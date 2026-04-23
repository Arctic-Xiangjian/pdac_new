import os, sys
import pathlib
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute())   )
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from lightning_warnings import configure_lightning_warning_filters

configure_lightning_warning_filters()

import pytorch_lightning as pl
from fastmri.data.subsample import create_mask_for_mask_type

from data.data_transforms import PDACDataTransform
from pl_modules.fastmri_data_module import FastMriDataModule

from pdac_examples.utils import load_args_from_config

from pl_modules.pdac_module import PDACModule
from pl_modules.pdac_singlecoil_module import PDACModule_singlecoil

from pytorch_lightning.callbacks import LearningRateMonitor
from single_gpu_runtime import (
    normalize_single_gpu_args,
    uses_activation_checkpointing,
    validate_single_gpu_runtime,
)
from trainer_runtime import (
    add_lightning_runtime_args,
    build_experiment_logger,
    build_trainer_kwargs,
    resolve_fit_ckpt_path,
    save_hparams_yaml,
)


def build_args():
    parser = ArgumentParser()

    batch_size = 1

    # client arguments
    parser.add_argument(
        '--config_file', 
        default="pdac_examples/config/fastmri/singlecoil/pdac.yaml",   
        type=pathlib.Path,          
        help='If given, experiment configuration will be loaded from this yaml file.',
    )
    parser.add_argument(
        '--verbose', 
        default=False,   
        action='store_true',          
        help='If set, print all command line arguments at startup.',
    )
    parser.add_argument(
        '--logger_type', 
        default='tb',   
        type=str,          
        help='Set Pytorch Lightning training logger. Options "tb" - Tensorboard (default), "wandb" - Weights and Biases',
    )
    parser.add_argument(
        '--experiment_name', 
        default='pdac-fastmri',   
        type=str,          
        help='Used with wandb logger to define the project name.',
    )
    parser.add_argument(
        "--use_bf16",
        default=False,
        action="store_true",
        help="If set, train with bf16 mixed precision instead of fp16 mixed precision.",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="equispaced",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.04],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[8],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        mask_type="random",  # random masks for knee data
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
    )

    # module config
    parser = PDACModule.add_model_specific_args(parser)

    # runtime config
    parser = add_lightning_runtime_args(parser)

    args = parser.parse_args()
    
    # Load args if config file is given
    if args.config_file is not None:
        args = load_args_from_config(args)

    return args


def cli_main(args):
    validate_single_gpu_runtime(
        args, use_compile=args.use_compile, context="train_pdac_fastmri2.py"
    )
    if args.use_compile and uses_activation_checkpointing(args):
        raise ValueError(
            "train_pdac_fastmri2.py does not support `use_compile=true` together with "
            "`use_checkpointing=true`."
        )
    normalize_single_gpu_args(args)

    if args.verbose:
        print(args.__dict__)
    
    if args.default_root_dir is not None:
        os.makedirs(args.default_root_dir, exist_ok=True)

    pl.seed_everything(args.seed)
    # ------------
    # model
    # ------------
    if args.challenge == 'multicoil':
        model = PDACModule(
            num_list=args.num_list,
            num_cascades=args.num_cascades,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            img_size=args.uniform_train_resolution,
            patch_size=args.patch_size,
            window_size=args.window_size,
            embed_dim=args.embed_dim, 
            depths=args.depths,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio, 
            bottleneck_depth=args.bottleneck_depth,
            bottleneck_heads=args.bottleneck_heads,
            resi_connection=args.resi_connection,
            conv_downsample_first=args.conv_downsample_first,
            num_adj_slices=args.num_adj_slices,
            mask_center=(not args.no_center_masking),
            use_checkpoint=args.use_checkpointing,
            use_compile=args.use_compile,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            max_epoch=args.max_epochs,
            no_residual_learning=args.no_residual_learning,
            logger_type=args.logger_type,
        )
    elif args.challenge == 'singlecoil':
        model = PDACModule_singlecoil(
            num_list=args.num_list,
            num_cascades=args.num_cascades,
            sens_pools=args.sens_pools,
            sens_chans=args.sens_chans,
            img_size=args.uniform_train_resolution,
            patch_size=args.patch_size,
            window_size=args.window_size,
            embed_dim=args.embed_dim, 
            depths=args.depths,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio, 
            bottleneck_depth=args.bottleneck_depth,
            bottleneck_heads=args.bottleneck_heads,
            resi_connection=args.resi_connection,
            conv_downsample_first=args.conv_downsample_first,
            num_adj_slices=args.num_adj_slices,
            mask_center=(not args.no_center_masking),
            use_checkpoint=args.use_checkpointing,
            use_compile=args.use_compile,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
            max_epoch=args.max_epochs,
            no_residual_learning=args.no_residual_learning,
            logger_type=args.logger_type,
        )
    else:
        raise ValueError('Challenge should be multicoil or singlecoil.')
    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    
    # use random masks for train transform, fixed masks for val transform
    train_transform = PDACDataTransform(uniform_train_resolution=args.uniform_train_resolution, mask_func=mask, use_seed=False)
    val_transform = PDACDataTransform(uniform_train_resolution=args.uniform_train_resolution, mask_func=mask)
    test_transform = PDACDataTransform(uniform_train_resolution=args.uniform_train_resolution)
    
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        use_dataset_cache_file=args.use_dataset_cache_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=True,
        combine_train_val=args.combine_train_val,
        train_scanners=args.train_scanners,
        val_scanners=args.val_scanners,
        combined_scanner_val=args.combined_scanner_val,
        num_adj_slices=args.num_adj_slices,
    )

    # ------------
    # trainer
    # ------------
    # set up logger
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            save_top_k=3,
            verbose=True,
            monitor="val_metrics/ssim",
            mode="max",
            filename="epoch{epoch}-ssim{val_metrics/ssim:.4f}",
            auto_insert_metric_name=False,
            save_last=False,
        ),
        pl.callbacks.ModelCheckpoint(
            save_top_k=3,
            verbose=True,
            monitor="val_metrics/psnr",
            mode="max",
            filename="epoch{epoch}-psnr{val_metrics/psnr:.2f}",
            auto_insert_metric_name=False,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = build_experiment_logger(args)
    trainer = pl.Trainer(
        **build_trainer_kwargs(
            args,
            precision="bf16-mixed" if args.use_bf16 else "16-mixed",
            callbacks=callbacks,
            logger=logger,
        )
    )

    save_hparams_yaml(trainer, args)
            
    # ------------
    # run
    # ------------
    trainer.fit(model, datamodule=data_module, ckpt_path=resolve_fit_ckpt_path(args))


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
