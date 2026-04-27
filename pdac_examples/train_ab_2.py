import os
import pathlib
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from lightning_warnings import configure_lightning_warning_filters

configure_lightning_warning_filters()

import pytorch_lightning as pl
from fastmri.data.subsample import create_mask_for_mask_type
from pytorch_lightning.callbacks import LearningRateMonitor

from data.data_transforms import PDACDataTransform
from pdac_examples.utils import load_args_from_config
from pl_modules.pdac_module_ab2 import PDACModuleAB2
from pl_modules.stanford_data_module import StanfordDataModule
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

    parser.add_argument(
        "--config_file",
        default="pdac_examples/config/stanford2d/ab_2.yaml",
        type=pathlib.Path,
        help="If given, experiment configuration will be loaded from this yaml file.",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="If set, print all command line arguments at startup.",
    )
    parser.add_argument(
        "--logger_type",
        default="tb",
        type=str,
        help='Set Pytorch Lightning training logger. Options "tb" or "wandb".',
    )
    parser.add_argument(
        "--experiment_name",
        default="pdac-stanford-ab2",
        type=str,
        help="Used with wandb logger to define the project name.",
    )
    parser.add_argument(
        "--use_bf16",
        default=False,
        action="store_true",
        help="If set, train with bf16 mixed precision instead of fp16 mixed precision.",
    )

    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="random",
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

    parser = StanfordDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        mask_type="random",
        batch_size=batch_size,
        test_path=None,
    )

    parser = PDACModuleAB2.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=8,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
        lr=0.0001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
    )

    parser = add_lightning_runtime_args(parser)

    args = parser.parse_args()
    if args.config_file is not None:
        args = load_args_from_config(args)

    return args


def cli_main(args):
    validate_single_gpu_runtime(args, use_compile=args.use_compile, context="train_ab_2.py")
    if args.use_compile and uses_activation_checkpointing(args):
        raise ValueError(
            "train_ab_2.py does not support `use_compile=true` together with "
            "`use_checkpointing=true`."
        )
    normalize_single_gpu_args(args)

    if args.verbose:
        print(args.__dict__)

    if args.default_root_dir is not None:
        os.makedirs(args.default_root_dir, exist_ok=True)

    pl.seed_everything(args.seed)

    model = PDACModuleAB2(
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

    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )

    train_transform = PDACDataTransform(
        uniform_train_resolution=args.uniform_train_resolution,
        mask_func=mask,
        use_seed=False,
    )
    val_transform = PDACDataTransform(
        uniform_train_resolution=args.uniform_train_resolution,
        mask_func=mask,
    )
    test_transform = PDACDataTransform(uniform_train_resolution=args.uniform_train_resolution)

    data_module = StanfordDataModule(
        data_path=args.data_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=True,
        train_val_seed=args.train_val_seed,
        train_val_split=args.train_val_split,
        num_adj_slices=args.num_adj_slices,
    )

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
        LearningRateMonitor(logging_interval="epoch"),
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
    trainer.fit(model, datamodule=data_module, ckpt_path=resolve_fit_ckpt_path(args))


def run_cli():
    args = build_args()
    cli_main(args)


if __name__ == "__main__":
    run_cli()
