import os, sys
import pathlib
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute())   )
from lightning_warnings import configure_lightning_warning_filters

configure_lightning_warning_filters()

import pytorch_lightning as pl
from fastmri.data.subsample import create_mask_for_mask_type
from pl_modules.pdac_module import PDACModule
from pl_modules.pdac_singlecoil_module import PDACModule_singlecoil
from data.data_transforms import PDACDataTransform
from pl_modules.fastmri_data_module import FastMriDataModule

from single_gpu_runtime import (
    normalize_single_gpu_args,
    uses_activation_checkpointing,
    validate_single_gpu_runtime,
)
from trainer_runtime import add_lightning_runtime_args, build_trainer_kwargs


def build_args():
    parser = ArgumentParser()

    # basic args
    batch_size = 1

    # client arguments
    parser.add_argument(
        '--checkpoint_file', 
        type=pathlib.Path,          
        help='Path to the checkpoint to load the model from.',
        default='./pretrained/pdac_fastmri_multicoil_8x'
    )
    parser.add_argument(
        "--use_compile",
        default=None,
        action="store_true",
        help="Override the checkpoint setting and enable torch.compile during evaluation.",
    )

    # data transform params
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

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        mask_type="random",  # random masks for knee data
        batch_size=batch_size,  # number of samples per batch
        test_path=None,  # path for test split, overwrites data_path
    )
    
    # runtime config
    parser = add_lightning_runtime_args(parser)

    args = parser.parse_args()

    return args


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # model
    # ------------
    if args.challenge == 'multicoil':
        model = PDACModule.load_from_checkpoint(args.checkpoint_file, map_location="cpu")
        hparams = dict(model.hparams)
        num_adj_slices = hparams['num_adj_slices']
        uniform_train_resolution = hparams['img_size']
    elif args.challenge == 'singlecoil':
        model = PDACModule_singlecoil.load_from_checkpoint(
            args.checkpoint_file, map_location="cpu"
        )
        hparams = dict(model.hparams)
        num_adj_slices = hparams['num_adj_slices']
        uniform_train_resolution = hparams['img_size']
    else:
        raise ValueError('Challenge not supported.')
    model.eval()
    compile_requested = (
        args.use_compile if args.use_compile is not None else hparams.get("use_compile", False)
    )
    validate_single_gpu_runtime(
        args, use_compile=compile_requested, context="eval_pdac_fastmri.py"
    )
    if compile_requested and uses_activation_checkpointing(hparams):
        raise ValueError(
            "eval_pdac_fastmri.py does not support `use_compile=true` for checkpoints "
            "saved with activation checkpointing enabled."
        )
    normalize_single_gpu_args(args)
    model._apply_compile_if_requested(compile_requested)
    
    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
        
    # use fixed masks for val transform
    val_transform = PDACDataTransform(uniform_train_resolution=uniform_train_resolution, mask_func=mask)
    
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=None,
        val_transform=val_transform,
        test_transform=None,
        test_split=None,
        test_path=None,
        sample_rate=None,
        volume_sample_rate=1.0,
        batch_size=1,
        num_workers=4,
        distributed_sampler=True,
        combine_train_val=False,
        train_scanners=args.train_scanners,
        val_scanners=args.val_scanners,
        combined_scanner_val=args.combined_scanner_val,
        num_adj_slices=num_adj_slices,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer(
        **build_trainer_kwargs(
            args,
            precision="32-true",
            callbacks=None,
            logger=False,
        )
    )
        
    # ------------
    # run
    # ------------
    trainer.validate(model, datamodule=data_module)


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == "__main__":
    run_cli()
