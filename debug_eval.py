import pathlib

import pytorch_lightning as pl
import torch
import yaml
from fastmri.data.subsample import create_mask_for_mask_type

from checkpoint_utils import extract_prefixed_state_dict
from data.data_transforms import PDACDataTransform
from pl_modules.fastmri_data_module import FastMriDataModule
from pl_modules.pdac_singlecoil_module import PDACModule_singlecoil


ROOT = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "pdac_examples/config/fastmri/singlecoil/pdac.yaml"
CHECKPOINT_PATH = (
    ROOT
    / "experiments/fastmri_pdac/singlecoil/lightning_logs/version_0/checkpoints/epoch49-ssim0.6758.ckpt"
)
RUNTIME_NUM_LIST = [148, 198, 232, 256, 274, 290, 306, 320]


def build_model(cfg):
    model = PDACModule_singlecoil(
        lr=cfg["lr"],
        lr_step_size=cfg["lr_step_size"],
        lr_gamma=cfg["lr_gamma"],
        weight_decay=cfg["weight_decay"],
        num_adj_slices=cfg["num_adj_slices"],
        use_compile=False,
        logger_type="tb",
        num_list=RUNTIME_NUM_LIST,
        num_cascades=cfg["num_cascades"],
        sens_chans=cfg["sens_chans"],
        img_size=cfg["uniform_train_resolution"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depths=cfg["depths"],
        num_heads=cfg["num_heads"],
        window_size=cfg["window_size"],
        mlp_ratio=cfg["mlp_ratio"],
        use_checkpoint=cfg["use_checkpointing"],
        resi_connection=cfg["resi_connection"],
        bottleneck_depth=cfg["bottleneck_depth"],
        bottleneck_heads=cfg["bottleneck_heads"],
        conv_downsample_first=cfg["conv_downsample_first"],
    ).eval()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    missing_keys, unexpected_keys = model.model.load_state_dict(
        extract_prefixed_state_dict(checkpoint, "model"),
        strict=False,
    )
    print("checkpoint:", CHECKPOINT_PATH)
    print("runtime_num_list:", RUNTIME_NUM_LIST)
    print("missing_keys:", missing_keys)
    print("unexpected_keys:", unexpected_keys)
    return model


def build_data_module(cfg):
    mask = create_mask_for_mask_type(
        cfg["mask_type"],
        cfg["center_fractions"],
        cfg["accelerations"],
    )
    val_transform = PDACDataTransform(
        uniform_train_resolution=cfg["uniform_train_resolution"],
        mask_func=mask,
    )
    return FastMriDataModule(
        data_path=cfg["data_path"],
        challenge=cfg["challenge"],
        train_transform=None,
        val_transform=val_transform,
        test_transform=None,
        test_split=None,
        test_path=None,
        sample_rate=None,
        volume_sample_rate=cfg["volume_sample_rate"],
        batch_size=1,
        num_workers=4,
        distributed_sampler=False,
        combine_train_val=False,
        train_scanners=None,
        val_scanners=None,
        combined_scanner_val=False,
        num_adj_slices=cfg["num_adj_slices"],
    )


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    pl.seed_everything(cfg["seed"])

    model = build_model(cfg)
    data_module = build_data_module(cfg)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    metrics = trainer.validate(model, datamodule=data_module, verbose=True)
    print("validation_metrics:", metrics)


if __name__ == "__main__":
    main()
