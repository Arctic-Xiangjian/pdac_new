from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--legacy-repo", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("current_model", "legacy_model", "zero_filled"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-volume-csv", type=Path, required=True)
    parser.add_argument("--output-slice-csv", type=Path, required=True)
    parser.add_argument("--train-val-split", type=float, default=0.8)
    parser.add_argument("--train-val-seed", type=int, default=0)
    parser.add_argument("--center-fractions", nargs="+", type=float, default=[0.04])
    parser.add_argument("--accelerations", nargs="+", type=int, default=[8])
    parser.add_argument("--mask-type", type=str, default="random")
    parser.add_argument("--uniform-train-resolution", nargs=2, type=int, default=[384, 384])
    parser.add_argument("--num-adj-slices", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def configure_import_path(args: argparse.Namespace) -> None:
    sys.path = [str(args.repo_root)] + [p for p in sys.path if p != str(args.repo_root)]
    if args.mode == "legacy_model":
        if args.legacy_repo is None:
            raise ValueError("--legacy-repo is required for legacy_model mode.")
        sys.path = [str(args.legacy_repo)] + [p for p in sys.path if p != str(args.legacy_repo)]


def scalar(x: Any) -> float:
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_runtime_objects(args: argparse.Namespace):
    from fastmri import evaluate, ifft2c, rss_complex
    from fastmri.data import transforms
    from fastmri.data.subsample import create_mask_for_mask_type

    from data.data_transforms import PDACDataTransform
    from data.stanford.stanford_data import StanfordSliceDataset

    model = None
    if args.mode in ("current_model", "legacy_model"):
        from pl_modules.pdac_module import PDACModule

        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for model modes.")

        if args.mode == "current_model":
            model = PDACModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
        else:
            model = PDACModule.load_from_checkpoint(args.checkpoint)

    mask = create_mask_for_mask_type(
        args.mask_type,
        args.center_fractions,
        args.accelerations,
    )
    transform = PDACDataTransform(
        uniform_train_resolution=args.uniform_train_resolution,
        mask_func=mask,
    )
    dataset = StanfordSliceDataset(
        root=args.data_path,
        data_partition="val",
        train_val_split=args.train_val_split,
        train_val_seed=args.train_val_seed,
        transform=transform,
        sample_rate=1.0,
        volume_sample_rate=None,
        num_adj_slices=args.num_adj_slices,
    )

    return evaluate, ifft2c, rss_complex, transforms, dataset, model


def main() -> None:
    args = parse_args()
    configure_import_path(args)

    evaluate, ifft2c, rss_complex, transforms, dataset, model = load_runtime_objects(args)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is not None:
        model = model.to(device).eval()

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    per_volume: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "slice_mse": [],
            "slice_target_norm": [],
            "slice_ssim": [],
            "slice_output_target_max_ratio": [],
            "max_value": None,
            "num_slices": 0,
        }
    )
    slice_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            masked_kspace, mask_t, target, fname, slice_num, max_value, crop_size, full_kspace = batch

            volume_name = fname[0]
            slice_index = int(slice_num[0].cpu().item())
            max_value_f = scalar(max_value[0].cpu().numpy())

            if model is None:
                output = rss_complex(ifft2c(masked_kspace), dim=0)
                target_c, output_c = transforms.center_crop_to_smallest(target, output)
            else:
                masked_kspace = masked_kspace.to(device)
                mask_t = mask_t.to(device)
                target = target.to(device)
                eval_context = (
                    model._evaluation_autocast_context()
                    if hasattr(model, "_evaluation_autocast_context")
                    else torch.no_grad()
                )
                with eval_context:
                    output = model.forward(masked_kspace, mask_t)
                    if isinstance(output, tuple):
                        output = output[0]
                    target_c, output_c = transforms.center_crop_to_smallest(target, output)
                target_c = target_c.cpu()
                output_c = output_c.cpu()

            target_np = target_c[0].numpy()
            output_np = output_c[0].numpy()
            mse_val = scalar(evaluate.mse(target_np, output_np))
            target_norm = scalar(evaluate.mse(target_np, np.zeros_like(target_np)))
            ssim_val = scalar(
                evaluate.ssim(target_np[None, ...], output_np[None, ...], maxval=max_value_f)
            )
            output_target_ratio = float(
                output_c.max().item() / (target_c.max().item() + 1e-12)
            )

            per_volume_entry = per_volume[volume_name]
            per_volume_entry["slice_mse"].append(mse_val)
            per_volume_entry["slice_target_norm"].append(target_norm)
            per_volume_entry["slice_ssim"].append(ssim_val)
            per_volume_entry["slice_output_target_max_ratio"].append(output_target_ratio)
            per_volume_entry["max_value"] = max_value_f
            per_volume_entry["num_slices"] += 1

            slice_rows.append(
                {
                    "volume": volume_name,
                    "slice": slice_index,
                    "mse": mse_val,
                    "target_norm": target_norm,
                    "nmse": mse_val / target_norm,
                    "psnr": 20 * math.log10(max_value_f) - 10 * math.log10(mse_val),
                    "ssim": ssim_val,
                    "max_value": max_value_f,
                    "target_max": float(target_c.max().item()),
                    "output_max": float(output_c.max().item()),
                    "output_target_max_ratio": output_target_ratio,
                }
            )

    volume_rows: list[dict[str, Any]] = []
    for volume_name in sorted(per_volume.keys()):
        volume = per_volume[volume_name]
        mean_mse = float(np.mean(volume["slice_mse"]))
        mean_target_norm = float(np.mean(volume["slice_target_norm"]))
        mean_ssim = float(np.mean(volume["slice_ssim"]))
        max_value_f = float(volume["max_value"])
        volume_rows.append(
            {
                "volume": volume_name,
                "num_slices": volume["num_slices"],
                "mean_slice_mse": mean_mse,
                "mean_slice_target_norm": mean_target_norm,
                "nmse": mean_mse / mean_target_norm,
                "psnr": 20 * math.log10(max_value_f) - 10 * math.log10(mean_mse),
                "ssim": mean_ssim,
                "max_value": max_value_f,
                "mean_output_target_max_ratio": float(
                    np.mean(volume["slice_output_target_max_ratio"])
                ),
            }
        )

    slice_rows.sort(key=lambda row: (row["volume"], row["slice"]))
    volume_rows.sort(key=lambda row: row["volume"])

    write_csv(
        args.output_slice_csv,
        slice_rows,
        [
            "volume",
            "slice",
            "mse",
            "target_norm",
            "nmse",
            "psnr",
            "ssim",
            "max_value",
            "target_max",
            "output_max",
            "output_target_max_ratio",
        ],
    )
    write_csv(
        args.output_volume_csv,
        volume_rows,
        [
            "volume",
            "num_slices",
            "mean_slice_mse",
            "mean_slice_target_norm",
            "nmse",
            "psnr",
            "ssim",
            "max_value",
            "mean_output_target_max_ratio",
        ],
    )

    print(f"wrote volume csv: {args.output_volume_csv}")
    print(f"wrote slice csv: {args.output_slice_csv}")
    print(f"volumes: {len(volume_rows)}")
    print(f"slices: {len(slice_rows)}")


if __name__ == "__main__":
    main()
