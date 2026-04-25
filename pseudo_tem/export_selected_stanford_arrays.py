from __future__ import annotations

import argparse
import csv
import sys
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
    parser.add_argument("--selected-cases-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def slugify_case(volume: str, slice_index: int) -> str:
    return f"{Path(volume).stem}__slice{slice_index:03d}"


def main() -> None:
    args = parse_args()
    configure_import_path(args)

    from fastmri import ifft2c, rss_complex
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is not None:
        model = model.to(device).eval()

    selected_rows = list(csv.DictReader(args.selected_cases_csv.open()))
    selected_keys = {(row["volume"], int(row["slice"])): row for row in selected_rows}

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
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    output_dir = args.output_dir / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = set(selected_keys.keys())
    with torch.no_grad():
        for batch in loader:
            masked_kspace, mask_t, target, fname, slice_num, max_value, crop_size, full_kspace = batch
            volume = fname[0]
            slice_index = int(slice_num[0].cpu().item())
            key = (volume, slice_index)
            if key not in selected_keys:
                continue

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

            case_slug = slugify_case(volume, slice_index)
            case_dir = output_dir / selected_keys[key]["category"]
            case_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                case_dir / f"{case_slug}.npz",
                volume=volume,
                slice=slice_index,
                target=target_c[0].numpy().astype(np.float32),
                output=output_c[0].numpy().astype(np.float32),
                max_value=np.array([scalar(max_value[0].cpu().numpy())], dtype=np.float32),
            )
            pending.remove(key)
            if not pending:
                break

    if pending:
        missing = ", ".join(f"{v}:{s}" for v, s in sorted(pending))
        raise RuntimeError(f"Failed to export selected cases: {missing}")

    print(f"wrote arrays to {output_dir}")
    print(f"cases: {len(selected_keys)}")


if __name__ == "__main__":
    main()
