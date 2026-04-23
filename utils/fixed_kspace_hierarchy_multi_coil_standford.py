from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.stanford.stanford_data import StanfordSliceDataset

if __package__ in (None, ""):
    from fixed_kspace_hierarchy_multi_coil_common import (
        DEFAULT_CALIBRATION_WINDOW,
        DEFAULT_NUM_VIRTUAL_COILS,
        DEFAULT_REPRESENTATION,
        REPRESENTATION_CHOICES,
        run_hierarchy_job,
    )
else:
    from .fixed_kspace_hierarchy_multi_coil_common import (
        DEFAULT_CALIBRATION_WINDOW,
        DEFAULT_NUM_VIRTUAL_COILS,
        DEFAULT_REPRESENTATION,
        REPRESENTATION_CHOICES,
        run_hierarchy_job,
    )


DEFAULT_DATA_ROOT = "/working2/arctic/Recon/stanford_convert/"
DEFAULT_OUTPUT_PREFIX = (
    Path(__file__).resolve().parent / "gamma_stanford_multicoil_384_raw"
)
DEFAULT_TMP_DIR = "/tmp/cov_rank_stanford_multicoil_384"


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute raw fixed k-space hierarchy statistics for Stanford multi-coil data."
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-samples", type=int, default=1024*4)
    parser.add_argument(
        "--uniform-train-resolution", nargs=2, type=int, default=[384, 384]
    )
    parser.add_argument("--tmp-dir", type=str, default=DEFAULT_TMP_DIR)
    parser.add_argument(
        "--output-prefix", type=str, default=str(DEFAULT_OUTPUT_PREFIX)
    )
    parser.add_argument("--train-val-split", type=float, default=0.8)
    parser.add_argument("--train-val-seed", type=int, default=0)
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize-per-sample",
        dest="normalize_per_sample",
        action="store_true",
        help="Normalize each sample vector before Gram computation.",
    )
    normalize_group.add_argument(
        "--no-normalize-per-sample",
        dest="normalize_per_sample",
        action="store_false",
        help="Disable per-sample normalization.",
    )
    parser.set_defaults(normalize_per_sample=True)
    parser.add_argument(
        "--tau-mode",
        type=str,
        default="full_tail_median_ratio",
        choices=(
            "full_max_ratio",
            "full_tail_median_ratio",
            "full_tail_mean_ratio",
        ),
    )
    parser.add_argument("--tau-ratio", type=float, default=1.0)
    parser.add_argument("--tau-abs", type=float, default=None)
    parser.add_argument("--block-size-gram", type=int, default=256)
    parser.add_argument("--block-size-center", type=int, default=512)
    parser.add_argument(
        "--representation",
        type=str,
        default=DEFAULT_REPRESENTATION,
        choices=REPRESENTATION_CHOICES,
    )
    parser.add_argument(
        "--num-virtual-coils",
        type=int,
        default=DEFAULT_NUM_VIRTUAL_COILS,
    )
    parser.add_argument(
        "--calibration-window",
        type=int,
        default=DEFAULT_CALIBRATION_WINDOW,
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()
    dataset = StanfordSliceDataset(
        root=Path(args.data_root),
        data_partition="train",
        train_val_split=args.train_val_split,
        train_val_seed=args.train_val_seed,
        transform=None,
    )
    run_hierarchy_job(
        dataset=dataset,
        dataset_tag="stanford_multicoil_384",
        output_prefix=args.output_prefix,
        tmp_dir=args.tmp_dir,
        num_samples=args.num_samples,
        uniform_train_resolution=args.uniform_train_resolution,
        normalize_per_sample=args.normalize_per_sample,
        tau_mode=args.tau_mode,
        tau_ratio=args.tau_ratio,
        tau_abs=args.tau_abs,
        block_size_gram=args.block_size_gram,
        block_size_center=args.block_size_center,
        representation=args.representation,
        num_virtual_coils=args.num_virtual_coils,
        calibration_window=args.calibration_window,
        metadata={
            "data_root": str(args.data_root),
            "dataset_name": "stanford",
            "challenge": "multicoil",
            "data_partition": "train",
            "train_val_split": float(args.train_val_split),
            "train_val_seed": int(args.train_val_seed),
            "source": "raw_kspace"
            if args.representation == DEFAULT_REPRESENTATION
            else "reconstruction_rss",
            "representation": str(args.representation),
            "num_virtual_coils": int(args.num_virtual_coils),
            "calibration_window": int(args.calibration_window),
        },
    )


if __name__ == "__main__":
    main()
