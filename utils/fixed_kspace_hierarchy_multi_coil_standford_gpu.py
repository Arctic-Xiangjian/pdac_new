from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.stanford.stanford_data import StanfordSliceDataset

if __package__ in (None, ""):
    from fixed_kspace_hierarchy_multi_coil_common_gpu import (
        CACHE_MODE_CHOICES,
        DEFAULT_CALIBRATION_WINDOW,
        DEFAULT_NUM_VIRTUAL_COILS,
        DEFAULT_REPRESENTATION,
        RAW_COIL_REPRESENTATION,
        REPRESENTATION_CHOICES,
        infer_max_available_kspace_coils,
        run_hierarchy_job,
    )
else:
    from .fixed_kspace_hierarchy_multi_coil_common_gpu import (
        CACHE_MODE_CHOICES,
        DEFAULT_CALIBRATION_WINDOW,
        DEFAULT_NUM_VIRTUAL_COILS,
        DEFAULT_REPRESENTATION,
        RAW_COIL_REPRESENTATION,
        REPRESENTATION_CHOICES,
        infer_max_available_kspace_coils,
        run_hierarchy_job,
    )


DEFAULT_DATA_ROOT = "/working2/arctic/Recon/stanford_convert/"
DEFAULT_OUTPUT_PREFIX = (
    Path(__file__).resolve().parent / "gamma_stanford_multicoil_384_raw_gpu"
)
DEFAULT_TMP_DIR = "/tmp/cov_rank_stanford_multicoil_384_gpu"


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute raw fixed k-space hierarchy statistics for Stanford multi-coil data on GPU."
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-samples", type=int, default=1024 * 4)
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
    parser.add_argument("--block-size-gram", type=int, default=1024)
    parser.add_argument("--block-size-center", type=int, default=512)
    parser.add_argument(
        "--representation",
        type=str,
        default=RAW_COIL_REPRESENTATION,
        choices=REPRESENTATION_CHOICES,
    )
    parser.add_argument(
        "--num-virtual-coils",
        type=int,
        default=None,
        help=(
            "Number of output coil channels. Defaults to the maximum available "
            "coil count in the selected Stanford split for raw/virtual-coil "
            "representations."
        ),
    )
    parser.add_argument(
        "--calibration-window",
        type=int,
        default=DEFAULT_CALIBRATION_WINDOW,
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    raw_grouping = parser.add_mutually_exclusive_group()
    raw_grouping.add_argument(
        "--group-raw-coils",
        dest="group_raw_coils",
        action="store_true",
        help=(
            "For raw_coil representation, group samples by their real coil count "
            "and compute each group without channel padding."
        ),
    )
    raw_grouping.add_argument(
        "--no-group-raw-coils",
        dest="group_raw_coils",
        action="store_false",
        help="Use the legacy fixed-channel raw_coil path with zero padding.",
    )
    parser.set_defaults(group_raw_coils=True)
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="gpu",
        choices=CACHE_MODE_CHOICES,
        help=(
            "Where to store the represented k-space cache. Defaults to GPU for "
            "this GPU entry point; use 'auto' to allow fallback to CPU streaming."
        ),
    )
    parser.add_argument("--gpu-cache-max-gb", type=float, default=60.0)
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
    if args.num_virtual_coils is None and args.representation != "rss_pseudo":
        num_virtual_coils = infer_max_available_kspace_coils(dataset)
        num_virtual_coils_source = "max_available_kspace_coils"
        if args.representation == RAW_COIL_REPRESENTATION and args.group_raw_coils:
            print(
                "[*] Auto max raw coil count for metadata/legacy fallback: "
                f"{num_virtual_coils}"
            )
        else:
            print(
                "[*] Auto output coil channels from max available k-space coils: "
                f"{num_virtual_coils}"
            )
    else:
        num_virtual_coils = (
            DEFAULT_NUM_VIRTUAL_COILS
            if args.num_virtual_coils is None
            else int(args.num_virtual_coils)
        )
        num_virtual_coils_source = (
            "default_unused" if args.num_virtual_coils is None else "cli"
        )

    run_hierarchy_job(
        dataset=dataset,
        dataset_tag="stanford_multicoil_384_gpu",
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
        num_virtual_coils=num_virtual_coils,
        calibration_window=args.calibration_window,
        device=args.device,
        cache_mode=args.cache_mode,
        gpu_cache_max_gb=args.gpu_cache_max_gb,
        group_raw_coils=bool(args.group_raw_coils),
        metadata={
            "data_root": str(args.data_root),
            "dataset_name": "stanford",
            "challenge": "multicoil",
            "data_partition": "train",
            "train_val_split": float(args.train_val_split),
            "train_val_seed": int(args.train_val_seed),
            "source": "reconstruction_rss"
            if args.representation == "rss_pseudo"
            else "raw_kspace",
            "representation": str(args.representation),
            "num_virtual_coils": int(num_virtual_coils),
            "num_coils": int(num_virtual_coils),
            "num_virtual_coils_source": num_virtual_coils_source,
            "calibration_window": int(args.calibration_window),
            "raw_coil_grouping_requested": bool(args.group_raw_coils),
        },
    )


if __name__ == "__main__":
    main()
