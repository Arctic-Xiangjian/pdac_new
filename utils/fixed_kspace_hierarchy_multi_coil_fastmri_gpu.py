from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.fastmri_data import SliceDataset

if __package__ in (None, ""):
    from fixed_kspace_hierarchy_multi_coil_common_gpu import (
        DEFAULT_CALIBRATION_WINDOW,
        DEFAULT_NUM_VIRTUAL_COILS,
        DEFAULT_REPRESENTATION,
        EPS,
        RAW_COIL_REPRESENTATION,
        REPRESENTATION_CHOICES,
        _accumulate_shell_gram_inplace,
        _build_kspace_representation,
        _bytes_to_gb,
        _center_gram_inplace,
        _ensure_cuda_device,
        _estimate_cache_nbytes,
        _materialize_window_gram,
        _normalize_even_window,
        _normalize_positive_int,
        _normalize_resolution,
        _peak_cuda_memory_gb,
        _representation_num_channels,
        _resolve_output_paths,
        _synchronize,
        _write_json_log,
        _write_text_log,
        infer_max_available_kspace_coils,
    )
else:
    from .fixed_kspace_hierarchy_multi_coil_common_gpu import (
        DEFAULT_CALIBRATION_WINDOW,
        DEFAULT_NUM_VIRTUAL_COILS,
        DEFAULT_REPRESENTATION,
        EPS,
        RAW_COIL_REPRESENTATION,
        REPRESENTATION_CHOICES,
        _accumulate_shell_gram_inplace,
        _build_kspace_representation,
        _bytes_to_gb,
        _center_gram_inplace,
        _ensure_cuda_device,
        _estimate_cache_nbytes,
        _materialize_window_gram,
        _normalize_even_window,
        _normalize_positive_int,
        _normalize_resolution,
        _peak_cuda_memory_gb,
        _representation_num_channels,
        _resolve_output_paths,
        _synchronize,
        _write_json_log,
        _write_text_log,
        infer_max_available_kspace_coils,
    )


DEFAULT_DATA_ROOT = "/v/ai/nobackup/arctic/public_lowlevel/data/fastMRI/knee/multicoil_train"
DEFAULT_OUTPUT_PREFIX = (
    Path(__file__).resolve().parent / "gamma_fastmri_multicoil_384_raw_gpu"
)
DEFAULT_TMP_DIR = "/tmp/cov_rank_fastmri_multicoil_384_gpu"
GPU_ONLY_CACHE_MODE_CHOICES = ("gpu", "auto")


def _force_gpu_cache_mode(cache_mode: str) -> str:
    if cache_mode != "gpu":
        print(
            f"[*] FastMRI GPU-only path forces cache_mode='gpu' "
            f"(received {cache_mode!r})."
        )
    return "gpu"


def _fastmri_slice_indices(dataset: SliceDataset, dataslice: int, num_slices: int) -> int | list[int]:
    num_adj_slices = _normalize_positive_int(
        "num_adj_slices",
        int(getattr(dataset, "num_adj_slices", 1) or 1),
    )
    if num_adj_slices == 1:
        return int(dataslice)

    get_slice_indices = getattr(dataset, "_get_slice_indices", None)
    if callable(get_slice_indices):
        return list(map(int, get_slice_indices(int(dataslice), int(num_slices))))

    num_slices_per_side = (num_adj_slices - 1) // 2
    slice_idx_l = int(dataslice) - num_slices_per_side
    slice_idx_h = int(dataslice) + num_slices_per_side
    if slice_idx_l < 0:
        diff = -slice_idx_l
        slice_idx_list = list(range(0, slice_idx_h + 1))
        return [0] * diff + slice_idx_list
    if slice_idx_h >= num_slices:
        diff = num_slices - slice_idx_h + 1
        slice_idx_list = list(range(slice_idx_l, num_slices))
        return slice_idx_list + [num_slices - 1] * diff
    return list(range(slice_idx_l, slice_idx_h + 1))


def _read_fastmri_sample_for_gpu(
    dataset: SliceDataset,
    idx: int,
    *,
    representation: str,
) -> Tuple[Any, Optional[Any]]:
    raw_samples = getattr(dataset, "raw_samples", None)
    if raw_samples is None:
        raw_samples = getattr(dataset, "examples", None)
    if raw_samples is None:
        sample = dataset[idx]
        return sample[0], sample[2] if representation == "rss_pseudo" and len(sample) > 2 else None

    fname, dataslice, _ = raw_samples[idx]
    with h5py.File(fname, "r") as hf:
        target = None
        if representation == "rss_pseudo":
            recons_key = getattr(dataset, "recons_key", "reconstruction_rss")
            target = hf[recons_key][dataslice] if recons_key in hf else None
            raw_kspace = None
        else:
            kspace_ds = hf["kspace"]
            slice_indices = _fastmri_slice_indices(
                dataset,
                int(dataslice),
                int(kspace_ds.shape[0]),
            )
            if isinstance(slice_indices, int):
                raw_kspace = kspace_ds[slice_indices]
            else:
                raw_kspace = np.concatenate(
                    [kspace_ds[slice_idx] for slice_idx in slice_indices],
                    axis=0,
                )

    return raw_kspace, target


def _load_fastmri_centered_kspace_cache_gpu(
    dataset: SliceDataset,
    num_samples: int,
    uniform_train_resolution: Sequence[int],
    *,
    representation: str,
    num_virtual_coils: int,
    calibration_window: int,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    height, width = _normalize_resolution(uniform_train_resolution)
    num_channels = _representation_num_channels(representation, num_virtual_coils)
    estimated_cache_nbytes = _estimate_cache_nbytes(num_samples, num_channels, height, width)

    kspace_cache = torch.empty(
        (num_samples, num_channels, height, width, 2),
        dtype=torch.float32,
        device=device,
    )

    print(
        f"[*] Caching {num_samples} fastMRI samples of centered {height}x{width} "
        f"{representation} k-space with {num_channels} channel(s) directly on GPU "
        f"(estimated={_bytes_to_gb(estimated_cache_nbytes):.2f} GB)"
    )

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Caching centered k-space on GPU"):
            raw_kspace, target = _read_fastmri_sample_for_gpu(
                dataset,
                idx,
                representation=representation,
            )
            represented = _build_kspace_representation(
                raw_kspace,
                target=target,
                uniform_train_resolution=(height, width),
                representation=representation,
                num_virtual_coils=num_virtual_coils,
                calibration_window=calibration_window,
                device=device,
            )
            expected_shape = (num_channels, height, width, 2)
            if tuple(represented.shape) != expected_shape:
                raise ValueError(
                    f"Expected represented k-space shape {expected_shape}, got {tuple(represented.shape)}"
                )
            kspace_cache[idx].copy_(represented)

    _synchronize(device)
    return kspace_cache, _bytes_to_gb(estimated_cache_nbytes)


def _spectral_stats_from_centered_gram_gpu(
    gram: torch.Tensor,
    n_samples: int,
    eps_ratio: float = 1e-12,
) -> Tuple[float, torch.Tensor]:
    print("[*] Eigendecomposition of centered Gram on GPU...")
    gram = 0.5 * (gram + gram.T)
    eigvals_gram = torch.linalg.eigvalsh(gram)
    eigvals_gram = torch.clamp(eigvals_gram, min=0.0)

    if eigvals_gram.numel() == 0:
        return 1.0, eigvals_gram

    thresh = eigvals_gram[-1] * float(eps_ratio)
    eigvals_gram = eigvals_gram[eigvals_gram > thresh]
    if eigvals_gram.numel() == 0:
        return 1.0, eigvals_gram

    probs = eigvals_gram / torch.clamp(eigvals_gram.sum(), min=EPS)
    entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=EPS)))
    r_eff = float(torch.exp(entropy).item())
    eigvals_cov = eigvals_gram / max(n_samples - 1, 1)
    return r_eff, eigvals_cov


def _median_sorted(values: torch.Tensor) -> torch.Tensor:
    count = int(values.numel())
    if count <= 0:
        raise ValueError("Cannot take the median of an empty tensor.")
    midpoint = count // 2
    if count % 2 == 1:
        return values[midpoint]
    return 0.5 * (values[midpoint - 1] + values[midpoint])


def _estimate_tau_gpu(
    eigvals_cov_by_window: Mapping[int, torch.Tensor],
    window_list: Sequence[int],
    *,
    tau_mode: str,
    tau_ratio: float,
    tau_abs: Optional[float],
    device: torch.device,
    tail_frac: float = 0.1,
    tail_count: int = 32,
) -> torch.Tensor:
    if tau_abs is not None:
        return torch.tensor(float(tau_abs), dtype=torch.float64, device=device)

    full_window = max(window_list)
    full_eigs = eigvals_cov_by_window[full_window]
    if full_eigs.numel() == 0:
        return torch.ones((), dtype=torch.float64, device=device)

    if tau_mode == "full_max_ratio":
        return torch.clamp(float(tau_ratio) * torch.max(full_eigs), min=EPS)

    if tau_mode in {"full_tail_median_ratio", "full_tail_mean_ratio"}:
        tail_n = max(
            1,
            min(
                int(full_eigs.numel()),
                max(tail_count, int(np.ceil(tail_frac * int(full_eigs.numel())))),
            ),
        )
        tail = torch.sort(full_eigs).values[:tail_n]
        if tau_mode == "full_tail_median_ratio":
            tau = float(tau_ratio) * _median_sorted(tail)
        else:
            tau = float(tau_ratio) * torch.mean(tail)
        return torch.clamp(tau, min=EPS)

    raise ValueError(f"Unsupported tau_mode: {tau_mode}")


def _gaussian_info_from_cov_eigs_gpu(eigvals_cov: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    if eigvals_cov.numel() == 0:
        return torch.zeros((), dtype=torch.float64, device=tau.device)
    return 0.5 * torch.sum(torch.log1p(eigvals_cov / torch.clamp(tau, min=EPS)))


def _compute_delta_infos_gpu(
    info_by_window: Mapping[int, torch.Tensor],
    *,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_values = sorted(info_by_window.keys())
    windows_t = torch.tensor(window_values, dtype=torch.int64, device=device)
    infos_t = torch.stack([info_by_window[window] for window in window_values]).to(torch.float64)
    infos_t = torch.cummax(infos_t, dim=0).values
    deltas_t = torch.diff(
        torch.cat([torch.zeros(1, dtype=torch.float64, device=device), infos_t])
    )
    deltas_t = torch.clamp(deltas_t, min=0.0)
    return (
        windows_t.detach().cpu().numpy(),
        infos_t.detach().cpu().numpy(),
        deltas_t.detach().cpu().numpy(),
    )


def calc_fastmri_cov_effective_rank_sizes_gpu_only(
    dataset: SliceDataset,
    tmp_dir: os.PathLike[str] | str,
    num_samples: int,
    window_list: Sequence[int],
    uniform_train_resolution: Sequence[int],
    *,
    block_size_gram: int,
    block_size_center: int,
    normalize_per_sample: bool,
    tau_mode: str,
    tau_ratio: float,
    tau_abs: Optional[float],
    representation: str,
    num_virtual_coils: int,
    calibration_window: int,
    device: str | torch.device,
) -> Dict[str, Any]:
    del tmp_dir
    device_obj = _ensure_cuda_device(device)
    uniform_train_resolution = _normalize_resolution(uniform_train_resolution)
    num_samples = min(int(num_samples), len(dataset))
    if num_samples <= 0:
        raise ValueError("num_samples must be positive after dataset truncation.")

    if normalize_per_sample:
        warnings.warn(
            "normalize_per_sample=True makes the complexity more shape-sensitive "
            "and less energy-sensitive. This is suitable for hierarchy discovery.",
            stacklevel=2,
        )

    num_virtual_coils = _normalize_positive_int("num_virtual_coils", num_virtual_coils)
    if representation == DEFAULT_REPRESENTATION:
        calibration_window = _normalize_even_window(
            calibration_window,
            min(uniform_train_resolution),
            name="calibration_window",
        )

    with torch.cuda.device(device_obj):
        torch.cuda.reset_peak_memory_stats()

    kspace_cache, represented_cache_gb = _load_fastmri_centered_kspace_cache_gpu(
        dataset=dataset,
        num_samples=num_samples,
        uniform_train_resolution=uniform_train_resolution,
        representation=representation,
        num_virtual_coils=num_virtual_coils,
        calibration_window=calibration_window,
        device=device_obj,
    )

    cumulative_gram = torch.zeros(
        (num_samples, num_samples),
        dtype=torch.float64,
        device=device_obj,
    )
    cumulative_norm_sq = torch.zeros(num_samples, dtype=torch.float64, device=device_obj)

    r_eff_by_window: Dict[int, float] = {}
    eigvals_cov_by_window: Dict[int, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for window_size in window_list:
                _accumulate_shell_gram_inplace(
                    kspace_cache,
                    cumulative_gram=cumulative_gram,
                    cumulative_norm_sq=cumulative_norm_sq,
                    window_size=window_size,
                    block_size=block_size_gram,
                    device=device_obj,
                )

                gram = _materialize_window_gram(
                    cumulative_gram,
                    cumulative_norm_sq,
                    normalize_per_sample=normalize_per_sample,
                    block_size=block_size_center,
                )
                gram = _center_gram_inplace(gram, block_size=block_size_center)

                r_eff, eigvals_cov = _spectral_stats_from_centered_gram_gpu(
                    gram,
                    n_samples=num_samples,
                )
                r_eff_by_window[window_size] = r_eff
                eigvals_cov_by_window[window_size] = eigvals_cov

                print(
                    f"[CovRank] window={window_size:3d}x{window_size:<3d} | "
                    f"r_eff={r_eff:10.4f} | nonzero_eigs={int(eigvals_cov.numel())}"
                )

                del gram
                gc.collect()

        tau_t = _estimate_tau_gpu(
            eigvals_cov_by_window,
            window_list,
            tau_mode=tau_mode,
            tau_ratio=tau_ratio,
            tau_abs=tau_abs,
            device=device_obj,
        )
        info_by_window = {
            window: _gaussian_info_from_cov_eigs_gpu(eigvals_cov_by_window[window], tau_t)
            for window in window_list
        }
        windows, infos, deltas = _compute_delta_infos_gpu(info_by_window, device=device_obj)
        return {
            "r_eff_by_window": r_eff_by_window,
            "info_by_window": {int(k): float(v.item()) for k, v in info_by_window.items()},
            "windows": windows,
            "infos": infos,
            "delta_infos": deltas,
            "tau": float(tau_t.item()),
            "num_samples": num_samples,
            "cache_mode": "gpu",
            "represented_cache_gb": represented_cache_gb,
            "peak_cuda_memory_gb": _peak_cuda_memory_gb(device_obj),
            "device": str(device_obj),
        }
    finally:
        del cumulative_gram, cumulative_norm_sq
        del kspace_cache
        gc.collect()
        torch.cuda.empty_cache()


def run_fastmri_hierarchy_job_gpu_only(
    *,
    dataset: SliceDataset,
    dataset_tag: str,
    output_prefix: os.PathLike[str] | str,
    tmp_dir: os.PathLike[str] | str,
    num_samples: int,
    uniform_train_resolution: Sequence[int],
    normalize_per_sample: bool,
    tau_mode: str,
    tau_ratio: float,
    tau_abs: Optional[float],
    block_size_gram: int,
    block_size_center: int,
    metadata: Optional[Mapping[str, Any]],
    representation: str,
    num_virtual_coils: int,
    calibration_window: int,
    device: str,
    cache_mode: str,
    gpu_cache_max_gb: float,
) -> Dict[str, Any]:
    del gpu_cache_max_gb
    device_obj = _ensure_cuda_device(device)
    cache_mode = _force_gpu_cache_mode(cache_mode)
    uniform_train_resolution = _normalize_resolution(uniform_train_resolution)
    max_window = min(uniform_train_resolution)
    window_list = list(range(2, max_window + 2, 2))
    _, text_path, json_path = _resolve_output_paths(output_prefix)

    print(f"[*] Dataset tag: {dataset_tag}")
    print(f"[*] Output text log: {text_path}")
    print(f"[*] Output json log: {json_path}")
    print(f"[*] Uniform train resolution: {uniform_train_resolution}")
    print(f"[*] Requested samples: {num_samples}")
    print(
        f"[*] Representation: {representation} | num_virtual_coils={num_virtual_coils} "
        f"| calibration_window={calibration_window}"
    )
    print(f"[*] Device: {device_obj} | cache_mode={cache_mode} | CPU stream disabled")

    results = calc_fastmri_cov_effective_rank_sizes_gpu_only(
        dataset=dataset,
        tmp_dir=tmp_dir,
        num_samples=num_samples,
        window_list=window_list,
        uniform_train_resolution=uniform_train_resolution,
        block_size_gram=block_size_gram,
        block_size_center=block_size_center,
        normalize_per_sample=normalize_per_sample,
        tau_mode=tau_mode,
        tau_ratio=tau_ratio,
        tau_abs=tau_abs,
        representation=representation,
        num_virtual_coils=num_virtual_coils,
        calibration_window=calibration_window,
        device=str(device_obj),
    )

    runtime_metadata = dict(metadata or {})
    runtime_metadata.update(
        {
            "device": str(device_obj),
            "cache_mode": str(results["cache_mode"]),
            "represented_cache_gb": float(results["represented_cache_gb"]),
            "peak_cuda_memory_gb": float(results["peak_cuda_memory_gb"]),
            "gpu_only_fastmri_entry": True,
        }
    )

    _write_text_log(
        text_path=text_path,
        windows=results["windows"],
        infos=results["infos"],
        deltas=results["delta_infos"],
        r_eff_by_window=results["r_eff_by_window"],
    )
    _write_json_log(
        json_path=json_path,
        dataset_tag=dataset_tag,
        uniform_train_resolution=uniform_train_resolution,
        num_samples=results["num_samples"],
        window_list=window_list,
        normalize_per_sample=normalize_per_sample,
        tau_mode=tau_mode,
        tau_ratio=tau_ratio,
        tau_abs=tau_abs,
        tau=results["tau"],
        windows=results["windows"],
        infos=results["infos"],
        deltas=results["delta_infos"],
        r_eff_by_window=results["r_eff_by_window"],
        metadata=runtime_metadata,
    )

    print(f"[*] Resolved cache mode: {results['cache_mode']}")
    print(f"[*] Represented cache size estimate: {results['represented_cache_gb']:.2f} GB")
    print(f"[*] Peak CUDA memory: {results['peak_cuda_memory_gb']:.2f} GB")
    print("\n========== spectral summary ==========")
    for window, info, delta in zip(
        results["windows"],
        results["infos"],
        results["delta_infos"],
    ):
        print(
            f"window={window:3d}x{window:<3d} | "
            f"r_eff={results['r_eff_by_window'][int(window)]:9.4f} | "
            f"gaussian_info={info:12.6f} | "
            f"delta_info={delta:10.6f}"
        )

    print(f"\n[*] global tau = {results['tau']:.6e}")
    print(f"[✔] Saved raw logs to {text_path} and {json_path}")
    return {
        **results,
        "text_path": text_path,
        "json_path": json_path,
    }


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute raw fixed k-space hierarchy statistics for fastMRI multi-coil data on GPU."
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
            "coil count in the fastMRI dataset for raw/virtual-coil representations."
        ),
    )
    parser.add_argument(
        "--calibration-window",
        type=int,
        default=DEFAULT_CALIBRATION_WINDOW,
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="gpu",
        choices=GPU_ONLY_CACHE_MODE_CHOICES,
        help="Compatibility option. 'auto' is accepted but forced to GPU; CPU stream is disabled.",
    )
    parser.add_argument("--gpu-cache-max-gb", type=float, default=80.0)
    return parser.parse_args()


def main() -> None:
    args = build_args()
    dataset = SliceDataset(
        root=Path(args.data_root),
        challenge="multicoil",
        transform=None,
    )
    if args.num_virtual_coils is None and args.representation != "rss_pseudo":
        num_virtual_coils = infer_max_available_kspace_coils(dataset)
        num_virtual_coils_source = "max_available_kspace_coils"
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

    run_fastmri_hierarchy_job_gpu_only(
        dataset=dataset,
        dataset_tag="fastmri_multicoil_384_gpu",
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
        metadata={
            "data_root": str(args.data_root),
            "dataset_name": "fastmri",
            "challenge": "multicoil",
            "source": "reconstruction_rss"
            if args.representation == "rss_pseudo"
            else "raw_kspace",
            "representation": str(args.representation),
            "num_virtual_coils": int(num_virtual_coils),
            "num_coils": int(num_virtual_coils),
            "num_virtual_coils_source": num_virtual_coils_source,
            "calibration_window": int(args.calibration_window),
        },
    )


if __name__ == "__main__":
    main()
