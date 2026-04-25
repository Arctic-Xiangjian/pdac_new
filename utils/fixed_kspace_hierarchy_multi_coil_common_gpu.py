from __future__ import annotations

import gc
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from fastmri import fft2c, ifft2c
from tqdm import tqdm


if __package__ in (None, ""):
    from fixed_kspace_hierarchy_multi_coil_common import (
        DEFAULT_CALIBRATION_WINDOW,
        DEFAULT_NUM_VIRTUAL_COILS,
        DEFAULT_REPRESENTATION,
        DEFAULT_UNIFORM_TRAIN_RESOLUTION,
        EPS,
        RAW_COIL_REPRESENTATION,
        REPRESENTATION_CHOICES,
        _build_raw_coil_representation,
        _compute_delta_infos,
        _crop_if_needed,
        _estimate_tau,
        _gaussian_info_from_cov_eigs,
        _normalize_even_window,
        _normalize_positive_int,
        _normalize_resolution,
        _pad_if_needed,
        _resolve_output_paths,
        _window_slice,
        _write_json_log,
        _write_text_log,
        infer_max_available_kspace_coils,
    )
else:
    from .fixed_kspace_hierarchy_multi_coil_common import (
        DEFAULT_CALIBRATION_WINDOW,
        DEFAULT_NUM_VIRTUAL_COILS,
        DEFAULT_REPRESENTATION,
        DEFAULT_UNIFORM_TRAIN_RESOLUTION,
        EPS,
        RAW_COIL_REPRESENTATION,
        REPRESENTATION_CHOICES,
        _build_raw_coil_representation,
        _compute_delta_infos,
        _crop_if_needed,
        _estimate_tau,
        _gaussian_info_from_cov_eigs,
        _normalize_even_window,
        _normalize_positive_int,
        _normalize_resolution,
        _pad_if_needed,
        _resolve_output_paths,
        _window_slice,
        _write_json_log,
        _write_text_log,
        infer_max_available_kspace_coils,
    )


CACHE_MODE_CHOICES = ("auto", "gpu", "cpu_stream")


def _ensure_cuda_device(device: str | torch.device) -> torch.device:
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        raise ValueError(f"GPU hierarchy expects a CUDA device, got {device_obj!s}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable, but a GPU hierarchy run was requested.")
    torch.cuda.set_device(device_obj)
    return device_obj


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.synchronize()


def _bytes_to_gb(num_bytes: int | float) -> float:
    return float(num_bytes) / float(1024**3)


def _peak_cuda_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    with torch.cuda.device(device):
        return _bytes_to_gb(torch.cuda.max_memory_allocated())


def _estimate_cache_nbytes(
    num_samples: int,
    num_channels: int,
    height: int,
    width: int,
) -> int:
    return int(num_samples) * int(num_channels) * int(height) * int(width) * 2 * 4


def _resolve_cache_mode(
    cache_mode: str,
    *,
    estimated_cache_nbytes: int,
    gpu_cache_max_gb: float,
    device: Optional[torch.device] = None,
) -> str:
    if cache_mode not in CACHE_MODE_CHOICES:
        raise ValueError(
            f"Unsupported cache_mode: {cache_mode!r}. Expected one of {CACHE_MODE_CHOICES}."
        )
    if cache_mode != "auto":
        return cache_mode

    max_cache_nbytes = int(float(gpu_cache_max_gb) * 1024**3)
    if device is not None and device.type == "cuda":
        free_nbytes, _ = torch.cuda.mem_get_info(device)
        max_cache_nbytes = min(max_cache_nbytes, int(free_nbytes * 0.5))

    if int(estimated_cache_nbytes) <= max_cache_nbytes:
        return "gpu"
    return "cpu_stream"


def _representation_num_channels(representation: str, num_virtual_coils: int) -> int:
    if representation == "rss_pseudo":
        return 1
    if representation in (DEFAULT_REPRESENTATION, RAW_COIL_REPRESENTATION):
        return _normalize_positive_int("num_virtual_coils", num_virtual_coils)
    raise ValueError(
        f"Unsupported representation: {representation!r}. "
        f"Expected one of {REPRESENTATION_CHOICES}."
    )


def target_to_uniform_kspace(
    target: np.ndarray | torch.Tensor,
    uniform_train_resolution: Sequence[int],
    *,
    device: str | torch.device,
) -> torch.Tensor:
    if target is None:
        raise ValueError("Dataset target is missing; expected reconstruction_rss.")

    device_obj = _ensure_cuda_device(device)
    uniform_train_resolution = _normalize_resolution(uniform_train_resolution)
    target_t = torch.as_tensor(target, dtype=torch.float32, device=device_obj)
    if target_t.ndim != 2:
        raise ValueError(f"Expected target with shape [H, W], got {tuple(target_t.shape)}")

    image = torch.stack([target_t, torch.zeros_like(target_t)], dim=-1).unsqueeze(0)
    image = _crop_if_needed(image, uniform_train_resolution)
    image = _pad_if_needed(image, uniform_train_resolution)
    return fft2c(image).squeeze(0).contiguous()


def _raw_kspace_to_tensor(kspace: Any, *, device: str | torch.device) -> torch.Tensor:
    device_obj = _ensure_cuda_device(device)

    if torch.is_tensor(kspace):
        if kspace.is_complex():
            kspace_t = kspace.detach().to(device=device_obj, dtype=torch.complex64)
        elif kspace.ndim >= 1 and kspace.shape[-1] == 2:
            kspace_t = kspace.detach().to(device=device_obj, dtype=torch.float32)
        else:
            raise ValueError(f"Expected complex-valued k-space tensor, got {tuple(kspace.shape)}")
    else:
        kspace_np = np.asarray(kspace)
        if np.iscomplexobj(kspace_np):
            kspace_t = torch.from_numpy(np.asarray(kspace_np, dtype=np.complex64)).to(
                device=device_obj
            )
        elif kspace_np.ndim >= 1 and kspace_np.shape[-1] == 2:
            kspace_t = torch.from_numpy(np.asarray(kspace_np, dtype=np.float32)).to(
                device=device_obj
            )
        else:
            raise ValueError(f"Expected complex-valued k-space array, got {kspace_np.shape}")

    if torch.is_complex(kspace_t):
        if kspace_t.ndim == 2:
            kspace_t = kspace_t.unsqueeze(0)
        elif kspace_t.ndim == 4:
            height, width = kspace_t.shape[-2:]
            kspace_t = kspace_t.reshape(-1, height, width)
        if kspace_t.ndim != 3:
            raise ValueError(f"Expected k-space with shape [coils, H, W], got {tuple(kspace_t.shape)}")
        return torch.view_as_real(kspace_t.to(torch.complex64).contiguous())

    if kspace_t.ndim == 2:
        kspace_t = kspace_t.unsqueeze(0)
    elif kspace_t.ndim == 4:
        height, width = kspace_t.shape[-3:-1]
        kspace_t = kspace_t.reshape(-1, height, width, 2)

    if kspace_t.ndim != 4 or kspace_t.shape[-1] != 2:
        raise ValueError(f"Expected k-space with shape [coils, H, W, 2], got {tuple(kspace_t.shape)}")

    return kspace_t.to(dtype=torch.float32).contiguous()


def _resize_kspace_to_uniform_resolution(
    kspace: torch.Tensor,
    uniform_train_resolution: Sequence[int],
) -> torch.Tensor:
    uniform_train_resolution = _normalize_resolution(uniform_train_resolution)
    image = ifft2c(kspace)
    image = _crop_if_needed(image, uniform_train_resolution)
    image = _pad_if_needed(image, uniform_train_resolution)
    return fft2c(image).contiguous()


def _apply_phase_anchor(
    full_kspace_complex: torch.Tensor,
    calibration_images_complex: torch.Tensor,
) -> torch.Tensor:
    mean_complex = calibration_images_complex.reshape(calibration_images_complex.shape[0], -1).mean(
        dim=1
    )
    phase = torch.ones_like(mean_complex)
    valid = mean_complex.abs() > EPS
    phase[valid] = mean_complex[valid] / mean_complex[valid].abs()
    return full_kspace_complex * phase.conj()[:, None, None]


def _build_virtual_coil_representation(
    resized_kspace: torch.Tensor,
    *,
    num_virtual_coils: int,
    calibration_window: int,
) -> torch.Tensor:
    num_virtual_coils = _normalize_positive_int("num_virtual_coils", num_virtual_coils)
    num_input_coils, height, width, _ = resized_kspace.shape
    calibration_window = _normalize_even_window(
        calibration_window,
        min(height, width),
        name="calibration_window",
    )

    ys, xs = _window_slice(height, width, calibration_window)
    calibration_kspace = resized_kspace[:, ys, xs, :]
    calibration_images_complex = torch.view_as_complex(ifft2c(calibration_kspace).contiguous())
    full_kspace_complex = torch.view_as_complex(resized_kspace.contiguous())

    coil_matrix = calibration_images_complex.reshape(num_input_coils, -1)
    basis, singular_values, _ = torch.linalg.svd(coil_matrix, full_matrices=False)
    if singular_values.numel() == 0:
        raise ValueError("Virtual-coil compression received an empty calibration matrix.")

    rank = min(num_virtual_coils, num_input_coils)
    basis = basis[:, :rank]

    compressed_kspace = torch.einsum("cr,chw->rhw", basis.conj(), full_kspace_complex)
    compressed_calibration_images = torch.einsum(
        "cr,chw->rhw", basis.conj(), calibration_images_complex
    )
    compressed_kspace = _apply_phase_anchor(
        compressed_kspace,
        compressed_calibration_images,
    )

    padded = torch.zeros(
        num_virtual_coils,
        height,
        width,
        dtype=torch.complex64,
        device=resized_kspace.device,
    )
    padded[:rank] = compressed_kspace.to(torch.complex64)
    return torch.view_as_real(padded).to(dtype=torch.float32).contiguous()


def _build_kspace_representation(
    raw_kspace: Any,
    *,
    target: Optional[np.ndarray],
    uniform_train_resolution: Sequence[int],
    representation: str,
    num_virtual_coils: int,
    calibration_window: int,
    device: str | torch.device,
) -> torch.Tensor:
    if representation == "rss_pseudo":
        return target_to_uniform_kspace(
            target,
            uniform_train_resolution,
            device=device,
        ).unsqueeze(0)

    if representation not in (DEFAULT_REPRESENTATION, RAW_COIL_REPRESENTATION):
        raise ValueError(
            f"Unsupported representation: {representation!r}. "
            f"Expected one of {REPRESENTATION_CHOICES}."
        )

    resized_kspace = _resize_kspace_to_uniform_resolution(
        _raw_kspace_to_tensor(raw_kspace, device=device),
        uniform_train_resolution,
    )
    if representation == RAW_COIL_REPRESENTATION:
        return _build_raw_coil_representation(
            resized_kspace,
            num_coils=num_virtual_coils,
        )

    return _build_virtual_coil_representation(
        resized_kspace,
        num_virtual_coils=num_virtual_coils,
        calibration_window=calibration_window,
    )


def _load_centered_kspace_cache(
    dataset,
    num_samples: int,
    cache_path: os.PathLike[str] | str,
    uniform_train_resolution: Sequence[int],
    *,
    representation: str,
    num_virtual_coils: int,
    calibration_window: int,
    device: str | torch.device,
    cache_mode: str,
    gpu_cache_max_gb: float,
) -> Tuple[torch.Tensor | np.memmap, str, Optional[Path], float]:
    device_obj = _ensure_cuda_device(device)
    height, width = _normalize_resolution(uniform_train_resolution)
    num_channels = _representation_num_channels(representation, num_virtual_coils)
    estimated_cache_nbytes = _estimate_cache_nbytes(num_samples, num_channels, height, width)
    resolved_cache_mode = _resolve_cache_mode(
        cache_mode,
        estimated_cache_nbytes=estimated_cache_nbytes,
        gpu_cache_max_gb=gpu_cache_max_gb,
        device=device_obj,
    )

    cache_path_obj: Optional[Path] = None
    if resolved_cache_mode == "gpu":
        kspace_cache: torch.Tensor | np.memmap = torch.empty(
            (num_samples, num_channels, height, width, 2),
            dtype=torch.float32,
            device=device_obj,
        )
    else:
        cache_path_obj = Path(cache_path)
        kspace_cache = np.memmap(
            cache_path_obj,
            mode="w+",
            dtype=np.float32,
            shape=(num_samples, num_channels, height, width, 2),
        )

    print(
        f"[*] Caching {num_samples} samples of centered {height}x{width} {representation} "
        f"k-space with {num_channels} channel(s) to {resolved_cache_mode} cache "
        f"(estimated={_bytes_to_gb(estimated_cache_nbytes):.2f} GB)"
    )

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Caching centered k-space"):
            sample = dataset[idx]
            raw_kspace = sample[0]
            target = sample[2] if len(sample) > 2 else None
            represented = _build_kspace_representation(
                raw_kspace,
                target=target,
                uniform_train_resolution=(height, width),
                representation=representation,
                num_virtual_coils=num_virtual_coils,
                calibration_window=calibration_window,
                device=device_obj,
            )
            expected_shape = (num_channels, height, width, 2)
            if tuple(represented.shape) != expected_shape:
                raise ValueError(
                    f"Expected represented k-space shape {expected_shape}, got {tuple(represented.shape)}"
                )

            if isinstance(kspace_cache, torch.Tensor):
                kspace_cache[idx].copy_(represented)
            else:
                kspace_cache[idx] = represented.detach().cpu().numpy().astype(np.float32)

    _synchronize(device_obj)
    if isinstance(kspace_cache, np.memmap):
        kspace_cache.flush()

    return (
        kspace_cache,
        resolved_cache_mode,
        cache_path_obj,
        _bytes_to_gb(estimated_cache_nbytes),
    )


def _cache_block_to_device(
    kspace_cache: torch.Tensor | np.memmap,
    i0: int,
    i1: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(kspace_cache, torch.Tensor):
        return kspace_cache[i0:i1]

    return torch.from_numpy(np.asarray(kspace_cache[i0:i1], dtype=np.float32)).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )


def _extract_window_vectors_torch(kspace_block: torch.Tensor, window_size: int) -> torch.Tensor:
    _, _, height, width, comp = kspace_block.shape
    if comp != 2:
        raise ValueError(f"Expected last dimension 2, got {comp}")

    ys, xs = _window_slice(height, width, window_size)
    return (
        kspace_block[:, :, ys, xs, :]
        .reshape(kspace_block.shape[0], -1)
        .to(dtype=torch.float64)
    )


def _extract_shell_vectors_torch(kspace_block: torch.Tensor, window_size: int) -> torch.Tensor:
    _, _, height, width, comp = kspace_block.shape
    if comp != 2:
        raise ValueError(f"Expected last dimension 2, got {comp}")

    ys, xs = _window_slice(height, width, window_size)
    parts = [
        kspace_block[:, :, ys.start : ys.start + 1, xs, :].reshape(kspace_block.shape[0], -1),
        kspace_block[:, :, ys.stop - 1 : ys.stop, xs, :].reshape(kspace_block.shape[0], -1),
    ]

    inner_y_start = ys.start + 1
    inner_y_stop = ys.stop - 1
    if inner_y_start < inner_y_stop:
        parts.extend(
            [
                kspace_block[:, :, inner_y_start:inner_y_stop, xs.start : xs.start + 1, :].reshape(
                    kspace_block.shape[0], -1
                ),
                kspace_block[:, :, inner_y_start:inner_y_stop, xs.stop - 1 : xs.stop, :].reshape(
                    kspace_block.shape[0], -1
                ),
            ]
        )

    return torch.cat(parts, dim=1).to(dtype=torch.float64)


def _compute_window_gram_direct(
    kspace_cache: torch.Tensor | np.ndarray | np.memmap,
    window_size: int,
    *,
    block_size: int = 256,
    normalize_per_sample: bool = False,
    device: str | torch.device = "cuda:0",
) -> torch.Tensor:
    device_obj = _ensure_cuda_device(device)
    if isinstance(kspace_cache, np.ndarray) and not isinstance(kspace_cache, np.memmap):
        kspace_cache = torch.from_numpy(kspace_cache).to(device=device_obj, dtype=torch.float32)

    num_samples = int(kspace_cache.shape[0])
    gram = torch.zeros((num_samples, num_samples), dtype=torch.float64, device=device_obj)

    with torch.no_grad():
        for i0 in range(0, num_samples, block_size):
            i1 = min(i0 + block_size, num_samples)
            xi = _extract_window_vectors_torch(
                _cache_block_to_device(kspace_cache, i0, i1, device=device_obj),
                window_size,
            )

            if normalize_per_sample:
                xi = xi / torch.clamp(torch.linalg.norm(xi, dim=1, keepdim=True), min=EPS)

            for j0 in range(i0, num_samples, block_size):
                j1 = min(j0 + block_size, num_samples)
                xj = _extract_window_vectors_torch(
                    _cache_block_to_device(kspace_cache, j0, j1, device=device_obj),
                    window_size,
                )

                if normalize_per_sample:
                    xj = xj / torch.clamp(torch.linalg.norm(xj, dim=1, keepdim=True), min=EPS)

                block = xi @ xj.T
                gram[i0:i1, j0:j1] = block
                if j0 != i0:
                    gram[j0:j1, i0:i1] = block.T

    return gram


def _accumulate_shell_gram_inplace(
    kspace_cache: torch.Tensor | np.memmap,
    *,
    cumulative_gram: torch.Tensor,
    cumulative_norm_sq: torch.Tensor,
    window_size: int,
    block_size: int = 256,
    device: str | torch.device = "cuda:0",
) -> None:
    device_obj = _ensure_cuda_device(device)
    num_samples = int(kspace_cache.shape[0])

    print(f"[*] Accumulating shell contribution for window={window_size}")
    with torch.no_grad():
        for i0 in tqdm(range(0, num_samples, block_size), desc=f"Shell {window_size}x{window_size}"):
            i1 = min(i0 + block_size, num_samples)
            xi = _extract_shell_vectors_torch(
                _cache_block_to_device(kspace_cache, i0, i1, device=device_obj),
                window_size,
            )
            cumulative_norm_sq[i0:i1].add_(torch.einsum("ij,ij->i", xi, xi))

            for j0 in range(i0, num_samples, block_size):
                j1 = min(j0 + block_size, num_samples)
                xj = _extract_shell_vectors_torch(
                    _cache_block_to_device(kspace_cache, j0, j1, device=device_obj),
                    window_size,
                )
                block = xi @ xj.T

                cumulative_gram[i0:i1, j0:j1].add_(block)
                if j0 != i0:
                    cumulative_gram[j0:j1, i0:i1].add_(block.T)


def _materialize_window_gram(
    cumulative_gram: torch.Tensor,
    cumulative_norm_sq: torch.Tensor,
    gram_path: os.PathLike[str] | str | None = None,
    *,
    block_size: int = 512,
    normalize_per_sample: bool = False,
) -> torch.Tensor:
    del gram_path, block_size
    gram = cumulative_gram.clone()
    if normalize_per_sample:
        inv_norm = torch.rsqrt(torch.clamp(cumulative_norm_sq, min=EPS))
        gram = gram * inv_norm[:, None] * inv_norm[None, :]
    return gram


def _center_gram_inplace(gram: torch.Tensor, block_size: int = 512) -> torch.Tensor:
    del block_size
    print("[*] Centering Gram matrix...")
    row_mean = gram.mean(dim=1)
    grand_mean = row_mean.mean()
    gram.sub_(row_mean[:, None])
    gram.sub_(row_mean[None, :])
    gram.add_(grand_mean)
    return gram


def _spectral_stats_from_centered_gram(
    gram: torch.Tensor,
    n_samples: int,
    eps_ratio: float = 1e-12,
) -> Tuple[float, np.ndarray, np.ndarray]:
    print("[*] Eigendecomposition of centered Gram...")
    gram = 0.5 * (gram + gram.T)
    eigvals_gram = torch.linalg.eigvalsh(gram)
    eigvals_gram = torch.clamp(eigvals_gram, min=0.0)

    if eigvals_gram.numel() == 0 or float(eigvals_gram[-1].item()) <= 0.0:
        eigvals_cpu = eigvals_gram.detach().cpu().numpy()
        return 1.0, eigvals_cpu.copy(), eigvals_cpu.copy()

    thresh = eigvals_gram[-1] * float(eps_ratio)
    eigvals_gram = eigvals_gram[eigvals_gram > thresh]
    if eigvals_gram.numel() == 0:
        eigvals_cpu = eigvals_gram.detach().cpu().numpy()
        return 1.0, eigvals_cpu.copy(), eigvals_cpu.copy()

    probs = eigvals_gram / torch.clamp(eigvals_gram.sum(), min=EPS)
    entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=EPS)))
    r_eff = float(torch.exp(entropy).item())

    eigvals_cov = eigvals_gram / max(n_samples - 1, 1)
    return (
        r_eff,
        eigvals_gram.detach().cpu().numpy(),
        eigvals_cov.detach().cpu().numpy(),
    )


def calc_cov_effective_rank_sizes(
    dataset,
    tmp_dir: os.PathLike[str] | str,
    num_samples: int,
    window_list: Sequence[int],
    uniform_train_resolution: Sequence[int] = DEFAULT_UNIFORM_TRAIN_RESOLUTION,
    block_size_gram: int = 256,
    block_size_center: int = 512,
    normalize_per_sample: bool = True,
    tau_mode: str = "full_tail_median_ratio",
    tau_ratio: float = 1.0,
    tau_abs: Optional[float] = None,
    representation: str = DEFAULT_REPRESENTATION,
    num_virtual_coils: int = DEFAULT_NUM_VIRTUAL_COILS,
    calibration_window: int = DEFAULT_CALIBRATION_WINDOW,
    device: str = "cuda:0",
    cache_mode: str = "auto",
    gpu_cache_max_gb: float = 60.0,
) -> Dict[str, Any]:
    device_obj = _ensure_cuda_device(device)
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

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

    cache_path = tmp_dir / f"kspace_cache_{num_samples}_gpu.dat"
    kspace_cache, resolved_cache_mode, cache_file_path, represented_cache_gb = _load_centered_kspace_cache(
        dataset=dataset,
        num_samples=num_samples,
        cache_path=cache_path,
        uniform_train_resolution=uniform_train_resolution,
        representation=representation,
        num_virtual_coils=num_virtual_coils,
        calibration_window=calibration_window,
        device=device_obj,
        cache_mode=cache_mode,
        gpu_cache_max_gb=gpu_cache_max_gb,
    )

    cumulative_gram = torch.zeros(
        (num_samples, num_samples),
        dtype=torch.float64,
        device=device_obj,
    )
    cumulative_norm_sq = torch.zeros(num_samples, dtype=torch.float64, device=device_obj)

    r_eff_by_window: Dict[int, float] = {}
    eigvals_cov_by_window: Dict[int, np.ndarray] = {}
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

                r_eff, eigvals_gram, eigvals_cov = _spectral_stats_from_centered_gram(
                    gram,
                    n_samples=num_samples,
                )
                r_eff_by_window[window_size] = r_eff
                eigvals_cov_by_window[window_size] = eigvals_cov

                print(
                    f"[CovRank] window={window_size:3d}x{window_size:<3d} | "
                    f"r_eff={r_eff:10.4f} | nonzero_eigs={len(eigvals_cov)}"
                )

                del gram, eigvals_gram
                gc.collect()

        tau = _estimate_tau(
            eigvals_cov_by_window=eigvals_cov_by_window,
            window_list=window_list,
            tau_mode=tau_mode,
            tau_ratio=tau_ratio,
            tau_abs=tau_abs,
        )
        info_by_window = {
            window: _gaussian_info_from_cov_eigs(eigvals_cov_by_window[window], tau=tau)
            for window in window_list
        }
        windows, infos, deltas = _compute_delta_infos(info_by_window)
        return {
            "r_eff_by_window": r_eff_by_window,
            "info_by_window": info_by_window,
            "windows": windows,
            "infos": infos,
            "delta_infos": deltas,
            "tau": tau,
            "num_samples": num_samples,
            "cache_mode": resolved_cache_mode,
            "represented_cache_gb": represented_cache_gb,
            "peak_cuda_memory_gb": _peak_cuda_memory_gb(device_obj),
            "device": str(device_obj),
        }
    finally:
        del cumulative_gram, cumulative_norm_sq
        del kspace_cache
        gc.collect()
        torch.cuda.empty_cache()
        if cache_file_path is not None:
            try:
                cache_file_path.unlink()
            except OSError:
                pass


def run_hierarchy_job(
    *,
    dataset,
    dataset_tag: str,
    output_prefix: os.PathLike[str] | str,
    tmp_dir: os.PathLike[str] | str,
    num_samples: int,
    uniform_train_resolution: Sequence[int] = DEFAULT_UNIFORM_TRAIN_RESOLUTION,
    normalize_per_sample: bool = True,
    tau_mode: str = "full_tail_median_ratio",
    tau_ratio: float = 1.0,
    tau_abs: Optional[float] = None,
    block_size_gram: int = 256,
    block_size_center: int = 512,
    metadata: Optional[Mapping[str, Any]] = None,
    representation: str = DEFAULT_REPRESENTATION,
    num_virtual_coils: int = DEFAULT_NUM_VIRTUAL_COILS,
    calibration_window: int = DEFAULT_CALIBRATION_WINDOW,
    device: str = "cuda:0",
    cache_mode: str = "auto",
    gpu_cache_max_gb: float = 60.0,
) -> Dict[str, Any]:
    device_obj = _ensure_cuda_device(device)
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
    print(
        f"[*] Device: {device_obj} | cache_mode={cache_mode} | "
        f"gpu_cache_max_gb={float(gpu_cache_max_gb):.1f}"
    )

    results = calc_cov_effective_rank_sizes(
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
        cache_mode=cache_mode,
        gpu_cache_max_gb=gpu_cache_max_gb,
    )

    runtime_metadata = dict(metadata or {})
    runtime_metadata.update(
        {
            "device": str(device_obj),
            "cache_mode": str(results["cache_mode"]),
            "gpu_cache_max_gb": float(gpu_cache_max_gb),
            "represented_cache_gb": float(results["represented_cache_gb"]),
            "peak_cuda_memory_gb": float(results["peak_cuda_memory_gb"]),
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


def benchmark_hierarchy_job(**kwargs: Any) -> Dict[str, Any]:
    device_obj = _ensure_cuda_device(kwargs.get("device", "cuda:0"))
    with torch.cuda.device(device_obj):
        torch.cuda.reset_peak_memory_stats()
    _synchronize(device_obj)
    start = time.perf_counter()
    results = run_hierarchy_job(**kwargs)
    _synchronize(device_obj)
    elapsed_seconds = time.perf_counter() - start
    benchmark = {
        "elapsed_seconds": float(elapsed_seconds),
        "peak_cuda_memory_gb": _peak_cuda_memory_gb(device_obj),
        "cache_mode": str(results["cache_mode"]),
        "device": str(device_obj),
    }
    return {
        **results,
        "benchmark": benchmark,
    }
