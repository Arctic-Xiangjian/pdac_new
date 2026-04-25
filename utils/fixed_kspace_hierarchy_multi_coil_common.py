from __future__ import annotations

import gc
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastmri import fft2c, ifft2c
from tqdm import tqdm


EPS = 1e-12
DEFAULT_UNIFORM_TRAIN_RESOLUTION = (384, 384)
DEFAULT_REPRESENTATION = "virtual_coil_pca"
RAW_COIL_REPRESENTATION = "raw_coil"
REPRESENTATION_CHOICES = (DEFAULT_REPRESENTATION, RAW_COIL_REPRESENTATION, "rss_pseudo")
DEFAULT_NUM_VIRTUAL_COILS = 4
DEFAULT_CALIBRATION_WINDOW = 64


def _normalize_resolution(resolution: Sequence[int]) -> Tuple[int, int]:
    if len(resolution) != 2:
        raise ValueError(
            f"uniform_train_resolution must contain 2 integers, got {resolution!r}"
        )
    height, width = int(resolution[0]), int(resolution[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"uniform_train_resolution must be positive, got {resolution!r}")
    return height, width


def _normalize_positive_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return normalized


def _kspace_shape_num_coils(shape: Sequence[int]) -> int:
    if len(shape) in (2, 3):
        return 1
    if len(shape) >= 4:
        return int(shape[1])
    raise ValueError(f"Expected k-space shape with at least 2 dims, got {tuple(shape)}")


def infer_max_available_kspace_coils(dataset) -> int:
    raw_samples = getattr(dataset, "raw_samples", None)
    num_adj_slices = _normalize_positive_int(
        "num_adj_slices",
        int(getattr(dataset, "num_adj_slices", 1) or 1),
    )

    max_coils = 0
    if raw_samples is not None:
        sample_paths = sorted({Path(sample[0]) for sample in raw_samples})
        if sample_paths:
            import h5py

            for sample_path in sample_paths:
                with h5py.File(sample_path, "r") as hf:
                    if "kspace" not in hf:
                        raise KeyError(f"{sample_path} does not contain a 'kspace' dataset")
                    coil_count = _kspace_shape_num_coils(hf["kspace"].shape)
                    max_coils = max(
                        max_coils,
                        coil_count * num_adj_slices,
                    )
            if max_coils > 0:
                return max_coils

    for idx in range(len(dataset)):
        sample = dataset[idx]
        max_coils = max(max_coils, int(_raw_kspace_to_tensor(sample[0]).shape[0]))

    if max_coils <= 0:
        raise ValueError("Could not infer a positive k-space coil count from dataset.")
    return max_coils


def _normalize_even_window(
    window_size: int, max_size: int, *, name: str = "window_size"
) -> int:
    normalized = _normalize_positive_int(name, window_size)
    if normalized % 2 != 0:
        raise ValueError(f"{name} must be even, got {window_size!r}")
    if normalized > max_size:
        raise ValueError(f"{name}={normalized} exceeds max_size={max_size}")
    return normalized


def _flush_array(arr: np.ndarray) -> None:
    flush = getattr(arr, "flush", None)
    if callable(flush):
        flush()


def _ensure_realimag_last2(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"Expected [H, W, 2], got shape={arr.shape}")
    return arr


def _crop_if_needed(
    image: torch.Tensor, uniform_train_resolution: Tuple[int, int]
) -> torch.Tensor:
    target_h, target_w = uniform_train_resolution
    current_h, current_w = image.shape[-3], image.shape[-2]

    h_from = max((current_h - target_h) // 2, 0)
    w_from = max((current_w - target_w) // 2, 0)
    h_to = h_from + min(target_h, current_h)
    w_to = w_from + min(target_w, current_w)

    return image[..., h_from:h_to, w_from:w_to, :]


def _pad_if_needed(
    image: torch.Tensor, uniform_train_resolution: Tuple[int, int]
) -> torch.Tensor:
    target_h, target_w = uniform_train_resolution
    pad_h = target_h - image.shape[-3]
    pad_w = target_w - image.shape[-2]

    if pad_h <= 0 and pad_w <= 0:
        return image

    pad_h_top = max(pad_h // 2, 0)
    pad_h_bottom = max(pad_h - pad_h_top, 0)
    pad_w_left = max(pad_w // 2, 0)
    pad_w_right = max(pad_w - pad_w_left, 0)

    if image.ndim != 4 or image.shape[-1] != 2:
        raise ValueError(f"Expected [N, H, W, 2], got shape={tuple(image.shape)}")

    image = image.permute(0, 3, 1, 2)
    image = F.pad(
        image,
        (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom),
        mode="reflect",
    )
    return image.permute(0, 2, 3, 1)


def target_to_uniform_kspace(
    target: np.ndarray, uniform_train_resolution: Sequence[int]
) -> np.ndarray:
    if target is None:
        raise ValueError("Dataset target is missing; expected reconstruction_rss.")

    uniform_train_resolution = _normalize_resolution(uniform_train_resolution)
    target_t = torch.as_tensor(target, dtype=torch.float32)
    if target_t.ndim != 2:
        raise ValueError(f"Expected target with shape [H, W], got {tuple(target_t.shape)}")

    image = torch.stack([target_t, torch.zeros_like(target_t)], dim=-1).unsqueeze(0)
    image = _crop_if_needed(image, uniform_train_resolution)
    image = _pad_if_needed(image, uniform_train_resolution)
    kspace = fft2c(image).squeeze(0)
    return _ensure_realimag_last2(kspace.cpu().numpy().astype(np.float32))


def _raw_kspace_to_tensor(kspace: Any) -> torch.Tensor:
    if torch.is_tensor(kspace):
        kspace_t = kspace.detach().cpu()
        if kspace_t.is_complex():
            pass
        elif kspace_t.ndim >= 1 and kspace_t.shape[-1] == 2:
            kspace_t = torch.view_as_complex(kspace_t.contiguous())
        else:
            raise ValueError(f"Expected complex-valued k-space tensor, got {tuple(kspace_t.shape)}")
    else:
        kspace_np = np.asarray(kspace)
        if np.iscomplexobj(kspace_np):
            kspace_t = torch.from_numpy(np.asarray(kspace_np, dtype=np.complex64))
        elif kspace_np.ndim >= 1 and kspace_np.shape[-1] == 2:
            kspace_t = torch.view_as_complex(
                torch.from_numpy(np.asarray(kspace_np, dtype=np.float32)).contiguous()
            )
        else:
            raise ValueError(f"Expected complex-valued k-space array, got {kspace_np.shape}")

    if kspace_t.ndim == 2:
        kspace_t = kspace_t.unsqueeze(0)
    elif kspace_t.ndim == 4:
        height, width = kspace_t.shape[-2:]
        kspace_t = kspace_t.reshape(-1, height, width)

    if kspace_t.ndim != 3:
        raise ValueError(f"Expected k-space with shape [coils, H, W], got {tuple(kspace_t.shape)}")

    return torch.view_as_real(kspace_t.to(torch.complex64).contiguous())


def _resize_kspace_to_uniform_resolution(
    kspace: torch.Tensor, uniform_train_resolution: Sequence[int]
) -> torch.Tensor:
    uniform_train_resolution = _normalize_resolution(uniform_train_resolution)
    image = ifft2c(kspace)
    image = _crop_if_needed(image, uniform_train_resolution)
    image = _pad_if_needed(image, uniform_train_resolution)
    return fft2c(image)


def _window_slice(height: int, width: int, window_size: int) -> Tuple[slice, slice]:
    if window_size % 2 != 0:
        raise ValueError(f"window_size must be even, got {window_size}")
    cy, cx = height // 2, width // 2
    half = window_size // 2
    return slice(cy - half, cy + half), slice(cx - half, cx + half)


def _apply_phase_anchor(
    full_kspace_complex: torch.Tensor, calibration_images_complex: torch.Tensor
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
        compressed_kspace, compressed_calibration_images
    )

    padded = torch.zeros(
        num_virtual_coils,
        height,
        width,
        dtype=torch.complex64,
    )
    padded[:rank] = compressed_kspace.to(torch.complex64)
    return torch.view_as_real(padded)


def _build_raw_coil_representation(
    resized_kspace: torch.Tensor,
    *,
    num_coils: int,
) -> torch.Tensor:
    num_coils = _normalize_positive_int("num_coils", num_coils)
    num_input_coils, height, width, components = resized_kspace.shape
    if components != 2:
        raise ValueError(f"Expected real/imag component dimension 2, got {components}")
    if num_input_coils > num_coils:
        raise ValueError(
            f"Input has {num_input_coils} coils, but output was configured for "
            f"{num_coils} coils."
        )
    if num_input_coils == num_coils:
        return resized_kspace.to(dtype=torch.float32).contiguous()

    padded = torch.zeros(
        num_coils,
        height,
        width,
        2,
        dtype=torch.float32,
        device=resized_kspace.device,
    )
    padded[:num_input_coils] = resized_kspace.to(dtype=torch.float32)
    return padded.contiguous()


def _build_kspace_representation(
    raw_kspace: Any,
    *,
    target: Optional[np.ndarray],
    uniform_train_resolution: Sequence[int],
    representation: str,
    num_virtual_coils: int,
    calibration_window: int,
) -> torch.Tensor:
    if representation == "rss_pseudo":
        return torch.from_numpy(
            target_to_uniform_kspace(target, uniform_train_resolution)
        ).unsqueeze(0)

    if representation not in (DEFAULT_REPRESENTATION, RAW_COIL_REPRESENTATION):
        raise ValueError(
            f"Unsupported representation: {representation!r}. "
            f"Expected one of {REPRESENTATION_CHOICES}."
        )

    resized_kspace = _resize_kspace_to_uniform_resolution(
        _raw_kspace_to_tensor(raw_kspace),
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


def _representation_num_channels(representation: str, num_virtual_coils: int) -> int:
    if representation == "rss_pseudo":
        return 1
    if representation in (DEFAULT_REPRESENTATION, RAW_COIL_REPRESENTATION):
        return _normalize_positive_int("num_virtual_coils", num_virtual_coils)
    raise ValueError(
        f"Unsupported representation: {representation!r}. "
        f"Expected one of {REPRESENTATION_CHOICES}."
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
) -> np.memmap:
    height, width = _normalize_resolution(uniform_train_resolution)
    num_channels = _representation_num_channels(representation, num_virtual_coils)
    kspace_cache = np.memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, num_channels, height, width, 2),
    )

    print(
        f"[*] Caching {num_samples} samples of centered {height}x{width} {representation} "
        f"k-space with {num_channels} channel(s) to: {cache_path}"
    )
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
        )
        expected_shape = (num_channels, height, width, 2)
        if tuple(represented.shape) != expected_shape:
            raise ValueError(
                f"Expected represented k-space shape {expected_shape}, got {tuple(represented.shape)}"
            )
        kspace_cache[idx] = represented.cpu().numpy().astype(np.float32)

    kspace_cache.flush()
    return kspace_cache


def _extract_window_vectors(kspace_block: np.ndarray, window_size: int) -> np.ndarray:
    _, _, height, width, comp = kspace_block.shape
    if comp != 2:
        raise ValueError(f"Expected last dimension 2, got {comp}")

    ys, xs = _window_slice(height, width, window_size)
    return np.asarray(kspace_block[:, :, ys, xs, :], dtype=np.float64).reshape(
        kspace_block.shape[0], -1
    )


def _extract_shell_vectors(kspace_block: np.ndarray, window_size: int) -> np.ndarray:
    _, _, height, width, comp = kspace_block.shape
    if comp != 2:
        raise ValueError(f"Expected last dimension 2, got {comp}")

    ys, xs = _window_slice(height, width, window_size)
    parts = [
        np.asarray(
            kspace_block[:, :, ys.start : ys.start + 1, xs, :],
            dtype=np.float64,
        ).reshape(kspace_block.shape[0], -1),
        np.asarray(
            kspace_block[:, :, ys.stop - 1 : ys.stop, xs, :],
            dtype=np.float64,
        ).reshape(kspace_block.shape[0], -1),
    ]

    inner_y_start = ys.start + 1
    inner_y_stop = ys.stop - 1
    if inner_y_start < inner_y_stop:
        parts.extend(
            [
                np.asarray(
                    kspace_block[:, :, inner_y_start:inner_y_stop, xs.start : xs.start + 1, :],
                    dtype=np.float64,
                ).reshape(kspace_block.shape[0], -1),
                np.asarray(
                    kspace_block[:, :, inner_y_start:inner_y_stop, xs.stop - 1 : xs.stop, :],
                    dtype=np.float64,
                ).reshape(kspace_block.shape[0], -1),
            ]
        )

    return np.concatenate(parts, axis=1)


def _compute_window_gram_direct(
    kspace_cache: np.ndarray,
    window_size: int,
    *,
    block_size: int = 256,
    normalize_per_sample: bool = False,
) -> np.ndarray:
    num_samples = int(kspace_cache.shape[0])
    gram = np.zeros((num_samples, num_samples), dtype=np.float64)

    for i0 in range(0, num_samples, block_size):
        i1 = min(i0 + block_size, num_samples)
        xi = _extract_window_vectors(kspace_cache[i0:i1], window_size)

        if normalize_per_sample:
            xi = xi / np.maximum(np.linalg.norm(xi, axis=1, keepdims=True), EPS)

        for j0 in range(i0, num_samples, block_size):
            j1 = min(j0 + block_size, num_samples)
            xj = _extract_window_vectors(kspace_cache[j0:j1], window_size)

            if normalize_per_sample:
                xj = xj / np.maximum(np.linalg.norm(xj, axis=1, keepdims=True), EPS)

            block = xi @ xj.T
            gram[i0:i1, j0:j1] = block
            if j0 != i0:
                gram[j0:j1, i0:i1] = block.T

    return gram


def _accumulate_shell_gram_inplace(
    kspace_cache: np.memmap,
    *,
    cumulative_gram: np.ndarray,
    cumulative_norm_sq: np.ndarray,
    window_size: int,
    block_size: int = 256,
) -> None:
    num_samples = int(kspace_cache.shape[0])

    print(f"[*] Accumulating shell contribution for window={window_size}")
    for i0 in tqdm(range(0, num_samples, block_size), desc=f"Shell {window_size}x{window_size}"):
        i1 = min(i0 + block_size, num_samples)
        xi = _extract_shell_vectors(kspace_cache[i0:i1], window_size)
        cumulative_norm_sq[i0:i1] += np.einsum("ij,ij->i", xi, xi, dtype=np.float64)

        for j0 in range(i0, num_samples, block_size):
            j1 = min(j0 + block_size, num_samples)
            xj = _extract_shell_vectors(kspace_cache[j0:j1], window_size)
            block = xi @ xj.T

            cumulative_gram[i0:i1, j0:j1] += block
            if j0 != i0:
                cumulative_gram[j0:j1, i0:i1] += block.T

    _flush_array(cumulative_gram)


def _materialize_window_gram(
    cumulative_gram: np.ndarray,
    cumulative_norm_sq: np.ndarray,
    gram_path: os.PathLike[str] | str,
    *,
    block_size: int = 512,
    normalize_per_sample: bool = False,
) -> np.memmap:
    num_samples = cumulative_gram.shape[0]
    gram = np.memmap(gram_path, mode="w+", dtype=np.float64, shape=(num_samples, num_samples))

    inv_norm = None
    if normalize_per_sample:
        inv_norm = 1.0 / np.sqrt(np.maximum(cumulative_norm_sq, EPS))

    for i0 in range(0, num_samples, block_size):
        i1 = min(i0 + block_size, num_samples)
        block = np.asarray(cumulative_gram[i0:i1], dtype=np.float64)
        if inv_norm is not None:
            block = inv_norm[i0:i1, None] * block * inv_norm[None, :]
        gram[i0:i1] = block

    gram.flush()
    return gram


def _center_gram_inplace(gram: np.ndarray, block_size: int = 512) -> np.ndarray:
    num_samples = gram.shape[0]
    row_mean = np.zeros(num_samples, dtype=np.float64)

    print("[*] Centering Gram matrix...")
    for i0 in tqdm(range(0, num_samples, block_size), desc="Row means"):
        i1 = min(i0 + block_size, num_samples)
        row_mean[i0:i1] = np.asarray(gram[i0:i1], dtype=np.float64).mean(axis=1)

    grand_mean = row_mean.mean()

    for i0 in tqdm(range(0, num_samples, block_size), desc="Center Gram"):
        i1 = min(i0 + block_size, num_samples)
        block = np.asarray(gram[i0:i1], dtype=np.float64)
        block = block - row_mean[i0:i1, None] - row_mean[None, :] + grand_mean
        gram[i0:i1] = block

    _flush_array(gram)
    return gram


def _spectral_stats_from_centered_gram(
    gram: np.ndarray, n_samples: int, eps_ratio: float = 1e-12
) -> Tuple[float, np.ndarray, np.ndarray]:
    print("[*] Eigendecomposition of centered Gram...")
    gram_arr = np.asarray(gram, dtype=np.float64)
    gram_arr = 0.5 * (gram_arr + gram_arr.T)

    eigvals_gram = np.linalg.eigvalsh(gram_arr)
    eigvals_gram = np.clip(eigvals_gram, a_min=0.0, a_max=None)

    if eigvals_gram.size == 0 or eigvals_gram[-1] <= 0:
        return 1.0, eigvals_gram.copy(), eigvals_gram.copy()

    thresh = eigvals_gram[-1] * eps_ratio
    eigvals_gram = eigvals_gram[eigvals_gram > thresh]
    if eigvals_gram.size == 0:
        return 1.0, eigvals_gram.copy(), eigvals_gram.copy()

    probs = eigvals_gram / np.maximum(eigvals_gram.sum(), EPS)
    entropy = -np.sum(probs * np.log(np.maximum(probs, EPS)))
    r_eff = float(np.exp(entropy))

    eigvals_cov = eigvals_gram / max(n_samples - 1, 1)
    return r_eff, eigvals_gram, eigvals_cov


def _gaussian_info_from_cov_eigs(eigvals_cov: np.ndarray, tau: float) -> float:
    if eigvals_cov.size == 0:
        return 0.0
    return float(0.5 * np.sum(np.log1p(eigvals_cov / max(tau, EPS))))


def _estimate_tau(
    eigvals_cov_by_window: Mapping[int, np.ndarray],
    window_list: Sequence[int],
    tau_mode: str = "full_tail_median_ratio",
    tau_ratio: float = 1.0,
    tau_abs: Optional[float] = None,
    tail_frac: float = 0.1,
    tail_count: int = 32,
) -> float:
    if tau_abs is not None:
        return float(tau_abs)

    full_window = max(window_list)
    full_eigs = np.asarray(eigvals_cov_by_window[full_window], dtype=np.float64)
    if full_eigs.size == 0:
        return 1.0

    if tau_mode == "full_max_ratio":
        return float(tau_ratio * np.max(full_eigs))

    if tau_mode in {"full_tail_median_ratio", "full_tail_mean_ratio"}:
        tail_n = max(
            1,
            min(full_eigs.size, max(tail_count, int(np.ceil(tail_frac * full_eigs.size)))),
        )
        tail = np.sort(full_eigs)[:tail_n]
        if tau_mode == "full_tail_median_ratio":
            tau = tau_ratio * float(np.median(tail))
        else:
            tau = tau_ratio * float(np.mean(tail))
        return max(tau, EPS)

    raise ValueError(f"Unsupported tau_mode: {tau_mode}")


def _compute_delta_infos(
    info_by_window: Mapping[int, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = np.array(sorted(info_by_window.keys()), dtype=np.int64)
    infos = np.array([info_by_window[w] for w in windows], dtype=np.float64)
    infos = np.maximum.accumulate(infos)
    deltas = np.diff(np.concatenate([[0.0], infos]))
    deltas = np.clip(deltas, a_min=0.0, a_max=None)
    return windows, infos, deltas


def _write_text_log(
    text_path: os.PathLike[str] | str,
    windows: np.ndarray,
    infos: np.ndarray,
    deltas: np.ndarray,
    r_eff_by_window: Mapping[int, float],
) -> None:
    with open(text_path, "w", encoding="utf-8") as text_file:
        for window, info, delta in zip(windows, infos, deltas):
            text_file.write(
                f"window={window:3d}x{window:<3d} | "
                f"r_eff={r_eff_by_window[int(window)]:9.4f} | "
                f"gaussian_info={info:12.6f} | "
                f"delta_info={delta:10.6f}\n"
            )


def _write_json_log(
    json_path: os.PathLike[str] | str,
    *,
    dataset_tag: str,
    uniform_train_resolution: Sequence[int],
    num_samples: int,
    window_list: Sequence[int],
    normalize_per_sample: bool,
    tau_mode: str,
    tau_ratio: float,
    tau_abs: Optional[float],
    tau: float,
    windows: np.ndarray,
    infos: np.ndarray,
    deltas: np.ndarray,
    r_eff_by_window: Mapping[int, float],
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = {
        "dataset_tag": dataset_tag,
        "info_kind": "raw",
        "uniform_train_resolution": list(map(int, uniform_train_resolution)),
        "num_samples": int(num_samples),
        "window_list": list(map(int, window_list)),
        "windows": list(map(int, windows.tolist())),
        "infos": [float(x) for x in infos.tolist()],
        "delta_infos": [float(x) for x in deltas.tolist()],
        "r_eff_by_window": {str(k): float(v) for k, v in r_eff_by_window.items()},
        "tau_mode": tau_mode,
        "tau_ratio": float(tau_ratio),
        "tau_abs": None if tau_abs is None else float(tau_abs),
        "tau": float(tau),
        "normalize_per_sample": bool(normalize_per_sample),
        "metadata": dict(metadata or {}),
    }

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _resolve_output_paths(
    output_prefix: os.PathLike[str] | str,
) -> Tuple[Path, Path, Path]:
    prefix = Path(output_prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return prefix, prefix.with_suffix(".txt"), prefix.with_suffix(".json")


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
) -> Dict[str, Any]:
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

    cache_path = tmp_dir / f"kspace_cache_{num_samples}.dat"
    cumulative_gram_path = tmp_dir / f"gram_cumulative_N{num_samples}.dat"
    kspace_cache = _load_centered_kspace_cache(
        dataset=dataset,
        num_samples=num_samples,
        cache_path=cache_path,
        uniform_train_resolution=uniform_train_resolution,
        representation=representation,
        num_virtual_coils=num_virtual_coils,
        calibration_window=calibration_window,
    )

    cumulative_gram = np.memmap(
        cumulative_gram_path,
        mode="w+",
        dtype=np.float64,
        shape=(num_samples, num_samples),
    )
    cumulative_gram[:] = 0.0
    cumulative_gram.flush()
    cumulative_norm_sq = np.zeros(num_samples, dtype=np.float64)

    r_eff_by_window: Dict[int, float] = {}
    eigvals_cov_by_window: Dict[int, np.ndarray] = {}
    try:
        for window_size in window_list:
            _accumulate_shell_gram_inplace(
                kspace_cache,
                cumulative_gram=cumulative_gram,
                cumulative_norm_sq=cumulative_norm_sq,
                window_size=window_size,
                block_size=block_size_gram,
            )

            gram_path = tmp_dir / f"gram_w{window_size}_N{num_samples}.dat"
            gram = _materialize_window_gram(
                cumulative_gram,
                cumulative_norm_sq,
                gram_path,
                block_size=block_size_center,
                normalize_per_sample=normalize_per_sample,
            )
            gram = _center_gram_inplace(gram, block_size=block_size_center)

            r_eff, eigvals_gram, eigvals_cov = _spectral_stats_from_centered_gram(
                gram, n_samples=num_samples
            )
            r_eff_by_window[window_size] = r_eff
            eigvals_cov_by_window[window_size] = eigvals_cov

            print(
                f"[CovRank] window={window_size:3d}x{window_size:<3d} | "
                f"r_eff={r_eff:10.4f} | nonzero_eigs={len(eigvals_cov)}"
            )

            del gram, eigvals_gram
            gc.collect()
            try:
                gram_path.unlink()
            except OSError:
                pass

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
        }
    finally:
        del kspace_cache, cumulative_gram
        gc.collect()
        for path in (cache_path, cumulative_gram_path):
            try:
                path.unlink()
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
) -> Dict[str, Any]:
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
        metadata=metadata,
    )

    print("\n========== spectral summary ==========")
    for window, info, delta in zip(
        results["windows"], results["infos"], results["delta_infos"]
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
