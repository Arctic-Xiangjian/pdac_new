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
from fastmri import fft2c
from tqdm import tqdm


EPS = 1e-12
DEFAULT_UNIFORM_TRAIN_RESOLUTION = (384, 384)


def _normalize_resolution(resolution: Sequence[int]) -> Tuple[int, int]:
    if len(resolution) != 2:
        raise ValueError(
            f"uniform_train_resolution must contain 2 integers, got {resolution!r}"
        )
    height, width = int(resolution[0]), int(resolution[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"uniform_train_resolution must be positive, got {resolution!r}")
    return height, width


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


def _load_centered_kspace_cache(
    dataset,
    num_samples: int,
    cache_path: os.PathLike[str] | str,
    uniform_train_resolution: Sequence[int],
) -> np.memmap:
    height, width = _normalize_resolution(uniform_train_resolution)
    kspace_cache = np.memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, height, width, 2),
    )

    print(
        f"[*] Caching {num_samples} samples of centered {height}x{width} simulated k-space to: {cache_path}"
    )
    for idx in tqdm(range(num_samples), desc="Caching centered k-space"):
        target = dataset[idx][2]
        kspace_cache[idx] = target_to_uniform_kspace(target, (height, width))

    kspace_cache.flush()
    return kspace_cache


def _window_slice(height: int, width: int, window_size: int) -> Tuple[slice, slice]:
    if window_size % 2 != 0:
        raise ValueError(f"window_size must be even, got {window_size}")
    cy, cx = height // 2, width // 2
    half = window_size // 2
    return slice(cy - half, cy + half), slice(cx - half, cx + half)


def _compute_gram_from_cache(
    kspace_cache: np.memmap,
    window_size: int,
    gram_path: os.PathLike[str] | str,
    block_size: int = 256,
    normalize_per_sample: bool = False,
) -> np.memmap:
    num_samples, height, width, channels = kspace_cache.shape
    if channels != 2:
        raise ValueError(f"Expected k-space cache with 2 channels, got {channels}")

    ys, xs = _window_slice(height, width, window_size)
    gram = np.memmap(gram_path, mode="w+", dtype=np.float64, shape=(num_samples, num_samples))

    print(f"[*] Computing Gram for window={window_size}: {gram_path}")
    for i0 in tqdm(range(0, num_samples, block_size), desc=f"Gram {window_size}x{window_size}"):
        i1 = min(i0 + block_size, num_samples)
        xi = np.asarray(kspace_cache[i0:i1, ys, xs, :], dtype=np.float64).reshape(i1 - i0, -1)

        if normalize_per_sample:
            norms = np.linalg.norm(xi, axis=1, keepdims=True)
            xi = xi / np.maximum(norms, EPS)

        for j0 in range(i0, num_samples, block_size):
            j1 = min(j0 + block_size, num_samples)
            xj = np.asarray(kspace_cache[j0:j1, ys, xs, :], dtype=np.float64).reshape(
                j1 - j0, -1
            )

            if normalize_per_sample:
                norms = np.linalg.norm(xj, axis=1, keepdims=True)
                xj = xj / np.maximum(norms, EPS)

            block = xi @ xj.T
            gram[i0:i1, j0:j1] = block
            if j0 != i0:
                gram[j0:j1, i0:i1] = block.T

    gram.flush()
    return gram


def _center_gram_inplace(gram: np.memmap, block_size: int = 512) -> np.memmap:
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

    gram.flush()
    return gram


def _spectral_stats_from_centered_gram(
    gram: np.memmap, n_samples: int, eps_ratio: float = 1e-12
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
) -> Dict[str, Any]:
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(int(num_samples), len(dataset))
    if num_samples <= 0:
        raise ValueError("num_samples must be positive after dataset truncation.")

    if normalize_per_sample:
        warnings.warn(
            "normalize_per_sample=True makes the complexity more shape-sensitive "
            "and less energy-sensitive. This is suitable for hierarchy discovery.",
            stacklevel=2,
        )

    cache_path = tmp_dir / f"kspace_cache_{num_samples}.dat"
    kspace_cache = _load_centered_kspace_cache(
        dataset=dataset,
        num_samples=num_samples,
        cache_path=cache_path,
        uniform_train_resolution=uniform_train_resolution,
    )

    r_eff_by_window: Dict[int, float] = {}
    eigvals_cov_by_window: Dict[int, np.ndarray] = {}
    try:
        for window_size in window_list:
            gram_path = tmp_dir / f"gram_w{window_size}_N{num_samples}.dat"
            gram = _compute_gram_from_cache(
                kspace_cache=kspace_cache,
                window_size=window_size,
                gram_path=gram_path,
                block_size=block_size_gram,
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
        del kspace_cache
        gc.collect()
        try:
            cache_path.unlink()
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
