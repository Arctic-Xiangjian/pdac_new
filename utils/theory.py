from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.center_mask_scheduler import validate_center_mask_schedule


EPS = 1e-12
CURVE_MODES = {"raw", "power"}
RATIO_ORDERS = {"coarse_to_fine", "fine_to_coarse"}
GAUSSIAN_INFO_PATTERN = re.compile(
    r"window=\s*(\d+)x\d+\s*\|.*?gaussian_info=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def parse_txt_info_file(file_path: Path) -> dict[int, float]:
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    log_text = file_path.read_text(encoding="utf-8")
    matches = GAUSSIAN_INFO_PATTERN.findall(log_text)
    if not matches:
        raise ValueError(f"No gaussian_info entries found in {file_path}")

    return {int(window): float(info) for window, info in matches}


def parse_info_file(file_path: Path) -> dict[int, float]:
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    if file_path.suffix.lower() == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        windows = payload.get("windows")
        infos = payload.get("infos")
        if not isinstance(windows, list) or not isinstance(infos, list):
            raise ValueError(f"Invalid raw info json format in {file_path}")
        if len(windows) != len(infos):
            raise ValueError(f"windows/infos length mismatch in {file_path}")
        return {int(window): float(info) for window, info in zip(windows, infos)}

    return parse_txt_info_file(file_path)


def build_theory_ratios(num_blocks: int = 8, ratio_base: float = 0.5) -> dict[str, np.ndarray]:
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if not (0.0 < ratio_base < 1.0):
        raise ValueError(f"ratio_base must be in (0, 1), got {ratio_base}")

    coarse_to_fine_raw = np.array(
        [ratio_base ** (block_idx + 1) for block_idx in range(num_blocks)],
        dtype=np.float64,
    )
    coarse_to_fine_ratios_normalized = coarse_to_fine_raw / coarse_to_fine_raw.sum()
    fine_to_coarse_raw = coarse_to_fine_raw[::-1].copy()

    return {
        "coarse_to_fine_ratios_raw": coarse_to_fine_raw,
        "coarse_to_fine_ratios_normalized": coarse_to_fine_ratios_normalized,
        "fine_to_coarse_ratios_raw": fine_to_coarse_raw,
        "fine_to_coarse_ratios_normalized": coarse_to_fine_ratios_normalized[::-1].copy(),
    }


def build_information_curve(
    info_by_window: dict[int, float],
    acs_size: int,
    curve_mode: str = "raw",
    shell_power: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if curve_mode not in CURVE_MODES:
        raise ValueError(f"Unsupported curve_mode: {curve_mode}")

    candidate_windows = np.array(
        sorted(
            int(window)
            for window in info_by_window.keys()
            if int(window) >= int(acs_size) and int(window) % 2 == 0
        ),
        dtype=np.int64,
    )
    if candidate_windows.size < 2:
        raise ValueError(
            f"Need at least 2 windows >= acs_size={acs_size}, got {candidate_windows.size}."
        )
    if int(candidate_windows[0]) != int(acs_size):
        raise ValueError(
            f"acs_size={acs_size} must exist in info_by_window; first available window is "
            f"{candidate_windows[0]}."
        )

    infos = np.array([info_by_window[int(window)] for window in candidate_windows], dtype=np.float64)
    infos = np.maximum.accumulate(infos)
    delta_infos = np.diff(infos)
    delta_infos = np.clip(delta_infos, a_min=0.0, a_max=None)

    if curve_mode == "raw":
        increments = delta_infos
    else:
        increments = np.power(delta_infos + EPS, shell_power)

    total_increment = float(increments.sum())
    if total_increment <= 0.0:
        raise ValueError("Information increments sum to zero; cannot build a valid cumulative curve.")

    cumulative_information = np.insert(np.cumsum(increments), 0, 0.0)
    cumulative_information = cumulative_information / cumulative_information[-1]
    return candidate_windows, increments, cumulative_information


def _round_to_even(value: float) -> int:
    rounded = int(np.round(float(value) / 2.0)) * 2
    return max(2, rounded)


def _eligible_windows(info_by_window: dict[int, float], acs_size: int) -> np.ndarray:
    return np.array(
        sorted(
            int(window)
            for window in info_by_window.keys()
            if int(window) >= int(acs_size) and int(window) % 2 == 0
        ),
        dtype=np.int64,
    )


def _validate_full_size(
    requested_full_size: int | None,
    available_windows: np.ndarray,
    acs_size: int,
    num_blocks: int,
) -> int:
    if available_windows.size == 0:
        raise ValueError(f"No even windows are available at or above acs_size={acs_size}.")

    max_available = int(available_windows[-1])
    full_size = max_available if requested_full_size is None else _round_to_even(requested_full_size)

    if full_size > max_available:
        raise ValueError(
            f"full_size={requested_full_size} exceeds the largest available window {max_available}."
        )
    if full_size <= acs_size:
        raise ValueError(f"full_size must be > acs_size, got full_size={full_size}, acs_size={acs_size}.")
    if full_size < acs_size + 2 * num_blocks:
        raise ValueError(
            f"full_size={full_size} is too small for {num_blocks} strictly increasing even windows "
            f"starting above acs_size={acs_size}."
        )

    return full_size


def _enforce_strictly_increasing_even(
    raw_windows: np.ndarray,
    acs_size: int,
    full_size: int,
) -> list[int]:
    final_windows: list[int] = []
    prev_window = int(acs_size)
    num_blocks = int(len(raw_windows))

    for block_idx, raw_window in enumerate(raw_windows):
        if block_idx == num_blocks - 1:
            window = int(full_size)
        else:
            window = _round_to_even(raw_window)
            if window <= prev_window:
                window = prev_window + 2

        final_windows.append(int(window))
        prev_window = int(window)

    final_windows[-1] = int(full_size)
    for block_idx in range(num_blocks - 2, -1, -1):
        max_allowed = final_windows[block_idx + 1] - 2
        if final_windows[block_idx] > max_allowed:
            final_windows[block_idx] = max_allowed

    prev_window = int(acs_size)
    for block_idx, window in enumerate(final_windows):
        min_allowed = prev_window + 2
        if window < min_allowed:
            final_windows[block_idx] = min_allowed
        prev_window = final_windows[block_idx]

    if any(curr >= nxt for curr, nxt in zip(final_windows, final_windows[1:])):
        raise RuntimeError(f"Failed to build a strictly increasing schedule: {final_windows}")
    if any(window % 2 != 0 for window in final_windows):
        raise RuntimeError(f"Failed to keep schedule even-valued: {final_windows}")
    if final_windows[-1] != int(full_size):
        raise RuntimeError(
            f"Failed to pin the final schedule window to full_size={full_size}: {final_windows}"
        )

    return final_windows


def generate_theory_windows(
    info_by_window: dict[int, float],
    acs_size: int = 16,
    num_blocks: int = 8,
    ratio_base: float = 0.5,
    curve_mode: str = "raw",
    shell_power: float = 0.25,
    full_size: int | None = 384,
    ratio_order: str = "coarse_to_fine",
) -> dict[str, Any]:
    ratio_info = build_theory_ratios(num_blocks=num_blocks, ratio_base=ratio_base)
    if ratio_order not in RATIO_ORDERS:
        raise ValueError(f"Unsupported ratio_order: {ratio_order}")

    available_windows = _eligible_windows(info_by_window, acs_size=acs_size)
    resolved_full_size = _validate_full_size(
        requested_full_size=full_size,
        available_windows=available_windows,
        acs_size=acs_size,
        num_blocks=num_blocks,
    )

    bounded_info_by_window = {
        int(window): float(info)
        for window, info in info_by_window.items()
        if acs_size <= int(window) <= resolved_full_size and int(window) % 2 == 0
    }
    candidate_windows, info_increments, cumulative_information = build_information_curve(
        info_by_window=bounded_info_by_window,
        acs_size=acs_size,
        curve_mode=curve_mode,
        shell_power=shell_power,
    )

    target_loads = ratio_info[f"{ratio_order}_ratios_normalized"]
    target_cumulative = np.cumsum(target_loads)
    interp_curve, interp_indices = np.unique(cumulative_information, return_index=True)
    interp_windows = candidate_windows[interp_indices]
    if interp_curve.size < 2:
        raise ValueError("Need at least two distinct cumulative information values for interpolation.")

    raw_windows = np.interp(target_cumulative, interp_curve, interp_windows)
    window_list = _enforce_strictly_increasing_even(
        raw_windows=raw_windows,
        acs_size=acs_size,
        full_size=resolved_full_size,
    )

    cumulative_at_windows = np.interp(
        np.asarray(window_list, dtype=np.float64),
        candidate_windows.astype(np.float64),
        cumulative_information,
    )
    achieved_block_loads = np.diff(np.insert(cumulative_at_windows, 0, 0.0))
    absolute_errors = np.abs(achieved_block_loads - target_loads)

    return {
        "window_list": [int(window) for window in window_list],
        "coarse_to_fine_ratios_raw": ratio_info["coarse_to_fine_ratios_raw"].tolist(),
        "coarse_to_fine_ratios_normalized": ratio_info["coarse_to_fine_ratios_normalized"].tolist(),
        "fine_to_coarse_ratios_raw": ratio_info["fine_to_coarse_ratios_raw"].tolist(),
        "fine_to_coarse_ratios_normalized": ratio_info["fine_to_coarse_ratios_normalized"].tolist(),
        "target_loads_used": target_loads.tolist(),
        "achieved_block_loads": achieved_block_loads.tolist(),
        "absolute_errors": absolute_errors.tolist(),
        "curve_mode": str(curve_mode),
        "ratio_order": str(ratio_order),
        "shell_power": float(shell_power),
        "acs_size": int(acs_size),
        "full_size": int(resolved_full_size),
        "candidate_windows": candidate_windows.tolist(),
        "information_increments": info_increments.tolist(),
    }


def format_schedule_report(result: dict[str, Any], acs_size: int) -> str:
    window_list = [int(value) for value in result["window_list"]]
    target_loads = np.asarray(result["target_loads_used"], dtype=np.float64)
    achieved_loads = np.asarray(result["achieved_block_loads"], dtype=np.float64)
    absolute_errors = np.asarray(result["absolute_errors"], dtype=np.float64)

    lines = [
        (
            f"[*] Theory schedule | acs_size={acs_size} | curve_mode={result['curve_mode']} "
            f"| shell_power={result['shell_power']} | ratio_order={result['ratio_order']}"
        ),
        "-" * 108,
    ]

    prev_window = int(acs_size)
    for block_idx, end_window in enumerate(window_list):
        lines.append(
            (
                f"Block {block_idx + 1}: Window {prev_window:3d} -> {int(end_window):3d} | "
                f"target={target_loads[block_idx] * 100:7.3f}% | "
                f"achieved={achieved_loads[block_idx] * 100:7.3f}% | "
                f"|error|={absolute_errors[block_idx] * 100:7.3f}%"
            )
        )
        prev_window = int(end_window)

    lines.append("-" * 108)
    return "\n".join(lines)


def _format_percentage_list(values: list[float]) -> str:
    return ", ".join(f"{float(value) * 100:.4f}%" for value in values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate theory-based multi-coil 384 center-mask schedules from raw info logs."
    )
    parser.add_argument("--info-file", type=str, required=True)
    parser.add_argument("--acs-size", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--ratio-base", type=float, default=0.5)
    parser.add_argument("--shell-power", type=float, default=1)
    parser.add_argument("--full-size", type=int, default=384)
    parser.add_argument("--curve-mode", type=str, default="raw", choices=["raw", "power", "both"])
    parser.add_argument(
        "--ratio-order",
        type=str,
        default="both",
        choices=["coarse_to_fine", "fine_to_coarse", "both"],
    )
    return parser.parse_args()


def _curve_modes_for_cli(curve_mode: str) -> list[str]:
    if curve_mode == "both":
        return ["raw", "power"]
    return [curve_mode]


def _ratio_orders_for_cli(ratio_order: str) -> list[str]:
    if ratio_order == "both":
        return ["coarse_to_fine", "fine_to_coarse"]
    return [ratio_order]


def _print_schedule(result: dict[str, Any], num_blocks: int) -> None:
    schedule_tag = f"[{result['curve_mode']}][{result['ratio_order']}]"
    schedule = [int(window) for window in result["window_list"]]
    validate_center_mask_schedule(schedule, num_blocks, [result["full_size"], result["full_size"]])

    print(f"\n{schedule_tag} coarse-to-fine raw ratios:")
    print(_format_percentage_list(result["coarse_to_fine_ratios_raw"]))
    print(f"{schedule_tag} coarse-to-fine normalized target loads:")
    print(_format_percentage_list(result["coarse_to_fine_ratios_normalized"]))
    print(f"{schedule_tag} fine-to-coarse raw ratios:")
    print(_format_percentage_list(result["fine_to_coarse_ratios_raw"]))
    print(f"{schedule_tag} fine-to-coarse normalized target loads:")
    print(_format_percentage_list(result["fine_to_coarse_ratios_normalized"]))
    print(f"{schedule_tag} target loads used for this schedule:")
    print(_format_percentage_list(result["target_loads_used"]))
    print(format_schedule_report(result, acs_size=result["acs_size"]))
    print(f"{schedule_tag} achieved block loads:")
    print(_format_percentage_list(result["achieved_block_loads"]))
    print(f"{schedule_tag} window_list:")
    print(schedule)
    print(f"{schedule_tag} JSON-copyable window_list:")
    print(json.dumps(schedule))


def main() -> None:
    args = _parse_args()
    info_by_window = parse_info_file(Path(args.info_file))

    for curve_mode in _curve_modes_for_cli(args.curve_mode):
        for ratio_order in _ratio_orders_for_cli(args.ratio_order):
            result = generate_theory_windows(
                info_by_window=info_by_window,
                acs_size=args.acs_size,
                num_blocks=args.num_blocks,
                ratio_base=args.ratio_base,
                curve_mode=curve_mode,
                shell_power=args.shell_power,
                full_size=args.full_size,
                ratio_order=ratio_order,
            )
            _print_schedule(result, num_blocks=args.num_blocks)


if __name__ == "__main__":
    main()
