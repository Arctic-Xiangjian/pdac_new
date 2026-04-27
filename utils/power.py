from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.center_mask_scheduler import validate_center_mask_schedule


EPS = 1e-12


def parse_txt_info_file(file_path: Path) -> Dict[int, float]:
    log_text = file_path.read_text(encoding="utf-8")
    pattern = r"window=\s*(\d+)x\d+\s*\|.*?gaussian_info=\s*([-\d\.eE]+)"
    matches = re.findall(pattern, log_text)
    if not matches:
        raise ValueError(f"No gaussian_info entries found in {file_path}")
    return {int(window): float(info) for window, info in matches}


def parse_info_file(file_path: Path) -> Dict[int, float]:
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


def _alpha_key(log_alpha: float) -> str:
    if np.isinf(log_alpha):
        return "inf"
    if float(log_alpha).is_integer():
        return str(int(log_alpha))
    return str(log_alpha)


def _round_to_even(value: float, min_value: int, max_value: int) -> int:
    rounded = int(np.round(value))
    if rounded % 2 != 0:
        rounded += 1
    return max(min_value, min(rounded, max_value))


def _strict_even_schedule(raw_windows: np.ndarray, num_blocks: int, img_size: int) -> List[int]:
    final_windows: List[int] = []
    prev = 0
    for index, raw_window in enumerate(raw_windows):
        even_window = _round_to_even(raw_window, min_value=2, max_value=img_size)
        even_window = max(even_window, prev + 2)
        if index == num_blocks - 1:
            even_window = img_size
        final_windows.append(even_window)
        prev = even_window

    for index in range(num_blocks - 2, -1, -1):
        final_windows[index] = min(final_windows[index], final_windows[index + 1] - 2)

    if any(window <= 0 for window in final_windows):
        raise ValueError(f"Could not build a valid strictly increasing schedule: {final_windows}")
    return final_windows


def generate_progressive_windows(
    info_by_window: Dict[int, float],
    *,
    acs_size: int = 16,
    num_blocks: int = 8,
    shell_power: float = 0.25,
    log_alpha: float = 100.0,
    img_size: int = 384,
) -> List[int]:
    windows = np.array(
        sorted(window for window in info_by_window.keys() if acs_size <= window <= img_size),
        dtype=np.int64,
    )
    if windows.size == 0:
        raise ValueError("No windows remain after applying acs_size/img_size bounds.")

    infos = np.array([info_by_window[window] for window in windows], dtype=np.float64)
    infos = np.maximum.accumulate(infos)
    delta_infos = np.diff(infos)
    delta_infos = np.clip(delta_infos, a_min=0.0, a_max=None)

    calibrated_increments = np.power(delta_infos + EPS, shell_power)
    cum_calibrated = np.insert(np.cumsum(calibrated_increments), 0, 0.0)
    cum_calibrated_norm = cum_calibrated / max(cum_calibrated[-1], EPS)

    linear_steps = np.linspace(1 / num_blocks, 1.0, num_blocks)
    if log_alpha > 0 and not np.isinf(log_alpha):
        targets = np.log1p(log_alpha * linear_steps) / np.log1p(log_alpha)
    else:
        targets = linear_steps

    raw_windows = np.interp(targets, cum_calibrated_norm, windows)
    final_windows = _strict_even_schedule(raw_windows, num_blocks=num_blocks, img_size=img_size)

    print(f"\n[*] ACS {acs_size}x{acs_size} | shell_power={shell_power} | log_alpha={log_alpha}")
    print("-" * 84)
    prev_window = acs_size
    for block_index, window in enumerate(final_windows, start=1):
        target_idx = int(np.where(windows == window)[0][0])
        prev_candidates = np.where(windows == prev_window)[0]
        prev_idx = int(prev_candidates[0]) if prev_candidates.size else 0
        calibrated_load = cum_calibrated_norm[target_idx] - cum_calibrated_norm[prev_idx]
        span = window - prev_window
        print(
            f"Block {block_index}: Window {prev_window:3d} -> {window:3d} | "
            f"physical span: +{span:3d} | calibrated load: {calibrated_load * 100:6.2f}%"
        )
        prev_window = window
    print("-" * 84)

    return final_windows


def update_output_json(output_json: Path, log_alpha: float, windows: List[int]) -> None:
    payload = {}
    if output_json.exists():
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict-shaped json in {output_json}")

    payload[_alpha_key(log_alpha)] = list(map(int, windows))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multi-coil 384 center-mask schedules from raw hierarchy logs."
    )
    parser.add_argument("--info-file", type=str, required=True)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--shell-power", type=float, default=1)
    parser.add_argument("--log-alpha", type=float, default=0)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--acs-size", type=int, default=16)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = build_args()
    info_file = Path(args.info_file)
    info_by_window = parse_info_file(info_file)
    schedule = generate_progressive_windows(
        info_by_window,
        acs_size=args.acs_size,
        num_blocks=args.num_blocks,
        shell_power=args.shell_power,
        log_alpha=args.log_alpha,
        img_size=args.img_size,
    )
    validate_center_mask_schedule(schedule, args.num_blocks, [args.img_size, args.img_size])

    print("\n[✔] Final num_list:")
    print(schedule)

    if args.output_json:
        output_json = Path(args.output_json)
        update_output_json(output_json, args.log_alpha, schedule)
        print(f"[*] Updated schedule json: {output_json}")


if __name__ == "__main__":
    main()
