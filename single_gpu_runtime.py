from __future__ import annotations

import numbers
from typing import Any


def _config_get(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def uses_activation_checkpointing(source: Any) -> bool:
    return bool(
        _config_get(source, "use_checkpointing", False)
        or _config_get(source, "use_checkpoint", False)
    )


def _count_devices(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned == "auto":
            return None
        if "," in cleaned:
            return len([part for part in cleaned.split(",") if part.strip()])
        return int(cleaned)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return None


def requested_device_count(args: Any) -> int:
    counts = []
    for key in ("devices", "gpus"):
        count = _count_devices(_config_get(args, key))
        if count is not None:
            counts.append(count)
    return max(counts) if counts else 1


def validate_single_gpu_runtime(args: Any, *, use_compile: bool, context: str) -> None:
    device_count = requested_device_count(args)
    num_nodes = int(_config_get(args, "num_nodes", 1) or 1)
    strategy = _config_get(args, "strategy")

    if device_count > 1 or num_nodes > 1:
        if use_compile:
            raise ValueError(
                f"{context} only supports torch.compile on a single GPU. "
                f"Got devices={device_count}, num_nodes={num_nodes}."
            )
        raise ValueError(
            f"{context} currently supports single-GPU execution only. "
            f"Got devices={device_count}, num_nodes={num_nodes}."
        )

    if strategy not in (None, "auto"):
        raise ValueError(
            f"{context} currently expects single-GPU strategy='auto'. "
            f"Got strategy={strategy!r}."
        )


def normalize_single_gpu_args(args: Any) -> Any:
    if hasattr(args, "accelerator"):
        args.accelerator = "gpu"
    if hasattr(args, "devices"):
        args.devices = 1
    if hasattr(args, "gpus"):
        args.gpus = None
    if hasattr(args, "num_nodes"):
        args.num_nodes = 1
    if hasattr(args, "strategy"):
        args.strategy = "auto"
    return args
