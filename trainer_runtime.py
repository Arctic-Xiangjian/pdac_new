from __future__ import annotations

import inspect
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import yaml

_UNSERIALIZABLE = object()


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def add_lightning_runtime_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--default_root_dir",
        default=None,
        type=str,
        help="Root directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        type=str,
        help="Checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--accelerator",
        default="gpu",
        type=str,
        help="Accelerator to use. Active path is single-GPU only.",
    )
    parser.add_argument(
        "--devices",
        default=1,
        type=int,
        help="Number of devices to use. Active path only supports 1.",
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes to use. Active path only supports 1.",
    )
    parser.add_argument(
        "--strategy",
        default="auto",
        type=str,
        help="Lightning strategy. Active path only supports 'auto'.",
    )
    parser.add_argument("--seed", default=42, type=int, help="Global random seed.")
    parser.add_argument(
        "--deterministic",
        default=False,
        type=_str_to_bool,
        help="Enable deterministic mode.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        type=_str_to_bool,
        help="Override torch benchmark mode.",
    )
    parser.add_argument("--max_epochs", default=None, type=int)
    parser.add_argument("--min_epochs", default=None, type=int)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--min_steps", default=None, type=int)
    parser.add_argument("--max_time", default=None, type=str)
    parser.add_argument("--limit_train_batches", default=None, type=float)
    parser.add_argument("--limit_val_batches", default=None, type=float)
    parser.add_argument("--limit_test_batches", default=None, type=float)
    parser.add_argument("--limit_predict_batches", default=None, type=float)
    parser.add_argument("--overfit_batches", default=0.0, type=float)
    parser.add_argument("--val_check_interval", default=None, type=float)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--num_sanity_val_steps", default=2, type=int)
    parser.add_argument("--log_every_n_steps", default=50, type=int)
    parser.add_argument(
        "--enable_checkpointing",
        default=True,
        type=_str_to_bool,
    )
    parser.add_argument(
        "--enable_progress_bar",
        default=True,
        type=_str_to_bool,
    )
    parser.add_argument(
        "--enable_model_summary",
        default=True,
        type=_str_to_bool,
    )
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--gradient_clip_val", default=None, type=float)
    parser.add_argument("--gradient_clip_algorithm", default=None, type=str)
    parser.add_argument("--profiler", default=None, type=str)
    parser.add_argument(
        "--detect_anomaly",
        default=False,
        type=_str_to_bool,
    )
    parser.add_argument(
        "--reload_dataloaders_every_n_epochs",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--fast_dev_run",
        default=False,
        type=_str_to_bool,
    )
    parser.add_argument(
        "--inference_mode",
        default=True,
        type=_str_to_bool,
    )

    return parser


def _trainer_supported_keys() -> set[str]:
    return set(inspect.signature(pl.Trainer.__init__).parameters.keys())


def build_trainer_kwargs(
    args: Namespace,
    *,
    precision: str,
    logger: Any,
    callbacks: list[Any] | None = None,
) -> dict[str, Any]:
    supported = _trainer_supported_keys()
    kwargs: dict[str, Any] = {
        "accelerator": "gpu",
        "devices": 1,
        "num_nodes": 1,
        "strategy": "auto",
        "precision": precision,
        "logger": logger,
        "use_distributed_sampler": False,
    }
    if callbacks is not None:
        kwargs["callbacks"] = callbacks

    passthrough_keys = {
        "default_root_dir",
        "fast_dev_run",
        "max_epochs",
        "min_epochs",
        "max_steps",
        "min_steps",
        "max_time",
        "limit_train_batches",
        "limit_val_batches",
        "limit_test_batches",
        "limit_predict_batches",
        "overfit_batches",
        "val_check_interval",
        "check_val_every_n_epoch",
        "num_sanity_val_steps",
        "log_every_n_steps",
        "enable_checkpointing",
        "enable_progress_bar",
        "enable_model_summary",
        "accumulate_grad_batches",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "deterministic",
        "benchmark",
        "inference_mode",
        "profiler",
        "detect_anomaly",
        "reload_dataloaders_every_n_epochs",
        "sync_batchnorm",
    }
    for key in passthrough_keys:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                kwargs[key] = value

    return {key: value for key, value in kwargs.items() if key in supported}


def build_experiment_logger(args: Namespace) -> Any:
    logger_type = getattr(args, "logger_type", "tb")
    default_root_dir = getattr(args, "default_root_dir", None) or "."
    if logger_type == "tb":
        return pl.loggers.TensorBoardLogger(
            save_dir=str(default_root_dir),
            name="lightning_logs",
        )
    if logger_type == "wandb":
        return pl.loggers.WandbLogger(
            project=args.experiment_name,
            save_dir=str(default_root_dir),
        )
    raise ValueError("Unknown logger type.")


def resolve_fit_ckpt_path(args: Namespace) -> str | None:
    ckpt_path = getattr(args, "resume_from", None)
    return str(ckpt_path) if ckpt_path else None


def resolve_log_dir(trainer: pl.Trainer) -> Path:
    log_dir = getattr(trainer, "log_dir", None)
    if log_dir:
        return Path(log_dir)
    logger = getattr(trainer, "logger", None)
    logger_log_dir = getattr(logger, "log_dir", None)
    if logger_log_dir:
        return Path(logger_log_dir)
    default_root_dir = getattr(trainer, "default_root_dir", None)
    return Path(default_root_dir) if default_root_dir else Path.cwd()


def _serialize_hparam(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Namespace):
        return _serialize_hparam(vars(value))
    if isinstance(value, dict):
        serialized = {}
        for key, item in value.items():
            serialized_item = _serialize_hparam(item)
            if serialized_item is not _UNSERIALIZABLE:
                serialized[str(key)] = serialized_item
        return serialized
    if isinstance(value, (list, tuple)):
        serialized = []
        for item in value:
            serialized_item = _serialize_hparam(item)
            if serialized_item is not _UNSERIALIZABLE:
                serialized.append(serialized_item)
        return serialized
    return _UNSERIALIZABLE


def save_hparams_yaml(trainer: pl.Trainer, args: Namespace) -> Path:
    log_dir = resolve_log_dir(trainer)
    log_dir.mkdir(parents=True, exist_ok=True)

    serializable_args = {}
    for key, value in vars(args).items():
        serialized = _serialize_hparam(value)
        if serialized is not _UNSERIALIZABLE:
            serializable_args[key] = serialized

    output_path = log_dir / "hparams.yaml"
    with output_path.open("w") as handle:
        yaml.safe_dump(serializable_args, handle, sort_keys=False)

    return output_path
