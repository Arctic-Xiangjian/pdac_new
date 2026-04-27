#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=2

uv run python pdac_examples/train_ab_1.py \
  --config_file pdac_examples/config/stanford2d/ab_1.yaml \
  "$@"

uv run python pdac_examples/train_ab_2.py \
  --config_file pdac_examples/config/stanford2d/ab_2.yaml \
  "$@"

uv run python pdac_examples/train_ab_3.py \
  --config_file pdac_examples/config/stanford2d/ab_3.yaml \
  "$@"
