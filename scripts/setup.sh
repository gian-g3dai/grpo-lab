#!/usr/bin/env bash
# One-time environment setup with uv (https://docs.astral.sh/uv/).
# Installs the exact versions in uv.lock (torch from the CUDA 12.8 index, see pyproject.toml).
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --python 3.13 --extra dev
echo "ok. activate with: source .venv/bin/activate"
