#!/usr/bin/env bash
# One-time environment setup with uv (https://docs.astral.sh/uv/).
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.13 .venv
# CUDA 12.8 wheels; change the index for other CUDA versions (see pytorch.org/get-started).
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e ".[dev]"
echo "ok. activate with: source .venv/bin/activate"
