#!/usr/bin/env bash
# Full pipeline for one config: baseline eval -> GRPO -> post-train eval -> plots.
#   scripts/run_experiment.sh configs/qwen2.5-0.5b-gsm8k.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:?usage: $0 <config.yaml> [eval_limit]}"
EVAL_LIMIT="${2:-}"          # empty = full GSM8K test set (1319)
PY=.venv/bin/python

NAME=$($PY -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['run_name'])")
MODEL=$($PY -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['model'])")
OUT=outputs/$NAME
RES=results/$NAME
mkdir -p "$OUT" "$RES"
LIMIT_ARG=(); [[ -n "$EVAL_LIMIT" ]] && LIMIT_ARG=(--limit "$EVAL_LIMIT")

echo "== baseline eval: $MODEL"
$PY -m grpo_lab.evaluate --model "$MODEL" --out "$RES/eval_before.json" "${LIMIT_ARG[@]}"

echo "== GRPO training: $CONFIG"
$PY -m grpo_lab.train --config "$CONFIG" 2>&1 | tee "$OUT/train.log"

echo "== post-train eval"
$PY -m grpo_lab.evaluate --model "$MODEL" --adapter "$OUT/final" --out "$RES/eval_after.json" "${LIMIT_ARG[@]}"

echo "== plots"
$PY -m grpo_lab.plot --state "$OUT/final/trainer_state.json" --out "$RES/curves.png" --csv "$RES/train_log.csv"
$PY -m grpo_lab.summarize --results "$RES"
