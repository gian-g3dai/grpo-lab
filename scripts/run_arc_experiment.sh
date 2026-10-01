#!/usr/bin/env bash
# ARC pipeline for one config: baseline evals -> GRPO -> post-train evals -> plots.
#   scripts/run_arc_experiment.sh configs/qwen2.5-1.5b-arc-pilot.yaml
# Evals (before and after): fresh re-arc samples of the trained tasks, the original test pairs
# of those tasks, and the public ARC-AGI-1 evaluation set (pass@2, official metric).
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:?usage: $0 <config.yaml>}"
PY=.venv/bin/python

NAME=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['run_name'])")
MODEL=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model'])")
OUT=outputs/$NAME
RES=results/$NAME
mkdir -p "$OUT" "$RES"

evals() {  # $1 = before|after, rest = extra args (e.g. --adapter path)
  local tag=$1; shift
  $PY -m grpo_lab.evaluate_arc --model "$MODEL" --config "$CONFIG" --split rearc-holdout --out "$RES/holdout_$tag.json" "$@"
  $PY -m grpo_lab.evaluate_arc --model "$MODEL" --config "$CONFIG" --split training --out "$RES/training_$tag.json" "$@"
  $PY -m grpo_lab.evaluate_arc --model "$MODEL" --split evaluation --attempts 2 --out "$RES/eval1_$tag.json" "$@"
}

echo "== baseline evals: $MODEL"
evals before

echo "== GRPO training: $CONFIG"
$PY -m grpo_lab.train --config "$CONFIG" 2>&1 | tee "$OUT/train.log"

echo "== post-train evals"
evals after --adapter "$OUT/final"

echo "== plots"
$PY -m grpo_lab.plot --state "$OUT/final/trainer_state.json" --out "$RES/curves.png" --csv "$RES/train_log.csv"
