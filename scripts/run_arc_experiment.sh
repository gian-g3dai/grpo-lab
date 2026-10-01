#!/usr/bin/env bash
# ARC pipeline for one config: baseline evals -> GRPO -> post-train evals -> plots.
#   scripts/run_arc_experiment.sh configs/qwen2.5-1.5b-arc-pilot.yaml
# Evals (before and after): fresh re-arc samples of the trained tasks, the original test pairs
# of those tasks, and the public ARC-AGI-1 evaluation set (pass@2, official metric).
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:?usage: $0 <config.yaml>}"
PY=.venv/bin/python
# Triton (TRL's fused log-softmax) needs a C compiler at runtime; fall back to a conda gcc if none is on PATH.
if ! command -v "${CC:-cc}" >/dev/null 2>&1; then
  CONDA_GCC=$HOME/.conda/envs/cc/bin/x86_64-conda-linux-gnu-gcc
  [[ -x $CONDA_GCC ]] && export CC=$CONDA_GCC
fi

NAME=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['run_name'])")
MODEL=$($PY -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['model'])")
OUT=outputs/$NAME
RES=results/$NAME
mkdir -p "$OUT" "$RES"

# Eval batching: inference only, so the whole 16 GB is available; ~24k prompt chars per batch is ~8 GB.
EVAL_BATCH=(--batch-size 16 --batch-chars 24000)
task_evals() {  # fresh re-arc samples + original test pairs of the trained tasks. $1 = before|after, rest = extra args
  local tag=$1; shift
  # holdout queries have <= max_output_cells (300) cells, so 512 new tokens is plenty
  $PY -m grpo_lab.evaluate_arc --model "$MODEL" --config "$CONFIG" --split rearc-holdout --max-new-tokens 512 "${EVAL_BATCH[@]}" --out "$RES/holdout_$tag.json" "$@"
  $PY -m grpo_lab.evaluate_arc --model "$MODEL" --config "$CONFIG" --split training "${EVAL_BATCH[@]}" --out "$RES/training_$tag.json" "$@"
}
official_eval() {  # public ARC-AGI-1 evaluation set, two attempts (slow: long prompts). $1 = before|after
  local tag=$1; shift
  $PY -m grpo_lab.evaluate_arc --model "$MODEL" --split evaluation --attempts 2 "${EVAL_BATCH[@]}" --out "$RES/eval1_$tag.json" "$@"
}

echo "== baseline task evals: $MODEL"
task_evals before

echo "== GRPO training: $CONFIG"
$PY -m grpo_lab.train --config "$CONFIG" 2>&1 | tee "$OUT/train.log"

echo "== post-train task evals"
task_evals after --adapter "$OUT/final"

echo "== plots"
$PY -m grpo_lab.plot --state "$OUT/final/trainer_state.json" --out "$RES/curves.png" --csv "$RES/train_log.csv"

echo "== official ARC-AGI-1 evaluation set, before / after (slowest part, last)"
official_eval before
official_eval after --adapter "$OUT/final"
