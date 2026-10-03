#!/usr/bin/env bash
# ARC pipeline for one config: baseline evals -> training (GRPO or SFT) -> post-train evals -> plots.
#   scripts/run_arc_experiment.sh configs/qwen2.5-1.5b-arc-pilot.yaml      # GRPO (config has a `grpo:` block)
#   scripts/run_arc_experiment.sh configs/qwen2.5-7b-arc-sft.yaml          # augmented SFT (config has an `sft:` block)
#   scripts/run_arc_experiment.sh configs/qwen2.5-7b-arc-grpo-from-sft.yaml  # GRPO continuing the SFT adapter
# Evals (before and after): fresh re-arc samples of the trained tasks, the original test pairs
# of those tasks, and the public ARC-AGI-1 evaluation set (pass@2, official metric). When the
# config has `init_adapter`, the "before" evals score that adapter, so the deltas are the stage's own.
#   EVAL_LIMIT=1000      cap the re-arc holdout / training-split evals (an all-400-task SFT holdout is 8,000 prompts)
#   OFFICIAL_LIMIT=100   cap the official ARC-1 eval (400 tasks, the slowest part)
#   SKIP_OFFICIAL=1      skip the official eval entirely
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:?usage: $0 <config.yaml>}"
PY=.venv/bin/python
# Triton (TRL's fused log-softmax) needs a C compiler at runtime; fall back to a conda gcc if none is on PATH.
if ! command -v "${CC:-cc}" >/dev/null 2>&1; then
  CONDA_GCC=$HOME/.conda/envs/cc/bin/x86_64-conda-linux-gnu-gcc
  [[ -x $CONDA_GCC ]] && export CC=$CONDA_GCC
fi

cfg() { $PY -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('$1') or '')"; }
NAME=$(cfg run_name)
MODEL=$(cfg model)
INIT=$(cfg init_adapter)
STAGE=grpo_lab.train
[[ -n $(cfg sft) ]] && STAGE=grpo_lab.sft
OUT=outputs/$NAME
RES=results/$NAME
mkdir -p "$OUT" "$RES"

# Eval batching: inference only, so the whole GPU is available. ~24k prompt chars per batch is ~8 GB
# on a 16 GB card; on an 8 GB card use e.g. EVAL_BATCH_CHARS=10000.
EVAL_BATCH=(--batch-size "${EVAL_BATCH_SIZE:-16}" --batch-chars "${EVAL_BATCH_CHARS:-24000}")
LIMIT=(); [[ -n ${EVAL_LIMIT:-} ]] && LIMIT=(--limit "$EVAL_LIMIT")
OLIMIT=(); [[ -n ${OFFICIAL_LIMIT:-} ]] && OLIMIT=(--limit "$OFFICIAL_LIMIT")
BEFORE=(); [[ -n $INIT ]] && BEFORE=(--adapter "$INIT")
task_evals() {  # fresh re-arc samples + original test pairs of the trained tasks. $1 = before|after, rest = extra args
  local tag=$1; shift
  # holdout queries have <= max_output_cells cells, so 512 new tokens is plenty
  $PY -m grpo_lab.evaluate_arc --model "$MODEL" --config "$CONFIG" --split rearc-holdout --max-new-tokens 512 "${EVAL_BATCH[@]}" "${LIMIT[@]}" --out "$RES/holdout_$tag.json" "$@"
  $PY -m grpo_lab.evaluate_arc --model "$MODEL" --config "$CONFIG" --split training "${EVAL_BATCH[@]}" "${LIMIT[@]}" --out "$RES/training_$tag.json" "$@"
}
official_eval() {  # public ARC-AGI-1 evaluation set, two attempts (slow: long prompts). $1 = before|after
  local tag=$1; shift
  $PY -m grpo_lab.evaluate_arc --model "$MODEL" --split evaluation --attempts 2 "${EVAL_BATCH[@]}" "${OLIMIT[@]}" --out "$RES/eval1_$tag.json" "$@"
}

echo "== baseline task evals: $MODEL ${INIT:+(adapter $INIT)}"
task_evals before "${BEFORE[@]}"

echo "== $STAGE: $CONFIG"
$PY -m "$STAGE" --config "$CONFIG" 2>&1 | tee "$OUT/train.log"

echo "== post-train task evals"
task_evals after --adapter "$OUT/final"

echo "== plots"
$PY -m grpo_lab.plot --state "$OUT/final/trainer_state.json" --out "$RES/curves.png" --csv "$RES/train_log.csv"

if [[ -z ${SKIP_OFFICIAL:-} ]]; then
  echo "== official ARC-AGI-1 evaluation set, before / after (slowest part, last)"
  official_eval before "${BEFORE[@]}"
  official_eval after --adapter "$OUT/final"
fi
