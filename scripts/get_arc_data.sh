#!/usr/bin/env bash
# Fetch the ARC data used by the `task: arc` configs into data/ (git-ignored):
#   data/arc-agi-1   official ARC-AGI-1 tasks (400 training + 400 evaluation)
#   data/arc-agi-2   official ARC-AGI-2 tasks (1000 training + 120 evaluation)
#   data/re-arc      re-arc repo (generators, verifiers, DSL)
#   data/re_arc      pre-generated re-arc samples: 1000 verified examples per ARC-1 training task
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
[[ -d data/arc-agi-1 ]] || git clone --depth 1 https://github.com/fchollet/ARC-AGI.git data/arc-agi-1
[[ -d data/arc-agi-2 ]] || git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git data/arc-agi-2
[[ -d data/re-arc ]]    || git clone --depth 1 https://github.com/michaelhodel/re-arc.git data/re-arc
[[ -d data/re_arc/tasks ]] || python3 -c "import zipfile; zipfile.ZipFile('data/re-arc/re_arc.zip').extractall('data')"
echo "arc-agi-1: $(ls data/arc-agi-1/data/training | wc -l) train / $(ls data/arc-agi-1/data/evaluation | wc -l) eval"
echo "arc-agi-2: $(ls data/arc-agi-2/data/training | wc -l) train / $(ls data/arc-agi-2/data/evaluation | wc -l) eval"
echo "re_arc:    $(ls data/re_arc/tasks | wc -l) tasks"
