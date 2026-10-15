#!/usr/bin/env bash
# End-to-end for each example: generate data -> zero-shot eval -> LoRA train -> fine-tuned eval -> summary.
#   EXAMPLES="conference-attend travel-budget" MODELS="Qwen/Qwen2.5-1.5B-Instruct" scripts/run_pipeline.sh
# Expects torch + requirements.txt installed. A single GPU is plenty for 1.5B; Apple MPS works for small runs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
EXAMPLES="${EXAMPLES:-cosmetics-preference conference-attend travel-budget}"
MODELS="${MODELS:-Qwen/Qwen2.5-1.5B-Instruct}"
GEN_ARGS="${GEN_ARGS:-}"; TRAIN_ARGS="${TRAIN_ARGS:-}"; EVAL_ARGS="${EVAL_ARGS:-}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for EX in $EXAMPLES; do
  D="data/$EX"; R="results/$EX"; O="outputs/$EX"
  # shellcheck disable=SC2086
  python "examples/$EX/gen.py" --out "$D" $GEN_ARGS
  mkdir -p "$R"
  for M in $MODELS; do
    TAG=$(basename "$M" | tr '[:upper:]' '[:lower:]')
    echo "===== $EX / $M ====="
    # shellcheck disable=SC2086
    python -m decision_model.evaluate --model "$M" --data "$D" --out "$R/$TAG/zero_shot" --task "$D/task.json" $EVAL_ARGS
    # shellcheck disable=SC2086
    python -m decision_model.train --model "$M" --data "$D" --out "$O/$TAG" --task "$D/task.json" $TRAIN_ARGS
    cp "$O/$TAG/train_log.json" "$R/$TAG/"
    # shellcheck disable=SC2086
    python -m decision_model.evaluate --model "$M" --adapter "$O/$TAG/adapter" --data "$D" \
      --out "$R/$TAG/finetuned" --task "$D/task.json" $EVAL_ARGS
    # models/<ex>-<tag>/ is what decision_model.score loads
    mkdir -p "models/$EX-$TAG" && cp -r "$O/$TAG/adapter" "models/$EX-$TAG/" \
      && cp "$R/$TAG/finetuned/metrics.json" "models/$EX-$TAG/"
  done
  python -m decision_model.summarize "$R" | tee "$R/SUMMARY.md"
done
