#!/usr/bin/env bash
# Stage 2 of 3 in the data chain: annotation. Needs GPUs and the abstracts from stage 1.
#
#   bash data_annotation/run_data_annotation.sh            # every cell, one GPU
#   NSHARD=5 bash data_annotation/run_data_annotation.sh   # shard each cell across 5 GPUs
#
# Each cell is one (model, prompt, read-out) combination; config/annotation_models.json is the
# index. Cells are independent: a cell can be re-run without touching the others.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python3}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
A=data_annotation
NSHARD="${NSHARD:-1}"

shard() {   # shard <script> [extra args...] -- fan one cell across NSHARD GPUs and wait
  local s="$1"; shift
  if [ "$NSHARD" -le 1 ]; then "$PY" "$s" --gpu 0 "$@"; return $?; fi
  local pids=()
  for g in $(seq 0 $((NSHARD - 1))); do
    "$PY" "$s" --gpu "$g" --shard "$g" --nshards "$NSHARD" "$@" & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; return $rc
}

echo "== primary cell: Llama-3.1-8B, P1, first-token read-out (the deployed pipeline) =="
if [ "$NSHARD" -le 1 ]; then "$PY" $A/01_annotate_primary.py --gpu 0
else
  pids=(); for g in $(seq 0 $((NSHARD - 1))); do
    "$PY" $A/01_annotate_primary.py --gpu "$g" --shard "$g" --nshard "$NSHARD" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p"; done
fi
"$PY" $A/01_annotate_primary.py --summary-only

echo "== second pipeline: Qwen2.5-7B, P2, full-string =="
shard $A/02_annotate_second_pipeline.py

echo "== attribution cell C: Qwen2.5-7B under the original Llama prompt =="
shard $A/03_annotate_second_pipeline_origprompt.py

echo "== controlled cells A' and B: Llama full-string, P1 and P2 =="
for cell in Aprime B; do "$PY" $A/04_annotate_controlled_cells.py --cell "$cell" --gpu 0; done

echo "== third annotation family: Mistral-7B-Instruct-v0.3, P2, full-string =="
echo "   (needs text_full.json: the records snapshot carries no abstracts)"
"$PY" $A/05_annotate_third_family.py --cell M --gpu 0 --text text_full \
  || echo "   skipped: see data_annotation/README.md"

echo
echo "External-1/2 re-annotation is separate (retained retrieval texts only):"
echo "  python $A/external/build_items.py && python $A/external/annotate_external.py --gpu 0"
