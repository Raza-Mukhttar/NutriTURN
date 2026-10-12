#!/usr/bin/env bash
# Stage 3 of 3 in the data chain: features and derived tables. CPU, except the embeddings.
#
#   bash data_preprocessing/run_data_preprocessing.sh
#
# Turns records + annotations into everything the audit consumes: the 55 pre-cutoff variables at
# every cutoff, the parsed effect-estimate cache, the LLM-independent numeric anchor and the
# Compact-7 system scores.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python3}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
S=data_preprocessing

echo "== 1. the claim x cutoff benchmark: 55 variables per unit =="
"$PY" $S/01_build_benchmark.py

echo "== 2. parsed effect estimates cache (lets the anchor analyses run without abstracts) =="
"$PY" $S/02_build_effects_cache.py

echo "== 3. LLM-independent numeric anchor =="
echo "   NOTE: this REWRITES data/derived/numeric_anchor_v2_*.csv. Run it only with the"
echo "   abstracts present; without them it writes empty tables and the anchor analyses"
echo "   lose their input."
"$PY" $S/03_build_numeric_anchor.py

echo "== 4. Compact-7 system scores =="
"$PY" $S/04_build_system_scores.py

echo "== 5. record embeddings for the text-reading baseline (GPU) =="
"$PY" $S/05_build_record_embeddings.py embed --gpu 0 || echo "   skipped (no GPU / no abstracts)"

echo "== 6. external claim-cutoff units =="
"$PY" $S/external/driver.py || echo "   skipped: see config/external_sources.json"
