#!/usr/bin/env bash
# Stage 1 of 3 in the data chain: retrieval. This is the only stage that needs the network.
#
#   bash data_preparation/run_data_preparation.sh
#
# Produces data/protocol_snapshot/records/<claim>.json: identifiers, years, titles, publication
# types and abstract text for every claim in config/claim_queries.json.
#
# PubMed keeps growing, so a retrieval run today returns MORE records than the September 2026
# snapshot the paper reports. That is expected and is why the paper's numbers are tied to the
# archived identifier lists, not to a live query. To reproduce the paper exactly, retrieve the
# archived PMIDs only (step 2); to extend the benchmark, retrieve from the queries (step 1).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-python3}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
D=data_preparation

echo "== 1. retrieve from the frozen queries (network; grows with PubMed) =="
"$PY" $D/01_retrieve_pubmed.py --all --overwrite

echo "== 2. restore abstract text for the archived identifiers (network) =="
"$PY" $D/02_fetch_abstracts.py

echo "== 3. dated re-retrieval audit: one UTC timestamp and SHA-256 per query =="
"$PY" $D/03_reretrieve_audit.py

echo
echo "External-1/2 retrieval is NOT regenerable: the query strings were never released."
echo "See config/external_sources.json. Their units are rebuilt in data_preprocessing instead."
