# data_preprocessing — stage 3 of 3

Turns records and annotations into everything the audit consumes. CPU, apart from the embeddings.

```bash
bash data_preprocessing/run_data_preprocessing.sh
```

| Script | Produces |
|---|---|
| `01_build_benchmark.py` | the claim × cutoff benchmark: the 55 pre-cutoff variables at every cutoff, the operational state, and the post-cutoff evaluation fields |
| `02_build_effects_cache.py` | parsed effect estimates per (claim, record) — the cache that lets the anchor analyses run without abstracts |
| `03_build_numeric_anchor.py` | the LLM-independent numeric anchor |
| `04_build_system_scores.py` | Compact-7 and its leave-one-claim-out predictions |
| `05_build_record_embeddings.py` | PubMedBERT record embeddings for the text-reading baseline (GPU) |
| `lib/nutrimature_features.py` | **the frozen feature builder** — the authoritative definition of all 55 variables |
| `external/driver.py` | the External-1/2 claim-cutoff units, rebuilt from the released unit files |


