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

## What a unit is

A unit is a (claim, cutoff) pair. A claim contributes units only if it has **≥ 45** dated labelled
records. Then `lo` is the publication year of the 40th record and `hi` is the last record year − 1;
if `hi − lo < 4` the claim contributes nothing. The cutoffs are the distinct integers among 7
equally spaced points from `lo + 2` to `hi` — so 3 to 7 cutoffs per claim.

Pre-cutoff means publication year **< t**. Post-cutoff means **≥ t**: the cutoff year itself is
post-cutoff. Every one of the 55 variables is computed from pre-cutoff records only.

## The 55 variables

Six blocks — direction, heterogeneity, decline, design, volume, derived. The paper's feature
inventory table gives the operational definition of each column; `lib/nutrimature_features.py` is
the definition that actually runs. Seven of them form **Compact-7**: `pooled_direction`,
`agreement`, `n_directional`, `n_pre_t`, `val_mean`, `val_sd`, `n_rr`.

A few are exact duplicates of others by construction (`n_records` = `n_pre_t`, `n_rr_studies` =
`n_numeric` = `n_rr`, `val_spread` = `spread`). They are kept because the ablation blocks are
reported as defined, not as de-duplicated.

## The numeric anchor

`03_build_numeric_anchor.py` locates every 95% confidence interval in every record, reads the
effect-measure phrase immediately before it (or, failing that, immediately after), and places ratio
measures on the log scale and additive measures on the linear scale. **The scale is never inferred
from the sign of the limits.** An interval is attributed to the claim only when the exposure term
and the outcome term both occur in the interval's own sentence or in the one preceding it.

**This script rewrites `data/derived/numeric_anchor_v2_*.csv`.** Run it only with the abstracts
present. Without them it writes empty tables, and every anchor analysis downstream silently loses
its input.
