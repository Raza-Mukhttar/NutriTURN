# NutriTURN

Code for *High Scores, Fragile Targets: Auditing Change in LLM-Annotated Diet-Health Evidence*.

NutriTURN audits whether a learned system that predicts **change in LLM-annotated evidence
direction** is measuring the evidence or the annotation. The repository contains the whole chain:
the retrieval that builds the corpus, the annotation that labels it, the feature construction, the
audit pipeline, and one script per experiment reported in the paper.

**The data is not redistributed.** PubMed abstract text cannot be republished under its licence, so
this repository ships the *scripts that rebuild the dataset exactly as the study built it* rather
than the dataset itself. Run the three data stages in order and you get the same corpus, the same
annotations and the same features; run the audit on them and you get the paper's numbers.

---

## Layout

```
NutriTURN-Repository/
├── config/                 frozen protocol constants, model/read-out index, the 165 claim queries
├── requirement.txt         pinned environment
├── data_preparation/       stage 1 — PubMed retrieval (the only stage that needs the network)
├── data_annotation/        stage 2 — every annotation cell, one per (model, prompt, read-out)
├── data_preprocessing/     stage 3 — the 55 pre-cutoff variables, caches, anchors, system scores
├── audit_pipeline/         the frozen audit method, its metrics, and the driver that runs it
└── audit_experiments/      one script per experiment in the paper (51), plus figures and tables
```

Produced at run time, and deliberately absent from the repository: `data/`, `results/`, `tables/`,
`figures/`, `logs/`.

---

## Quickstart

```bash
pip install -r requirement.txt
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1      # required, not optional

bash data_preparation/run_data_preparation.sh          # network,  ~2-4 h
bash data_annotation/run_data_annotation.sh            # GPUs,     ~6-10 h on one card
bash data_preprocessing/run_data_preprocessing.sh      # CPU,      ~1 h

python audit_pipeline/run_audit.py                     # the core audit, ~20 min
bash audit_experiments/run_all_experiments.sh          # every reported number, a few hours
```

`PYTHONHASHSEED=0` is a correctness requirement: `audit_experiments/null_controls.py` derives each
draw's seed from Python's `hash()` of the control's name, and Python salts string hashes per process
unless the variable is fixed. Without it the null controls produce different — still valid, but
different — draws. `config/environment.json` records this and the rest of the environment.

### If you already hold the data

With `data/` in place, the audit verifies itself against the paper in about a minute:

```
$ python audit_pipeline/verify_release.py
  annotation    units  events   exp  AUROC two-var    exp  ok
  primary        1012      71    71         0.9480  0.948  yes
  second         1012      52    52         0.8572  0.857  yes
  consensus      1012      55    55         0.8984  0.898  yes
   horizon  units   exp  events   exp    AUROC    exp  ok
  H=3         756   756      58    58   0.9127  0.913  yes
  H=5         625   625      45    45   0.9204  0.920  yes
  H=10        492   492      35    35   0.8756  0.876  yes
  ALL CHECKS PASS
```

---

## The three stages, and what each one fixes

### Stage 1 — `data_preparation/`

| Script | What it does |
|---|---|
| `01_retrieve_pubmed.py` | ESearch + EFetch for each of the 165 claim queries in `config/claim_queries.json`, no record cap; merges with the claim's existing history so a history never loses a record |
| `02_fetch_abstracts.py` | restores abstract text for the **archived** identifiers |
| `03_reretrieve_audit.py` | a dated re-retrieval: one UTC timestamp, the returned PMID list and its SHA-256 per query, plus a version-to-version comparison |
| `external/` | the External-1/2 input manifest and the primary-reconstruction probe |

**PubMed keeps growing.** A retrieval run today returns *more* records than the September 2026
snapshot the paper reports, which is why the paper's numbers are tied to the archived identifier
lists rather than to a live query. To reproduce the paper exactly, use `02_fetch_abstracts.py`
against the archived PMIDs. To extend the benchmark, use `01_retrieve_pubmed.py`.

### Stage 2 — `data_annotation/`

Six annotation cells, each a single (model, prompt, read-out) combination at an identical text
budget — 3,000 characters, 1,600-token truncation, bfloat16, one forward pass, no sampling. Holding
everything else fixed is what makes a cell-to-cell difference attributable to the annotator.
`config/annotation_models.json` is the index.

| Cell | Model | Prompt | Read-out | Script |
|---|---|---|---|---|
| primary | Llama-3.1-8B-Instruct | P1 | first-token | `01_annotate_primary.py` |
| D (second pipeline) | Qwen2.5-7B-Instruct | P2 | full-string | `02_annotate_second_pipeline.py` |
| C | Qwen2.5-7B-Instruct | P1 | full-string | `03_annotate_second_pipeline_origprompt.py` |
| A′ | Llama-3.1-8B-Instruct | P1 | full-string | `04_annotate_controlled_cells.py --cell Aprime` |
| B | Llama-3.1-8B-Instruct | P2 | full-string | `04_annotate_controlled_cells.py --cell B` |
| M (third family) | Mistral-7B-Instruct-v0.3 | P2 | full-string | `05_annotate_third_family.py --cell M` |

Two read-outs, defined precisely because the paper's cells differ by them: **first-token** is a
softmax over the leading token of each label at the answer position; **full-string** sums the
log-probabilities of *all* of a label's tokens and takes a softmax over the four sums, without
length normalisation.

The third family needs `text_full.json`, not the records snapshot — the snapshot carries no
abstracts, and running it against the snapshot silently produces a cell with zero abstracts.

### Stage 3 — `data_preprocessing/`

| Script | What it does |
|---|---|
| `01_build_benchmark.py` | the claim × cutoff benchmark: the **55 pre-cutoff variables** at up to seven cutoffs per claim, plus the post-cutoff evaluation fields |
| `02_build_effects_cache.py` | parsed effect estimates per (claim, record), so the anchor analyses run without abstracts |
| `03_build_numeric_anchor.py` | the LLM-independent numeric anchor: effect measures read next to each 95% interval, attributed to a claim only when exposure and outcome both occur in the interval's sentence or the one before |
| `04_build_system_scores.py` | Compact-7 and its leave-one-claim-out predictions |
| `05_build_record_embeddings.py` | PubMedBERT record embeddings for the text-reading baseline (GPU) |
| `lib/nutrimature_features.py` | the frozen feature builder — the authoritative definition of all 55 variables |

**The cutoff grid**, which determines what a unit is: a claim needs ≥ 45 dated labelled records;
`lo` is the publication year of the 40th record, `hi` the last record year − 1; the claim is skipped
if `hi − lo < 4`; the cutoffs are the distinct integers among 7 equally spaced points from `lo + 2`
to `hi`. That yields 3–7 cutoffs per claim.

`03_build_numeric_anchor.py` **rewrites** `data/derived/numeric_anchor_v2_*.csv`. Run it only with
the abstracts present; without them it writes empty tables and the anchor analyses lose their input.

---

## The audit pipeline

`audit_pipeline/` is the study's method, not an experiment:

| File | Role |
|---|---|
| `lib/nutrimature.py` | paths, the frozen protocol, leave-one-claim-out evaluation, metrics |
| `lib/nfx_common.py` | the endpoint contract, the sparse gate, the claim-clustered bootstrap |
| `lib/pro_common.py`, `lib/astra_common.py`, `lib/plan_common.py` | result IO, the count-aware reference, transfer helpers |
| `endpoint_contract.py` | checks the closed-form identities and monotonicity of the endpoint |
| `verify_release.py` | rebuilds the cohorts from the labels and checks them against the paper |
| `run_audit.py` | runs the core audit in dependency order (`--list` shows it) |

**The endpoint, stated once.** The pooled pre-cutoff direction is `p_t = (h − l) / n` over signed
pre-cutoff annotations when at least `SPARSE_GATE = 8` signed records are present, and **0**
otherwise. The post-cutoff direction `p_post` is **ungated**. A unit is an event when
`sgn0(p_post) ≠ sgn0(p_t)`. Pre-cutoff means year `< t`; post-cutoff means year `≥ t`, so the cutoff
year itself is post-cutoff. `config/protocol.json` carries these values for inspection, but
`lib/nfx_common.py` remains authoritative.

Stage order matters: several stages read what earlier ones wrote. `run_audit.py --list` prints the
order; `--from N` resumes.

---

## Reproducing the paper's numbers

### Headline claims

| Claim | Script | Output |
|---|---|---|
| 1,012 units; 71 / 52 / 55 events | `audit_pipeline/verify_release.py` | stdout |
| Source AUROCs .948 / .857 / .898 | `audit_pipeline/verify_release.py` | stdout |
| Calendar 756/58, 625/45, 492/35; .913 / .920 / .876 | `audit_pipeline/verify_release.py` | stdout |
| Two-variable vs margin **+0.0667 [+0.017, +0.120]** | `audit_experiments/stationary_baselines.py` | `results/issue1/stationary_baselines.json` |
| Its exact pair decomposition +0.0609 / +0.0068 / −0.0010 | `audit_experiments/auroc_pair_decomposition.py` | `results/R3_pair_decomposition.json` |
| Benchmark rebuilds byte-identically | `audit_experiments/smoke_test.py` | `results/guide/smoke_test.json` |
| Endpoint contract identities and monotonicity | `audit_pipeline/endpoint_contract.py` | `results/guide/endpoint_contract.json` |

### Tables and figures

Every number in the paper comes from a file in `results/`, and every file in `results/` is written
by one script in `audit_experiments/`. Each script prints the path of each file it writes as its
last line, so the mapping can always be re-derived by running it.

| Paper object | Script(s) in `audit_experiments/` | Output |
|---|---|---|
| Protocol table (the five checks) | `full_metric_suite.py`, `core_reanalyses.py` | `results/issue4/full_metrics.{csv,json}` |
| Stationary references | `stationary_baselines.py` | `results/issue1/` |
| Temporal offset / order features | `temporal_offset.py`, `recalibrated_order_features.py` | `results/issue2/` |
| Sign-invariant and flip-augmented scores | `sign_invariant_scores.py` | `results/issue3/` |
| Calendar cohort census | `origin_calendar_cohort.py` | `results/issue5/` |
| The three "not evaluated" census rows | `incomplete_origin_windows.py` | `results/R6_incomplete_origins.json` |
| Extractor agreement | `extractor_agreement.py` | `results/issue6/` |
| Feature inventory / dataset card | `dataset_card.py` | `results/issue7/` |
| Agreement definition × annotation × support | `agreement_definition_variants.py` | `results/R1_agreement_branch.json` |
| Branch-free intervention | `agreement_definition_audit.py`, `agreement_branch_and_class_prior.py` | `results/agreement_v2/`, `results/N2_N5_branch.json` |
| Alternative inference procedures | `wild_cluster_bootstrap.py`, `clustered_roc_all_cells.py --refit`, `calendar_refit_bootstrap.py` | `results/N6_wild_bootstrap.json`, `results/R5b_table23.json`, `results/R5c_calendar_refit.json` |
| Cluster-aware inference | `cluster_aware_inference.py` | `results/R5_inference.json` |
| Null controls, partition grid | `null_controls.py` | `results/E2_null_*.json`, `results/E2_draws_*.npz` |
| Null C invariance / Null D audit | `null_c_invariance.py`, `null_d_optimizer_audit.py` | `results/R2_nullc_invariance.json`, `results/R8_nullD_audit.json` |
| Stationary null distributions | `stationary_null_{permutation,simulation,summary}.py` | `results/stationary_null/` |
| Numeric anchor by era and scale | `numeric_anchor_by_era.py`, `anchor_paired_strata.py`, `anchor_slope_differences.py` | `results/N1_N3_anchor.json`, `results/R4*.json` |
| Two-variable transfer, external cohorts | `two_variable_transfer.py` | `results/plan/` |
| Cumulative and three-state endpoints | `cumulative_endpoints.py` | `results/cumulative_target/` |
| Orientation audit | `orientation_audit.py` | `results/orientation_clean/` |
| Documented reversals | `documented_reversals.py` | `results/N4_N8_reversals_coef.json` |
| Retrieval cap sensitivity / audit reconciliation | `retrieval_cap_sensitivity.py`, `retrieval_audit_reconciliation.py` | `results/E15_cap_census.csv`, `results/E7_retrieval.json` |
| Controlled annotation design (model × prompt × read-out) | `controlled_annotation_design.py` | `results/E8_cells.json` |
| Annotation drift by decade | `annotation_drift_by_decade.py drift`, `within_claim_label_drift.py` | `results/E13_drift.json`, `results/E17_drift_within.json` |
| Accrual and censoring / resampling power | `accrual_and_censoring.py`, `resampling_power.py` | `results/E6_accrual.json`, `results/E11_power.json` |
| Figure 1 (protocol schematic) | `figures/src/nutrifig.tex` (TikZ) | `figures/nutrifig.pdf` |
| Figures 4–6 | `figures/regenerate_main_figures.py` | `figures/fig{4,5,6}_*.pdf` |
| LaTeX tables | `reporting/build_latex_tables_{main,audit}.py` | `tables/` |

`regenerate_main_figures.py` asserts at run time that no text, legend or mark overlaps, and refuses
to save a figure that fails the check. Figure 1 compiles with any LaTeX engine:
`tectonic audit_experiments/figures/src/nutrifig.tex`.

---

## The three levels of reproducibility

The paper distinguishes three claims, and this repository supports them to different degrees. Being
explicit about which level a given number sits at is the point.

1. **Replay the stored metrics** — from per-unit prediction files, if you hold `results/`.
2. **Rebuild from the released annotations** — fully: counts, pooled directions, agreement under
   every definition, every endpoint, the stationary probabilities and the null controls. This is
   what `audit_pipeline/verify_release.py` demonstrates.
3. **Re-execute the text-dependent methods** — possible for the source and calendar cells once
   `data_preparation/02_fetch_abstracts.py` has restored the abstracts; **not** possible for the
   external corpora.

## What this repository cannot reproduce

Stated plainly, because each is a real limit and not an oversight:

- **External-1/2 retrieval.** The query strings were never released, so their retrieval cannot be
  re-run from the query forward; `03_reretrieve_audit.py` marks them non-regenerable by design.
  Their *units* are rebuilt from the released unit files, and their re-annotation runs on the
  retained retrieval texts (κ = 0.959 on External-1, 0.967 on External-2, 0.964 [0.957, 0.971]
  pooled against the archived labels). Population alignment was established — identifier sets and
  pre/post-cutoff record counts match for all 46 External-1 and 180 External-2 claims — which is
  not the same as identical input text.
- **Text-reading methods on the external corpora.** Those corpora ship as unit-level label counts
  with no record identifiers. This is what restricts text-reading methods to checks 1–4.
- **The original retrieval timestamp**, which was not recorded; the paper gives a bound instead.
- **The candidate-claim-list provenance**, which was not recorded.
- **Human validation of the target.** The adjudication frame is released, unexecuted. No human or
  expert label in this repository was generated by a model.

A note on duplicate keys, since re-execution can produce them: where an association key occurs more
than once in a re-executed cell, **the first occurrence in sorted file order** is used. The
last-occurrence rule disagrees on 476 of 33,737 duplicated keys and yields the same event count
(78).

## Provenance

`PROVENANCE.json` records, for every script here, its path in the original study tree and the
SHA-256 of both the source and the copy. The only edit applied during assembly was to the path
bootstrap at the top of each file, so that scripts resolve this repository's root instead of the
original tree's; the analysis code is otherwise byte-for-byte the code that produced the paper.

## Local verification

`data/` and `results/` are absent from the repository by design. If you hold a local copy of the
protocol snapshot, point `data/` at it (a symlink is fine) and `audit_pipeline/verify_release.py`
will run immediately; `results/` is then created by the audit stages. Both paths are in
`.gitignore`, as are `tables/`, `figures/` and `logs/`, which the library creates on import.
