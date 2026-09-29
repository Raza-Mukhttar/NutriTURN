# Reproducing the paper

Every number in `paper/NutriTURN.pdf` comes from a file in `results/`, and every file in
`results/` is written by one script in `scripts/`. This file maps between them.

```bash
pip install -r requirements.txt
PYTHONHASHSEED=0 python scripts/verify_release.py    # 1 minute, rebuilds the cohorts
PYTHONHASHSEED=0 bash scripts/run_all.sh             # everything, a few hours
```

`PYTHONHASHSEED=0` is mandatory — see `ENVIRONMENT.md`. Every script prints the path of each
file it writes as its last line, so the mapping below can always be re-derived by running it.

## Headline claims

| Claim in the paper | Script | Output |
|---|---|---|
| 1,012 units; 71 / 52 / 55 events | `scripts/verify_release.py` | stdout |
| Source AUROCs .948 / .857 / .898 | `scripts/verify_release.py` | stdout |
| Calendar 756/58, 625/45, 492/35; .913 / .920 / .876 | `scripts/verify_release.py` | stdout |
| Benchmark rebuilds byte-identically from records and labels | `experiments/smoke_test.py` | `results/guide/smoke_test.json` |
| Endpoint contract, Eq. (2) identities, monotonicity | `experiments/endpoint_contract.py` | `results/guide/endpoint_contract.json` |

## Tables

| Table | Script(s) | Output |
|---|---|---|
| Protocol table (the five checks) | `experiments/full_metric_suite.py`, `experiments/core_reanalyses.py` | `results/issue4/full_metrics.{csv,json}` |
| Full metric suite, source cohort | `experiments/full_metric_suite.py` | `results/issue4/full_metrics.csv` |
| Stationary references | `experiments/stationary_baselines.py` | `results/issue1/stationary_baselines.json`, `results/issue1/source_scores.csv` |
| Temporal offset / order features | `experiments/temporal_offset.py`, `experiments/recalibrated_order_features.py` | `results/issue2/` |
| Sign-invariant and flip-augmented scores | `experiments/sign_invariant_scores.py` | `results/issue3/` |
| Calendar cohort census | `experiments/origin_calendar_cohort.py` | `results/issue5/origin_cohort.json`, `pooled_scores_H{3,5,10}.csv` |
| The three "not evaluated" census rows | `experiments/incomplete_origin_windows.py` | `results/R6_incomplete_origins.json` |
| Extractor agreement | `experiments/extractor_agreement.py` | `results/issue6/` |
| Feature inventory / dataset card | `experiments/dataset_card.py` | `results/issue7/` |
| Agreement definition x annotation x support | `experiments/agreement_definition_variants.py` | `results/R1_agreement_branch.json` |
| Branch-free intervention | `experiments/agreement_definition_audit.py`, `experiments/agreement_branch_and_class_prior.py` | `results/agreement_v2/`, `results/N2_N5_branch.json` |
| Exact AUROC pair decomposition | `experiments/auroc_pair_decomposition.py` | `results/R3_pair_decomposition.json` |
| Alternative inference procedures (all seven cells) | `experiments/wild_cluster_bootstrap.py`, `experiments/clustered_roc_all_cells.py --refit`, `experiments/calendar_refit_bootstrap.py` | `results/N6_wild_bootstrap.json`, `results/R5b_table23.json`, `results/R5c_calendar_refit.json` |
| Cluster-aware inference, development cells | `experiments/cluster_aware_inference.py` | `results/R5_inference.json` |
| Null controls, partition grid (all annotations) | `experiments/null_controls.py` | `results/E2_null_{primary,second,consensus}.json`, `results/E2_draws_*.npz` |
| Null C invariance | `experiments/null_c_invariance.py` | `results/R2_nullc_invariance.json` |
| Null D optimiser settings and case counts | `experiments/null_d_optimizer_audit.py` | `results/R8_nullD_audit.json` |
| Stationary null distributions | `experiments/stationary_null_{permutation,simulation,summary}.py` | `results/stationary_null/` |
| Numeric anchor, by era and by scale | `experiments/numeric_anchor_by_era.py`, `experiments/anchor_paired_strata.py`, `experiments/anchor_slope_differences.py` | `results/N1_N3_anchor.json`, `results/R4*.json` |
| Intersection anchor, second-pipeline confusion | `experiments/intersection_anchor_and_confusion.py` | `results/R6_misc.json` |
| Two-variable transfer, external cohorts | `experiments/two_variable_transfer.py` | `results/plan/` |
| Cumulative and three-state endpoints | `experiments/cumulative_endpoints.py` | `results/cumulative_target/` |
| Orientation audit | `experiments/orientation_audit.py` | `results/orientation_clean/` |
| Documented reversals; second logistic coefficient | `experiments/documented_reversals.py` | `results/N4_N8_reversals_coef.json` |
| Pivotal source resolution (network) | `experiments/pivotal_source_resolution.py` | `results/R4c_pivotal_sources.json` |
| Eligible cutoffs before each pivotal year | `experiments/documented_reversals.py` | `results/N4_eligible_cutoffs.json` |
| Retrieval cap sensitivity | `experiments/retrieval_cap_sensitivity.py` | `results/E15_cap_census.csv` |
| Retrieval audit reconciliation | `experiments/retrieval_audit_reconciliation.py` | `results/E7_retrieval.json` |
| Snapshot provenance bound | `experiments/snapshot_provenance.py` | `results/guide/snapshot_provenance.json` |
| Controlled annotation design (model x prompt x read-out) | `experiments/controlled_annotation_design.py` | `results/E8_cells.json` |
| Annotation drift by decade | `experiments/annotation_drift_by_decade.py drift`, `experiments/within_claim_label_drift.py` | `results/E13_drift.json`, `results/E17_drift_within.json` |
| Accrual and censoring | `experiments/accrual_and_censoring.py` | `results/E6_accrual.json` |
| Resampling power | `experiments/resampling_power.py` | `results/E11_power.json` |

LaTeX versions of the tables are regenerated into `tables/` by
`scripts/build_latex_tables_main.py` (the main-text tables) and
`scripts/build_latex_tables_audit.py` (the audit tables `tab:astra1`-`tab:astra5`
and `tab:astra7`).

## Figures

| Figure | Source | Output |
|---|---|---|
| Figure 1, protocol schematic | `figures/src/nutrifig.tex` (TikZ) | `figures/nutrifig.pdf` |
| Figure 4, analytic vs plug-in simulation | `scripts/figures/regenerate_main_figures.py` | `figures/fig4_analytic_vs_simulation.pdf` |
| Figure 5, calendar cohort | `scripts/figures/regenerate_main_figures.py` | `figures/fig5_calendar_cohort.pdf` |
| Figure 6, null audit | `scripts/figures/regenerate_main_figures.py` | `figures/fig6_null_audit.pdf` |

Figure 1 is compiled with any LaTeX engine: `tectonic figures/src/nutrifig.tex`.
`regenerate_main_figures.py` asserts at run time that no text, legend or mark overlaps and
refuses to save a figure that fails the check.


## Building NutriTURN from scratch

`scripts/pipeline/` holds the construction stages, in order. These are **not** part of
`run_all.sh`: they need GPUs, network access and the abstracts.

| Stage | Script |
|---|---|
| Re-fetch abstracts by released PMID | `pipeline/fetch_abstracts.py` |
| Dated re-retrieval audit | `pipeline/reretrieve_pubmed.py` |
| Primary and controlled-cell annotation | `pipeline/annotate_controlled_cells.py` |
| Second-pipeline annotation | `pipeline/annotate_second_pipeline.py`, `pipeline/annotate_second_pipeline_origprompt.py` |
| Parse effect estimates into the cache | `pipeline/build_effects_cache.py` |
| Build the numeric anchor | `pipeline/build_numeric_anchor.py` |
| Build the claim-cutoff benchmark | `pipeline/build_benchmark.py` |
| Build the system scores | `pipeline/build_system_scores.py` |

`pipeline/build_numeric_anchor.py` **rewrites** `data/derived/numeric_anchor_v2_*.csv`. Run it
only with the abstracts present; without them it writes empty tables and the anchor analyses
lose their input.

## Not reproducible from this release

- Anything needing abstract text, until `pipeline/fetch_abstracts.py` has been run.
- Text-reading methods on the external corpora: those ship as unit-level label counts with no
  identifiers, which is why the paper restricts text-reading methods to checks 1-4.
- The exact original retrieval timestamp, which was not recorded; the paper gives a bound.
- The candidate-claim-list provenance, which was not recorded.
- Human validation of the target: the adjudication frame is released, unexecuted.
