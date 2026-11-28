# Results added in response to review

This file collects, in one place, every result cited in the response to the ARR October 2026
reviews (reviewers zhvc, asRP and AqvV), with the full tables behind each answer. It adds no
claim that is not in the responses. Each table names its source: a table of the submitted paper,
a file of the submitted data archive (`data_release/`), or a folder beside this file
(`rebuttal/` in the repository) that regenerates it.

`data_release/` below means the submitted archive, which unpacks as `data_release_nutriturn/`.

## How to read this file

**Intervals.** Unless a table says otherwise, an interval is a 95% percentile bootstrap over
claims (2,000 resamples, claim-clustered), and a contrast is *established* when that interval
excludes zero. AUROC differences are learned score minus reference.

**Terms.** A *claim* pairs an exposure with an outcome; an *association* is one (claim, PubMed
record) pair with one label (PROTECTIVE, HARMFUL, NULL or UNCLEAR); a *unit* is one (claim, cutoff
year) pair. For a unit, p_t = (h - l)/(h + l) over the HARMFUL (h) and PROTECTIVE (l) records
before the cutoff, set to 0 when h + l < 8 (the *sparse gate*); p_t+ is the same ratio after the
cutoff, without the gate. A unit is an *event* when the two signs differ. The *margin rule* scores
1 - |p_t|; the *stationary reference* is the probability that the future majority differs from
the current sign if later labels were exchangeable with earlier ones. *NULL-aware agreement*
(definition A) is the NULL share when that share is at least 0.5 and the signed majority share
otherwise; *signed-only agreement* (definition C) is the signed majority share (1 + |p_t|)/2
throughout. The *agreement swap* replaces A with C with annotation, units and outcomes fixed.

**When each result was specified.**

| Status | Results |
|---|---|
| Confirmatory: specification frozen and timestamped before outcomes existed | frozen cohort: prespecified primary (vs stationary) and secondary (vs margin) |
| Retrospective: outcomes already existed | development backtest, calendar cohort, frozen transfer to External-1 and External-2, re-annotation, audit cells, 48-cell multiverse, sparse-gate and endpoint variants, Bashir audit |
| Exploratory | agreement-swap comparisons and their decomposition, and everything marked *new* below |

**New since submission** (exploratory, each with its own folder and README):

| Folder | What it holds | Answers | Runs on |
|---|---|---|---|
| `full_precision_70b/` | Llama-3.1-70B annotation of all 112,453 associations in bfloat16; scripts, logs, output, `fp70_numbers.json` (input text not included) | zhvc C3 | GPU for the annotation; analysis on CPU with `data/` in place |
| `equivalence_tests/` | two one-sided tests on 28 contrasts | asRP W6 | CPU, about three minutes, with `data/` in place |
| `cto_audit_proof/` | audit of the CTO clinical-trial-outcome system (Gao et al., 2024); 136 values checked against expected output | zhvc C1, C4; asRP W2, Q7 | CPU, about three minutes |
| `noise_null_calibration/` | noise-null calibration of event overlap | AqvV C10 | CPU, about eight minutes |
| `worked_example/` | endpoint rebuilt for all 1,012 units from released labels; one claim printed | AqvV C3 | CPU, seconds |
| `appendix_font_fix/` | appendix font patch, builds before and after, font and margin measurements | AqvV C6 | tectonic, PyMuPDF |

**Concern index.**

| Reviewer | Concern | Section |
|---|---|---|
| zhvc | C1 scope beyond PubMed | [1.1](#11-scope-beyond-the-source-corpus-zhvc-c1) |
| zhvc | C2 "only the Qwen family" | [1.2](#12-annotator-inventory-zhvc-c2) |
| zhvc | C3 weak annotators: read-out, capacity, 4-bit | [1.3](#13-annotator-strength-zhvc-c3) |
| zhvc | C4 predictor is the authors' own | [1.4](#14-predictors-and-systems-not-built-by-the-authors-zhvc-c4) |
| asRP | W1/Q1 ground truth | [2.1](#21-ground-truth-asrp-w1-q1) |
| asRP | W3 generalisability | [2.2](#22-generalisability-asrp-w3) |
| asRP | W4 evaluation design | [2.3](#23-evaluation-design-asrp-w4) |
| asRP | W5 parameterisation | [2.4](#24-parameterisation-of-the-agreement-input-asrp-w5) |
| asRP | W6 absence of evidence vs equivalence | [2.5](#25-equivalence-tests-asrp-w6) |
| asRP | Q4 event definition | [2.6](#26-event-definition-asrp-q4) |
| asRP | Q5 future-evidence conditioning | [2.7](#27-conditioning-on-future-evidence-asrp-q5) |
| asRP | Q6 error vs interpretation | [2.8](#28-error-versus-interpretation-asrp-q6) |
| asRP | Q7 human-validated benchmark | [2.9](#29-human-validated-benchmarks-asrp-q7) |
| AqvV | C1-C2 motivation and hypotheses | [3.1](#31-hypotheses-and-results-aqvv-c1-c2) |
| AqvV | C3 corpus, terminology, worked example | [3.2](#32-corpus-terminology-and-a-worked-example-aqvv-c3) |
| AqvV | C4 provenance | [3.3](#33-provenance-aqvv-c4) |
| AqvV | C5 labelling quality | [3.4](#34-labelling-quality-aqvv-c5) |
| AqvV | C6 appendix font | [3.5](#35-appendix-font-and-spacing-aqvv-c6) |
| AqvV | C7 comparisons and concepts | [3.6](#36-what-each-comparison-holds-fixed-aqvv-c7) |
| AqvV | C8 the 61-claim cohort | [3.7](#37-the-specification-frozen-cohort-aqvv-c8) |
| AqvV | C9 pair attributions | [3.8](#38-exact-pair-decomposition-aqvv-c9) |
| AqvV | C10 why Jaccard | [3.9](#39-event-overlap-and-its-noise-null-aqvv-c10) |

---

## 1. Reviewer zhvc

### 1.1 Scope beyond the source corpus (zhvc C1)

Source: paper Tables 5 and 32; CTO row new (`cto_audit_proof/`).

| Setting | Domain | Size | Events | Learned score minus untrained reference |
|---|---|---|---|---|
| Source cohort | diet-health | 165 claims, 1,012 units | 71 | +0.067 [0.017, 0.120] |
| External-1 | physical activity, sleep, environment | 46 claims, 322 units | 19 | +0.055 [-0.052, 0.106] |
| External-2 | medication, screening, occupational exposure, clinical risk | 180 claims, 1,257 units | 95 | -0.074 [-0.123, -0.024] |
| Bashir et al. (2021) review updates | human-built labels | 256 pairs | 43 | +0.042 [-0.048, 0.139] |
| CTO (Gao et al., 2024), *new* | clinical-trial outcomes | 11,210 trials | n/a | see 1.4 |

No claim is shared between the source cohort and either external corpus. On External-2 the two
core annotators define 95 and 191 events and share 20 of 266 (Jaccard 0.075).

### 1.2 Annotator inventory (zhvc C2)

Source: paper Table 1 and Appendix B.2; bfloat16 row new (`full_precision_70b/`).

| Annotator | Family | Prompt | Read-out | Precision | Role | Associations |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | Llama | P1 | first-token logits | bfloat16 | primary | 112,453 |
| Qwen2.5-7B | Qwen | P2 | full label string | bfloat16 | second | 112,453 |
| Llama-3.1-70B | Llama | P2 | full label string | 4-bit NF4 | scale diagnostic | 112,453 |
| Llama-3.1-70B, *new* | Llama | P2 | full label string | bfloat16 | full-precision 70B annotator | 112,453 |
| Qwen2.5-72B | Qwen | P2 | full label string | 4-bit NF4 | scale diagnostic | 112,453 |
| Mistral-7B-v0.3 | Mistral | P2 | full label string | bfloat16 | third family | 112,453 |
| PubMedBERT | encoder | n/a | fine-tuned reader | n/a | text baseline | n/a |

### 1.3 Annotator strength (zhvc C3)

**(a) Read-out.** One factor changes per row and the events are rebuilt each time. Source: paper
Table 7; first-token mass from `data_release/04_results/results/issue6/extractor_agreement.json`
(`mean_first_token_mass` = 0.99909).

| Factor changed | Between | Label kappa | Event Jaccard | Events |
|---|---|---|---|---|
| Batching | primary re-run | 0.980 | 0.846 | 71 to 73 |
| Read-out | Llama P1: first token to full string | 0.986 | 0.899 | 73 to 77 |
| Prompt | Llama: P1 to P2 | 0.633 | 0.520 | 77 to 75 |
| Prompt | Qwen: P1 to P2 | 0.823 | 0.538 | 49 to 51 |
| Model | P1: Llama to Qwen | 0.561 | 0.312 | 77 to 49 |
| Model | P2: Llama to Qwen | 0.402 | 0.312 | 75 to 51 |
| Model | P2: Llama to Mistral | 0.493 | 0.267 | 75 to 96 |
| Model | P2: Qwen to Mistral | 0.383 | 0.246 | 51 to 96 |

The read-out moves the event set no more than a batch-size rerun; prompt and model family are
what move the target.

**(b) Capacity.** Agreement with an LLM-independent extractor of typed 95% confidence intervals on
the same 1,841 records. Source: paper Table 6; bfloat16 row new.

| Annotator | Agreement, all | Resolved (%) | Agreement, resolved | Balanced accuracy |
|---|---|---|---|---|
| Llama-3.1-8B (primary) | 0.807 [0.782, 0.830] | 99.5 | 0.811 | 0.754 |
| Qwen2.5-7B (second) | 0.758 [0.733, 0.782] | 97.7 | 0.776 | 0.724 |
| Llama-3.1-70B, 4-bit | 0.686 | 90.4 | 0.758 | 0.655 |
| Llama-3.1-70B, bfloat16, *new* | 0.691 | 92.0 | 0.751 | 0.661 |
| Qwen2.5-72B, 4-bit | 0.715 | 91.7 | 0.779 | 0.674 |

Larger annotators are not closer to the independent anchor at either precision.

**(c) Weight precision (new).** Llama-3.1-70B-Instruct from the official checkpoint in bfloat16,
over all 112,453 associations, with prompt, read-out, text budget and truncation unchanged. Before
the bfloat16 analysis, the script reproduced all 56 published values of the 4-bit cell. Source:
`full_precision_70b/output/fp70_numbers.json`.

Labels and events:

| Precision | Labels: PROTECTIVE / HARMFUL / NULL / UNCLEAR (%) | Events (from sparse-gated histories) | Anchor agreement, all | Resolved (%) | Anchor agreement, resolved | Balanced accuracy |
|---|---|---|---|---|---|---|
| 4-bit NF4 | 36.7 / 18.1 / 14.1 / 31.1 | 55 (21) | 0.686 | 90.4 | 0.758 | 0.655 |
| bfloat16 | 38.1 / 20.5 / 10.4 / 31.0 | 34 (14) | 0.691 | 92.0 | 0.751 | 0.661 |

Precision alone, with model, prompt and read-out fixed:

| Quantity | Value |
|---|---|
| Labels changed, 4-bit to bfloat16 | 19,105 of 112,453 (17.0%) |
| Label kappa / raw agreement | 0.761 / 0.830 |
| Largest per-record label-probability change: median / 95th percentile / share above 0.1 | 0.074 / 0.510 / 44.1% |
| Event Jaccard, 4-bit against bfloat16 (paper endpoint) | 0.589 (33/56); event kappa 0.730 |
| Event Jaccard, sparse-gated histories as non-events | 0.543 (19/35) |
| Two equivalent bfloat16 scoring implementations (6,569 records) | label kappa 0.983, 55 labels differ |
| Published 4-bit values reproduced before the bfloat16 run | 56 of 56 |

For comparison, a batch-size rerun of the primary annotator gives label kappa 0.980 and event
Jaccard 0.846 (66/78).

Event overlap of the Llama-3.1-70B cell with the other annotations, at each precision:

| Llama-3.1-70B against | Precision | Label kappa | Event Jaccard, paper endpoint | Sparse-gated as non-event | Common support | Event kappa |
|---|---|---|---|---|---|---|
| Llama-3.1-8B (primary) | 4-bit NF4 | 0.406 | 0.235 (24/102) | 0.209 (18/86) | 0.215 (17/79) | 0.341 |
| Llama-3.1-8B (primary) | bfloat16 | 0.408 | 0.207 (18/87) | 0.184 (14/76) | 0.186 (13/70) | 0.312 |
| Qwen2.5-7B (second) | 4-bit NF4 | 0.384 | 0.321 (26/81) | 0.288 (17/59) | 0.281 (16/57) | 0.457 |
| Qwen2.5-7B (second) | bfloat16 | 0.353 | 0.284 (19/67) | 0.216 (11/51) | 0.204 (10/49) | 0.418 |
| Qwen2.5-72B (4-bit) | 4-bit NF4 | 0.615 | 0.527 (39/74) | 0.392 (20/51) | 0.408 (20/49) | 0.672 |
| Qwen2.5-72B (4-bit) | bfloat16 | 0.640 | 0.460 (29/63) | 0.357 (15/42) | 0.366 (15/41) | 0.614 |
| Llama-3.1-8B, prompt P2 | 4-bit NF4 | 0.456 | 0.250 (26/104) | 0.224 (19/85) | 0.222 (18/81) | 0.360 |
| Llama-3.1-8B, prompt P2 | bfloat16 | 0.492 | 0.211 (19/90) | 0.169 (13/77) | 0.162 (12/74) | 0.317 |

Contrasts, both precisions, on all units and on the common support:

| Support | Contrast | 4-bit NF4 | bfloat16 |
|---|---|---|---|
| all 1,012 units | units / events | 1012 / 55 | 1012 / 34 |
| all 1,012 units | AUROC: NULL-aware / signed-only / margin / stationary | 0.899 / 0.965 / 0.963 / 0.977 | 0.958 / 0.953 / 0.942 / 0.958 |
| all 1,012 units | NULL-aware learned minus margin | -0.064 [-0.133, 0.009] | +0.016 [-0.023, 0.051] |
| all 1,012 units | NULL-aware learned minus stationary | -0.078 [-0.140, -0.013] (established) | -0.000 [-0.029, 0.024] |
| all 1,012 units | Signed-only learned minus margin | +0.002 [-0.012, 0.017] | +0.011 [-0.018, 0.038] |
| all 1,012 units | Signed-only learned minus stationary | -0.012 [-0.039, 0.007] | -0.005 [-0.034, 0.022] |
| all 1,012 units | Agreement swap (signed-only minus NULL-aware) | +0.067 [-0.004, 0.134] | -0.005 [-0.019, 0.008] |
| common support | units / events | 987 / 33 | 990 / 19 |
| common support | AUROC: NULL-aware / signed-only / margin / stationary | 0.847 / 0.941 / 0.939 / 0.962 | 0.926 / 0.916 / 0.909 / 0.933 |
| common support | NULL-aware learned minus margin | -0.092 [-0.235, 0.028] | +0.017 [-0.042, 0.079] |
| common support | NULL-aware learned minus stationary | -0.115 [-0.239, -0.017] (established) | -0.007 [-0.048, 0.042] |
| common support | Signed-only learned minus margin | +0.002 [-0.024, 0.027] | +0.007 [-0.032, 0.049] |
| common support | Signed-only learned minus stationary | -0.021 [-0.069, 0.014] | -0.017 [-0.072, 0.049] |
| common support | Agreement swap (signed-only minus NULL-aware) | +0.094 [-0.018, 0.237] | -0.010 [-0.048, 0.016] |

The 4-bit cells on all 1,012 units are shown as published (paper Section 5.3 and Appendix B);
the recomputation in `fp70_numbers.json` matches them to within 0.001 (two interval limits and the 4-bit anchor agreement, 0.685 recomputed).

What the cell changes:

- Precision is an annotator change in its own right: 17.0% of labels change, mostly at the
  NULL/UNCLEAR boundary, and the event count moves from 55 to 34 (33 shared).
- Two 4-bit point estimates move to zero. The NULL-aware loss against the stationary reference,
  established at 4-bit (-0.078 [-0.140, -0.013], paper Section 5.3), is -0.000 [-0.029, 0.024]
  in bfloat16. The agreement swap, +0.067 [-0.004, 0.134] at 4-bit and never established, is
  -0.005 [-0.019, 0.008]. The -0.078 is not maintained as a 70B finding.
- The central findings hold. Under signed-only agreement the bfloat16 cell establishes no
  difference from either reference, and both contrasts are equivalent to zero within 0.034
  (section 2.5). It shares 18 of 87 union events with the primary annotation.

### 1.4 Predictors and systems not built by the authors (zhvc C4)

Richer predictors on the source cohort. Source: paper Table 22.

| Model (inputs) | AUROC [95% CI] | Compact score minus model |
|---|---|---|
| Compact score (7 inputs) | 0.946 [0.918, 0.969] | reference |
| Random forest (all inputs) | 0.938 [0.908, 0.965] | +0.007 [-0.011, 0.027] |
| Elastic net (all inputs) | 0.938 [0.909, 0.963] | +0.008 [-0.013, 0.034] |
| XGBoost (all inputs) | 0.932 [0.888, 0.967] | +0.014 [-0.013, 0.047] |
| Generalised additive model (7) | 0.925 [0.880, 0.964] | +0.020 [0.000, 0.050] |

The published Bashir feature family rises from .744 to .945 only when composition is added.

**CTO audit (new).** CTO (Gao et al., 2024) labels clinical-trial outcomes with fifteen weak
labelling functions aggregated by a random forest; one function is LLM-derived. That function was
re-derived with the two core annotators on the text the original model saw (148,400 scored
prompts, 74,200 per annotator), and the system was rebuilt on 11,210 distinct trials. Source:
`cto_audit_proof/` (CTO's files unmodified, the labels, one script, expected values).

| Variant | AUROC | Change from refit | Established |
|---|---|---|---|
| CTO shipped probabilities | 0.8573 | n/a | n/a |
| Refit, original 15 functions | 0.8598 | reference | n/a |
| LLM function deleted | 0.8595 | -0.0004 [-0.0031, +0.0023] | no |
| LLM function replaced, Llama-3.1-8B | 0.9093 | +0.0495 [+0.0445, +0.0544] | yes |
| LLM function replaced, Qwen2.5-7B | 0.9058 | +0.0459 [+0.0412, +0.0508] | yes |
| Both added, original kept | 0.9101 | +0.0503 [+0.0452, +0.0553] | yes |
| Control: coverage indicator only | 0.8600 | +0.0001 [-0.0023, +0.0024] | no |
| Control: our labels shuffled | 0.8591 | -0.0008 [-0.0033, +0.0016] | no |

| Target | Registry status alone | Forest, refit | CTO shipped | Re-derived functions (Llama / Qwen) |
|---|---|---|---|---|
| 11,012 expert labels, accuracy | 97.0% (majority class 67.8%) | n/a | n/a | n/a |
| 9,710 expert-labelled trials with predictions, AUROC | 0.9783 | 0.9790 (human set held out) | 0.9850 | n/a |
| Completed trials only (3,432; 285 failures), AUROC | 0.500 (constant) | n/a | 0.655 | 0.764 / 0.757 |

The two core annotators agree at kappa 0.937 on the 4,020 trials where both commit. Scope: model,
prompt and read-out were changed together, and the weak target is not the human subset (80.7%
agreement on 207 shared trials). CTO disclose their two-stage label design; the audit quantifies
it and alleges no defect. The training pool of CTO's shipped probabilities is not visible to us.

---

## 2. Reviewer asRP

### 2.1 Ground truth (asRP W1, Q1)

Events are defined by LLM annotation and have **not** been adjudicated by human coders. The
checks below need no human labels; they bound how far events can be annotation artefacts and do
not validate them. Source: paper Table 6 and Appendix B.

| Check | Result |
|---|---|
| Agreement with an LLM-independent numeric anchor, primary annotator (1,841 records) | 0.807 [0.782, 0.830] |
| Same, second annotator | 0.758 [0.733, 0.782] |
| Opposite-sign record disagreement, discordant events vs joint non-events | 9.7% vs 3.4%; difference +0.063 [0.013, 0.117] |
| Primary events surviving 1 / 3 / 5 record relabellings | 64 / 50 / 46 of 71 |
| Discordant source units differing only in the post-cutoff sign / only in the pre-cutoff sign | 44 / 23 of 69 |

The adjudication instrument (`data_release/06_adjudication_instrument/`) and a six-event,
19-record dossier (`data_release/07_dossier/`) are released for independent coders.

### 2.2 Generalisability (asRP W3)

Source: paper Tables 5, 31 and 32; CTO row new.

| Setting | Domain | How the score is fitted | Units / events | Learned score minus untrained reference |
|---|---|---|---|---|
| Source cohort (development) | diet-health, 165 claims | leave-one-claim-out | 1,012 / 71 | +0.067 [0.017, 0.120] |
| Frozen cohort | 61 claims, 60 of them WCRF cancer relationships | leave-one-claim-out, specifications frozen before annotation | 234 / 50 | -0.006 [-0.043, 0.034] |
| External-1 | activity, sleep, environment, 46 claims | source coefficients, unchanged | 322 / 19 | +0.055 [-0.052, 0.106] |
| External-2 | medication, screening, occupational exposure, 180 claims | source coefficients, unchanged | 1,257 / 95 | -0.074 [-0.123, -0.024] |
| CTO (Gao et al., 2024), *new* | clinical-trial outcomes; another group's system | their 15-function random forest, refitted | 11,012 trials | +0.0007 over registry status alone |
| Bashir et al. (2021) | Cochrane review updates; another group's target | their random forest, refitted | 256 / 43 | +0.042 [-0.048, 0.139] over trials added |

Outside development no setting establishes a gain over an untrained composition reference. The
frozen cohort is the only setting frozen before annotation, and it is almost entirely cancer.

### 2.3 Evaluation design (asRP W4)

Units share claims, so every interval resamples claims (160 clusters, 23 with events). Source:
paper Table 21 and Appendix C.5.

| Contrast | Estimate | Percentile | Wild cluster-t | Jackknife | Clustered ROC | Refitting |
|---|---|---|---|---|---|---|
| Primary, learned minus margin | +0.067 | [.017, .120] | [.018, .115] | [.012, .121] | [.016, .117] | [.014, .117] |
| Primary, learned minus class-blind | +0.051 | [.013, .092] | [.013, .088] | [.010, .091] | [.013, .088] | [.015, .120] |
| Second, learned minus margin | -0.096 | [-.166, -.037] | [-.157, -.034] | [-.163, -.028] | [-.158, -.033] | [-.242, -.039] |
| Intersection, learned minus margin | -0.030 | [-.094, .023] | [-.082, .022] | [-.094, .034] | [-.089, .029] | [-.177, .020] |

| Concentration and accrual | Result |
|---|---|
| Largest AUROC shift when any one positive claim is left out | 0.0052 |
| Units accrued at horizon 3 / 5 / 10 | 756/800, 625/635, 492/492 |
| Largest change in any difference under accrual weighting | 0.014 |

Temporal knowledge contamination is a stated limitation: the design controls document
availability, not the annotator's parametric knowledge, and no pre-cutoff annotator was run.

### 2.4 Parameterisation of the agreement input (asRP W5)

Source: paper Tables 4b and 10, Appendix D; bfloat16 row new.

| Annotator | Events | AUROC, NULL-aware to signed-only | Swap effect |
|---|---|---|---|
| Llama-3.1-8B | 71 | .948 to .925 | -0.023 [-0.050, +0.002] |
| Qwen2.5-7B | 52 | .857 to .952 | +0.095 [+0.046, +0.148] |
| Llama-3.1-70B, 4-bit | 55 | .899 to .965 | +0.067 [-0.004, +0.134] |
| Llama-3.1-70B, bfloat16, *new* | 34 | .958 to .953 | -0.005 [-0.019, +0.008] |
| Qwen2.5-72B, 4-bit | 58 | .941 to .955 | +0.014 [-0.000, +0.031] |

Only the Qwen-7B effect is established. Its 332 NULL-dominant units move from .632 to .878
(+0.246 [0.073, 0.405]), against +0.041 [-0.009, 0.101] on its 670 signed-branch units. Adding a
branch indicator and its interaction to definition A removes the second-annotation deficit
(+0.001 [-0.031, 0.032]) and leaves the primary gain at +0.067.

### 2.5 Equivalence tests (asRP W6)

*New.* Two one-sided tests (alpha 0.05) on 28 contrasts, from the same 2,000 claim resamples as
the published intervals; all published estimates and intervals in the list were reproduced first.
A contrast is equivalent to zero within a bound d when its 90% interval lies inside (-d, +d). No
bound was fixed in advance, so the table gives the smallest such bound (post hoc) and the result
at the fixed bounds 0.02 and 0.05. For reference, the development gain is +0.067 over the margin
rule and +0.038 over the stationary reference. Source: `equivalence_tests/equivalence_tests.json`
(the Bashir row needs the public data file of Bashir et al., passed with `--bashir`).

| Contrast | Units / events | Estimate | 95% interval | 90% interval | Established | Smallest equivalence bound | Equivalent within 0.02 | within 0.05 |
|---|---|---|---|---|---|---|---|---|
| development, primary: signed minus margin | 1012 / 71 | +0.067 | [0.017, 0.120] | [0.024, 0.110] | yes | 0.110 | no | no |
| development, primary: signed minus class-blind | 1012 / 71 | +0.051 | [0.013, 0.092] | [0.019, 0.084] | yes | 0.084 | no | no |
| development, second: signed minus margin | 1012 / 52 | -0.096 | [-0.166, -0.037] | [-0.154, -0.044] | yes | 0.154 | no | no |
| development, second: signed minus class-blind | 1012 / 52 | -0.091 | [-0.164, -0.028] | [-0.150, -0.037] | yes | 0.150 | no | no |
| development, consensus: signed minus margin | 1012 / 55 | -0.030 | [-0.094, 0.023] | [-0.083, 0.015] | no | 0.083 | no | no |
| development, consensus: signed minus class-blind | 1012 / 55 | -0.024 | [-0.089, 0.029] | [-0.077, 0.022] | no | 0.077 | no | no |
| calendar H3: signed minus margin | 756 / 58 | +0.006 | [-0.012, 0.023] | [-0.009, 0.021] | no | 0.021 | no | yes |
| calendar H5: signed minus margin | 625 / 45 | +0.023 | [-0.005, 0.053] | [-0.001, 0.048] | no | 0.048 | no | yes |
| calendar H10: signed minus margin | 492 / 35 | +0.008 | [-0.032, 0.047] | [-0.024, 0.040] | no | 0.040 | no | yes |
| External-1, frozen transfer: signed minus margin | 322 / 19 | +0.055 | [-0.052, 0.106] | [-0.037, 0.099] | no | 0.099 | no | no |
| External-2, frozen transfer: signed minus margin | 1257 / 95 | -0.074 | [-0.123, -0.024] | [-0.115, -0.032] | yes | 0.115 | no | no |
| frozen cohort, primary: signed minus stationary (prespecified primary) | 234 / 50 | -0.040 | [-0.107, 0.013] | [-0.092, 0.006] | no | 0.092 | no | no |
| frozen cohort, primary: signed minus margin (prespecified secondary) | 234 / 50 | -0.006 | [-0.043, 0.034] | [-0.037, 0.027] | no | 0.037 | no | yes |
| frozen cohort, second (own cohort): signed minus stationary | 192 / 28 | -0.095 | [-0.224, 0.007] | [-0.201, -0.006] | no | 0.201 | no | no |
| frozen cohort, second (own cohort): signed minus margin | 192 / 28 | -0.095 | [-0.215, 0.006] | [-0.196, -0.009] | no | 0.196 | no | no |
| frozen cohort, second (primary cohort): signed minus stationary | 234 / 39 | -0.076 | [-0.192, 0.022] | [-0.174, 0.007] | no | 0.174 | no | no |
| frozen cohort, second (primary cohort): signed minus margin | 234 / 39 | -0.050 | [-0.157, 0.045] | [-0.142, 0.032] | no | 0.142 | no | no |
| signed-only, Llama-8B: C minus margin | 1012 / 71 | +0.044 | [-0.005, 0.091] | [0.001, 0.083] | no | 0.083 | no | no |
| signed-only, Llama-8B: C minus stationary | 1012 / 71 | +0.015 | [-0.024, 0.049] | [-0.018, 0.044] | no | 0.044 | no | yes |
| signed-only, Qwen-7B: C minus margin | 1012 / 52 | -0.001 | [-0.028, 0.018] | [-0.022, 0.016] | no | 0.022 | no | yes |
| signed-only, Qwen-7B: C minus stationary | 1012 / 52 | -0.010 | [-0.043, 0.010] | [-0.037, 0.008] | no | 0.037 | no | yes |
| signed-only, Llama-70B (4-bit): C minus margin | 1012 / 55 | +0.002 | [-0.012, 0.017] | [-0.009, 0.014] | no | 0.014 | yes | yes |
| signed-only, Llama-70B (4-bit): C minus stationary | 1012 / 55 | -0.012 | [-0.038, 0.007] | [-0.033, 0.004] | no | 0.033 | no | yes |
| signed-only, Qwen-72B (4-bit): C minus margin | 1012 / 58 | -0.003 | [-0.021, 0.010] | [-0.018, 0.008] | no | 0.018 | yes | yes |
| signed-only, Qwen-72B (4-bit): C minus stationary | 1012 / 58 | -0.010 | [-0.036, 0.007] | [-0.030, 0.005] | no | 0.030 | no | yes |
| signed-only, Llama-70B (bfloat16): C minus margin | 1012 / 34 | +0.011 | [-0.018, 0.038] | [-0.013, 0.034] | no | 0.034 | no | yes |
| signed-only, Llama-70B (bfloat16): C minus stationary | 1012 / 34 | -0.005 | [-0.034, 0.022] | [-0.029, 0.019] | no | 0.029 | no | yes |
| Bashir et al. review updates: random forest minus trials added | 256 / 43 | +0.042 | [-0.048, 0.139] | [-0.035, 0.120] | no | 0.120 | no | no |

Summary:

- Signed-only agreement: nine of ten annotator contrasts are equivalent to zero within 0.05,
  including the new bfloat16 70B cell (0.034). The exception is Llama-8B against the margin rule
  (0.083), whose 90% interval excludes zero on the positive side.
- Frozen cohort: the prespecified secondary is equivalent within 0.037. The prespecified primary is
  not equivalent, but its 90% upper limit (+0.006) excludes, one-sided, any gain over the stationary
  reference above 0.006.
- Calendar evaluation: all three horizons are equivalent within 0.05 (0.021 to 0.048).
- Inconclusive (neither a difference nor equivalence): intersection labels, External-1, Bashir.

### 2.6 Event definition (asRP Q4)

Source: paper Tables 8, 11 and 12, Appendix B.4.

| Endpoint variant | Events (primary / second) | Cross-annotator Jaccard | Primary, learned minus margin |
|---|---|---|---|
| Paper endpoint | 71 / 52 | 0.281 | +0.067 [0.017, 0.120] |
| Sparse-gated units excluded | n/a | n/a | +0.068 [0.017, 0.125] |
| Three-state, deadband 0.05 | 91 / 57 | 0.287 | +0.068 [0.016, 0.131] |
| Three-state, deadband 0.10 | 100 / 64 | 0.291 | +0.057 [0.005, 0.119] |
| Three-state, deadband 0.20 | 117 / 84 | 0.256 | +0.038 [-0.012, 0.092] |
| Strict reversal, deadband 0.05 | 37 / 27 | 0.280 | +0.074 [0.014, 0.143] |
| Systematic reviews and meta-analyses removed | 72 / n/a | n/a | +0.066 [0.014, 0.119] |
| All review articles removed | 70 / n/a | n/a | +0.019 [-0.032, 0.065] |

### 2.7 Conditioning on future evidence (asRP Q5)

A target built from a truncated future window, compared with the full-window target. Source:
`data_release/04_results/results/R6_incomplete_origins.json`.

| Horizon | Truncated window | Units with the same target |
|---|---|---|
| 5 | 4 years | 606 / 618 = 98.1% |
| 10 | 7 years | 481 / 491 = 98.0% |

Frozen-cohort attrition is censused: 84 and 48 origins failed the two record criteria.

### 2.8 Error versus interpretation (asRP Q6)

Stage two of the released instrument codes this split with a seven-cause codebook. Its
distribution requires human adjudication, which has not been carried out; no result is reported.

### 2.9 Human-validated benchmarks (asRP Q7)

Bashir et al. (2021), 256 review pairs, 43 changed conclusions. Source: paper Table 32.

| Model | AUROC | Minus untrained reference |
|---|---|---|
| Trials added at the update (untrained) | 0.694 | reference |
| Random forest, refit of the published specification | 0.736 | +0.042 [-0.048, 0.139] |
| Logistic regression, published | 0.701 | +0.007 [-0.075, 0.096] |
| Decision tree, published | 0.638 | -0.056 [-0.181, 0.070] |

CTO expert-reviewed labels (new): see the tables in section 1.4.

---

## 3. Reviewer AqvV

### 3.1 Hypotheses and results (AqvV C1-C2)

Source: paper Tables 3 and 4, Sections 5.1 to 5.3.

| Hypothesis | What is measured | Result |
|---|---|---|
| H1. Composition predictability: label composition alone ranks events well | AUROC of the margin rule, the stationary reference and the learned score (1,012 units, 71 events) | 0.881, 0.909 and 0.948; learned minus margin +0.067 [0.017, 0.120] |
| H2. Event-set instability: a second annotator defines different events | events shared by the two core annotators | 27 of 96 union events (Jaccard 0.281), with 93.2% unit agreement |
| H3. Representation dependence: the ranking depends on one feature definition | agreement swap with annotation, units and outcomes fixed | Qwen-7B +0.095 [+0.046, +0.148]; Llama-8B -0.023 [-0.050, +0.002] |

### 3.2 Corpus, terminology and a worked example (AqvV C3)

| Term | Definition | Count |
|---|---|---|
| Claim | an exposure (e.g. processed meat) paired with an outcome (e.g. colorectal cancer) | 165 (145 from the systematic candidate list, 20 added by hand) |
| Record | a PubMed document retrieved by the claim's query | |
| Association | one (claim, record) pair, labelled for that claim | 112,453 |
| Unit | one (claim, cutoff year) pair: records before the cutoff are inputs, records after it define the outcome | 1,012 |
| Event | a unit whose pre- and post-cutoff directions have different signs | 71, in 23 claims |

**Worked example (new).** `worked_example/worked_example.py` rebuilds the endpoint for all 1,012
units from `data_release/02_records/cohort_tables/`, checks the published totals (71 primary
events, 52 second, 27 shared; it exits with an error if any differs) and prints one claim.

Claim: exposure *coffee*, outcome *cardiovascular disease*; query
`("coffee"[MeSH Terms] OR "coffee"[Title/Abstract]) AND ("cardiovascular disease"[MeSH Terms] OR "cardiovascular disease"[Title/Abstract])`;
496 associations; seven cutoffs. H/P are HARMFUL/PROTECTIVE records before the cutoff.

| Cutoff | Records before / after | Primary: H/P | p_t to p_t+ | Event | Second: H/P | p_t to p_t+ | Event |
|---|---|---|---|---|---|---|---|
| 1997 | 45 / 451 | 17 / 3 | +0.70 to -0.53 | yes | 11 / 5 | +0.38 to -0.66 | yes |
| 2001 | 60 / 436 | 26 / 6 | +0.63 to -0.57 | yes | 19 / 7 | +0.46 to -0.71 | yes |
| 2006 | 92 / 404 | 39 / 18 | +0.37 to -0.62 | yes | 30 / 18 | +0.25 to -0.77 | yes |
| 2011 | 140 / 356 | 44 / 43 | +0.01 to -0.62 | **yes** | 32 / 40 | -0.11 to -0.77 | **no** |
| 2015 | 202 / 294 | 56 / 83 | -0.19 to -0.63 | no | 36 / 70 | -0.32 to -0.77 | no |
| 2020 | 313 / 183 | 69 / 150 | -0.37 to -0.61 | no | 41 / 129 | -0.52 to -0.73 | no |
| 2025 | 448 / 48 | 89 / 237 | -0.45 to -0.57 | no | 54 / 208 | -0.59 to -0.77 | no |

At 2011 the primary annotator's earlier records split 44 to 43, so the unit is an event by one
record; the second annotator's 32 to 40 makes it a non-event.

### 3.3 Provenance (AqvV C4)

| Check | Result |
|---|---|
| Archived associations returned by re-retrieval on 22 September 2026 | 91.7% |
| Primary result with capped claims removed | +0.070 [0.009, 0.131] |
| Primary result on all claims | +0.067 [0.017, 0.120] |

Timeline of the specification-frozen cohort, from the released execution log and RFC 3161 token
(`data_release/05_freeze_evidence/`):

| UTC | Step |
|---|---|
| 2 Oct 2026, 16:00:38 | SHA-256 manifest of claims, queries, annotation configuration, endpoint, model specifications and analysis plan externally timestamped |
| 2 Oct, 16:07 to 17:22 | retrieval: 91,418 of 91,418 records |
| 2 Oct 17:39 to 3 Oct 06:50 | annotation by both core annotators; audit: no duplicate, missing or extra record |
| 3 Oct, 07:02:14 | manifest digest re-verified against the token; endpoint construction and the analysis run once, as written |

### 3.4 Labelling quality (AqvV C5)

Agreement with the LLM-independent anchor is in section 1.3(b). Between the two core annotators
(source: paper Appendix A.1 and Table 7):

| Agreement between the two core annotators | Result |
|---|---|
| Label kappa, all 112,453 associations | 0.521 |
| Label kappa, 56,423 records both call signed | 0.829 |
| Labels changed by re-running the same annotator | 1.4% (kappa 0.980) |

Disagreement sits on the signed/NULL/UNCLEAR boundary, not on direction.

### 3.5 Appendix font and spacing (AqvV C6)

*New.* The appendix opened with a bare `\small` and five float-spacing reductions (`\textfloatsep`,
`\floatsep`, `\intextsep`, `\abovecaptionskip`, `\belowcaptionskip`), and one table was set in
`\tiny`. The patch removes the six commands and sets that table in `\scriptsize`. The source was
compiled before and after with the same toolchain (tectonic) and the PDFs measured. Source:
`appendix_font_fix/` (`font_fix.patch`, builds, `font_measurements.json`, `margin_check.log`,
`aclpubcheck_fast_checks.log`).

| Measured in the PDF | Submitted PDF | Submitted source, rebuilt | After the fix |
|---|---|---|---|
| Main-text body font | 11.0 pt | 10.9 pt | 10.9 pt |
| Appendix body font | 9.1 pt | 9.0 pt | 10.9 pt |
| Main text ends on | page 8 | page 8 | page 8 |
| Total pages | 24 | 22 | 26 |
| aclpubcheck: page size, page limit, font family | pass | pass | pass |
| aclpubcheck margin rule (camera-ready build) | n/a | pass | pass |

The submitted PDF was built with a different engine, hence 24 pages against 22 for the rebuilt
source; the fix adds four pages to the rebuilt source and none to the main text. aclpubcheck checks
the font family, not its size. Its margin check needs ImageMagick, so its rule was reproduced in
`check_margins.py`.

### 3.6 What each comparison holds fixed (AqvV C7)

| Comparison | Inputs | Outcomes | Coefficients | What it measures |
|---|---|---|---|---|
| Development | Llama-8B | Llama-8B | refitted, held-out claims | ranking on the source cohort |
| Re-annotation | second annotator | second annotator | refitted | how much of the target is annotator-specific |
| Agreement swap | one agreement feature replaced | fixed | refitted for each definition | sensitivity to one representation choice |
| Frozen transfer (External-1, External-2) | Llama-8B | Llama-8B | frozen from source | a new claim population |
| Frozen cohort | both annotators | both annotators | refitted | a frame fixed before outcomes exist |

| Term | Meaning |
|---|---|
| Margin rule | untrained score 1 - abs(p_t) |
| Stationary reference | probability that the future majority differs from the current sign if later labels were exchangeable with earlier ones |
| Agreement input | NULL-aware: the NULL share when at least 0.5, else the signed majority share; signed-only: the signed majority share throughout |
| Compact-7 | the seven hand-fixed variables of the compact score |
| Cross-, both-, neither-on-branch | event/non-event pairs with exactly one, both or neither unit on a NULL-dominant history |

### 3.7 The specification-frozen cohort (AqvV C8)

Source: paper Appendix E.2 and Table 31.

| Specification-frozen cohort | Llama-8B | Qwen-7B |
|---|---|---|
| Claims in the frame | 61 | 61 |
| Units / events | 234 / 50 (in 20 claims) | 192 / 28 |
| Learned minus stationary (prespecified primary) | -0.040 [-0.107, 0.013] | -0.095 [-0.224, 0.007] |
| Learned minus margin (prespecified secondary) | -0.006 [-0.043, 0.034] | n/a |
| Events shared by the two annotators on the 234 units | 21 of 68 (Jaccard 0.309) | |

### 3.8 Exact pair decomposition (AqvV C9)

AUROC is a mean over event/non-event pairs, so a difference between two scores splits exactly by
pair type: each contribution is the pair type's share of all pairs times the AUROC difference
within that type. Source: paper Tables 18 and 28; `data_release/04_results/results/R3_pair_decomposition.json`.

| Pair type | Pairs | Contribution to learned minus margin |
|---|---|---|
| Across majority classes | 34,650 | +0.0609 [0.0159, 0.1081] |
| Within a majority class | 24,633 | +0.0068 [-0.0039, 0.0195] |
| Zero direction | 7,528 | -0.0010 [-0.0032, -0.0001] |
| Total | 66,811 | +0.0667 |

### 3.9 Event overlap and its noise null (AqvV C10)

Events are rare (71 of 1,012 units), so a measure that counts shared non-events is dominated by
them: the core annotations agree on 93.2% of units yet share 27 of 96 union events. Jaccard ignores
joint non-events and needs no reference annotation. It is a monotone transform of F1 with either
annotation as reference, F1 = 2J/(1 + J), and event kappa ranks all six pairs the same way. Source: paper Table 9.

| Pair | Shared / union events | Jaccard | Equivalent F1 | Event kappa |
|---|---|---|---|---|
| Llama-8B -- Qwen-7B (core pair) | 27 / 96 | 0.281 | 0.439 | 0.404 |
| Llama-8B -- Llama-70B | 24 / 102 | 0.235 | 0.381 | 0.341 |
| Llama-8B -- Qwen-72B | 26 / 103 | 0.252 | 0.403 | 0.363 |
| Qwen-7B -- Llama-70B | 26 / 81 | 0.321 | 0.486 | 0.457 |
| Qwen-7B -- Qwen-72B | 28 / 82 | 0.341 | 0.509 | 0.481 |
| Llama-70B -- Qwen-72B | 39 / 74 | 0.527 | 0.690 | 0.672 |

70B and 72B rows are 4-bit; bfloat16 overlaps are in section 1.3(c).

**Noise null (new).** Each annotator's labels are redrawn at the measured record-level
disagreement rate with the disagreements placed at random, and the events rebuilt: null mean and
95% range over 1,000 draws from each side of the pair, pooled, with the confusion matrix estimated
globally (N1), per claim (N2) or per claim and decade (N3). Source: `noise_null_calibration/`.

Paper endpoint:

| Pair | Observed | N1 global | N2 per claim | N3 per claim and decade |
|---|---|---|---|---|
| Core pair | 0.281 (27/96) | 0.300 [0.130, 0.522] | 0.217 [0.125, 0.315] | 0.258 [0.178, 0.351] |
| 70B-class pair | 0.527 (39/74) | 0.326 [0.227, 0.443] | 0.399 [0.308, 0.493] | 0.424 [0.316, 0.526] |
| 8B to 70B, fixed prompt | 0.250 (26/104) | 0.244 [0.116, 0.423] | 0.225 [0.141, 0.326] | 0.228 [0.153, 0.314] |
| Family swap, fixed prompt | 0.312 (30/96) | 0.296 [0.086, 0.569] | 0.247 [0.161, 0.326] | 0.279 [0.213, 0.352] |
| Batch-size rerun (positive control) | 0.846 (66/78) | 0.876 [0.800, 0.936] | 0.840 [0.747, 0.920] | 0.853 [0.775, 0.921] |

Sparse-gated histories treated as non-events:

| Pair | Observed | N1 global | N2 per claim | N3 per claim and decade |
|---|---|---|---|---|
| Core pair | 0.258 (23/89) | 0.292 [0.134, 0.500] | 0.209 [0.127, 0.307] | 0.250 [0.174, 0.351] |
| 70B-class pair | 0.392 (20/51) | 0.292 [0.200, 0.396] | 0.351 [0.250, 0.455] | 0.324 [0.215, 0.426] |
| 8B to 70B, fixed prompt | 0.224 (19/85) | 0.211 [0.115, 0.349] | 0.197 [0.121, 0.284] | 0.194 [0.129, 0.265] |
| Family swap, fixed prompt | 0.261 (23/88) | 0.284 [0.087, 0.533] | 0.215 [0.143, 0.286] | 0.246 [0.179, 0.312] |
| Batch-size rerun (positive control) | 0.857 (66/77) | 0.878 [0.800, 0.936] | 0.844 [0.747, 0.920] | 0.860 [0.783, 0.932] |

The core pair, the family swap and the 8B-to-70B change fall inside every null range on both
endpoints; the positive control falls inside its ranges. The 70B-class pair lies just above all
three ranges on the paper endpoint (by 0.001 for N3) and inside them once sparse-gated histories
are non-events. Falling inside a range shows consistency with random placement, not that the
disagreement is unstructured.

---

## 4. What changes in the paper

Presentation changes listed in the responses for the camera-ready:

- A model-inventory table and the external corpora's domains in the main text.
- The bfloat16 cell as the Llama-70B annotator, replacing the -0.078 in Section 5.3 and the
  Llama-70B row of Table 4b; the 4-bit cell kept as a precision control.
- A new appendix for the CTO audit.
- Definitions and a notation table before Section 3; the worked example (section 3.2) in the main
  text; an annotator-quality table after the annotation description; a provenance timeline figure.
- "Specification-frozen cohort" used throughout; each results subsection opening with what was
  measured and what it means for an LLM-annotation pipeline.
- The appendix at the main-text size (`appendix_font_fix/font_fix.patch`).

## 5. Not done

Human adjudication of events (asRP W1, Q1, Q6) has not been carried out; no human or expert
labels exist for the source benchmark. It remains a stated limitation, and the instrument and
dossier are released for it.
