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
├── config/                 frozen protocol constants, model/read-out index, the claim queries
├── requirement.txt         environment
├── data_preparation/       stage 1 — PubMed retrieval
├── data_annotation/        stage 2 — every annotation cell, one per (model, prompt, read-out)
├── data_preprocessing/     stage 3 — the 55 pre-cutoff variables, caches, anchors, system scores
├── audit_pipeline/         the frozen audit method, its metrics, and the driver that runs it
└── audit_experiments/      one script per experiment in the paper
```

---

## Quickstart

```bash
pip install -r requirement.txt
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1      

bash data_preparation/run_data_preparation.sh          
bash data_annotation/run_data_annotation.sh            
bash data_preprocessing/run_data_preprocessing.sh      

python audit_pipeline/run_audit.py                     
bash audit_experiments/run_all_experiments.sh          
```


## The three stages, and what each one fixes

### Stage 1 — `data_preparation/`

| Script | What it does |
|---|---|
| `01_retrieve_pubmed.py` | ESearch + EFetch for each of the 165 claim queries in `config/claim_queries.json` |
| `02_fetch_abstracts.py` | restores abstract text for the **archived** identifiers |
| `03_reretrieve_audit.py` | a dated re-retrieval: one UTC timestamp, the returned PMID list and its SHA-256 per query, plus a version-to-version comparison |
| `external/` | the External-1/2 input manifest and the primary-reconstruction probe |


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



### Stage 3 — `data_preprocessing/`

| Script | What it does |
|---|---|
| `01_build_benchmark.py` | the claim × cutoff benchmark: the **55 pre-cutoff variables** at up to seven cutoffs per claim, plus the post-cutoff evaluation fields |
| `02_build_effects_cache.py` | parsed effect estimates per (claim, record), so the anchor analyses run without abstracts |
| `03_build_numeric_anchor.py` | the LLM-independent numeric anchor: effect measures read next to each 95% interval, attributed to a claim only when exposure and outcome both occur in the interval's sentence or the one before |
| `04_build_system_scores.py` | Compact-7 and its leave-one-claim-out predictions |
| `05_build_record_embeddings.py` | PubMedBERT record embeddings for the text-reading baseline (GPU) |
| `lib/nutrimature_features.py` | the frozen feature builder — the authoritative definition of all 55 variables |


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

