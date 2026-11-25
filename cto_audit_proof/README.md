# CTO audit: one script, every number

The audit of the paper applied to a system we did not build. CTO (Gao et al., 2024,
arXiv:2406.10292) labels clinical-trial outcomes with fifteen weak labelling functions aggregated
by a random forest; one function, `gpt`, is LLM-derived (GPT-3.5-turbo over the two most similar
PubMed abstracts per trial). We re-derived that function with Llama-3.1-8B-Instruct and
Qwen2.5-7B-Instruct on the abstracts CTO released, rebuilt the system, and scored it.

## Run

    pip install numpy scipy scikit-learn
    python run_cto_audit.py            # about 3 minutes on CPU, no GPU

The script recomputes every reported quantity, writes `output/cto_audit_numbers.json` and
`output/cto_audit_log.txt`, compares the result with `expected_numbers.json`, and exits 0 only if
all 136 numbers match. Options: `--jobs N` (forest threads), `--tol` (default 1e-6).

Reference environment: Python 3.10.12, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2. Counts,
rates and kappa are exact in any environment. Forest AUROCs depend on the scikit-learn version;
with another version use `--tol 3e-3`, which is the seed-to-seed spread of the forest.

## What is here

| path | content |
|---|---|
| `run_cto_audit.py` | the whole analysis |
| `expected_numbers.json` | the reference values, each with its provenance |
| `output/` | numbers and log from the reference run, with SHA-256 of every input |
| `inputs/cto_released/` | CTO's files, unmodified: `phase{1,2,3}_CTO_rf.csv`, `human_labels_2020_2024.csv` (Hugging Face `chufangao/CTO`, MIT licence) |
| `inputs/annotations/{llama8b,qwen7b}/` | our labels: 74,200 scored prompts per annotator, one JSON line per (trial, abstract slot) with the label and its probability vector |
| `annotation_scripts/` | the code that produced the labels (`prep_items.py`, `annotate_cto.py`) |


## Re-running the annotation

Not needed for the check above. `annotation_scripts/` holds the two scripts exactly as they were
run, so they read and write a folder named `data/` beside them: `prep_items.py` expects CTO's
`pubmed_gpt_outcomes` file at `data/pubmed_gpt_outcomes.csv` and `annotate_cto.py` writes
`data/annot_<model>/`. The label files they produced are stored here under
`inputs/annotations/<model>/`. Settings: bfloat16, full-string read-out, 3,000-character text
budget, 1,600-token limit with left truncation, one forward pass, no sampling.
