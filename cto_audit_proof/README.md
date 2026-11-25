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

## Design

- **Unit.** The phase files repeat trials: 24,478 labelled rows cover 11,210 distinct trials. One
  row per trial is kept (lowest phase); intervals resample trials. Keeping the highest phase
  instead changes the main contrast from +0.0495 to +0.0493.
- **Trial label.** Up to two abstracts per trial; the first that commits to SUCCESS or FAILURE
  gives the label, otherwise the function abstains.
- **Forest.** 300 trees, minimum leaf 2, seed 3, out-of-fold over 5 stratified folds. For the
  human subset the forest is fitted on the weak target with human-labelled trials held out.
- **Intervals.** Percentile, 2,000 resamples of trials, seed 3. "Established" means the interval
  excludes zero.

## Results

Weak target (`hint_train`), 11,210 trials:

| variant | AUROC | vs R0 |
|---|---|---|
| CTO shipped `pred_proba` | 0.8573 | |
| R0 original fifteen functions | 0.8598 | reference |
| R1 drop `gpt` | 0.8595 | -0.0004 [-0.0031, +0.0023] |
| R2 `gpt` replaced by Llama-3.1-8B | 0.9093 | +0.0495 [+0.0445, +0.0544] |
| R2 `gpt` replaced by Qwen2.5-7B | 0.9058 | +0.0459 [+0.0412, +0.0508] |
| R3 `gpt` replaced by consensus | 0.9079 | +0.0481 [+0.0433, +0.0530] |
| R4 both added, `gpt` kept | 0.9101 | +0.0503 [+0.0452, +0.0553] |
| L1 coverage indicator only | 0.8600 | +0.0001 [-0.0023, +0.0024] |
| L2 ours under CTO's abstention pattern | 0.9093 | +0.0495 [+0.0445, +0.0544] |
| L3 ours shuffled among committed trials | 0.8591 | -0.0008 [-0.0033, +0.0016] |

L2 equals R2 by construction: wherever our function commits, CTO's does too. Forest seed spread
is 0.0023 (R0) and 0.0015 (R2). Our two annotators agree at kappa 0.9365 on 4,020 co-committed
trials.

Human-reviewed subset, 11,012 trials: "success if and only if COMPLETED" is correct on 10,687
(97.0%; majority baseline 67.8%). On the 9,710 of these trials present in the phase files:
status alone 0.9783, the held-out forest 0.9790 (+0.0007), CTO's shipped probabilities 0.9850.

Completed trials only (3,432 trials, 285 failures), where status is constant: ours 0.7636 and
0.7573; best released function (`results_reported`) 0.6950; CTO shipped 0.6553; `gpt` 0.4263.

## Scope

- CTO document the two-stage construction of the human subset themselves. This audit measures it
  and alleges no defect.
- The weak target is not the human subset: the two share 207 trials and agree on 80.7%.
- Model, prompt and read-out changed together relative to CTO's GPT-3.5 set-up.
- Intervals cluster at the trial only, not by drug, sponsor or indication.
- R0 is our out-of-fold refit, not CTO's training protocol; it lands within 0.0025 of their
  shipped probabilities on the weak target.
- The completed-only ranking rests on 285 failures.

## Re-running the annotation

Not needed for the check above. `annotation_scripts/` holds the two scripts exactly as they were
run, so they read and write a folder named `data/` beside them: `prep_items.py` expects CTO's
`pubmed_gpt_outcomes` file at `data/pubmed_gpt_outcomes.csv` and `annotate_cto.py` writes
`data/annot_<model>/`. The label files they produced are stored here under
`inputs/annotations/<model>/`. Settings: bfloat16, full-string read-out, 3,000-character text
budget, 1,600-token limit with left truncation, one forward pass, no sampling.
