# Environment

## Interpreter

Python **3.10.12** (CPython), Linux 5.15.0, glibc 2.35, x86-64.
`pip install -r requirements.txt`.

## Two deterministic settings that matter

**`PYTHONHASHSEED=0` is required**, not optional. `scripts/experiments/null_controls.py`
derives each draw's seed from Python's `hash()` of the control's name:

```python
jobs = [(kind, 10_000 * (1 + hash(kind) % 17) + i) for i in range(draws)]
```

Python salts string hashes per process unless `PYTHONHASHSEED` is fixed, so a run without
it produces different (still valid, but different) draws. Under `PYTHONHASHSEED=0` the
Null D seeds are 130000–130999 and the shipped `results/E2_draws_primary_D.npz`
reproduces bit for bit.

**`PYTHONDONTWRITEBYTECODE=1`** is used throughout so no `__pycache__` is written into the
release.

Every script in this release is run as:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=<n> python scripts/<...>.py
```

## Randomness

Randomness enters only through seeded resamplers and seeded learners. The shared seed is
`SEED = 3` (`scripts/lib/nfx_common.py`), and the claim-clustered percentile bootstrap uses
2,000 resamples at that seed. The wild cluster bootstrap uses 9,999 Webb six-point draws at
the same seed. No result depends on wall-clock time or on process scheduling.

`OMP_NUM_THREADS` affects speed only, never values: all reported quantities are computed in
deterministic NumPy/scikit-learn code paths.

## Hardware

The reported analyses are CPU-only and were produced on a 64-core x86-64 host. The heaviest
analysis step is the calendar refitting bootstrap
(`scripts/experiments/calendar_refit_bootstrap.py`), about 3 minutes across 40 worker
processes; single-core it is roughly 25 minutes.

GPUs are needed **only** to re-run annotation (`scripts/pipeline/`) or the text-reading
baselines. The shipped annotations were produced on NVIDIA GPUs in bfloat16 with
Llama-3.1-8B-Instruct (primary) and Qwen2.5-7B-Instruct (second pipeline); the prompted
forecaster runs additionally used 32B and 72B models. Model weights are **not** redistributed
here; `scripts/pipeline/` downloads them from the Hugging Face Hub at the pinned revisions
recorded in `results/issue6/` and in the per-run metadata of `data/llm_runs/`.

## What cannot be re-executed from this release

- **Abstract text is not redistributed.** `data/protocol_snapshot/records/` carries
  identifiers, years, titles and publication types only. Text-dependent stages
  (re-annotation, the embedding baseline, the prompted forecasters, effect-estimate parsing)
  need the abstracts back; `scripts/pipeline/fetch_abstracts.py` re-fetches them from PubMed
  by the released PMIDs. The parsed effect estimates are shipped as a derived cache
  (`data/protocol_snapshot/effects_cache/`) so the numeric-anchor analyses reproduce without
  the abstracts.
- **The external corpora ship as unit-level label counts without identifiers**, so
  text-reading methods cannot be evaluated on them. This is what restricts text-reading
  methods to checks 1–4 in the paper's protocol.
- **The original retrieval timestamp was not recorded**, and the candidate-list provenance is
  not reconstructible. See the paper's snapshot appendix; the harvest window is bounded, not
  exact.
