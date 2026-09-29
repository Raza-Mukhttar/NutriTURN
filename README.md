# NutriTURN

Benchmark, data and code for *High Scores, Fragile Targets: Auditing Change in LLM-Annotated
Diet-Health Evidence*. Everything needed to reproduce every number, table and figure in
`paper/NutriTURN.pdf` is in this directory.

**Start here:** `python scripts/verify_release.py` — rebuilds the cohorts from the shipped
labels and checks them against the counts and AUROCs the paper states. It should print
`ALL CHECKS PASS` in about a minute.

```
NutriTURN release verification
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

Then see **`REPRODUCE.md`** for which script produces which table or figure, and
**`ENVIRONMENT.md`** for the interpreter, the pinned versions and the two environment
variables that determinism depends on.

## Layout

```
NutriTURN/
├── README.md              this file
├── REPRODUCE.md           paper object -> script -> output file
├── ENVIRONMENT.md         interpreter, determinism, what cannot be re-executed
├── requirements.txt       pinned versions
├── MANIFEST.json          sha256 of every shipped file, with sizes
├── paper/                 NutriTURN.tex and the compiled PDF
├── data/                  see "Data" below
├── scripts/
│   ├── lib/               shared libraries: feature builder, LOCO fitting, metrics, bootstrap
│   ├── pipeline/          NutriTURN construction: retrieval, annotation, benchmark build
│   ├── experiments/       one script per experiment, named for what it does
│   ├── figures/           figure generators
│   ├── verify_release.py  the rebuild check above
│   ├── run_all.sh         launcher: regenerates every table and figure
│   ├── build_latex_tables_main.py, build_latex_tables_audit.py
│   └── build_results_report.py, collect_paper_numbers.py, build_file_manifest.py
├── results/               every result table as CSV and JSON
├── figures/               the four figures the paper includes, plus sources and previews
├── tables/                generated LaTeX tables
└── logs/                  run logs from the production runs
```

## Data

| Path | What it is |
|---|---|
| `data/protocol_snapshot/` | the frozen protocol snapshot the paper is built on |
| ├ `claim_queries.json` | the 165 claims with the fixed PubMed query of each |
| ├ `records/` | retrieved identifiers, years, titles and publication types, one file per claim |
| ├ `directions/` | primary extractor direction labels with confidences |
| ├ `benchmark/` | the claim-cutoff benchmark and the 145-claim subset |
| ├ `external/` | the two external benchmarks as claim-cutoff units |
| ├ `effects_cache/` | parsed effect estimates, derived, so the anchor analyses run without abstracts |
| └ `analyses_v2/tables/` | out-of-fold predictions used by the metric suite |
| `data/units.csv` | the 1,012-unit evaluation cohort |
| `data/associations.csv` | the 112,453 claim-record associations with both pipelines' labels |
| `data/claims.csv` | claim-level metadata |
| `data/reretrieved_years.json` | dated re-retrieval manifest behind the retrieval audit |
| `data/llm_runs/controlled_cells/` | stored option probabilities for the model x prompt x read-out design |
| `data/llm_runs/forecasters/` | stored outputs of the prompted LLM forecasters |
| `data/derived/` | derived per-analysis tables (anchor, agreement, cumulative targets) |
| `data/record_embeddings.npz` | abstract embeddings for the text-reading baseline |
| `results/reextraction*/` | the second pipeline's per-record label store, where the code reads it |

**Abstract text is not redistributed.** Records carry identifiers, years, titles and
publication types. Everything reported in the paper reproduces without the abstracts; see
`ENVIRONMENT.md` for the stages that do need them and how to fetch them back.

## The three reproducibility levels

The paper distinguishes three claims, and this release supports them to different degrees.

1. **Reproduce the stored metrics** — fully, from the per-unit files in `results/`.
2. **Reconstruct from the released labels** — fully: counts, pooled directions, agreement
   under every definition, every endpoint, the stationary probabilities and the null
   controls. This is what `scripts/verify_release.py` demonstrates.
3. **Re-execute text-dependent methods** — possible for the source and calendar cells after
   re-fetching abstracts by the released PMIDs; **not** possible for the external corpora,
   which ship as unit-level label counts without identifiers.

## Running everything

```bash
pip install -r requirements.txt
PYTHONHASHSEED=0 bash scripts/run_all.sh
```

`run_all.sh` runs the analysis layer in dependency order and writes into `results/`,
`tables/` and `figures/`. It does not re-run annotation: that needs GPUs and the abstracts,
and is in `scripts/pipeline/`.

## Naming

Every script is named for what it does. Nothing in this directory is unrelated to
reproducing the paper: analyses whose outputs the paper does not use (the descriptive
state rule, the forward-complete cohort, the explanatory source split) are not shipped.
