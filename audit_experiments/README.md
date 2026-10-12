# audit_experiments

One script per experiment in the paper, named for what it does. 51 experiment scripts, plus figure
and table generation.

```bash
export PYTHONHASHSEED=0
bash audit_experiments/run_all_experiments.sh      # everything, in dependency order, a few hours
python audit_experiments/stationary_baselines.py   # or any single experiment
```

Every script prints the path of each file it writes as its last line, so the mapping from a paper
number to the script that produced it can always be re-derived by running it. The top-level
`README.md` carries the full table of paper object → script → output.

| Subdirectory | Contents |
|---|---|
| `figures/` | `regenerate_main_figures.py` (Figures 4–6) and `src/nutrifig.tex` (Figure 1, TikZ) |
| `reporting/` | the LaTeX table builders, the results report, the paper-number collector, the file manifest |
| `external/` | External-1/2 scoring, the v2 scorer and the external anchor |

## Things worth knowing before you run these

**`PYTHONHASHSEED=0` is required.** `null_controls.py` derives each draw's seed from Python's
`hash()` of the control's name. Python salts string hashes per process unless the variable is fixed,
so without it you get different — still valid, but different — draws, and the shipped
`E2_draws_primary_D.npz` will not reproduce.

**Scripts import each other.** `full_metric_suite.py` imports `two_variable_transfer`,
`clustered_roc_all_cells.py` imports `wild_cluster_bootstrap`, and several others do the same. The
filenames are therefore part of the interface: renaming a script here breaks its importers. They all
sit in one directory on `sys.path` for exactly this reason.

**The order in `run_all_experiments.sh` is a dependency order, not a preference.** Section 1 writes
the core scores that sections 2–4 read. Running a single late experiment in a fresh checkout will
fail with a missing-input error, which is the correct behaviour — build the core first with
`audit_pipeline/run_audit.py`.

**The slowest stage** is `calendar_refit_bootstrap.py`: about 3 minutes across 40 worker processes,
roughly 25 minutes single-core. `null_controls.py` and the stationary-null trio are the next
heaviest.

**`regenerate_main_figures.py` refuses to save a broken figure.** It asserts at run time that no
text, legend or mark overlaps, and fails rather than writing a figure that does.
