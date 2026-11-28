# Equivalence tests on the reference contrasts

Added after submission, in response to review (asRP W6). Nothing else in the repository depends on
this folder.

For each learned-minus-reference AUROC contrast, `equivalence_tests.py` redraws the same 2,000
claim resamples as the published intervals (same seed and cell definitions), checks the 95%
interval against the published value, and reports the 90% interval. That is the two one-sided
tests (TOST) interval at alpha 0.05: a contrast is equivalent to zero within +/- d exactly when its
90% interval lies inside (-d, +d). No bound was fixed before the data were seen, so the smallest
such d is the primary quantity; verdicts at 0.02 and 0.05 are a reading aid.

## Run

    FP70_CELL=rebuttal/full_precision_70b/output/cell_llama70_bf16.jsonl \
    python rebuttal/equivalence_tests/equivalence_tests.py . [--bashir /path/to/extracted_info.txt]

Pass the repository root (the folder holding `.nutriturn_root`); CPU, about three minutes. The
script needs the repository's `audit_pipeline/lib` on the path (it adds it itself from the root)
and the record and annotation tables in `data/` (`units.csv`, the associations, the labels).
`data/` is not part of the repository: build it with the repository Quickstart or link a local
snapshot as `data/`. The outputs are included, so the results can be read without a run.
`FP70_CELL` adds the two full-precision Llama-70B rows. The Bashir et al. row needs their public
`extracted_info.txt` (github.com/dsurian/rules_cochranereviews), which is not redistributed here;
without `--bashir` that row is skipped.

## Output

`equivalence_tests.json` and `equivalence_tests_log.txt`: 28 contrasts (development, calendar,
frozen transfer, the frozen cohort under both annotations, signed-only agreement for four
annotators plus the full-precision Llama-70B cell, and Bashir et al.). A rerun from a clean copy
of the repository reproduced the 27 rows that need no third-party file byte for byte.

`first_run/` (5 contrasts) and `without_bf16/` (26 contrasts) are earlier runs of the same script,
superseded by the files above and kept for the record.
