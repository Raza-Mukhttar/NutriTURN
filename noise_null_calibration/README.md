# Noise-null calibration of event overlap

Added after submission, in response to review. Nothing else in the repository depends on this
folder, and it changes no existing file.

**Question.** Two annotations of the same records share few events (Jaccard 0.281 for the
headline pair). Is that less overlap than ordinary record-level disagreement would produce once
it passes through a thresholded endpoint?

**Method.** For a pair (X, Y), every label of X is redrawn from the measured P(Y label | X label),
the endpoint is rebuilt, and the event overlap with X is computed; the same is done from Y, and
the two sets of draws are pooled. The synthetic annotator has the same label shift and the same
record-level agreement with X as Y has, but its disagreements are placed at random. Three nulls
estimate the confusion matrix globally (N1), per claim (N2) and per claim and decade (N3); a
stratum with fewer than 40 records falls back to the global matrix. A batch-size rerun of the
primary annotator is the positive control: its overlap should sit inside the null range.

## Run

    python noise_null_calibration/noise_null_overlap.py     # from the repository root, CPU only

1,000 draws per direction (set `REPS` to change), seed fixed, draws independent of
`PYTHONHASHSEED`. Writes `noise_null_overlap.json`; `noise_null_log.txt` is the log of the
reference run. The script needs numpy only and reads only `data_release/` (labels, identifiers and
years): `02_records/cohort_tables/{associations,units}.csv` and
`03_annotations/{controlled_cells,controlled_cells_raw,large_cells}/`. If `data_release/` is not
beside this folder, set `NUTRITURN_DATA_RELEASE` to its path. A run takes about eight minutes.




