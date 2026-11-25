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

## Result

Observed event Jaccard against the null mean and 95% range, on the endpoint of the paper:

| pair | observed | N1 global | N2 per claim | N3 per claim and decade |
|---|---|---|---|---|
| headline pair | 0.281 (27/96) | 0.300 [0.130, 0.522] | 0.217 [0.125, 0.315] | 0.258 [0.178, 0.351] |
| 70B-class pair | 0.527 (39/74) | 0.326 [0.227, 0.443] | 0.399 [0.308, 0.493] | 0.424 [0.316, 0.526] |
| 8B to 70B, fixed prompt | 0.250 (26/104) | 0.244 [0.116, 0.423] | 0.225 [0.141, 0.326] | 0.228 [0.153, 0.314] |
| family swap, fixed prompt | 0.312 (30/96) | 0.296 [0.086, 0.569] | 0.247 [0.161, 0.326] | 0.279 [0.213, 0.352] |
| batch-size rerun (positive control) | 0.846 (66/78) | 0.876 [0.800, 0.936] | 0.840 [0.747, 0.920] | 0.853 [0.775, 0.921] |

With a sparse-gated history treated as a non-event:

| pair | observed | N1 global | N2 per claim | N3 per claim and decade |
|---|---|---|---|---|
| headline pair | 0.258 (23/89) | 0.292 [0.134, 0.500] | 0.209 [0.127, 0.307] | 0.250 [0.174, 0.351] |
| 70B-class pair | 0.392 (20/51) | 0.292 [0.200, 0.396] | 0.351 [0.250, 0.455] | 0.324 [0.215, 0.426] |
| 8B to 70B, fixed prompt | 0.224 (19/85) | 0.211 [0.115, 0.349] | 0.197 [0.121, 0.284] | 0.194 [0.129, 0.265] |
| family swap, fixed prompt | 0.261 (23/88) | 0.284 [0.087, 0.533] | 0.215 [0.143, 0.286] | 0.246 [0.179, 0.312] |
| batch-size rerun (positive control) | 0.857 (66/77) | 0.878 [0.800, 0.936] | 0.844 [0.747, 0.920] | 0.860 [0.783, 0.932] |

## Reading

- The headline overlap, the family swap and the 8B-to-70B change all fall inside every null range
  on both endpoints: record-level disagreement at the measured rate, placed at random, is enough
  to produce event sets that overlap this little.
- The positive control falls inside every range, so the method does not manufacture low overlap.
- The 70B-class pair is the exception on the paper's endpoint: 0.527 is above all three ranges
  (by 0.001 for N3). It is inside all three once sparse-gated histories are non-events.
- The ranges are wide, N1 especially. Falling inside a range shows consistency with random
  placement; it does not show the disagreement is unstructured.
