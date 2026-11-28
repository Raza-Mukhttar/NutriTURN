# Worked example: one claim from records to event labels

Added after submission, in response to review (AqvV C3). Nothing else in the repository depends on
this folder.

`worked_example.py` rebuilds the endpoint of the paper (Section 3, eq. 1) for all 1,012 units from
the released labels, PMIDs and years in `data_release/02_records/cohort_tables/`
(`claims.csv`, `associations.csv`, `units.csv`). It checks the totals against the paper (1,012
units; 71 primary events, 52 second, 27 shared), exits with an error if any differs, and prints
every unit of one claim under both core annotators.

## Run

    python rebuttal/worked_example/worked_example.py [DATA_RELEASE] [CLAIM_ID]   # CPU, seconds; needs pandas

Defaults: the repository's `data_release/` (or `NUTRITURN_DATA_RELEASE`) and the claim
`coffee_cvd`. No abstract text and no `data/` folder are needed.

## Output

`worked_example.json` (the totals check and every unit of the claim, with both annotators' counts, directions and event
labels) and `worked_example_log.txt` (the printed table of the reference run).

For coffee and cardiovascular disease (496 associations, seven cutoffs), the earlier records lean
harmful and the later ones protective, so the 1997, 2001 and 2006 cutoffs are events under both
annotators. At 2011 the primary annotator's earlier signed records split 44 HARMFUL to 43
PROTECTIVE (direction +0.011, later -0.616), so the unit is an event by one record; the second
annotator's 32 to 40 (-0.111) makes it a non-event.
