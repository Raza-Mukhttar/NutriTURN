# data_preparation — stage 1 of 3

Builds the corpus. **The only stage that needs the network**, and the only one whose output depends
on when you run it.

```bash
bash data_preparation/run_data_preparation.sh
```

| Script | Needs | Produces |
|---|---|---|
| `01_retrieve_pubmed.py` | network | `data/protocol_snapshot/records/<claim>.json` from the queries in `config/claim_queries.json` |
| `02_fetch_abstracts.py` | network | abstract text restored into those record files, for the **archived** identifiers |
| `03_reretrieve_audit.py` | network | `results/snapshot_<UTCdate>/` — one timestamp, PMID list and SHA-256 per query, with a version-to-version comparison |
| `external/build_input_manifest.py` | — | the External-1/2 input manifest |
| `external/probe_primary_reconstruction.py` | — | the probe behind the κ agreement between re-executed and archived external labels |

## Retrieval

For each claim the query is sent to NCBI ESearch with `usehistory=y`; the whole server-side result
set is then fetched with EFetch in blocks of 500 and parsed with the frozen parser. There is no
record cap. A claim counts as fully retrieved when the unique PMIDs reach PubMed's hit count. A new
retrieval is **merged** with the claim's existing history, so records PubMed no longer returns are
kept and a history never loses a record. Rate limiting is 0.36 s between requests (NCBI allows three
per second without an API key).

## Two things that will surprise you

**PubMed grows.** A retrieval today returns more records than the September 2026 snapshot the paper
reports. This is not a bug and it is why the paper's numbers are tied to the archived identifier
lists, not to a live query. Reproducing the paper exactly means restoring abstracts for the archived
PMIDs (`02_fetch_abstracts.py`); extending the benchmark means retrieving from the queries
(`01_retrieve_pubmed.py`), after which the annotation and feature stages must be re-run.

**External-1/2 retrieval cannot be re-run at all.** Their query strings were never released.
`03_reretrieve_audit.py` marks them non-regenerable by design rather than silently skipping them.
Their units are rebuilt in `data_preprocessing/external/` from the released unit files, and their
re-annotation runs on the retained retrieval texts. See `config/external_sources.json`.
