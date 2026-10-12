# config

The frozen settings of the study. Nothing here is an input you are meant to tune: these values
define the reported numbers, and changing one changes the paper.

| File | Contents |
|---|---|
| `claim_queries.json` | the 165 claims with the fixed PubMed retrieval query of each. This is the protocol input that `data_preparation/01_retrieve_pubmed.py` replays, which is why it ships as configuration rather than as data |
| `protocol.json` | the endpoint contract, the sparse gate, the cutoff grid rule, the seeds and bootstrap counts, the agreement definitions, the Compact-7 variable list |
| `annotation_models.json` | every annotation cell: model, Hugging Face revision, prompt, read-out, batch size, and the script that produces it |
| `environment.json` | interpreter, the two environment variables determinism depends on, and why |
| `external_sources.json` | External-1/2 provenance, and an explicit split between what is and is not regenerable |

**`protocol.json` and `annotation_models.json` are documentation, not the source of truth.** The
executable definitions live in `audit_pipeline/lib/nfx_common.py` (endpoint, gate, bootstrap) and
`data_preprocessing/lib/nutrimature_features.py` (the 55 variables). If the two ever disagree, the
code is right and this directory is stale — the scripts never read these files.
