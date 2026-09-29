"""Issue 7: dataset card and manifest for the stored (derived) snapshot and the new timestamped re-retrieval.
Records: the query file hash, per-claim record/direction file hashes, counts by year, the 145-claim curated list
(fixed artifact), the external cohort files (non-regenerable), and the re-retrieval comparison. Writes
results/issue7/{manifest.json, DATASET_CARD.md, version_comparison.csv}."""

# ---- NutriTURN portable path bootstrap (added by the release build) ----
import os as _os, sys as _sys
_NT = _os.path.abspath(__file__)
for _ in range(5):
    _NT = _os.path.dirname(_NT)
    if _os.path.isdir(_os.path.join(_NT, 'data')) and _os.path.isdir(_os.path.join(_NT, 'scripts')):
        break
for _p in ('lib', 'experiments', 'pipeline', 'figures', ''):
    _q = _os.path.join(_NT, 'scripts', _p)
    if _q not in _sys.path:
        _sys.path.insert(0, _q)
# ------------------------------------------------------------------------
import os, sys, json, hashlib, csv, glob, collections, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_common as pc
SNAP = _os.path.join(_NT, 'data', 'protocol_snapshot')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''): h.update(ch)
    return h.hexdigest()


def main():
    snaps = sorted(glob.glob(os.path.join(pc.RES, 'snapshot_*'))); new = snaps[-1]; summ = json.load(open(os.path.join(new, 'summary.json')))
    comp = list(csv.DictReader(open(os.path.join(new, 'comparison.csv'))))
    Q = json.load(open(f'{SNAP}/claim_queries.json')); man = {'stored_snapshot': {'path': 'astra_reviews/source_snapshot/data', 'claim_queries_sha256': sha(f'{SNAP}/claim_queries.json'), 'claims': len(Q), 'files': {}}, 'reretrieval': summ}
    years = collections.Counter(); nrec = 0; ndir = 0; per = []
    for cid in Q:
        rp = f'{SNAP}/records/{cid}.json'; dp = f'{SNAP}/directions/{cid}.json'; R = json.load(open(rp))['records']; D = json.load(open(dp))['directions']
        for r in R:
            if r.get('year'): years[int(r['year'])] += 1
        nrec += len(R); ndir += len(D); man['stored_snapshot']['files'][cid] = {'records': len(R), 'directions': len(D), 'records_sha256': sha(rp), 'directions_sha256': sha(dp)}
    man['stored_snapshot'].update(records_total=nrec, directions_total=ndir, records_by_year={str(k): years[k] for k in sorted(years)}, last_complete_year=pc.LAST_COMPLETE_YEAR,
                                  retrieval_timestamp='not recorded in the stored files (derived snapshot; see DATASET_CARD.md)')
    ext = {}
    for k in (1, 2):
        p = getattr(pc.nm, f'EXTERNAL{k}'); ext[f'External-{k}'] = {'path': p, 'sha256': sha(p) if os.path.exists(p) else None, 'regenerable': False, 'reason': 'query strings and record files were not shipped; unit-level features only'}
    man['external'] = ext
    man['curated_claim_list'] = {'n_claims': len(Q), 'status': 'fixed curated artifact (v1); not a random sample of nutrition claims', 'sha256': man['stored_snapshot']['claim_queries_sha256']}
    tot = summ; jac = sorted(float(r['jaccard']) for r in comp)
    man['version_comparison'] = {'stored_ids': tot['stored_ids_total'], 'new_ids': tot['returned_ids_total'], 'in_both': tot['in_both_total'], 'new_only': tot['new_only_total'], 'stored_only': tot['stored_only_total'],
                                 'recall_of_stored_in_new': round(tot['in_both_total'] / tot['stored_ids_total'], 4), 'median_jaccard': jac[len(jac) // 2], 'min_jaccard': jac[0], 'max_jaccard': jac[-1],
                                 'claims_with_stored_only_gt_10pct': sum(1 for r in comp if int(r['stored_only']) > 0.1 * int(r['stored_ids']))}
    pc.save_json(man, 'issue7/manifest.json'); pc.save_csv(comp, 'issue7/version_comparison.csv')
    card = f"""# Dataset card: PubMed evidence-stream benchmark (derived snapshot v1) and re-retrieval {summ['snapshot_timestamp_utc'][:10]}

## Stored snapshot (the one every experiment uses)
* Files: `claim_queries.json` (145 curated claims; 165 query entries incl. duplicates/variants), `records/<claim>.json`, `directions/<claim>.json`.
* Records: {nrec:,} PubMed records with an automatic direction annotation ({ndir:,} annotations); years {min(years)}-{max(years)}; {years.get(2026, 0):,} records dated 2026 (partial year), last complete year {pc.LAST_COMPLETE_YEAR}.
* Provenance: the retrieval timestamp, per-query PubMed counts and query translations were NOT recorded when the snapshot was built. The stored files are therefore a **non-regenerable derived snapshot**: the exact PMID sets cannot be reproduced from PubMed, but every downstream number is reproducible from the files, whose SHA-256 hashes are listed in `manifest.json`.
* Annotation: Llama-3.1-8B-Instruct first-token read-out (deployed extractor); an independent Qwen2.5-7B-Instruct re-annotation with complete verbalizer scoring is stored under `results/reextraction/` (issue 6).
* Curated claim list: fixed artifact, hand-assembled diet/exposure-outcome pairs; not a random sample of the nutrition literature.

## Re-retrieval at one UTC timestamp ({summ['snapshot_timestamp_utc']})
* Every stored query string re-run with ESearch (retmax 200,000) at one timestamp; per query: PubMed count, query translation, full PMID list, SHA-256 of the sorted PMID list (`snapshot_*/queries.jsonl`).
* Comparison with the stored snapshot: {tot['in_both_total']:,} of {tot['stored_ids_total']:,} stored PMIDs are returned again ({100*tot['in_both_total']/tot['stored_ids_total']:.1f}%); {tot['new_only_total']:,} PMIDs are new (mostly records added since the original retrieval and translation drift); {tot['stored_only_total']:,} stored PMIDs are no longer returned (median per-claim Jaccard {jac[len(jac)//2]:.3f}, range {jac[0]:.3f}-{jac[-1]:.3f}).
* Abstracts were not re-fetched and the benchmark was not rebuilt on the new PMID sets; the re-retrieval documents drift and supplies a reproducible version 2 identifier set.

## External cohorts
* External-1 ({'322 units'}) and External-2 ({'1,257 units'}) were shipped as unit-level feature files without query strings or records; they cannot be re-retrieved (non-regenerable). Selection and freeze are documented in the paper's appendix.
"""
    open(os.path.join(pc.RES, 'issue7', 'DATASET_CARD.md'), 'w').write(card); print(card)


if __name__ == '__main__':
    main()
