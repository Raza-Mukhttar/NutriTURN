"""Independent setting at scale, step 1 -- assemble the External-1/2 re-annotation item set.

The experiment: re-annotate the two external (non-diet) corpora with the SECOND pipeline
(Qwen2.5-7B-Instruct) and report (a) event overlap against the shipped primary annotation and
(b) the signed-minus-margin contrast under each annotation. This tests the annotator-dependence
claim OUTSIDE diet, at 1,582 units across 226 claims rather than the 256 pairs of Appendix G.

The chosen option is the second of the two offered. The first -- the Alharbi and Stevenson
3,326 review-update pairs -- remains blocked: that dataset has no data or code link on the ACL
Anthology page or in the paper, as recorded in plan_v2/COMPLETED_EXPERIMENTS.md.

Record text is NOT in the release (which ships counts only for the external corpora), so it is
read from the original retrieval inputs:
  External-1  data/external_benchmark/abstracts            46 of 46 benchmark claims present
  External-2  $NUTRITURN_EXTERNAL2_DIR/abstracts            180 of 180 present
Both carry title + abstract per record, so nothing is refetched from PubMed.
"""

# ---- NutriTURN repository path bootstrap ----
import os as _os, sys as _sys
_NT = _os.path.abspath(__file__)
for _ in range(6):
    _NT = _os.path.dirname(_NT)
    if _os.path.isfile(_os.path.join(_NT, '.nutriturn_root')):
        break
else:
    raise SystemExit('cannot locate the repository root (.nutriturn_root marker) from ' + __file__)
for _p in ('audit_pipeline/lib', 'data_preprocessing/lib', 'audit_experiments',
           'audit_experiments/figures', 'data_annotation', 'data_preparation', ''):
    _q = _os.path.join(_NT, *_p.split('/')) if _p else _NT
    if _q not in _sys.path:
        _sys.path.insert(0, _q)
# ---------------------------------------------

import csv, json, os, sys

# External-1/2 corpora are not redistributed. Point these at a local copy, or set
# NUTRITURN_EXTERNAL1_DIR / NUTRITURN_EXTERNAL2_DIR. Each root holds abstracts/, directions/
# and claim_queries.json. See config/external_sources.json.
EXT_ROOT = {1: os.environ.get('NUTRITURN_EXTERNAL1_DIR', os.path.join(_NT, 'data', 'external', 'external1')),
            2: os.environ.get('NUTRITURN_EXTERNAL2_DIR', os.path.join(_NT, 'data', 'external', 'external2'))}
SRC = {k: (os.path.join(v, 'abstracts'), os.path.join(v, 'claim_queries.json'))
       for k, v in EXT_ROOT.items()}


def main():
    items, meta = [], {}
    for which, (adir, qpath) in SRC.items():
        bench = list(csv.DictReader(open(os.path.join(
            NT, 'data', 'protocol_snapshot', 'external', f'external{which}_benchmark.csv'))))
        claims = sorted({r['claim_id'] for r in bench})
        Q = json.load(open(qpath))
        n_rec = n_ab = n_miss = 0
        for c in claims:
            p = os.path.join(adir, f'{c}.json')
            if not os.path.exists(p):
                n_miss += 1
                continue
            q = Q.get(c) or {}
            exp, outc = q.get('exposure', ''), q.get('outcome', '')
            if not (exp and outc):
                n_miss += 1
                continue
            for r in json.load(open(p)).get('records') or []:
                if not r.get('year'):
                    continue
                items.append(dict(domain=which, claim_id=c, pmid=str(r['pmid']),
                                  year=int(r['year']), exposure=exp, outcome=outc,
                                  title=r.get('title') or '', abstract=r.get('abstract') or ''))
                n_rec += 1
                n_ab += bool((r.get('abstract') or '').strip())
        meta[f'external{which}'] = dict(claims=len(claims), claims_missing=n_miss,
                                        units=len(bench), records=n_rec, with_abstract=n_ab)
        print(f'External-{which}: {len(claims)} claims ({n_miss} unusable), {len(bench):,} units, '
              f'{n_rec:,} records, {n_ab:,} with abstract ({100*n_ab/max(n_rec,1):.1f}%)')
    items.sort(key=lambda r: (r['domain'], r['claim_id'], r['pmid']))
    with open(os.path.join(HERE, 'data', 'items.jsonl'), 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    json.dump(meta, open(os.path.join(HERE, 'data', 'scope.json'), 'w'), indent=1)
    print(f'\ntotal records to annotate: {len(items):,}')
    print(f'written {os.path.join(HERE, "data", "items.jsonl")}')


if __name__ == '__main__':
    main()
