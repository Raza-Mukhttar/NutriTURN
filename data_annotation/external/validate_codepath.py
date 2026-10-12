"""CHECK C -- does this code path reproduce the ARCHIVED second-pipeline labels?

The external corpora ship as unit-level files and were re-annotated here with a freshly written
loader, so the low label-level kappa could in principle be a fault in this run rather than a
property of the domain. The decisive test is to point the SAME code path at SOURCE associations,
whose second-pipeline labels are archived, and check that it reproduces them.

Sample construction: source associations whose title+abstract is available from the ORIGINAL
extraction input (the abstract cache, NUTRITURN_ABSTRACT_CACHE), i.e. byte-identical to the text the shipped
pipeline saw, so a mismatch cannot be blamed on refetched text. Stratified by the archived label so
every class is represented.

A high reproduction rate licenses the external numbers. A low one contaminates them.
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

import argparse, glob, json, os, random, sys
import numpy as np

SNAP = os.path.join(_NT, 'data', 'protocol_snapshot')
# The deployed pipeline read abstracts from the extraction input, not from the protocol
# snapshot (which carries no abstract text). Point this at a local copy, or set
# NUTRITURN_ABSTRACT_CACHE.
CACHE = os.environ.get('NUTRITURN_ABSTRACT_CACHE', os.path.join(_NT, 'data', 'abstract_cache'))
import nfx_common as N
LABELS = ['PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR']


def build(n_per_class=750, seed=3):
    sec = N.label_column('second')
    Q = json.load(open(os.path.join(SNAP, 'claim_queries.json')))
    cache = {}
    for f in glob.glob(CACHE + '/*.json'):
        d = json.load(open(f))
        cache[d['claim_id']] = {str(r['pmid']): r for r in d['records']}
    pool = {l: [] for l in LABELS}
    for (c, p), lab in sec.items():
        r = cache.get(c, {}).get(str(p))
        if not r or not (r.get('abstract') or '').strip():
            continue
        q = Q.get(c) or {}
        if not (q.get('exposure') and q.get('outcome')):
            continue
        if lab in pool:
            pool[lab].append(dict(claim_id=c, pmid=str(p), archived=lab,
                                  exposure=q['exposure'], outcome=q['outcome'],
                                  title=r.get('title') or '', abstract=r['abstract']))
    rng = random.Random(seed)
    items = []
    for l in LABELS:
        rng.shuffle(pool[l])
        items += pool[l][:n_per_class]
        print(f'  archived {l:11s}: pool {len(pool[l]):6,d}  sampled {min(len(pool[l]), n_per_class)}')
    rng.shuffle(items)
    p = os.path.join(HERE, 'validation', 'items.jsonl')
    with open(p, 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    print(f'  total sampled {len(items):,} -> {p}')
    return items


def compare():
    arch = {}
    for l in open(os.path.join(HERE, 'validation', 'items.jsonl')):
        if l.strip():
            d = json.loads(l); arch[(d['claim_id'], d['pmid'])] = d['archived']
    got = {}
    for f in sorted(glob.glob(os.path.join(HERE, 'validation', 'annot', '*.jsonl'))):
        for l in open(f):
            if l.strip():
                d = json.loads(l); got[(d['claim_id'], d['pmid'])] = d['label_full']
    keys = sorted(k for k in arch if k in got)
    a = np.array([arch[k] for k in keys]); b = np.array([got[k] for k in keys])
    agree = float((a == b).mean())
    cats = sorted(set(a.tolist()) | set(b.tolist()))
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    kap = (agree - pe) / (1 - pe) if pe < 1 else float('nan')
    print(f'\nreproduction of archived second labels on {len(keys):,} source associations')
    print(f'  exact agreement {agree:.4f}   kappa {kap:.4f}')
    print(f'\n  {"archived/rerun":16s}' + ''.join(f'{x[:10]:>12s}' for x in LABELS) + f'{"recall":>9s}')
    for x in LABELS:
        row = [int(((a == x) & (b == y)).sum()) for y in LABELS]
        tot = sum(row)
        rec = row[LABELS.index(x)] / tot if tot else float('nan')
        print(f'  {x:16s}' + ''.join(f'{v:12,d}' for v in row) + f'{rec:9.3f}')
    json.dump(dict(n=len(keys), agreement=agree, kappa=float(kap),
                   confusion={x: {y: int(((a == x) & (b == y)).sum()) for y in LABELS}
                              for x in LABELS}),
              open(os.path.join(HERE, 'results', 'codepath_validation.json'), 'w'), indent=1)
    print('\nwritten results/codepath_validation.json')
    return agree


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['build', 'compare'])
    ap.add_argument('--n', type=int, default=750)
    a = ap.parse_args()
    if a.cmd == 'build':
        build(a.n)
    else:
        compare()
