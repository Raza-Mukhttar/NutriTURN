"""Derived artifact for the release: the parsed effect estimates per (claim, record), so that the
origin-defined cohort analysis runs without redistributing abstract text. Output mirrors
calendar_forward_eval.text_feats_cache(cid): a list of [year, [[y, se], ...]] per claim."""

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
import os, sys, json
sys.path.insert(0, _os.path.join(_NT, 'scripts', 'lib'))
import astra_common as ac
nm, nf = ac.nm, ac.nf
OUT = sys.argv[1] if len(sys.argv) > 1 else _os.path.join(_NT, 'data', 'protocol_snapshot', 'effects_cache')
os.makedirs(OUT, exist_ok=True)
Q = json.load(open(nm.QUERIES)); n_eff = 0
for i, cid in enumerate(Q):
    recs = json.load(open(f'{nm.RECORDS}/{cid}.json'))['records']
    Dj = {d['pmid'] for d in json.load(open(f'{nm.DIRECTIONS}/{cid}.json'))['directions']}
    cache = [[int(r['year']), [[float(y), float(se)] for y, se in nf.effects(r)]] for r in recs if r['pmid'] in Dj and r.get('year')]
    n_eff += sum(len(e) for _, e in cache)
    json.dump(cache, open(f'{OUT}/{cid}.json', 'w'))
    if i % 40 == 0: print(f'  {i}/{len(Q)}', flush=True)
print('DONE claims', len(Q), 'effect estimates', n_eff, 'bytes', sum(os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT)))
