"""NutriMATURE system, step 3: build the uncapped claim x cutoff benchmark from records and direction labels.

For each claim (data/claim_queries.json order) the frozen feature builder computes, at up to seven historical cutoffs,
the 55 pre-cutoff variables from records published before the cutoff, the operational maturity state, and the
post-cutoff evaluation fields (later categorical direction change = l2_sign_change). The systematic-only benchmark drops
the hand-added claims.
Outputs: data/benchmark/uncapped_benchmark.csv, data/benchmark/uncapped_benchmark_systematic_only.csv,
         data/benchmark/benchmark_meta.json, tables/03_benchmark_summary.{csv,json}
The script reports whether the rebuilt files are byte-identical to the files they replace.
"""

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
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import nutrimature as nm, nutrimature_features as nf

Q = json.load(open(nm.QUERIES)); hand = set(json.load(open(nm.HAND_ADDED))['hand_added_claim_ids'])
before = {p: nm.sha(p) for p in (nm.BENCHMARK, nm.SYSTEMATIC) if os.path.exists(p)}
rows = []
for cid in Q:
    rows += nf.build_claim_rows(cid, json.load(open(f'{nm.RECORDS}/{cid}.json'))['records'],
                                json.load(open(f'{nm.DIRECTIONS}/{cid}.json'))['directions'])
rows = nf.labelled(rows)
nf.write_corpus(rows, nm.BENCHMARK)
nf.write_corpus([r for r in rows if r['claim_id'] not in hand], nm.SYSTEMATIC)
P = [r for r in rows if r['n_post_t'] >= nm.POST20]; ev = [r for r in P if r[nm.TARGET] == 1]
meta = {'units': len(rows), 'claims': len({r['claim_id'] for r in rows}), 'prospective_units': len(P),
        'prospective_claims': len({r['claim_id'] for r in P}), 'events': len(ev), 'positive_claims': len({r['claim_id'] for r in ev}),
        'claims_with_full_history_after_800_cap': sum(1 for c in Q if Q[c]['history_retrieved_in_full_after_800_cap']),
        'systematic_only_units': sum(1 for r in rows if r['claim_id'] not in hand), 'hand_added_claims': len(hand & {r['claim_id'] for r in rows}),
        'records': sum(len(json.load(open(f'{nm.RECORDS}/{c}.json'))['records']) for c in Q),
        'sha256': {os.path.basename(p): nm.sha(p) for p in (nm.BENCHMARK, nm.SYSTEMATIC)}}
json.dump(meta, open(f'{nm.DATA}/benchmark/benchmark_meta.json', 'w'), indent=1)
st = collections.Counter(r['maturity_state'] for r in rows); stp = collections.Counter(r['maturity_state'] for r in P)
summary = [{'quantity': k, 'value': v} for k, v in (('claims', meta['claims']), ('claim x cutoff units', meta['units']),
           ('records (full histories)', meta['records']), ('prospective units (n_post_t >= 20)', meta['prospective_units']),
           ('prospective claims', meta['prospective_claims']), ('later direction changes (events)', meta['events']),
           ('claims with an event', meta['positive_claims']), ('event prevalence', round(meta['events'] / meta['prospective_units'], 4)),
           ('systematic-only units', meta['systematic_only_units']), ('hand-added claims', meta['hand_added_claims']),
           *[(f'units in state {s}', st[s]) for s in ('stable', 'still_forming', 'unstable', 'unassessable')],
           *[(f'prospective units in state {s}', stp[s]) for s in ('stable', 'still_forming', 'unstable', 'unassessable')])]
nm.save_table(summary, '03_benchmark_summary', meta)
for p, h in before.items():
    print(f'{os.path.basename(p)}: {"byte-identical to the previous file" if nm.sha(p) == h else "CHANGED (new records or labels)"}', flush=True)
print(json.dumps({k: v for k, v in meta.items() if k != 'sha256'}), flush=True)
