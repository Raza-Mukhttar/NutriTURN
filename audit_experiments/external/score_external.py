"""Independent setting at scale, step 3 -- event overlap and signed-minus-margin on External-1/2.

VALIDATION FIRST. The shipped benchmark columns `pooled_direction`, `agreement` and
`post_pooled_direction` are recomputed from the PRIMARY direction files using the frozen contract.
If the rebuild does not reproduce the shipped columns, the rebuild is wrong and no result is
reported -- that check is what licenses applying the identical code to the second pipeline's labels.

Frozen contract, as in nfx_common:
    p_t = (h - l) / (h + l) when h + l >= SPARSE_GATE (8), else 0
    a_t = 0 if h + l < 8 else (null_share if null_share >= 0.5 else (1 + |p_t|) / 2)
    y_sub = 1[ sgn0(post_pooled_direction) != sgn0(pooled_direction) ]
UNCLEAR (code 99) is excluded from every count. Future-eligibility is n_post >= 20, the external
criterion, applied to the SHIPPED counts so the evaluation population does not move with the
annotation.

Reported per domain and pooled:
  * event counts under each annotation, their intersection and union (the annotator-dependence
    claim: how much of the target is shared?)
  * Cohen's kappa and raw agreement on the unit-level endpoint
  * AUROC(signed two-variable) - AUROC(margin), refit leave-one-claim-out under EACH annotation,
    with a claim-clustered percentile bootstrap, 2,000 draws, seed 3
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

import csv, glob, json, os, sys
import numpy as np

import nfx_common as N

# External-1/2 corpora are not redistributed. Point these at a local copy, or set
# NUTRITURN_EXTERNAL1_DIR / NUTRITURN_EXTERNAL2_DIR. Each root holds abstracts/, directions/
# and claim_queries.json. See config/external_sources.json.
EXT_ROOT = {1: os.environ.get('NUTRITURN_EXTERNAL1_DIR', os.path.join(_NT, 'data', 'external', 'external1')),
            2: os.environ.get('NUTRITURN_EXTERNAL2_DIR', os.path.join(_NT, 'data', 'external', 'external2'))}
DIRS = {k: os.path.join(v, 'directions') for k, v in EXT_ROOT.items()}
ABS = {k: os.path.join(v, 'abstracts') for k, v in EXT_ROOT.items()}
GATE, MIN_POST = N.SPARSE_GATE, 20
SEED, BOOT = N.SEED, N.BOOT


def primary_streams(which):
    S = {}
    for f in sorted(glob.glob(DIRS[which] + '/*.json')):
        c = os.path.basename(f)[:-5]
        Dj = {d['pmid']: d['direction'] for d in json.load(open(f))['directions']}
        recs = json.load(open(os.path.join(ABS[which], f'{c}.json'))).get('records') or []
        sel = [r for r in recs if str(r['pmid']) in Dj and r.get('year')]
        S[c] = (np.array([int(r['year']) for r in sel]),
                np.array([N.CODE.get(Dj[str(r['pmid'])], 99) for r in sel]))
    return S


def second_streams():
    per = {}
    for f in sorted(glob.glob(os.path.join(HERE, 'data', 'annot_second', '*.jsonl'))):
        for l in open(f):
            if l.strip():
                d = json.loads(l)
                per.setdefault(d['claim_id'], []).append((int(d['year']),
                                                          N.CODE.get(d['label_full'], 99)))
    return {c: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for c, v in per.items()}


def unit(yrs, codes, cut):
    pre = (yrs < cut) & (codes != 99)
    post = (yrs >= cut) & (codes != 99)
    h = int((codes[pre] == 1).sum()); l = int((codes[pre] == -1).sum())
    n0 = int((codes[pre] == 0).sum()); n = h + l
    p = (h - l) / n if n >= GATE else 0.0
    res = h + l + n0
    ns = n0 / res if res else 0.0
    a = 0.0 if n < GATE else (ns if ns >= 0.5 else (1 + abs(p)) / 2)
    ph = int((codes[post] == 1).sum()); pl = int((codes[post] == -1).sum()); m = ph + pl
    pp = (ph - pl) / m if m >= GATE else 0.0
    # accrual_5y must be computed from THIS corpus's stream; nfx_common.build_cohort_accrual5
    # resolves claim ids against the source diet streams and cannot be used here
    rate5 = int(((yrs >= cut - 5) & (yrs < cut) & (codes != 99) & (codes != 0)).sum()) / 5.0
    return dict(h=h, l=l, n0=n0, n_signed=n, p=p, s=N.sgn0(p), a=a, null_share=ns,
                accrual_5y=rate5, cutoff=cut,
                n_pre=int(((yrs < cut) & (codes != 99)).sum()),
                n_post=int(post.sum()), m_signed=m, p_post=pp,
                y=int(N.sgn0(pp) != N.sgn0(p)))


def contrast(rows):
    idx, s2, y, g = N.loco_two_var(rows, features=('p', 'a'))
    if len(np.unique(y)) < 2:
        return None
    m = N.margin_score(rows, idx)
    d, lo, hi, _ = N.paired_ci(y, s2, m, g, n=BOOT, seed=SEED)
    return dict(units=int(len(y)), events=int(y.sum()),
                claims=int(len(np.unique(g))),
                auroc=float(N.auroc(y, s2)), auroc_margin=float(N.auroc(y, m)),
                delta=float(d), ci=[float(lo), float(hi)])


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = sum(float((a == k).mean()) * float((b == k).mean()) for k in (0, 1))
    return (po - pe) / (1 - pe) if pe < 1 else float('nan')


def main():
    SEC = second_streams()
    out = {}
    pooled = {'primary': [], 'second': []}
    for which in (1, 2):
        bench = list(csv.DictReader(open(os.path.join(
            NT, 'data', 'protocol_snapshot', 'external', f'external{which}_benchmark.csv'))))
        PRI = primary_streams(which)
        rebuilt_ok = mism = 0
        rp, rs, keep = [], [], []
        for b in bench:
            c, cut = b['claim_id'], int(float(b['cutoff_year']))
            if c not in PRI or c not in SEC:
                continue
            up = unit(*PRI[c], cut)
            # validate the rebuild against the shipped primary columns
            try:
                shipped_p = float(b['pooled_direction']); shipped_a = float(b['agreement'])
                if abs(up['p'] - shipped_p) < 1e-6 and abs(up['a'] - shipped_a) < 1e-6:
                    rebuilt_ok += 1
                else:
                    mism += 1
            except (ValueError, KeyError):
                mism += 1
            if float(b['n_post_t']) < MIN_POST:
                continue
            us = unit(*SEC[c], cut)
            up['claim_id'] = c; us['claim_id'] = c
            rp.append(up); rs.append(us); keep.append(b)
        n = len(rp)
        print(f'\n=== External-{which} ===')
        print(f'  rebuild validation against shipped primary columns: '
              f'{rebuilt_ok} match, {mism} mismatch of {rebuilt_ok+mism}')
        print(f'  future-eligible units (shipped n_post_t >= {MIN_POST}): {n:,}')
        yp = np.array([r['y'] for r in rp]); ys = np.array([r['y'] for r in rs])
        inter = int(((yp == 1) & (ys == 1)).sum()); union = int(((yp == 1) | (ys == 1)).sum())
        print(f'  events  primary {int(yp.sum()):4d}   second {int(ys.sum()):4d}   '
              f'intersection {inter:4d}   union {union:4d}   '
              f'Jaccard {inter/max(union,1):.3f}')
        print(f'  unit-level endpoint agreement {float((yp==ys).mean()):.4f}   '
              f"Cohen's kappa {kappa(yp, ys):.4f}")
        res = {'n_units': n, 'rebuild_match': rebuilt_ok, 'rebuild_mismatch': mism,
               'events_primary': int(yp.sum()), 'events_second': int(ys.sum()),
               'events_intersection': inter, 'events_union': union,
               'jaccard': inter / max(union, 1),
               'endpoint_agreement': float((yp == ys).mean()), 'kappa': float(kappa(yp, ys))}
        print(f"\n  {'annotation':12s} {'units':>6s} {'ev':>4s} {'cl':>4s} {'AUROC':>7s} "
              f"{'margin':>7s} {'delta':>8s} {'[95% CI]':>20s}")
        for nm, rows in (('primary', rp), ('second', rs)):
            cres = contrast([dict(r) for r in rows])
            res[f'contrast_{nm}'] = cres
            if cres:
                print(f"  {nm:12s} {cres['units']:6d} {cres['events']:4d} {cres['claims']:4d} "
                      f"{cres['auroc']:7.4f} {cres['auroc_margin']:7.4f} {cres['delta']:+8.4f} "
                      f"[{cres['ci'][0]:+.4f}, {cres['ci'][1]:+.4f}]")
            else:
                print(f'  {nm:12s} endpoint single-class -> not estimable')
        out[f'external{which}'] = res
        pooled['primary'] += rp; pooled['second'] += rs
    print(f'\n=== POOLED External-1 + External-2 ===')
    yp = np.array([r['y'] for r in pooled['primary']]); ys = np.array([r['y'] for r in pooled['second']])
    inter = int(((yp == 1) & (ys == 1)).sum()); union = int(((yp == 1) | (ys == 1)).sum())
    print(f'  units {len(yp):,}   events primary {int(yp.sum())}  second {int(ys.sum())}  '
          f'intersection {inter}  union {union}  Jaccard {inter/max(union,1):.3f}')
    print(f"  endpoint agreement {float((yp==ys).mean()):.4f}   kappa {kappa(yp,ys):.4f}")
    print(f"\n  {'annotation':12s} {'units':>6s} {'ev':>4s} {'cl':>4s} {'AUROC':>7s} "
          f"{'margin':>7s} {'delta':>8s} {'[95% CI]':>20s}")
    pres = {}
    for nm in ('primary', 'second'):
        cres = contrast([dict(r) for r in pooled[nm]])
        pres[nm] = cres
        if cres:
            print(f"  {nm:12s} {cres['units']:6d} {cres['events']:4d} {cres['claims']:4d} "
                  f"{cres['auroc']:7.4f} {cres['auroc_margin']:7.4f} {cres['delta']:+8.4f} "
                  f"[{cres['ci'][0]:+.4f}, {cres['ci'][1]:+.4f}]")
    out['pooled'] = dict(units=len(yp), events_primary=int(yp.sum()), events_second=int(ys.sum()),
                         events_intersection=inter, events_union=union,
                         jaccard=inter / max(union, 1),
                         endpoint_agreement=float((yp == ys).mean()),
                         kappa=float(kappa(yp, ys)), **{f'contrast_{k}': v for k, v in pres.items()})
    json.dump(out, open(os.path.join(HERE, 'results', 'external_reannotation_results.json'), 'w'),
              indent=1, default=float)
    print('\nwritten results/external_reannotation_results.json')


if __name__ == '__main__':
    main()
