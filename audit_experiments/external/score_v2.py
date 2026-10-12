"""Independent setting at scale, step 4 -- the reviewer's checks.

1  RESOLVED WITHOUT COMPUTATION. Table 24's External-2 row (.823, -.074 [-.123, -.024]) is the
   FROZEN score: the caption says "external rows apply the source-fitted model unchanged". The
   paper also reports the REFITTED variant in the main text (l.477) and in app:crossdomain
   (l.1933): "refitted inside External-2 ... -0.017 [-0.047, 0.009] (External-1: +0.054
   [-0.068, 0.116])". Our primary rows reproduce those refitted numbers exactly
   (External-2 -0.0169 [-0.0474, +0.0087]; External-1 +0.0540 [-0.0685, +0.1164]), because
   loco_two_var refits leave-one-claim-out inside the cohort. There is no contradiction: the two
   numbers are the frozen and refitted rows the paper already distinguishes.

2  Events by branch (sparse / NULL-dominant / signed), then Jaccard and delta on COMMON SUPPORT
   (>= 8 signed pre-cutoff records under BOTH pipelines).
3  External-1 under the second annotation: counts only, 3 events is degenerate.
4  Label-level agreement and kappa on the external associations, against 0.52 on the source.
5  Positive claims per cell, since the bootstrap is clustered by claim.
6  Definition C (branch-free, c = (1+|p|)/2 with the sparse gate) under the second annotation.
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

import csv, glob, json, os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NT = os.path.join(os.path.dirname(HERE), 'NutriTURN')
for p in ('lib', 'experiments'):
    sys.path.insert(0, os.path.join(NT, 'scripts', p))
import nfx_common as N
sys.path.insert(0, os.path.join(HERE, 'scripts'))
from score_external import primary_streams, second_streams, unit, kappa, MIN_POST

GATE, SEED, BOOT = N.SPARSE_GATE, N.SEED, N.BOOT


def branch(r):
    if r['n_signed'] < GATE:
        return 'sparse'
    return 'null_dominant' if r['null_share'] >= 0.5 else 'signed'


def prep(rows):
    for r in rows:
        n = r['n_signed']
        r['c'] = (1 + abs(r['p'])) / 2 if n >= GATE else 0.0
        r['q'] = r['null_share'] if n >= GATE else 0.0
        r['branch'] = branch(r)
    return rows


def contrast(rows, feats=('p', 'a'), mask=None):
    rows = [dict(r) for r in rows]
    m = None if mask is None else np.asarray(mask)
    idx, s2, y, g = N.loco_two_var(rows, features=feats, subset=m)
    if len(np.unique(y)) < 2:
        return dict(units=int(len(y)), events=int(y.sum()), estimable=False)
    mg = np.array([1 - abs(rows[i]['p']) for i in idx])
    d, lo, hi, _ = N.paired_ci(y, s2, mg, g, n=BOOT, seed=SEED)
    return dict(units=int(len(y)), events=int(y.sum()),
                claims=int(len(np.unique(g))),
                positive_claims=int(len(np.unique(g[y == 1]))),
                auroc=float(N.auroc(y, s2)), auroc_margin=float(N.auroc(y, mg)),
                delta=float(d), ci=[float(lo), float(hi)], estimable=True)


def main():
    SEC = second_streams()
    out = {}
    for which in (1, 2):
        bench = list(csv.DictReader(open(os.path.join(
            NT, 'data', 'protocol_snapshot', 'external', f'external{which}_benchmark.csv'))))
        PRI = primary_streams(which)
        rp, rs = [], []
        for b in bench:
            c, cut = b['claim_id'], int(float(b['cutoff_year']))
            if c not in PRI or c not in SEC or float(b['n_post_t']) < MIN_POST:
                continue
            up = unit(*PRI[c], cut); us = unit(*SEC[c], cut)
            up['claim_id'] = c; us['claim_id'] = c
            rp.append(up); rs.append(us)
        prep(rp); prep(rs)
        yp = np.array([r['y'] for r in rp]); ys = np.array([r['y'] for r in rs])
        print(f'\n{"="*86}\nEXTERNAL-{which}   {len(rp):,} future-eligible units\n{"="*86}')

        # --- CHECK 2a: events by branch
        print('\n  CHECK 2a  events by branch')
        print(f'    {"branch":16s} {"primary units":>14s} {"ev":>5s} {"rate":>7s} '
              f'{"second units":>14s} {"ev":>5s} {"rate":>7s}')
        bres = {}
        for bn in ('sparse', 'null_dominant', 'signed'):
            kp = np.array([r['branch'] == bn for r in rp])
            ks = np.array([r['branch'] == bn for r in rs])
            ep = int(yp[kp].sum()) if kp.any() else 0
            es = int(ys[ks].sum()) if ks.any() else 0
            print(f'    {bn:16s} {int(kp.sum()):14d} {ep:5d} '
                  f'{(ep/max(kp.sum(),1)):7.3f} {int(ks.sum()):14d} {es:5d} '
                  f'{(es/max(ks.sum(),1)):7.3f}')
            bres[bn] = dict(primary_units=int(kp.sum()), primary_events=ep,
                            second_units=int(ks.sum()), second_events=es)
        print(f'    {"TOTAL":16s} {len(rp):14d} {int(yp.sum()):5d} {yp.mean():7.3f} '
              f'{len(rs):14d} {int(ys.sum()):5d} {ys.mean():7.3f}')
        sg = bres['sparse']
        print(f"\n    -> of the second pipeline's {int(ys.sum())} events, "
              f"{sg['second_events']} ({100*sg['second_events']/max(int(ys.sum()),1):.1f}%) are "
              f"SPARSE-GATE units (p_t forced to 0)")
        print(f"       primary: {sg['primary_events']} of {int(yp.sum())} "
              f"({100*sg['primary_events']/max(int(yp.sum()),1):.1f}%)")

        # --- CHECK 2b: common support
        cs = np.array([(a['n_signed'] >= GATE) and (b_['n_signed'] >= GATE)
                       for a, b_ in zip(rp, rs)])
        ypc, ysc = yp[cs], ys[cs]
        inter = int(((ypc == 1) & (ysc == 1)).sum()); union = int(((ypc == 1) | (ysc == 1)).sum())
        print(f'\n  CHECK 2b  COMMON SUPPORT (>= {GATE} signed pre-cutoff records under BOTH)')
        print(f'    units {int(cs.sum()):,} of {len(rp):,} ({100*cs.mean():.1f}%)')
        print(f'    events  primary {int(ypc.sum())}   second {int(ysc.sum())}   '
              f'intersection {inter}   union {union}   Jaccard {inter/max(union,1):.3f}')
        print(f'    endpoint agreement {float((ypc==ysc).mean()):.4f}   '
              f'kappa {kappa(ypc, ysc):.4f}')
        allJ = int(((yp==1)&(ys==1)).sum())/max(int(((yp==1)|(ys==1)).sum()),1)
        print(f'    (full support Jaccard was {allJ:.3f})')

        # --- CHECKS 3/5/6: contrasts with positive-claim counts, A and C, full and common support
        print(f'\n  CHECKS 3,5,6  contrasts  (def A = (p,a);  def C = (p,c) branch-free)')
        print(f'    {"annot":8s} {"def":4s} {"support":14s} {"units":>6s} {"ev":>4s} '
              f'{"posCl":>6s} {"AUROC":>7s} {"margin":>7s} {"delta":>8s} {"[95% CI]":>20s}')
        cres = {}
        for nm, rows in (('primary', rp), ('second', rs)):
            for dn, feats in (('A', ('p', 'a')), ('C', ('p', 'c'))):
                for sup, mk in (('full', None), ('common', cs)):
                    r = contrast(rows, feats, mk)
                    cres[f'{nm}|{dn}|{sup}'] = r
                    if r.get('estimable'):
                        print(f"    {nm:8s} {dn:4s} {sup:14s} {r['units']:6d} {r['events']:4d} "
                              f"{r['positive_claims']:6d} {r['auroc']:7.4f} {r['auroc_margin']:7.4f} "
                              f"{r['delta']:+8.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
                    else:
                        print(f"    {nm:8s} {dn:4s} {sup:14s} {r['units']:6d} {r['events']:4d} "
                              f"{'':6s} {'':7s} {'':7s} NOT ESTIMABLE (single-class endpoint)")
        out[f'external{which}'] = dict(branches=bres, common_support=dict(
            units=int(cs.sum()), events_primary=int(ypc.sum()), events_second=int(ysc.sum()),
            intersection=inter, union=union, jaccard=inter/max(union,1),
            agreement=float((ypc==ysc).mean()), kappa=float(kappa(ypc,ysc))), contrasts=cres)

        # --- CHECK 4: label-level agreement
        lp = []; ls = []
        for c in sorted(set(PRI) & set(SEC)):
            yr_p, cd_p = PRI[c]; yr_s, cd_s = SEC[c]
            n = min(len(cd_p), len(cd_s))
            lp.append(cd_p[:n]); ls.append(cd_s[:n])
        lp = np.concatenate(lp); ls = np.concatenate(ls)
        agree = float((lp == ls).mean())
        cats = sorted(set(lp.tolist()) | set(ls.tolist()))
        pe = sum(float((lp == k).mean()) * float((ls == k).mean()) for k in cats)
        kap = (agree - pe) / (1 - pe) if pe < 1 else float('nan')
        print(f'\n  CHECK 4  label-level agreement on {len(lp):,} associations: '
              f'{agree:.4f}   kappa {kap:.4f}   (source: 0.52)')
        out[f'external{which}']['label_agreement'] = dict(n=int(len(lp)), agreement=agree,
                                                         kappa=float(kap))
    json.dump(out, open(os.path.join(HERE, 'results', 'external_v2_results.json'), 'w'),
              indent=1, default=float)
    print('\nwritten results/external_v2_results.json')


if __name__ == '__main__':
    main()
