"""N4 and N8: face validity against documented reversals, and the second logistic coefficient.

N4. Eight claims in the corpus have a documented conclusion change with a pivotal trial or
guidance date taken from the external literature, not chosen by us. For each, this prints the
cutoff grid, the pooled direction at each cutoff, the endpoint under all three annotations, and
the earliest event cutoff relative to the pivotal date. It also records whether the direction of
change matches the documented direction, and flags capped claims whose archive begins after the
pivotal date, where the reversal cannot be observable in principle.

N8. The signed two-variable logit is b0 + b1 p + b2 |p| on the signed branch. Reporting b2 and the
two within-class slopes b2+b1 and b2-b1 replaces an inferential paragraph with numbers: the slope
in |p| is b2+b1 among HARMFUL majorities and b2-b1 among PROTECTIVE ones.

  python analyses/nfx_reversals.py
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
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# pivotal dates from the external literature (Prasad et al. 2013 where listed), not selected by us
PIVOTAL = [
    ('hormone_replacement_cvd', 2002, 'WHI', 'PROTECTIVE->HARMFUL'),
    ('hormone_replacement_breast_cancer', 2002, 'WHI', 'PROTECTIVE->HARMFUL'),
    ('beta_carotene_cancer', 1996, 'ATBC 1994, CARET 1996', 'PROTECTIVE->HARMFUL'),
    ('vitamin_e_cvd', 2002, 'HOPE 2000, HPS 2002', 'PROTECTIVE->NULL/HARMFUL'),
    ('selenium_prostate_cancer', 2009, 'SELECT', 'PROTECTIVE->NULL'),
    ('folic_acid_colorectal', 2007, 'Cole et al.', 'PROTECTIVE->HARMFUL'),
    ('niacin_cvd', 2014, 'AIM-HIGH 2011, HPS2-THRIVE 2014', 'PROTECTIVE->NULL'),
    ('dietary_cholesterol_cvd', 2015, '2015 dietary guidance', 'HARMFUL->NULL'),
]


def main():
    out = {'N4': [], 'N8': {}}
    tabs = {a: N.unit_table(a) for a in ('primary', 'second', 'consensus')}
    idx = {a: {(r['claim_id'], r['cutoff']): r for r in tabs[a]} for a in tabs}
    cap = set()
    import csv as _csv
    for r in _csv.DictReader(open(os.path.join(N.RESULTS, 'E15_cap_census.csv'))):
        if 780 <= int(r['archived']) <= 800:
            cap.add(r['claim_id'])
    S = N.streams('primary')

    print('N4 documented reversals against the vote-count endpoint\n')
    for cid, piv, src, direction in PIVOTAL:
        cuts = sorted(k[1] for k in idx['primary'] if k[0] == cid)
        if not cuts:
            print(f'{cid}: not future-eligible at any cutoff'); continue
        yrs, _ = S[cid]
        first = {}
        for a in ('primary', 'second', 'consensus'):
            ev = [c for c in cuts if idx[a][(cid, c)]['y']]
            first[a] = min(ev) if ev else None
        row = dict(claim_id=cid, pivotal_year=piv, source=src, documented=direction,
                   archive_starts=int(yrs.min()), capped=cid in cap,
                   cutoffs=cuts,
                   p_at_cutoff={c: round(idx['primary'][(cid, c)]['p'], 3) for c in cuts},
                   p_post={c: round(idx['primary'][(cid, c)]['p_post'], 3) for c in cuts},
                   first_event=first,
                   observable=bool(int(yrs.min()) <= piv))
        for a in ('primary', 'second', 'consensus'):
            row[f'lag_{a}'] = (first[a] - piv) if first[a] is not None else None
        out['N4'].append(row)
        flag = '  CAPPED, archive starts after the pivotal date' if (cid in cap and yrs.min() > piv) else ''
        print(f'{cid}  (pivotal {piv}, {src})')
        print(f'   archive starts {int(yrs.min())}   cutoffs {cuts}{flag}')
        print(f'   p_t at cutoffs  ' + '  '.join(f'{c}:{idx["primary"][(cid,c)]["p"]:+.2f}' for c in cuts))
        for a in ('primary', 'second', 'consensus'):
            fe = first[a]
            lag = 'no event' if fe is None else (f'{fe} ({fe-piv:+d} yr)')
            print(f'   first event, {a:10s} {lag}')
        print()
    obs = [r for r in out['N4'] if r['observable']]
    caught = [r for r in obs if r['first_event']['primary'] is not None]
    early = [r for r in caught if r['lag_primary'] <= 0]
    out['summary'] = dict(claims=len(out['N4']), observable=len(obs), caught_primary=len(caught),
                          caught_at_or_before_pivotal=len(early),
                          capped_unobservable=sum(1 for r in out['N4'] if not r['observable']))
    print(f'summary: {len(out["N4"])} documented reversals; {len(obs)} have an archive reaching '
          f'the pivotal date; the endpoint flags {len(caught)} of those under the primary '
          f'annotation, {len(early)} at or before the pivotal year')
    print(f'         {out["summary"]["capped_unobservable"]} are unobservable because the archive '
          f'begins after the pivotal date')

    # ---------------- N8
    print('\n\nN8 the second coefficient and the within-class slopes\n')
    rows = tabs['primary']
    X = np.array([[r['p'], r['a']] for r in rows], float)
    y = np.array([r['y'] for r in rows]); g = np.array([r['claim_id'] for r in rows])
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000).fit(sc.transform(X), y)
    cp, ca = float(m.coef_[0][0]), float(m.coef_[0][1]); c0 = float(m.intercept_[0])
    rp, ra = cp / sc.scale_[0], ca / sc.scale_[1]
    r0 = c0 - cp * sc.mean_[0] / sc.scale_[0] - ca * sc.mean_[1] / sc.scale_[1]
    b0, b1, b2 = r0 + ra * 0.5, rp, ra * 0.5
    out['N8'] = dict(standardised_p=cp, standardised_a=ca,
                     raw_b0=b0, raw_b1=b1, raw_b2=b2,
                     slope_in_abs_p_harmful_majority=b2 + b1,
                     slope_in_abs_p_protective_majority=b2 - b1)
    print(f'   standardised: b on p {cp:+.3f}   b on a {ca:+.3f}')
    print(f'   raw signed-branch logit b0 + b1 p + b2 |p|:')
    print(f'      b0 {b0:+.3f}   b1 {b1:+.3f}   b2 {b2:+.3f}')
    print(f'   within-class slope in |p|:  HARMFUL majority b2+b1 = {b2+b1:+.3f}   '
          f'PROTECTIVE majority b2-b1 = {b2-b1:+.3f}')
    # fold stability
    both_neg = 0; folds = 0
    for c in np.unique(g):
        tr = g != c
        if len(np.unique(y[tr])) < 2:
            continue
        sct = StandardScaler().fit(X[tr])
        mt = LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000)
        mt.fit(sct.transform(X[tr]), y[tr])
        p_, a_ = mt.coef_[0]
        rp_, ra_ = p_ / sct.scale_[0], a_ / sct.scale_[1]
        b1_, b2_ = rp_, ra_ * 0.5
        folds += 1
        if (b2_ + b1_) < 0 and (b2_ - b1_) < 0:
            both_neg += 1
    out['N8']['folds'] = folds
    out['N8']['folds_both_slopes_negative'] = both_neg
    out['N8']['share_both_negative'] = both_neg / max(folds, 1)
    print(f'   leave-one-claim-out folds in which BOTH within-class slopes are negative: '
          f'{both_neg} of {folds} ({100*both_neg/max(folds,1):.1f}%)')
    print('\nwritten', N.save_json(out, 'N4_N8_reversals_coef.json'))


if __name__ == '__main__':
    main()
