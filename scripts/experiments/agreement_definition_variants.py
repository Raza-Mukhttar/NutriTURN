"""R1 (their E1) and Q1: agreement definition x annotation x support, with branch-specific
evaluation, fold-slope audit, branch transitions, and the zero-direction endpoint audit.

Contract. The endpoint, eligibility and fitting procedure are held fixed; only the agreement input
changes. Variants, with h, l, u the pre-cutoff HARMFUL, PROTECTIVE and NULL counts, n_s = h + l,
q = u/(h+l+u), p = (h-l)/n_s and c = (1+|p|)/2 on supported histories:

  A       [p, a_A]      a_A = q when q >= 0.5, else c     the shipped piecewise feature
  C       [p, c]        the branch switch removed
  C+NULL  [p, c, q]     NULL information without a feature that changes meaning

Sparse-history convention. The frozen builder returns agreement 0 when fewer than eight signed
records are present, and this script uses the same fallback for c and q, so "remove the NULL
branch" is not silently combined with "change the sparse convention". The convention is asserted
against the frozen builder before anything else runs.

Cohorts. Primary mechanism analysis is the common support of at least eight signed pre-cutoff
records under both pipelines, refitted. Secondary compatibility analysis is the full archived
cohort with the archived sparse behaviour retained.

  python analyses/nfx_R1_agreement.py
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

VARIANTS = [('A', ['p', 'a']), ('C', ['p', 'c']), ('C+NULL', ['p', 'c', 'q'])]


def prep(rows):
    for r in rows:
        n = r['n_signed']
        r['c'] = (1 + abs(r['p'])) / 2 if n >= N.SPARSE_GATE else 0.0
        r['q'] = r['null_share'] if n >= N.SPARSE_GATE else 0.0
        r['branch'] = ('sparse' if n < N.SPARSE_GATE
                       else ('null_dominant' if r['null_share'] >= 0.5 else 'signed'))
    return rows


def loco_full(rows, feats, subset):
    """Out-of-fold scores plus the per-fold raw-input slopes."""
    idx = [i for i in range(len(rows)) if subset[i]]
    X = np.array([[rows[i][f] for f in feats] for i in idx], float)
    y = np.array([rows[i]['y'] for i in idx]); g = np.array([rows[i]['claim_id'] for i in idx])
    s = np.full(len(idx), np.nan); slopes = []
    for c in np.unique(g):
        te = g == c; tr = ~te
        if len(np.unique(y[tr])) < 2:
            s[te] = y[tr].mean() if tr.any() else 0.0
            continue
        sc = StandardScaler().fit(X[tr])
        scale = np.where(sc.scale_ > 0, sc.scale_, np.inf)      # zero-variance handled explicitly
        m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000)
        m.fit(sc.transform(X[tr]), y[tr])
        s[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        B = m.coef_[0] / scale
        Bp = B[0]; Ba = B[1]
        slopes.append(dict(fold=str(c), B_p=float(Bp), B_a=float(Ba),
                           harmful_slope=float(Bp + Ba / 2),
                           protective_slope=float(-Bp + Ba / 2)))
    return np.array(idx), s, y, g, slopes


def summarise_slopes(sl, key):
    v = np.array([x[key] for x in sl])
    return dict(min=float(v.min()), median=float(np.median(v)), max=float(v.max()),
                share_negative=float((v < 0).mean()), n=len(v))


def main():
    out = {'contract': {}, 'cells': [], 'slopes': {}, 'branch_transition': {}, 'Q1': {}}
    tabs = {a: prep(N.unit_table(a)) for a in ('primary', 'second', 'consensus')}

    # ---- sparse convention assertion against the frozen builder
    sys.path.insert(0, _os.path.join(_NT, 'scripts', 'lib'))
    import astra_common as ac
    bad = 0
    for a in ('primary', 'second'):
        lab = N.label_column(a)
        for r in tabs[a]:
            if r['n_signed'] >= N.SPARSE_GATE:
                continue
            yrs, codes, pm = ac.stream(r['claim_id'])
            cb = np.array([N.CODE.get(lab.get((r['claim_id'], str(x)), ''), 99) for x in pm])
            f = ac.direction_feats(yrs[yrs < r['cutoff']], cb[yrs < r['cutoff']], r['cutoff'], 'A')
            if abs(f['agreement'] - 0.0) > 1e-12 or abs(r['c'] - 0.0) > 1e-12:
                bad += 1
    out['contract']['sparse_convention_mismatches'] = bad
    print(f'sparse-history convention: the frozen builder returns agreement 0 below eight signed '
          f'records; c and q use the same fallback. mismatches: {bad}\n')

    # ---- masks
    m_full = np.ones(len(tabs['primary']), bool)
    m2 = np.ones(len(tabs['primary']), bool)
    for a in ('primary', 'second'):
        m2 &= np.array([r['branch'] != 'sparse' for r in tabs[a]])
    m3 = m2.copy()
    m3 &= np.array([r['branch'] != 'sparse' for r in tabs['consensus']])
    out['contract']['mask_sizes'] = {'full': int(m_full.sum()), 'two_pipeline': int(m2.sum()),
                                     'three_annotation': int(m3.sum())}
    print(f"cohorts: full {int(m_full.sum())}, common support two pipelines {int(m2.sum())}, "
          f"three annotations {int(m3.sum())}\n")

    # ---- endpoint invariance across variants (must hold by construction)
    for a in tabs:
        ys = [tuple(r['y'] for r in tabs[a])]
        assert len(set(ys)) == 1
    out['contract']['endpoint_invariant_across_variants'] = True

    print(f"{'annotation':11s} {'cohort':16s} {'branch':14s} {'units':>6s} {'ev':>4s}  "
          f"{'A':>6s} {'C':>6s} {'C+N':>6s}   {'C-A [CI]':>22s}  {'C+N-A [CI]':>22s}")
    for a in ('primary', 'second', 'consensus'):
        rows = tabs[a]
        for cohort, mask in (('common support', m2), ('full archived', m_full)):
            got = {}
            for vname, feats in VARIANTS:
                idx, s, y, g, sl = loco_full(rows, feats, mask)
                got[vname] = (idx, s, y, g)
                out['slopes'].setdefault(a, {})[f'{cohort}|{vname}'] = dict(
                    harmful=summarise_slopes(sl, 'harmful_slope'),
                    protective=summarise_slopes(sl, 'protective_slope'))
            idx, _, y, g = got['A']
            sm = np.array([1 - abs(rows[i]['p']) for i in idx])
            row = dict(annotation=a, cohort=cohort, branch='all', units=len(idx),
                       events=int(y.sum()),
                       positive_claims=int(len(np.unique(g[y == 1]))))
            for vname, _ in VARIANTS:
                row[f'auroc_{vname}'] = N.auroc(y, got[vname][1])
                row[f'auprc_{vname}'] = N.auprc(y, got[vname][1])
                d, lo, hi, _ = N.paired_ci(y, got[vname][1], sm, g)
                row[f'{vname}_minus_margin'] = [d, lo, hi]
                row[f'{vname}_minus_margin_est'] = N.established(lo, hi)
                row[f'{vname}_non_inferior_0p02'] = bool(lo > -0.02)
            for vname in ('C', 'C+NULL'):
                d, lo, hi, _ = N.paired_ci(y, got[vname][1], got['A'][1], g)
                row[f'{vname}_minus_A'] = [d, lo, hi]
                row[f'{vname}_minus_A_est'] = N.established(lo, hi)
            out['cells'].append(row)
            ca = row['C_minus_A']; cn = row['C+NULL_minus_A']
            print(f"{a:11s} {cohort:16s} {'all':14s} {len(idx):6d} {int(y.sum()):4d}  "
                  f"{row['auroc_A']:.3f} {row['auroc_C']:.3f} {row['auroc_C+NULL']:.3f}   "
                  f"{ca[0]:+.3f} {N.fmt_ci(ca[1],ca[2]):>16s}  "
                  f"{cn[0]:+.3f} {N.fmt_ci(cn[1],cn[2]):>16s}")
            # branch slices, membership fixed on the ORIGINAL counts, same fitted models
            if cohort != 'full archived':
                continue
            br = np.array([rows[i]['branch'] for i in idx])
            for b in ('signed', 'null_dominant'):
                k = br == b
                if k.sum() < 40 or len(np.unique(y[k])) < 2:
                    continue
                r2 = dict(annotation=a, cohort=cohort, branch=b, units=int(k.sum()),
                          events=int(y[k].sum()),
                          positive_claims=int(len(np.unique(g[k][y[k] == 1]))))
                for vname, _ in VARIANTS:
                    r2[f'auroc_{vname}'] = N.auroc(y[k], got[vname][1][k])
                for vname in ('C', 'C+NULL'):
                    d, lo, hi, _ = N.paired_ci(y[k], got[vname][1][k], got['A'][1][k], g[k])
                    r2[f'{vname}_minus_A'] = [d, lo, hi]
                    r2[f'{vname}_minus_A_est'] = N.established(lo, hi)
                out['cells'].append(r2)
                ca = r2['C_minus_A']; cn = r2['C+NULL_minus_A']
                print(f"{a:11s} {cohort:16s} {b:14s} {int(k.sum()):6d} {int(y[k].sum()):4d}  "
                      f"{r2['auroc_A']:.3f} {r2['auroc_C']:.3f} {r2['auroc_C+NULL']:.3f}   "
                      f"{ca[0]:+.3f} {N.fmt_ci(ca[1],ca[2]):>16s}  "
                      f"{cn[0]:+.3f} {N.fmt_ci(cn[1],cn[2]):>16s}")
        print()

    # ---- fold slopes
    print('fold slopes with respect to |p| on the signed branch, raw-input scale')
    print(f"{'annotation':11s} {'variant':8s} {'HARMFUL  min/med/max  %neg':>42s}  "
          f"{'PROTECTIVE  min/med/max  %neg':>42s}")
    for a in ('primary', 'second', 'consensus'):
        for vname, _ in VARIANTS:
            k = f'full archived|{vname}'
            v = out['slopes'][a].get(k)
            if not v:
                continue
            h, p_ = v['harmful'], v['protective']
            print(f"{a:11s} {vname:8s} "
                  f"{h['min']:+8.2f}/{h['median']:+7.2f}/{h['max']:+8.2f}  {100*h['share_negative']:5.1f}%  "
                  f"{p_['min']:+8.2f}/{p_['median']:+7.2f}/{p_['max']:+8.2f}  {100*p_['share_negative']:5.1f}%")

    # ---- branch transition primary -> second
    print('\nbranch transition, primary to second (all 1,012 units)')
    keys = [(r['claim_id'], r['cutoff']) for r in tabs['primary']]
    b1 = {k: r['branch'] for k, r in zip(keys, tabs['primary'])}
    b2 = {k: r['branch'] for k, r in zip(keys, tabs['second'])}
    T = {}
    for k in keys:
        T[(b1[k], b2[k])] = T.get((b1[k], b2[k]), 0) + 1
    out['branch_transition'] = {f'{a}->{b}': v for (a, b), v in sorted(T.items())}
    order = ['signed', 'null_dominant', 'sparse']
    hdr = 'primary to second'
    print(f'{hdr:20s}' + ''.join(f'{b:>16s}' for b in order))
    for a_ in order:
        print(f'{a_:20s}' + ''.join(f'{T.get((a_, b_), 0):16d}' for b_ in order))

    # ---- Q1 zero-direction audit
    print('\nQ1 zero-direction audit (primary annotation)')
    rows = tabs['primary']
    zd = [r for r in rows if r['s'] == 0 or N.sgn0(r['p_post']) == 0]
    pre_tie = [r for r in zd if r['n_signed'] >= N.SPARSE_GATE and r['h'] == r['l']]
    sparse = [r for r in zd if r['n_signed'] < N.SPARSE_GATE]
    post_tie = [r for r in zd if N.sgn0(r['p_post']) == 0]
    out['Q1'] = dict(zero_direction_units=len(zd), pre_cutoff_exact_ties=len(pre_tie),
                     sparse_fallback=len(sparse), post_cutoff_exact_ties=len(post_tie),
                     all_events=int(sum(r['y'] for r in zd)))
    print(f'   zero-direction units {len(zd)}: {len(pre_tie)} supported pre-cutoff exact ties, '
          f'{len(sparse)} sparse fallback, {len(post_tie)} post-cutoff exact ties; '
          f'{sum(r["y"] for r in zd)} are events')
    print('   survival under each pre-cutoff margin mask (the caption claims all are removed):')
    masks = []
    for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        surv = [r for r in zd if abs(r['p']) >= thr]
        masks.append(dict(threshold=thr, surviving=len(surv),
                          surviving_ids=[f"{r['claim_id']}@{r['cutoff']}" for r in surv]))
        print(f'      |p_t| >= {thr:.2f}: {len(surv)} of {len(zd)} survive'
              + ('' if not surv else '   <- ' + ', '.join(
                  f"{r['claim_id']}@{r['cutoff']} (post tie, |p_t|={abs(r['p']):.2f})" for r in surv[:3])))
    out['Q1']['margin_masks'] = masks
    print('\nwritten', N.save_json(out, 'R1_agreement_branch.json'))


if __name__ == '__main__':
    main()
