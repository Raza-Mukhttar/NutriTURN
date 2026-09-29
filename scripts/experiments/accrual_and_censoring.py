"""E6: accrued vs censored candidates, and accrual-weighted AUROC on the calendar cohort.

The calendar cohort scores only candidates whose window later accrues at least 10 signed records,
so its AUROC estimates P(change | sufficient later literature). This script

  1. tabulates standardised mean differences between accrued and censored candidates on
     pre-origin features (signed count, |p_t|, NULL share, recent accrual, years of history);
  2. fits a claim-disjoint logistic model of accrual on those features;
  3. reweights each accrued unit by the inverse predicted probability of accrual and recomputes
     the margin, stationary and two-variable AUROCs and their paired differences;
  4. skips the weighting at H = 10, where nothing is censored.

  python analyses/nfx_accrual.py
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

FEATS = ['n_signed', 'abs_p', 'null_share', 'accrual_5y', 'span_years']


def add_feats(rows):
    for r in rows:
        r['abs_p'] = abs(r['p'])
    return rows


def weighted_auroc(y, s, w):
    """AUROC over positive-negative pairs weighted by w_i * w_j; 0.5 credit for ties."""
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    d = s[pos][:, None] - s[neg][None, :]
    ww = w[pos][:, None] * w[neg][None, :]
    num = (ww * ((d > 0) + 0.5 * (d == 0))).sum()
    return float(num / ww.sum())


def boot_weighted(y, sa, sb, g, w, n=N.BOOT, seed=N.SEED):
    vals = []
    for ix in N.claim_boot_pairs(y, g, n, seed):
        if len(np.unique(y[ix])) < 2:
            continue
        vals.append(weighted_auroc(y[ix], sa[ix], w[ix]) - weighted_auroc(y[ix], sb[ix], w[ix]))
    v = np.array([x for x in vals if np.isfinite(x)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def stationary_cohort(rows):
    """Count-predicted operational Beta-Binomial probability for cohort rows (per-origin count model)."""
    sys.path.insert(0, os.path.join(N.GPTPRO, 'scripts'))
    import pro_common as pc
    rng = np.random.default_rng(N.SEED)
    out = np.zeros(len(rows))
    g = np.array([r['claim_id'] for r in rows])
    ln_pre = np.log1p(np.array([r['n_signed'] for r in rows], float))
    ln_rate = np.log1p(np.array([r['accrual_5y'] for r in rows], float))
    ln_m = np.log1p(np.array([max(r['m_signed'], 0) for r in rows], float))
    for c in np.unique(g):
        tr = g != c
        b, sd = pc.fit_count_model(ln_pre[tr], ln_rate[tr], ln_m[tr])
        for i in np.where(g == c)[0]:
            ms = pc.draw_counts(b, sd, ln_pre[i], ln_rate[i], rng)
            vals, cnts = np.unique(ms, return_counts=True)
            out[i] = sum(cn * pc.bb_change_prob(rows[i]['h'], rows[i]['l'], int(v),
                                                s0=N.sgn0(rows[i]['p']))
                         for v, cn in zip(vals, cnts)) / cnts.sum()
    return out


def main():
    co = add_feats(N.build_cohort('primary'))
    out = {'smd': [], 'cells': []}
    print('E6 accrued vs censored candidates\n')
    print('standardised mean differences (accrued minus censored), per horizon')
    for H in (3, 5, 10):
        d = [r for r in co if r['horizon'] == H]
        acc = [r for r in d if r['accrued']]; cen = [r for r in d if not r['accrued']]
        if not cen:
            print(f'  H{H}: no censored candidates (0 of {len(d)}); no comparison and no weighting needed')
            out['smd'].append(dict(horizon=H, censored=0))
            continue
        row = dict(horizon=H, candidates=len(d), accrued=len(acc), censored=len(cen))
        line = []
        for f in FEATS:
            av = np.array([r[f] for r in acc], float); cv = np.array([r[f] for r in cen], float)
            sd = np.array([r[f] for r in d], float).std()
            smd = float((av.mean() - cv.mean()) / sd) if sd > 0 else 0.0
            row[f'smd_{f}'] = smd
            line.append(f'{f} {smd:+.2f}')
        out['smd'].append(row)
        print(f'  H{H} (accrued {len(acc)}, censored {len(cen)}): ' + '  '.join(line))
    mx = max(abs(r.get(f'smd_{f}', 0.0)) for r in out['smd'] for f in FEATS if f'smd_{f}' in r)
    print(f'\nlargest absolute standardised mean difference across all horizons and features: {mx:.2f}\n')
    out['max_abs_smd'] = mx

    print('accrual-weighted AUROC among accrued units (inverse predicted probability of accrual)')
    for H in (3, 5, 10):
        d = [r for r in co if r['horizon'] == H]
        acc_idx = [i for i, r in enumerate(d) if r['accrued']]
        yA = np.array([d[i]['y'] for i in acc_idx])
        gA = np.array([d[i]['claim_id'] for i in acc_idx])
        s2, ys, gs, os_ = N.cohort_loco_two_var(d, H)
        assert len(s2) == len(acc_idx) and (ys == yA).all()
        sm = np.array([1 - abs(d[i]['p']) for i in acc_idx])
        sb = stationary_cohort([d[i] for i in acc_idx])
        cen = sum(1 for r in d if not r['accrued'])
        if cen == 0:
            w = np.ones(len(acc_idx))
            note = 'no censoring at this horizon; weights are all 1'
        else:
            # claim-disjoint logistic model of accrual on pre-origin features
            X = np.array([[r[f] for f in FEATS] for r in d], float)
            ya = np.array([r['accrued'] for r in d])
            g_all = np.array([r['claim_id'] for r in d])
            ph = np.zeros(len(d))
            for c in np.unique(g_all):
                te = g_all == c; tr = ~te
                if len(np.unique(ya[tr])) < 2:
                    ph[te] = ya[tr].mean(); continue
                sc = StandardScaler().fit(X[tr])
                m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
                m.fit(sc.transform(X[tr]), ya[tr])
                ph[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
            pa = np.clip(ph[acc_idx], 0.05, 1.0)
            w = 1.0 / pa
            w = w / w.mean()
            note = (f'accrual model AUC {N.auroc(ya, ph):.3f}; weights in '
                    f'[{w.min():.2f}, {w.max():.2f}]')
        cell = dict(horizon=H, units=len(acc_idx), events=int(yA.sum()), censored=cen, note=note)
        for nm_, sc_ in (('margin', sm), ('stationary', sb), ('two_var', s2)):
            cell[f'{nm_}_auroc'] = N.auroc(yA, sc_)
            cell[f'{nm_}_auroc_weighted'] = weighted_auroc(yA, sc_, w)
        for nm_, sc_ in (('stationary', sb), ('two_var', s2)):
            d_u, lo_u, hi_u, _ = N.paired_ci(yA, sc_, sm, gA)
            dw = weighted_auroc(yA, sc_, w) - weighted_auroc(yA, sm, w)
            low, hiw = boot_weighted(yA, sc_, sm, gA, w)
            cell[f'{nm_}_minus_margin'] = [d_u, lo_u, hi_u]
            cell[f'{nm_}_minus_margin_weighted'] = [dw, low, hiw]
            cell[f'{nm_}_conclusion_changes'] = bool(N.established(lo_u, hi_u) != N.established(low, hiw))
        out['cells'].append(cell)
        print(f"  H{H} ({len(acc_idx)} units, {int(yA.sum())} events, {cen} censored) {note}")
        print(f"     unweighted: margin {cell['margin_auroc']:.3f}  stationary {cell['stationary_auroc']:.3f}  "
              f"two-var {cell['two_var_auroc']:.3f}")
        print(f"     weighted  : margin {cell['margin_auroc_weighted']:.3f}  "
              f"stationary {cell['stationary_auroc_weighted']:.3f}  two-var {cell['two_var_auroc_weighted']:.3f}")
        for nm_ in ('stationary', 'two_var'):
            du, lu, hu = cell[f'{nm_}_minus_margin']; dw2, lw, hw = cell[f'{nm_}_minus_margin_weighted']
            print(f"     {nm_:10s} minus margin  unweighted {du:+.3f} {N.fmt_ci(lu,hu)}"
                  f"{' EST' if N.established(lu,hu) else '    '}   "
                  f"weighted {dw2:+.3f} {N.fmt_ci(lw,hw)}"
                  f"{' EST' if N.established(lw,hw) else '    '}"
                  f"{'   <- CONCLUSION CHANGES' if cell[f'{nm_}_conclusion_changes'] else ''}")
    changed = [c for c in out['cells'] if any(c.get(f'{k}_conclusion_changes') for k in ('stationary', 'two_var'))]
    mxd = max(abs(c[f'{k}_minus_margin_weighted'][0] - c[f'{k}_minus_margin'][0])
              for c in out['cells'] for k in ('stationary', 'two_var'))
    out['any_conclusion_changes'] = bool(changed)
    out['max_abs_change_in_difference'] = mxd
    print(f'\nlargest change in any paired difference from weighting: {mxd:.3f}')
    print('conclusions changed by weighting:', 'NONE' if not changed else [c['horizon'] for c in changed])
    print('written', N.save_json(out, 'E6_accrual.json'))


if __name__ == '__main__':
    main()
