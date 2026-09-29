"""Issue 2: does temporal order add information beyond stationary composition?
Incremental model: logit P(change) = logit(stationary posterior-predictive probability [deployable Beta-Binomial,
claim-held-out, from issue 1]) + b0 + b'z, where z are sign-invariant, order-sensitive features of the pre-cutoff
signed sequence expressed relative to the current majority: recent-vs-historical dissent, recency-weighted margin,
cumulative-margin slope, latest opposing run length, contraction toward zero, CUSUM range, recent acceleration of
contradiction. Fitted by class-balanced penalised likelihood with the offset fixed; leave-one-claim-out; the ridge
strength is chosen by nested grouped 5-fold log loss inside each training fold (grid 0.1/1/10); unweighted likelihood. Reference
fits: offset + composition (|p|, agreement, log signed count) to check that order features are not proxies for
composition; composition-only two-variable model. Permutation check: the order of the pre-cutoff labels is shuffled
within each claim (composition preserved, order destroyed) 200 times and the offset model refitted (C=1) -> null
distribution of the AUROC gain. Writes results/issue2/."""

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
import os, sys, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_common as pc
from scipy.special import logit, expit
ac, nm = pc.ac, pc.nm
NPERM = 200


def nested_loco(X, y, g, offset, grid=(0.1, 1.0, 10.0)):
    s = np.zeros(len(y)); chosen = []
    claims = np.unique(g)
    for c in claims:
        te = g == c; tr = ~te
        best, bestC = None, 1.0
        for C in grid:
            # inner: grouped 5-fold over the training claims (claims held out together), log loss
            from sklearn.model_selection import GroupKFold
            from sklearn.preprocessing import StandardScaler
            si = np.zeros(tr.sum()); Xtr, ytr, gtr, otr = X[tr], y[tr], g[tr], offset[tr]
            for itr, ite in GroupKFold(5).split(Xtr, ytr, gtr):
                sc = StandardScaler().fit(Xtr[itr]); th = pc.fit_offset_logistic(sc.transform(Xtr[itr]), ytr[itr], otr[itr], C); si[ite] = expit(otr[ite] + sc.transform(Xtr[ite]) @ th[1:] + th[0])
            ll = pc.LOGLOSS(ytr, si)
            if best is None or ll < best: best, bestC = ll, C
        chosen.append(bestC)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X[tr]); th = pc.fit_offset_logistic(sc.transform(X[tr]), y[tr], offset[tr], bestC)
        s[te] = expit(offset[te] + sc.transform(X[te]) @ th[1:] + th[0])
    return s, chosen


def main():
    U = pc.source_units(); y, g, P = U['y'], U['g'], U['P']; n = len(y); rng = np.random.default_rng(3)
    S1 = {(r['claim_id'], int(float(r['cutoff_year']))): r for r in csv.DictReader(open(os.path.join(pc.RES, 'issue1', 'source_scores.csv')))}
    dep = np.array([float(S1[(r['claim_id'], int(r['cutoff_year']))]['bb_deployable']) for r in P]); off = logit(np.clip(dep, 1e-4, 1 - 1e-4))
    Z = np.array([[o[k] for k in pc.ORDER_KEYS] for o in U['order']])
    comp = np.column_stack([np.abs(U['p']), U['a'], np.log1p(U['npl'] + U['nmi'])])
    su = 1 - np.abs(U['p']); s2 = ac.loco_scores(U['X7'][:, :2], y, g)
    print('fitting nested offset models', flush=True)
    s_ord, C_ord = nested_loco(Z, y, g, off); s_comp, C_comp = nested_loco(comp, y, g, off); s_both, C_both = nested_loco(np.column_stack([comp, Z]), y, g, off)
    s_ord_nooff, _ = nested_loco(Z, y, g, np.zeros(n))
    res = {'protocol': __doc__, 'n_units': int(n), 'events': int(y.sum()), 'chosen_C': {'order': C_ord, 'composition': C_comp, 'both': C_both}, 'models': {}}
    scores = {'stationary_offset_only': dep, 'offset_plus_order': s_ord, 'offset_plus_composition': s_comp, 'offset_plus_composition_plus_order': s_both, 'order_only_no_offset': s_ord_nooff, 'two_var': s2, 'margin': su}
    for k, s in scores.items():
        d = pc.full_metrics(y, s, s if k != 'two_var' and k != 'margin' else pc.loco_calibrated(s, y, g)); d['auroc_ci'] = pc.ci(y, s, g, pc.AUROC); d['auprc_ci'] = pc.ci(y, s, g, pc.AUPRC)
        d['d_auroc_vs_offset'], d['d_auroc_vs_offset_ci'] = pc.paired_ci(y, s, dep, g, pc.AUROC); d['d_auprc_vs_offset'], d['d_auprc_vs_offset_ci'] = pc.paired_ci(y, s, dep, g, pc.AUPRC)
        d['d_logloss_vs_offset'], d['d_logloss_vs_offset_ci'] = pc.paired_ci(y, s, dep, g, lambda yy, pp: -pc.LOGLOSS(yy, pp)) if k not in ('two_var', 'margin') else (None, None)
        d['d_auroc_vs_two_var'], d['d_auroc_vs_two_var_ci'] = pc.paired_ci(y, s, s2, g, pc.AUROC)
        res['models'][k] = d; print(f"  {k:36s} AUROC {d['auroc']:.4f} {d['auroc_ci']} AUPRC {d['auprc']:.4f} logloss {d['logloss']:.4f}  d_offset {d['d_auroc_vs_offset']:+.4f} {d['d_auroc_vs_offset_ci']}  d_2v {d['d_auroc_vs_two_var']:+.4f} {d['d_auroc_vs_two_var_ci']}", flush=True)
    # coefficients of the full-data offset+order fit (standardised) for reporting
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Z); th = pc.fit_offset_logistic(sc.transform(Z), y, off, 1.0); res['order_coefficients_full_fit_C1'] = dict(zip(['intercept'] + pc.ORDER_KEYS, [round(float(v), 3) for v in th]))
    # permutation check: shuffle the order of the pre-cutoff labels within each claim; composition identical
    print('permutation check', flush=True); obs = res['models']['offset_plus_order']['d_auroc_vs_offset']; null = []
    streams = {c: ac.stream(c) for c in set(g)}
    for b in range(NPERM):
        Zb = np.zeros_like(Z)
        for i, r in enumerate(P):
            yrs, codes, _ = streams[r['claim_id']]; t = int(r['cutoff_year']); m = yrs < t; cp = codes.copy(); cp[m] = rng.permutation(codes[m])
            Zb[i] = [pc.order_feats(yrs, cp, t)[k] for k in pc.ORDER_KEYS]
        sb = pc.loco_offset(Zb, y, g, off, 1.0); null.append(float(pc.AUROC(y, sb) - pc.AUROC(y, dep)))
        if b % 20 == 0: print(f'  perm {b}: gain {null[-1]:+.4f}', flush=True)
    null = np.array(null); s_c1 = pc.loco_offset(Z, y, g, off, 1.0); obs_c1 = float(pc.AUROC(y, s_c1) - pc.AUROC(y, dep))
    res['permutation'] = {'draws': NPERM, 'observed_gain_C1': obs_c1, 'observed_gain_nested': obs, 'null_mean': float(null.mean()), 'null_sd': float(null.std()), 'null_q025': float(np.quantile(null, .025)), 'null_q975': float(np.quantile(null, .975)),
                          'p_value_one_sided': float((1 + (null >= obs_c1).sum()) / (NPERM + 1))}
    print('  permutation:', res['permutation'], flush=True)
    pc.save_json(res, 'issue2/temporal_offset.json')
    pc.save_csv([{'claim_id': r['claim_id'], 'cutoff_year': r['cutoff_year'], 'y': int(y[i]), **{k: float(f'{float(s[i]):.10g}') for k, s in scores.items()}, **{k: round(float(Z[i, j]), 5) for j, k in enumerate(pc.ORDER_KEYS)}} for i, r in enumerate(P)], 'issue2/source_scores.csv')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
