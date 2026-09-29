"""Issue 3: sign-invariant primary model and claim-wise sign-flip experiments (source cohort, leave-one-claim-out).
Sign-invariant feature set: |p_t|, agreement, NULL share, log signed count, log record count, |mean effect|, effect sd,
n_rr, plus the seven order features relative to the current majority (issue 2). The target is already defined relative
to the current majority (change of the subsequent-stream sign), so it is invariant to flipping all signed labels
of a claim. Experiments: (i) sign-invariant model vs the signed two-variable and Compact-7 models; (ii) claim-wise
sign-flip: for 200 random draws each claim's signed labels are flipped with probability 1/2 (all units of the claim
together), signed models refitted and re-evaluated; also the deterministic full flip; (iii) flip augmentation:
signed models trained on original + flipped copies, tested on the original. Writes results/issue3/."""

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
ac, nm = pc.ac, pc.nm
NDRAW = 200


def main():
    U = pc.source_units(); y, g, P = U['y'], U['g'], U['P']; n = len(y); rng = np.random.default_rng(3)
    X7 = U['X7']; X2 = X7[:, :2]; p = U['p']; a = U['a']
    Z = np.array([[o[k] for k in pc.ORDER_KEYS] for o in U['order']])
    nullshare = U['nnull'] / np.maximum(U['npl'] + U['nmi'] + U['nnull'], 1)
    Xinv = np.column_stack([np.abs(p), a, nullshare, np.log1p(U['npl'] + U['nmi']), np.log1p(np.array([r['n_pre_t'] for r in P])), np.abs(X7[:, 4]), X7[:, 5], X7[:, 6], Z])
    Xinv_small = np.column_stack([np.abs(p), a])
    su = 1 - np.abs(p); s2 = ac.loco_scores(X2, y, g); s7 = ac.loco_scores(X7, y, g); sinv = ac.loco_scores(Xinv, y, g); sinv2 = ac.loco_scores(Xinv_small, y, g)
    res = {'protocol': __doc__, 'n_units': int(n), 'events': int(y.sum()), 'models': {}}
    for k, s in (('margin', su), ('two_var_signed', s2), ('compact7_signed', s7), ('sign_invariant_two_var', sinv2), ('sign_invariant_full', sinv)):
        d = pc.full_metrics(y, s, pc.loco_calibrated(s, y, g)); d['auroc_ci'] = pc.ci(y, s, g, pc.AUROC); d['auprc_ci'] = pc.ci(y, s, g, pc.AUPRC)
        d['d_auroc_vs_margin'], d['d_auroc_vs_margin_ci'] = pc.paired_ci(y, s, su, g, pc.AUROC); d['d_auroc_vs_two_var'], d['d_auroc_vs_two_var_ci'] = pc.paired_ci(y, s, s2, g, pc.AUROC)
        d['d_auprc_vs_two_var'], d['d_auprc_vs_two_var_ci'] = pc.paired_ci(y, s, s2, g, pc.AUPRC)
        res['models'][k] = d; print(f"  {k:26s} AUROC {d['auroc']:.4f} {d['auroc_ci']} AUPRC {d['auprc']:.4f}  d_2v {d['d_auroc_vs_two_var']:+.4f} {d['d_auroc_vs_two_var_ci']}", flush=True)
    # direction strata of the signed two-variable model (reference)
    for name, m in (('p_negative', p < 0), ('p_positive', p > 0)):
        res['models'][f'two_var_signed_{name}'] = {'units': int(m.sum()), 'events': int(y[m].sum()), 'auroc': float(pc.AUROC(y[m], s2[m])), 'margin_auroc': float(pc.AUROC(y[m], su[m])), 'sign_invariant_full_auroc': float(pc.AUROC(y[m], sinv[m]))}
    # (ii) claim-wise sign flips
    claims = np.unique(g); flips = {'two_var_signed': [], 'compact7_signed': [], 'sign_invariant_full': []}
    def flipped(X, fl):
        Xf = X.copy(); Xf[fl, 0] = -Xf[fl, 0]; return Xf
    for d_ in range(NDRAW):
        fc = set(rng.choice(claims, size=len(claims) // 2, replace=False)); fl = np.array([c in fc for c in g])
        flips['two_var_signed'].append(float(pc.AUROC(y, ac.loco_scores(flipped(X2, fl), y, g)))); flips['compact7_signed'].append(float(pc.AUROC(y, ac.loco_scores(flipped(X7, fl), y, g))))
        if d_ < 5: flips['sign_invariant_full'].append(float(pc.AUROC(y, ac.loco_scores(Xinv, y, g))))   # invariant by construction: features do not change
        if d_ % 40 == 0: print(f'  flip draw {d_}: 2v {flips["two_var_signed"][-1]:.4f} C7 {flips["compact7_signed"][-1]:.4f}', flush=True)
    full = np.ones(n, bool)
    res['sign_flip'] = {'draws': NDRAW, 'design': 'each claim flipped with probability 1/2 (exactly half of the claims per draw), all units of a claim together',
                        'two_var_signed': {'original': float(pc.AUROC(y, s2)), 'mean': float(np.mean(flips['two_var_signed'])), 'sd': float(np.std(flips['two_var_signed'])), 'min': float(np.min(flips['two_var_signed'])), 'max': float(np.max(flips['two_var_signed'])), 'full_flip': float(pc.AUROC(y, ac.loco_scores(flipped(X2, full), y, g)))},
                        'compact7_signed': {'original': float(pc.AUROC(y, s7)), 'mean': float(np.mean(flips['compact7_signed'])), 'sd': float(np.std(flips['compact7_signed'])), 'min': float(np.min(flips['compact7_signed'])), 'max': float(np.max(flips['compact7_signed'])), 'full_flip': float(pc.AUROC(y, ac.loco_scores(flipped(X7, full), y, g)))},
                        'sign_invariant_full': {'original': float(pc.AUROC(y, sinv)), 'under_flips': float(np.mean(flips['sign_invariant_full'])), 'note': 'identical by construction'}}
    print('  flips:', {k: (round(v['mean'], 4), round(v['min'], 4), round(v['max'], 4), round(v['full_flip'], 4)) for k, v in res['sign_flip'].items() if isinstance(v, dict) and 'full_flip' in v}, flush=True)
    # (iii) flip augmentation, LOCO: training set = original + fully flipped copies of the training claims
    def loco_aug(X):
        s = np.zeros(n)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        for c in claims:
            te = g == c; tr = ~te; Xtr = np.vstack([X[tr], flipped(X[tr], np.ones(tr.sum(), bool))]); ytr = np.concatenate([y[tr], y[tr]])
            sc = StandardScaler().fit(Xtr); m = LogisticRegression(max_iter=3000, class_weight='balanced').fit(sc.transform(Xtr), ytr); s[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        return s
    s2a = loco_aug(X2); s7a = loco_aug(X7)
    res['flip_augmentation'] = {}
    for k, s in (('two_var_augmented', s2a), ('compact7_augmented', s7a)):
        d = pc.full_metrics(y, s, pc.loco_calibrated(s, y, g)); d['auroc_ci'] = pc.ci(y, s, g, pc.AUROC); d['d_auroc_vs_two_var'], d['d_auroc_vs_two_var_ci'] = pc.paired_ci(y, s, s2, g, pc.AUROC)
        d['auroc_under_full_flip'] = float(pc.AUROC(y, loco_aug(flipped(X2 if k.startswith('two') else X7, full))))
        res['flip_augmentation'][k] = d; print(f"  {k:26s} AUROC {d['auroc']:.4f} {d['auroc_ci']} d_2v {d['d_auroc_vs_two_var']:+.4f} {d['d_auroc_vs_two_var_ci']} under full flip {d['auroc_under_full_flip']:.4f}", flush=True)
    pc.save_json(res, 'issue3/sign_invariant.json')
    pc.save_csv([{'claim_id': r['claim_id'], 'cutoff_year': r['cutoff_year'], 'y': int(y[i]), 'margin': float(f'{float(su[i]):.10g}'), 'two_var_signed': float(f'{float(s2[i]):.10g}'), 'compact7_signed': float(f'{float(s7[i]):.10g}'), 'sign_invariant_full': float(f'{float(sinv[i]):.10g}'), 'sign_invariant_two_var': float(f'{float(sinv2[i]):.10g}'), 'two_var_augmented': float(f'{float(s2a[i]):.10g}')} for i, r in enumerate(P)], 'issue3/source_scores.csv')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
