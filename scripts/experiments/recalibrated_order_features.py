"""Guide A6 (optional): isolate the order-feature increment in proper scores with a matched comparison.
Model R: logistic on [logit pi_t] (Platt-style recalibration of the stationary probability: slope + intercept).
Model R+Z: logistic on [logit pi_t, z_t] (same recalibration plus the seven order features).
Identical estimator (unweighted L2 logistic, C = 1), identical leave-one-claim-out folds, identical bootstrap resamples."""

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
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
U = pc.source_units(); y, g, P = U['y'], U['g'], U['P']
S1 = {(r['claim_id'], str(int(float(r['cutoff_year'])))): r for r in csv.DictReader(open(os.path.join(pc.RES, 'issue1', 'source_scores.csv')))}
pi = np.array([float(S1[(r['claim_id'], str(int(r['cutoff_year'])))]['bb_deployable']) for r in P]); lp = logit(np.clip(pi, 1e-6, 1 - 1e-6))
Z = np.array([[o[k] for k in pc.ORDER_KEYS] for o in U['order']])
def loco(X):
    s = np.zeros(len(y))
    for c in np.unique(g):
        te = g == c; tr = ~te; sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[tr]), y[tr]); s[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return s
R = loco(lp.reshape(-1, 1)); RZ = loco(np.column_stack([lp, Z])); out = {'protocol': __doc__}
for k, s in (('recalibrated_stationary', R), ('recalibrated_stationary_plus_order', RZ), ('raw_stationary', pi)):
    d = pc.full_metrics(y, s, s); d['auroc_ci'] = pc.ci(y, s, g, pc.AUROC); out[k] = d
for name, fn, sign in (('auroc', pc.AUROC, 1), ('auprc', pc.AUPRC, 1), ('logloss', pc.LOGLOSS, -1), ('brier', pc.BRIER, -1)):
    d, c = pc.paired_ci(y, RZ, R, g, (lambda yy, ss: sign * fn(yy, ss))); out[f'd_{name}_RZ_minus_R (positive = RZ better)'] = [d, c]
print({k: (v if not isinstance(v, dict) else {kk: round(vv, 4) for kk, vv in v.items() if kk in ('auroc', 'auprc', 'logloss', 'brier', 'cal_slope', 'ece')}) for k, v in out.items() if k != 'protocol'})
pc.save_json(out, 'guide/recalibrated_order.json')
