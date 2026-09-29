"""Shared pieces for the 4/5 revision-plan analyses (W3, W5, W6): the past-only count-aware stationary
reference, the temporal comparator features, and the fixed-target order shuffle. Namespace: results/plan."""

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
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astra_common as ac
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
nm, nf = ac.nm, ac.nf
PLAN_RESULTS = ac.ns('results', 'plan'); PLAN_TABLES = ac.ns('tables', 'plan'); PLAN_FIGURES = ac.ns('figures', 'plan')
LAST_COMPLETE_YEAR = 2025          # 2026 is partially observed (Appendix A); a window [t, t+H) is complete iff t+H-1 <= 2025
ALPHA = 1.0                        # Beta(1,1) prior on the harmful share: fixed, not tuned
MC = 400                           # Monte Carlo draws per unit for the count-aware reference


def sgn0(x): return 0 if x == 0 else (1 if x > 0 else -1)


def signed_counts(codes):
    """(n_harmful, n_protective) among codes -1/0/+1 (99 = UNCLEAR)."""
    return int((codes == 1).sum()), int((codes == -1).sum())


def count_aware_score(n_plus, n_minus, ratios, rng, alpha=ALPHA, mc=MC):
    """Past-only stationary reference: P(sign of a future signed majority differs from the current sign)
    under theta ~ Beta(n+ + alpha, n- + alpha) and a future signed count m = round(r * (n+ + n-)) with r
    drawn from `ratios` (the empirical future/past signed-count ratios of the TRAINING units). Exact tie
    convention sgn(0) = 0: a tied future majority counts as a change when the past sign is non-zero."""
    n = n_plus + n_minus
    if n == 0: return 0.5
    s0 = sgn0(n_plus - n_minus)
    r = rng.choice(ratios, size=mc); m = np.maximum(1, np.rint(r * n)).astype(int)
    theta = rng.beta(n_plus + alpha, n_minus + alpha, size=mc)
    k = rng.binomial(m, theta)
    s1 = np.sign(2 * k - m).astype(int)
    return float(np.mean(s1 != s0))


def temporal_feats(years, codes, cut):
    """Dated directional summaries of the pre-cutoff signed labels: pooled direction and log count in four
    bins ([t-5,t), [t-10,t-5), [t-20,t-10), < t-20). Zero direction and zero count for empty bins."""
    m = (years < cut) & (codes != 99) & (codes != 0); y = years[m]; c = codes[m].astype(float)
    out = {}
    for name, lo, hi in (('b1', cut - 5, cut), ('b2', cut - 10, cut - 5), ('b3', cut - 20, cut - 10), ('b4', -10 ** 9, cut - 20)):
        w = (y >= lo) & (y < hi); out[f'dir_{name}'] = float(c[w].mean()) if w.sum() else 0.0; out[f'logn_{name}'] = float(np.log1p(w.sum()))
    return out


TEMPORAL_KEYS = ['pooled_direction', 'agreement', 'dir_b1', 'logn_b1', 'dir_b2', 'logn_b2', 'dir_b3', 'logn_b3', 'dir_b4', 'logn_b4']


def shuffled_years(years, codes, cut, rng):
    """Fixed-target perturbation: permute the publication years among the pre-cutoff records (the set of
    labels is unchanged, so p and a are unchanged; only the dated summaries move)."""
    yrs = years.copy(); m = years < cut; idx = np.where(m)[0]
    yrs[idx] = yrs[rng.permutation(idx)]
    return yrs


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); m = LogisticRegression(max_iter=3000, class_weight='balanced').fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


def loco_scores(X, y, g):
    return ac.loco_scores(np.asarray(X, float), np.asarray(y), np.asarray(g))


def metrics(y, s, g, ref=None, label=''):
    """AUROC/AUPRC with claim-clustered CIs; paired differences against a reference score if given."""
    y = np.asarray(y); s = np.asarray(s); g = np.asarray(g)
    d = {'model': label, 'units': int(len(y)), 'events': int(y.sum()), 'positive_claims': int(len(set(g[y == 1]))), 'prevalence': round(float(y.mean()), 4),
         'auroc': round(float(roc_auc_score(y, s)), 4), 'auprc': round(float(average_precision_score(y, s)), 4)}
    lo, hi, _ = nm.ci_metric(y, s, g, 'auroc'); d['auroc_ci'] = nm.fmt_ci(lo, hi)
    lo, hi, _ = nm.ci_metric(y, s, g, 'auprc'); d['auprc_ci'] = nm.fmt_ci(lo, hi)
    if ref is not None:
        for met in ('auroc', 'auprc'):
            delta, ci = ac.paired(y, s, np.asarray(ref), g, met); d[f'delta_vs_margin_{met}'] = delta; d[f'delta_vs_margin_{met}_ci'] = ci
    return d


def registry_row(experiment_id, cohort, d, comparator='margin rule 1-|p_t|', protocol='claim-held-out', prediction_file='', data_version='source_snapshot 2026-09-20'):
    rows = []
    for met in ('auroc', 'auprc'):
        rows.append({'experiment_id': experiment_id, 'data_version': data_version, 'cohort_id': cohort, 'model_id': d['model'], 'training_protocol': protocol,
                     'n_units': d['units'], 'n_events': d['events'], 'n_positive_claims': d['positive_claims'], 'metric_name': met.upper(),
                     'metric_implementation': 'sklearn roc_auc_score / average_precision_score', 'point_estimate': d[met], 'interval': d[f'{met}_ci'],
                     'interval_type': 'claim-clustered percentile bootstrap, 2,000 resamples, seed 3', 'cluster_unit': 'claim',
                     'paired_comparison': f"model minus {comparator}: {d.get(f'delta_vs_margin_{met}', '')} {d.get(f'delta_vs_margin_{met}_ci', '')}".strip(),
                     'prediction_file': prediction_file})
    return rows
