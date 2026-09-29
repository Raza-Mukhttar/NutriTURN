"""Shared code for the Revise_V2 fixes (gptpro). Reads the frozen snapshot through astra_common (read-only);
writes only under gptpro/. Provides: stationary posterior-predictive baselines (Beta-Binomial oracle /
deployable / Dirichlet-multinomial), sign-invariant order features relative to the current majority,
offset logistic regression, and the full metric set (ranking, probability, calibration, budget)."""

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
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, expit, logit
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, _os.path.join(_NT, 'scripts', 'lib'))
import astra_common as ac
nm, nf = ac.nm, ac.nf
G = _NT
RES = os.path.join(G, 'results'); os.makedirs(RES, exist_ok=True)
LAST_COMPLETE_YEAR = 2025
MC = 400


def sgn0(x): return 0 if x == 0 else (1 if x > 0 else -1)


# ---------------------------------------------------------------- stationary posterior-predictive baselines
def bb_change_prob(n_plus, n_minus, m, alpha=1.0, rng=None, mc=MC, s0=None):
    """Beta-Binomial posterior predictive: P(sign of a future signed majority of size m differs from the
    current sign s0) with theta ~ Beta(n+ + alpha, n- + alpha). Exact tie convention sgn(0)=0. m may be an int
    (realised count) or an array of draws of the future count (count-predicted: exact tail averaged over the draws).
    s0 = the current sign used by the scored endpoint: the OPERATIONAL sign sgn(p_t) of the frozen builder (0 when
    fewer than 8 signed records precede the cutoff). s0=None gives the raw-majority sign sgn(n+ - n-) (theoretical
    reference, diagnostic only). The posterior always uses the actual counts; only the comparator changes."""
    n = n_plus + n_minus; s0 = sgn0(n_plus - n_minus) if s0 is None else int(s0)
    if np.isscalar(m):
        m = int(max(1, m)); k = np.arange(m + 1)
        # exact beta-binomial pmf
        lp = (gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1) + betaln(k + n_plus + alpha, m - k + n_minus + alpha) - betaln(n_plus + alpha, n_minus + alpha))
        p = np.exp(lp); s1 = np.sign(2 * k - m)
        return float(p[s1 != s0].sum())
    # deployable: exact Beta-Binomial tail for every drawn future count m, averaged over the draws (no sampling of k or theta)
    m = np.maximum(1, np.asarray(m, int)); vals, cnts = np.unique(m, return_counts=True)
    return float(sum(c * bb_change_prob(n_plus, n_minus, int(v), alpha, s0=s0) for v, c in zip(vals, cnts)) / cnts.sum())


def dm_change_prob(n_plus, n_minus, n_null, m_total, rng, alpha=1.0, mc=MC, s0=None):
    """Dirichlet-multinomial sensitivity: theta ~ Dir(n+ + a, n- + a, n0 + a); future m_total records
    (signed + NULL) drawn multinomially; the sign of the signed majority is compared with the current sign;
    a future window with no signed record is a change under sgn(0)=0 when s0 != 0."""
    s0 = sgn0(n_plus - n_minus) if s0 is None else int(s0); m_total = np.maximum(1, np.asarray(m_total, int))
    th = rng.dirichlet([n_plus + alpha, n_minus + alpha, n_null + alpha], size=len(m_total))
    out = np.empty(len(m_total))
    for i in range(len(m_total)):
        c = rng.multinomial(m_total[i], th[i]); out[i] = np.sign(c[0] - c[1]) != s0
    return float(out.mean())


def fit_count_model(log_n_pre, log_rate5, log_m):
    """Deployable future-count model: log m = b0 + b1 log(n_pre_signed) + b2 log(1 + accrual rate of signed
    records over the last five years), fitted by least squares on TRAINING units; returns (coef, resid sd)."""
    X = np.column_stack([np.ones(len(log_n_pre)), log_n_pre, log_rate5]); b, *_ = np.linalg.lstsq(X, log_m, rcond=None)
    resid = log_m - X @ b; return b, float(resid.std() + 1e-6)


def draw_counts(b, sd, log_n_pre, log_rate5, rng, mc=MC):
    mu = b[0] + b[1] * log_n_pre + b[2] * log_rate5; return np.rint(np.exp(mu + sd * rng.standard_normal(mc))).astype(int)


# ---------------------------------------------------------------- sign-invariant order features (relative to the current majority)
def order_feats(years, codes, cut):
    """Features of the pre-cutoff signed sequence expressed relative to the current majority sign, so they are
    invariant to flipping all signed labels of the claim. Returns dict."""
    m = (years < cut) & (codes != 99) & (codes != 0); y = years[m]; c = codes[m].astype(float)
    n = len(c); f = {}
    if n < 4:
        for k in ('dissent_recent_vs_hist', 'recency_weighted_margin', 'cum_margin_slope', 'opp_run_len', 'contraction', 'cusum_max', 'contra_accel', 'log_n_signed'):
            f[k] = 0.0
        f['log_n_signed'] = math.log1p(n); return f
    s0 = np.sign(c.sum()) or 1.0; z = c * s0                      # +1 agrees with current majority, -1 contradicts
    order = np.argsort(y, kind='stable'); z = z[order]; yy = y[order]
    rec = yy >= cut - 6; hist = ~rec
    d_rec = float((z[rec] < 0).mean()) if rec.sum() else 0.0; d_hist = float((z[hist] < 0).mean()) if hist.sum() else 0.0
    f['dissent_recent_vs_hist'] = d_rec - d_hist
    w = np.exp(-(cut - 1 - yy) / 5.0); f['recency_weighted_margin'] = float((w * z).sum() / w.sum())
    cm = np.cumsum(z) / np.arange(1, n + 1); h = n // 2
    f['cum_margin_slope'] = float(cm[-1] - cm[max(h - 1, 0)])
    run = 0; best = 0
    for v in z[::-1]:
        if v < 0: run += 1; best = max(best, run)
        else: break
    f['opp_run_len'] = float(best)                                  # length of the latest run of contradicting labels
    ea = abs(z[:h].mean()) if h else 0.0; la = abs(z[h:].mean()) if n - h else 0.0; f['contraction'] = float(ea - la)
    mean_z = z.mean(); cs = np.cumsum(z - mean_z); f['cusum_max'] = float((cs.max() - cs.min()) / max(n, 1))
    q = max(n // 4, 1); f['contra_accel'] = float((z[-q:] < 0).mean() - (z[-2 * q:-q] < 0).mean()) if n >= 2 * q else 0.0
    f['log_n_signed'] = math.log1p(n)
    return f


ORDER_KEYS = ['dissent_recent_vs_hist', 'recency_weighted_margin', 'cum_margin_slope', 'opp_run_len', 'contraction', 'cusum_max', 'contra_accel']


# ---------------------------------------------------------------- offset logistic regression (stationary probability as offset)
def fit_offset_logistic(X, y, offset, C=1.0):
    """Minimise log loss of sigmoid(offset + Xb + b0) with L2 penalty (1/(2C))||b||^2."""
    n, d = X.shape; wts = np.ones(n)          # unweighted likelihood: the offset is a calibrated probability and the increments are fitted as proper log-odds
    def obj(th):
        b0, b = th[0], th[1:]; eta = offset + X @ b + b0; p = expit(eta)
        ll = -(wts * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum() / n + (b @ b) / (2 * C * n)
        g = (wts * (p - y)) @ np.column_stack([np.ones(n), X]) / n; g[1:] += b / (C * n)
        return ll, g
    r = minimize(obj, np.zeros(d + 1), jac=True, method='L-BFGS-B'); return r.x


def loco_offset(X, y, g, offset, C=1.0):
    """Leave-one-claim-out out-of-fold probabilities for the offset model; X standardised inside each fold."""
    s = np.zeros(len(y))
    for c in np.unique(g):
        te = g == c; tr = ~te
        if len(np.unique(y[tr])) < 2: s[te] = expit(offset[te]); continue
        sc = StandardScaler().fit(X[tr]); th = fit_offset_logistic(sc.transform(X[tr]), y[tr], offset[tr], C)
        s[te] = expit(offset[te] + sc.transform(X[te]) @ th[1:] + th[0])
    return s


# ---------------------------------------------------------------- metrics
def platt_fit(s, y):
    z = logit(np.clip(s, 1e-6, 1 - 1e-6)); m = LogisticRegression(C=1e6, max_iter=2000).fit(z.reshape(-1, 1), y); return m


def platt_apply(m, s): return m.predict_proba(logit(np.clip(s, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]


def loco_calibrated(s, y, g):
    """Training-only Platt calibration under leave-one-claim-out: the calibrator for a held-out claim is fitted on
    the other claims' (uncalibrated) scores."""
    out = np.zeros(len(y))
    for c in np.unique(g):
        te = g == c; tr = ~te
        if len(np.unique(y[tr])) < 2: out[te] = s[te]; continue
        out[te] = platt_apply(platt_fit(s[tr], y[tr]), s[te])
    return out


def ece(p, y, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for k in range(bins):
        m = b == k
        if m.sum(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def calib_slope(p, y):
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    try:
        m = LogisticRegression(C=1e6, max_iter=2000).fit(z.reshape(-1, 1), y); return float(m.coef_[0][0]), float(m.intercept_[0])
    except Exception: return float('nan'), float('nan')


def budget_metrics(s, y, budgets=(0.05, 0.10, 0.20), target_recall=0.8):
    n = len(y); order = np.argsort(-s); ys = y[order]; out = {}
    for b in budgets:
        k = max(1, int(round(b * n))); tp = ys[:k].sum(); out[f'precision@{int(b*100)}'] = float(tp / k); out[f'recall@{int(b*100)}'] = float(tp / max(y.sum(), 1))
    cum = np.cumsum(ys); need = int(np.argmax(cum >= target_recall * y.sum()) + 1) if y.sum() else n
    out[f'share_reviewed_for_recall_{int(target_recall*100)}'] = float(need / n); out[f'workload_reduction_at_recall_{int(target_recall*100)}'] = float(1 - need / n)
    k10 = max(1, int(round(0.10 * n))); out['nnr@10'] = float(k10 / max(ys[:k10].sum(), 1))
    return out


def full_metrics(y, s, p=None):
    """Ranking metrics from s; probability metrics from p (calibrated) or from s if p is None."""
    y = np.asarray(y); s = np.asarray(s); p = np.clip(np.asarray(s if p is None else p), 1e-6, 1 - 1e-6)
    d = {'auroc': float(roc_auc_score(y, s)), 'auprc': float(average_precision_score(y, s)), 'logloss': float(log_loss(y, p)), 'brier': float(brier_score_loss(y, p)), 'ece': ece(p, y)}
    d['cal_slope'], d['cal_intercept'] = calib_slope(p, y); d.update(budget_metrics(s, y)); return d


def paired_ci(y, sa, sb, g, fn, n=2000, seed=3):
    lo, hi, _ = nm.claim_boot(g, lambda i: fn(y[i], sa[i]) - fn(y[i], sb[i]), n=n, seed=seed); return round(float(fn(y, sa) - fn(y, sb)), 4), nm.fmt_ci(lo, hi)


def ci(y, s, g, fn, n=2000, seed=3):
    lo, hi, _ = nm.claim_boot(g, lambda i: fn(y[i], s[i]), n=n, seed=seed); return nm.fmt_ci(lo, hi)


AUROC = lambda y, s: roc_auc_score(y, s); AUPRC = lambda y, s: average_precision_score(y, s)
LOGLOSS = lambda y, p: log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)); BRIER = lambda y, p: brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6))


def save_json(o, name):
    p = os.path.join(RES, name); os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(o, open(p, 'w'), indent=1, default=float); print('  ->', os.path.relpath(p, G), flush=True)


def save_csv(rows, name):
    import csv
    p = os.path.join(RES, name); os.makedirs(os.path.dirname(p), exist_ok=True); keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, 'w', newline='') as f: w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print('  ->', os.path.relpath(p, G), f'({len(rows)} rows)', flush=True)


def source_units():
    rows, keys = ac.load_units(); P, y, g = ac.prospective(rows); n = len(P)
    p = np.array([r['pooled_direction'] for r in P]); a = np.array([r['agreement'] for r in P])
    npl, nmi, nnull, m_signed, m_total, rate5, order = [], [], [], [], [], [], []
    for r in P:
        yrs, codes, _ = ac.stream(r['claim_id']); t = int(r['cutoff_year']); pre = (yrs < t) & (codes != 99); post = (yrs >= t) & (codes != 99)
        npl.append(int((codes[pre] == 1).sum())); nmi.append(int((codes[pre] == -1).sum())); nnull.append(int((codes[pre] == 0).sum()))
        m_signed.append(int(((codes[post] != 0)).sum())); m_total.append(int(post.sum()))
        rate5.append(int(((yrs >= t - 5) & (yrs < t) & (codes != 99) & (codes != 0)).sum()) / 5.0); order.append(order_feats(yrs, codes, t))
    return dict(P=P, y=y, g=g, p=p, a=a, npl=np.array(npl), nmi=np.array(nmi), nnull=np.array(nnull), m_signed=np.array(m_signed), m_total=np.array(m_total), rate5=np.array(rate5), order=order,
                X7=nm.X_of(P, nm.COMPACT7))
