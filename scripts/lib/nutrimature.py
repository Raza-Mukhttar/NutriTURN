"""NutriMATURE system library: paths, the frozen method, evaluation protocol and plotting style.

Every experiment script imports this module, so all of them use the same leave-one-claim-out split, the same
claim-clustered bootstrap and one metric implementation. The evaluation functions are the frozen NutriMATURE protocol
code, unchanged. Paths are relative to the FINAL_NutriMature folder, so the folder can be moved anywhere.
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
import os, csv, json, hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, cohen_kappa_score,
                             matthews_corrcoef, brier_score_loss, log_loss, confusion_matrix)

# ------------------------------------------------------------------ folders
ROOT = _NT
DATA, TABLES, FIGURES = f'{ROOT}/data/protocol_snapshot', f'{ROOT}/tables', f'{ROOT}/figures'
RECORDS, DIRECTIONS = f'{DATA}/records', f'{DATA}/directions'
BENCHMARK = f'{DATA}/benchmark/uncapped_benchmark.csv'
SYSTEMATIC = f'{DATA}/benchmark/uncapped_benchmark_systematic_only.csv'
HAND_ADDED = f'{DATA}/benchmark/hand_added_claims.json'
QUERIES = f'{DATA}/claim_queries.json'
EXTERNAL1 = f'{DATA}/external/external1_benchmark.csv'
EXTERNAL2 = f'{DATA}/external/external2_benchmark.csv'
EXTERNAL_ENDPOINT_UNITS = f'{DATA}/external/external_numeric_endpoint_units.csv'
CORPORA = {'complete': BENCHMARK, 'systematic': SYSTEMATIC}
for _d in (TABLES, FIGURES): os.makedirs(_d, exist_ok=True)

# ------------------------------------------------------------------ frozen method (fixed before any external evaluation)
COMPACT7 = ['pooled_direction', 'agreement', 'n_directional', 'n_pre_t', 'val_mean', 'val_sd', 'n_rr']
HP = {'C': 1.0, 'penalty': 'l2', 'class_weight': 'balanced'}      # StandardScaler + LogisticRegression
ID_COLS = ('claim_id', 'pmids_pre', 'maturity_state', 'cutoff_year', 'year_min', 'year_max')
POST_PREFIX = ('post_', 'l1_', 'l2_', 'l3_', 'n_post_t')          # post-cutoff columns: never model inputs
RULE = ['agreement', 'agreement_recent', 'dissent_recent', 'agreement_contraction']   # the 4 operational-state rule inputs
GATE = ['n_directional', 'n_nonnull']                                                 # the 2 unassessable-gate inputs
DIRECTION = ['pooled_direction', 'agreement', 'agreement_recent', 'agreement_contraction', 'direction_trajectory',
             'dissent_recent', 'null_frac', 'n_directional', 'n_nonnull', 'consensus_verdict', 'has_consensus']
POST20 = 20                                                        # prospective sample: >= 20 post-cutoff records
N_BOOT, SEED = 2000, 3                                             # claim-clustered percentile bootstrap
TARGET = 'l2_sign_change'                                          # later change of the categorical evidence direction


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ frozen protocol functions (unchanged)
def load(corpus):
    """Rows as dicts with floats; returns rows, and the 55-column predictive key list."""
    p = CORPORA[corpus] if corpus in CORPORA else corpus
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k, v in list(r.items()):
            if k not in ('claim_id', 'pmids_pre', 'maturity_state'):
                try: r[k] = float(v)
                except Exception: r[k] = 0.0
    keys = [k for k in rows[0] if k not in ID_COLS and not k.startswith(POST_PREFIX)]
    return rows, keys


def X_of(rows, keys):
    return np.nan_to_num(np.array([[r[k] for k in keys] for r in rows], float), posinf=0, neginf=0)


def prospective_mask(rows, min_post=POST20):
    return np.array([r['n_post_t'] >= min_post for r in rows])


def prospective(rows, min_post=POST20):
    m = prospective_mask(rows, min_post)
    R = [r for r, k in zip(rows, m) if k]
    return R, np.array([int(r[TARGET]) for r in R]), np.array([r['claim_id'] for r in R])


def loco_binary(X, y, g, model_factory=None, thresh_from_train=True, seed=0):
    """Out-of-fold P(y=1) under leave-one-claim-out. Threshold for hard predictions is chosen on the
    TRAINING fold only (the F1-maximising threshold on in-fold predictions), never on the held-out claim."""
    s = np.zeros(len(y)); thr = np.zeros(len(y))
    for u in np.unique(g):
        te = g == u; tr = ~te
        if len(np.unique(y[tr])) < 2: s[te] = float(y[tr].mean()); thr[te] = 0.5; continue
        sc = StandardScaler().fit(X[tr]); Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        m = (model_factory or (lambda: LogisticRegression(max_iter=3000, class_weight='balanced')))()
        m.fit(Xtr, y[tr]); s[te] = m.predict_proba(Xte)[:, 1]
        if thresh_from_train:
            ptr = m.predict_proba(Xtr)[:, 1]; cands = np.unique(np.round(ptr, 3))
            f1s = [f1_score(y[tr], (ptr >= c).astype(int), zero_division=0) for c in cands]
            thr[te] = cands[int(np.argmax(f1s))] if len(cands) else 0.5
        else: thr[te] = 0.5
    return s, thr


def loco_multiclass(X, y, g, model_factory=None):
    p = np.empty(len(y), object); conf = np.zeros(len(y))
    for u in np.unique(g):
        te = g == u; tr = ~te
        if len(np.unique(y[tr])) < 2: p[te] = y[tr][0]; conf[te] = 1 / 3; continue
        sc = StandardScaler().fit(X[tr])
        m = (model_factory or (lambda: LogisticRegression(max_iter=3000, class_weight='balanced')))()
        m.fit(sc.transform(X[tr]), y[tr]); pr = m.predict_proba(sc.transform(X[te]))
        p[te] = m.classes_[pr.argmax(1)]; conf[te] = pr.max(1)
    return p, conf


def binary_metrics(y, s, thr=None):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    out = {'n': int(len(y)), 'events': int(y.sum()), 'base_rate': float(y.mean()) if len(y) else float('nan')}
    if y.sum() == 0 or y.sum() == len(y):
        out.update({k: float('nan') for k in ('auroc', 'auprc', 'brier', 'nll')}); return out
    out['auroc'] = float(roc_auc_score(y, s)); out['auprc'] = float(average_precision_score(y, s))
    sc = np.clip(s, 1e-6, 1 - 1e-6)
    out['brier'] = float(brier_score_loss(y, sc)); out['nll'] = float(log_loss(y, sc))
    if thr is not None:
        pred = (s >= (thr if np.ndim(thr) else np.full(len(s), thr))).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        out.update({'precision': tp / (tp + fp) if tp + fp else float('nan'), 'recall': tp / (tp + fn) if tp + fn else float('nan'),
                    'specificity': tn / (tn + fp) if tn + fp else float('nan'), 'flagged': tp + fp})
        out['sensitivity'] = out['recall']
        out['f1'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float('nan')
        out['balacc'] = float(np.nanmean([out['recall'], out['specificity']]))
    return out


def multiclass_metrics(y, p, labels=('stable', 'still_forming', 'unstable')):
    y = np.asarray(y); p = np.asarray(p); labs = [l for l in labels if (y == l).any()]
    rec = {l: float((p[y == l] == l).mean()) for l in labs}
    prec = {l: float((y[p == l] == l).mean()) if (p == l).any() else float('nan') for l in labs}
    f1 = {l: (2 * prec[l] * rec[l] / (prec[l] + rec[l]) if prec[l] == prec[l] and prec[l] + rec[l] > 0 else 0.0) for l in labs}
    lab_all = sorted(set(y) | set(p))
    return {'n': int(len(y)), 'balacc': float(np.mean(list(rec.values()))), 'macro_f1': float(f1_score(y, p, average='macro', zero_division=0)),
            'kappa': float(cohen_kappa_score(y, p, labels=lab_all)),
            'mcc': float(matthews_corrcoef([lab_all.index(a) for a in y], [lab_all.index(a) for a in p])),
            'recall': rec, 'precision': prec, 'f1': f1, 'support': {l: int((y == l).sum()) for l in labs},
            'confusion': confusion_matrix(y, p, labels=lab_all).tolist(), 'confusion_labels': lab_all}


def calibration(y, s, bins=10):
    y = np.asarray(y).astype(int); s = np.clip(np.asarray(s, float), 1e-6, 1 - 1e-6)
    edges = np.linspace(0, 1, bins + 1); ece = 0.0; rel = []
    for i in range(bins):
        m = (s > edges[i]) & (s <= edges[i + 1]) if i else (s >= edges[i]) & (s <= edges[i + 1])
        if m.any(): ece += m.mean() * abs(y[m].mean() - s[m].mean()); rel.append((float(s[m].mean()), float(y[m].mean()), int(m.sum())))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(s / (1 - s)).reshape(-1, 1), y) if len(np.unique(y)) > 1 else None
    return {'ece': float(ece), 'brier': float(brier_score_loss(y, s)), 'nll': float(log_loss(y, s)),
            'slope': float(lr.coef_[0][0]) if lr else float('nan'), 'intercept': float(lr.intercept_[0]) if lr else float('nan'),
            'reliability': rel}


def claim_boot(g, fn, n=N_BOOT, seed=SEED):
    """Claim-clustered percentile bootstrap of fn(idx). fn must accept an index array."""
    g = np.asarray(g); rng = np.random.default_rng(seed); cl = np.unique(g)
    idx = {c: np.where(g == c)[0] for c in cl}; vals = []
    for _ in range(n):
        ii = np.concatenate([idx[c] for c in rng.choice(cl, len(cl), replace=True)])
        try:
            v = fn(ii)
            if v == v: vals.append(v)
        except Exception: pass
    if not vals: return float('nan'), float('nan'), 0
    v = np.sort(vals); return float(v[int(.025 * len(v))]), float(v[int(.975 * len(v)) - 1]), len(v)


def ci_metric(y, s, g, metric='auroc', n=N_BOOT, seed=SEED):
    y = np.asarray(y); s = np.asarray(s)
    f = {'auroc': lambda i: roc_auc_score(y[i], s[i]), 'auprc': lambda i: average_precision_score(y[i], s[i]),
         'brier': lambda i: brier_score_loss(y[i], np.clip(s[i], 1e-6, 1 - 1e-6))}[metric]
    return claim_boot(g, f, n, seed)


def ci_paired(y, sa, sb, g, metric='auroc', n=N_BOOT, seed=SEED):
    """Paired difference a - b on the same claim resamples."""
    y = np.asarray(y); sa = np.asarray(sa); sb = np.asarray(sb)
    f = {'auroc': lambda i: roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]),
         'auprc': lambda i: average_precision_score(y[i], sa[i]) - average_precision_score(y[i], sb[i]),
         'brier': lambda i: brier_score_loss(y[i], np.clip(sa[i], 1e-6, 1 - 1e-6)) - brier_score_loss(y[i], np.clip(sb[i], 1e-6, 1 - 1e-6))}[metric]
    return claim_boot(g, f, n, seed)


def fmt_ci(lo, hi, nd=3):
    if lo != lo or hi != hi: return 'NA'
    return f'[{lo:.{nd}f}, {hi:.{nd}f}]'


def perf(y, s, g):
    """Standard performance summary: sample counts, AUROC and AUPRC with claim-clustered 95% intervals, Brier."""
    y = np.asarray(y); s = np.asarray(s); g = np.asarray(g)
    d = {'units': len(y), 'claims': len(set(g)), 'events': int(y.sum()), 'positive_claims': len(set(g[y == 1])), 'base_rate': round(float(y.mean()), 4)}
    if 0 < y.sum() < len(y):
        b = binary_metrics(y, s); a = ci_metric(y, s, g, 'auroc'); p = ci_metric(y, s, g, 'auprc')
        d.update({'auroc': round(b['auroc'], 4), 'auroc_ci': fmt_ci(*a[:2]), 'auprc': round(b['auprc'], 4), 'auprc_ci': fmt_ci(*p[:2]), 'brier': round(b['brier'], 4)})
    else:
        d.update({'auroc': 'NA', 'auroc_ci': 'NA (single class)', 'auprc': 'NA', 'auprc_ci': 'NA', 'brier': 'NA'})
    return d


# ------------------------------------------------------------------ outputs
def save_table(rows, name, meta=None):
    """Write tables/<name>.csv and tables/<name>.json (rows + meta)."""
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(f'{TABLES}/{name}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow({k: r.get(k, '') for k in keys})
    json.dump({'meta': meta or {}, 'rows': rows}, open(f'{TABLES}/{name}.json', 'w'), indent=1, default=float)


def save_preds(name, ids, y, s, extra=None):
    """Per-unit out-of-fold scores: tables/<name>.csv."""
    with open(f'{TABLES}/{name}.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['claim_id', 'cutoff_year', 'y', 'score'] + (list(extra) if extra else []))
        for i, (c, t) in enumerate(ids):
            w.writerow([c, int(t), int(y[i]), float(s[i])] + ([extra[k][i] for k in extra] if extra else []))


def table(name):
    return json.load(open(f'{TABLES}/{name}.json'))


# ------------------------------------------------------------------ record streams (records + direction labels per claim)
DIRCODE = {'PROTECTIVE': -1, 'HARMFUL': 1, 'NULL': 0, 'UNCLEAR': 99}


def stream(cid):
    """Publication years and direction codes (-1, 0, +1; 99 = UNCLEAR) of a claim's dated, labelled records."""
    Dj = {d['pmid']: d['direction'] for d in json.load(open(f'{DIRECTIONS}/{cid}.json'))['directions']}
    recs = [r for r in json.load(open(f'{RECORDS}/{cid}.json'))['records'] if r['pmid'] in Dj and r.get('year')]
    return np.array([int(r['year']) for r in recs]), np.array([DIRCODE.get(Dj[r['pmid']], 99) for r in recs])


# ------------------------------------------------------------------ plotting style (validated categorical palette, light surface)
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'   # categorical slots 1-5


def plt_setup():
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.expanduser('~'), '.cache', 'matplotlib'))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'figure.facecolor': SURF,
                         'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'legend.frameon': False})
    return plt


def style(ax, grid='x'):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8); ax.set_axisbelow(True)
    if grid: ax.grid(True, axis=grid, color=GRID, lw=0.6)


def save_fig(fig, name):
    fig.savefig(f'{FIGURES}/{name}.png', dpi=200, facecolor=SURF, bbox_inches='tight')
    import matplotlib.pyplot as plt; plt.close(fig)
    print(f'figure written: figures/{name}.png', flush=True)


def parse_ci(s):
    import re
    m = re.findall(r'-?\d+(?:\.\d+)?', str(s)); return (float(m[0]), float(m[1])) if len(m) >= 2 else (float('nan'), float('nan'))
