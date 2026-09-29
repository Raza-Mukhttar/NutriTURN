"""Shared infrastructure for the Astra-review workstreams.

Everything reads from astra_reviews/source_snapshot (a copy of the read-only project) and writes only
under astra_reviews/. The frozen protocol code (nutrimature.py) is imported from the snapshot, so the
leave-one-claim-out split, the claim-clustered bootstrap and the metric implementations are the
project's own. The target convention is the paper's final one: sgn(0) = 0.
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
import os, sys, csv, json, math, collections
import numpy as np
ROOT = _NT
SNAP = os.path.join(_NT, 'data', 'protocol_snapshot')
sys.path.insert(0, os.path.join(SNAP, 'scripts'))
import nutrimature as nm            # noqa: E402  (paths resolve inside the snapshot)
import nutrimature_features as nf   # noqa: E402
RESULTS, TABLES, FIGURES, LOGS = [os.path.join(ROOT, d) for d in ('results', 'tables', 'figures', 'logs')]
DATA = os.path.join(ROOT, 'data', 'derived')   # per-analysis derived tables
for _d in (DATA, RESULTS, TABLES, FIGURES, LOGS): os.makedirs(_d, exist_ok=True)
SEED = nm.SEED
COMPACT7 = nm.COMPACT7
CODE = {'PROTECTIVE': -1, 'HARMFUL': 1, 'NULL': 0, 'UNCLEAR': 99}


def sgn0(x): return 0 if x == 0 else (1 if x > 0 else -1)


def ns(kind, space):
    p = os.path.join({'data': DATA, 'results': RESULTS, 'tables': TABLES, 'figures': FIGURES}[kind], space)
    os.makedirs(p, exist_ok=True); return p


# ---------------------------------------------------------------- benchmark units
def load_units(corpus='complete'):
    rows, keys = nm.load(corpus)
    for r in rows:
        r['y_sub'] = int(sgn0(r['post_pooled_direction']) != sgn0(r['pooled_direction']))
    return rows, keys


def prospective(rows, target='y_sub'):
    P = [r for r in rows if r['n_post_t'] >= nm.POST20]
    return P, np.array([int(r[target]) for r in P]), np.array([r['claim_id'] for r in P])


def load_external(which):
    p = nm.EXTERNAL1 if which == 1 else nm.EXTERNAL2
    rows, keys = nm.load(p)
    for r in rows:
        r['y_sub'] = int(sgn0(r['post_pooled_direction']) != sgn0(r['pooled_direction']))
    return rows, keys


# ---------------------------------------------------------------- record streams (file order, as the builder uses)
_STREAMS = {}


def stream(cid):
    """(years, codes, pmids) of a claim's dated labelled records in file order; codes -1/0/+1, 99 = UNCLEAR."""
    if cid not in _STREAMS:
        Dj = {d['pmid']: d['direction'] for d in json.load(open(f'{nm.DIRECTIONS}/{cid}.json'))['directions']}
        recs = [r for r in json.load(open(f'{nm.RECORDS}/{cid}.json'))['records'] if r['pmid'] in Dj and r.get('year')]
        _STREAMS[cid] = (np.array([int(r['year']) for r in recs]), np.array([CODE.get(Dj[r['pmid']], 99) for r in recs]),
                         np.array([r['pmid'] for r in recs]))
    return _STREAMS[cid]


def split(cid, cut):
    y, c, _ = stream(cid); m = y < cut
    return y[m], c[m], y[~m], c[~m]


def pooled(codes):
    s = codes[(codes != 99) & (codes != 0)]
    return float(s.mean()) if len(s) else 0.0


# ---------------------------------------------------------------- agreement definitions (Issue 2)
def counts(codes):
    r = codes[codes != 99]
    return int((r == -1).sum()), int((r == 1).sum()), int((r == 0).sum())


def agreement_defs(codes):
    """A: frozen agg_verdict; B: max share of P/H/NULL among resolved; C: signed-only majority share;
    D: 1 - normalised entropy of the P/H/NULL shares. Also the NULL share and the frozen verdict."""
    nP, nH, nN = counts(codes); n = nP + nH + nN; ns_ = nP + nH
    out = {'n_resolved': n, 'n_signed': ns_, 'null_share': nN / n if n else 0.0}
    v, agA = nf.agg_verdict([int(x) for x in codes if x != 99])
    out['A'] = agA; out['verdict_A'] = v
    sh = np.array([nP, nH, nN], float) / n if n else np.zeros(3)
    out['B'] = float(sh.max()) if n else 0.0
    out['verdict_B'] = int([-1, 1, 0][int(sh.argmax())]) if n else 0
    out['C'] = max(nP, nH) / ns_ if ns_ else 0.0
    out['verdict_C'] = (1 if nH >= nP else -1) if ns_ else 0
    ent = -sum(p * math.log(p) for p in sh if p > 0) if n else 0.0
    out['D'] = 1 - ent / math.log(3) if n else 0.0
    out['verdict_D'] = out['verdict_B']
    return out


def direction_feats(years, codes, cut, defn='A'):
    """The frozen direction_block, with the consensus function swapped for definition `defn`.
    For defn 'A' this reproduces the shipped columns exactly (asserted by the audit script)."""
    res = codes != 99; d_codes = codes[res]; d_years = years[res]
    nn = d_codes != 0
    f = {'n_directional': int(res.sum()), 'n_nonnull': int(nn.sum()), 'null_frac': float((~nn).sum() / max(res.sum(), 1))}
    if nn.sum() < 8:
        f.update(pooled_direction=0.0, agreement=0.0, agreement_recent=0.0, dissent_recent=0.0, agreement_contraction=0.0, verdict=0)
        return f
    v = d_codes[nn]; yr = d_years[nn].astype(float)
    p = int((v > 0).sum()); f['pooled_direction'] = float(v.mean())
    ag = agreement_defs(d_codes); f['agreement'] = ag[defn]; verdict = ag[f'verdict_{defn}']
    f['verdict'] = verdict
    est = verdict if verdict != 0 else (1 if p >= len(v) - p else -1)
    h = len(v) // 2
    ea = max((v[:h] > 0).sum(), h - (v[:h] > 0).sum()) / max(h, 1)
    la = max((v[h:] > 0).sum(), len(v) - h - (v[h:] > 0).sum()) / max(len(v) - h, 1)
    f['agreement_contraction'] = float(la - ea)
    rec_all = d_codes[d_years >= cut - 6]; rec = v[yr >= cut - 6]
    if len(rec) >= 4:
        f['agreement_recent'] = agreement_defs(rec_all)[defn]
        f['dissent_recent'] = float((rec != est).mean())
    else:
        f['agreement_recent'] = f['agreement']; f['dissent_recent'] = 0.0
    return f


def state_of(f):
    return nf.label({'n_directional': f['n_directional'], 'n_nonnull': f['n_nonnull'], 'agreement': f['agreement'],
                     'agreement_recent': f['agreement_recent'], 'dissent_recent': f['dissent_recent'],
                     'agreement_contraction': f['agreement_contraction']})


# ---------------------------------------------------------------- evaluation helpers (frozen protocol code underneath)
def loco(X, y, g):
    return nm.loco_binary(np.asarray(X, float), np.asarray(y), np.asarray(g))


def loco_scores(X, y, g):
    """Same out-of-fold scores as loco() (identical fits) without the training-fold F1 threshold search,
    which dominates run time; for AUROC/AUPRC-only uses such as the null draws."""
    return nm.loco_binary(np.asarray(X, float), np.asarray(y), np.asarray(g), thresh_from_train=False)[0]


def perf(y, s, g, thr=None):
    d = nm.perf(y, s, g)
    if thr is not None and 0 < np.sum(y) < len(y):
        b = nm.binary_metrics(y, s, thr); d.update({k: round(float(b[k]), 4) for k in ('f1', 'precision', 'recall', 'balacc')})
    return d


def paired(y, sa, sb, g, metric='auroc'):
    from sklearn.metrics import roc_auc_score, average_precision_score
    f = {'auroc': roc_auc_score, 'auprc': average_precision_score}[metric]
    lo, hi, _ = nm.ci_paired(y, sa, sb, g, metric)
    return round(float(f(y, sa) - f(y, sb)), 4), nm.fmt_ci(lo, hi)


def claim_level(y, s, g):
    """Claim-level: the claim's score is its maximum over cutoffs; positive if any cutoff is an event."""
    cl = sorted(set(g)); ys = np.array([int(y[g == c].max()) for c in cl]); ss = np.array([float(s[g == c].max()) for c in cl])
    return nm.perf(ys, ss, np.array(cl))


def rate_ci(y, g):
    y = np.asarray(y, float); lo, hi, _ = nm.claim_boot(g, lambda i: float(y[i].mean()))
    return round(float(y.mean()), 4), nm.fmt_ci(lo, hi)


# ---------------------------------------------------------------- output helpers
def save_csv(rows, path):
    rows = [rows] if isinstance(rows, dict) else rows
    keys = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f'  -> {os.path.relpath(path, ROOT)} ({len(rows)} rows)', flush=True)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, 'w'), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f'  -> {os.path.relpath(path, ROOT)}', flush=True)


def save_frame(rows, path_parquet):
    """Write parquet when an engine is available, else CSV with the same stem (recorded in the manifest)."""
    import pandas as pd
    df = pd.DataFrame(rows)
    try:
        df.to_parquet(path_parquet, index=False); out = path_parquet
    except Exception as e:
        out = path_parquet[:-8] + '.csv'; df.to_csv(out, index=False)
        print(f'  (parquet engine unavailable: {type(e).__name__}; wrote CSV instead)', flush=True)
    print(f'  -> {os.path.relpath(out, ROOT)} ({len(df)} rows)', flush=True); return out


def latex_table(rows, cols, path, caption, label, fmt=None):
    """A minimal LaTeX tabular for the appendix; cols = [(key, header)]."""
    fmt = fmt or {}
    lines = ['\\begin{table}[!t]\\centering\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
             '\\resizebox{\\columnwidth}{!}{%', '\\begin{tabular}{@{}l' + 'r' * (len(cols) - 1) + '@{}}', '\\toprule',
             ' & '.join(h for _, h in cols) + ' \\\\', '\\midrule']
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, '')
            if k in fmt: v = fmt[k](v)
            elif isinstance(v, float): v = f'{v:.3f}'
            cells.append(str(v).replace('_', '\\_').replace('%', '\\%'))
        lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}}', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\end{table}']
    os.makedirs(os.path.dirname(path), exist_ok=True); open(path, 'w').write('\n'.join(lines) + '\n')
    print(f'  -> {os.path.relpath(path, ROOT)}', flush=True)


def plt_setup():
    os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, 'logs', 'mpl'))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
    return plt
