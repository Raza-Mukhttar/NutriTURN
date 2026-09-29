"""Shared layer for the newfab_exp reanalyses.

Reads the frozen protocol and the archived snapshot read-only; every write goes under
newfab_exp/. Nothing outside this folder is modified.

Annotations
  primary    Llama-3.1-8B-Instruct, first-token read-out      (the deployed pipeline)
  second     Qwen2.5-7B-Instruct, full-string read-out        (the second pipeline)
  consensus  disagreements between the two become UNCLEAR
Extra annotations appear once experiment E8 has run (cells A', B, C).

Endpoint contract (frozen builder, reproduced exactly)
  p_t = (h-l)/(h+l) when h+l >= 8, else 0;  s_t = sgn(p_t)
  p+_t = raw signed mean of the post-cutoff signed sample, 0 when that sample is empty
  R    = 1[ sgn(p+_t) != s_t ]
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
import os, sys, json, glob, csv, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = _NT
DATA = os.path.join(BASE, 'data')
RESULTS = os.path.join(BASE, 'results')
LOGS = os.path.join(BASE, 'logs')
FIGURES = os.path.join(BASE, 'figures')
ANNOT = os.path.join(BASE, 'data', 'llm_runs', 'controlled_cells')
for d in (DATA, RESULTS, LOGS, FIGURES, ANNOT):
    os.makedirs(d, exist_ok=True)

# ---- read-only sources
GPTPRO = _NT
ASTRA = _NT
SNAPD = os.path.join(_NT, 'data', 'protocol_snapshot')
REEXTRACT = os.path.join(GPTPRO, 'results', 'reextraction')

CODE = {'PROTECTIVE': -1, 'HARMFUL': 1, 'NULL': 0, 'UNCLEAR': 99}
LABELS = ['PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR']
SEED = 3
BOOT = 2000
SPARSE_GATE = 8            # signed pre-cutoff records needed before p_t is non-zero
MIN_FUTURE = 20            # later records needed for future-eligibility


def sgn0(x):
    return 0 if x == 0 else (1 if x > 0 else -1)


# ---------------------------------------------------------------- inputs
def build_associations():
    """claim_id, pmid, year, label_primary, label_second -> data/associations.csv"""
    prim = {}
    for p in sorted(glob.glob(os.path.join(SNAPD, 'directions', '*.json'))):
        d = json.load(open(p))
        for r in d['directions']:
            prim[(d['claim_id'], str(r['pmid']))] = (int(r['year']), r['direction'])
    sec = {}
    for p in sorted(glob.glob(os.path.join(REEXTRACT, '*.jsonl'))):
        for line in open(p):
            r = json.loads(line)
            sec[(r['claim_id'], str(r['pmid']))] = r['qwen_label_full']
    rows = []
    for (cid, pmid), (yr, lp) in prim.items():
        rows.append({'claim_id': cid, 'pmid': pmid, 'year': yr,
                     'label_primary': lp, 'label_second': sec.get((cid, pmid), '')})
    rows.sort(key=lambda r: (r['claim_id'], r['year'], r['pmid']))
    out = os.path.join(DATA, 'associations.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, ['claim_id', 'pmid', 'year', 'label_primary', 'label_second'])
        w.writeheader(); w.writerows(rows)
    return out, len(rows)


def build_claims():
    q = json.load(open(os.path.join(SNAPD, 'claim_queries.json')))
    out = os.path.join(DATA, 'claims.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, ['claim_id', 'exposure', 'outcome', 'query'])
        w.writeheader()
        for cid in sorted(q):
            w.writerow({'claim_id': cid, 'exposure': q[cid]['exposure'],
                        'outcome': q[cid]['outcome'], 'query': q[cid]['query']})
    return out, len(q)


def build_units():
    """The archived cutoff grid, restricted to future-eligible units, from the frozen benchmark."""
    src = os.path.join(SNAPD, 'benchmark', 'uncapped_benchmark.csv')
    rows = list(csv.DictReader(open(src)))
    keep = [r for r in rows if int(float(r['n_post_t'])) >= MIN_FUTURE]
    out = os.path.join(DATA, 'units.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, ['claim_id', 'cutoff', 'n_pre_t', 'n_post_t'])
        w.writeheader()
        for r in keep:
            w.writerow({'claim_id': r['claim_id'], 'cutoff': int(float(r['cutoff_year'])),
                        'n_pre_t': int(float(r['n_pre_t'])), 'n_post_t': int(float(r['n_post_t']))})
    return out, len(keep), len(rows)


# ---------------------------------------------------------------- annotation streams
_ASSOC = None


def assoc():
    global _ASSOC
    if _ASSOC is None:
        p = os.path.join(DATA, 'associations.csv')
        if not os.path.exists(p):
            build_associations()
        rows = list(csv.DictReader(open(p)))
        for r in rows:
            r['year'] = int(r['year'])
        _ASSOC = rows
    return _ASSOC


def consensus_label(a, b):
    return a if a == b else 'UNCLEAR'


def label_column(annotation):
    """Return {(claim_id, pmid): LABEL} for a named annotation."""
    A = assoc()
    if annotation == 'primary':
        return {(r['claim_id'], r['pmid']): r['label_primary'] for r in A}
    if annotation == 'second':
        return {(r['claim_id'], r['pmid']): r['label_second'] for r in A}
    if annotation == 'consensus':
        return {(r['claim_id'], r['pmid']): consensus_label(r['label_primary'], r['label_second']) for r in A}
    # an E8 cell: annotations/<name>.jsonl with claim_id, pmid, label
    p = os.path.join(ANNOT, annotation + '.jsonl')
    if not os.path.exists(p):
        raise SystemExit(f'unknown annotation {annotation!r} (no {p})')
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[(r['claim_id'], str(r['pmid']))] = r['label']
    return out


def streams(annotation):
    """{claim_id: (years array, codes array)} in archived file order."""
    lab = label_column(annotation)
    per = {}
    for r in assoc():
        k = (r['claim_id'], r['pmid'])
        L = lab.get(k, '')
        per.setdefault(r['claim_id'], []).append((r['year'], CODE.get(L, 99)))
    return {c: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for c, v in per.items()}


# ---------------------------------------------------------------- endpoint
def unit_table(annotation):
    """Rebuild inputs and endpoint for every future-eligible unit under one annotation.

    Future-eligibility is fixed by the archived record counts (units.csv), so the evaluation
    population is the same 1,012 units for every annotation; only labels change.
    """
    up = os.path.join(DATA, 'units.csv')
    if not os.path.exists(up):
        build_units()
    units = list(csv.DictReader(open(up)))
    S = streams(annotation)
    rows = []
    for u in units:
        cid, t = u['claim_id'], int(u['cutoff'])
        yrs, codes = S[cid]
        pre = (yrs < t) & (codes != 99)
        post = (yrs >= t) & (codes != 99)
        h = int((codes[pre] == 1).sum()); l = int((codes[pre] == -1).sum())
        n0 = int((codes[pre] == 0).sum())
        n = h + l
        r_t = (h - l) / n if n else 0.0
        p_t = r_t if n >= SPARSE_GATE else 0.0
        s_t = sgn0(p_t)
        # agreement, shipped definition: the sparse gate fires FIRST (frozen direction_block
        # returns pooled_direction = agreement = 0 when fewer than 8 signed records are present),
        # and only then the NULL-dominant / signed-majority branch applies.
        resolved = h + l + n0
        null_share = n0 / resolved if resolved else 0.0
        if n < SPARSE_GATE:
            a_t = 0.0
        else:
            a_t = null_share if null_share >= 0.5 else (1 + abs(p_t)) / 2
        ph = int((codes[post] == 1).sum()); pl = int((codes[post] == -1).sum())
        m = ph + pl
        p_post = (ph - pl) / m if m else 0.0
        rows.append(dict(claim_id=cid, cutoff=t, h=h, l=l, n0=n0, n_signed=n,
                         r_t=r_t, p=p_t, s=s_t, a=a_t, null_share=null_share,
                         m_signed=m, p_post=p_post,
                         y=int(sgn0(p_post) != s_t),
                         sparse=int(n < SPARSE_GATE)))
    return rows


# ---------------------------------------------------------------- models
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score


def loco_two_var(rows, class_blind=False, subset=None, features=('p', 'a')):
    """Leave-one-claim-out out-of-fold scores of the class-balanced L2 logistic score.

    subset: optional boolean mask; when given, both fitting and scoring use only those rows
            (this is the 'refit on the restricted population' that E3 and E5 require).
    """
    idx = [i for i in range(len(rows)) if (subset is None or subset[i])]
    X = np.array([[abs(rows[i]['p']) if (class_blind and f == 'p') else rows[i][f] for f in features] for i in idx], float)
    y = np.array([rows[i]['y'] for i in idx])
    g = np.array([rows[i]['claim_id'] for i in idx])
    s = np.full(len(idx), np.nan)
    for c in np.unique(g):
        te = g == c; tr = ~te
        if len(np.unique(y[tr])) < 2:
            s[te] = y[tr].mean() if tr.any() else 0.0
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
        m.fit(sc.transform(X[tr]), y[tr])
        s[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return np.array(idx), s, y, g


def margin_score(rows, idx):
    return np.array([1 - abs(rows[i]['p']) for i in idx])


# ---------------------------------------------------------------- statistics
def claim_boot_pairs(y, g, n=BOOT, seed=SEED):
    """Shared claim-clustered resamples: yields index arrays."""
    rng = np.random.default_rng(seed)
    claims = np.unique(g)
    by = {c: np.where(g == c)[0] for c in claims}
    out = []
    for _ in range(n):
        pick = rng.choice(claims, size=len(claims), replace=True)
        out.append(np.concatenate([by[c] for c in pick]))
    return out


def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float('nan')


def auprc(y, s):
    return float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float('nan')


def ci(y, s, g, fn=auroc, n=BOOT, seed=SEED):
    vals = []
    for ix in claim_boot_pairs(y, g, n, seed):
        if len(np.unique(y[ix])) < 2:
            continue
        vals.append(fn(y[ix], s[ix]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def paired_ci(y, sa, sb, g, fn=auroc, n=BOOT, seed=SEED):
    """Interval for fn(sa) - fn(sb) on shared resamples."""
    vals = []
    for ix in claim_boot_pairs(y, g, n, seed):
        if len(np.unique(y[ix])) < 2:
            continue
        vals.append(fn(y[ix], sa[ix]) - fn(y[ix], sb[ix]))
    v = np.array(vals)
    return float(fn(y, sa) - fn(y, sb)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def established(lo, hi):
    return (lo > 0) or (hi < 0)


def fmt_ci(lo, hi, d=3):
    return f'[{lo:.{d}f}, {hi:.{d}f}]'


def save_json(obj, name):
    p = os.path.join(RESULTS, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, 'w'), indent=1, default=float)
    return p


def save_csv(rows, name):
    p = os.path.join(RESULTS, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if rows:
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return p


# ---------------------------------------------------------------- calendar cohort
ORIGINS = (2004, 2007, 2010, 2013, 2016, 2019, 2022)
HORIZONS = (3, 5, 10)
LAST_COMPLETE_YEAR = 2025
CAND_MIN_RECORDS = 40       # records (any label) before the origin
ACCRUAL_MIN = 10            # signed records inside the window


def build_cohort(annotation='primary', include_incomplete=False):
    """Origin-defined cohort: one row per (claim, origin, horizon) candidate.

    Columns: claim_id, origin, horizon, h, l, n0, n_signed, p, a, null_share, accrual_5y,
             span_years, n_pre, accrued, m_signed, y (defined only when accrued).
    Windows ending after 2025 are excluded unless include_incomplete.
    """
    S = streams(annotation)
    rows = []
    for H in HORIZONS:
        for Y in ORIGINS:
            if not include_incomplete and Y + H - 1 > LAST_COMPLETE_YEAR:
                continue
            for cid, (yrs, codes) in S.items():
                if (yrs < Y).sum() < CAND_MIN_RECORDS:
                    continue
                pre = (yrs < Y) & (codes != 99)
                h = int((codes[pre] == 1).sum()); l = int((codes[pre] == -1).sum())
                n0 = int((codes[pre] == 0).sum()); n = h + l
                p = (h - l) / n if n >= SPARSE_GATE else 0.0
                resolved = h + l + n0
                ns_ = n0 / resolved if resolved else 0.0
                a = 0.0 if n < SPARSE_GATE else (ns_ if ns_ >= 0.5 else (1 + abs(p)) / 2)
                rec5 = int(((yrs >= Y - 5) & (yrs < Y) & (codes != 99) & (codes != 0)).sum()) / 5.0
                pv = yrs[yrs < Y]
                w = (yrs >= Y) & (yrs < Y + H) & (codes != 99)
                ph = int((codes[w] == 1).sum()); pl = int((codes[w] == -1).sum()); m = ph + pl
                accrued = m >= ACCRUAL_MIN
                rows.append(dict(claim_id=cid, origin=Y, horizon=H, h=h, l=l, n0=n0, n_signed=n,
                                 p=p, a=a, null_share=ns_, accrual_5y=rec5,
                                 span_years=int(pv.max() - pv.min()) if len(pv) else 0,
                                 n_pre=int((yrs < Y).sum()),
                                 accrued=int(accrued), m_signed=m,
                                 y=int(sgn0((ph - pl) / m) != sgn0(p)) if accrued else -1))
    return rows


def cohort_unit(cid, t, H, yrs, codes):
    """One cohort unit at an arbitrary cutoff, with the accrual indicator (never excluded)."""
    m = yrs < t
    if m.sum() < CAND_MIN_RECORDS:
        return None
    pre = (yrs < t) & (codes != 99)
    h = int((codes[pre] == 1).sum()); l = int((codes[pre] == -1).sum())
    n0 = int((codes[pre] == 0).sum()); n = h + l
    p = (h - l) / n if n >= SPARSE_GATE else 0.0
    resolved = h + l + n0
    ns_ = n0 / resolved if resolved else 0.0
    a = 0.0 if n < SPARSE_GATE else (ns_ if ns_ >= 0.5 else (1 + abs(p)) / 2)
    w = (yrs >= t) & (yrs < t + H) & (codes != 99)
    ph = int((codes[w] == 1).sum()); pl = int((codes[w] == -1).sum()); m_s = ph + pl
    acc = m_s >= ACCRUAL_MIN
    pv = yrs[yrs < t]
    return dict(claim_id=cid, origin=t, horizon=H, h=h, l=l, n0=n0, n_signed=n, p=p, a=a,
                null_share=ns_,
                accrual_5y=int(((yrs >= t - 5) & (yrs < t) & (codes != 99) & (codes != 0)).sum()) / 5.0,
                span_years=int(pv.max() - pv.min()) if len(pv) else 0,
                n_pre=int(m.sum()), accrued=int(acc), m_signed=m_s,
                y=int(sgn0((ph - pl) / m_s) != sgn0(p)) if acc else -1)


TRAIN_STEP = 2      # the training cutoff grid is two-yearly, as in the published protocol
TRAIN_MIN = 30      # minimum training units for a fit


def cohort_training_units(H, Y, S, drop_claim=None):
    """Accrued units on a two-year cutoff grid whose horizon outcome is observable by the origin."""
    out = []
    for cid, (yrs, codes) in S.items():
        if drop_claim is not None and cid == drop_claim:
            continue
        for t in range(1975 + (Y % TRAIN_STEP), Y - H + 1, TRAIN_STEP):
            u = cohort_unit(cid, t, H, yrs, codes)
            if u and u['accrued']:
                out.append(u)
    return out


def cohort_loco_two_var(rows, H, class_blind=False, features=('p', 'a'), annotation='primary',
                        extra=None):
    """Per-origin, claim-disjoint fit on the two-year training grid; score the accrued test units.

    extra: optional {(claim_id, origin): {feature: value}} supplying additional features (used by
           the text-reading forecasters), which must also be available for the training units.
    """
    S = streams(annotation)
    d = [r for r in rows if r['horizon'] == H]
    out_s, out_y, out_g, out_o = [], [], [], []
    for Y in sorted({r['origin'] for r in d}):
        te = [r for r in d if r['origin'] == Y and r['accrued'] == 1]
        if not te:
            continue
        base = cohort_training_units(H, Y, S)
        for r in te:
            tr = [q for q in base if q['claim_id'] != r['claim_id']]
            yt = np.array([q['y'] for q in tr])
            if len(tr) < TRAIN_MIN or len(np.unique(yt)) < 2:
                out_s.append(0.5)
            else:
                def X(rows_):
                    return np.array([[abs(q['p']) if (class_blind and f == 'p') else q[f]
                                      for f in features] for q in rows_], float)
                sc = StandardScaler().fit(X(tr))
                m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
                m.fit(sc.transform(X(tr)), yt)
                out_s.append(float(m.predict_proba(sc.transform(X([r])))[0, 1]))
            out_y.append(r['y']); out_g.append(r['claim_id']); out_o.append(Y)
    return np.array(out_s), np.array(out_y), np.array(out_g), np.array(out_o)


def build_cohort_accrual5(rows):
    """Signed records per year over the five years before each unit's cutoff (for the count model)."""
    S = streams('primary')
    out = []
    for r in rows:
        yrs, codes = S[r['claim_id']]; t = r['cutoff']
        out.append(int(((yrs >= t - 5) & (yrs < t) & (codes != 99) & (codes != 0)).sum()) / 5.0)
    return out
