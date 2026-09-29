"""NutriMATURE evidence representation: the frozen pre-cutoff feature builder and operational-state label rule.

build_row() and its helpers compute the 55 pre-cutoff variables (plus identifiers and the post-cutoff evaluation
fields) for one claim at one cutoff year. label() is the operational three-state rule. Both are the frozen NutriMATURE
code, unchanged; build_claim_rows()/labelled() reproduce the benchmark driver exactly (including the string round trip
of the original CSV pipeline), so 03_build_benchmark.py rebuilds the shipped benchmark byte for byte.
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
import re, math, csv
import numpy as np

DIRMAP = {'PROTECTIVE': -1, 'HARMFUL': 1, 'NULL': 0, 'UNCLEAR': None}
NUMV = r'\b(?:aOR|aHR|aRR|OR|RR|HR|SMD|WMD|MD)\b[\s:=\]\[()]{0,4}(-?\d+\.\d+)'
CIV  = r'(?i)95\s*%\s*(?:ci|confidence interval)[^0-9\-]{0,14}(-?\d+\.\d+)\s*(?:to|[-–,])\s*(-?\d+\.\d+)'
SYN  = r'(?i)\b(meta-analys\w+|systematic review|pooled analys\w+)\b'
RCT  = r'(?i)\b(randomi[sz]ed|double-blind|placebo-controlled|clinical trial)\b'
COH  = r'(?i)\b(cohort|prospective|longitudinal|follow-up)\b'
ADJ  = r'(?i)\b(adjust\w+|multivariable|confound\w+|propensity|covariate)\b'
BIG  = r'(?i)\b(\d{1,3},\d{3}|\d{5,})\s+(participants|subjects|patients|individuals|women|men)\b'
CONSENSUS = 0.60   # validated consensus threshold: below a clear majority, no consensus
COLUMNS = ['claim_id', 'cutoff_year', 'n_pre_t', 'n_post_t', 'year_min', 'year_max', 'span_years', 'n_directional', 'n_nonnull', 'null_frac',
           'pooled_direction', 'agreement', 'consensus_verdict', 'has_consensus', 'agreement_contraction', 'direction_trajectory',
           'agreement_recent', 'dissent_recent', 'tau2_DL', 'tau2_REML', 'tau2_Hedges', 'mean_tau2', 'i_squared', 'spread', 'n_rr',
           'n_rr_studies', 'decline_signal', 'early_magnitude', 'late_magnitude', 'shrink_ratio', 'decline_sufficient', 'naive_tsa',
           'accrual_rate_5yr', 'n_large_recent', 'synth_frac', 'rct_frac', 'cohort_frac', 'synth_trend', 'rct_trend', 'synth_recent',
           'design_upgrade_frac', 'confounding_trend', 'n_ci', 'ci_rate', 'ci_width_mean', 'ci_width_sd', 'ci_crosses_null', 'val_mean',
           'val_sd', 'val_spread', 'val_cv', 'n_numeric', 'n_records', 'numeric_rate', 'numeric_rate_recent', 'numeric_trend',
           'recency_density', 'vol_last5', 'vol_growth', 'volume_ratio', 'post_pooled_direction', 'post_agreement', 'l2_sign_change',
           'l1_interval_shift', 'l3_confirmed_reversal', 'pmids_pre', 'maturity_state']


def trend(x, y):
    if len(x) < 4 or np.ptp(x) == 0: return 0.0
    return float(np.polyfit(x, y, 1)[0])


def effects(rec):
    """Effect estimates with a standard error, needed for real heterogeneity estimation."""
    raw = (rec.get('title') or '') + ' ' + (rec.get('abstract') or '')
    out = []
    for lo, hi in re.findall(CIV, raw):
        lo, hi = float(lo), float(hi)
        if hi <= lo: continue
        if lo > 0 and hi > 0:                       # ratio measure: work on the log scale
            y = (math.log(lo) + math.log(hi)) / 2; se = (math.log(hi) - math.log(lo)) / (2 * 1.96)
        else:                                        # difference measure
            y = (lo + hi) / 2; se = (hi - lo) / (2 * 1.96)
        if se > 1e-6 and abs(y) < 10: out.append((y, se))
    return out


def tau2_DL(y, v):
    w = 1 / v; mu = (w * y).sum() / w.sum()
    Q = (w * (y - mu) ** 2).sum(); k = len(y)
    C = w.sum() - (w ** 2).sum() / w.sum()
    return max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0, Q, k


def tau2_REML(y, v, it=40):
    t2 = tau2_DL(y, v)[0]
    for _ in range(it):
        w = 1 / (v + t2); mu = (w * y).sum() / w.sum()
        num = (w ** 2 * ((y - mu) ** 2 - v)).sum() + 1 / w.sum() * 0
        den = (w ** 2).sum()
        new = max(0.0, num / den) if den > 0 else 0.0
        if abs(new - t2) < 1e-8: break
        t2 = new
    return t2


def tau2_Hedges(y, v):
    k = len(y)
    if k < 2: return 0.0
    s2 = float(np.var(y, ddof=1))
    return max(0.0, s2 - float(np.mean(v)))


def heterogeneity(eff):
    if len(eff) < 3:
        return dict(tau2_DL=0.0, tau2_REML=0.0, tau2_Hedges=0.0, mean_tau2=0.0,
                    i_squared=0.0, spread=0.0, n_rr=len(eff), n_rr_studies=len(eff))
    y = np.array([a for a, _ in eff]); v = np.array([b ** 2 for _, b in eff])
    dl, Q, k = tau2_DL(y, v)
    reml = tau2_REML(y, v); hed = tau2_Hedges(y, v)
    i2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return dict(tau2_DL=float(dl), tau2_REML=float(reml), tau2_Hedges=float(hed),
                mean_tau2=float(np.mean([dl, reml, hed])), i_squared=float(i2),
                spread=float(np.percentile(y, 90) - np.percentile(y, 10)),
                n_rr=len(eff), n_rr_studies=len(eff))


def agg_verdict(v):
    """Aggregate direction with the validated consensus rule: below a clear majority the verdict is no-consensus."""
    if not v: return 0, 0.0
    nz = sum(1 for x in v if x == 0); nn = [x for x in v if x != 0]
    if not nn or nz / len(v) >= 0.50: return 0, nz / len(v)
    p = sum(1 for x in nn if x > 0); ag = max(p, len(nn) - p) / len(nn)
    if ag < CONSENSUS: return 0, ag
    return (1 if p >= len(nn) - p else -1), ag


def direction_block(dirs, years, cut):
    d = [(y, DIRMAP[x]) for y, x in zip(years, dirs) if DIRMAP[x] is not None]
    nn = [(y, v) for y, v in d if v != 0]
    f = dict(n_directional=len(d), n_nonnull=len(nn), null_frac=(len(d) - len(nn)) / max(len(d), 1))
    if len(nn) < 8:
        f.update(pooled_direction=0.0, direction_trajectory=0.0, agreement_contraction=0.0,
                 agreement=0.0, agreement_recent=0.0, dissent_recent=0.0,
                 consensus_verdict=0.0, has_consensus=0.0)
        return f
    v = np.array([x for _, x in nn]); yr = np.array([y for y, _ in nn], float)
    p = int((v > 0).sum()); f['pooled_direction'] = float(v.mean())
    allv = [x for _, x in d]
    verdict, ag = agg_verdict(allv)
    f['agreement'] = ag; f['consensus_verdict'] = float(verdict)
    f['has_consensus'] = float(ag >= CONSENSUS and verdict != 0)
    est = verdict if verdict != 0 else (1 if p >= len(v) - p else -1)
    h = len(v) // 2
    ea = max((v[:h] > 0).sum(), h - (v[:h] > 0).sum()) / max(h, 1)
    la = max((v[h:] > 0).sum(), len(v) - h - (v[h:] > 0).sum()) / max(len(v) - h, 1)
    f['agreement_contraction'] = float(la - ea)
    f['direction_trajectory'] = trend(yr, v.astype(float))
    recall_ = [x for y, x in d if y >= cut - 6]
    rec = v[yr >= cut - 6]
    if len(rec) >= 4:
        _, ar = agg_verdict(recall_)
        f['agreement_recent'] = ar
        f['dissent_recent'] = float((rec != est).mean())
    else:
        f['agreement_recent'] = f['agreement']; f['dissent_recent'] = 0.0
    return f


def decline_block(eff_years):
    """Proteus / decline effect: early-extreme then shrink."""
    if len(eff_years) < 6:
        return dict(decline_signal=0.0, early_magnitude=0.0, late_magnitude=0.0,
                    shrink_ratio=0.0, decline_sufficient=False)
    e = sorted(eff_years); k = max(2, len(e) // 3)
    early = np.mean([abs(y) for _, y, _ in e[:k]]); late = np.mean([abs(y) for _, y, _ in e[-k:]])
    return dict(decline_signal=float((early - late) / early) if early > 1e-9 else 0.0,
                early_magnitude=float(early), late_magnitude=float(late),
                shrink_ratio=float(late / early) if early > 1e-9 else 1.0, decline_sufficient=True)


def build_row(cid, recs, dirs, cut):
    pre = [(r, d) for r, d in zip(recs, dirs) if r['year'] < cut]
    post = [(r, d) for r, d in zip(recs, dirs) if r['year'] >= cut]
    if len(pre) < 40: return None
    R = [r for r, _ in pre]; D = [d for _, d in pre]
    yrs = np.array([r['year'] for r in R], float)
    raw = [((r.get('title') or '') + ' ' + (r.get('abstract') or '')) for r in R]
    f = dict(claim_id=cid, cutoff_year=int(cut), n_pre_t=len(R), n_post_t=len(post),
             year_min=int(yrs.min()), year_max=int(yrs.max()), span_years=float(yrs.max() - yrs.min()))
    f.update(direction_block(D, [r['year'] for r in R], cut))
    eff_all, eff_years = [], []
    for r, t in zip(R, raw):
        for y, se in effects(r):
            eff_all.append((y, se)); eff_years.append((r['year'], y, se))
    f.update(heterogeneity(eff_all))
    f.update(decline_block(eff_years))
    f['naive_tsa'] = float(min(1.0, sum(1 / (se ** 2) for _, se in eff_all) / 400.0)) if eff_all else 0.0
    f['accrual_rate_5yr'] = float((yrs >= cut - 5).sum()) / 5.0
    f['n_large_recent'] = float(sum(1 for r, t in zip(R, raw) if r['year'] >= cut - 5 and re.search(BIG, t)))
    syn = np.array([1.0 if re.search(SYN, t) else 0.0 for t in raw])
    rct = np.array([1.0 if re.search(RCT, t) else 0.0 for t in raw])
    coh = np.array([1.0 if re.search(COH, t) else 0.0 for t in raw])
    adj = np.array([1.0 if re.search(ADJ, t) else 0.0 for t in raw])
    f['synth_frac'], f['rct_frac'], f['cohort_frac'] = float(syn.mean()), float(rct.mean()), float(coh.mean())
    f['synth_trend'], f['rct_trend'] = trend(yrs, syn), trend(yrs, rct)
    f['synth_recent'] = float(syn[yrs >= cut - 5].mean()) if (yrs >= cut - 5).any() else 0.0
    f['design_upgrade_frac'] = float((syn[yrs >= cut - 5].mean() if (yrs >= cut-5).any() else 0) -
                                     (syn[yrs < cut - 5].mean() if (yrs < cut-5).any() else 0))
    f['confounding_trend'] = trend(yrs, adj)
    cis = [c for t in raw for c in re.findall(CIV, t)]
    f['n_ci'] = len(cis); f['ci_rate'] = len(cis) / max(len(R), 1)
    if len(eff_all) >= 3:
        w = [2 * 1.96 * se for _, se in eff_all]
        f['ci_width_mean'] = float(np.mean(w)); f['ci_width_sd'] = float(np.std(w))
        f['ci_crosses_null'] = float(np.mean([1.0 if abs(y) < 1.96 * se else 0.0 for y, se in eff_all]))
        f['val_mean'] = float(np.mean([y for y, _ in eff_all])); f['val_sd'] = float(np.std([y for y, _ in eff_all]))
        f['val_spread'] = f['spread']; f['val_cv'] = f['val_sd'] / max(abs(f['val_mean']), 1e-6)
    else:
        for k in ('ci_width_mean','ci_width_sd','ci_crosses_null','val_mean','val_sd','val_spread','val_cv'): f[k] = 0.0
    f['n_numeric'] = float(len(eff_all)); f['n_records'] = float(len(R))
    f['numeric_rate'] = float(np.mean([1.0 if effects(r) else 0.0 for r in R]))
    f['numeric_rate_recent'] = float(np.mean([1.0 if effects(r) else 0.0 for r in R if r['year'] >= cut-5])) if (yrs>=cut-5).any() else 0.0
    f['numeric_trend'] = trend(yrs, np.array([1.0 if effects(r) else 0.0 for r in R]))
    f['recency_density'] = float((yrs >= cut - 5).mean())
    f['vol_last5'] = float((yrs >= cut - 5).sum())
    f['vol_growth'] = float(((yrs >= cut-5).sum()+1) / (((yrs>=cut-10)&(yrs<cut-5)).sum()+1))
    h = len(yrs)//2; f['volume_ratio'] = float(len(yrs)-h)/max(h,1)
    if post:                                         # post-cutoff stream: evaluation only, never a model input
        pd_ = [DIRMAP[d] for _, d in post if DIRMAP[d] is not None]
        pnn = [x for x in pd_ if x != 0]
        f['post_pooled_direction'] = float(np.mean(pnn)) if pnn else 0.0
        pp = sum(1 for x in pnn if x > 0)
        f['post_agreement'] = max(pp, len(pnn)-pp)/len(pnn) if pnn else 0.0
        pre_dir = 1 if f.get('pooled_direction',0) > 0 else -1
        f['l2_sign_change'] = int(bool(pnn) and (1 if f['post_pooled_direction'] > 0 else -1) != pre_dir)
        f['l1_interval_shift'] = float(abs(f['post_agreement'] - f.get('agreement', 0)))
        f['l3_confirmed_reversal'] = int(f['l2_sign_change'] and f['post_agreement'] >= 0.65 and len(pnn) >= 20)
    else:
        f.update(post_pooled_direction=0.0, post_agreement=0.0, l1_interval_shift=0.0,
                 l2_sign_change=0, l3_confirmed_reversal=0)
    f['pmids_pre'] = ';'.join(r['pmid'] for r in R[:40])
    return f


def label(r):
    """Operational maturity state (frozen rule)."""
    if r['n_directional'] < 40 or r['n_nonnull'] < 25: return 'unassessable'
    a, ar, dr, ac = r['agreement'], r['agreement_recent'], r['dissent_recent'], r['agreement_contraction']
    if ar <= a - 0.12 and dr >= 0.40: return 'unstable'          # established position breaking down
    if ar >= 0.70 and dr < 0.30 and ac >= -0.05: return 'stable' # holding
    return 'still_forming'                                        # accumulating, not settled


def build_claim_rows(cid, records, directions):
    """All claim x cutoff rows of one claim: >= 45 dated labelled records; cutoffs = distinct integers of 7 equally spaced
    points from (year of the 40th record + 2) to (last year - 1), skipped if that span is < 4 years."""
    Dj = {d['pmid']: d['direction'] for d in directions}
    recs = [r for r in records if r['pmid'] in Dj and r.get('year')]
    dirs = [Dj[r['pmid']] for r in recs]
    if len(recs) < 45: return []
    yrs = sorted(r['year'] for r in recs)
    lo, hi = yrs[39], yrs[-1] - 1
    if hi - lo < 4: return []
    out = []
    for cut in sorted({int(x) for x in np.linspace(lo + 2, hi, 7)}):
        r = build_row(cid, recs, dirs, cut)
        if r: out.append(r)
    return out


def labelled(rows):
    """Float-convert every field except claim_id/pmids_pre after a str() round trip (as the original CSV pipeline did,
    so booleans become 0.0), then apply the label rule."""
    out = []
    for r in rows:
        r = dict(r)
        for k, v in r.items():
            if k not in ('claim_id', 'pmids_pre'):
                try: r[k] = float(str(v))
                except Exception: r[k] = 0.0
        r['maturity_state'] = label(r); out.append(r)
    return out


def write_corpus(rows, path):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in COLUMNS})
