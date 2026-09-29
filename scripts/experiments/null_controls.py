"""E2: exchangeable null models on every annotation, with the paired null distribution.

  python analyses/nfx_nulls.py --annotation primary --draws 1000

Null A  within-claim permutation of the direction labels (publication years fixed), so each
        claim's realised label counts are preserved exactly.
Null B  each record's label drawn i.i.d. from its claim's own label frequencies, so the
        generating probabilities are preserved but the realised counts are not.

Every draw rebuilds the inputs and the endpoint from the permuted or simulated stream and reruns
the leave-one-claim-out two-variable fit. Recorded per annotation: observed events; the null event
mean, 95% range and upper-tail p-value; the null AUROC distributions; and the paired null
distribution of (two-variable minus margin) with the upper-tail p-value of the observed value.
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
import os, sys, json, argparse
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_G = {}


def prep(annotation):
    """Per-claim year-sorted streams plus the cutoff boundary of every unit."""
    S = N.streams(annotation)
    units = list(__import__('csv').DictReader(open(os.path.join(N.DATA, 'units.csv'))))
    claims = sorted(S)
    cs = {}
    for c in claims:
        yrs, codes = S[c]
        o = np.argsort(yrs, kind='stable')
        cs[c] = dict(yrs=yrs[o], codes=codes[o])
    U = []
    for u in units:
        c = u['claim_id']; t = int(u['cutoff'])
        b = int(np.searchsorted(cs[c]['yrs'], t, side='left'))
        U.append((c, t, b))
    return cs, U, claims


def rebuild(cs, U, codes_by_claim):
    """Vectorised rebuild of (p, a, y) for all units from a set of per-claim code arrays."""
    pre = {}
    for c, cd in codes_by_claim.items():
        c1 = np.concatenate([[0], np.cumsum(cd == 1)])
        cm = np.concatenate([[0], np.cumsum(cd == -1)])
        c0 = np.concatenate([[0], np.cumsum(cd == 0)])
        pre[c] = (c1, cm, c0)
    p = np.empty(len(U)); a = np.empty(len(U)); y = np.empty(len(U), int)
    for i, (c, t, b) in enumerate(U):
        c1, cm, c0 = pre[c]
        h = int(c1[b]); l = int(cm[b]); n0 = int(c0[b]); n = h + l
        pv = (h - l) / n if n >= N.SPARSE_GATE else 0.0
        res = h + l + n0
        ns = n0 / res if res else 0.0
        a[i] = 0.0 if n < N.SPARSE_GATE else (ns if ns >= 0.5 else (1 + abs(pv)) / 2)
        p[i] = pv
        ph = int(c1[-1] - c1[b]); pl = int(cm[-1] - cm[b]); m = ph + pl
        pp = (ph - pl) / m if m else 0.0
        y[i] = int(N.sgn0(pp) != N.sgn0(pv))
    return p, a, y


def loco(p, a, y, g, gidx):
    X = np.column_stack([p, a]); s = np.empty(len(y))
    for c, te in gidx.items():
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            s[te] = float(y[tr].mean()) if tr.any() else 0.0
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
        m.fit(sc.transform(X[tr]), y[tr])
        s[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return s


def fast_auroc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = np.argsort(np.argsort(s, kind='stable'), kind='stable').astype(float) + 1
    # average ranks over ties
    order = np.argsort(s, kind='stable'); ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def one_draw(args):
    kind, seed = args
    cs, U, claims, gidx = _G['cs'], _G['U'], _G['claims'], _G['gidx']
    rng = np.random.default_rng(seed)
    cb = {}
    for c in claims:
        cd = cs[c]['codes']
        if kind == 'A':
            cb[c] = rng.permutation(cd)
        elif kind == 'B':
            lab, cnt = np.unique(cd, return_counts=True)
            cb[c] = rng.choice(lab, size=len(cd), replace=True, p=cnt / cnt.sum())
        elif kind.startswith('C'):
            # Null C: permute inside each claim-by-block period, so the claim's composition AND
            # the label mix of every block are preserved exactly. Anything the observed stream
            # has beyond era-level drift must survive this control. 'C' is 10-year blocks aligned
            # to decades; 'C5'/'C20' change the width, 'Cs' shifts the boundary by five years.
            width = 10; shift = 0
            if kind == 'C5':
                width = 5
            elif kind == 'C20':
                width = 20
            elif kind == 'Cs':
                shift = 5
            yr = cs[c]['yrs']; out = cd.copy()
            blk = ((yr - shift) // width) * width
            for b in np.unique(blk):
                m = blk == b
                out[m] = rng.permutation(cd[m])
            cb[c] = out
        else:
            # Null D: a parametric trend-preserving control. Fit a within-claim logistic trend of
            # P(HARMFUL | signed) on year and simulate signed labels from it at the observed dates,
            # leaving NULL and UNCLEAR positions untouched. This tests whether drift alone
            # reproduces the target without conditioning on block totals.
            yr = cs[c]['yrs']; out = cd.copy()
            sg = (cd == 1) | (cd == -1)
            if sg.sum() >= 8:
                t = yr[sg].astype(float); h = (cd[sg] == 1).astype(float)
                tm = t.mean(); ts = t.std() or 1.0
                z = (t - tm) / ts
                b0, b1 = 0.0, 0.0
                for _ in range(60):                       # Newton steps on the logistic likelihood
                    eta = b0 + b1 * z
                    p_ = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
                    w = np.clip(p_ * (1 - p_), 1e-6, None)
                    r_ = h - p_
                    X = np.column_stack([np.ones_like(z), z])
                    XtWX = X.T @ (X * w[:, None]) + 1e-6 * np.eye(2)
                    step = np.linalg.solve(XtWX, X.T @ r_)
                    b0 += step[0]; b1 += step[1]
                    if abs(step).max() < 1e-8:
                        break
                pr = 1.0 / (1.0 + np.exp(-np.clip(b0 + b1 * z, -30, 30)))
                draw = rng.random(len(pr)) < pr
                out[np.where(sg)[0]] = np.where(draw, 1, -1)
            cb[c] = out
    p, a, y = rebuild(cs, U, cb)
    if len(np.unique(y)) < 2:
        return None
    g = _G['g']
    s2 = loco(p, a, y, g, gidx)
    sm = 1 - np.abs(p)
    A2, AM = fast_auroc(y, s2), fast_auroc(y, sm)
    return dict(events=int(y.sum()), two_var=A2, margin=AM, diff=A2 - AM)


def init(cs, U, claims, g, gidx):
    _G.update(cs=cs, U=U, claims=claims, g=g, gidx=gidx)


def run(annotation, draws, workers, kinds=('A', 'B')):
    cs, U, claims = prep(annotation)
    g = np.array([u[0] for u in U])
    gidx = {c: (g == c) for c in np.unique(g)}
    # observed
    obs_codes = {c: cs[c]['codes'] for c in claims}
    p, a, y = rebuild(cs, U, obs_codes)
    s2 = loco(p, a, y, g, gidx); sm = 1 - np.abs(p)
    o_ev = int(y.sum()); o_2 = fast_auroc(y, s2); o_m = fast_auroc(y, sm); o_d = o_2 - o_m
    print(f'observed [{annotation}]: events {o_ev}  two-var {o_2:.3f}  margin {o_m:.3f}  '
          f'two-var minus margin {o_d:+.3f}')
    out = {'annotation': annotation, 'draws_requested': draws,
           'observed': dict(events=o_ev, two_var_auroc=o_2, margin_auroc=o_m, diff=o_d)}
    init(cs, U, claims, g, gidx)
    for kind in kinds:
        jobs = [(kind, 10_000 * (1 + hash(kind) % 17) + i) for i in range(draws)]
        with Pool(workers, initializer=init, initargs=(cs, U, claims, g, gidx)) as pool:
            res = [r for r in pool.map(one_draw, jobs, chunksize=4) if r]
        ev = np.array([r['events'] for r in res], float)
        a2 = np.array([r['two_var'] for r in res]); am = np.array([r['margin'] for r in res])
        df = np.array([r['diff'] for r in res])
        up = lambda arr, obs: float((np.sum(arr >= obs) + 1) / (len(arr) + 1))
        lo = lambda arr, obs: float((np.sum(arr <= obs) + 1) / (len(arr) + 1))
        blk = dict(
            draws=len(res),
            events=dict(mean=float(ev.mean()), lo=float(np.percentile(ev, 2.5)), hi=float(np.percentile(ev, 97.5)),
                        p_upper=up(ev, o_ev), observed_above_range=bool(o_ev > np.percentile(ev, 97.5))),
            two_var=dict(mean=float(np.nanmean(a2)), lo=float(np.nanpercentile(a2, 2.5)),
                         hi=float(np.nanpercentile(a2, 97.5)), p_upper=up(a2[np.isfinite(a2)], o_2)),
            margin=dict(mean=float(np.nanmean(am)), lo=float(np.nanpercentile(am, 2.5)),
                        hi=float(np.nanpercentile(am, 97.5)), p_upper=up(am[np.isfinite(am)], o_m),
                        p_lower=lo(am[np.isfinite(am)], o_m)),
            paired_diff=dict(mean=float(np.nanmean(df)), lo=float(np.nanpercentile(df, 2.5)),
                             hi=float(np.nanpercentile(df, 97.5)), p_upper=up(df[np.isfinite(df)], o_d)))
        out[f'null_{kind}'] = blk
        # keep the raw per-draw values so the figure can show the distributions themselves
        np.savez_compressed(os.path.join(N.RESULTS, f'E2_draws_{annotation}_{kind}.npz'),
                            events=ev, two_var=a2, margin=am, diff=df)
        nm = {'A': 'Null A (within-claim permutation)',
              'B': 'Null B (stationary voting)',
              'C': 'Null C (within claim-by-decade permutation)',
              'C5': 'Null C, 5-year blocks',
              'C20': 'Null C, 20-year blocks',
              'Cs': 'Null C, decade boundary shifted 5 years',
              'D': 'Null D (fitted within-claim logistic trend)'}.get(kind, f'Null {kind}')
        print(f'  {nm}: draws {len(res)}')
        print(f'    events       mean {blk["events"]["mean"]:6.1f}  95% range '
              f'[{blk["events"]["lo"]:.0f}, {blk["events"]["hi"]:.0f}]  p(obs>=null) {blk["events"]["p_upper"]:.3f}'
              f'  observed above range: {blk["events"]["observed_above_range"]}')
        print(f'    two-var      mean {blk["two_var"]["mean"]:.3f}  95% range '
              f'[{blk["two_var"]["lo"]:.3f}, {blk["two_var"]["hi"]:.3f}]  p {blk["two_var"]["p_upper"]:.3f}')
        print(f'    margin       mean {blk["margin"]["mean"]:.3f}  95% range '
              f'[{blk["margin"]["lo"]:.3f}, {blk["margin"]["hi"]:.3f}]  p(lower) {blk["margin"]["p_lower"]:.3f}')
        print(f'    PAIRED diff  mean {blk["paired_diff"]["mean"]:+.3f}  95% range '
              f'[{blk["paired_diff"]["lo"]:+.3f}, {blk["paired_diff"]["hi"]:+.3f}]  '
              f'p(obs>=null) {blk["paired_diff"]["p_upper"]:.3f}')
    p_ = N.save_json(out, f'E2_null_{annotation}.json')
    print('written', p_)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotation', default='primary')
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--kinds', nargs='+', default=['A', 'B'])
    a = ap.parse_args()
    run(a.annotation, a.draws, a.workers, tuple(a.kinds))
