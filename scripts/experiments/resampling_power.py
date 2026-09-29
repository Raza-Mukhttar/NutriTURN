"""E11: resampling power analysis, so the half-width statement becomes a detectable effect.

For each cell, synthetic scores with a known gain delta over the margin rule are built as
  s_lambda = rank(margin) + lambda * (y - 0.5) * U,  U ~ Uniform(0,1) drawn once,
and lambda is bisected until AUROC(s_lambda) - AUROC(margin) = delta on the full cell. An outer loop
of R claim resamples then asks how often the paired claim-clustered interval (B inner resamples)
establishes the gain. Power(delta) is that fraction; the MDE is the smallest delta with power >= 0.8.

The simulation assumes the score is correlated with the margin rule in the way this construction
implies; the reported MDE is conditional on that assumption.

  python analyses/nfx_power.py --deltas 0.01 0.02 0.03 0.04 0.05 0.06 --R 200 --B 500
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


def auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    o = np.argsort(s, kind='stable'); ss = s[o]
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1, dtype=float)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def make_score(base_rank, y, U, lam, noise=None, nu=0.0):
    s = base_rank + lam * (y - 0.5) * U
    if noise is not None and nu > 0:
        s = s + nu * noise
    return s


def match_noise(base_rank, target_rho, Z):
    """Scale independent noise so the competing score's Spearman correlation with the margin rule
    matches the value actually observed for a real competing score in that cell."""
    from scipy.stats import spearmanr
    if target_rho is None or not np.isfinite(target_rho) or target_rho >= 0.999:
        return 0.0
    lo, hi = 0.0, 10.0 * (base_rank.std() + 1e-9)
    for _ in range(60):
        mid = (lo + hi) / 2
        r = abs(spearmanr(base_rank + mid * Z, base_rank).statistic)
        if r > target_rho:
            lo = mid
        else:
            hi = mid
    return hi


def calibrate(base_rank, y, U, delta, a0, noise=None, nu=0.0):
    """Find lambda so the competing score's AUROC exceeds the MARGIN RULE's by delta.

    The reference is always the margin rule, because that is what the paired interval tests. In
    realistic mode the added noise first costs AUROC, so lambda must recover that loss as well as
    supply the gain; the realised rank correlation with the margin rule is reported alongside.
    """
    lo, hi = 0.0, 10.0
    for _ in range(60):
        if auc(y, make_score(base_rank, y, U, hi, noise, nu)) - a0 >= delta:
            break
        hi *= 2
        if hi > 1e9:
            return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if auc(y, make_score(base_rank, y, U, mid, noise, nu)) - a0 < delta:
            lo = mid
        else:
            hi = mid
    return hi


_S = {}


def one_rep(args):
    seed, B = args
    y, g, sm, sl, claims, by = _S['y'], _S['g'], _S['sm'], _S['sl'], _S['claims'], _S['by']
    rng = np.random.default_rng(seed)
    pick = rng.choice(claims, size=len(claims), replace=True)
    ix = np.concatenate([by[c] for c in pick])
    yy = y[ix]
    if len(np.unique(yy)) < 2:
        return None
    gg = g[ix]
    cl = np.unique(gg)
    b2 = {c: np.where(gg == c)[0] for c in cl}
    vals = np.empty(B)
    rng2 = np.random.default_rng(seed + 777_777)
    for b in range(B):
        p2 = rng2.choice(cl, size=len(cl), replace=True)
        jx = np.concatenate([b2[c] for c in p2])
        if len(np.unique(yy[jx])) < 2:
            vals[b] = np.nan; continue
        vals[b] = auc(yy[jx], sl[ix][jx]) - auc(yy[jx], sm[ix][jx])
    v = vals[np.isfinite(vals)]
    if len(v) < 50:
        return None
    return float(np.percentile(v, 2.5)) > 0


def cell_power(name, y, g, sm, deltas, R, B, workers, seed=N.SEED, rho=None, mode='paired'):
    y = np.asarray(y); g = np.asarray(g); sm = np.asarray(sm, float)
    a0 = auc(y, sm)
    order = np.argsort(np.argsort(sm, kind='stable'), kind='stable').astype(float)
    rng = np.random.default_rng(seed)
    U = rng.random(len(y))
    Z = rng.standard_normal(len(y))
    nu = match_noise(order, rho, Z) if mode == 'realistic' else 0.0
    claims = np.unique(g); by = {c: np.where(g == c)[0] for c in claims}
    out = {'cell': name, 'mode': mode, 'units': int(len(y)), 'events': int(y.sum()),
           'margin_auroc': float(a0), 'target_rho': rho, 'noise_scale': nu, 'power': {}}
    print(f'\n{name} [{mode}]: {len(y)} units, {int(y.sum())} events, margin AUROC {a0:.3f}'
          + (f', target rank correlation with margin {rho:.3f}' if mode == 'realistic' and rho else ''))
    mde = None
    for d in deltas:
        lam = calibrate(order, y, U, d, a0, Z, nu)
        if lam is None:
            print(f'  delta {d:.4f}: not reachable'); continue
        sl = make_score(order, y, U, lam, Z, nu)
        _S.update(y=y, g=g, sm=sm, sl=sl, claims=claims, by=by)
        jobs = [(seed * 1000 + i, B) for i in range(R)]
        with Pool(workers, initializer=_S.update, initargs=({'y': y, 'g': g, 'sm': sm, 'sl': sl,
                                                             'claims': claims, 'by': by},)) as pool:
            res = [r for r in pool.map(one_rep, jobs, chunksize=2) if r is not None]
        pw = float(np.mean(res)) if res else float('nan')
        from scipy.stats import spearmanr
        out['power'][f'{d:.4f}'] = dict(power=pw, replicates=len(res),
                                       realised_gain_over_margin=float(auc(y, sl) - a0),
                                       realised_rank_corr_with_margin=float(abs(spearmanr(sl, sm).statistic)))
        print(f'  delta {d:.4f}: power {pw:.2f} ({len(res)} replicates)')
        if mde is None and pw >= 0.8:
            mde = d
    out['mde_80'] = mde
    print(f'  MDE at 80% power: {mde if mde is not None else "> max delta tested"}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deltas', nargs='+', type=float, default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    ap.add_argument('--R', type=int, default=200)
    ap.add_argument('--B', type=int, default=500)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    from scipy.stats import spearmanr
    cells = []
    # development cells, primary and second; rho is the observed rank correlation between the real
    # two-variable score and the margin rule in that cell
    for ann in ('primary', 'second'):
        rows = N.unit_table(ann)
        idx, s2, y, g = N.loco_two_var(rows)
        sm = N.margin_score(rows, idx)
        cells.append((f'development, {ann} annotation', y, g, sm,
                      abs(float(spearmanr(s2, sm).statistic))))
    # calendar cells
    co = N.build_cohort('primary')
    for H in (3, 5, 10):
        d = [r for r in co if r['horizon'] == H]
        acc = [r for r in d if r['accrued']]
        y = np.array([r['y'] for r in acc]); g = np.array([r['claim_id'] for r in acc])
        sm = np.array([1 - abs(r['p']) for r in acc])
        s2c, _, _, _ = N.cohort_loco_two_var(d, H)
        cells.append((f'calendar cohort, H = {H}', y, g, sm,
                      abs(float(spearmanr(s2c, sm).statistic))))
    # frozen external cohorts
    sys.path.insert(0, os.path.join(_NT, 'scripts'))
    import astra_common as ac
    for which, label in ((1, 'External-1, frozen'), (2, 'External-2, frozen')):
        rows_e, _ = ac.load_external(which)
        P, y, g = ac.prospective(rows_e)
        sm = np.array([1 - abs(r['pooled_direction']) for r in P])
        a_ = np.array([r['agreement'] for r in P])
        s2e = N.loco_two_var([dict(p=r['pooled_direction'], a=r['agreement'], y=yy, claim_id=gg)
                              for r, yy, gg in zip(P, y, g)])[1]
        cells.append((label, np.asarray(y), np.asarray(g), sm,
                      abs(float(spearmanr(s2e, sm).statistic))))
    out = {'protocol': __doc__, 'R': a.R, 'B': a.B, 'deltas': a.deltas, 'cells': []}
    print('E11 resampling power analysis')
    for nm_, y, g, sm, rho in cells:
        for mode in ('paired', 'realistic'):
            out['cells'].append(cell_power(nm_, y, g, sm, a.deltas, a.R, a.B, a.workers,
                                           rho=rho, mode=mode))
    print('\nMDE summary (80% power)')
    print(f"  {'cell':34s} {'units':>6s} {'events':>7s} {'MDE paired':>11s} {'MDE realistic':>14s}")
    names = []
    for c in out['cells']:
        if c['cell'] not in names:
            names.append(c['cell'])
    for nm_ in names:
        cs = {c['mode']: c for c in out['cells'] if c['cell'] == nm_}
        f = lambda m: (f"{cs[m]['mde_80']:.4f}" if cs[m]['mde_80'] is not None
                       else '>' + str(max(a.deltas))) if m in cs else 'n/a'
        c0 = cs['paired']
        print(f"  {nm_:34s} {c0['units']:6d} {c0['events']:7d} {f('paired'):>11s} {f('realistic'):>14s}")
    print('written', N.save_json(out, 'E11_power.json'))


if __name__ == '__main__':
    main()
