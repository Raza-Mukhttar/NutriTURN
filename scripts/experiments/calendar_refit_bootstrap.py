"""R5c: the refitting bootstrap on the three calendar cells of Table 23.

Why this is feasible after all. The per-origin two-year training grid depends only on (H, origin)
and the annotation stream, never on which claims a bootstrap replicate happens to draw. A claim
resample changes only WHICH grid rows enter training and with what multiplicity. So the grid is
built once per (H, origin) and each replicate re-weights it, instead of rebuilding it 2,000 times.

The procedure mirrors the source-cohort refitting bootstrap exactly:
  - resample claims with replacement from the cohort's claim universe (training grid union test),
  - every copy of a claim is a separate training group, so a claim drawn k times contributes k
    copies of each of its grid rows,
  - all copies of the held-out claim are excluded from training, per origin, as in the frozen
    leave-one-claim-out procedure,
  - standardisation and the logistic model are refitted inside every replicate at every origin,
  - the paired difference is AUROC(two-variable) - AUROC(margin rule) on the resampled test set.

Replicates are independent, so they are spread over worker processes.

  python analyses/nfx_R5c_calendar_refit.py --replicates 2000 --workers 32
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
import os, sys, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PUBLISHED = {3: (+0.006, -0.012, 0.023), 5: (+0.023, -0.005, 0.053), 10: (+0.008, -0.032, 0.047)}
GRID = {}          # H -> (claims, [(Xb, yb, gb, Xt, yt, gt, mt), ...])


def build(H):
    """Precompute, per origin, the training grid and the accrued test units, claim-indexed."""
    S = N.streams('primary')
    co = N.build_cohort('primary')
    d = [r for r in co if r['horizon'] == H]
    per, names = [], set()
    for Y in sorted({r['origin'] for r in d}):
        te = [r for r in d if r['origin'] == Y and r['accrued'] == 1]
        if not te:
            continue
        base = N.cohort_training_units(H, Y, S)
        names |= {q['claim_id'] for q in base} | {r['claim_id'] for r in te}
        per.append((Y, base, te))
    claims = sorted(names)
    ix = {c: i for i, c in enumerate(claims)}
    out = []
    for Y, base, te in per:
        Xb = np.array([[q['p'], q['a']] for q in base], float)
        yb = np.array([q['y'] for q in base], int)
        gb = np.array([ix[q['claim_id']] for q in base], int)
        Xt = np.array([[r['p'], r['a']] for r in te], float)
        yt = np.array([r['y'] for r in te], int)
        gt = np.array([ix[r['claim_id']] for r in te], int)
        mt = np.array([1 - abs(r['p']) for r in te], float)
        out.append((Xb, yb, gb, Xt, yt, gt, mt))
    return claims, out


def score_multiset(H, cnt):
    """Out-of-fold two-variable scores and margin scores for one claim multiset."""
    s_, y_, m_ = [], [], []
    for Xb, yb, gb, Xt, yt, gt, mt in GRID[H][1]:
        trep = cnt[gt]
        if trep.sum() == 0:
            continue
        brep = cnt[gb]
        pool = np.repeat(np.arange(len(gb)), brep)
        Xp, yp, gp = Xb[pool], yb[pool], gb[pool]
        tsel = np.repeat(np.arange(len(gt)), trep)
        for c in np.unique(gt[tsel]):
            keep = gp != c
            ytr = yp[keep]
            rows = np.flatnonzero(gt[tsel] == c)
            if keep.sum() < N.TRAIN_MIN or len(np.unique(ytr)) < 2:
                s = np.full(len(rows), 0.5)
            else:
                sc = StandardScaler().fit(Xp[keep])
                lm = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
                lm.fit(sc.transform(Xp[keep]), ytr)
                src = tsel[rows]
                s = lm.predict_proba(sc.transform(Xt[src]))[:, 1]
            src = tsel[rows]
            s_.append(s); y_.append(yt[src]); m_.append(mt[src])
    if not s_:
        return None
    return np.concatenate(s_), np.concatenate(y_), np.concatenate(m_)


def delta(H, cnt):
    r = score_multiset(H, cnt)
    if r is None:
        return None
    s, y, m = r
    if len(np.unique(y)) < 2:
        return None
    return float(N.auroc(y, s) - N.auroc(y, m))


def _work(a):
    H, seed_off = a
    ncl = len(GRID[H][0])
    rng = np.random.default_rng([N.SEED, H, seed_off])
    cnt = np.bincount(rng.integers(0, ncl, size=ncl), minlength=ncl)
    return delta(H, cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replicates', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--horizons', type=int, nargs='+', default=[3, 5, 10])
    a_ = ap.parse_args()
    print('R5c refitting bootstrap on the calendar cells of Table 23\n')
    for H in a_.horizons:
        t0 = time.time()
        GRID[H] = build(H)
        ncl = len(GRID[H][0])
        full = delta(H, np.ones(ncl, int))
        d0, plo, phi = PUBLISHED[H]
        ok = abs(full - d0) < 6e-4
        print(f'H={H:<3d} claim universe {ncl:3d}   origins {len(GRID[H][1])}   '
              f'point delta {full:+.4f} vs published {d0:+.3f}   '
              f'{"reproduces" if ok else "DOES NOT REPRODUCE"}   '
              f'grid built in {time.time()-t0:.1f}s')
        if not ok:
            print('   refusing to bootstrap a cell that does not reproduce'); continue
    import multiprocessing as mp
    out = {}
    print(f"\n{'cell':16s} {'delta':>7s} {'percentile (published)':>24s} "
          f"{'REFIT 95%':>20s} {'repl':>6s} {'excl 0':>7s} {'mins':>6s}")
    for H in a_.horizons:
        if H not in GRID:
            continue
        t0 = time.time()
        with mp.Pool(a_.workers) as pool:
            vals = pool.map(_work, [(H, i) for i in range(a_.replicates)], chunksize=4)
        v = np.array([x for x in vals if x is not None and np.isfinite(x)])
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        d0, plo, phi = PUBLISHED[H]
        est = N.established(lo, hi)
        out[f'calendar H={H}'] = dict(delta=delta(H, np.ones(len(GRID[H][0]), int)),
                                      published_percentile=[plo, phi],
                                      refit=[float(v.mean()), lo, hi],
                                      refit_replicates=int(len(v)),
                                      refit_established=bool(est),
                                      claims=len(GRID[H][0]))
        print(f"{'Calendar H=' + str(H):16s} {d0:+7.3f} {N.fmt_ci(plo, phi):>24s} "
              f"{N.fmt_ci(lo, hi):>19s}{'*' if est else ' '} {len(v):6d} "
              f"{'yes' if est else 'no':>7s} {(time.time()-t0)/60:6.1f}")
    print('\nwritten', N.save_json({'calendar_refit': out}, 'R5c_calendar_refit.json'))


if __name__ == '__main__':
    main()
