"""E17: is the decade-level label shift composition, or drift inside claims?

The pooled HARMFUL share falls from 0.463 to 0.225 across decades. Two explanations differ in what
they imply for the benchmark:

  composition  different claims publish in different decades, and claims differ in their label mix.
               The endpoint compares a claim against itself, so composition cannot manufacture
               sign changes.
  within-claim the annotator's output drifts over calendar time inside the same claim. Then a
               claim's pre-cutoff and post-cutoff majorities can differ with no change in the
               literature, and the endpoint records an event that is an annotation artefact.

The test is the same regression with and without claim fixed effects. Let s be the signed label
(+1 HARMFUL, -1 PROTECTIVE) of each signed record and t its publication year.

  pooled slope       s ~ t                 mixes composition and drift
  within-claim slope s ~ t + claim effects removes composition by centring both within claim

A pooled slope that is large while the within-claim slope is near zero means composition. A
within-claim slope that is itself large means the annotator drifts inside claims, which is the case
that matters. Intervals are claim-clustered percentile bootstraps, 2,000 resamples, seed 3.

  python analyses/nfx_drift_within.py
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
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N


def slopes(year, sgn, claim, within):
    """Least-squares slope of sgn on year, optionally after centring both within claim."""
    y = year.astype(float); s = sgn.astype(float)
    if within:
        for c in np.unique(claim):
            m = claim == c
            if m.sum() < 2:
                y[m] = 0.0; s[m] = 0.0; continue
            y[m] = y[m] - y[m].mean(); s[m] = s[m] - s[m].mean()
    else:
        y = y - y.mean(); s = s - s.mean()
    d = (y * y).sum()
    return float((y * s).sum() / d) if d > 0 else float('nan')


def boot(year, sgn, claim, within, n=N.BOOT, seed=N.SEED):
    rng = np.random.default_rng(seed)
    cl = np.unique(claim); by = {c: np.where(claim == c)[0] for c in cl}
    out = []
    for _ in range(n):
        pick = rng.choice(cl, size=len(cl), replace=True)
        ix = np.concatenate([by[c] for c in pick])
        # relabel the resampled claims so repeated draws are separate fixed effects
        lab = np.concatenate([np.full(len(by[c]), i) for i, c in enumerate(pick)])
        v = slopes(year[ix], sgn[ix], lab, within)
        if np.isfinite(v):
            out.append(v)
    a = np.array(out)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    print('E17 within-claim drift\n')
    A = N.assoc()
    out = {'annotations': {}}
    for ann, col in (('primary', 'label_primary'), ('second', 'label_second')):
        rows = [r for r in A if r[col] in ('PROTECTIVE', 'HARMFUL')]
        year = np.array([r['year'] for r in rows])
        sgn = np.array([1 if r[col] == 'HARMFUL' else -1 for r in rows])
        claim = np.array([r['claim_id'] for r in rows])
        pooled = slopes(year.copy(), sgn.copy(), claim, False)
        within = slopes(year.copy(), sgn.copy(), claim, True)
        pl, ph = boot(year, sgn, claim, False)
        wl, wh = boot(year, sgn, claim, True)
        share = 1 - abs(within / pooled) if pooled else float('nan')
        out['annotations'][ann] = dict(
            signed_records=len(rows), claims=int(len(np.unique(claim))),
            pooled_slope_per_year=pooled, pooled_ci=[pl, ph],
            within_claim_slope_per_year=within, within_claim_ci=[wl, wh],
            share_of_pooled_removed_by_claim_effects=share,
            within_established=N.established(wl, wh))
        print(f'{ann} annotation  ({len(rows):,} signed records in {len(np.unique(claim))} claims)')
        print(f'   pooled slope        {pooled*10:+.4f} per decade  {N.fmt_ci(pl*10, ph*10, 4)}')
        print(f'   within-claim slope  {within*10:+.4f} per decade  {N.fmt_ci(wl*10, wh*10, 4)}'
              f"{'  ESTABLISHED' if N.established(wl, wh) else '  not established'}")
        print(f'   claim effects remove {100*share:.1f}% of the pooled slope')
        # how much of the decade shift survives inside claims
        dec = (year // 10) * 10
        keep = np.isin(dec, [1990, 2000, 2010, 2020])
        pooled_shift, within_shift = {}, {}
        for d in (1990, 2000, 2010, 2020):
            m = dec == d
            pooled_shift[d] = float((sgn[m] == 1).mean())
        # within-claim: only claims present in both 1990s and 2020s
        c90 = set(claim[dec == 1990]); c20 = set(claim[dec == 2020])
        both = sorted(c90 & c20)
        if both:
            d90 = np.array([ (sgn[(claim == c) & (dec == 1990)] == 1).mean() for c in both ])
            d20 = np.array([ (sgn[(claim == c) & (dec == 2020)] == 1).mean() for c in both ])
            diff = d20 - d90
            rng = np.random.default_rng(N.SEED)
            bs = [np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(N.BOOT)]
            lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            out['annotations'][ann]['harmful_share_1990s'] = float(d90.mean())
            out['annotations'][ann]['harmful_share_2020s'] = float(d20.mean())
            out['annotations'][ann]['paired_within_claim_change'] = [float(diff.mean()), lo, hi]
            out['annotations'][ann]['paired_claims'] = len(both)
            print(f'   paired within-claim HARMFUL share, {len(both)} claims present in both the '
                  f'1990s and 2020s:')
            print(f'      1990s {d90.mean():.3f} -> 2020s {d20.mean():.3f}  '
                  f'change {diff.mean():+.3f} {N.fmt_ci(lo, hi)}'
                  f"{'  ESTABLISHED' if (lo > 0 or hi < 0) else '  not established'}")
            print(f'   unpaired pooled shares: ' +
                  '  '.join(f'{d}s {pooled_shift[d]:.3f}' for d in (1990, 2000, 2010, 2020)))
        print()
    print('written', N.save_json(out, 'E17_drift_within.json'))


if __name__ == '__main__':
    main()
