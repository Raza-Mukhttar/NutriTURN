"""R8: an audit of the Null D per-claim logistic fits, reporting the optimiser settings that were
actually used and how many claims fall into each numerical case.

The fitting code is reproduced here verbatim from the Null D branch of nfx_nulls.py (lines
130-155) with instrumentation added and nothing else changed, so the counts describe the fits that
produced the published Null D row. Null D was run for the primary annotation only.

  python analyses/nfx_R8_nullD_audit.py
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
from null_controls import prep

MAXIT, TOL, RIDGE, ETA_CLIP, W_FLOOR, MIN_SIGNED = 60, 1e-8, 1e-6, 30.0, 1e-6, 8


def fit_one(t, h):
    """Verbatim Null D fit, instrumented. t: years of signed records, h: 1 if HARMFUL."""
    tm = t.mean(); ts = t.std() or 1.0
    z = (t - tm) / ts
    b0, b1 = 0.0, 0.0
    it = 0
    converged = False
    for k in range(MAXIT):
        it = k + 1
        eta = b0 + b1 * z
        p_ = 1.0 / (1.0 + np.exp(-np.clip(eta, -ETA_CLIP, ETA_CLIP)))
        w = np.clip(p_ * (1 - p_), W_FLOOR, None)
        r_ = h - p_
        X = np.column_stack([np.ones_like(z), z])
        XtWX = X.T @ (X * w[:, None]) + RIDGE * np.eye(2)
        step = np.linalg.solve(XtWX, X.T @ r_)
        b0 += step[0]; b1 += step[1]
        if abs(step).max() < TOL:
            converged = True
            break
    eta_f = b0 + b1 * z
    return b0, b1, it, converged, float(np.abs(eta_f).max()), z


def separable(z, h):
    """1-D complete separation: the two classes do not overlap on z."""
    z1, z0 = z[h == 1], z[h == 0]
    if len(z1) == 0 or len(z0) == 0:
        return 'single_class'
    if z1.max() < z0.min() or z0.max() < z1.min():
        return 'complete'
    return 'none'


def main():
    ann = 'primary'
    cs, U, claims = prep(ann)
    rows = []
    for c in claims:
        cd = cs[c]['codes']; yr = cs[c]['yrs']
        sg = (cd == 1) | (cd == -1)
        n = int(sg.sum())
        if n < MIN_SIGNED:
            rows.append(dict(claim_id=c, n_signed=n, case='skipped_sparse'))
            continue
        t = yr[sg].astype(float); h = (cd[sg] == 1).astype(float)
        b0, b1, it, conv, emax, z = fit_one(t, h)
        sep = separable(z, h)
        rows.append(dict(claim_id=c, n_signed=n, case=('single_class' if sep == 'single_class'
                                                       else 'separated' if sep == 'complete'
                                                       else 'regular'),
                         iterations=it, converged=bool(conv), b0=float(b0), b1=float(b1),
                         max_abs_eta=emax, eta_clip_binds=bool(emax > ETA_CLIP)))
    tot = len(rows)
    sk = [r for r in rows if r['case'] == 'skipped_sparse']
    ft = [r for r in rows if r['case'] != 'skipped_sparse']
    sc = [r for r in ft if r['case'] == 'single_class']
    sp = [r for r in ft if r['case'] == 'separated']
    rg = [r for r in ft if r['case'] == 'regular']
    nc = [r for r in ft if not r['converged']]
    cl = [r for r in ft if r['eta_clip_binds']]
    sig_all = sum(r['n_signed'] for r in rows)
    sig_sk = sum(r['n_signed'] for r in sk)

    print('R8  Null D per-claim logistic fits: optimiser settings and case counts')
    print(f'    source: analyses/nfx_nulls.py lines 130-155, annotation "{ann}"\n')
    print('SETTINGS (read from the code, not defaults)')
    print(f'   implementation          hand-written Newton/IRLS loop; numpy.linalg.solve on the')
    print(f'                           ridge-augmented information matrix. No statsmodels, no')
    print(f'                           scikit-learn, no scipy.optimize for this fit.')
    print(f'   convergence criterion   max absolute PARAMETER step < {TOL:g}')
    print(f'                           (not log-likelihood, not gradient norm)')
    print(f'   maximum iterations      {MAXIT}')
    print(f'   ridge                   {RIDGE:g} * I(2) added to X^T W X only')
    print(f'   linear-predictor clip   eta clipped to [-{ETA_CLIP:g}, {ETA_CLIP:g}] in the fit and')
    print(f'                           again when simulating, so p in [{1/(1+np.exp(ETA_CLIP)):.3e}, '
          f'{1-1/(1+np.exp(ETA_CLIP)):.12f}]')
    print(f'   IRLS weight floor       p(1-p) floored at {W_FLOOR:g}')
    print(f'   predictor scaling       year centred and scaled by its within-claim SD '
          f'(SD 0 -> 1.0)')
    print(f'   start values            b0 = b1 = 0')
    print(f'   post-loop check         NONE: the last iterate is kept unconditionally\n')
    print('CASE COUNTS (claims)')
    print(f'   claims in the archive                     {tot}')
    print(f'   not fitted, fewer than {MIN_SIGNED} signed records   {len(sk)}'
          f'   ({sig_sk} of {sig_all} signed records, {100*sig_sk/sig_all:.2f}%)')
    print(f'      -> their signed labels are NOT redrawn; they pass through at observed values')
    print(f'   fitted                                    {len(ft)}')
    print(f'      regular (classes overlap on year)      {len(rg)}')
    print(f'      single signed class                    {len(sc)}')
    print(f'      complete separation on year            {len(sp)}')
    print(f'   of the fitted claims:')
    print(f'      converged, step < {TOL:g} within {MAXIT}     {len(ft)-len(nc)}')
    print(f'      hit the {MAXIT}-iteration cap              {len(nc)}')
    print(f'      eta clip binds at the fitted params    {len(cl)}')
    if ft:
        its = np.array([r['iterations'] for r in ft])
        b1s = np.array([abs(r['b1']) for r in ft])
        print(f'      iterations  min {its.min()}  median {np.median(its):.0f}  max {its.max()}')
        print(f'      |b1|        median {np.median(b1s):.3f}  p95 {np.percentile(b1s,95):.3f}'
              f'  max {b1s.max():.3f}')
    print(f'\n   NOTE the ridge is added to the Hessian only, never to the score, so the fixed')
    print(f'   point of the iteration solves the UNPENALISED likelihood equation X^T(h-p)=0.')
    print(f'   It changes the step, not the solution, which is what "no substantive penalty" means.')
    out = dict(annotation=ann, settings=dict(
        implementation='hand-written Newton/IRLS, numpy.linalg.solve',
        source='analyses/nfx_nulls.py lines 130-155',
        convergence='max abs parameter step < 1e-8', tolerance=TOL, max_iterations=MAXIT,
        ridge_on_hessian_only=RIDGE, eta_clip=[-ETA_CLIP, ETA_CLIP], irls_weight_floor=W_FLOOR,
        start_values=[0.0, 0.0], predictor='year centred and scaled by within-claim SD',
        non_converged_policy='last iterate kept unconditionally; no post-loop check',
        sparse_policy=f'claims with fewer than {MIN_SIGNED} signed records are not fitted and '
                      f'their signed labels are not redrawn'),
        counts=dict(claims=tot, skipped_sparse=len(sk), fitted=len(ft), regular=len(rg),
                    single_class=len(sc), separated=len(sp), hit_iteration_cap=len(nc),
                    eta_clip_binds=len(cl), signed_records_total=int(sig_all),
                    signed_records_in_skipped=int(sig_sk)),
        per_claim=rows)
    print('\nwritten', N.save_json(out, 'R8_nullD_audit.json'))


if __name__ == '__main__':
    main()
