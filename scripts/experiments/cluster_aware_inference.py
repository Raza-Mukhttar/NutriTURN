"""R5 (their E5): cluster-aware inference and a model-refitting bootstrap.

Two separate checks on the headline contrasts.

A. Clustered ROC. AUROC differences on correlated, clustered observations need a variance that
accounts for the cluster, not a coefficient test. This uses DeLong's placement values to form the
per-observation influence of the paired difference, aggregates it within claim, and takes the
cluster-robust variance as the sum of squared claim totals. That is the clustered analogue of
DeLong's variance and is the same object Obuchowski's clustered-ROC method estimates.

B. Model-refitting bootstrap. Resampling stored predictions holds the fitted models fixed and so
understates uncertainty from the fitting itself. Here each replicate resamples claims with
replacement, keeps all of a sampled claim's units and multiplicities, refits standardisation and
the model inside the claim-held-out procedure with every copy of a held-out claim excluded from
training, and recomputes the paired difference.

Non-inferiority is judged at the paper's prespecified margin of -0.02 on the lower bound.

  python analyses/nfx_R5_inference.py --replicates 400
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N
from wild_cluster_bootstrap import placements
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def clustered_roc_ci(y, sa, sb, g):
    V10a, V01a = placements(y, sa); V10b, V01b = placements(y, sb)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    delta = float(V10a.mean() - V10b.mean())
    infl = np.zeros(len(y))
    infl[y == 1] = (V10a - V10b - delta) / n1
    infl[y == 0] = (V01a - V01b - delta) / n0
    h = np.array([infl[g == c].sum() for c in np.unique(g)])
    se = float(np.sqrt((h ** 2).sum()))
    return delta, delta - 1.96 * se, delta + 1.96 * se, se


def refit_delta(rows, idx_claims, feats=('p', 'a')):
    """Refit the claim-held-out pipeline on a claim multiset and return the paired difference."""
    units, gid = [], []
    for j, c in enumerate(idx_claims):
        for r in rows:
            if r['claim_id'] == c:
                units.append(r); gid.append(j)          # copies are separate training groups
    if len(units) < 60:
        return None
    y = np.array([u['y'] for u in units]); g = np.array(gid)
    orig = np.array([u['claim_id'] for u in units])
    if len(np.unique(y)) < 2:
        return None
    X = np.array([[u[f] for f in feats] for u in units], float)
    s = np.full(len(units), np.nan)
    for c in np.unique(orig):                            # hold out ALL copies of the original claim
        te = orig == c; tr = ~te
        if len(np.unique(y[tr])) < 2:
            s[te] = y[tr].mean() if tr.any() else 0.0; continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000)
        m.fit(sc.transform(X[tr]), y[tr]); s[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    sm = np.array([1 - abs(u['p']) for u in units])
    if len(np.unique(y)) < 2:
        return None
    return float(N.auroc(y, s) - N.auroc(y, sm))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--replicates', type=int, default=2000)
    a_ = ap.parse_args()
    print('R5 cluster-aware inference and model-refitting bootstrap\n')
    out = {'contrasts': []}
    tabs = {a: N.unit_table(a) for a in ('primary', 'second')}
    for a in tabs:
        for r, acc in zip(tabs[a], N.build_cohort_accrual5(tabs[a])):
            r['accrual_5y'] = acc

    print(f"{'contrast':40s} {'delta':>7s} {'percentile':>18s} {'clustered ROC':>20s} "
          f"{'refit bootstrap':>20s}  {'non-inf -0.02':>13s}")
    for a in ('primary', 'second'):
        rows = tabs[a]
        idx, s2, y, g = N.loco_two_var(rows)
        sm = N.margin_score(rows, idx)
        d0, plo, phi, _ = N.paired_ci(y, s2, sm, g)
        dc, clo, chi, se = clustered_roc_ci(y, s2, sm, g)
        cl = np.unique(np.array([r['claim_id'] for r in rows]))
        rng = np.random.default_rng(N.SEED)
        vals = []
        for b in range(a_.replicates):
            pick = rng.choice(cl, size=len(cl), replace=True)
            v = refit_delta(rows, pick)
            if v is not None and np.isfinite(v):
                vals.append(v)
        v = np.array(vals)
        rlo, rhi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        rec = dict(contrast=f'development {a}: signed minus margin', delta=d0,
                   percentile=[plo, phi], clustered_roc=[clo, chi], clustered_se=se,
                   refit=[float(v.mean()), rlo, rhi], refit_replicates=len(v),
                   percentile_established=N.established(plo, phi),
                   clustered_established=N.established(clo, chi),
                   refit_established=N.established(rlo, rhi),
                   non_inferior_refit=bool(rlo > -0.02))
        out['contrasts'].append(rec)
        print(f"{rec['contrast']:40s} {d0:+7.3f} {N.fmt_ci(plo,phi):>18s}"
              f"{'*' if rec['percentile_established'] else ' '}"
              f"{N.fmt_ci(clo,chi):>19s}{'*' if rec['clustered_established'] else ' '}"
              f"{N.fmt_ci(rlo,rhi):>19s}{'*' if rec['refit_established'] else ' '}"
              f"  {'yes' if rec['non_inferior_refit'] else 'no':>13s}")
    # within-class (restricted-pair) cells, clustered ROC on the restricted statistic
    for a in ('primary', 'second'):
        rows = tabs[a]
        idx, s2, y, g = N.loco_two_var(rows)
        idxb, s2b, yb, gb = N.loco_two_var(rows, class_blind=True)
        sm = N.margin_score(rows, idx)
        sign = np.array([N.sgn0(rows[i]['p']) for i in idx])
        keep = sign != 0                                  # restricted pairs only
        d0, plo, phi, _ = N.paired_ci(y[keep], s2[keep], sm[keep], g[keep])
        dc, clo, chi, se = clustered_roc_ci(y[keep], s2[keep], sm[keep], g[keep])
        out['contrasts'].append(dict(contrast=f'within-class {a}: signed minus margin',
                                     delta=d0, percentile=[plo, phi], clustered_roc=[clo, chi],
                                     percentile_established=N.established(plo, phi),
                                     clustered_established=N.established(clo, chi)))
        print(f"{'within-class ' + a + ': signed minus margin':40s} {d0:+7.3f} "
              f"{N.fmt_ci(plo,phi):>18s} {N.fmt_ci(clo,chi):>19s}")

    # calendar H5 and External-2 on stored predictions, clustered ROC only
    co = N.build_cohort('primary')
    d5 = [r for r in co if r['horizon'] == 5]
    s2, y, g, _ = N.cohort_loco_two_var(d5, 5)
    sm = np.array([1 - abs(r['p']) for r in d5 if r['accrued']])
    d0, plo, phi, _ = N.paired_ci(y, s2, sm, g)
    dc, clo, chi, se = clustered_roc_ci(y, s2, sm, g)
    out['contrasts'].append(dict(contrast='calendar H=5: signed minus margin', delta=d0,
                                 percentile=[plo, phi], clustered_roc=[clo, chi],
                                 percentile_established=N.established(plo, phi),
                                 clustered_established=N.established(clo, chi)))
    print(f"{'calendar H=5: signed minus margin':40s} {d0:+7.3f} {N.fmt_ci(plo,phi):>18s} "
          f"{N.fmt_ci(clo,chi):>19s}")
    sys.path.insert(0, _os.path.join(_NT, 'scripts', 'lib'))
    import astra_common as ac
    rows_e, _ = ac.load_external(2)
    P, ye, ge = ac.prospective(rows_e)
    er = [dict(p=r['pooled_direction'], a=r['agreement'], y=int(yy), claim_id=gg)
          for r, yy, gg in zip(P, ye, ge)]
    idx, se2, y2, g2 = N.loco_two_var(er)
    sm2 = np.array([1 - abs(er[i]['p']) for i in idx])
    d0, plo, phi, _ = N.paired_ci(y2, se2, sm2, g2)
    dc, clo, chi, _ = clustered_roc_ci(y2, se2, sm2, g2)
    out['contrasts'].append(dict(contrast='External-2: signed minus margin', delta=d0,
                                 percentile=[plo, phi], clustered_roc=[clo, chi],
                                 percentile_established=N.established(plo, phi),
                                 clustered_established=N.established(clo, chi)))
    print(f"{'External-2: signed minus margin':40s} {d0:+7.3f} {N.fmt_ci(plo,phi):>18s} "
          f"{N.fmt_ci(clo,chi):>19s}")
    rows_e1, _ = ac.load_external(1)
    P1, y1, g1 = ac.prospective(rows_e1)
    er1 = [dict(p=r['pooled_direction'], a=r['agreement'], y=int(yy), claim_id=gg)
           for r, yy, gg in zip(P1, y1, g1)]
    i1, se1, yy1, gg1 = N.loco_two_var(er1)
    sm1 = np.array([1 - abs(er1[i]['p']) for i in i1])
    d0, plo, phi, _ = N.paired_ci(yy1, se1, sm1, gg1)
    dc, clo, chi, _ = clustered_roc_ci(yy1, se1, sm1, gg1)
    out['contrasts'].append(dict(contrast='External-1: signed minus margin', delta=d0,
                                 percentile=[plo, phi], clustered_roc=[clo, chi],
                                 percentile_established=N.established(plo, phi),
                                 clustered_established=N.established(clo, chi)))
    print(f"{'External-1: signed minus margin':40s} {d0:+7.3f} {N.fmt_ci(plo,phi):>18s} "
          f"{N.fmt_ci(clo,chi):>19s}")
    print('\nwritten', N.save_json(out, 'R5_inference.json'))


if __name__ == '__main__':
    main()
