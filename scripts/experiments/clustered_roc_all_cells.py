"""R5b: the clustered-ROC column on the seven cells of Table 23, and a diagnosis of the
mislabelled rows in the first R5 run.

The first R5 run reported seven rows, but only three of them are cells of tab:wild. This script
does two things.

1. Clustered ROC on the seven cells of tab:wild, using the same cell definitions as the wild
   cluster bootstrap that already fills that table (nfx_wildboot.cells) and the same resample
   seed for the percentile column. Each cell's delta and percentile interval are checked against
   the published values before its clustered interval is reported, so a cell can only be entered
   once it is shown to be the same cell.

2. What the four unusable rows were scoring. The "within-class" rows dropped the eight
   zero-direction units and then took an UNRESTRICTED AUROC on the remaining 1,004; tab:protocol's
   within-class cell restricts the PAIRS to a shared majority class on those same units. The
   "External" rows refitted the two-variable model inside the external cohort with claim-held-out
   folds and used every prospective unit; tab:protocol's external cells apply the frozen
   source-fitted model unchanged to the future-eligible units (n_post_t >= 20). Both alternatives
   are computed here side by side.

  python analyses/nfx_R5b_table23.py            # parts 1, 2
  python analyses/nfx_R5b_table23.py --refit    # also the refitting column on the two open cells
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
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N
import wild_cluster_bootstrap as W
from cluster_aware_inference import clustered_roc_ci, refit_delta

ASTRA_S = _os.path.join(_NT, 'scripts', 'lib')
GP_S = os.path.join(N.GPTPRO, 'scripts')

# tab:wild as published: cell label in nfx_wildboot.cells() -> (row name, delta, lo, hi)
TAB23 = [
    ('development, primary: signed minus margin',      'Prim., signed - margin',      +0.067,  0.017,  0.120),
    ('development, primary: signed minus class-blind', 'Prim., signed - class-blind', +0.051,  0.013,  0.092),
    ('development, second: signed minus margin',       'Second, signed - margin',     -0.096, -0.166, -0.037),
    ('development, consensus: signed minus margin',    'Intersect., signed - margin', -0.030, -0.094,  0.023),
    ('calendar H3: signed minus margin',               'Calendar H=3',                +0.006, -0.012,  0.023),
    ('calendar H5: signed minus margin',               'Calendar H=5',                +0.023, -0.005,  0.053),
    ('calendar H10: signed minus margin',              'Calendar H=10',               +0.008, -0.032,  0.047),
]


def stratified_roc_ci(y, sa, sb, g, strata):
    """Within-stratum (same-class pairs only) AUROC difference with a claim-clustered variance.

    The pooled statistic is the pair-count weighted mean of the per-stratum differences, which is
    what a pair restriction to a shared majority class computes.  Each stratum contributes its own
    DeLong influence, scaled by its pair weight; the influences are then summed within claim.
    """
    ks = [k for k in np.unique(strata)
          if (y[strata == k] == 1).any() and (y[strata == k] == 0).any()]
    w = {k: int((y[strata == k] == 1).sum()) * int((y[strata == k] == 0).sum()) for k in ks}
    T = sum(w.values())
    delta = 0.0
    infl = np.zeros(len(y))
    per = {}
    for k in ks:
        m = strata == k
        yk = y[m]
        V10a, V01a = W.placements(yk, sa[m]); V10b, V01b = W.placements(yk, sb[m])
        n1 = int((yk == 1).sum()); n0 = int((yk == 0).sum())
        dk = float(V10a.mean() - V10b.mean())
        per[k] = (dk, float(N.auroc(yk, sa[m])), float(N.auroc(yk, sb[m])), n1, n0)
        delta += (w[k] / T) * dk
        ik = np.zeros(int(m.sum()))
        ik[yk == 1] = (V10a - V10b - dk) / n1
        ik[yk == 0] = (V01a - V01b - dk) / n0
        infl[m] = (w[k] / T) * ik
    h = np.array([infl[g == c].sum() for c in np.unique(g)])
    se = float(np.sqrt((h ** 2).sum()))
    return delta, delta - 1.96 * se, delta + 1.96 * se, se, T, per


def part1(do_refit, reps):
    print('PART 1  clustered ROC on the seven cells of Table 23 (tab:wild)\n')
    have = {lab: (y, sa, sb, g) for lab, y, sa, sb, g in W.cells()}
    print(f"{'row (tab:wild)':30s} {'delta':>7s} {'published':>18s} {'this run':>18s} "
          f"{'match':>6s} {'clustered ROC':>20s} {'SE':>7s}  pos.cl")
    out = []
    for lab, row, pd_, plo_, phi_ in TAB23:
        y, sa, sb, g = have[lab]
        y = np.asarray(y); sa = np.asarray(sa, float); sb = np.asarray(sb, float); g = np.asarray(g)
        d0, plo, phi, _ = N.paired_ci(y, sa, sb, g)
        ok = (abs(d0 - pd_) < 6e-4 and abs(plo - plo_) < 1.5e-3 and abs(phi - phi_) < 1.5e-3)
        dc, clo, chi, se = clustered_roc_ci(y, sa, sb, g)
        pos = len(np.unique(g[y == 1]))
        rec = dict(row=row, cell=lab, delta=d0, percentile=[plo, phi],
                   published=[pd_, plo_, phi_], cell_verified=bool(ok),
                   clustered_roc=[clo, chi], clustered_se=se,
                   clustered_established=N.established(clo, chi),
                   clusters=int(len(np.unique(g))), positive_clusters=int(pos))
        out.append(rec)
        print(f"{row:30s} {d0:+7.3f} {N.fmt_ci(plo_, phi_):>18s} {N.fmt_ci(plo, phi):>18s} "
              f"{'yes' if ok else 'NO':>6s} {N.fmt_ci(clo, chi):>19s}"
              f"{'*' if rec['clustered_established'] else ' '} {se:7.4f}  {pos}/{len(np.unique(g))}")
    bad = [r['row'] for r in out if not r['cell_verified']]
    print(f"\n   cells whose delta or percentile interval does not reproduce the published value: "
          f"{bad if bad else 'none'}")
    if do_refit:
        print(f'\n   refitting bootstrap on the two open source-cohort cells, {reps} replicates, '
              f'seed {N.SEED}')
        for a, feats_b in (('primary', 'class_blind'), ('consensus', 'margin')):
            rows = N.unit_table(a)
            for r, acc in zip(rows, N.build_cohort_accrual5(rows)):
                r['accrual_5y'] = acc
            cl = np.unique(np.array([r['claim_id'] for r in rows]))
            rng = np.random.default_rng(N.SEED)
            vals = []
            for _ in range(reps):
                pick = rng.choice(cl, size=len(cl), replace=True)
                v = (refit_delta_classblind(rows, pick) if feats_b == 'class_blind'
                     else refit_delta(rows, pick))
                if v is not None and np.isfinite(v):
                    vals.append(v)
            v = np.array(vals)
            lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
            row = ('Prim., signed - class-blind' if feats_b == 'class_blind'
                   else 'Intersect., signed - margin')
            rec = [r for r in out if r['row'] == row][0]
            rec['refit'] = [float(v.mean()), lo, hi]
            rec['refit_replicates'] = len(v)
            rec['refit_established'] = N.established(lo, hi)
            print(f"      {row:30s} {N.fmt_ci(lo, hi):>18s}"
                  f"{'*' if rec['refit_established'] else ' '}  {len(v)} usable replicates")
    return out


def refit_delta_classblind(rows, idx_claims):
    """Signed minus class-blind, both models refitted inside the replicate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    units, = [[r for c in idx_claims for r in rows if r['claim_id'] == c]]
    if len(units) < 60:
        return None
    y = np.array([u['y'] for u in units])
    orig = np.array([u['claim_id'] for u in units])
    if len(np.unique(y)) < 2:
        return None
    Xs = np.array([[u['p'], u['a']] for u in units], float)
    Xb = np.array([[abs(u['p']), u['a']] for u in units], float)
    ss = np.full(len(units), np.nan); sb = np.full(len(units), np.nan)
    for c in np.unique(orig):
        te = orig == c; tr = ~te
        if len(np.unique(y[tr])) < 2:
            ss[te] = sb[te] = (y[tr].mean() if tr.any() else 0.0); continue
        for X, dest in ((Xs, ss), (Xb, sb)):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000)
            m.fit(sc.transform(X[tr]), y[tr])
            dest[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return float(N.auroc(y, ss) - N.auroc(y, sb))


def part2():
    print('\n\nPART 2  what the four unusable rows of the first R5 run were scoring\n')
    out = {}

    # ---- within-class -------------------------------------------------------------------
    print('2a  within-class.  tab:protocol row 4 is a PAIR restriction; the R5 row was a UNIT '
          'restriction.\n')
    print(f"   {'annotation':10s} {'statistic':52s} {'units':>6s} {'ev':>4s} {'delta':>7s} "
          f"{'percentile':>18s} {'clustered ROC':>20s}")
    for a in ('primary', 'second'):
        rows = N.unit_table(a)
        for r, acc in zip(rows, N.build_cohort_accrual5(rows)):
            r['accrual_5y'] = acc
        idx, s2, y, g = N.loco_two_var(rows)
        sm = N.margin_score(rows, idx)
        sign = np.array([N.sgn0(rows[i]['p']) for i in idx])
        keep = sign != 0
        # (a) exactly what the first R5 run did
        d0, plo, phi, _ = N.paired_ci(y[keep], s2[keep], sm[keep], g[keep])
        _, clo, chi, _ = clustered_roc_ci(y[keep], s2[keep], sm[keep], g[keep])
        print(f"   {a:10s} {'zero-direction units dropped, pairs UNRESTRICTED (the R5 row)':52s} "
              f"{int(keep.sum()):6d} {int(y[keep].sum()):4d} {d0:+7.3f} "
              f"{N.fmt_ci(plo, phi):>18s} {N.fmt_ci(clo, chi):>19s}")
        out[f'within_class_{a}_unrestricted_pairs'] = dict(
            delta=d0, percentile=[plo, phi], clustered_roc=[clo, chi],
            units=int(keep.sum()), events=int(y[keep].sum()))
        # (b) the published cell: same-class pairs only
        dS, sloS, shiS, seS, T, per = stratified_roc_ci(y[keep], s2[keep], sm[keep], g[keep],
                                                        sign[keep])
        dp, plo2, phi2, _ = N.paired_ci_stratified(y[keep], s2[keep], sm[keep], g[keep],
                                                   sign[keep]) \
            if hasattr(N, 'paired_ci_stratified') else (dS, np.nan, np.nan, None)
        print(f"   {a:10s} {'same-class pairs only (tab:protocol row 4)':52s} "
              f"{int(keep.sum()):6d} {int(y[keep].sum()):4d} {dS:+7.3f} "
              f"{'(see below)':>18s} {N.fmt_ci(sloS, shiS):>19s}")
        for k in sorted(per):
            dk, ak, bk, n1, n0 = per[k]
            nm_ = 'HARM-majority' if k > 0 else 'PROT-majority'
            print(f"      {nm_:16s} units {n1 + n0:5d}  events {n1:3d}  margin {bk:.3f}  "
                  f"score {ak:.3f}  delta {dk:+.3f}  pairs {n1 * n0:,}")
        print(f"      pooled over {T:,} same-class pairs: delta {dS:+.4f}, clustered SE {seS:.4f}")
        out[f'within_class_{a}_same_class_pairs'] = dict(
            delta=dS, clustered_roc=[sloS, shiS], clustered_se=seS, pairs=int(T),
            per_class={('HARM' if k > 0 else 'PROT'): dict(
                delta=per[k][0], score_auroc=per[k][1], margin_auroc=per[k][2],
                events=per[k][3], non_events=per[k][4]) for k in per})
        print()

    # ---- external -----------------------------------------------------------------------
    print('2b  external.  tab:protocol row 5 applies the FROZEN source fit to the future-eligible '
          'units;\n    the R5 row refitted inside the external cohort and used every prospective '
          'unit.\n')
    sys.path.insert(0, GP_S); sys.path.insert(0, ASTRA_S)
    import pro_common as pc
    import two_variable_transfer as tv
    ac, nm = pc.ac, pc.nm
    rows_src, _ = ac.load_units()
    P, ys, gs = ac.prospective(rows_src)
    X7 = nm.X_of(P, nm.COMPACT7)
    f2 = tv.fit_frozen(X7[:, :2], ys)[0]
    print(f"   {'cohort':11s} {'statistic':52s} {'units':>6s} {'ev':>4s} {'delta':>7s} "
          f"{'percentile':>18s} {'clustered ROC':>20s}")
    for k_ in (1, 2):
        E0, _ = ac.load_external(k_)
        # (a) exactly what the first R5 run did
        Pe, ye0, ge0 = ac.prospective(E0)
        er = [dict(p=r['pooled_direction'], a=r['agreement'], y=int(yy), claim_id=gg)
              for r, yy, gg in zip(Pe, ye0, ge0)]
        i1, sr, yr, gr = N.loco_two_var(er)
        smr = np.array([1 - abs(er[i]['p']) for i in i1])
        d0, plo, phi, _ = N.paired_ci(yr, sr, smr, gr)
        _, clo, chi, _ = clustered_roc_ci(yr, sr, smr, gr)
        print(f"   External-{k_} {'refit inside the cohort, all prospective units (the R5 row)':52s} "
              f"{len(yr):6d} {int(yr.sum()):4d} {d0:+7.3f} {N.fmt_ci(plo, phi):>18s} "
              f"{N.fmt_ci(clo, chi):>19s}")
        out[f'external_{k_}_refit_in_domain'] = dict(delta=d0, percentile=[plo, phi],
                                                     clustered_roc=[clo, chi], units=int(len(yr)),
                                                     events=int(yr.sum()))
        # (b) the published cell: frozen source fit, future-eligible units
        E = [r for r in E0 if float(r['n_post_t']) >= 20]
        ye = np.array([int(r['y_sub']) for r in E]); ge = np.array([r['claim_id'] for r in E])
        Xe2 = nm.X_of(E, nm.COMPACT7)[:, :2]
        su = 1 - np.abs(Xe2[:, 0]); s2 = f2(Xe2)
        d1, qlo, qhi, _ = N.paired_ci(ye, s2, su, ge)
        _, dlo, dhi, dse = clustered_roc_ci(ye, s2, su, ge)
        print(f"   External-{k_} {'frozen source fit, future-eligible units (tab:protocol row 5)':52s} "
              f"{len(ye):6d} {int(ye.sum()):4d} {d1:+7.3f} {N.fmt_ci(qlo, qhi):>18s} "
              f"{N.fmt_ci(dlo, dhi):>19s}"
              f"{'*' if N.established(dlo, dhi) else ' '}")
        print(f"      margin AUROC {N.auroc(ye, su):.3f}   frozen two-variable AUROC "
              f"{N.auroc(ye, s2):.3f}   claims {len(np.unique(ge))}")
        out[f'external_{k_}_frozen_transfer'] = dict(
            delta=d1, percentile=[qlo, qhi], clustered_roc=[dlo, dhi], clustered_se=dse,
            units=int(len(ye)), events=int(ye.sum()),
            margin_auroc=float(N.auroc(ye, su)), score_auroc=float(N.auroc(ye, s2)),
            clustered_established=N.established(dlo, dhi))
        print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refit', action='store_true')
    ap.add_argument('--replicates', type=int, default=2000)
    a_ = ap.parse_args()
    t23 = part1(a_.refit, a_.replicates)
    diag = part2()
    print('written', N.save_json({'table23_cells': t23, 'mislabelled_row_diagnosis': diag},
                                 'R5b_table23.json'))


if __name__ == '__main__':
    main()
