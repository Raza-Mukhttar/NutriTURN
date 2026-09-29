"""newfab_exp reanalyses: E1, E3, E4, E5 and the E12 sampling frame.

  python analyses/nutrimature_reanalyses.py rebuild
  python analyses/nutrimature_reanalyses.py coef
  python analyses/nutrimature_reanalyses.py common
  python analyses/nutrimature_reanalyses.py stratum  --annotation primary
  python analyses/nutrimature_reanalyses.py decisive --deltas 0.1 0.2
  python analyses/nutrimature_reanalyses.py union    --out union_events.csv

Every subcommand first reproduces the archived cell counts and stops if it cannot.
Statistics: claim-clustered percentile bootstrap, 2,000 resamples, seed 3; "established"
means the 95% interval excludes zero.
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
import os, sys, json, argparse, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

EXPECT = {'primary': (71, 23), 'second': (52, 16), 'consensus': (55, 19)}
# published two-variable leave-one-claim-out AUROC per annotation (Tables 2 and 28)
EXPECT_AUROC = {'primary': 0.948, 'second': 0.857, 'consensus': 0.898}
ANNOTATIONS = ('primary', 'second', 'consensus')


def gate(verbose=True):
    """Reproduce 1,012 units and the event counts of all three annotations, or stop."""
    got = {}
    for a in ANNOTATIONS:
        rows = N.unit_table(a)
        ev = sum(r['y'] for r in rows)
        pc = len({r['claim_id'] for r in rows if r['y']})
        assert len(rows) == 1012, f'{a}: {len(rows)} units, expected 1012'
        got[a] = dict(units=len(rows), claims=len({r['claim_id'] for r in rows}), events=ev,
                      positive_claims=pc, sparse=sum(r['sparse'] for r in rows),
                      sparse_events=sum(r['sparse'] and r['y'] for r in rows),
                      matches_paper=bool((ev, pc) == EXPECT[a]))
        if verbose:
            m = 'OK ' if got[a]['matches_paper'] else 'MISMATCH'
            print(f'  {m} {a:10s} units {len(rows)} events {ev} (expected {EXPECT[a][0]}) '
                  f'positive claims {pc} (expected {EXPECT[a][1]})')
    if not all(g['matches_paper'] for g in got.values()):
        raise SystemExit('rebuild gate failed; not proceeding')
    # cross-check (p_t, a_t) against the frozen feature builder, and the two-variable AUROC
    sys.path.insert(0, os.path.join(_NT, 'scripts'))
    import astra_common as ac
    for a in ANNOTATIONS:
        lab = N.label_column(a); rows = N.unit_table(a); bad = 0
        for r in rows:
            yrs, codes, pmids = ac.stream(r['claim_id']); t = r['cutoff']
            cb = np.array([N.CODE.get(lab.get((r['claim_id'], str(q)), ''), 99) for q in pmids])
            f = ac.direction_feats(yrs[yrs < t], cb[yrs < t], t, 'A')
            if abs(f['agreement'] - r['a']) > 1e-9 or abs(f['pooled_direction'] - r['p']) > 1e-9:
                bad += 1
        idx, s2, y, g = N.loco_two_var(rows)
        au = N.auroc(y, s2)
        got[a]['feature_mismatches_vs_frozen_builder'] = bad
        got[a]['two_var_auroc'] = au
        got[a]['two_var_auroc_matches_paper'] = bool(abs(au - EXPECT_AUROC[a]) < 0.0015)
        if verbose:
            print(f"  {'OK ' if bad == 0 and got[a]['two_var_auroc_matches_paper'] else 'MISMATCH'} "
                  f'{a:10s} (p,a) vs frozen builder: {bad} mismatches | two-var AUROC {au:.4f} '
                  f'(expected {EXPECT_AUROC[a]})')
        if bad or not got[a]['two_var_auroc_matches_paper']:
            raise SystemExit(f'feature/AUROC gate failed for {a}; not proceeding')
    return got


def cohort_gate(verbose=True):
    co = N.build_cohort('primary')
    want = {3: (756, 58), 5: (625, 45), 10: (492, 35)}
    got = {}
    for H, (ua, ea) in want.items():
        d = [r for r in co if r['horizon'] == H]
        a = [r for r in d if r['accrued']]
        got[H] = dict(candidates=len(d), accrued=len(a), censored=len(d) - len(a),
                      events=sum(r['y'] for r in a), matches_paper=(len(a), sum(r['y'] for r in a)) == (ua, ea))
        if verbose:
            print(f"  {'OK ' if got[H]['matches_paper'] else 'MISMATCH'} calendar H{H}: "
                  f"accrued {len(a)} (expected {ua}) events {sum(r['y'] for r in a)} (expected {ea})")
    # published calendar two-variable and margin AUROCs (Table 3)
    WANT = {3: (0.913, 0.907), 5: (0.920, 0.897), 10: (0.876, 0.867)}
    for H in (3, 5, 10):
        d = [r for r in co if r['horizon'] == H]
        s2, y, g, _ = N.cohort_loco_two_var(d, H)
        sm = np.array([1 - abs(r['p']) for r in d if r['accrued']])
        a2, am = N.auroc(y, s2), N.auroc(y, sm)
        ok = abs(a2 - WANT[H][0]) < 0.0015 and abs(am - WANT[H][1]) < 0.0015
        got[H]['two_var_auroc'] = a2; got[H]['margin_auroc'] = am
        got[H]['auroc_matches_paper'] = bool(ok)
        if verbose:
            print(f"  {'OK ' if ok else 'MISMATCH'} calendar H{H}: two-var {a2:.4f} "
                  f'(expected {WANT[H][0]})  margin {am:.4f} (expected {WANT[H][1]})')
        if not ok:
            raise SystemExit(f'calendar AUROC gate failed at H{H}; not proceeding')
    if not all(g['matches_paper'] for g in got.values()):
        raise SystemExit('calendar gate failed; not proceeding')
    return got


# ------------------------------------------------------------------ E1
def cmd_rebuild(args):
    print('E1 rebuild check')
    u = gate()
    c = cohort_gate()
    out = {'development': u, 'calendar': {f'H{k}': v for k, v in c.items()},
           'matches_paper': True,
           'note': 'inputs and endpoint rebuilt from associations.csv under the frozen contract'}
    print('\nmatches_paper: true (all three annotations and all three calendar horizons)')
    print('written', N.save_json(out, 'E1_rebuild.json'))


def cmd_coef(args):
    print('E1 coefficient scale\n')
    gate(verbose=False)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    res = {}
    for a in ANNOTATIONS:
        rows = N.unit_table(a)
        X = np.array([[r['p'], r['a']] for r in rows], float)
        y = np.array([r['y'] for r in rows])
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(sc.transform(X), y)
        cp, ca = float(m.coef_[0][0]), float(m.coef_[0][1])
        c0 = float(m.intercept_[0])
        sdp, sda = float(sc.scale_[0]), float(sc.scale_[1])
        mp, ma = float(sc.mean_[0]), float(sc.mean_[1])
        # unstandardise, then substitute the signed-branch identity a = (1+|p|)/2
        rp, ra = cp / sdp, ca / sda
        r0 = c0 - cp * mp / sdp - ca * ma / sda
        b0, b1, b2 = r0 + ra * 0.5, rp, ra * 0.5
        res[a] = dict(standardised_p=cp, standardised_a=ca, standardised_intercept=c0,
                      sd_p=sdp, sd_a=sda,
                      raw_b0=b0, raw_b1=b1, raw_b2=b2,
                      note='raw b0,b1,b2 for logit = b0 + b1 p + b2 |p| on the signed branch, '
                           'obtained by substituting a = (1+|p|)/2')
        print(f'{a:10s} standardised: p {cp:+.3f}  a {ca:+.3f}  intercept {c0:+.3f}')
        print(f'{"":10s} raw signed-branch: b0 {b0:+.3f}  b1 {b1:+.3f}  b2 {b2:+.3f}')
    p = res['primary']
    print()
    print(f"paper value 3.0 matches the STANDARDISED coefficient on p_t ({p['standardised_p']:+.3f}), "
          f"not the raw b1 ({p['raw_b1']:+.3f})")
    res['verdict'] = ('standardised coefficient on p_t is +%.2f; raw b1 is %+.2f' %
                      (p['standardised_p'], p['raw_b1']))
    print('written', N.save_json(res, 'E1_coef.json'))


# ------------------------------------------------------------------ E3
def _cell(rows, mask, label, refit):
    """AUROCs and paired intervals on a population, refitting inside it when refit is True."""
    if refit:
        idx, s2, y, g = N.loco_two_var(rows, subset=mask)
    else:
        idx_all, s_all, y_all, g_all = N.loco_two_var(rows, subset=None)
        keep = np.array([bool(mask[i]) for i in idx_all])
        idx, s2, y, g = idx_all[keep], s_all[keep], y_all[keep], g_all[keep]
    sm = N.margin_score(rows, idx)
    sb = _stationary(rows, idx)
    d2, lo2, hi2, _ = N.paired_ci(y, s2, sm, g)
    db, lob, hib, _ = N.paired_ci(y, sb, sm, g)
    a2lo, a2hi, _ = N.ci(y, s2, g)
    return dict(cell=label, refit=bool(refit), units=len(idx), claims=len(np.unique(g)),
                events=int(y.sum()), positive_claims=len(np.unique(g[y == 1])),
                margin_auroc=N.auroc(y, sm), margin_auprc=N.auprc(y, sm),
                stationary_auroc=N.auroc(y, sb),
                two_var_auroc=N.auroc(y, s2), two_var_auprc=N.auprc(y, s2),
                two_var_ci=[a2lo, a2hi],
                signed_minus_margin=[d2, lo2, hi2], signed_minus_margin_established=N.established(lo2, hi2),
                stationary_minus_margin=[db, lob, hib]), (idx, s2, sm, sb, y, g)


_BB = {}


def _stationary(rows, idx):
    """Count-predicted operational Beta-Binomial probability, training-fitted count model."""
    key = id(rows)
    if key not in _BB:
        _BB[key] = _stationary_all(rows)
    return _BB[key][idx]


def _stationary_all(rows):
    sys.path.insert(0, os.path.join(N.GPTPRO, 'scripts'))
    import pro_common as pc
    y = np.array([r['y'] for r in rows]); g = np.array([r['claim_id'] for r in rows])
    n_pre = np.array([r['h'] + r['l'] + r['n0'] for r in rows], float)
    rate5 = np.array([r.get('accrual_5y', 0.0) for r in rows], float)
    m_obs = np.array([r['m_signed'] for r in rows], float)
    out = np.zeros(len(rows))
    rng = np.random.default_rng(N.SEED)
    for c in np.unique(g):
        te = g == c; tr = ~te
        b, sd = pc.fit_count_model(np.log1p(n_pre[tr]), np.log1p(rate5[tr]), np.log1p(m_obs[tr]))
        for i in np.where(te)[0]:
            ms = pc.draw_counts(b, sd, np.log1p(n_pre[i]), np.log1p(rate5[i]), rng)
            vals, cnts = np.unique(ms, return_counts=True)
            pr = sum(cn * pc.bb_change_prob(rows[i]['h'], rows[i]['l'], int(v), s0=rows[i]['s'])
                     for v, cn in zip(vals, cnts)) / cnts.sum()
            out[i] = pr
    return out


def common_support_mask(annotations):
    """Units with at least 8 signed pre-cutoff records under every listed annotation."""
    tabs = {a: N.unit_table(a) for a in annotations}
    n = len(next(iter(tabs.values())))
    mask = np.ones(n, bool)
    for a in annotations:
        mask &= np.array([r['sparse'] == 0 for r in tabs[a]])
    return mask, tabs


def cmd_common(args):
    print('E3 common-support refit\n')
    gate(verbose=False)
    ann = list(ANNOTATIONS)
    # two masks: the published one (the two pipelines, 1,002 units, as in Table 37) and the
    # stricter one that also requires adequate support under consensus.
    mask2, _ = common_support_mask(('primary', 'second'))
    mask3, tabs = common_support_mask(ANNOTATIONS)
    print(f'common support, two pipelines : {int(mask2.sum())} of {len(mask2)} units '
          f'({int((~mask2).sum())} excluded)   <- the published mask')
    print(f'common support, all three     : {int(mask3.sum())} of {len(mask3)} units '
          f'({int((~mask3).sum())} excluded)\n')
    rows_out = []
    for a in ann:
        rows = tabs[a]
        for r, acc in zip(rows, N.build_cohort_accrual5(rows)):
            r['accrual_5y'] = acc
        full, _ = _cell(rows, np.ones(len(rows), bool), f'{a}: full cohort', refit=True)
        rest, _ = _cell(rows, mask2, f'{a}: common support (2 pipe), restricted', refit=False)
        refit, _ = _cell(rows, mask2, f'{a}: common support (2 pipe), REFITTED', refit=True)
        refit3, _ = _cell(rows, mask3, f'{a}: common support (all 3), REFITTED', refit=True)
        for c in (full, rest, refit, refit3):
            rows_out.append(c)
            d, lo, hi = c['signed_minus_margin']
            print(f"{c['cell']:52s} units {c['units']:5d} events {c['events']:3d} "
                  f"margin {c['margin_auroc']:.3f} stat {c['stationary_auroc']:.3f} "
                  f"signed {c['two_var_auroc']:.3f}  signed-margin {d:+.3f} {N.fmt_ci(lo,hi)}"
                  f"{'  ESTABLISHED' if c['signed_minus_margin_established'] else ''}")
        print()
    N.save_csv(rows_out, 'E3_common_support.csv')
    sec = [c for c in rows_out if c['cell'].startswith('second: common support (2 pipe), REFITTED')][0]
    d, lo, hi = sec['signed_minus_margin']
    verdict = ('refitted second-annotation signed-minus-margin is %+.3f %s and IS established as negative'
               % (d, N.fmt_ci(lo, hi))) if (hi < 0) else (
              'refitted second-annotation signed-minus-margin is %+.3f %s and is NOT established'
              % (d, N.fmt_ci(lo, hi)))
    print('verdict:', verdict)
    print('written', N.save_json({'cells': rows_out,
                                  'common_support_units_two_pipelines': int(mask2.sum()),
                                  'common_support_units_all_three': int(mask3.sum()),
                                  'verdict': verdict}, 'E3_common_support.json'))


# ------------------------------------------------------------------ E4
def cmd_stratum(args):
    a = args.annotation
    print(f'E4 within-class AUROC, annotation = {a}\n')
    gate(verbose=False)
    rows = N.unit_table(a)
    for r, acc in zip(rows, N.build_cohort_accrual5(rows)):
        r['accrual_5y'] = acc
    idx, s2, y, g = N.loco_two_var(rows)
    idxb, s2b, yb, gb = N.loco_two_var(rows, class_blind=True)
    sm = N.margin_score(rows, idx)
    sign = np.array([N.sgn0(rows[i]['p']) for i in idx])
    res = {'annotation': a, 'per_class': [], 'pooled_within_class': {}}
    names = {1: 'HARMFUL-majority', -1: 'PROTECTIVE-majority', 0: 'no majority (tie or sparse)'}
    for v in (1, -1, 0):
        k = sign == v
        if k.sum() == 0 or len(np.unique(y[k])) < 2:
            res['per_class'].append(dict(stratum=names[v], units=int(k.sum()), events=int(y[k].sum()),
                                         margin=None, two_var=None, class_blind=None))
            print(f'{names[v]:28s} units {int(k.sum()):4d} events {int(y[k].sum()):3d}  (AUROC undefined)')
            continue
        # per-class intervals, claim-clustered inside the stratum, plus the paired difference
        # against each reference, so every row of the table carries an interval
        gk = g[k]
        cm_lo, cm_hi, _ = N.ci(y[k], sm[k], gk)
        c2_lo, c2_hi, _ = N.ci(y[k], s2[k], gk)
        cb_lo, cb_hi, _ = N.ci(yb[k], s2b[k], gk)
        d2, d2lo, d2hi, _ = N.paired_ci(y[k], s2[k], sm[k], gk)
        db, dblo, dbhi, _ = N.paired_ci(yb[k], s2b[k], sm[k], gk)
        row = dict(stratum=names[v], units=int(k.sum()), events=int(y[k].sum()),
                   margin=N.auroc(y[k], sm[k]), margin_ci=[cm_lo, cm_hi],
                   two_var=N.auroc(y[k], s2[k]), two_var_ci=[c2_lo, c2_hi],
                   class_blind=N.auroc(yb[k], s2b[k]), class_blind_ci=[cb_lo, cb_hi],
                   two_var_minus_margin=[d2, d2lo, d2hi],
                   two_var_minus_margin_established=N.established(d2lo, d2hi),
                   class_blind_minus_margin=[db, dblo, dbhi],
                   class_blind_minus_margin_established=N.established(dblo, dbhi))
        res['per_class'].append(row)
        print(f"{names[v]:28s} units {row['units']:4d} events {row['events']:3d}")
        print(f"{'':28s}   margin      {row['margin']:.3f} {N.fmt_ci(cm_lo, cm_hi)}")
        print(f"{'':28s}   two-var     {row['two_var']:.3f} {N.fmt_ci(c2_lo, c2_hi)}   "
              f"minus margin {d2:+.3f} {N.fmt_ci(d2lo, d2hi)}"
              f"{'  ESTABLISHED' if N.established(d2lo, d2hi) else ''}")
        print(f"{'':28s}   class-blind {row['class_blind']:.3f} {N.fmt_ci(cb_lo, cb_hi)}   "
              f"minus margin {db:+.3f} {N.fmt_ci(dblo, dbhi)}"
              f"{'  ESTABLISHED' if N.established(dblo, dbhi) else ''}")
    # pooled within-class: concordant pairs counted only inside a stratum
    def pooled_within(y_, s_, sign_):
        num = den = 0.0
        for v in np.unique(sign_):
            k = sign_ == v
            pos = s_[k][y_[k] == 1]; neg = s_[k][y_[k] == 0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            d = pos[:, None] - neg[None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum()
            den += d.size
        return float(num / den) if den else float('nan')

    def pooled_boot(y_, s_, sign_, g_):
        vals = []
        for ix in N.claim_boot_pairs(y_, g_):
            vals.append(pooled_within(y_[ix], s_[ix], sign_[ix]))
        v = np.array([x for x in vals if np.isfinite(x)])
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    def pooled_paired(y_, sa, sb_, sign_, g_):
        vals = []
        for ix in N.claim_boot_pairs(y_, g_):
            va = pooled_within(y_[ix], sa[ix], sign_[ix]); vb = pooled_within(y_[ix], sb_[ix], sign_[ix])
            if np.isfinite(va) and np.isfinite(vb):
                vals.append(va - vb)
        v = np.array(vals)
        return (pooled_within(y_, sa, sign_) - pooled_within(y_, sb_, sign_),
                float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    print()
    for nm_, sc in (('margin rule', sm), ('two-variable', s2), ('class-blind two-variable', s2b)):
        pw = pooled_within(y, sc, sign)
        lo, hi = pooled_boot(y, sc, sign, g)
        res['pooled_within_class'][nm_] = dict(auroc=pw, ci=[lo, hi])
        print(f'pooled within-class AUROC, {nm_:26s} {pw:.3f} {N.fmt_ci(lo,hi)}')
    for nm_, sc in (('two-variable', s2), ('class-blind two-variable', s2b)):
        d, lo, hi = pooled_paired(y, sc, sm, sign, g)
        res['pooled_within_class'][nm_]['minus_margin'] = [d, lo, hi]
        res['pooled_within_class'][nm_]['established'] = N.established(lo, hi)
        print(f'  {nm_} minus margin: {d:+.3f} {N.fmt_ci(lo,hi)}'
              f"{'  ESTABLISHED' if N.established(lo,hi) else '  not established'}")
    print('written', N.save_json(res, f'E4_stratum_{a}.json'))


# ------------------------------------------------------------------ E5
def cmd_decisive(args):
    print(f'E5 stricter endpoints, deltas = {args.deltas}\n')
    gate(verbose=False)
    mask_cs, tabs = common_support_mask(ANNOTATIONS)
    out = []
    for a in ANNOTATIONS:
        rows = tabs[a]
        for r, acc in zip(rows, N.build_cohort_accrual5(rows)):
            r['accrual_5y'] = acc
        for delta in args.deltas:
            # decisive: both sides must be decisive, |p| >= delta and |p_post| >= delta
            keep = np.array([abs(r['p']) >= delta and abs(r['p_post']) >= delta for r in rows])
            if keep.sum() < 50 or len({rows[i]['y'] for i in np.where(keep)[0]}) < 2:
                print(f'{a} delta={delta}: too few units ({int(keep.sum())}) or one class only; skipped')
                continue
            c, _ = _cell(rows, keep, f'{a}: decisive delta={delta}', refit=True)
            c['delta'] = delta; c['annotation'] = a
            c['excluded_indecisive'] = int((~keep).sum())
            out.append(c)
            d, lo, hi = c['signed_minus_margin']
            print(f"{a:10s} delta {delta}: kept {c['units']:4d} events {c['events']:3d} "
                  f"excluded {c['excluded_indecisive']:4d}  margin {c['margin_auroc']:.3f} "
                  f"stat {c['stationary_auroc']:.3f} signed {c['two_var_auroc']:.3f}  "
                  f"signed-margin {d:+.3f} {N.fmt_ci(lo,hi)}"
                  f"{'  ESTABLISHED' if N.established(lo,hi) else ''}")
        c, _ = _cell(rows, mask_cs, f'{a}: common support (refitted)', refit=True)
        c['delta'] = None; c['annotation'] = a; c['excluded_indecisive'] = int((~mask_cs).sum())
        out.append(c)
        d, lo, hi = c['signed_minus_margin']
        print(f"{a:10s} common support: kept {c['units']:4d} events {c['events']:3d}  "
              f"margin {c['margin_auroc']:.3f} stat {c['stationary_auroc']:.3f} "
              f"signed {c['two_var_auroc']:.3f}  signed-margin {d:+.3f} {N.fmt_ci(lo,hi)}"
              f"{'  ESTABLISHED' if N.established(lo,hi) else ''}")
        print()
    N.save_csv(out, 'E5_decisive.csv')
    print('written', N.save_json({'cells': out}, 'E5_decisive.json'))


# ------------------------------------------------------------------ E12 frame only
def cmd_union(args):
    print('E12 sampling frame (union events). No human labels are produced by this script.\n')
    gate(verbose=False)
    tp, ts = N.unit_table('primary'), N.unit_table('second')
    rows = []
    for rp, rs in zip(tp, ts):
        if rp['y'] or rs['y']:
            rows.append(dict(claim_id=rp['claim_id'], cutoff=rp['cutoff'],
                             event_primary=rp['y'], event_second=rs['y'],
                             p_primary=round(rp['p'], 4), p_second=round(rs['p'], 4),
                             abs_p_primary=round(abs(rp['p']), 4),
                             sparse_primary=rp['sparse'], sparse_second=rs['sparse']))
    # 50 matched non-event units: same claims where possible, nearest |p_t|
    ev_claims = {r['claim_id'] for r in rows}
    cand = [dict(claim_id=rp['claim_id'], cutoff=rp['cutoff'], abs_p_primary=round(abs(rp['p']), 4))
            for rp, rs in zip(tp, ts) if not (rp['y'] or rs['y'])]
    tgt = np.array([r['abs_p_primary'] for r in rows])
    same = [c for c in cand if c['claim_id'] in ev_claims]
    pool = same if len(same) >= 50 else cand
    order = sorted(pool, key=lambda c: min(abs(c['abs_p_primary'] - t) for t in tgt))
    matched = order[:50]
    N.save_csv(rows, args.out)
    N.save_csv(matched, 'E12_matched_nonevents.csv')
    print(f'union events: {len(rows)} units in {len(ev_claims)} claims '
          f'(primary-only {sum(1 for r in rows if r["event_primary"] and not r["event_second"])}, '
          f'second-only {sum(1 for r in rows if r["event_second"] and not r["event_primary"])}, '
          f'both {sum(1 for r in rows if r["event_primary"] and r["event_second"])})')
    print(f'matched non-events drawn: {len(matched)} (from {"the same claims" if pool is same else "all non-event units"})')
    print('written', os.path.join(N.RESULTS, args.out), 'and E12_matched_nonevents.csv')
    print('\nE12 requires two human annotators with biomedical training. This script only builds the')
    print('sampling frame and guideline; it does not and must not generate human labels.')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('rebuild')
    sub.add_parser('coef')
    sub.add_parser('common')
    s = sub.add_parser('stratum'); s.add_argument('--annotation', default='primary')
    s = sub.add_parser('decisive'); s.add_argument('--deltas', nargs='+', type=float, default=[0.1, 0.2])
    s = sub.add_parser('union'); s.add_argument('--out', default='E12_union_events.csv')
    a = ap.parse_args()
    {'rebuild': cmd_rebuild, 'coef': cmd_coef, 'common': cmd_common,
     'stratum': cmd_stratum, 'decisive': cmd_decisive, 'union': cmd_union}[a.cmd](a)


if __name__ == '__main__':
    main()
