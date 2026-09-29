"""N6: wild cluster bootstrap-t for the headline paired AUROC differences.

With 23 positive claims the claim-clustered percentile bootstrap can under-cover. This recomputes
every headline difference with inference built for a small number of clusters.

Method. AUROC is a two-sample U-statistic, so it has an exact influence representation through
DeLong's placement values. For scores a and b, each positive i contributes
(V10a_i - V10b_i - delta)/n1 and each negative j contributes (V01a_j - V01b_j - delta)/n0. Summing
those inside a claim gives one influence score per claim, and the cluster-robust standard error is
the root sum of their squares. The wild cluster bootstrap-t then perturbs the claim scores with
Webb six-point weights, which is the recommended choice when clusters are few, and inverts the
resulting t distribution. A cluster jackknife interval is reported as a second check.

  python analyses/nfx_wildboot.py
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

WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def placements(y, s):
    """DeLong placement values: V10 for positives, V01 for negatives."""
    pos = s[y == 1]; neg = s[y == 0]
    n1, n0 = len(pos), len(neg)
    order = np.argsort(neg, kind='stable'); ns = neg[order]
    lt = np.searchsorted(ns, pos, side='left')
    le = np.searchsorted(ns, pos, side='right')
    V10 = (lt + 0.5 * (le - lt)) / n0
    order2 = np.argsort(pos, kind='stable'); ps = pos[order2]
    gt = n1 - np.searchsorted(ps, neg, side='right')
    ge = n1 - np.searchsorted(ps, neg, side='left')
    V01 = (gt + 0.5 * (ge - gt)) / n1
    return V10, V01


def claim_influence(y, sa, sb, g):
    """One influence score per claim for delta = AUROC(sa) - AUROC(sb)."""
    V10a, V01a = placements(y, sa)
    V10b, V01b = placements(y, sb)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    delta = float(V10a.mean() - V10b.mean())
    infl = np.zeros(len(y))
    infl[y == 1] = (V10a - V10b - delta) / n1
    infl[y == 0] = (V01a - V01b - delta) / n0
    cl = np.unique(g)
    h = np.array([infl[g == c].sum() for c in cl])
    return delta, h, cl


def wild_ci(delta, h, draws=9999, seed=N.SEED):
    se = float(np.sqrt((h ** 2).sum()))
    if se == 0:
        return delta, delta, se
    rng = np.random.default_rng(seed)
    G = len(h)
    t = np.empty(draws)
    for b in range(draws):
        w = rng.choice(WEBB, size=G)
        hb = w * h
        num = hb.sum()
        den = np.sqrt((hb ** 2).sum())
        t[b] = num / den if den > 0 else 0.0
    lo_q, hi_q = np.percentile(t, [2.5, 97.5])
    return delta - hi_q * se, delta - lo_q * se, se


def jack_ci(y, sa, sb, g):
    cl = np.unique(g)
    full = N.auroc(y, sa) - N.auroc(y, sb)
    vals = []
    for c in cl:
        k = g != c
        if len(np.unique(y[k])) < 2:
            continue
        vals.append(N.auroc(y[k], sa[k]) - N.auroc(y[k], sb[k]))
    v = np.array(vals); G = len(v)
    ps = G * full - (G - 1) * v
    se = float(np.std(ps, ddof=1) / np.sqrt(G))
    from scipy import stats
    t = stats.t.ppf(0.975, G - 1)
    return full - t * se, full + t * se, se


def cells():
    out = []
    for a in ('primary', 'second', 'consensus'):
        rows = N.unit_table(a)
        for r, acc in zip(rows, N.build_cohort_accrual5(rows)):
            r['accrual_5y'] = acc
        idx, s2, y, g = N.loco_two_var(rows)
        sm = N.margin_score(rows, idx)
        out.append((f'development, {a}: signed minus margin', y, s2, sm, g))
        idxb, s2b, yb, gb = N.loco_two_var(rows, class_blind=True)
        out.append((f'development, {a}: signed minus class-blind', y, s2, s2b, g))
    co = N.build_cohort('primary')
    for H in (3, 5, 10):
        d = [r for r in co if r['horizon'] == H]
        s2, y, g, _ = N.cohort_loco_two_var(d, H)
        sm = np.array([1 - abs(r['p']) for r in d if r['accrued']])
        out.append((f'calendar H{H}: signed minus margin', y, s2, sm, g))
    return out


def main():
    print('N6 wild cluster bootstrap-t, Webb six-point weights, 9,999 draws\n')
    res = []
    print(f"{'cell':46s} {'delta':>7s} {'percentile':>18s} {'wild cluster':>20s} "
          f"{'jackknife':>20s}  clusters")
    for lab, y, sa, sb, g in cells():
        y = np.asarray(y); sa = np.asarray(sa, float); sb = np.asarray(sb, float); g = np.asarray(g)
        d0, plo, phi, _ = N.paired_ci(y, sa, sb, g)
        delta, h, cl = claim_influence(y, sa, sb, g)
        wlo, whi, se = wild_ci(delta, h)
        jlo, jhi, jse = jack_ci(y, sa, sb, g)
        pos_cl = len(np.unique(g[y == 1]))
        r = dict(cell=lab, delta=d0, percentile=[plo, phi], wild=[wlo, whi],
                 jackknife=[jlo, jhi], clusters=int(len(cl)), positive_clusters=int(pos_cl),
                 percentile_established=N.established(plo, phi),
                 wild_established=N.established(wlo, whi),
                 jackknife_established=N.established(jlo, jhi))
        res.append(r)
        print(f"{lab:46s} {d0:+7.3f} {N.fmt_ci(plo,phi):>18s}"
              f"{'*' if r['percentile_established'] else ' '}"
              f"{N.fmt_ci(wlo,whi):>19s}"
              f"{'*' if r['wild_established'] else ' '}"
              f"{N.fmt_ci(jlo,jhi):>19s}"
              f"{'*' if r['jackknife_established'] else ' '}"
              f"  {pos_cl}/{len(cl)}")
    lost = [r['cell'] for r in res if r['percentile_established'] and not r['wild_established']]
    print(f"\ncells established under the percentile interval but NOT under the wild cluster "
          f"bootstrap: {lost if lost else 'none'}")
    print('written', N.save_json({'cells': res, 'lost_under_wild': lost}, 'N6_wild_bootstrap.json'))


if __name__ == '__main__':
    main()
