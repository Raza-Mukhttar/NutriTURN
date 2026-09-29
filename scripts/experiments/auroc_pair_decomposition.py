"""R3 (their E3): exact additive decomposition of the AUROC gain, and class rates in common
margin bins.

The paper contrasts an overall signed-minus-margin gain of about +0.067 with a pooled within-class
gain of about +0.018. Those use different pair populations, so subtracting them does not quantify
the across-class contribution. This computes the exact decomposition instead.

Every positive-negative pair falls in exactly one group: within-class when both units share the
same non-zero pre-cutoff majority, across-class when one is HARMFUL-majority and the other
PROTECTIVE-majority, zero-direction when either has p_t = 0. With

  H(z) = 1 if z > 0, 0.5 if z = 0, 0 if z < 0

the contribution of group k is the sum over its pairs of H(s_i - s_j) - H(m_i - m_j), divided by
N+ x N-. By construction the three contributions add to the overall difference, which is asserted
numerically.

Margin bins are cut once on the pooled primary |p_t| over supported non-zero histories, without
using outcomes, then frozen and applied to both annotations, so the classes are compared at
comparable margins rather than on separate per-class quantiles.

  python analyses/nfx_R3_decomp.py
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


def H(z):
    return np.where(z > 0, 1.0, np.where(z == 0, 0.5, 0.0))


def decompose(y, s, m, cls):
    """Exact additive split of AUROC(s) - AUROC(m) into within, across and zero-direction pairs."""
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n1, n0 = len(pos), len(neg)
    ds = s[pos][:, None] - s[neg][None, :]
    dm = m[pos][:, None] - m[neg][None, :]
    contrib = (H(ds) - H(dm)) / (n1 * n0)
    cp = cls[pos][:, None]; cn = cls[neg][None, :]
    zero = (cp == 0) | (cn == 0)
    within = (~zero) & (cp == cn)
    across = (~zero) & (cp != cn)
    out = {}
    for name, mask in (('within', within), ('across', across), ('zero', zero)):
        npair = int(mask.sum())
        out[name] = dict(pairs=npair, share_of_pairs=npair / (n1 * n0),
                         contribution=float(contrib[mask].sum()),
                         within_group_auroc_diff=float(
                             (H(ds[mask]) - H(dm[mask])).mean()) if npair else float('nan'))
    out['overall'] = float(contrib.sum())
    return out


def main():
    out = {'annotations': {}, 'margin_bins': {}}
    tabs = {a: N.unit_table(a) for a in ('primary', 'second', 'consensus')}

    # frozen margin bins from the pooled primary |p| on supported non-zero histories
    base = [r for r in tabs['primary'] if r['n_signed'] >= N.SPARSE_GATE and r['p'] != 0]
    edges = np.percentile([abs(r['p']) for r in base], [20, 40, 60, 80])
    out['margin_bins']['edges'] = [float(x) for x in edges]
    print('R3 exact AUROC pair decomposition\n')
    print(f'frozen |p_t| bin edges from the pooled primary supported non-zero histories: '
          f'{np.round(edges, 3).tolist()}\n')

    print(f"{'annotation':11s} {'group':8s} {'pairs':>12s} {'share':>7s} {'contribution':>13s} "
          f"{'[95% CI]':>20s} {'group diff':>11s}")
    for a in ('primary', 'second', 'consensus'):
        rows = tabs[a]
        idx, s, y, g = N.loco_two_var(rows)
        m = N.margin_score(rows, idx)
        cls = np.array([N.sgn0(rows[i]['p']) for i in idx])
        d = decompose(y, s, m, cls)
        overall = N.auroc(y, s) - N.auroc(y, m)
        assert abs(d['overall'] - overall) < 1e-9, (d['overall'], overall)
        # claim-clustered intervals on each contribution
        cis = {k: [] for k in ('within', 'across', 'zero')}
        for ix in N.claim_boot_pairs(y, g):
            if len(np.unique(y[ix])) < 2:
                continue
            dd = decompose(y[ix], s[ix], m[ix], cls[ix])
            for k in cis:
                cis[k].append(dd[k]['contribution'])
        d['overall_check'] = overall
        for k in ('within', 'across', 'zero'):
            v = np.array(cis[k])
            d[k]['ci'] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            print(f"{a:11s} {k:8s} {d[k]['pairs']:12,d} {d[k]['share_of_pairs']:7.3f} "
                  f"{d[k]['contribution']:+13.4f} "
                  f"{N.fmt_ci(d[k]['ci'][0], d[k]['ci'][1], 4):>20s} "
                  f"{d[k]['within_group_auroc_diff']:+11.4f}")
        print(f"{a:11s} {'TOTAL':8s} {n1n0(y):12,d} {1.0:7.3f} {d['overall']:+13.4f}"
              f"   (matches AUROC difference {overall:+.4f})")
        out['annotations'][a] = d
        print()

    # ---- class rates in the frozen margin bins
    print('class event rates within the frozen margin bins')
    for a in ('primary', 'second'):
        rows = tabs[a]
        print(f'\n   {a} annotation')
        print(f"      {'bin':16s} {'HARMFUL n/ev/claims/rate':>30s} {'PROTECTIVE n/ev/claims/rate':>32s}")
        tab = []
        for i in range(5):
            lo = -np.inf if i == 0 else edges[i - 1]
            hi = np.inf if i == 4 else edges[i]
            sel = [r for r in rows if r['n_signed'] >= N.SPARSE_GATE and r['p'] != 0
                   and lo < abs(r['p']) <= hi]
            row = dict(bin=i + 1, lo=float(lo), hi=float(hi))
            for v, lab in ((1, 'harmful'), (-1, 'protective')):
                k = [r for r in sel if N.sgn0(r['p']) == v]
                ev = sum(r['y'] for r in k)
                row[f'{lab}_n'] = len(k); row[f'{lab}_events'] = ev
                row[f'{lab}_claims'] = len({r['claim_id'] for r in k})
                row[f'{lab}_rate'] = ev / len(k) if k else float('nan')
            tab.append(row)
            print(f"      Q{i+1} ({'-inf' if i==0 else f'{lo:.2f}'}, "
                  f"{'inf' if i==4 else f'{hi:.2f}'}]".ljust(22)
                  + f"{row['harmful_n']:5d}/{row['harmful_events']:3d}/"
                    f"{row['harmful_claims']:3d}/{row['harmful_rate']:.3f}".rjust(24)
                  + f"{row['protective_n']:5d}/{row['protective_events']:3d}/"
                    f"{row['protective_claims']:3d}/{row['protective_rate']:.3f}".rjust(30))
        out['margin_bins'][a] = tab
    print('\nwritten', N.save_json(out, 'R3_pair_decomposition.json'))


def n1n0(y):
    return int((y == 1).sum() * (y == 0).sum())


if __name__ == '__main__':
    main()
