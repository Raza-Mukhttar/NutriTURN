"""N2 and N5: is the second-annotation loss the agreement branch, and is the class prior a class
effect or a margin confound?

N2. The shipped agreement input a_t has two branches: the NULL share when NULL dominates, the
signed-majority share otherwise. Under the second annotation the NULL-dominant branch grows from
42 units to 332, so the input changes meaning for a third of the cohort. Rebuilding a_t without the
branch separates "the score loses under a second annotator" from "the shipped feature does not
port". Definitions: A shipped, C signed-only share with no branch, C+null adds the NULL share as a
third input.

N5. HARMFUL majorities change about five times as often as PROTECTIVE ones. That is a class effect
only if it survives adjustment for |p_t|, since HARMFUL majorities may simply be thinner.

  python analyses/nfx_branch.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def agreement_variants(rows):
    """Attach a_C (signed-only share, no branch) and the NULL share to every unit."""
    for r in rows:
        n = r['n_signed']
        r['a_C'] = (1 + abs(r['p'])) / 2 if n >= N.SPARSE_GATE else 0.0
        r['a_null'] = r['null_share']
    return rows


def loco(rows, feats, subset=None, class_blind=False):
    idx = [i for i in range(len(rows)) if (subset is None or subset[i])]
    X = np.array([[abs(rows[i]['p']) if (class_blind and f == 'p') else rows[i][f]
                   for f in feats] for i in idx], float)
    y = np.array([rows[i]['y'] for i in idx]); g = np.array([rows[i]['claim_id'] for i in idx])
    s = np.full(len(idx), np.nan)
    for c in np.unique(g):
        te = g == c; tr = ~te
        if len(np.unique(y[tr])) < 2:
            s[te] = y[tr].mean() if tr.any() else 0.0; continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000)
        m.fit(sc.transform(X[tr]), y[tr]); s[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return np.array(idx), s, y, g


def cell(rows, feats, mask, label, class_blind=False):
    idx, s, y, g = loco(rows, feats, mask, class_blind)
    sm = np.array([1 - abs(rows[i]['p']) for i in idx])
    d, lo, hi, _ = N.paired_ci(y, s, sm, g)
    return dict(cell=label, features='+'.join(feats) + (' (class-blind)' if class_blind else ''),
                units=len(idx), events=int(y.sum()), margin_auroc=N.auroc(y, sm),
                score_auroc=N.auroc(y, s), minus_margin=[d, lo, hi],
                established=N.established(lo, hi),
                non_inferior_at_0p02=bool(lo > -0.02))


def main():
    out = {'N2': [], 'N5': {}}
    print('N2 branch-free agreement under each annotation\n')
    tabs = {a: agreement_variants(N.unit_table(a)) for a in ('primary', 'second', 'consensus')}
    cs, _ = None, None
    m2 = np.ones(len(tabs['primary']), bool)
    for a in ('primary', 'second'):
        m2 &= np.array([r['sparse'] == 0 for r in tabs[a]])
    defs = [('A shipped (p, a)', ['p', 'a'], False),
            ('C signed-only (p, a_C)', ['p', 'a_C'], False),
            ('C+null (p, a_C, null share)', ['p', 'a_C', 'a_null'], False),
            ('C class-blind (|p|, a_C)', ['p', 'a_C'], True)]
    for a in ('primary', 'second', 'consensus'):
        rows = tabs[a]
        print(f'--- {a} annotation ---')
        for lab, feats, cb in defs:
            c = cell(rows, feats, np.ones(len(rows), bool), f'{a}: {lab}', cb)
            out['N2'].append(c)
            d, lo, hi = c['minus_margin']
            print(f"   {lab:30s} units {c['units']:5d} score {c['score_auroc']:.3f}  "
                  f"minus margin {d:+.3f} {N.fmt_ci(lo,hi)}"
                  f"{'  ESTABLISHED' if c['established'] else ''}"
                  f"{'  non-inferior at -0.02' if c['non_inferior_at_0p02'] else ''}")
        # common support, refitted
        c = cell(rows, ['p', 'a_C'], m2, f'{a}: C on common support (refitted)')
        out['N2'].append(c)
        d, lo, hi = c['minus_margin']
        print(f"   {'C, common support refit':30s} units {c['units']:5d} score {c['score_auroc']:.3f}  "
              f"minus margin {d:+.3f} {N.fmt_ci(lo,hi)}"
              f"{'  ESTABLISHED' if c['established'] else ''}")
        print()

    # branch stratification under the shipped definition, second annotation
    print('branch stratification under the shipped definition A (second annotation)')
    rows = tabs['second']
    idx, s, y, g = loco(rows, ['p', 'a'])
    sm = np.array([1 - abs(rows[i]['p']) for i in idx])
    br = np.array([('sparse' if rows[i]['sparse'] else
                    ('null_dominant' if rows[i]['null_share'] >= 0.5 else 'signed'))
                   for i in idx])
    out['branch'] = []
    for b in ('signed', 'null_dominant'):
        k = br == b
        if k.sum() < 50 or len(np.unique(y[k])) < 2:
            continue
        d, lo, hi, _ = N.paired_ci(y[k], s[k], sm[k], g[k])
        # and refitted inside the branch
        mask = np.zeros(len(rows), bool); mask[idx[k]] = True
        cr = cell(rows, ['p', 'a'], mask, f'second: branch {b}, refitted')
        out['branch'].append(dict(branch=b, units=int(k.sum()), events=int(y[k].sum()),
                                  restricted_minus_margin=[d, lo, hi],
                                  refit=cr))
        print(f'   {b:14s} units {int(k.sum()):4d} events {int(y[k].sum()):3d}  '
              f'restricted {d:+.3f} {N.fmt_ci(lo,hi)}'
              f"{'  EST' if N.established(lo,hi) else '     '}"
              f"   refitted {cr['minus_margin'][0]:+.3f} "
              f"{N.fmt_ci(cr['minus_margin'][1], cr['minus_margin'][2])}"
              f"{'  EST' if cr['established'] else ''}")

    # ---- N5 class effect within margin strata
    print('\n\nN5 class effect within margin strata (primary annotation)\n')
    rows = tabs['primary']
    p = np.array([r['p'] for r in rows]); ap = np.abs(p)
    y = np.array([r['y'] for r in rows]); g = np.array([r['claim_id'] for r in rows])
    cls = np.array([N.sgn0(x) for x in p])
    for v, lab in ((1, 'HARMFUL-majority'), (-1, 'PROTECTIVE-majority')):
        k = cls == v
        q = np.percentile(ap, [20, 40, 60, 80])
        print(f'   {lab:20s} units {k.sum():4d} events {int(y[k].sum()):3d}  '
              f'|p| median {np.median(ap[k]):.3f}  IQR '
              f'[{np.percentile(ap[k],25):.3f}, {np.percentile(ap[k],75):.3f}]  '
              f'share in lowest quintile {np.mean(ap[k] <= q[0]):.3f}')
        out['N5'][lab] = dict(units=int(k.sum()), events=int(y[k].sum()),
                              abs_p_median=float(np.median(ap[k])),
                              rate=float(y[k].mean()))
    print()
    q = np.percentile(ap, [20, 40, 60, 80])
    edges = [-1e9] + list(q) + [1e9]
    print(f"   {'|p| quintile':22s} {'HARMFUL n/ev/rate':>24s} {'PROTECTIVE n/ev/rate':>24s}   OR")
    ors, ws = [], []
    strata = []
    for i in range(5):
        m = (ap > edges[i]) & (ap <= edges[i + 1])
        a1 = int(((cls == 1) & m & (y == 1)).sum()); b1 = int(((cls == 1) & m & (y == 0)).sum())
        c1 = int(((cls == -1) & m & (y == 1)).sum()); d1 = int(((cls == -1) & m & (y == 0)).sum())
        n1, n0 = a1 + b1, c1 + d1
        r1 = a1 / n1 if n1 else float('nan'); r0 = c1 / n0 if n0 else float('nan')
        oR = (a1 * d1) / (b1 * c1) if b1 and c1 else float('nan')
        strata.append(dict(quintile=i + 1, harmful_n=n1, harmful_events=a1, harmful_rate=r1,
                           protective_n=n0, protective_events=c1, protective_rate=r0,
                           odds_ratio=oR))
        tot = a1 + b1 + c1 + d1
        if tot and b1 * c1:
            ors.append((a1 * d1) / tot); ws.append((b1 * c1) / tot)
        print(f'   Q{i+1} (|p|<= {edges[i+1] if i<4 else float("inf"):.3f})'.ljust(24)
              + f'{n1:6d}/{a1:3d}/{r1:.3f}'.rjust(24)
              + f'{n0:6d}/{c1:3d}/{r0:.3f}'.rjust(24)
              + (f'   {oR:.2f}' if np.isfinite(oR) else '   n/a'))
    mh = sum(ors) / sum(ws) if ws and sum(ws) else float('nan')
    out['N5']['strata'] = strata
    out['N5']['mantel_haenszel_or'] = float(mh)
    crude = ((y[cls == 1].mean()) / (1 - y[cls == 1].mean())) / \
            ((y[cls == -1].mean()) / (1 - y[cls == -1].mean()))
    out['N5']['crude_or'] = float(crude)
    print(f'\n   crude odds ratio (HARMFUL vs PROTECTIVE majority): {crude:.2f}')
    print(f'   Mantel-Haenszel odds ratio across |p| quintiles   : {mh:.2f}')
    # adjusted logistic with claim-clustered interval
    Xb = np.column_stack([(cls == 1).astype(float), ap])
    def fit(ix):
        m = LogisticRegression(C=1e6, max_iter=3000)
        m.fit(Xb[ix], y[ix]); return float(m.coef_[0][0])
    b = fit(np.arange(len(y)))
    vals = []
    for ix in N.claim_boot_pairs(y, g):
        if len(np.unique(y[ix])) < 2:
            continue
        try:
            vals.append(fit(ix))
        except Exception:
            pass
    v = np.array(vals)
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    out['N5']['adjusted_log_or'] = [b, lo, hi]
    out['N5']['adjusted_or'] = [float(np.exp(b)), float(np.exp(lo)), float(np.exp(hi))]
    print(f'   adjusted odds ratio for class, controlling |p| : {np.exp(b):.2f} '
          f'[{np.exp(lo):.2f}, {np.exp(hi):.2f}]'
          f"{'  ESTABLISHED' if (lo>0 or hi<0) else '  not established'}")
    print('\nwritten', N.save_json(out, 'N2_N5_branch_class.json'))


if __name__ == '__main__':
    main()
