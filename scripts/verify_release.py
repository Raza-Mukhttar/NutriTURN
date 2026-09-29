"""Verify that this release reproduces the rebuild checks stated in the paper's
Reproducibility appendix, using only files inside this directory.

  python scripts/verify_release.py
"""
import os, sys
import numpy as np
_NT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ('lib', 'experiments', 'pipeline', 'figures', ''):
    sys.path.insert(0, os.path.join(_NT, 'scripts', _p))
import nfx_common as N

EXPECT_EVENTS = {'primary': 71, 'second': 52, 'consensus': 55}
EXPECT_AUROC = {'primary': 0.948, 'second': 0.857, 'consensus': 0.898}
EXPECT_CAL = {3: (756, 58, 0.913), 5: (625, 45, 0.920), 10: (492, 35, 0.876)}

def main():
    print('NutriTURN release verification')
    print(f'  root        {_NT}')
    print(f'  data        {N.DATA}')
    print(f'  snapshot    {N.SNAPD}\n')
    bad = 0
    for f in ('units.csv', 'associations.csv', 'claims.csv'):
        p = os.path.join(N.DATA, f)
        print(f'  {f:22s} {"found" if os.path.exists(p) else "MISSING":8s} '
              f'{os.path.getsize(p)/1e6:8.2f} MB' if os.path.exists(p) else f'  {f} MISSING')
        bad += not os.path.exists(p)

    print('\n  source cohort: 1,012 units, one annotation per column')
    print(f"  {'annotation':12s} {'units':>6s} {'events':>7s} {'exp':>5s} "
          f"{'AUROC two-var':>14s} {'exp':>6s}  ok")
    for a in ('primary', 'second', 'consensus'):
        rows = N.unit_table(a)
        for r, acc in zip(rows, N.build_cohort_accrual5(rows)):
            r['accrual_5y'] = acc
        ev = sum(r['y'] for r in rows)
        idx, s2, y, g = N.loco_two_var(rows)
        au = N.auroc(y, s2)
        ok = (len(rows) == 1012 and ev == EXPECT_EVENTS[a]
              and abs(au - EXPECT_AUROC[a]) < 5e-4)
        bad += not ok
        print(f'  {a:12s} {len(rows):6d} {ev:7d} {EXPECT_EVENTS[a]:5d} '
              f'{au:14.4f} {EXPECT_AUROC[a]:6.3f}  {"yes" if ok else "NO"}')

    print('\n  calendar cohort, complete windows')
    print(f"  {'horizon':>8s} {'units':>6s} {'exp':>5s} {'events':>7s} {'exp':>5s} "
          f"{'AUROC':>8s} {'exp':>6s}  ok")
    co = N.build_cohort('primary')
    for H in (3, 5, 10):
        d = [r for r in co if r['horizon'] == H]
        s2, y, g, _ = N.cohort_loco_two_var(d, H)
        u, e, a_ = EXPECT_CAL[H]
        au = N.auroc(y, s2)
        ok = len(y) == u and int(y.sum()) == e and abs(au - a_) < 5e-4
        bad += not ok
        print(f'  H={H:<6d} {len(y):6d} {u:5d} {int(y.sum()):7d} {e:5d} '
              f'{au:8.4f} {a_:6.3f}  {"yes" if ok else "NO"}')
    print(f'\n  {"ALL CHECKS PASS" if bad == 0 else str(bad) + " CHECK(S) FAILED"}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
