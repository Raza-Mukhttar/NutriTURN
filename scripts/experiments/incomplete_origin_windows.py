"""R6: can the three "not evaluated" rows of the census table be evaluated?

Those rows are H=5 at origin 2022 (window [2022,2027)), H=10 at 2019 ([2019,2029)) and H=10 at
2022 ([2022,2032)). Each window ends after the archive's last complete year, 2025, so the
full-window endpoint is undefined: the records do not exist yet. What DOES exist is a prefix of
each window -- 4 of 5 years, 7 of 10, and 4 of 10.

Two questions, answered separately.

A. What is observable on the prefix? The cohort is rebuilt with the horizon set to the observed
   number of years, which is exactly what cohort_unit does for any H, and the census columns are
   recomputed: candidates, accrued, censored, events, rate, and the claim-disjoint margin and
   two-variable AUROCs.

B. Would a prefix evaluation agree with the real one? This is the question that decides whether
   the prefix numbers may be reported as the cell. For every origin whose FULL window is complete,
   the endpoint is recomputed on the same prefix length and compared with the full-window endpoint
   on the units accrued under both. The disagreement rate is the error a prefix row would carry.

  python analyses/nfx_R6_incomplete_origins.py
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
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

# (horizon, origin, years of the window inside the archive)
TARGETS = [(5, 2022, 4), (10, 2019, 7), (10, 2022, 4)]


def units_at(Y, H, S):
    out = []
    for cid, (yrs, codes) in S.items():
        u = N.cohort_unit(cid, Y, H, yrs, codes)
        if u:
            out.append(u)
    return out


def main():
    S = N.streams('primary')
    out = {'prefix_census': [], 'prefix_fidelity': []}

    print('R6  the three "window extends past 2025" rows of the census table\n')
    print('A. what is observable on the part of each window that is inside the archive\n')
    print(f"{'cell':22s} {'window':18s} {'observed':10s} {'cand':>5s} {'accr':>5s} {'cens':>5s} "
          f"{'ev':>4s} {'rate':>6s}  {'margin':>7s} {'two-var':>8s}")
    for H, Y, Ho in TARGETS:
        u = units_at(Y, Ho, S)
        for r in u:
            r['origin'] = Y
        acc = [r for r in u if r['accrued']]
        ev = int(sum(r['y'] for r in acc))
        s2, y, g, _ = N.cohort_loco_two_var(u, Ho)
        sm = np.array([1 - abs(r['p']) for r in u if r['accrued']])
        am = N.auroc(y, sm) if len(np.unique(y)) > 1 else float('nan')
        a2 = N.auroc(y, s2) if len(np.unique(y)) > 1 else float('nan')
        rec = dict(horizon=H, origin=Y, observed_years=Ho,
                   window=f'[{Y}, {Y+H})', observed_window=f'[{Y}, {Y+Ho})',
                   candidates=len(u), accrued=len(acc), censored=len(u) - len(acc),
                   events=ev, rate=ev / len(acc) if acc else None,
                   margin_auroc=float(am), two_var_auroc=float(a2))
        out['prefix_census'].append(rec)
        print(f"{'H=' + str(H) + ' @ ' + str(Y):22s} {rec['window']:18s} "
              f"{str(Ho) + ' of ' + str(H) + ' yr':10s} {len(u):5d} {len(acc):5d} "
              f"{len(u)-len(acc):5d} {ev:4d} {ev/len(acc) if acc else 0:6.3f}  "
              f"{am:7.3f} {a2:8.3f}")

    print('\nB. would a prefix of that length reproduce the full-window endpoint?')
    print('   measured on every origin whose FULL window is complete\n')
    print(f"{'comparison':26s} {'origins':22s} {'accr both':>9s} {'agree':>7s} "
          f"{'rate':>7s} {'ev full':>8s} {'ev prefix':>10s} {'missed':>7s} {'false':>6s}")
    for H, Yt, Ho in TARGETS:
        origins = [Y for Y in N.ORIGINS if Y + H - 1 <= N.LAST_COMPLETE_YEAR]
        nb = ag = evf = evp = miss = false = 0
        accf = accp = 0
        for Y in origins:
            uf = {r['claim_id']: r for r in units_at(Y, H, S)}
            up = {r['claim_id']: r for r in units_at(Y, Ho, S)}
            for c in uf:
                if c not in up:
                    continue
                accf += uf[c]['accrued']; accp += up[c]['accrued']
                if not (uf[c]['accrued'] and up[c]['accrued']):
                    continue
                nb += 1
                a, b = uf[c]['y'], up[c]['y']
                ag += int(a == b); evf += a; evp += b
                miss += int(a == 1 and b == 0)      # real event the prefix does not show
                false += int(a == 0 and b == 1)     # prefix event that the full window undoes
        rec = dict(horizon=H, prefix_years=Ho, target_origin=Yt,
                   origins=[int(o) for o in origins], accrued_full=accf, accrued_prefix=accp,
                   accrued_both=nb, agree=ag, agreement=ag / nb if nb else None,
                   events_full=evf, events_prefix=evp, missed=miss, false_events=false)
        out['prefix_fidelity'].append(rec)
        print(f"{str(Ho) + ' of ' + str(H) + ' yr (for H=' + str(H) + '@' + str(Yt) + ')':26s} "
              f"{str(origins[0]) + '-' + str(origins[-1]) + ' (' + str(len(origins)) + ')':22s} "
              f"{nb:9d} {ag:7d} {ag/nb if nb else 0:7.3f} {evf:8d} {evp:10d} {miss:7d} {false:6d}")
        print(f"{'':26s} accrual: {accf} units accrue on the full window, {accp} on the "
              f"{Ho}-year prefix")
    print('\nwritten', N.save_json(out, 'R6_incomplete_origins.json'))


if __name__ == '__main__':
    main()
