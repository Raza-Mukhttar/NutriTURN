"""Two one-sided equivalence tests (TOST) on the learned-minus-reference AUROC contrasts.

For every contrast the paper reports with a claim-clustered percentile interval, this script
re-draws the same 2,000 claim resamples (same seed, same cell definitions as the wild-bootstrap
and clustered-ROC tables) and reports

  * the 95% interval, checked against the published value before anything else is reported;
  * the 90% interval, which is the TOST interval at alpha = 0.05: the contrast is equivalent to
    zero within +/- d exactly when the 90% interval lies inside (-d, +d);
  * the smallest d for which that holds (max absolute end of the 90% interval);
  * the verdict at each bound in BOUNDS.

The bounds were not fixed before the data were seen. The smallest equivalence bound is therefore
the primary quantity; verdicts at fixed bounds are a reading aid.

  python equivalence_tests.py /path/to/NutriTURN-Repository
"""
import os, sys, json
import numpy as np

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('NUTRITURN_ROOT', '.'))
if not os.path.isfile(os.path.join(ROOT, '.nutriturn_root')):
    raise SystemExit('pass the repository root (the folder holding .nutriturn_root)')
for p in ('audit_pipeline/lib', 'data_preprocessing/lib', 'audit_experiments',
          'audit_experiments/figures', 'data_annotation', 'data_preparation', ''):
    sys.path.insert(0, os.path.join(ROOT, *p.split('/')) if p else ROOT)
import nfx_common as N
import wild_cluster_bootstrap as W

HERE = os.path.dirname(os.path.abspath(__file__))
BOUNDS = (0.02, 0.05)
# published delta and 95% interval, used only to confirm each cell is the cell of the paper
PUBLISHED = {
    'development, primary: signed minus margin':      (+0.067,  0.017,  0.120),
    'development, primary: signed minus class-blind': (+0.051,  0.013,  0.092),
    'development, second: signed minus margin':       (-0.096, -0.166, -0.037),
    'development, consensus: signed minus margin':    (-0.030, -0.094,  0.023),
    'calendar H3: signed minus margin':               (+0.006, -0.012,  0.023),
    'calendar H5: signed minus margin':               (+0.023, -0.005,  0.053),
    'calendar H10: signed minus margin':              (+0.008, -0.032,  0.047),
    'External-1, frozen transfer: signed minus margin': (+0.055, -0.052, 0.106),
    'External-2, frozen transfer: signed minus margin': (-0.074, -0.123, -0.024),
}


def draws(y, sa, sb, g):
    v = []
    for ix in N.claim_boot_pairs(y, g, N.BOOT, N.SEED):
        if len(np.unique(y[ix])) < 2:
            continue
        v.append(N.auroc(y[ix], sa[ix]) - N.auroc(y[ix], sb[ix]))
    return np.array(v)


def external_cells():
    import pro_common as pc
    import two_variable_transfer as tv
    ac, nm = pc.ac, pc.nm
    rows_src, _ = ac.load_units()
    P, ys, _ = ac.prospective(rows_src)
    f2 = tv.fit_frozen(nm.X_of(P, nm.COMPACT7)[:, :2], ys)[0]
    out = []
    for k in (1, 2):
        E0, _ = ac.load_external(k)
        E = [r for r in E0 if float(r['n_post_t']) >= 20]
        y = np.array([int(r['y_sub']) for r in E]); g = np.array([r['claim_id'] for r in E])
        X = nm.X_of(E, nm.COMPACT7)[:, :2]
        out.append((f'External-{k}, frozen transfer: signed minus margin', y, f2(X), 1 - np.abs(X[:, 0]), g))
    return out


def main():
    res = []
    for lab, y, sa, sb, g in list(W.cells()) + external_cells():
        y = np.asarray(y); sa = np.asarray(sa, float); sb = np.asarray(sb, float); g = np.asarray(g)
        d0 = float(N.auroc(y, sa) - N.auroc(y, sb))
        v = draws(y, sa, sb, g)
        lo95, hi95 = (float(np.percentile(v, q)) for q in (2.5, 97.5))
        lo90, hi90 = (float(np.percentile(v, q)) for q in (5, 95))
        pub = PUBLISHED.get(lab)
        ok = None if pub is None else bool(abs(d0 - pub[0]) < 6e-4 and abs(lo95 - pub[1]) < 1.5e-3
                                           and abs(hi95 - pub[2]) < 1.5e-3)
        est = N.established(lo95, hi95)
        r = dict(cell=lab, units=int(len(y)), events=int(y.sum()), clusters=int(len(np.unique(g))),
                 delta=d0, ci95=[lo95, hi95], established=bool(est), matches_published=ok,
                 ci90=[lo90, hi90], smallest_equivalence_bound=float(max(abs(lo90), abs(hi90))),
                 p_lower={f'{b:.2f}': float(np.mean(v <= -b)) for b in BOUNDS},
                 p_upper={f'{b:.2f}': float(np.mean(v >= b)) for b in BOUNDS},
                 equivalent={f'{b:.2f}': bool(lo90 > -b and hi90 < b) for b in BOUNDS},
                 usable_resamples=int(len(v)))
        res.append(r)
    hdr = (f"{'contrast':50s} {'units':>5s} {'ev':>3s} {'delta':>7s} {'95% interval':>18s} est pub "
           f"{'90% interval':>18s} {'min bound':>9s} " + ' '.join(f'eq@{b:.2f}' for b in BOUNDS))
    lines = [hdr]
    for r in res:
        lines.append(f"{r['cell']:50s} {r['units']:5d} {r['events']:3d} {r['delta']:+7.4f} "
                     f"{N.fmt_ci(*r['ci95'], d=4):>18s} {'yes' if r['established'] else 'no ':3s} "
                     f"{ {True: 'ok ', False: 'NO ', None: '-  '}[r['matches_published']] } "
                     f"{N.fmt_ci(*r['ci90'], d=4):>18s} {r['smallest_equivalence_bound']:9.4f} "
                     + ' '.join(f"{'yes' if r['equivalent'][f'{b:.2f}'] else 'no':>7s}" for b in BOUNDS))
    txt = '\n'.join(lines)
    print(txt)
    open(os.path.join(HERE, 'equivalence_tests_log.txt'), 'w').write(txt + '\n')
    json.dump(dict(protocol=__doc__, resamples=N.BOOT, seed=N.SEED, alpha=0.05, bounds=BOUNDS, contrasts=res),
              open(os.path.join(HERE, 'equivalence_tests.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
