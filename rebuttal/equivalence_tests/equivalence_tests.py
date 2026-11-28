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

  python equivalence_tests.py /path/to/NutriTURN-Repository [--bashir /path/to/extracted_info.txt]

Contrasts covered: the development, calendar and frozen-transfer cells; the frozen cohort under both
annotations; the signed-only (definition C) reference contrasts for all four annotators; and the
Bashir et al. review-update contrast. The Bashir data are theirs and are not redistributed: pass
the path to their released extracted_info.txt (github.com/dsurian/rules_cochranereviews) to include
that row; it is skipped otherwise.
"""
import os, sys, json
import numpy as np

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('root', nargs='?', default=os.environ.get('NUTRITURN_ROOT', '.'))
_ap.add_argument('--bashir', default=os.environ.get('BASHIR_DATA'))
ARGS = _ap.parse_args()
ROOT = os.path.abspath(ARGS.root)
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
    'frozen cohort, primary: signed minus stationary (prespecified primary)': (-0.0397, -0.1067, 0.0132),
    'frozen cohort, primary: signed minus margin (prespecified secondary)': (-0.0059, -0.0429, 0.0338),
    'frozen cohort, second (own cohort): signed minus stationary': (-0.0949, -0.2236, 0.0072),
    'frozen cohort, second (own cohort): signed minus margin': (-0.0952, -0.2155, 0.0060),
    'frozen cohort, second (primary cohort): signed minus stationary': (-0.0756, -0.1918, 0.0224),
    'frozen cohort, second (primary cohort): signed minus margin': (-0.0497, -0.1572, 0.0454),
    'signed-only, Llama-8B: C minus margin': (+0.0436, -0.005, 0.091),
    'signed-only, Llama-8B: C minus stationary': (+0.0150, -0.024, 0.049),
    'signed-only, Qwen-7B: C minus margin': (-0.0008, -0.028, 0.018),
    'signed-only, Qwen-7B: C minus stationary': (-0.0103, -0.043, 0.010),
    'signed-only, Llama-70B (4-bit): C minus margin': (+0.0023, -0.012, 0.017),
    'signed-only, Llama-70B (4-bit): C minus stationary': (-0.0116, -0.039, 0.007),
    'signed-only, Qwen-72B (4-bit): C minus margin': (-0.0025, -0.022, 0.010),
    'signed-only, Qwen-72B (4-bit): C minus stationary': (-0.0102, -0.036, 0.007),
    'Bashir et al. review updates: random forest minus trials added': (+0.0421, -0.0482, 0.1390),
}
# published values given to three decimals are compared at that precision
TOL3 = {k for k, v in PUBLISHED.items() if all(abs(round(x, 3) - x) < 1e-12 for x in v[1:])}


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


def frozen_cells():
    """The 61-claim frozen cohort, rebuilt with the frozen contract from the released annotations."""
    import collections, glob
    import core_reanalyses as C
    FC = os.path.join(ROOT, 'data_release', '03_annotations', 'frozen_cohort')
    ORIGINS, H, CAND, ACCR = (2004, 2007, 2010, 2013, 2016, 2019), 5, 40, 10
    G8 = N.SPARSE_GATE

    def streams(pipe):
        per = {}
        for p in sorted(glob.glob(os.path.join(FC, pipe, '*.jsonl'))):
            for line in open(p):
                if line.strip():
                    d = json.loads(line)
                    per.setdefault(d['claim_id'], []).append((int(d['year']), N.CODE.get(d['label_full'], 99)))
        return {c: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for c, v in per.items()}

    def unit(cid, Y, yrs, codes):
        reason = 'not_candidate' if (yrs < Y).sum() < CAND else None
        pre = (yrs < Y) & (codes != 99)
        h, l, n0 = (int((codes[pre] == v).sum()) for v in (1, -1, 0))
        n = h + l
        p = (h - l) / n if n >= G8 else 0.0
        s_ = N.sgn0(p)
        res_ = h + l + n0
        ns_ = n0 / res_ if res_ else 0.0
        a = 0.0 if n < G8 else (ns_ if ns_ >= 0.5 else (1 + abs(p)) / 2)
        rate5 = int(((yrs >= Y - 5) & (yrs < Y) & (codes != 99) & (codes != 0)).sum()) / 5.0
        w = (yrs >= Y) & (yrs < Y + H) & (codes != 99)
        ph, pl = int((codes[w] == 1).sum()), int((codes[w] == -1).sum())
        m = ph + pl
        if m < ACCR:
            reason = reason or 'not_accrued'
        if m == 0:
            return None, reason or 'empty'
        return dict(claim_id=cid, origin=Y, h=h, l=l, n0=n0, n_signed=n, p=p, s=s_, a=a,
                    null_share=ns_, accrual_5y=rate5, m_signed=m, y=int(N.sgn0((ph - pl) / m) != s_)), reason

    def protocol(S):
        rows = []
        for cid in sorted(S):
            for Y in ORIGINS:
                r, why = unit(cid, Y, *S[cid])
                if r is not None and not why:
                    rows.append(r)
        return rows

    def cells(rows, tag):
        idx, A, y, g = N.loco_two_var(rows, features=('p', 'a'))
        stat = C._stationary_all(rows)[idx]
        marg = np.array([1 - abs(rows[i]['p']) for i in idx])
        sfx = ' (prespecified primary)' if tag == 'primary' else ''
        sfx2 = ' (prespecified secondary)' if tag == 'primary' else ''
        return [(f'frozen cohort, {tag}: signed minus stationary{sfx}', y, A, stat, g),
                (f'frozen cohort, {tag}: signed minus margin{sfx2}', y, A, marg, g)]

    Sp, Ss = streams('primary'), streams('second')
    prim = protocol(Sp)
    out = cells(prim, 'primary') + cells(protocol(Ss), 'second (own cohort)')
    fixed = []
    for r0 in prim:
        r, _ = unit(r0['claim_id'], r0['origin'], *Ss[r0['claim_id']])
        if r is not None:
            fixed.append(r)
    return out + cells(fixed, 'second (primary cohort)')


def signed_only_cells():
    """Definition C (branch-free agreement) against both references, 1,012 source units."""
    import csv
    import core_reanalyses as C
    G8 = N.SPARSE_GATE
    LC = os.path.join(ROOT, 'data_release', '03_annotations', 'large_cells')

    def first(path):
        o = {}
        for line in open(path):
            if line.strip():
                r = json.loads(line); k = (r['claim_id'], str(r['pmid']))
                if k not in o:
                    o[k] = r['label_full']
        return o

    def table(lab):
        per = {}
        for r in N.assoc():
            per.setdefault(r['claim_id'], []).append((r['year'], N.CODE.get(lab.get((r['claim_id'], r['pmid']), ''), 99)))
        S = {c: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for c, v in per.items()}
        rows = []
        for u in csv.DictReader(open(os.path.join(N.DATA, 'units.csv'))):
            cid, t = u['claim_id'], int(u['cutoff'])
            yrs, codes = S[cid]
            pre, post = (yrs < t) & (codes != 99), (yrs >= t) & (codes != 99)
            h, l, n0 = (int((codes[pre] == v).sum()) for v in (1, -1, 0))
            n = h + l
            p = ((h - l) / n) if n >= G8 else 0.0
            s_ = N.sgn0(p)
            res_ = h + l + n0
            ns_ = n0 / res_ if res_ else 0.0
            a = 0.0 if n < G8 else (ns_ if ns_ >= 0.5 else (1 + abs(p)) / 2)
            ph, pl = int((codes[post] == 1).sum()), int((codes[post] == -1).sum())
            m = ph + pl
            pp = (ph - pl) / m if m else 0.0
            rows.append(dict(claim_id=cid, cutoff=t, h=h, l=l, n0=n0, n_signed=n, p=p, s=s_, a=a,
                             c=0.0 if n < G8 else (1 + abs(p)) / 2, null_share=ns_, m_signed=m,
                             y=int(N.sgn0(pp) != s_), sparse=int(n < G8)))
        for r, acc in zip(rows, N.build_cohort_accrual5(rows)):
            r['accrual_5y'] = acc
        return rows

    labs = [('Llama-8B', N.label_column('primary')), ('Qwen-7B', N.label_column('second')),
            ('Llama-70B (4-bit)', first(os.path.join(LC, 'cell_llama70.jsonl'))),
            ('Qwen-72B (4-bit)', first(os.path.join(LC, 'cell_qwen72.jsonl')))]
    extra = os.environ.get('FP70_CELL')
    if extra and os.path.exists(extra):
        labs.append(('Llama-70B (bfloat16)', first(extra)))
    out = []
    for name, lab in labs:
        t = table(lab)
        idx, SC, y, g = N.loco_two_var(t, features=('p', 'c'))
        stat = C._stationary_all(t)[idx]
        marg = np.array([1 - abs(t[i]['p']) for i in idx])
        out += [(f'signed-only, {name}: C minus margin', y, SC, marg, g),
                (f'signed-only, {name}: C minus stationary', y, SC, stat, g)]
    return out


def bashir_cells():
    """Bashir et al. (2021) review-update pairs: refitted published random forest vs trials added."""
    if not ARGS.bashir or not os.path.exists(ARGS.bashir):
        print('Bashir data not given (--bashir); that row is skipped', flush=True)
        return []
    import csv, datetime
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    rows = []
    for r in csv.DictReader(open(ARGS.bashir), delimiter='\t'):
        d2 = datetime.datetime.strptime(r['Search date pub 2'].strip(), '%d %B %Y')
        d3 = datetime.datetime.strptime(r['Search date pub 3'].strip(), '%d %B %Y')
        t2, t3 = float(r['Total trials pub2']), float(r['Total trials pub3'])
        p2, p3 = float(r['Total numb participants pub2']), float(r['Total numb participants pub3'])
        rows.append(dict(y=int(r['Conclusion']), trials_v2=t2, part_v2=p2, coverage=p2 / max(p3, 1.0),
                         elapsed_days=(d3 - d2).days, new_trials=t3 - t2))
    feats = ['trials_v2', 'part_v2', 'coverage', 'elapsed_days']
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r['y'] for r in rows])
    Xt = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    s = np.full(len(rows), np.nan)
    for tr, te in StratifiedKFold(n_splits=10, shuffle=True, random_state=N.SEED).split(Xt, y):
        m = RandomForestClassifier(n_estimators=500, random_state=N.SEED, class_weight='balanced', n_jobs=-1)
        m.fit(Xt[tr], y[tr])
        s[te] = m.predict_proba(Xt[te])[:, 1]
    ref = np.array([r['new_trials'] for r in rows], float)
    return [('Bashir et al. review updates: random forest minus trials added', y, s, ref, np.arange(len(rows)))]


def main():
    res = []
    for lab, y, sa, sb, g in list(W.cells()) + external_cells() + frozen_cells() + signed_only_cells() + bashir_cells():
        y = np.asarray(y); sa = np.asarray(sa, float); sb = np.asarray(sb, float); g = np.asarray(g)
        d0 = float(N.auroc(y, sa) - N.auroc(y, sb))
        v = draws(y, sa, sb, g)
        lo95, hi95 = (float(np.percentile(v, q)) for q in (2.5, 97.5))
        lo90, hi90 = (float(np.percentile(v, q)) for q in (5, 95))
        pub = PUBLISHED.get(lab)
        tl = 1.5e-3 if (lab in TOL3 or lab.startswith(('development', 'calendar', 'External'))) else 1.5e-4
        ok = None if pub is None else bool(abs(d0 - pub[0]) < 6e-4 and abs(lo95 - pub[1]) < tl
                                           and abs(hi95 - pub[2]) < tl)
        est = N.established(lo95, hi95)
        r = dict(cell=lab, units=int(len(y)), events=int(y.sum()), clusters=int(len(np.unique(g))),
                 delta=d0, ci95=[lo95, hi95], established=bool(est), matches_published=ok,
                 ci90=[lo90, hi90], smallest_equivalence_bound=float(max(abs(lo90), abs(hi90))),
                 p_lower={f'{b:.2f}': float(np.mean(v <= -b)) for b in BOUNDS},
                 p_upper={f'{b:.2f}': float(np.mean(v >= b)) for b in BOUNDS},
                 equivalent={f'{b:.2f}': bool(lo90 > -b and hi90 < b) for b in BOUNDS},
                 usable_resamples=int(len(v)))
        res.append(r)
    hdr = (f"{'contrast':72s} {'units':>5s} {'ev':>3s} {'delta':>7s} {'95% interval':>18s} est pub "
           f"{'90% interval':>18s} {'min bound':>9s} " + ' '.join(f'eq@{b:.2f}' for b in BOUNDS))
    lines = [hdr]
    for r in res:
        lines.append(f"{r['cell']:72s} {r['units']:5d} {r['events']:3d} {r['delta']:+7.4f} "
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
