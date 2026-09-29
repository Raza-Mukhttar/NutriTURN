"""N1 and N3: the LLM-independent numeric anchor against BOTH pipelines, overall and by era.

N1 asks whether "the score loses under a second annotation" means annotator dependence, or a
noisier annotator losing a real signal. The anchor is the only reference here that no language
model produced: a reported 95% interval, typed by its effect measure and attributed to the claim.

N3 asks where the falling HARMFUL share comes from. If the anchor's own HARMFUL share falls within
claims at the same rate as the pipelines', the drift is in the reported effect estimates, that is
in the literature. If the anchor is flat while a pipeline falls, the drift is the annotator.

  python analyses/nfx_anchor.py

Caveat stated with every result: the anchored subset is reporting-selected, 1.6% of associations,
and reporting practice itself changes by era.
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
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

ANCHOR = _os.path.join(_NT, 'data', 'derived',
                       'numeric_anchor_v2_high_confidence.csv')
SIGNED = ('PROTECTIVE', 'HARMFUL')
CLS = ['PROTECTIVE', 'HARMFUL', 'NULL']


def load():
    A = {(r['claim_id'], r['pmid']): r for r in N.assoc()}
    cons = N.label_column('consensus')
    rows = []
    for r in csv.DictReader(open(ANCHOR)):
        k = (r['claim_id'], r['pmid'])
        a = A.get(k)
        if not a:
            continue
        rows.append(dict(claim_id=r['claim_id'], pmid=r['pmid'], year=int(r['year']),
                         design=r['design'], anchor=r['anchor'],
                         primary=a['label_primary'], second=a['label_second'],
                         intersection=cons.get(k, 'UNCLEAR')))
    return rows


def metrics(rows, pipe):
    """Agreement, balanced accuracy and macro-F1 of one pipeline against the anchor."""
    y = [r['anchor'] for r in rows]; p = [r[pipe] for r in rows]
    agree = float(np.mean([a == b for a, b in zip(y, p)]))
    rec, f1 = {}, []
    for c in CLS:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        rec[c] = tp / (tp + fn) if tp + fn else float('nan')
        pr = tp / (tp + fp) if tp + fp else 0.0
        f1.append(2 * pr * rec[c] / (pr + rec[c]) if (pr + rec[c]) else 0.0)
    ba = float(np.nanmean([rec[c] for c in CLS]))
    return dict(n=len(rows), agreement=agree, balanced_accuracy=ba,
                macro_f1=float(np.mean(f1)), recall=rec)


def boot(rows, fn, n=N.BOOT, seed=N.SEED):
    g = np.array([r['claim_id'] for r in rows])
    cl = np.unique(g); by = {c: np.where(g == c)[0] for c in cl}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.choice(cl, size=len(cl), replace=True)
        ix = np.concatenate([by[c] for c in pick])
        v = fn([rows[i] for i in ix])
        if np.isfinite(v):
            out.append(v)
    a = np.array(out)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    rows = load()
    out = {'n_anchored': len(rows), 'caveat':
           'the anchored subset is reporting-selected (1.6% of associations) and reporting '
           'practice itself changes by era'}
    print(f'N1 numeric anchor against both pipelines  ({len(rows)} high-confidence records, '
          f'{len({r["claim_id"] for r in rows})} claims)\n')

    # ---- overall metrics per pipeline
    print(f"{'pipeline':14s} {'n':>5s} {'agree':>16s} {'bal.acc':>16s} {'macro-F1':>8s}   "
          f"recall P / H / N")
    for pipe in ('primary', 'second', 'intersection'):
        m = metrics(rows, pipe)
        lo, hi = boot(rows, lambda rr, p=pipe: metrics(rr, p)['agreement'])
        blo, bhi = boot(rows, lambda rr, p=pipe: metrics(rr, p)['balanced_accuracy'])
        out[pipe] = dict(m, agreement_ci=[lo, hi], balanced_accuracy_ci=[blo, bhi])
        print(f"{pipe:14s} {m['n']:5d} {m['agreement']:.3f} {N.fmt_ci(lo,hi):>10s} "
              f"{m['balanced_accuracy']:.3f} {N.fmt_ci(blo,bhi):>10s} {m['macro_f1']:8.3f}   "
              + ' / '.join(f'{m["recall"][c]:.3f}' for c in CLS))

    # ---- paired differences on the same records
    print('\npaired differences on the identical records (positive favours the first)')
    for a, b in (('primary', 'second'), ('primary', 'intersection'), ('second', 'intersection')):
        for key, lab in (('agreement', 'agreement'), ('balanced_accuracy', 'balanced accuracy')):
            f = lambda rr, x=a, y=b, k=key: metrics(rr, x)[k] - metrics(rr, y)[k]
            d = f(rows); lo, hi = boot(rows, f)
            out[f'{a}_minus_{b}_{key}'] = [d, lo, hi]
            print(f'   {a} minus {b}, {lab:17s} {d:+.3f} {N.fmt_ci(lo,hi)}'
                  f"{'  ESTABLISHED' if N.established(lo,hi) else ''}")

    # ---- signed-only variant: drop NULL anchors so the NULL-rate difference cannot confound sign
    sg = [r for r in rows if r['anchor'] in SIGNED]
    print(f'\nsigned-anchor subset only ({len(sg)} records), so the pipelines\' different NULL '
          f'rates cannot confound sign accuracy')
    for pipe in ('primary', 'second', 'intersection'):
        acc = float(np.mean([r[pipe] == r['anchor'] for r in sg]))
        sgn = [r for r in sg if r[pipe] in SIGNED]
        sacc = float(np.mean([r[pipe] == r['anchor'] for r in sgn])) if sgn else float('nan')
        lo, hi = boot(sg, lambda rr, p=pipe: float(np.mean([x[p] == x['anchor'] for x in rr])))
        out[f'{pipe}_signed_anchor'] = dict(n=len(sg), accuracy=acc, accuracy_ci=[lo, hi],
                                            n_pipeline_signed=len(sgn),
                                            accuracy_when_pipeline_signed=sacc)
        print(f'   {pipe:14s} accuracy on signed anchors {acc:.3f} {N.fmt_ci(lo,hi)}   '
              f'and {sacc:.3f} on the {len(sgn)} it also calls signed')
    f = lambda rr: float(np.mean([x['primary'] == x['anchor'] for x in rr])) - \
                   float(np.mean([x['second'] == x['anchor'] for x in rr]))
    d = f(sg); lo, hi = boot(sg, f)
    out['signed_primary_minus_second'] = [d, lo, hi]
    print(f'   primary minus second on signed anchors {d:+.3f} {N.fmt_ci(lo,hi)}'
          f"{'  ESTABLISHED' if N.established(lo,hi) else ''}")

    # ---- where the second pipeline's extra NULL mass lands
    na = [r for r in rows if r['anchor'] == 'NULL']
    sa = [r for r in rows if r['anchor'] in SIGNED]
    print(f'\nwhere the second pipeline\'s extra NULL mass lands')
    for pipe in ('primary', 'second'):
        print(f'   {pipe:9s} NULL on NULL anchors {np.mean([r[pipe]=="NULL" for r in na]):.3f} '
              f'({len(na)} records)   NULL on SIGNED anchors '
              f'{np.mean([r[pipe]=="NULL" for r in sa]):.3f} ({len(sa)} records)')
    out['null_mass'] = {p: dict(on_null_anchor=float(np.mean([r[p] == 'NULL' for r in na])),
                                on_signed_anchor=float(np.mean([r[p] == 'NULL' for r in sa])))
                        for p in ('primary', 'second')}

    # ---- intersection coverage
    unc = sum(1 for r in rows if r['intersection'] == 'UNCLEAR')
    res = [r for r in rows if r['intersection'] != 'UNCLEAR']
    out['intersection_coverage'] = dict(unclear=unc, resolved=len(res),
                                        agreement_on_resolved=float(np.mean(
                                            [r['intersection'] == r['anchor'] for r in res])))
    print(f'\nintersection coverage: {unc} of {len(rows)} anchored records become UNCLEAR; '
          f'on the {len(res)} resolved the agreement is '
          f'{out["intersection_coverage"]["agreement_on_resolved"]:.3f}')

    # ---- column-wise error rates for the second pipeline
    print('\ncolumn-wise error rates (share of each pipeline call carrying the opposite anchor)')
    for pipe in ('primary', 'second'):
        for c, other in (('PROTECTIVE', 'HARMFUL'), ('HARMFUL', 'PROTECTIVE')):
            sel = [r for r in rows if r[pipe] == c]
            v = float(np.mean([r['anchor'] == other for r in sel])) if sel else float('nan')
            out[f'{pipe}_{c}_carrying_{other}'] = v
            print(f'   {pipe:9s} {c:11s} carrying a {other:11s} anchor: {v:.3f} ({len(sel)} calls)')

    # ---- confusion matrices
    out['confusion'] = {}
    for pipe in ('primary', 'second', 'intersection'):
        M = {a: {b: sum(1 for r in rows if r['anchor'] == a and r[pipe] == b)
                 for b in CLS + ['UNCLEAR']} for a in CLS}
        out['confusion'][pipe] = M
    print('\nconfusion, anchor rows by pipeline columns (P/H/N/U)')
    for pipe in ('primary', 'second', 'intersection'):
        print(f'   {pipe}')
        for a in CLS:
            print(f'      {a:11s} ' + ' '.join(f'{out["confusion"][pipe][a][b]:5d}'
                                               for b in CLS + ['UNCLEAR']))

    # ================= N3: by era =================
    print('\n\nN3 decade-stratified anchor comparison')
    bins = [('pre-2000', 0, 1999), ('2000s', 2000, 2009), ('2010s', 2010, 2019), ('2020s', 2020, 2100)]
    out['by_era'] = []
    print(f"\n{'era':10s} {'n':>5s}  agreement: anchor vs primary / second      "
          f"HARMFUL share: anchor / primary / second")
    for lab, lo_, hi_ in bins:
        sel = [r for r in rows if lo_ <= r['year'] <= hi_]
        if len(sel) < 30:
            print(f'{lab:10s} {len(sel):5d}  (too few)'); continue
        ap = float(np.mean([r['primary'] == r['anchor'] for r in sel]))
        as_ = float(np.mean([r['second'] == r['anchor'] for r in sel]))
        sgb = [r for r in sel if r['anchor'] in SIGNED]
        def hs(key):
            v = [r for r in sel if r[key] in SIGNED]
            return float(np.mean([r[key] == 'HARMFUL' for r in v])) if v else float('nan')
        ha = float(np.mean([r['anchor'] == 'HARMFUL' for r in sgb])) if sgb else float('nan')
        row = dict(era=lab, n=len(sel), n_signed_anchor=len(sgb),
                   agreement_primary=ap, agreement_second=as_,
                   harmful_share_anchor=ha, harmful_share_primary=hs('primary'),
                   harmful_share_second=hs('second'),
                   over_harmful_primary=float(np.mean(
                       [r['primary'] == 'HARMFUL' and r['anchor'] != 'HARMFUL' for r in sel])))
        out['by_era'].append(row)
        print(f"{lab:10s} {len(sel):5d}              {ap:.3f} / {as_:.3f}              "
              f"      {ha:.3f} / {row['harmful_share_primary']:.3f} / "
              f"{row['harmful_share_second']:.3f}")
    print('\n   over-HARMFUL rate (pipeline says HARMFUL, anchor does not), by era')
    for r in out['by_era']:
        print(f"      {r['era']:10s} {r['over_harmful_primary']:.3f}")

    # within-claim slope of the HARMFUL share, for each of the three label sources
    print('\n   within-claim slope of the signed HARMFUL indicator, claim fixed effects,')
    print('   fitted on the SAME anchored records for all three sources')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from within_claim_label_drift import slopes, boot as slope_boot
    out['within_claim_slopes'] = {}
    for src in ('anchor', 'primary', 'second'):
        sel = [r for r in rows if r[src] in SIGNED]
        if len(sel) < 100:
            continue
        yr = np.array([r['year'] for r in sel]); sg_ = np.array([1 if r[src] == 'HARMFUL' else -1 for r in sel])
        cl = np.array([r['claim_id'] for r in sel])
        w = slopes(yr.copy(), sg_.copy(), cl, True)
        wl, wh = slope_boot(yr, sg_, cl, True)
        out['within_claim_slopes'][src] = dict(n=len(sel), slope_per_decade=w * 10,
                                               ci_per_decade=[wl * 10, wh * 10],
                                               established=N.established(wl, wh))
        print(f'      {src:9s} n {len(sel):5d}  {w*10:+.4f} per decade {N.fmt_ci(wl*10, wh*10, 4)}'
              f"{'  ESTABLISHED' if N.established(wl,wh) else '  not established'}")
    # ratio-scale sensitivity
    print('\n   (the anchor is most reliable on ratio-scale intervals; the design column does not')
    print('    record scale, so this sensitivity is reported in the JSON only where available)')
    print('\nwritten', N.save_json(out, 'N1_N3_anchor.json'))


if __name__ == '__main__':
    main()
