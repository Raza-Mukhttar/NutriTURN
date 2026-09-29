"""R4 (their E4): paired cross-annotator anchor comparison with scale and era strata.

Extends R1/N1 with the pieces their specification asks for and the earlier run omitted:
the ratio-scale subset reported separately from the complete anchor set, the paired difference
stated as second minus primary, the declared pre-2000 versus 2000-onward split, disagreement
patterns broken out by anchor class rather than only overall agreement, and the early-versus-late
comparison restricted to claims represented in both periods so that changing claim composition is
separated from change inside the same claims.

Intersection labels are reported three ways, as required: coverage, conditional agreement among
retained labels, and agreement over the full anchor population with UNCLEAR counted as a
non-matching prediction. Greater agreement on the retained subset is not independent validation,
because the retained labels are exactly where the first two pipelines already agree.

  python analyses/nfx_R4_anchor_ext.py
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

AD = _os.path.join(_NT, 'data', 'derived')
HIGH = os.path.join(AD, 'numeric_anchor_v2_high_confidence.csv')
ALL = os.path.join(AD, 'numeric_anchor_v2_all_candidates.csv')
CLS = ['PROTECTIVE', 'HARMFUL', 'NULL']
SIGNED = ('PROTECTIVE', 'HARMFUL')


def load():
    # effect_type per (claim, pmid) from the candidate file; a record is ratio-scale only if every
    # attributed typed interval it carries is a ratio measure
    scale = {}
    for r in csv.DictReader(open(ALL)):
        k = (r['claim_id'], r['pmid'])
        et = (r.get('effect_type') or '').strip().lower()
        if not et:
            continue
        scale.setdefault(k, set()).add(et)
    A = {(r['claim_id'], r['pmid']): r for r in N.assoc()}
    cons = N.label_column('consensus')
    rows = []
    for r in csv.DictReader(open(HIGH)):
        k = (r['claim_id'], r['pmid'])
        a = A.get(k)
        if not a:
            continue
        et = scale.get(k, set())
        rows.append(dict(claim_id=r['claim_id'], pmid=r['pmid'], year=int(r['year']),
                         anchor=r['anchor'], primary=a['label_primary'], second=a['label_second'],
                         intersection=cons.get(k, 'UNCLEAR'),
                         ratio_only=bool(et) and et <= {'ratio'},
                         additive_only=bool(et) and et <= {'additive'}))
    return rows


def agree(rows, pipe):
    return float(np.mean([r[pipe] == r['anchor'] for r in rows])) if rows else float('nan')


def boot_diff(rows, f, n=N.BOOT, seed=N.SEED):
    g = np.array([r['claim_id'] for r in rows])
    cl = np.unique(g); by = {c: np.where(g == c)[0] for c in cl}
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        pick = rng.choice(cl, size=len(cl), replace=True)
        ix = np.concatenate([by[c] for c in pick])
        v = f([rows[i] for i in ix])
        if np.isfinite(v):
            out.append(v)
    a = np.array(out)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    rows = load()
    out = {'n': len(rows)}
    print(f'R4 paired cross-annotator anchor comparison  ({len(rows)} records, '
          f'{len({r["claim_id"] for r in rows})} claims)\n')

    # ---- scale strata
    subsets = [('complete anchor set', rows),
               ('ratio-scale only', [r for r in rows if r['ratio_only']]),
               ('additive-scale only', [r for r in rows if r['additive_only']])]
    print(f"{'subset':22s} {'n':>5s} {'primary':>8s} {'second':>8s} {'second - primary [95% CI]':>30s}")
    out['by_scale'] = []
    for lab, sub in subsets:
        if len(sub) < 30:
            print(f'{lab:22s} {len(sub):5d}   (too few)'); continue
        ap, as_ = agree(sub, 'primary'), agree(sub, 'second')
        f = lambda rr: agree(rr, 'second') - agree(rr, 'primary')
        d = f(sub); lo, hi = boot_diff(sub, f)
        out['by_scale'].append(dict(subset=lab, n=len(sub), primary=ap, second=as_,
                                    second_minus_primary=[d, lo, hi],
                                    established=N.established(lo, hi)))
        print(f'{lab:22s} {len(sub):5d} {ap:8.3f} {as_:8.3f}   {d:+.3f} {N.fmt_ci(lo,hi)}'
              f"{'  ESTABLISHED' if N.established(lo,hi) else ''}")

    # ---- declared era split
    print('\ndeclared era split (pre-2000 versus 2000 onward)')
    out['by_era'] = []
    for lab, sel in (('pre-2000', [r for r in rows if r['year'] < 2000]),
                     ('2000 onward', [r for r in rows if r['year'] >= 2000])):
        ap, as_ = agree(sel, 'primary'), agree(sel, 'second')
        f = lambda rr: agree(rr, 'second') - agree(rr, 'primary')
        d = f(sel); lo, hi = boot_diff(sel, f)
        out['by_era'].append(dict(era=lab, n=len(sel), primary=ap, second=as_,
                                  second_minus_primary=[d, lo, hi],
                                  established=N.established(lo, hi)))
        print(f'   {lab:14s} n {len(sel):5d}  primary {ap:.3f}  second {as_:.3f}  '
              f'second-primary {d:+.3f} {N.fmt_ci(lo,hi)}'
              f"{'  ESTABLISHED' if N.established(lo,hi) else ''}")

    # ---- disagreement patterns by anchor class
    print('\ndisagreement patterns by anchor class (share of anchored records)')
    pats = [('signed anchor, opposite sign',
             lambda r, p: r['anchor'] in SIGNED and r[p] in SIGNED and r[p] != r['anchor']),
            ('signed anchor, called NULL',
             lambda r, p: r['anchor'] in SIGNED and r[p] == 'NULL'),
            ('signed anchor, called UNCLEAR',
             lambda r, p: r['anchor'] in SIGNED and r[p] == 'UNCLEAR'),
            ('NULL anchor, given a sign',
             lambda r, p: r['anchor'] == 'NULL' and r[p] in SIGNED)]
    out['patterns'] = []
    print(f"{'pattern':34s} {'primary':>9s} {'second':>9s} {'intersection':>13s}")
    for lab, fn in pats:
        v = {}
        for p in ('primary', 'second', 'intersection'):
            base = [r for r in rows if (r['anchor'] in SIGNED if 'signed' in lab else r['anchor'] == 'NULL')]
            v[p] = float(np.mean([fn(r, p) for r in base])) if base else float('nan')
        out['patterns'].append(dict(pattern=lab, **v))
        print(f"{lab:34s} {v['primary']:9.3f} {v['second']:9.3f} {v['intersection']:13.3f}")

    # ---- paired early/late restricted to claims present in both periods
    early = {r['claim_id'] for r in rows if r['year'] < 2000}
    late = {r['claim_id'] for r in rows if r['year'] >= 2000}
    both = sorted(early & late)
    print(f'\npaired early-versus-late, restricted to the {len(both)} claims represented in both '
          f'periods (separates changing claim composition from change inside claims)')
    out['paired_claims'] = len(both)
    for p in ('primary', 'second'):
        de = []
        for c in both:
            e = [r for r in rows if r['claim_id'] == c and r['year'] < 2000]
            l = [r for r in rows if r['claim_id'] == c and r['year'] >= 2000]
            if len(e) < 3 or len(l) < 3:
                continue
            de.append(agree(l, p) - agree(e, p))
        if not de:
            continue
        de = np.array(de)
        rng = np.random.default_rng(N.SEED)
        bs = [np.mean(rng.choice(de, size=len(de), replace=True)) for _ in range(N.BOOT)]
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        out[f'paired_late_minus_early_{p}'] = [float(de.mean()), lo, hi, len(de)]
        print(f'   {p:9s} late minus early agreement, {len(de)} claims: '
              f'{de.mean():+.3f} {N.fmt_ci(lo,hi)}'
              f"{'  ESTABLISHED' if (lo>0 or hi<0) else '  not established'}")

    # ---- intersection three ways
    res = [r for r in rows if r['intersection'] != 'UNCLEAR']
    out['intersection'] = dict(coverage=len(res) / len(rows),
                               retained=len(res), abstained=len(rows) - len(res),
                               conditional_agreement=agree(res, 'intersection'),
                               full_population_agreement=agree(rows, 'intersection'))
    i = out['intersection']
    print(f"\nintersection labels, reported three ways")
    print(f"   coverage {i['coverage']:.3f} ({i['retained']} retained, {i['abstained']} abstain)")
    print(f"   conditional agreement on retained  {i['conditional_agreement']:.3f}")
    print(f"   agreement over the full population {i['full_population_agreement']:.3f} "
          f"(UNCLEAR counted as non-matching)")
    print('   the retained labels are exactly where the two pipelines already agree, so the higher '
          'conditional value is not independent validation')
    print('\nwritten', N.save_json(out, 'R4_anchor_extended.json'))


if __name__ == '__main__':
    main()
