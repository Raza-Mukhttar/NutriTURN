"""Remaining table numbers: intersection anchor on retained records, the second pipeline's
confusion with denominators, the pair-count partition check, post-cutoff tie signs, the
query-unconfirmed manifest field, and per-case cutoff coverage for the historical sources.
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
import os, sys, csv, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

AD = _os.path.join(_NT, 'data', 'derived')
CLS = ['PROTECTIVE', 'HARMFUL', 'NULL']
PIV = {'hormone_replacement_cvd': 2002, 'hormone_replacement_breast_cancer': 2002,
       'beta_carotene_cancer': 1996, 'vitamin_e_cvd': 2002,
       'selenium_prostate_cancer': 2009, 'folic_acid_colorectal': 2007,
       'niacin_cvd': 2014, 'dietary_cholesterol_cvd': 2015}
PIV_PMID = {'hormone_replacement_cvd': '12117397', 'hormone_replacement_breast_cancer': '12117397',
            'beta_carotene_cancer': '8127329', 'vitamin_e_cvd': '10639540',
            'selenium_prostate_cancer': '19066370', 'folic_acid_colorectal': '17551129',
            'niacin_cvd': '22085343', 'dietary_cholesterol_cvd': None}


def anchor_rows():
    A = {(r['claim_id'], r['pmid']): r for r in N.assoc()}
    cons = N.label_column('consensus')
    out = []
    for r in csv.DictReader(open(os.path.join(AD, 'numeric_anchor_v2_high_confidence.csv'))):
        k = (r['claim_id'], r['pmid'])
        a = A.get(k)
        if a:
            out.append(dict(claim_id=r['claim_id'], year=int(r['year']), anchor=r['anchor'],
                            primary=a['label_primary'], second=a['label_second'],
                            intersection=cons.get(k, 'UNCLEAR')))
    return out


def metrics(rows, pipe):
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
    return dict(n=len(rows), agreement=agree, balanced_accuracy=float(np.nanmean(list(rec.values()))),
                macro_f1=float(np.mean(f1)), recall=rec)


def boot(rows, fn, n=N.BOOT, seed=N.SEED):
    g = np.array([r['claim_id'] for r in rows]); cl = np.unique(g)
    by = {c: np.where(g == c)[0] for c in cl}; rng = np.random.default_rng(seed); o = []
    for _ in range(n):
        ix = np.concatenate([by[c] for c in rng.choice(cl, size=len(cl), replace=True)])
        v = fn([rows[i] for i in ix])
        if np.isfinite(v):
            o.append(v)
    a = np.array(o); return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    out = {}
    rows = anchor_rows()

    # ---- 6. intersection on retained records, and the second pipeline's confusion
    ret = [r for r in rows if r['intersection'] != 'UNCLEAR']
    m = metrics(ret, 'intersection')
    lo, hi = boot(ret, lambda rr: metrics(rr, 'intersection')['agreement'])
    blo, bhi = boot(ret, lambda rr: metrics(rr, 'intersection')['balanced_accuracy'])
    out['intersection_retained'] = dict(m, agreement_ci=[lo, hi], balanced_accuracy_ci=[blo, bhi])
    print('6. Intersection labels restricted to the records they retain\n')
    print(f"   n {m['n']}   agreement {m['agreement']:.4f} {N.fmt_ci(lo,hi,4)}   "
          f"balanced accuracy {m['balanced_accuracy']:.4f} {N.fmt_ci(blo,bhi,4)}   "
          f"macro-F1 {m['macro_f1']:.4f}")
    print(f"   recall  " + '  '.join(f'{c} {m["recall"][c]:.4f}' for c in CLS))
    print('\n   second pipeline confusion, anchor rows by pipeline columns, with row denominators')
    print(f"   {'anchor':12s} {'PROTECTIVE':>11s} {'HARMFUL':>9s} {'NULL':>7s} {'UNCLEAR':>8s} {'total':>7s}")
    conf = {}
    for a in CLS:
        r_ = [x for x in rows if x['anchor'] == a]
        counts = {b: sum(1 for x in r_ if x['second'] == b) for b in CLS + ['UNCLEAR']}
        conf[a] = dict(counts, total=len(r_))
        print(f"   {a:12s} " + ' '.join(f'{counts[b]:>9d}' for b in CLS + ['UNCLEAR'])
              + f" {len(r_):>7d}")
    out['second_confusion'] = conf

    # ---- 1. pair-count partition check
    print('\n1. Pair-count partition check (primary)')
    t = N.unit_table('primary')
    idx, s, y, g = N.loco_two_var(t)
    cls = np.array([N.sgn0(t[i]['p']) for i in idx])
    n_h = int((cls == 1).sum()); n_p = int((cls == -1).sum()); n_z = int((cls == 0).sum())
    pos = y == 1; neg = y == 0
    within = int(((cls[pos][:, None] == cls[neg][None, :]) &
                  (cls[pos][:, None] != 0) & (cls[neg][None, :] != 0)).sum())
    across = int(((cls[pos][:, None] != cls[neg][None, :]) &
                  (cls[pos][:, None] != 0) & (cls[neg][None, :] != 0)).sum())
    zero = int(((cls[pos][:, None] == 0) | (cls[neg][None, :] == 0)).sum())
    out['pair_partition'] = dict(harmful_units=n_h, protective_units=n_p, zero_units=n_z,
                                 within=within, across=across, zero=zero,
                                 total=int(pos.sum() * neg.sum()))
    print(f'   unit split: HARMFUL {n_h}, PROTECTIVE {n_p}, zero-direction {n_z}  '
          f'(assumed 274 / 730 / 8)')
    print(f'   pairs: within {within:,}  across {across:,}  zero {zero:,}  '
          f'total {int(pos.sum()*neg.sum()):,}')
    print(f'   matches the reported 24,633 / 34,650 / 7,528: '
          f'{within==24633 and across==34650 and zero==7528}')

    # ---- 14. post-cutoff ties surviving the margin masks
    print('\n14. Post-cutoff ties surviving the margin masks, with pre-cutoff signs')
    zd = [r for r in t if r['s'] == 0 or N.sgn0(r['p_post']) == 0]
    surv = []
    for thr in (0.05, 0.10, 0.15, 0.20):
        for r in zd:
            if abs(r['p']) >= thr:
                surv.append(dict(threshold=thr, claim_id=r['claim_id'], cutoff=r['cutoff'],
                                 pre_sign=r['s'], abs_p=round(abs(r['p']), 4),
                                 h=r['h'], l=r['l'], post_tie=int(N.sgn0(r['p_post']) == 0),
                                 event=r['y']))
    seen = set(); uniq = []
    for x in surv:
        k = (x['claim_id'], x['cutoff'])
        if k not in seen:
            seen.add(k); uniq.append(x)
    out['post_cutoff_ties'] = uniq
    for x in uniq:
        sg = {1: 'HARMFUL', -1: 'PROTECTIVE', 0: 'none'}[x['pre_sign']]
        print(f"   {x['claim_id']}@{x['cutoff']}  pre-cutoff sign {sg}  |p_t| {x['abs_p']}  "
              f"h={x['h']} l={x['l']}  post-cutoff tie {bool(x['post_tie'])}  event {x['event']}")
    print('   -> under the alternative tie mapping (tie assigned to PROTECTIVE) a PROTECTIVE '
          'pre-cutoff sign would leave the endpoint unchanged for these units')

    # ---- 15. query_unconfirmed manifest field
    adir = sorted(glob.glob(os.path.join(N.GPTPRO, 'results', 'snapshot_*')))[-1]
    zero_q = []
    for line in open(os.path.join(adir, 'queries.jsonl')):
        d = json.loads(line)
        if not d.get('pmids'):
            zero_q.append(d['claim_id'])
    man = [dict(claim_id=c, query_unconfirmed=int(c in set(zero_q)))
           for c in sorted(json.load(open(os.path.join(N.SNAPD, 'claim_queries.json'))))]
    N.save_csv(man, 'R6_query_unconfirmed_manifest.csv')
    out['query_unconfirmed'] = sorted(zero_q)
    print(f'\n15. query_unconfirmed = true for {len(zero_q)} claims '
          f'(written to results/R6_query_unconfirmed_manifest.csv)')

    # ---- 13 extra. eligible cutoffs before the pivotal year, pivotal PMID in archive
    print('\n13. Historical cases: eligible cutoffs before the pivotal year, pivotal PMID in archive')
    units = {}
    for r in t:
        units.setdefault(r['claim_id'], []).append(r['cutoff'])
    arch = {}
    for p in sorted(glob.glob(os.path.join(N.SNAPD, 'records', '*.json'))):
        d = json.load(open(p))
        arch[d['claim_id']] = {str(x['pmid']) for x in d['records']}
    cov = []
    for cid, piv in PIV.items():
        cuts = sorted(units.get(cid, []))
        before = [c for c in cuts if c < piv]
        pm = PIV_PMID.get(cid)
        cov.append(dict(claim_id=cid, pivotal_year=piv, cutoffs=len(cuts),
                        cutoffs_before_pivotal=len(before),
                        pivotal_pmid=pm,
                        pivotal_pmid_in_archive=(None if pm is None else int(pm in arch.get(cid, set())))))
        print(f'   {cid:36s} pivotal {piv}  cutoffs {len(cuts)}  before pivotal {len(before)}  '
              f"pivotal PMID {pm or 'n/a':>9s} in archive "
              f"{'n/a' if pm is None else bool(pm in arch.get(cid, set()))}")
    out['historical_coverage'] = cov
    print('\nwritten', N.save_json(out, 'R6_table_numbers.json'))


if __name__ == '__main__':
    main()
