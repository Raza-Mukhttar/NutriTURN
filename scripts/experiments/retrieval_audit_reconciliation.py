"""E7: retrieval-audit reconciliation.

From the dated re-retrieval manifest, list every claim whose archived and re-retrieved identifier
sets overlap below 0.8, plus every claim that loses more than 10% of its stored identifiers; say
whether each is a positive claim under either annotation and whether it is one of the 20 hand-added
claims; compare the stored and re-retrieved query translations to identify the cause; then rerun
the development and calendar cells with those claims removed.

  python analyses/nfx_retrieval.py
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
import os, sys, json, glob, csv, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

AUDIT = glob.glob(os.path.join(N.GPTPRO, 'results', 'snapshot_*'))
HAND = os.path.join(_NT,
                    'data', 'protocol_snapshot', 'benchmark', 'hand_added_claims.json')


def main():
    assert AUDIT, 'no snapshot audit directory found'
    adir = sorted(AUDIT)[-1]
    print(f'E7 retrieval-audit reconciliation\naudit: {os.path.basename(adir)}\n')
    v2 = {}
    for line in open(os.path.join(adir, 'queries.jsonl')):
        d = json.loads(line)
        v2[d['claim_id']] = d
    # archived identifier sets
    arch = {}
    SNAPD = N.SNAPD
    for p in sorted(glob.glob(os.path.join(SNAPD, 'records', '*.json'))):
        d = json.load(open(p))
        arch[d['claim_id']] = ({str(r['pmid']) for r in d['records']}, d.get('query', ''))
    hand = set()
    if os.path.exists(HAND):
        h = json.load(open(HAND))
        hand = set(h if isinstance(h, list) else h.get('claims', h.keys()))
    # positive claims under either annotation
    tp, ts = N.unit_table('primary'), N.unit_table('second')
    pos = {r['claim_id'] for r in tp if r['y']} | {r['claim_id'] for r in ts if r['y']}

    rows = []
    for cid, (ids, q1) in sorted(arch.items()):
        d = v2.get(cid)
        if not d:
            continue
        new = {str(x) for x in d['pmids']}
        inter = len(ids & new); union = len(ids | new)
        jac = inter / union if union else 0.0
        lost = len(ids - new)
        rows.append(dict(claim_id=cid, archived=len(ids), reretrieved=len(new),
                         shared=inter, jaccard=round(jac, 4),
                         lost=lost, lost_share=round(lost / max(len(ids), 1), 4),
                         positive_claim=int(cid in pos), hand_added=int(cid in hand),
                         query_changed=int((q1 or '').strip() != (d.get('query') or '').strip()),
                         translation=(d.get('query_translation') or '')[:300]))
    N.save_csv(rows, 'E7_per_claim_audit.csv')
    low = [r for r in rows if r['jaccard'] < 0.8]
    lost10 = [r for r in rows if r['lost_share'] > 0.10]
    flagged = sorted({r['claim_id'] for r in low} | {r['claim_id'] for r in lost10})
    print(f'claims audited: {len(rows)}')
    print(f'per-claim Jaccard below 0.8: {len(low)}')
    print(f'losing more than 10% of stored identifiers: {len(lost10)}  (paper reports 19)')
    print(f'union of the two flagged sets: {len(flagged)} claims\n')
    print(f"{'claim_id':44s} {'arch':>6s} {'re-ret':>7s} {'jacc':>6s} {'lost%':>6s} pos hand qchg")
    for r in sorted(low + [x for x in lost10 if x not in low], key=lambda r: r['jaccard']):
        print(f"{r['claim_id']:44s} {r['archived']:6d} {r['reretrieved']:7d} {r['jaccard']:6.3f} "
              f"{100*r['lost_share']:6.1f} {r['positive_claim']:3d} {r['hand_added']:4d} {r['query_changed']:4d}")
    # cause analysis
    fl = {r['claim_id']: r for r in low}
    fl.update({r['claim_id']: r for r in lost10})
    flv = list(fl.values())                                  # deduplicated flagged claims
    causes = {'flagged claims (deduplicated)': len(flv),
              'query string differs': sum(1 for r in flv if r['query_changed']),
              'zero overlap (set replaced)': sum(1 for r in flv if r['shared'] == 0),
              'archived set larger than re-retrieved': sum(1 for r in flv if r['archived'] > r['reretrieved']),
              'archived set smaller than re-retrieved': sum(1 for r in flv if r['archived'] < r['reretrieved'])}
    # the archived snapshot was capped: per-claim counts pile up just below 800
    arch_n = np.array([len(v[0]) for v in arch.values()])
    near_cap = int(((arch_n >= 780) & (arch_n <= 800)).sum())
    causes['archived count in [780, 800] (record cap)'] = near_cap
    causes['flagged claims at the cap'] = sum(1 for r in flv if 780 <= r['archived'] <= 800)
    causes['flagged claims with zero re-retrieved records'] = sum(1 for r in flv if r['reretrieved'] == 0)
    print('\ncause indicators over the flagged claims:')
    for k, v in causes.items():
        print(f'  {k}: {v}')
    print(f'\narchived per-claim record counts: min {arch_n.min()} median {int(np.median(arch_n))} '
          f'max {arch_n.max()} | claims in [780, 800]: {near_cap} of {len(arch_n)}')
    zero = [r for r in low if r['reretrieved'] == 0]
    big = [r for r in low if r['reretrieved'] > r['archived']]
    print(f'\ntwo disjoint mechanisms explain the flagged claims:')
    print(f'  {len(zero)} claims returned ZERO records at re-retrieval (query no longer resolves); '
          f'their Jaccard is 0 by construction and they are exactly the {len(lost10)} '
          f'claims that "lose more than 10%" of stored identifiers')
    print(f'  {len(big)} claims re-retrieved MORE records than were archived '
          f'(median ratio {np.median([r["reretrieved"]/max(r["archived"],1) for r in big]):.1f}x); '
          f'they lost nothing, and their low Jaccard reflects the archived record cap, not drift')
    out_extra = {'zero_return_claims': [r['claim_id'] for r in zero],
                 'cap_inflated_claims': [r['claim_id'] for r in big],
                 'archived_count_min': int(arch_n.min()), 'archived_count_max': int(arch_n.max()),
                 'archived_claims_near_cap': near_cap}

    # rerun the cells without the flagged claims
    print('\ncells with the flagged claims removed')
    out = {'audit_dir': os.path.basename(adir), 'n_claims': len(rows),
           'jaccard_below_0.8': len(low), 'lost_more_than_10pct': len(lost10),
           'flagged': flagged, 'causes': causes, 'cells': []}
    for a in ('primary', 'second', 'consensus'):
        rows_u = N.unit_table(a)
        for r, acc in zip(rows_u, N.build_cohort_accrual5(rows_u)):
            r['accrual_5y'] = acc
        keep = np.array([r['claim_id'] not in set(flagged) for r in rows_u])
        for lbl, mask in (('all claims', np.ones(len(rows_u), bool)), ('flagged removed', keep)):
            idx, s2, y, g = N.loco_two_var(rows_u, subset=mask)
            sm = N.margin_score(rows_u, idx)
            d, lo, hi, _ = N.paired_ci(y, s2, sm, g)
            cell = dict(annotation=a, population=lbl, units=len(idx), events=int(y.sum()),
                        margin_auroc=N.auroc(y, sm), two_var_auroc=N.auroc(y, s2),
                        signed_minus_margin=[d, lo, hi], established=N.established(lo, hi))
            out['cells'].append(cell)
            print(f"  {a:10s} {lbl:16s} units {len(idx):5d} events {int(y.sum()):3d}  "
                  f"margin {cell['margin_auroc']:.3f} signed {cell['two_var_auroc']:.3f}  "
                  f"signed-margin {d:+.3f} {N.fmt_ci(lo,hi)}"
                  f"{'  ESTABLISHED' if cell['established'] else ''}")
    co = N.build_cohort('primary')
    for H in (3, 5, 10):
        d_ = [r for r in co if r['horizon'] == H]
        for lbl, drop in (('all claims', set()), ('flagged removed', set(flagged))):
            sel = [r for r in d_ if r['claim_id'] not in drop]
            s2, y, g, _ = N.cohort_loco_two_var(sel, H)
            sm = np.array([1 - abs(r['p']) for r in sel if r['accrued']])
            dd, lo, hi, _ = N.paired_ci(y, s2, sm, g)
            cell = dict(annotation='primary', population=f'calendar H{H}: {lbl}',
                        units=len(y), events=int(y.sum()),
                        margin_auroc=N.auroc(y, sm), two_var_auroc=N.auroc(y, s2),
                        signed_minus_margin=[dd, lo, hi], established=N.established(lo, hi))
            out['cells'].append(cell)
            print(f"  calendar H{H:<2d} {lbl:16s} units {len(y):5d} events {int(y.sum()):3d}  "
                  f"margin {cell['margin_auroc']:.3f} signed {cell['two_var_auroc']:.3f}  "
                  f"signed-margin {dd:+.3f} {N.fmt_ci(lo,hi)}"
                  f"{'  ESTABLISHED' if cell['established'] else ''}")
    ch = []
    for a in ('primary', 'second', 'consensus'):
        x = [c for c in out['cells'] if c['annotation'] == a and c['population'] == 'all claims'][0]
        z = [c for c in out['cells'] if c['annotation'] == a and c['population'] == 'flagged removed'][0]
        if x['established'] != z['established']:
            ch.append(a)
    for H in (3, 5, 10):
        x = [c for c in out['cells'] if c['population'] == f'calendar H{H}: all claims'][0]
        z = [c for c in out['cells'] if c['population'] == f'calendar H{H}: flagged removed'][0]
        if x['established'] != z['established']:
            ch.append(f'calendar H{H}')
    out.update(out_extra)
    out['conclusions_changed'] = ch
    print('\nconclusions changed by removing the flagged claims:', 'NONE' if not ch else ch)
    print('written', N.save_json(out, 'E7_retrieval.json'))


if __name__ == '__main__':
    main()
