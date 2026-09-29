"""E15: retrieval-cap sensitivity, defined by the cap itself rather than by re-retrieval overlap.

E7 flagged 54 claims whose archived and re-retrieved identifier sets overlap below 0.8. That set is
defined by the 2026-09-22 re-retrieval, so it is not the same as the set of claims the cap actually
truncated, and a reviewer can reasonably ask about near-cap claims E7 never flagged. This script
defines the cap-affected set from the archive's own metadata and re-runs every protocol cell on it.

  1. Cap census. `claim_queries.json` records, per claim, `history_retrieved_in_full_after_800_cap`.
     A claim is CAP-TRUNCATED when that flag is False and the provider reported more records than
     were archived, so the archived stream is a strict subset of what the query returned.
  2. Set reconciliation against E7's 54 flagged claims, reported as a full cross-tabulation.
  3. Truncation-order diagnostic. PubMed identifiers are assigned in rough accession order, so the
     position of the archived identifiers inside the re-retrieved identifier range says how the
     truncation was ordered: high positions mean the most recent were kept, low positions the
     oldest, a uniform spread means relevance or arbitrary order. A non-uniform spread means the
     archived stream is non-randomly incomplete in calendar time, which is what would matter.
  4. Leave-out sensitivity. Every development and calendar cell recomputed without the
     cap-truncated claims.

  python analyses/nfx_cap.py
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
import os, sys, json, glob, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N


def main():
    Q = json.load(open(os.path.join(N.SNAPD, 'claim_queries.json')))
    arch = {}
    for p in sorted(glob.glob(os.path.join(N.SNAPD, 'records', '*.json'))):
        d = json.load(open(p))
        arch[d['claim_id']] = {str(r['pmid']) for r in d['records']}
    adir = sorted(glob.glob(os.path.join(N.GPTPRO, 'results', 'snapshot_*')))[-1]
    v2 = {}
    for line in open(os.path.join(adir, 'queries.jsonl')):
        d = json.loads(line)
        v2[d['claim_id']] = d

    print('E15 retrieval-cap sensitivity\n')
    print('1. Cap census (from the archive metadata, not from re-retrieval)')
    rows = []
    for cid in sorted(Q):
        full = Q[cid].get('history_retrieved_in_full_after_800_cap')
        n_arch = len(arch[cid])
        d = v2.get(cid, {})
        prov = int(d.get('pubmed_count') or 0)
        new = {str(x) for x in d.get('pmids', [])}
        truncated = (full is False) and prov > n_arch
        rows.append(dict(claim_id=cid, retrieved_in_full=bool(full), archived=n_arch,
                         provider_count_2026=prov, reretrieved_ids=len(new),
                         cap_truncated=int(truncated),
                         shared=len(arch[cid] & new),
                         jaccard=round(len(arch[cid] & new) / max(len(arch[cid] | new), 1), 4)))
    N.save_csv(rows, 'E15_cap_census.csv')
    infull = [r for r in rows if r['retrieved_in_full']]
    notfull = [r for r in rows if not r['retrieved_in_full']]
    trunc = [r for r in rows if r['cap_truncated']]
    nearcap = [r for r in rows if 780 <= r['archived'] <= 800]
    print(f'   claims: {len(rows)}')
    print(f'   retrieved in full after the 800 cap : {len(infull):3d}  '
          f'(archived {min(r["archived"] for r in infull)}-{max(r["archived"] for r in infull)})')
    print(f'   NOT retrieved in full              : {len(notfull):3d}  '
          f'(archived {min(r["archived"] for r in notfull)}-{max(r["archived"] for r in notfull)})')
    print(f'   archived count in [780, 800]       : {len(nearcap):3d}  <- the "cap-like" set')
    print(f'   CAP-TRUNCATED (flag False and provider count exceeds archived): {len(trunc):3d}')
    print(f'      of these, archived in [780,800]: {sum(1 for r in trunc if 780 <= r["archived"] <= 800)}')
    print(f'      of these, archived below 780   : {sum(1 for r in trunc if r["archived"] < 780)}')

    # ---- 2. reconciliation with E7
    e7 = json.load(open(os.path.join(N.RESULTS, 'E7_retrieval.json')))
    flagged = set(e7['flagged'])
    T = {r['claim_id'] for r in trunc}
    NC = {r['claim_id'] for r in nearcap}
    print('\n2. Reconciliation with the E7 overlap-flagged set')
    print(f'   E7 flagged (Jaccard < 0.8 or losing >10%)     : {len(flagged)}')
    print(f'   cap-truncated (this script)                   : {len(T)}')
    print(f'   in both                                       : {len(T & flagged)}')
    print(f'   cap-truncated but NOT flagged by E7           : {len(T - flagged)}')
    print(f'   flagged by E7 but NOT cap-truncated           : {len(flagged - T)}')
    print(f'   near-cap [780,800] but NOT flagged by E7      : {len(NC - flagged)}')
    zero = {r['claim_id'] for r in rows if r['reretrieved_ids'] == 0}
    print(f'   (E7 flagged that returned zero at re-retrieval: {len(zero & flagged)}, '
          f'of which cap-truncated: {len(zero & flagged & T)})')
    print(f'   UNION of cap-truncated and E7-flagged         : {len(T | flagged)}')

    # ---- 3. truncation-order diagnostic
    print('\n3. Truncation-order diagnostic (identifier position, a proxy for accession date)')
    print('   For each cap-truncated claim whose archived identifiers are a subset of the')
    print('   re-retrieved set, the mean percentile of the archived identifiers within that set.')
    print('   0.5 means an arbitrary or relevance-ordered subset; near 1 means newest-kept;')
    print('   near 0 means oldest-kept. A biased value means calendar-time incompleteness.')
    pct = []
    for r in trunc:
        cid = r['claim_id']
        new = {str(x) for x in v2.get(cid, {}).get('pmids', [])}
        if len(new) < 50:
            continue
        both = arch[cid] & new
        if len(both) < 50:
            continue
        allp = np.array(sorted(int(x) for x in new), dtype=np.int64)
        sub = np.array(sorted(int(x) for x in both), dtype=np.int64)
        ranks = np.searchsorted(allp, sub) / max(len(allp) - 1, 1)
        pct.append((cid, float(ranks.mean()), len(both), len(allp)))
    if pct:
        m = np.array([x[1] for x in pct])
        print(f'   claims evaluable: {len(pct)}   mean percentile {m.mean():.3f}  '
              f'median {np.median(m):.3f}  range [{m.min():.3f}, {m.max():.3f}]')
        print(f'   claims with mean percentile above 0.65 (newest-kept): {(m > 0.65).sum()}')
        print(f'   claims with mean percentile below 0.35 (oldest-kept): {(m < 0.35).sum()}')
        for cid, v, nb, na in sorted(pct, key=lambda x: x[1])[:4]:
            print(f'      {cid:44s} {v:.3f}  ({nb} of {na})')
        for cid, v, nb, na in sorted(pct, key=lambda x: -x[1])[:4]:
            print(f'      {cid:44s} {v:.3f}  ({nb} of {na})')
    else:
        print('   no evaluable claims')

    # ---- 4. leave-out sensitivity
    print('\n4. Leave-out sensitivity, cells recomputed without the cap-affected claims')
    out = {'census': {'claims': len(rows), 'retrieved_in_full': len(infull),
                      'cap_truncated': len(T), 'near_cap_780_800': len(nearcap),
                      'e7_flagged': len(flagged), 'both': len(T & flagged),
                      'cap_only': sorted(T - flagged), 'e7_only': sorted(flagged - T),
                      'union': len(T | flagged)},
           'identifier_percentile': {'n': len(pct), 'mean': float(np.mean([x[1] for x in pct])) if pct else None,
                                     'per_claim': [{'claim_id': c, 'mean_percentile': v} for c, v, _, _ in pct]},
           'cells': []}
    drops = [('all claims', set()), ('cap-truncated removed', T),
             ('near-cap [780,800] removed', NC), ('cap-truncated or E7-flagged removed', T | flagged)]
    for a in ('primary', 'second', 'consensus'):
        urows = N.unit_table(a)
        for r, acc in zip(urows, N.build_cohort_accrual5(urows)):
            r['accrual_5y'] = acc
        for lbl, drop in drops:
            mask = np.array([r['claim_id'] not in drop for r in urows])
            if mask.sum() < 100:
                continue
            idx, s2, y, g = N.loco_two_var(urows, subset=mask)
            sm = N.margin_score(urows, idx)
            d, lo, hi, _ = N.paired_ci(y, s2, sm, g)
            c = dict(cell=f'development, {a}', population=lbl, units=len(idx),
                     events=int(y.sum()), claims=len(np.unique(g)),
                     margin_auroc=N.auroc(y, sm), two_var_auroc=N.auroc(y, s2),
                     signed_minus_margin=[d, lo, hi], established=N.established(lo, hi))
            out['cells'].append(c)
            print(f"   {a:10s} {lbl:36s} units {len(idx):5d} events {int(y.sum()):3d}  "
                  f"margin {c['margin_auroc']:.3f} signed {c['two_var_auroc']:.3f}  "
                  f"signed-margin {d:+.3f} {N.fmt_ci(lo,hi)}"
                  f"{'  ESTABLISHED' if c['established'] else ''}")
        print()
    co = N.build_cohort('primary')
    for H in (3, 5, 10):
        base = [r for r in co if r['horizon'] == H]
        for lbl, drop in drops:
            sel = [r for r in base if r['claim_id'] not in drop]
            acc = [r for r in sel if r['accrued']]
            if len(acc) < 80:
                continue
            s2, y, g, _ = N.cohort_loco_two_var(sel, H)
            sm = np.array([1 - abs(r['p']) for r in acc])
            d, lo, hi, _ = N.paired_ci(y, s2, sm, g)
            c = dict(cell=f'calendar H{H}', population=lbl, units=len(y), events=int(y.sum()),
                     claims=len(np.unique(g)), margin_auroc=N.auroc(y, sm),
                     two_var_auroc=N.auroc(y, s2), signed_minus_margin=[d, lo, hi],
                     established=N.established(lo, hi))
            out['cells'].append(c)
            print(f"   calendar H{H:<2d} {lbl:36s} units {len(y):5d} events {int(y.sum()):3d}  "
                  f"margin {c['margin_auroc']:.3f} signed {c['two_var_auroc']:.3f}  "
                  f"signed-margin {d:+.3f} {N.fmt_ci(lo,hi)}"
                  f"{'  ESTABLISHED' if c['established'] else ''}")
        print()
    changed = []
    for cell in {c['cell'] for c in out['cells']}:
        ref = [c for c in out['cells'] if c['cell'] == cell and c['population'] == 'all claims']
        if not ref:
            continue
        for c in out['cells']:
            if c['cell'] == cell and c['population'] != 'all claims' and c['established'] != ref[0]['established']:
                changed.append(f"{cell} / {c['population']}")
    out['conclusions_changed'] = changed
    print('conclusions changed by any cap-based exclusion:', 'NONE' if not changed else changed)
    print('written', N.save_json(out, 'E15_cap_sensitivity.json'))


if __name__ == '__main__':
    main()
