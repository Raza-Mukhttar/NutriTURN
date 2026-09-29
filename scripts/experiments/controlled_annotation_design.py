"""E8 analysis: the model x prompt x read-out design.

Consolidates the raw cell files, then reports per cell the label distribution and NULL share, the
pairwise kappa between cells, the rebuilt endpoint (events, Jaccard against every other cell,
sparse-history events), and the method comparison on the common support of all cells. Finally it
decomposes the change in positive class into a prompt effect, a model effect and a read-out effect.

  python analyses/nfx_e8_analysis.py
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
import os, sys, json, glob, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

LABELS = ['PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR']
RAW = os.path.join(N.ANNOT, 'raw')
CELLS = ['Aarch', 'A', 'Aprime', 'B', 'C', 'D']
DESC = {'Aarch': "Llama-3.1-8B, prompt P1, first token (the ARCHIVED deployed labels)",
        'A': "Llama-3.1-8B, prompt P1, first token (rescored here)",
        'Aprime': "Llama-3.1-8B, prompt P1, full string",
        'B': "Llama-3.1-8B, prompt P2, full string",
        'C': "Qwen2.5-7B, prompt P1, full string",
        'D': "Qwen2.5-7B, prompt P2, full string"}


def consolidate():
    """Write annotations/<cell>.jsonl with a single 'label' field for every scored cell.

    Cell A is the first-token read-out of the SAME forward pass as cell A', so the read-out
    comparison is free of any other difference.
    """
    made = {}
    for cell in ('Aprime', 'B', 'C', 'D'):
        fs = sorted(glob.glob(os.path.join(RAW, f'cell{cell}_shard*.jsonl')))
        if not fs:
            continue
        rows = []
        for f in fs:
            for line in open(f):
                rows.append(json.loads(line))
        if len(rows) < 112453:
            print(f'  cell {cell}: only {len(rows)} of 112453 records; skipping')
            continue
        with open(os.path.join(N.ANNOT, f'cell{cell}.jsonl'), 'w') as fh:
            for r in rows:
                fh.write(json.dumps({'claim_id': r['claim_id'], 'pmid': r['pmid'],
                                     'label': r['label_full']}) + '\n')
        made[cell] = len(rows)
        if cell == 'Aprime':
            with open(os.path.join(N.ANNOT, 'cellA.jsonl'), 'w') as fh:
                for r in rows:
                    fh.write(json.dumps({'claim_id': r['claim_id'], 'pmid': r['pmid'],
                                         'label': r['label_first']}) + '\n')
            made['A'] = len(rows)
            with open(os.path.join(N.ANNOT, 'cellAarch.jsonl'), 'w') as fh:
                for r in rows:
                    fh.write(json.dumps({'claim_id': r['claim_id'], 'pmid': r['pmid'],
                                         'label': r['primary_label']}) + '\n')
            made['Aarch'] = len(rows)
            same = sum(1 for r in rows if r['label_first'] == r['primary_label'])
            ro = sum(1 for r in rows if r['label_first'] == r['label_full'])
            print(f"  read-out contrast (A vs A', identical forward pass): "
                  f"{ro}/{len(rows)} identical ({100*ro/len(rows):.2f}%)")
            print(f"  batching reproducibility (rescored first-token vs the ARCHIVED deployed "
                  f"labels): {same}/{len(rows)} identical ({100*same/len(rows):.2f}%). Same model, "
                  f"prompt, truncation and read-out; only the batch composition differs.")
    return made


def kappa(a, b):
    n = len(a)
    idx = {c: i for i, c in enumerate(LABELS)}
    M = np.zeros((4, 4))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    po = np.trace(M) / n
    pe = (M.sum(0) / n * M.sum(1) / n).sum()
    return float((po - pe) / (1 - pe)) if pe < 1 else float('nan')


def main():
    print('E8 controlled annotation: model x prompt x read-out\n')
    made = consolidate()
    have = [c for c in CELLS if c in made or os.path.exists(os.path.join(N.ANNOT, f'cell{c}.jsonl'))]
    if len(have) < 2:
        raise SystemExit('need at least two scored cells')
    print(f'cells available: {have}\n')
    A = N.assoc()
    keys = [(r['claim_id'], r['pmid']) for r in A]
    lab = {}
    for c in have:
        L = N.label_column(f'cell{c}')
        lab[c] = [L.get(k, 'UNCLEAR') for k in keys]
    out = {'cells': {}, 'kappa': {}, 'endpoint': {}, 'decomposition': {}}

    print('1. label distribution per cell (association level, n = %d)' % len(keys))
    print(f"{'cell':8s} {'description':44s} " + ' '.join(f'{l[:4]:>7s}' for l in LABELS) + '   NULL share')
    for c in have:
        cnt = {l: lab[c].count(l) / len(keys) for l in LABELS}
        out['cells'][c] = dict(description=DESC[c], shares=cnt)
        print(f"{c:8s} {DESC[c]:44s} " + ' '.join(f'{cnt[l]:7.3f}' for l in LABELS)
              + f'   {cnt["NULL"]:.3f}')

    print('\n2. pairwise Cohen kappa (all associations / both-signed subset)')
    print(f"{'pair':18s} {'kappa all':>10s} {'agree all':>10s} {'kappa signed':>13s} {'n signed':>9s}")
    for x, y in itertools.combinations(have, 2):
        k = kappa(lab[x], lab[y])
        ag = float(np.mean([p == q for p, q in zip(lab[x], lab[y])]))
        both = [(p, q) for p, q in zip(lab[x], lab[y])
                if p in ('PROTECTIVE', 'HARMFUL') and q in ('PROTECTIVE', 'HARMFUL')]
        ks = kappa([p for p, _ in both], [q for _, q in both]) if both else float('nan')
        out['kappa'][f'{x}|{y}'] = dict(kappa_all=k, agreement_all=ag, kappa_signed=ks,
                                        n_signed=len(both))
        print(f'{x + " vs " + y:18s} {k:10.3f} {ag:10.3f} {ks:13.3f} {len(both):9d}')

    print('\n3. rebuilt endpoint per cell')
    tab, ev = {}, {}
    for c in have:
        rows = N.unit_table(f'cell{c}')
        for r, acc in zip(rows, N.build_cohort_accrual5(rows)):
            r['accrual_5y'] = acc
        tab[c] = rows
        ev[c] = {i for i, r in enumerate(rows) if r['y']}
        out['endpoint'][c] = dict(events=len(ev[c]),
                                  positive_claims=len({r['claim_id'] for r in rows if r['y']}),
                                  sparse_units=sum(r['sparse'] for r in rows),
                                  sparse_events=sum(r['sparse'] and r['y'] for r in rows),
                                  null_dominant_units=sum(1 for r in rows
                                                          if r['null_share'] >= 0.5 and r['n_signed'] >= 8))
        e = out['endpoint'][c]
        print(f"  {c:8s} events {e['events']:3d}  positive claims {e['positive_claims']:3d}  "
              f"sparse units {e['sparse_units']:3d} (events {e['sparse_events']:3d})  "
              f"NULL-dominant units {e['null_dominant_units']:4d}")

    print('\n4. Jaccard of the positive-event sets')
    print('        ' + ' '.join(f'{c:>8s}' for c in have))
    jac = {}
    for x in have:
        line = []
        for y in have:
            j = len(ev[x] & ev[y]) / max(len(ev[x] | ev[y]), 1)
            jac[f'{x}|{y}'] = j
            line.append(f'{j:8.3f}')
        print(f'{x:8s} ' + ' '.join(line))
    out['event_jaccard'] = jac

    print('\n5. method comparison on the common support of all cells')
    mask = np.ones(len(tab[have[0]]), bool)
    for c in have:
        mask &= np.array([r['sparse'] == 0 for r in tab[c]])
    print(f'   common support: {int(mask.sum())} of {len(mask)} units')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core_reanalyses import _cell
    out['method'] = []
    for c in have:
        cell, _ = _cell(tab[c], mask, f'cell {c}', refit=True)
        cell['cell'] = c
        out['method'].append(cell)
        d, lo, hi = cell['signed_minus_margin']
        print(f"  {c:8s} units {cell['units']:4d} events {cell['events']:3d}  "
              f"margin {cell['margin_auroc']:.3f} stationary {cell['stationary_auroc']:.3f} "
              f"signed {cell['two_var_auroc']:.3f}  signed-margin {d:+.3f} {N.fmt_ci(lo,hi)}"
              f"{'  ESTABLISHED' if cell['signed_minus_margin_established'] else ''}")

    print('\n6. decomposition of the change in positive class')
    pairs = [('batching only (archived vs rescored, same config)', 'Aarch', 'A'),
             ('prompt effect, Llama (P1 vs P2)', 'Aprime', 'B'),
             ('prompt effect, Qwen  (P1 vs P2)', 'C', 'D'),
             ('model effect, P1 (Llama vs Qwen)', 'Aprime', 'C'),
             ('model effect, P2 (Llama vs Qwen)', 'B', 'D'),
             ('read-out effect, Llama P1 (first vs full)', 'A', 'Aprime')]
    print(f"{'contrast':44s} {'kappa':>7s} {'Jaccard':>8s} {'NULL shift':>11s} {'events':>12s}")
    for name, x, y in pairs:
        if x not in have or y not in have:
            continue
        kk = out['kappa'].get(f'{x}|{y}', out['kappa'].get(f'{y}|{x}'))['kappa_all']
        j = jac[f'{x}|{y}']
        dn = out['cells'][y]['shares']['NULL'] - out['cells'][x]['shares']['NULL']
        out['decomposition'][name] = dict(kappa=kk, jaccard=j, null_shift=dn,
                                          events=[out['endpoint'][x]['events'],
                                                  out['endpoint'][y]['events']])
        print(f'{name:44s} {kk:7.3f} {j:8.3f} {dn:+11.3f} '
              f"{out['endpoint'][x]['events']:5d} -> {out['endpoint'][y]['events']:<5d}")
    dec = out['decomposition']
    cand = {k: v for k, v in dec.items()
            if 'effect' in k and 'read-out' not in k and 'batching' not in k}
    if cand:
        driver = min(cand, key=lambda k: cand[k]['jaccard'])
        out['driver'] = driver
        print(f'\nlowest Jaccard among the prompt and model contrasts: {driver} '
              f'({cand[driver]["jaccard"]:.3f})')
        big_null = max(cand, key=lambda k: abs(cand[k]['null_shift']))
        print(f'largest NULL-share shift: {big_null} ({cand[big_null]["null_shift"]:+.3f})')
    print('\nwritten', N.save_json(out, 'E8_controlled_annotation.json'))


if __name__ == '__main__':
    main()
