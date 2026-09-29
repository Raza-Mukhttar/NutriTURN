"""Issue 5: the 2-variable model (pooled direction, agreement) on the source domain and, fitted once
and never refitted, on both external domains.

Agreement is computed under every definition of Issue 2 so the choice is transparent; the manuscript
uses the definition named by --final. On the source the agreement is rebuilt from the record streams;
on the external corpora, whose records are not distributed, it is rebuilt from the unit-level counts
(PROTECTIVE, HARMFUL and NULL counts are recoverable from pooled_direction, n_nonnull and
n_directional). The shipped 'agreement' column of each external file is asserted to equal the
count-based reconstruction of definition A. Namespace: agreement_v2.
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
import os, sys, json, argparse, math, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astra_common as ac
nm = ac.nm
DEFS = ['A', 'B', 'C', 'D']


def agreement_from_counts(nP, nH, nN):
    n = nP + nH + nN; ns_ = nP + nH
    if ns_ < 8: return {d: 0.0 for d in DEFS}
    A = nN / n if nN / n >= 0.5 else max(nP, nH) / ns_
    sh = np.array([nP, nH, nN], float) / n; ent = -sum(p * math.log(p) for p in sh if p > 0)
    return {'A': A, 'B': float(sh.max()), 'C': max(nP, nH) / ns_, 'D': 1 - ent / math.log(3)}


def counts_of_row(r):
    ns_ = int(round(r['n_nonnull'])); nH = int(round(ns_ * (1 + r['pooled_direction']) / 2)); nP = ns_ - nH; nN = int(round(r['n_directional'])) - ns_
    return nP, nH, nN


def fit_frozen(X, y):
    sc = StandardScaler().fit(X); m = LogisticRegression(max_iter=3000, class_weight='balanced').fit(sc.transform(X), y)
    return lambda Z: m.predict_proba(sc.transform(Z))[:, 1], m, sc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--final', default='B'); a = ap.parse_args()
    rows, keys = ac.load_units(); P, y, g = ac.prospective(rows)
    # source agreement under every definition, from the streams
    AG = {d: [] for d in DEFS}
    for r in P:
        py, pc, _, _ = ac.split(r['claim_id'], int(r['cutoff_year']))
        for d in DEFS: AG[d].append(ac.direction_feats(py, pc, int(r['cutoff_year']), d)['agreement'])
    AG = {d: np.array(v) for d, v in AG.items()}
    assert np.max(np.abs(AG['A'] - np.array([r['agreement'] for r in P]))) < 1e-9
    # count-based reconstruction must agree with the stream-based one on the source (validates the external route)
    recon = np.array([agreement_from_counts(*counts_of_row(r))['A'] for r in P]); assert np.max(np.abs(recon - AG['A'])) < 1e-6, np.max(np.abs(recon - AG['A']))
    pt = np.array([r['pooled_direction'] for r in P]); X7 = nm.X_of(P, nm.COMPACT7)
    s7, t7 = ac.loco(X7, y, g); ref7 = ac.perf(y, s7, g, t7)
    res = {'protocol': 'source: leave-one-claim-out, F1 threshold on the training fold; external: one fit on all 1,012 source prospective units, applied unchanged; 2,000 claim-clustered resamples, seed 3; target sgn0 subsequent-evidence change',
           'compact7_reference_source': ref7, 'source': {}, 'external': {}}
    tab = []
    ext = {1: ac.load_external(1), 2: ac.load_external(2)}
    for d in DEFS:
        X2 = np.column_stack([pt, AG[d]]); s2, t2 = ac.loco(X2, y, g); pf = ac.perf(y, s2, g, t2)
        X7d = X7.copy(); X7d[:, nm.COMPACT7.index('agreement')] = AG[d]; s7d, t7d = ac.loco(X7d, y, g); pf7 = ac.perf(y, s7d, g, t7d)
        src = {'two_variable': pf, 'two_variable_claim_level': ac.claim_level(y, s2, g), 'compact7_same_agreement': pf7,
               'paired_two_var_minus_compact7_reference': {'auroc': ac.paired(y, s2, s7, g, 'auroc'), 'auprc': ac.paired(y, s2, s7, g, 'auprc')},
               'paired_two_var_minus_compact7_same_agreement': {'auroc': ac.paired(y, s2, s7d, g, 'auroc'), 'auprc': ac.paired(y, s2, s7d, g, 'auprc')},
               'coefficients_full_fit': None}
        f2, m2, sc2 = fit_frozen(X2, y); f7, m7, sc7 = fit_frozen(X7d, y)
        src['coefficients_full_fit'] = {'standardised': dict(zip(['pooled_direction', f'agreement_{d}'], [round(float(c), 4) for c in m2.coef_[0]])), 'intercept': round(float(m2.intercept_[0]), 4)}
        res['source'][d] = src
        row = {'definition': d, 'domain': 'source (LOCO)', 'units': pf['units'], 'events': pf['events'], 'auroc': pf['auroc'], 'auroc_ci': pf['auroc_ci'], 'auprc': pf['auprc'], 'auprc_ci': pf['auprc_ci'],
               'f1': pf.get('f1'), 'claim_auroc': src['two_variable_claim_level']['auroc'], 'claim_auroc_ci': src['two_variable_claim_level']['auroc_ci'],
               'delta_vs_compact7_auroc': src['paired_two_var_minus_compact7_reference']['auroc'][0], 'delta_vs_compact7_auroc_ci': src['paired_two_var_minus_compact7_reference']['auroc'][1],
               'delta_vs_compact7_auprc': src['paired_two_var_minus_compact7_reference']['auprc'][0], 'delta_vs_compact7_auprc_ci': src['paired_two_var_minus_compact7_reference']['auprc'][1],
               'compact7_same_agreement_auroc': pf7['auroc']}
        tab.append(row); print(f"  {d} source 2-var {pf['auroc']} {pf['auroc_ci']} auprc {pf['auprc']}  d vs C7 {row['delta_vs_compact7_auroc']} {row['delta_vs_compact7_auroc_ci']}", flush=True)
        res['external'][d] = {}
        for k in (1, 2):
            erows, _ = ext[k]; EP, ey, eg = ac.prospective(erows)
            eag = np.array([agreement_from_counts(*counts_of_row(r))[d] for r in EP]); ept = np.array([r['pooled_direction'] for r in EP])
            if d == 'A': assert np.max(np.abs(eag - np.array([r['agreement'] for r in EP]))) < 1e-6, 'external agreement reconstruction'
            es2 = f2(np.column_stack([ept, eag])); EX7 = nm.X_of(EP, nm.COMPACT7); EX7[:, nm.COMPACT7.index('agreement')] = eag; es7 = f7(EX7)
            base = {'1 - |p_t|': 1 - np.abs(ept), f'1 - agreement ({d})': 1 - eag, 'recent dissent': np.array([r['dissent_recent'] for r in EP])}
            e = {'two_variable_frozen': nm.perf(ey, es2, eg), 'compact7_frozen_same_agreement': nm.perf(ey, es7, eg),
                 'paired_two_var_minus_compact7': {'auroc': ac.paired(ey, es2, es7, eg, 'auroc'), 'auprc': ac.paired(ey, es2, es7, eg, 'auprc')}, 'unfitted': {}}
            for bn, bs in base.items():
                e['unfitted'][bn] = {**nm.perf(ey, bs, eg), 'paired_two_var_minus_signal': {'auroc': ac.paired(ey, es2, bs, eg, 'auroc'), 'auprc': ac.paired(ey, es2, bs, eg, 'auprc')}}
            res['external'][d][f'External-{k}'] = e
            tab.append({'definition': d, 'domain': f'External-{k} (frozen)', 'units': e['two_variable_frozen']['units'], 'events': e['two_variable_frozen']['events'], 'auroc': e['two_variable_frozen']['auroc'],
                        'auroc_ci': e['two_variable_frozen']['auroc_ci'], 'auprc': e['two_variable_frozen']['auprc'], 'auprc_ci': e['two_variable_frozen']['auprc_ci'],
                        'delta_vs_compact7_auroc': e['paired_two_var_minus_compact7']['auroc'][0], 'delta_vs_compact7_auroc_ci': e['paired_two_var_minus_compact7']['auroc'][1],
                        'compact7_same_agreement_auroc': e['compact7_frozen_same_agreement']['auroc'],
                        'unfitted_1_minus_abs_p_auroc': e['unfitted']['1 - |p_t|']['auroc'], 'delta_vs_1_minus_abs_p_auroc': e['unfitted']['1 - |p_t|']['paired_two_var_minus_signal']['auroc'][0],
                        'delta_vs_1_minus_abs_p_auroc_ci': e['unfitted']['1 - |p_t|']['paired_two_var_minus_signal']['auroc'][1],
                        'unfitted_low_agreement_auroc': e['unfitted'][f'1 - agreement ({d})']['auroc'], 'unfitted_recent_dissent_auroc': e['unfitted']['recent dissent']['auroc']})
            print(f"     ext{k} frozen 2-var {e['two_variable_frozen']['auroc']} {e['two_variable_frozen']['auroc_ci']}  C7 {e['compact7_frozen_same_agreement']['auroc']}  1-|p| {e['unfitted']['1 - |p_t|']['auroc']}", flush=True)
    res['final_definition'] = a.final
    ac.save_json(res, os.path.join(ac.RESULTS, 'two_variable_source_external.json')); ac.save_csv(tab, os.path.join(ac.TABLES, 'two_variable_source_external.csv'))
    plt = ac.plt_setup(); fig, ax = plt.subplots(figsize=(7.2, 3.6)); doms = ['source (LOCO)', 'External-1 (frozen)', 'External-2 (frozen)']; W = .2
    for i, d in enumerate(DEFS):
        v = [next(t['auroc'] for t in tab if t['definition'] == d and t['domain'] == dm) for dm in doms]
        ax.bar(np.arange(3) + (i - 1.5) * W, v, W, label=f'2-variable, agreement {d}')
    v7 = [ref7['auroc']] + [res['external'][a.final][f'External-{k}']['compact7_frozen_same_agreement']['auroc'] for k in (1, 2)]
    ax.scatter(np.arange(3), v7, marker='_', s=600, color='#1A1A1A', label=f'Compact-7 (agreement {a.final})', zorder=5)
    vu = [None] + [res['external'][a.final][f'External-{k}']['unfitted']['1 - |p_t|']['auroc'] for k in (1, 2)]
    ax.scatter([1, 2], vu[1:], marker='x', s=50, color='#B3402F', label='1-|p_t| unfitted', zorder=5)
    ax.set_xticks(range(3)); ax.set_xticklabels(doms); ax.set_ylim(.5, 1); ax.set_ylabel('AUROC'); ax.legend(fontsize=6.5, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.12))
    ax.set_title('Direction + consensus: two variables, fitted once on the source', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(ac.FIGURES, 'two_variable_transfer.pdf')); fig.savefig(os.path.join(ac.FIGURES, 'two_variable_transfer.png'), dpi=200); print('DONE', flush=True)


if __name__ == '__main__':
    main()
