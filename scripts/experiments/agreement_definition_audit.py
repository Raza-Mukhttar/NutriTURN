"""Issue 2: audit the agreement definition and test alternatives.

The frozen `agg_verdict` returns the NULL share when NULL is at least half of the resolved records, and
otherwise the signed-only majority share, which equals (1 + |p_t|)/2. This script (i) quantifies the
two branches and the boundary at NULL share 0.50 on the shipped benchmark, (ii) verifies that the
rebuild reproduces the shipped columns exactly, (iii) rebuilds agreement, recent agreement and the
state rule under four definitions, and (iv) evaluates 2-variable and Compact-7 models under each with
paired claim-clustered intervals. Namespace: agreement_v2.
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
import os, sys, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astra_common as ac
nm = ac.nm
DEFS = {'A': 'current implementation (NULL share if NULL >= 0.50 of resolved, else signed majority share)',
        'B': 'continuous all-resolved consensus: max share of PROTECTIVE / HARMFUL / NULL',
        'C': 'signed-only majority share (= (1+|p_t|)/2), NULL share kept as a separate variable',
        'D': 'normalised-entropy consensus over the PROTECTIVE / HARMFUL / NULL shares'}


def main():
    rows, keys = ac.load_units(); P, y, g = ac.prospective(rows)
    print(f'{len(rows)} units, {len(P)} prospective, {int(y.sum())} events', flush=True)

    # ---- rebuild under every definition and verify the frozen one reproduces the shipped columns
    feats = {d: [] for d in DEFS}; audit_rows = []
    maxdiff = collections.defaultdict(float)
    for r in rows:
        cid, cut = r['claim_id'], int(r['cutoff_year'])
        py, pc, _, _ = ac.split(cid, cut)
        for d in DEFS:
            f = ac.direction_feats(py, pc, cut, d); f['state'] = ac.state_of(f)
            f.update(claim_id=cid, cutoff_year=cut, definition=d, n_post_t=r['n_post_t'], y_sub=r['y_sub'],
                     null_frac_shipped=r['null_frac'], **{k: r[k] for k in ('n_pre_t', 'val_mean', 'val_sd', 'n_rr')})
            feats[d].append(f)
        fA = feats['A'][-1]
        for k in ('pooled_direction', 'agreement', 'agreement_recent', 'dissent_recent', 'agreement_contraction', 'n_directional', 'n_nonnull'):
            maxdiff[k] = max(maxdiff[k], abs(fA[k] - r[k]))
        assert fA['state'] == r['maturity_state'], (cid, cut, fA['state'], r['maturity_state'])
        ag = ac.agreement_defs(pc[pc != 99] if True else pc)
        audit_rows.append({'claim_id': cid, 'cutoff_year': cut, 'prospective': int(r['n_post_t'] >= nm.POST20),
                           'n_resolved': ag['n_resolved'], 'n_signed': ag['n_signed'], 'null_share': round(ag['null_share'], 4),
                           'branch': 'zero (n_signed<8)' if ag['n_signed'] < 8 else ('NULL-dominant' if ag['null_share'] >= 0.5 else 'signed-vote'),
                           'agreement_A': round(ag['A'], 4), 'agreement_B': round(ag['B'], 4), 'agreement_C': round(ag['C'], 4),
                           'agreement_D': round(ag['D'], 4), 'abs_p_t': round(abs(r['pooled_direction']), 4),
                           'identity_check_A_eq_half_1_plus_abs_p': (round(ag['A'], 6) == round((1 + abs(r['pooled_direction'])) / 2, 6)) if ag['n_signed'] >= 8 and ag['null_share'] < 0.5 else ''})
    assert max(maxdiff.values()) < 1e-9, maxdiff
    print('rebuild under definition A reproduces every shipped direction column and every state label (max |diff| %.1e)' % max(maxdiff.values()), flush=True)

    # ---- audit of the current definition
    A = [a for a in audit_rows]; Ap = [a for a in A if a['prospective']]
    br = collections.Counter(a['branch'] for a in A); brp = collections.Counter(a['branch'] for a in Ap)
    ns_ = np.array([a['null_share'] for a in A]); bins = [(0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60)]
    near = {f'{lo:.2f}-{hi:.2f}': int(((ns_ >= lo) & (ns_ < hi)).sum()) for lo, hi in bins}
    ident = [a['identity_check_A_eq_half_1_plus_abs_p'] for a in A if a['identity_check_A_eq_half_1_plus_abs_p'] != '']
    sv = [a for a in A if a['branch'] == 'signed-vote']; nd = [a for a in A if a['branch'] == 'NULL-dominant']
    corr = lambda xs, ys: float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float('nan')
    dep = {'signed_vote_branch': {'n': len(sv), 'pearson_agreement_vs_abs_p': round(corr([a['agreement_A'] for a in sv], [a['abs_p_t'] for a in sv]), 4),
                                  'identity_holds_on_every_unit': bool(all(ident)), 'units_checked': len(ident)},
           'null_dominant_branch': {'n': len(nd), 'pearson_agreement_vs_abs_p': round(corr([a['agreement_A'] for a in nd], [a['abs_p_t'] for a in nd]), 4),
                                    'pearson_agreement_vs_null_share': round(corr([a['agreement_A'] for a in nd], [a['null_share'] for a in nd]), 4)},
           'all_units': {'n': len(A), 'pearson_agreement_vs_abs_p': round(corr([a['agreement_A'] for a in A], [a['abs_p_t'] for a in A]), 4),
                         'spearman_agreement_vs_abs_p': None}}
    from scipy.stats import spearmanr
    dep['all_units']['spearman_agreement_vs_abs_p'] = round(float(spearmanr([a['agreement_A'] for a in A], [a['abs_p_t'] for a in A]).correlation), 4)
    for d in 'BCD':
        dep[f'definition_{d}_vs_abs_p_all_units'] = round(corr([a[f'agreement_{d}'] for a in A], [a['abs_p_t'] for a in A]), 4)
    # boundary crossings: consecutive cutoffs of one claim on different branches
    by = collections.defaultdict(list)
    for a in A: by[a['claim_id']].append(a)
    cross = []
    for cid, L in by.items():
        L = sorted(L, key=lambda a: a['cutoff_year'])
        for a, b in zip(L, L[1:]):
            if a['branch'] != b['branch'] and 'zero' not in a['branch'] + b['branch']:
                cross.append({'claim_id': cid, 'from_cutoff': a['cutoff_year'], 'to_cutoff': b['cutoff_year'], 'from_branch': a['branch'], 'to_branch': b['branch'],
                              'null_share_from': a['null_share'], 'null_share_to': b['null_share'], 'agreement_A_from': a['agreement_A'], 'agreement_A_to': b['agreement_A'],
                              'jump_A': round(b['agreement_A'] - a['agreement_A'], 4), 'agreement_B_from': a['agreement_B'], 'agreement_B_to': b['agreement_B'],
                              'jump_B': round(b['agreement_B'] - a['agreement_B'], 4)})
    cross.sort(key=lambda c: -abs(c['jump_A']))
    audit = {'definition_A': DEFS['A'], 'branch_counts_all_units': dict(br), 'branch_counts_prospective_units': dict(brp),
             'units_with_null_share_in_bin': near, 'dependence': dep, 'boundary_crossings': len(cross), 'boundary_crossing_examples': cross[:12],
             'mean_abs_agreement_jump_at_crossings_A': round(float(np.mean([abs(c['jump_A']) for c in cross])), 4) if cross else None,
             'mean_abs_agreement_jump_at_crossings_B': round(float(np.mean([abs(c['jump_B']) for c in cross])), 4) if cross else None}
    ac.save_csv(audit_rows, os.path.join(ac.ns('data', 'agreement_v2'), 'agreement_audit_units.csv'))
    ac.save_csv(cross, os.path.join(ac.ns('data', 'agreement_v2'), 'agreement_boundary_crossings.csv'))

    # ---- experiments per definition (prospective units, y_sub)
    idx = {(r['claim_id'], int(r['cutoff_year'])): i for i, r in enumerate(rows)}
    pidx = [idx[(r['claim_id'], int(r['cutoff_year']))] for r in P]
    pt = np.array([r['pooled_direction'] for r in P]); nullf = np.array([r['null_frac'] for r in P])
    X7 = nm.X_of(P, nm.COMPACT7)
    res, tab, scores = {}, [], {}
    s_ref2 = s_ref7 = None
    for d in DEFS:
        F = [feats[d][i] for i in pidx]
        ag = np.array([f['agreement'] for f in F])
        X2 = np.column_stack([pt, ag]); s2, t2 = ac.loco(X2, y, g)
        X7d = X7.copy(); X7d[:, nm.COMPACT7.index('agreement')] = ag; s7, t7 = ac.loco(X7d, y, g)
        if d == 'A': s_ref2, s_ref7 = s2, s7
        st = np.array([f['state'] for f in F]); m = st != 'unassessable'
        rates = {}
        for sname in ('stable', 'still_forming', 'unstable'):
            k = st == sname
            rates[sname] = {'units': int(k.sum()), 'events': int(y[k].sum()), 'rate': ac.rate_ci(y[k], g[k]) if k.sum() else (None, None)}
        d2, d7 = ac.perf(y, s2, g, t2), ac.perf(y, s7, g, t7)
        e = {'definition': DEFS[d], 'two_variable': d2, 'compact7_variant': d7,
             'two_variable_claim_level': ac.claim_level(y, s2, g), 'state_rates': rates,
             'assessable_units': int(m.sum())}
        if d != 'A':
            e['paired_delta_vs_A'] = {'two_variable_auroc': ac.paired(y, s2, s_ref2, g, 'auroc'), 'two_variable_auprc': ac.paired(y, s2, s_ref2, g, 'auprc'),
                                      'compact7_auroc': ac.paired(y, s7, s_ref7, g, 'auroc'), 'compact7_auprc': ac.paired(y, s7, s_ref7, g, 'auprc')}
        res[d] = e; scores[d] = (s2, s7)
        tab.append({'definition': d, 'description': DEFS[d], 'two_var_auroc': d2['auroc'], 'two_var_auroc_ci': d2['auroc_ci'], 'two_var_auprc': d2['auprc'],
                    'two_var_auprc_ci': d2['auprc_ci'], 'two_var_f1': d2.get('f1'), 'compact7_auroc': d7['auroc'], 'compact7_auroc_ci': d7['auroc_ci'],
                    'compact7_auprc': d7['auprc'], 'compact7_auprc_ci': d7['auprc_ci'],
                    'delta_two_var_auroc_vs_A': e.get('paired_delta_vs_A', {}).get('two_variable_auroc', ('', ''))[0],
                    'delta_two_var_auroc_vs_A_ci': e.get('paired_delta_vs_A', {}).get('two_variable_auroc', ('', ''))[1],
                    'delta_compact7_auroc_vs_A': e.get('paired_delta_vs_A', {}).get('compact7_auroc', ('', ''))[0],
                    'delta_compact7_auroc_vs_A_ci': e.get('paired_delta_vs_A', {}).get('compact7_auroc', ('', ''))[1],
                    'stable_units': rates['stable']['units'], 'stable_rate': rates['stable']['rate'][0], 'stable_rate_ci': rates['stable']['rate'][1],
                    'still_forming_units': rates['still_forming']['units'], 'still_forming_rate': rates['still_forming']['rate'][0],
                    'still_forming_rate_ci': rates['still_forming']['rate'][1], 'unstable_units': rates['unstable']['units'],
                    'unstable_rate': rates['unstable']['rate'][0], 'unstable_rate_ci': rates['unstable']['rate'][1]})
        print(f"  {d}: 2-var AUROC {d2['auroc']} {d2['auroc_ci']}  C7 {d7['auroc']}  stable {rates['stable']['units']} @ {rates['stable']['rate'][0]}  still {rates['still_forming']['units']} @ {rates['still_forming']['rate'][0]}", flush=True)
    # C with NULL share as a third variable, and the two single signals
    ag_c = np.array([feats['C'][i]['agreement'] for i in pidx])
    s3, t3 = ac.loco(np.column_stack([pt, ag_c, nullf]), y, g); d3 = ac.perf(y, s3, g, t3)
    res['C_plus_null_share'] = {'definition': 'p_t, signed-only agreement, NULL share (3 variables)', 'three_variable': d3,
                                'paired_delta_vs_A_two_variable_auroc': ac.paired(y, s3, s_ref2, g, 'auroc')}
    tab.append({'definition': 'C+null', 'description': res['C_plus_null_share']['definition'], 'two_var_auroc': d3['auroc'], 'two_var_auroc_ci': d3['auroc_ci'],
                'two_var_auprc': d3['auprc'], 'two_var_auprc_ci': d3['auprc_ci'], 'two_var_f1': d3.get('f1'),
                'delta_two_var_auroc_vs_A': res['C_plus_null_share']['paired_delta_vs_A_two_variable_auroc'][0],
                'delta_two_var_auroc_vs_A_ci': res['C_plus_null_share']['paired_delta_vs_A_two_variable_auroc'][1]})
    s1, t1 = ac.loco(pt.reshape(-1, 1), y, g); d1 = ac.perf(y, s1, g, t1)
    du = nm.perf(y, 1 - np.abs(pt), g)
    res['p_t_only'] = d1; res['one_minus_abs_p_t_unfitted'] = du
    tab.append({'definition': 'p_t only', 'description': 'logistic on pooled direction alone', 'two_var_auroc': d1['auroc'], 'two_var_auroc_ci': d1['auroc_ci'], 'two_var_auprc': d1['auprc'], 'two_var_auprc_ci': d1['auprc_ci'], 'two_var_f1': d1.get('f1')})
    tab.append({'definition': '1-|p_t|', 'description': 'unfitted signal', 'two_var_auroc': du['auroc'], 'two_var_auroc_ci': du['auroc_ci'], 'two_var_auprc': du['auprc'], 'two_var_auprc_ci': du['auprc_ci']})

    # ---- outputs
    allf = [f for d in DEFS for f in feats[d]]
    ac.save_frame(allf, os.path.join(ac.DATA, 'agreement_sensitivity_features.parquet'))
    ac.save_json({'audit': audit, 'experiments': res, 'protocol': 'leave-one-claim-out, 2,000 claim-clustered resamples, seed 3; target sgn0 subsequent-evidence change; prospective units n_post_t >= 20'},
                 os.path.join(ac.RESULTS, 'agreement_sensitivity.json'))
    ac.save_csv(tab, os.path.join(ac.TABLES, 'agreement_sensitivity.csv'))
    np.save(os.path.join(ac.ns('results', 'agreement_v2'), 'scores_two_var.npy'), np.column_stack([scores[d][0] for d in DEFS]))
    # figure
    plt = ac.plt_setup(); fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.1))
    ax[0].hist(ns_, bins=np.linspace(0, 1, 41), color='#2D6CA2', alpha=.85); ax[0].axvline(.5, color='#B3402F', ls='--', lw=1.2)
    ax[0].set_xlabel('NULL share among resolved records'); ax[0].set_ylabel('claim-cutoff units'); ax[0].set_title('Where the branch switches', fontsize=9)
    col = {'signed-vote': '#2D6CA2', 'NULL-dominant': '#B3402F', 'zero (n_signed<8)': '#999999'}
    for b, c in col.items():
        S = [a for a in A if a['branch'] == b]
        ax[1].scatter([a['abs_p_t'] for a in S], [a['agreement_A'] for a in S], s=6, alpha=.6, color=c, label=f'{b} (n={len(S)})')
    xx = np.linspace(0, 1, 50); ax[1].plot(xx, (1 + xx) / 2, color='#1A1A1A', lw=.9, ls=':', label='(1+|p|)/2')
    ax[1].set_xlabel('|p_t|'); ax[1].set_ylabel('agreement (current definition)'); ax[1].legend(fontsize=6.5, frameon=False); ax[1].set_title('Algebraic dependence', fontsize=9)
    names = list(DEFS) + ['C+null', 'p_t only']; vals = [t['two_var_auroc'] for t in tab if t['definition'] in names]
    ax[2].bar(range(len(names)), vals, color='#2D6CA2'); ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, fontsize=7)
    ax[2].set_ylim(.8, 1); ax[2].set_ylabel('LOCO AUROC, 2-variable model'); ax[2].set_title('Definitions perform alike', fontsize=9)
    for i, v in enumerate(vals): ax[2].text(i, v + .004, f'{v:.3f}', ha='center', fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(ac.FIGURES, 'agreement_definition_comparison.pdf')); fig.savefig(os.path.join(ac.FIGURES, 'agreement_definition_comparison.png'), dpi=200)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
