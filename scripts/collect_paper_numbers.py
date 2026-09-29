"""Collect every placeholder number of the manuscript from the authoritative result files (one rounding, half-up)."""

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
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_latex_tables_main import f3, ci, dlt
G = _NT; R = os.path.join(G, 'results'); J = lambda *p: json.load(open(os.path.join(R, *p)))
z = lambda x, d=3: ('0' + f3(x, d)) if f3(x, d).startswith('.') else f3(x, d)
S = J('issue1', 'stationary_baselines.json')['source']; H2 = J('historical_rawsign', 'issue2', 'temporal_offset.json'); I2 = J('issue2', 'temporal_offset.json'); RC = J('guide', 'recalibrated_order.json'); I5 = J('issue5', 'origin_cohort.json')['pooled']; I6 = J('issue6', 'extractor_agreement.json'); C = J('guide', 'checks.json'); E = J('guide', 'endpoint_contract.json')
M4 = {(r['cohort'], r['method']): r for r in csv.DictReader(open(os.path.join(R, 'issue4', 'full_metrics.csv')))}
SP = J('guide', 'snapshot_provenance.json')
def pd(d, c):
    lo, hi = [float(x) for x in c.strip('[]').split(', ')]; sd = ('+' if d >= 0 else '-') + '0' + f3(abs(d)) if abs(d) < 1 else f"{d:+.3f}"
    return f"${sd}$ \\ci{{{lo:.3f}}}{{{hi:.3f}}}"
cs = I6['common_support']; U = I6['unit_level']
N = {
 'src_stat_auroc': z(S['bb_deployable']['auroc']), 'src_stat_ci': '\\ci{' + S['bb_deployable']['auroc_ci'].strip('[]').replace(', ', '}{') + '}', 'src_stat_auprc': z(S['bb_deployable']['auprc']), 'src_dm_auroc': z(S['dm_deployable']['auroc']), 'src_or_auroc': z(S['bb_oracle']['auroc']),
 'src_margin_minus_stat': pd(-S['bb_deployable']['d_auroc_vs_margin'], '[' + ', '.join(f'{-float(x):.3f}' for x in reversed(S['bb_deployable']['d_auroc_vs_margin_ci'].strip('[]').split(', '))) + ']'),
 'src_twovar_minus_stat': pd(-S['bb_deployable']['d_auroc_vs_two_var'], '[' + ', '.join(f'{-float(x):.3f}' for x in reversed(S['bb_deployable']['d_auroc_vs_two_var_ci'].strip('[]').split(', '))) + ']'),
 'src_stat_logloss': z(S['bb_deployable']['logloss']), 'src_stat_brier': z(S['bb_deployable']['brier']), 'src_stat_slope': z(S['bb_deployable']['cal_slope'], 2),
 'expected_events_op': f"{J('issue1', 'stationary_baselines.json')['issue9_validation']['expected_events_analytic']:.1f}", 'expected_events_raw': f"{sum(float(r['bb_oracle_rawsign']) for r in csv.DictReader(open(os.path.join(R, 'issue1', 'source_scores.csv')))):.1f}",
 'q_stat_auroc': z(U['qwen']['bb_deployable_auroc']), 'c_stat_auroc': z(U['consensus']['bb_deployable_auroc']),
 'off_gain': f"{I2['models']['offset_plus_order']['d_auroc_vs_offset']:+.3f}".replace('+', '$+$').replace('-', '$-$'), 'off_gain_ci': '\\ci{' + I2['models']['offset_plus_order']['d_auroc_vs_offset_ci'].strip('[]').replace(', ', '}{') + '}', 'off_auroc': z(I2['models']['offset_plus_order']['auroc']), 'off_auprc': z(I2['models']['offset_plus_order']['auprc']),
 'rc_auroc': pd(*RC['d_auroc_RZ_minus_R (positive = RZ better)']), 'rc_auprc': pd(*RC['d_auprc_RZ_minus_R (positive = RZ better)']), 'rc_logloss': pd(*RC['d_logloss_RZ_minus_R (positive = RZ better)']), 'rc_brier': pd(*RC['d_brier_RZ_minus_R (positive = RZ better)']),
 'h3_stat_auroc': z(M4[('forward_H3', 'bb_deployable')]['auroc']), 'h3_stat_auprc': z(M4[('forward_H3', 'bb_deployable')]['auprc']), 'h10_stat_auroc': z(M4[('forward_H10', 'bb_deployable')]['auroc']),
 'e1_2v_minus_margin': pd(float(M4[('external_1', 'two_var')]['d_auroc_vs_margin']), M4[('external_1', 'two_var')]['d_auroc_vs_margin_ci']),
 'e2_2v_minus_margin': pd(float(M4[('external_2', 'two_var')]['d_auroc_vs_margin']), M4[('external_2', 'two_var')]['d_auroc_vs_margin_ci']),
 'e1_stat_auroc': z(M4[('external_1', 'bb_deployable')]['auroc']), 'e2_stat_auroc': z(M4[('external_2', 'bb_deployable')]['auroc']), 'e2_stat_auprc': z(M4[('external_2', 'bb_deployable')]['auprc']),
 'pp_mae': f"{C['posterior_predictive_simulation']['mean_abs_error']:.4f}", 'pp_max': f"{C['posterior_predictive_simulation']['max_abs_error']:.3f}", 'pp_se': f"{C['posterior_predictive_simulation']['mean_mc_se']:.4f}", 'pp_within': f"{100*C['posterior_predictive_simulation']['share_within_2se']:.1f}",
 'plug_r_bb': f"{J('issue1', 'stationary_baselines.json')['issue9_validation_addendum']['pearson_bb_vs_nullB']:.3f}", 'plug_r_plug': f"{J('issue1', 'stationary_baselines.json')['issue9_validation_addendum']['pearson_plugin_vs_nullB']:.3f}",
 'audit_mismatch': str(E['audit_source']['archived_vs_reproduced_mismatches']), 'changed_units': str(E['change_log_E1']['units_with_changed_count_predicted_score']), 'max_change': f"{E['change_log_E1']['max_abs_change_count_predicted']:.3f}",
 'perm_null_mean': f"${I2['permutation']['null_mean']:+.3f}$", 'perm_null_sd': f"{I2['permutation']['null_sd']:.3f}", 'perm_obs': f"${I2['permutation']['observed_gain_C1']:+.3f}$",
 'perm_old_null_mean': f"${H2['permutation']['null_mean']:+.3f}$", 'perm_old_null_sd': f"{H2['permutation']['null_sd']:.3f}", 'perm_old_obs': f"${H2['permutation']['observed_gain_C1']:+.3f}$",
 'perm_r': f"{J('issue1', 'stationary_baselines.json')['issue9_validation']['pearson_r']:.2f}", 'perm_check_events': f"{J('issue1', 'stationary_baselines.json')['issue9_validation']['expected_events_simulated']:.1f}", 'expected_events_plugin_raw': f"{J('issue1', 'stationary_baselines.json')['issue9_validation_addendum']['expected_events_plugin_raw']:.1f}", 'expected_events_plugin_op': f"{J('issue1', 'stationary_baselines.json')['issue9_validation_addendum']['expected_events_plugin_op']:.1f}",
}
# snapshot provenance bounds (post hoc; no exact retrieval timestamp was recorded)
N['snap_lo'] = SP['lower_bound_utc']
N['snap_hi'] = SP['upper_bound_utc'][:16]                      # minute precision for prose
N['snap_window'] = f"{SP['lower_bound_utc']} through {SP['upper_bound_utc'][:16]} UTC"
N['snap_pmid'] = f"{SP['highest_archived_identifiers'][0]:,}".replace(',', '{,}')
N['snap_files'] = str(SP['upper_bound_detail']['files'])

# common-support sentences (E2), written from computed values
D, Q = cs['deployed'], cs['second']
N['common_support_sentence'] = (f"On the common support of {cs['units']} units with at least eight signed pre-cutoff records under both pipelines ({cs['excluded_units']} excluded), events number {D['events']} and {Q['events']} with {cs['shared_events']} shared (Jaccard {z(cs['event_jaccard'])}), and the ordering persists: "
                                f"margin rule {z(D['margin_auroc'])} against {z(Q['margin_auroc'])}, stationary baseline {z(D['bb_deployable_auroc'])} against {z(Q['bb_deployable_auroc'])}, signed score {z(D['two_var_auroc'])} against {z(Q['two_var_auroc'])} (signed minus margin {pd(*Q['two_var_minus_margin_auroc'])} under the second pipeline; \\cref{{tab:support}}).")
N['common_support_appendix'] = (f"\\Cref{{tab:support}} restricts the full-cohort out-of-fold predictions to the {cs['units']} units ({cs['claims']} claims) with at least eight signed pre-cutoff records under both pipelines; exact ties with adequate support are retained. Low event overlap persists (Jaccard {z(cs['event_jaccard'])}) and the method ordering under the second pipeline is unchanged "
                                f"(signed minus margin {pd(*Q['two_var_minus_margin_auroc'])}; signed minus stationary {pd(*Q['two_var_minus_bb_auroc'])}). The reversal persists when sparse histories are excluded from the evaluation population, using the existing full-cohort out-of-fold models; no model was refitted on the restricted population, so this sensitivity does not remove every possible training influence of sparse histories.")
json.dump(N, open(os.path.join(R, 'guide', 'numbers.json'), 'w'), indent=1); print('numbers collected', len(N))
