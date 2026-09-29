"""Build the six audit tables the paper uses in machine-readable (tables/Table_Astra_k_*.csv) and LaTeX
(tables/Table_Astra_k_*.tex) form from the workstream outputs. Run after all workstreams."""

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astra_common as ac
L = ac.TABLES; os.makedirs(L, exist_ok=True)
def rd(p):
    f = os.path.join(ac.TABLES, p)
    if not os.path.exists(f): print('  (missing, skipped)', p); return None
    return list(csv.DictReader(open(f)))
def f3(v):
    try: return f'{float(v):.3f}'.lstrip('0') if float(v) < 1 else f'{float(v):.3f}'
    except Exception: return str(v)
def emit(k, name, rows, cols, caption, star=False):
    if rows is None: return
    ac.save_csv(rows, os.path.join(ac.TABLES, f'Table_Astra_{k}_{name}.csv'))
    env = 'table*' if star else 'table'; width = '\\textwidth' if star else '\\columnwidth'
    lines = [f'\\begin{{{env}}}[!t]\\centering\\scriptsize', '\\setlength{\\tabcolsep}{3pt}', f'\\resizebox{{{width}}}{{!}}{{%',
             '\\begin{tabular}{@{}l' + 'r' * (len(cols) - 1) + '@{}}', '\\toprule', ' & '.join(h for _, h, _ in cols) + ' \\\\', '\\midrule']
    for r in rows:
        lines.append(' & '.join(fn(r.get(key, '')) if fn else str(r.get(key, '')).replace('_', '\\_') for key, _, fn in cols) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}}', f'\\caption{{{caption}}}', f'\\label{{tab:astra{k}}}', f'\\end{{{env}}}']
    open(os.path.join(L, f'Table_Astra_{k}_{name}.tex'), 'w').write('\n'.join(lines) + '\n'); print(f'  -> tables/Table_Astra_{k}_{name}.tex ({len(rows)} rows)')
ci = lambda v: str(v).replace('[', '[').replace(', ', ',\\,')
def main():
    # 1 anchor v2
    R = (rd('numeric_anchor_v2_main.csv') or []) + (rd('numeric_anchor_v2_by_design.csv') or []) or None
    emit(1, 'numeric_anchor_v2', R, [('evaluation', 'Evaluation', None), ('records', 'Records', str), ('claims', 'Claims', str), ('agreement', 'Agreement', f3), ('agreement_ci', '[95\\% CI]', ci),
         ('balanced_accuracy', 'Bal.\\ acc.', f3), ('macro_f1', 'Macro-F1', f3), ('macro_f1_ci', '[95\\% CI]', ci), ('recall_PROTECTIVE', 'R$_\\textsc{prot}$', f3), ('recall_HARMFUL', 'R$_\\textsc{harm}$', f3), ('recall_NULL', 'R$_\\textsc{null}$', f3)],
         'LLM-independent numeric anchor (v2) against the shipped extractor labels. The effect measure is read from the text next to each 95\\% interval and the interval is attributed to the claim only when the exposure and outcome terms co-occur in its sentence. The legacy row is the earlier reference whose scale was inferred from the sign of the limits. Claim-clustered 95\\% intervals, 2,000 resamples.', star=True)
    # 2 agreement
    emit(2, 'agreement_sensitivity', rd('agreement_sensitivity.csv'), [('definition', 'Def.', None), ('two_var_auroc', '2-var AUROC', f3), ('two_var_auroc_ci', '[95\\% CI]', ci), ('two_var_auprc', 'AUPRC', f3),
         ('delta_two_var_auroc_vs_A', '$\\Delta$ vs A', f3), ('delta_two_var_auroc_vs_A_ci', '[95\\% CI]', ci), ('compact7_auroc', 'Compact-7 AUROC', f3), ('stable_units', '$n_S$', str), ('stable_rate', 'rate$_S$', f3), ('still_forming_units', '$n_F$', str), ('still_forming_rate', 'rate$_F$', f3)],
         'Agreement-definition sensitivity on the 1,012 prospective units. A: shipped implementation; B: max share of \\textsc{protective}/\\textsc{harmful}/\\textsc{null}; C: signed-only majority share $(1+|p_t|)/2$; D: normalised-entropy consensus; C+null adds the \\textsc{null} share as a third input. Paired differences on identical claim resamples. State rates are the later-change rates after rebuilding the rule under each definition.', star=True)
    # 3 targets
    emit(3, 'subsequent_vs_cumulative_target', rd('cumulative_endpoint_results.csv'), [('endpoint', 'Endpoint', None), ('units', 'Units', str), ('events', 'Events', str), ('positive_claims', 'Pos.\\ claims', str), ('prevalence', 'Prev.', f3),
         ('compact7_auroc', 'Compact-7 AUROC', f3), ('compact7_auroc_ci', '[95\\% CI]', ci), ('compact7_auprc', 'AUPRC', f3), ('two_var_auroc', '2-var AUROC', f3), ('unfitted_1_minus_abs_p_auroc', '$1-|p_t|$', f3), ('claim_level_compact7_auroc', 'Claim AUROC', f3)],
         'Subsequent-evidence endpoint ($R_{\\mathrm{sub}}$, the primary target), cumulative updated-direction endpoint ($R_{\\mathrm{cum}}$), three-state endpoints with a no-predominant-direction band $|p|<\\delta$, and fixed-horizon variants, all on the same prospective units, leave-one-claim-out.', star=True)
    # 4 orientation
    emit(4, 'orientation_clean_sensitivity', rd('orientation_clean_results.csv'), [('benchmark', 'Benchmark', None), ('units', 'Units', str), ('claims', 'Claims', str), ('events', 'Events', str), ('compact7_auroc', 'Compact-7 AUROC', f3), ('compact7_auroc_ci', '[95\\% CI]', ci),
         ('two_var_auroc', '2-var AUROC', f3), ('claim_level_compact7_auroc', 'Claim AUROC', f3), ('stable_units', '$n_S$', str), ('stable_rate', 'rate$_S$', f3), ('still_forming_units', '$n_F$', str), ('still_forming_rate', 'rate$_F$', f3)],
         'Semantic-orientation sensitivity: the benchmark with every claim flagged by the deterministic orientation rules removed (core: deficiency, therapeutic or preventive, restriction or inversion, non-disease outcome term; strict: core plus benefit-valence outcomes), models refitted leave-one-claim-out inside the retained claims.', star=True)
    # 5 two-variable
    emit(5, 'two_variable_source_external', rd('two_variable_source_external.csv'), [('definition', 'Agr.', None), ('domain', 'Domain', None), ('units', 'Units', str), ('events', 'Events', str), ('auroc', 'AUROC', f3), ('auroc_ci', '[95\\% CI]', ci), ('auprc', 'AUPRC', f3),
         ('f1', 'F1', f3), ('claim_auroc', 'Claim AUROC', f3), ('delta_vs_compact7_auroc', '$\\Delta$ vs Compact-7', f3), ('delta_vs_compact7_auroc_ci', '[95\\% CI]', ci), ('unfitted_1_minus_abs_p_auroc', '$1-|p_t|$', f3)],
         'The two-variable model (pooled direction, agreement) under each agreement definition: leave-one-claim-out on the source and, fitted once on the 1,012 source units and never refitted, on External-1 and External-2. $\\Delta$ is paired on identical claim resamples.', star=True)
    # 7 nulls
    emit(7, 'stationary_null', rd('stationary_null_results.csv'), [('null', 'Null', None), ('metric', 'Metric', None), ('observed', 'Observed', f3), ('null_mean', 'Null mean', f3), ('null_95', 'Null 95\\% interval', ci), ('empirical_p_upper', '$p$ (observed $\\ge$ null)', f3), ('draws', 'Draws', str)],
         'Stationary-voting and permutation nulls. Null A permutes each claim\'s direction labels across its publication years; Null B draws labels i.i.d.\\ from each claim\'s stationary label distribution over the observed publication times. Every draw rebuilds the benchmark and reruns the leave-one-claim-out pipeline.')
if __name__ == '__main__': main()
