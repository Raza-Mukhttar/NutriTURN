"""Generate every LaTeX table of the revised manuscript from the result files under gptpro/results (no hand
transcription). Tables whose inputs are not yet available are written as a visible placeholder."""

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
G = _NT; RES = os.path.join(G, 'results'); TAB = os.path.join(G, 'tables'); os.makedirs(TAB, exist_ok=True)
SHORT = {'bb_deployable': 'Stationary (count-predicted)', 'margin': 'Margin', 'two_var': 'Signed two-variable', 'offset_plus_order': 'Stationary $+$ order', 'sign_invariant_full': 'Sign-invariant'}
NAMES = {'bb_deployable': 'Stationary Beta--Binomial, count-predicted (operational)', 'dm_deployable': 'Dirichlet--multinomial, count-predicted', 'bb_oracle': 'Beta--Binomial, realised future count (diagnostic)', 'bb_deployable_ratio': 'Beta--Binomial, training ratio of counts',
         'dm_oracle': 'Dirichlet--multinomial, realised count (diagnostic)', 'margin': 'Margin rule $1-|p_t|$, untrained', 'two_var': 'Two-variable logistic ($p$, $a$)', 'compact7': '\\compact logistic', 'random_forest': 'Random forest (all 55)', 'xgboost': 'XGBoost (all 55)',
         'offset_plus_order': 'Stationary offset $+$ order features', 'sign_invariant_full': 'Sign-invariant logistic', 'two_var_augmented': 'Two-variable, flip-augmented'}
COH = {'source': 'Source, claim-held-out (1{,}012 units, 71 events)', 'forward_H3': 'Calendar cohort, 3 y', 'forward_H5': 'Calendar cohort, 5 y', 'forward_H10': 'Calendar cohort, 10 y', 'external_1': 'External-1, frozen', 'external_2': 'External-2, frozen'}


from decimal import Decimal, ROUND_HALF_UP
def f3(x, d=3):
    """Round once, half-up, from the full-precision value; leading zero dropped."""
    try: x = float(x)
    except Exception: return '---'
    s = str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)); return s.replace('0.', '.', 1) if abs(x) < 1 else s


def ci(s):  # '[0.921, 0.971]' -> '[.921, .971]'
    if not s or s == '---': return '---'
    return s.replace('[0.', '[.').replace(' 0.', ' .').replace('-0.', '$-$.').replace('[-.', '[$-$.')


def dlt(d, c):
    if d in (None, '', '---'): return '---'
    d = float(d); return f"{'$+$' if d >= 0 else '$-$'}{f3(abs(d))} {ci(c)}"


def write(name, body):
    p = os.path.join(TAB, f'pro_{name}.tex'); open(p, 'w').write(body); print('  ->', os.path.relpath(p, G))


def placeholder(name, label, what):
    write(name, f"\\begin{{table}}[!t]\\centering\\small \\fbox{{\\parbox{{0.9\\columnwidth}}{{PENDING: {what}}}}}\\caption{{Pending.}}\\label{{{label}}}\\end{{table}}\n")


def load_metrics():
    p = os.path.join(RES, 'issue4', 'full_metrics.csv'); return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def tab_core():
    M = load_metrics()
    if M is None: return placeholder('core', 'tab:core', 'source comparison table (issue 4)')
    S = {r['method']: r for r in M if r['cohort'] == 'source'}
    order = ['bb_deployable', 'margin', 'two_var', 'offset_plus_order', 'sign_invariant_full']
    rows = []
    for k in order:
        if k not in S: continue
        r = S[k]; rows.append(f"{SHORT[k]} & {f3(r['auroc'])} {ci(r['auroc_ci'])} & {f3(r['auprc'])} & {f3(r['brier'], 4)} & {f3(r['logloss'])} & {f3(r['recall@10'])} & {dlt(r.get('d_auroc_vs_margin'), r.get('d_auroc_vs_margin_ci')) if k != 'margin' else 'reference'} \\\\")
        if k == 'two_var': rows.append('\\midrule')
    body = "\\begin{table*}[!t]\\centering\\footnotesize\n\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{@{}llrrrrl@{}}\n\\toprule\nScore & AUROC [95\\% CI] & AUPRC & Brier & Log loss & Recall@10\\% & $\\Delta$AUROC vs.\\ margin [95\\% CI] \\\\\n\\midrule\n" + '\n'.join(rows) + \
        "\n\\bottomrule\n\\end{tabular}\n\\caption{Source cohort, 1{,}012 future-eligible units (71 events in 23 claims, prevalence 0.070), leave-one-claim-out. Stationary (count-predicted) = the uncalibrated operational Beta--Binomial probability integrated over the training-fitted count model; Margin = the untrained rule $1-|p_t|$; Signed two-variable = the class-balanced logistic score on $(p,a)$; Stationary $+$ order = the fixed-offset model with the seven order features; Sign-invariant = the sign-invariant logistic model. Brier and log loss use training-only Platt calibration for all rows except the stationary row; Recall@10\\% is the fraction of all positive events recovered among the highest-scored 10\\% of units; $\\Delta$ is score minus margin rule, paired over the same 2{,}000 claim resamples. \\compact, the flexible learners, the realised-count and Dirichlet--multinomial references and every metric are in \\cref{app:metrics,tab:stationary}.}\n\\label{tab:core}\n\\end{table*}\n"
    write('core', body)


def tab_cross():
    M = load_metrics()
    if M is None: return placeholder('cross', 'tab:cross', 'cross-protocol table (issue 4)')
    rows = []
    for c in ['source', 'forward_H3', 'forward_H5', 'forward_H10', 'external_1', 'external_2']:
        S = {r['method']: r for r in M if r['cohort'] == c}
        if not S: continue
        u = S['margin']; lab = COH[c] if c == 'source' else f"{COH[c]} ({int(u['units']):,}, {u['events']})".replace(',', '{,}')
        cells = [f"{f3(S[k]['auroc'])} {ci(S[k]['auroc_ci'])}" for k in ('bb_deployable', 'margin', 'two_var')]
        aup = ' / '.join(f3(S[k]['auprc']) for k in ('bb_deployable', 'margin', 'two_var'))
        rows.append(f"{lab} & " + ' & '.join(cells) + f" & {aup} & {dlt(S['bb_deployable']['d_auroc_vs_margin'], S['bb_deployable']['d_auroc_vs_margin_ci'])} & {dlt(S['two_var']['d_auroc_vs_margin'], S['two_var']['d_auroc_vs_margin_ci'])} \\\\")
    body = "\\begin{table*}[!t]\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lllllll@{}}\n\\toprule\n & \\multicolumn{3}{c}{AUROC [95\\% CI]} & AUPRC & \\multicolumn{2}{c}{Paired $\\Delta$AUROC vs.\\ margin [95\\% CI]} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}\nProtocol / domain (units, events) & Stationary & Margin & Two-variable & stat.\\ / margin / 2-var & Stationary $-$ margin & Two-variable $-$ margin \\\\\n\\midrule\n" + '\n'.join(rows) + \
        "\n\\bottomrule\n\\end{tabular}}\n\\caption{Cross-protocol comparison. Calendar-cohort rows are origin-defined with complete windows (ending by 2025), claim-disjoint training and evaluation among accrued units (\\cref{app:cohort}); frozen rows apply the source fit, calibrator and count model unchanged. Stationary = count-predicted operational Beta--Binomial baseline (uncalibrated operational probabilities; count model and calibrators frozen from the source for the external rows); $\\Delta$ = score minus margin rule. The order-feature and sign-invariant diagnostics were evaluated on the source cohort only; \\compact was also evaluated in the calendar and external protocols, and its results with the complete metric suites are in \\cref{app:metrics}. }\n\\label{tab:cross}\n\\end{table*}\n"
    write('cross', body)


def tab_metrics():
    M = load_metrics()
    if M is None: return placeholder('metrics', 'tab:metrics_source', 'full metric suite (issue 4)')
    cols = [('auroc', 'AUROC'), ('auprc', 'AUPRC'), ('logloss', 'Log loss'), ('brier', 'Brier'), ('cal_slope', 'Cal.\\ slope'), ('cal_intercept', 'Cal.\\ int.'), ('ece', 'ECE'), ('precision@5', 'P@5'), ('recall@5', 'R@5'), ('precision@10', 'P@10'), ('recall@10', 'R@10'), ('precision@20', 'P@20'), ('recall@20', 'R@20'), ('nnr@10', 'NNR@10'), ('share_reviewed_for_recall_80', 'Share@R.8')]
    out = []
    for grp, cohs, lab in (('source', ['source'], 'tab:metrics_source'), ('forward', ['forward_H3', 'forward_H5', 'forward_H10'], 'tab:metrics_forward'), ('external', ['external_1', 'external_2'], 'tab:metrics_external')):
        rows = []
        for c in cohs:
            R = [r for r in M if r['cohort'] == c]
            if not R: continue
            rows.append(f"\\multicolumn{{{len(cols)+1}}}{{@{{}}l}}{{\\emph{{{COH[c]}: {int(R[0]['units']):,} units, {R[0]['events']} events}}}} \\\\".replace(',', '{,}', 1) if False else f"\\multicolumn{{{len(cols)+1}}}{{@{{}}l}}{{\\emph{{{COH[c]}, {R[0]['units']} units, {R[0]['events']} events}}}} \\\\")
            for r in R: rows.append(f"{NAMES.get(r['method'], r['method'])} & " + ' & '.join(f3(r[k], 4 if k == 'brier' else 3) for k, _ in cols) + ' \\\\')
            rows.append('\\midrule')
        if rows and rows[-1] == '\\midrule': rows.pop()
        body = "\\begin{table*}[!t]\\centering\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}l" + 'r' * len(cols) + "@{}}\n\\toprule\nScore & " + ' & '.join(l for _, l in cols) + " \\\\\n\\midrule\n" + '\n'.join(rows) + \
            f"\n\\bottomrule\n\\end{{tabular}}}}\n\\caption{{Full metric suite, {grp} cohort(s). P@$b$ and R@$b$: precision and recall when the $b$\\% highest-scored units are reviewed; NNR@10: units reviewed per event found at the 10\\% budget; Share@R.8: share of units that must be reviewed to reach recall 0.8 (workload reduction is one minus this share). Calibration slope and intercept are from a logistic regression of the outcome on the logit of the calibrated probability; ECE uses ten equal-width bins.}}\n\\label{{{lab}}}\n\\end{{table*}}\n"
        out.append(body)
    write('metrics', '\n'.join(out))


def tab_stationary():
    p = os.path.join(RES, 'issue1', 'stationary_baselines.json')
    if not os.path.exists(p): return placeholder('stationary', 'tab:stationary', 'stationary baselines (issue 1)')
    R = json.load(open(p))['source']; rows = []
    for k in ('bb_oracle', 'bb_deployable', 'bb_deployable_ratio', 'dm_oracle', 'dm_deployable', 'margin', 'two_var', 'compact7'):
        r = R[k]; rows.append(f"{NAMES[k]} & {f3(r['auroc'])} {ci(r['auroc_ci'])} & {f3(r['auprc'])} {ci(r['auprc_ci'])} & {f3(r['logloss'])} & {f3(r['brier'], 4)} & {f3(r['cal_slope'], 2)} & {f3(r['ece'])} & {dlt(r.get('d_auroc_vs_margin'), r.get('d_auroc_vs_margin_ci')) if k != 'margin' else 'reference'} & {dlt(r.get('d_auroc_vs_two_var'), r.get('d_auroc_vs_two_var_ci')) if k not in ('two_var',) else 'reference'} \\\\")
    pv = R['prevalence_only']
    body = "\\begin{table*}[!t]\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lllllllll@{}}\n\\toprule\nScore & AUROC [95\\% CI] & AUPRC [95\\% CI] & Log loss & Brier & Cal.\\ slope & ECE & $\\Delta$AUROC vs.\\ margin & $\\Delta$AUROC vs.\\ two-variable \\\\\n\\midrule\n" + '\n'.join(rows) + \
        f"\n\\bottomrule\n\\end{{tabular}}}}\n\\caption{{Stationary posterior-predictive references for the archived operational endpoint on the source cohort (1{{,}}012 units, 71 events); all rows evaluate the archived operational endpoint $\\pi^{{\\mathrm{{op}}}}_t$. The stationary posterior-predictive rows use uncalibrated probabilities; probability metrics for the margin, two-variable and \\compact scores use the training-only calibration of \\cref{{sec:protocols}} and \\cref{{app:metrics}}. Realised-count rows use observed future counts and are diagnostic rather than origin-available forecasts; count-predicted rows integrate over the training-fitted count model of \\cref{{app:bb}}; the ratio row draws the future/past count ratio from the training claims. The raw-majority probability of \\cref{{eq:bb}} is a separately labelled diagnostic in the released per-unit files (columns \\texttt{{*\\_rawsign}}), not a row of this table. A prevalence-only forecast has log loss {f3(pv['logloss'])} and Brier {f3(pv['brier'], 4)}. The calibrated two-variable score improves on the deployable baseline by {f3(R['two_var_calibrated_minus_bb_deployable_brier'][0], 4)} in Brier {ci(R['two_var_calibrated_minus_bb_deployable_brier'][1])} and {f3(R['two_var_calibrated_minus_bb_deployable_logloss'][0])} in log loss {ci(R['two_var_calibrated_minus_bb_deployable_logloss'][1])} (paired, positive = better).}}\n\\label{{tab:stationary}}\n\\end{{table*}}\n"
    write('stationary', body)


def tab_order():
    p = os.path.join(RES, 'issue2', 'temporal_offset.json')
    if not os.path.exists(p): return placeholder('order', 'tab:order', 'offset model (issue 2)')
    R = json.load(open(p)); M = R['models']; rows = []
    lab = {'stationary_offset_only': 'Stationary offset only ($\\pi_t$)', 'offset_plus_order': 'Offset $+$ order features', 'offset_plus_composition': 'Offset $+$ composition', 'offset_plus_composition_plus_order': 'Offset $+$ composition $+$ order', 'order_only_no_offset': 'Order features only, no offset', 'two_var': 'Two-variable logistic', 'margin': 'Margin rule'}
    for k in lab:
        r = M[k]; ll = f3(r['d_logloss_vs_offset']) if r.get('d_logloss_vs_offset') is not None else '---'; llc = ci(r['d_logloss_vs_offset_ci']) if r.get('d_logloss_vs_offset_ci') else ''
        rows.append(f"{lab[k]} & {f3(r['auroc'])} {ci(r['auroc_ci'])} & {f3(r['auprc'])} & {f3(r['logloss'])} & {dlt(r['d_auroc_vs_offset'], r['d_auroc_vs_offset_ci']) if k != 'stationary_offset_only' else 'reference'} & {dlt(r['d_auprc_vs_offset'], r['d_auprc_vs_offset_ci']) if k != 'stationary_offset_only' else 'reference'} & {(ll + ' ' + llc) if ll != '---' else '---'} \\\\")
    P = R['permutation']; co = R['order_coefficients_full_fit_C1']
    coefs = ', '.join(f"{k.replace('_', ' ')} {v:+.2f}" for k, v in co.items() if k != 'intercept')
    body = "\\begin{table*}[!t]\\centering\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lllllll@{}}\n\\toprule\nModel & AUROC [95\\% CI] & AUPRC & Log loss & $\\Delta$AUROC vs.\\ offset & $\\Delta$AUPRC vs.\\ offset & $\\Delta$log loss vs.\\ offset (positive = better) \\\\\n\\midrule\n" + '\n'.join(rows) + \
        f"\n\\bottomrule\n\\end{{tabular}}}}\n\\caption{{Fixed-offset diagnostic: incremental value of temporal order beyond the uncalibrated operational stationary probability (source cohort, leave-one-claim-out, nested ridge selection). The offset is the uncalibrated operational stationary probability. Superseded within-prefix randomisation ({P['draws']} independent prefix shuffles, $C=1$; see text), recalculated with the operational offset (run 2026-09-23): observed gain {P['observed_gain_C1']:+.4f} (nested fit {P['observed_gain_nested']:+.4f}), replicate mean {P['null_mean']:+.4f}, s.d.\\ {P['null_sd']:.4f}; no $p$-value is used. Standardised coefficients of the full-data fit: {coefs}.}}\n\\label{{tab:order}}\n\\end{{table*}}\n"
    write('order', body)


def tab_signflip():
    p = os.path.join(RES, 'issue3', 'sign_invariant.json')
    if not os.path.exists(p): return placeholder('signflip', 'tab:signflip', 'sign-flip (issue 3)')
    R = json.load(open(p)); M = R['models']; F = R['sign_flip']; A = R['flip_augmentation']; rows = []
    lab = {'margin': 'Margin rule', 'two_var_signed': 'Two-variable, signed', 'compact7_signed': '\\compact, signed', 'sign_invariant_two_var': 'Sign-invariant ($|p_t|$, $a_t$)', 'sign_invariant_full': 'Sign-invariant, full'}
    for k in lab:
        r = M[k]; fl = F.get(k)
        flip = f"{f3(fl['mean'])} [{f3(fl['min'])}, {f3(fl['max'])}]" if fl and 'mean' in fl else ('unchanged' if k.startswith('sign') else '---')
        rows.append(f"{lab[k]} & {f3(r['auroc'])} {ci(r['auroc_ci'])} & {f3(r['auprc'])} & {dlt(r['d_auroc_vs_margin'], r['d_auroc_vs_margin_ci']) if k != 'margin' else 'reference'} & {dlt(r['d_auroc_vs_two_var'], r['d_auroc_vs_two_var_ci']) if k != 'two_var_signed' else 'reference'} & {flip} \\\\")
    for k, l in (('two_var_augmented', 'Two-variable, flip-augmented'), ('compact7_augmented', '\\compact, flip-augmented')):
        r = A[k]; rows.append(f"{l} & {f3(r['auroc'])} {ci(r['auroc_ci'])} & {f3(r['auprc'])} & --- & {dlt(r['d_auroc_vs_two_var'], r['d_auroc_vs_two_var_ci'])} & {f3(r['auroc_under_full_flip'])} (full flip) \\\\")
    n_ = M['two_var_signed_p_negative']; p_ = M['two_var_signed_p_positive']
    body = "\\begin{table*}[!t]\\centering\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}llllll@{}}\n\\toprule\nModel & AUROC [95\\% CI] & AUPRC & $\\Delta$AUROC vs.\\ margin & $\\Delta$AUROC vs.\\ signed two-variable & AUROC under claim-wise flips, mean [min, max] \\\\\n\\midrule\n" + '\n'.join(rows) + \
        f"\n\\bottomrule\n\\end{{tabular}}}}\n\\caption{{Sign-invariance experiments on the source cohort. Claim-wise flips exchange the signed labels of exactly half of the claims (all cutoffs of a claim together) in each of {F['draws']} draws and refit the signed models leave-one-claim-out; a full exchange of every claim leaves the signed scores unchanged because the fitted sign of $\\beta$ reverses. Direction strata of the signed two-variable score: \\textsc{{protective}}-majority units ({n_['units']} units, {n_['events']} events) AUROC {f3(n_['auroc'])} (margin {f3(n_['margin_auroc'])}, sign-invariant {f3(n_['sign_invariant_full_auroc'])}); \\textsc{{harmful}}-majority units ({p_['units']}, {p_['events']}) {f3(p_['auroc'])} (margin {f3(p_['margin_auroc'])}, sign-invariant {f3(p_['sign_invariant_full_auroc'])}).}}\n\\label{{tab:signflip}}\n\\end{{table*}}\n"
    write('signflip', body)


def tab_census():
    p = os.path.join(RES, 'issue5', 'census.csv')
    if not os.path.exists(p): return placeholder('census', 'tab:census', 'cohort census (issue 5)')
    C = list(csv.DictReader(open(p))); B = {(r['horizon'], r['origin']): r for r in csv.DictReader(open(os.path.join(RES, 'issue5', 'by_origin.csv')))}; rows = []
    for H in ('3', '5', '10'):
        rows.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{Horizon {H} years}}}} \\\\")
        for r in [x for x in C if x['horizon'] == H]:
            b = B.get((H, r['origin'])); st = 'complete' if r['complete'] == 'True' else 'window extends past 2025: excluded'
            perf = f"{f3(b['bb_deployable_auroc_cond'])} / {f3(b['margin_auroc_cond'])} / {f3(b['two_var_auroc_cond'])} / {f3(b['compact7_auroc_cond'])}" if b else '---'
            rows.append(f"{r['origin']} & {r['window']} & {r['candidate_claims']} & {r['accrued']} & {r['censored_low_accrual']} & {r['events_among_accrued']} & {f3(r['event_rate_among_accrued'])} & {perf} & {st} \\\\")
        rows.append('\\midrule')
    rows.pop()
    body = "\\begin{table*}[!t]\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}llrrrrrll@{}}\n\\toprule\nOrigin & Window & Candidates & Accrued & Censored & Events & Rate & AUROC among accrued: stationary / margin / two-var / \\compact & Status \\\\\n\\midrule\n" + '\n'.join(rows) + \
        "\n\\bottomrule\n\\end{tabular}}\n\\caption{Census of the origin-defined calendar cohort. Candidates are claims with at least 40 records before the origin; censored units have fewer than ten signed records in the window; the rate is the event rate among accrued units. Per-origin AUROCs are claim-disjoint and rest on 4--11 events.}\n\\label{tab:census}\n\\end{table*}\n"
    write('census', body)


def tab_snapshot():
    p = os.path.join(RES, 'issue7', 'manifest.json')
    if not os.path.exists(p): return placeholder('snapshot', 'tab:snapshot', 'snapshot (issue 7)')
    M = json.load(open(p)); V = M['version_comparison']; S = M['stored_snapshot']; ts = M['reretrieval']['snapshot_timestamp_utc'][:19].replace('T', ' '); UQ = json.load(open(os.path.join(RES, 'guide', 'checks.json')))['reretrieval_unique']
    N = lambda n: f'{int(n):,}'.replace(',', '{,}')
    SP = json.load(open(os.path.join(RES, 'guide', 'snapshot_provenance.json')))
    SPW = f"{SP['lower_bound_utc']} to {SP['upper_bound_utc'][:16]} UTC"
    body = f"""\\begin{{table}}[!t]\\centering\\scriptsize
\\setlength{{\\tabcolsep}}{{4pt}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{@{{}}lr@{{}}}}
\\toprule
Quantity & Value \\\\
\\midrule
Archived snapshot (v1): claim queries & {S['claims']} \\\\
Archived claim--record associations & {N(S['records_total'])} \\\\
Archived records dated 2026 (partial year) & {N(S['records_by_year'].get('2026', 0))} \\\\
Original snapshot retrieval window & {SPW} \\\\
Exact original retrieval timestamp & not recorded \\\\
Re-retrieval timestamp (UTC) & {ts} \\\\
Associations returned at re-retrieval (v2) & {N(V['new_ids'])} \\\\
Archived associations returned again & {N(V['in_both'])} ({100*V['in_both']/V['stored_ids']:.1f}\\%) \\\\
Distinct archived identifiers / returned again & {N(UQ['stored_unique_pmids'])} / {N(UQ['stored_unique_recovered'])} ({100*UQ['recovered_share_unique']:.1f}\\%) \\\\
Distinct identifiers at re-retrieval & {N(UQ['new_unique_pmids'])} \\\\
Archived associations no longer returned & {N(V['stored_only'])} \\\\
Associations new at re-retrieval & {N(V['new_only'])} \\\\
Per-claim Jaccard, median [min, max] & {V['median_jaccard']:.3f} [{V['min_jaccard']:.3f}, {V['max_jaccard']:.3f}] \\\\
Claims losing $>10$\\% of stored identifiers & {V['claims_with_stored_only_gt_10pct']} \\\\
External corpora re-retrievable & no (unit-level files only) \\\\
\\bottomrule
\\end{{tabular}}}}
\\caption{{Snapshot versioning. Version 1 is the stored, non-regenerable derived snapshot on which every experiment runs; version 2 is the timestamped ESearch re-retrieval of the same query strings (identifier lists and hashes only). Per-query counts, translations and identifier hashes are in the released manifest.}}
\\label{{tab:snapshot}}
\\end{{table}}
"""
    write('snapshot', body)


def tab_reextract():
    p = os.path.join(RES, 'issue6', 'extractor_agreement.json')
    if not os.path.exists(p):
        placeholder('extractors', 'tab:extractors', 'extractor agreement, main (issue 6)'); placeholder('support', 'tab:support', 'common support'); return placeholder('reextract', 'tab:reextract', 'extractor agreement, appendix (issue 6)')
    R = json.load(open(p)); rec = R['record_level']; U = R['unit_level']; S = R['soft_labels']; L = ['PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR']
    C = json.load(open(os.path.join(RES, 'guide', 'checks.json'))); X = C['event_crosstab']; BR = C['branch_counts']; ER = C['event_retention_by_branch']; TS = C['tie_vs_sparse']
    conf = rec['confusion_llama_rows_qwen_cols']; N = lambda n: f'{int(n):,}'.replace(',', '{,}')
    q, c, l = U['qwen'], U['consensus'], U['llama']
    rows = '\n'.join(f"{lab} & {u['events']} ({u['positive_claims']}) & {u['events_both']} & {f3(u['event_jaccard_with_llama'])} & {f3(u['margin_auroc'])} & {f3(u['bb_deployable_auroc'])} & {f3(u['two_var_auroc'])} {ci(u['two_var_auroc_ci'])} & {dlt(u['two_var_minus_margin_auroc'][0], u['two_var_minus_margin_auroc'][1])} \\\\"
                     for lab, u in (('Deployed (Llama-3.1-8B, first token)', l), ('Second (Qwen2.5-7B, full string)', q), ('Consensus (disagreements $\\to$ \\textsc{unclear})', c)))
    main = f"""\\begin{{table*}}[!t]\\centering\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{@{{}}lrrrrrll@{{}}}}
\\toprule
Annotation pipeline & Events (claims) & Shared w/ deployed & Jaccard & Margin AUROC & Stationary AUROC & Signed AUROC [95\\% CI] & Signed $-$ margin [95\\% CI] \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}}}
\\caption{{Cross-pipeline comparison on the same 1{{,}}012 archived claim-cutoff units. Each row rebuilds the inputs and endpoint under the indicated annotation pipeline and refits the learned score leave-one-claim-out; the stationary column is the count-predicted operational probability. Shared-event counts and Jaccard are relative to the deployed pipeline. Neither annotation pipeline is a reference standard. Claim--record association-level agreement, sparse-evidence cases, the deployed score evaluated on the other endpoints, soft-label draws and the common-support sensitivity are in \\cref{{app:reextract}}.}}
\\label{{tab:extractors}}
\\end{{table*}}
"""
    write('extractors', main)
    crows = '\n'.join(f"\\textsc{{{a.lower()}}} & " + ' & '.join(N(conf[a][b]) for b in L) + ' \\\\' for a in L)
    brows = '\n'.join(f"{lab} & {ER[k]['units']} & {ER[k]['llama_events']} & {ER[k]['qwen_events']} & {ER[k]['both']} \\\\" for k, lab in (('llama:signed', 'Deployed: signed branch'), ('llama:null_dominant', 'Deployed: \\textsc{null}-dominant'), ('llama:fallback', 'Deployed: fewer than 8 signed'), ('qwen:signed', 'Second: signed branch'), ('qwen:null_dominant', 'Second: \\textsc{null}-dominant'), ('qwen:fallback', 'Second: fewer than 8 signed')))
    drows = '\n'.join(f"{lab} & {u['events_with_fewer_than_8_signed_pre']} & {f3(u.get('deployed_llama_two_var_score_evaluated_on_this_target_auroc', u['two_var_auroc']))} \\\\" for lab, u in (('Deployed', l), ('Second', q), ('Consensus', c)))
    app = f"""\\begin{{table*}}[!t]\\centering\\scriptsize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\\toprule
Deployed (rows) $\\backslash$ second (columns) & \\textsc{{protective}} & \\textsc{{harmful}} & \\textsc{{null}} & \\textsc{{unclear}} \\\\
\\midrule
{crows}
\\bottomrule
\\end{{tabular}}
\\hspace{{4mm}}
\\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\\toprule
Agreement branch of the unit & Units & Deployed events & Second events & Both \\\\
\\midrule
{brows}
\\bottomrule
\\end{{tabular}}
\\vspace{{4pt}}

\\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\\toprule
Endpoint cross-tabulation & Second: event & Second: non-event & Total \\\\
\\midrule
Deployed: event & {X['both']} & {X['llama_only']} & {X['both'] + X['llama_only']} \\\\
Deployed: non-event & {X['qwen_only']} & {X['neither']} & {X['qwen_only'] + X['neither']} \\\\
Total & {X['both'] + X['qwen_only']} & {X['llama_only'] + X['neither']} & {X['both'] + X['llama_only'] + X['qwen_only'] + X['neither']} \\\\
\\bottomrule
\\end{{tabular}}
\\hspace{{4mm}}
\\begin{{tabular}}{{@{{}}lrr@{{}}}}
\\toprule
Pipeline & Events on sparse histories ($<8$ signed) & Deployed score on this endpoint, AUROC \\\\
\\midrule
{drows}
\\bottomrule
\\end{{tabular}}
\\caption{{Cross-pipeline sensitivity, detail. Claim--record association-level agreement with its denominators: {100*rec['raw_agreement']:.1f}\\% of all {N(rec['records_scored'])} associations ($\\kappa={rec['kappa_4way']:.2f}$), {100*rec['resolved_both_agreement']:.1f}\\% of associations both pipelines resolve ($\\kappa={rec['resolved_both_kappa']:.2f}$), {100*rec['signed_only_agreement']:.1f}\\% of the {N(rec['signed_only_records'])} both call signed ($\\kappa={rec['signed_only_kappa']:.2f}$); agreement is not accuracy. {N(C['record_level_derived']['signed_to_null'])} deployed signed annotations become \\textsc{{null}} and {N(C['record_level_derived']['direct_sign_switch'])} switch sign (\\textsc{{null}} share {100*rec['llama_label_shares']['NULL']:.1f}\\% against {100*rec['qwen_label_shares']['NULL']:.1f}\\%). Exact ties with at least eight signed records: {TS['llama']['exact_ties_ge8']} units under the deployed pipeline (all events), {TS['qwen']['exact_ties_ge8']} under the second. The last block distinguishes the deployed score (deployed inputs, trained on the deployed endpoint) evaluated against each endpoint from the rebuilt-and-refitted scores of \\cref{{tab:extractors}}. Soft-label draws ({S['draws']}; variation across annotation draws, no claim-bootstrap layer): {S['events_q025']:.0f}--{S['events_q975']:.0f} events, signed score {f3(S['two_var_auroc_q025'])}--{f3(S['two_var_auroc_q975'])}, margin rule {f3(S['margin_auroc_q025'])}--{f3(S['margin_auroc_q975'])}. The Qwen first-token read-out defines the same {C['qwen_first_token_readout']['events']} events as the full-string read-out.}}
\\label{{tab:reextract}}
\\end{{table*}}
"""
    write('reextract', app)
    # ---- common-support sensitivity (E2)
    CS = R.get('common_support')
    if not CS: return placeholder('support', 'tab:support', 'common support (issue 6)')
    srows = '\n'.join(f"{lab} & {CS['units']} / {CS['claims']} & {CS[k]['events']} / {CS[k]['positive_claims']} & {CS['shared_events']} & {f3(CS['event_jaccard'])} & {f3(CS[k]['margin_auroc'])} / {f3(CS[k]['margin_auprc'])} & {f3(CS[k]['bb_deployable_auroc'])} / {f3(CS[k]['bb_deployable_auprc'])} & {f3(CS[k]['two_var_auroc'])} / {f3(CS[k]['two_var_auprc'])} & {dlt(CS[k]['two_var_minus_margin_auroc'][0], CS[k]['two_var_minus_margin_auroc'][1])} \\\\"
                      for lab, k in (('Deployed', 'deployed'), ('Second', 'second')))
    frows = '\n'.join(f"{lab} & 1{{,}}012 / 160 & {u['events']} / {u['positive_claims']} & {u['events_both']} & {f3(u['event_jaccard_with_llama'])} & {f3(u['margin_auroc'])} / {f3(u['margin_auprc'])} & {f3(u['bb_deployable_auroc'])} / {f3(u['bb_deployable_auprc'])} & {f3(u['two_var_auroc'])} / {f3(u['two_var_auprc'])} & {dlt(u['two_var_minus_margin_auroc'][0], u['two_var_minus_margin_auroc'][1])} \\\\"
                      for lab, u in (('Deployed', l), ('Second', q)))
    _nd, _ns = CS['deployed']['resamples_evaluable_of_2000'], CS['second']['resamples_evaluable_of_2000']
    _boot = ('All 2{,}000 claim-clustered resamples were evaluable for both pipeline comparisons.' if _nd == _ns == 2000
             else f'Claim-clustered resamples evaluable within the restricted population: {_nd} and {_ns} of 2{{,}}000.')
    sup = f"""\\begin{{table*}}[!t]\\centering\\scriptsize
\\setlength{{\\tabcolsep}}{{4pt}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{@{{}}llllllllll@{{}}}}
\\toprule
Support / training scope & Pipeline & Units / claims & Events / positive claims & Shared events & Jaccard & Margin AUROC / AUPRC & Stationary AUROC / AUPRC & Signed AUROC / AUPRC & Signed $-$ margin [95\\% CI] \\\\
\\midrule
\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{Full archived cohort, out-of-fold scores}}}} \\\\
{frows.replace(chr(10), chr(10)).replace('Deployed &', 'Full cohort & Deployed &').replace('Second &', 'Full cohort & Second &')}
\\midrule
\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{Common support ($\\ge8$ signed pre-cutoff records under both pipelines), full-cohort-trained out-of-fold scores restricted}}}} \\\\
{srows.replace('Deployed &', 'Common support, restricted & Deployed &').replace('Second &', 'Common support, restricted & Second &')}
\\bottomrule
\\end{{tabular}}}}
\\caption{{Common-support sensitivity. The support mask (hash in the released manifest) keeps units with at least eight signed pre-cutoff records under both pipelines ({CS['units']} units; {CS['excluded_units']} excluded, of which {CS['excluded_second_sparse']} sparse under the second pipeline and {CS['excluded_deployed_sparse']} under the deployed pipeline; the deployed sparse unit {'is' if CS['deployed_sparse_unit_within_second_sparse'] else 'is not'} among the second pipeline's sparse units). Each pipeline keeps its own operational endpoint; scores are the full-cohort leave-one-claim-out predictions restricted to the mask, not refitted. {_boot}}}
\\label{{tab:support}}
\\end{{table*}}
"""
    write('support', sup)


def tab_matched():
    p = os.path.join(RES, 'guide', 'recalibrated_order.json')
    if not os.path.exists(p): return placeholder('matched', 'tab:matched', 'matched recalibration (guide)')
    R = json.load(open(p)); Rr, Rz = R['recalibrated_stationary'], R['recalibrated_stationary_plus_order']
    rows = []
    for k, lab, sgn in (('auroc', 'AUROC', 1), ('auprc', 'AUPRC', 1), ('logloss', 'Log loss', -1), ('brier', 'Brier score', -1)):
        d, c = R[f'd_{k}_RZ_minus_R (positive = RZ better)']; rows.append(f"{lab} & {f3(Rr[k], 4 if k == 'brier' else 3)} & {f3(Rz[k], 4 if k == 'brier' else 3)} & {dlt(d, c)} \\\\")
    body = "\\begin{table}[!t]\\centering\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}lrrl@{}}\n\\toprule\nMetric & Recalibrated operational stationary & $+$ order features & Improvement [95\\% CI] \\\\\n\\midrule\n" + '\n'.join(rows) + \
        "\n\\bottomrule\n\\end{tabular}\n\\caption{Matched recalibration comparison on the source cohort. The reference is a fitted logistic recalibration of the uncalibrated operational stationary logit $\\logit\\pi^{\\mathrm{op}}_t$ (a different procedure from the uncalibrated baseline of \\cref{tab:stationary} and from the fixed-offset model of \\cref{tab:order}); the augmented model additionally uses the seven order features. Both use unweighted $L_2$ logistic regression with $C=1$, the same leave-one-claim-out folds and the same claim-bootstrap resamples. Positive values favour the augmented model: augmented minus reference for AUROC and AUPRC, reference minus augmented for log loss and Brier score, computed from unrounded paired predictions. None of the paired intervals excludes zero.}\n\\label{tab:matched}\n\\end{table}\n"
    write('matched', body)


if __name__ == '__main__':
    for fn in (tab_core, tab_cross, tab_metrics, tab_stationary, tab_order, tab_signflip, tab_census, tab_snapshot, tab_reextract, tab_matched): fn()
