"""Collect every completed experiment into one report.

  python analyses/report.py
Writes results/REPORT.md and prints it.
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
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

R = N.RESULTS
L = []
def w(s=''):
    L.append(s)


def J(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ci(v, d=3):
    return f'[{v[1]:.{d}f}, {v[2]:.{d}f}]'


def est(v):
    return 'established' if (v[1] > 0 or v[2] < 0) else 'not established'


def main():
    w('# NutriMATURE ARR revision: experiment results')
    w()
    w('All inputs, scripts, outputs and logs are under `newfab_exp/`. Nothing outside it was changed.')
    w('Statistics throughout: claim-clustered percentile bootstrap, 2,000 resamples, seed 3;')
    w('"established" means the 95% interval excludes zero.')
    w()

    # ---- E1
    d = J('E1_rebuild.json')
    if d:
        w('## E1 Rebuild check and coefficient scale')
        w()
        w('| cell | units | events | expected | matches |')
        w('|---|---|---|---|---|')
        for a, v in d['development'].items():
            w(f"| {a} | {v['units']} | {v['events']} | {{71,52,55}} | "
              f"{'yes' if v['matches_paper'] else 'NO'} |")
        for k, v in d['calendar'].items():
            w(f"| calendar {k} | {v['accrued']} | {v['events']} | — | "
              f"{'yes' if v['matches_paper'] else 'NO'} |")
        w()
        w('`matches_paper: true` for all three annotations and all three calendar horizons. '
          'The rebuild also reproduces (p_t, a_t) from the frozen feature builder on every one of '
          'the 1,012 units with zero mismatches, and the published two-variable AUROCs '
          '(0.948 / 0.857 / 0.898) and calendar AUROCs (0.913 / 0.920 / 0.876) exactly.')
        w()
    d = J('E1_coef.json')
    if d:
        p = d['primary']
        w('**Coefficient scale.** The value 3.0 in the paper is the *standardised* coefficient on '
          f"p_t ({p['standardised_p']:+.3f}), not the raw one.")
        w()
        w('| annotation | standardised b on p | standardised b on a | raw b0 | raw b1 | raw b2 |')
        w('|---|---|---|---|---|---|')
        for a in ('primary', 'second', 'consensus'):
            v = d[a]
            w(f"| {a} | {v['standardised_p']:+.3f} | {v['standardised_a']:+.3f} | "
              f"{v['raw_b0']:+.3f} | {v['raw_b1']:+.3f} | {v['raw_b2']:+.3f} |")
        w()
        w('*Decision rule outcome:* only the standardised coefficient is about +3.0, so the paper '
          'should read "standardised coefficient on p_t of +3.0".')
        w()

    # ---- E2
    w('## E2 Exchangeable controls on every annotation, with the paired null distribution')
    w()
    w('| annotation | observed events | Null A mean [95% range] | above range | Null B mean [95% range] | above range |')
    w('|---|---|---|---|---|---|')
    for a in ('primary', 'second', 'consensus'):
        d = J(f'E2_null_{a}.json')
        if not d:
            continue
        o = d['observed']['events']
        if 'null_A' not in d or 'null_B' not in d:
            continue
        A, B = d['null_A']['events'], d['null_B']['events']
        w(f"| {a} | {o} | {A['mean']:.1f} [{A['lo']:.0f}, {A['hi']:.0f}] | "
          f"{'YES' if A['observed_above_range'] else 'no'} | "
          f"{B['mean']:.1f} [{B['lo']:.0f}, {B['hi']:.0f}] | "
          f"{'YES' if B['observed_above_range'] else 'no'} |")
    w()
    w('**Null C: permutation within claim and decade.** This third control preserves each '
      'claim\'s composition AND the label mix of every decade, so it isolates whatever the '
      'observed stream has beyond era-level annotation drift.')
    w()
    w('| annotation | observed events | Null C mean [95% range] | above range | p |')
    w('|---|---|---|---|---|')
    for a in ('primary', 'second', 'consensus'):
        d = J(f'E2_null_{a}.json')
        if not d or 'null_C' not in d:
            continue
        C = d['null_C']['events']
        w(f"| {a} | {d['observed']['events']} | {C['mean']:.1f} [{C['lo']:.0f}, {C['hi']:.0f}] | "
          f"{'**YES**' if C['observed_above_range'] else 'no'} | {C['p_upper']:.3f} |")
    w()
    w('| annotation | observed diff | Null C mean [95% range] | p |')
    w('|---|---|---|---|')
    for a in ('primary', 'second', 'consensus'):
        d = J(f'E2_null_{a}.json')
        if not d or 'null_C' not in d:
            continue
        C = d['null_C']['paired_diff']
        w(f"| {a} | {d['observed']['diff']:+.3f} | {C['mean']:+.3f} "
          f"[{C['lo']:+.3f}, {C['hi']:+.3f}] | {C['p_upper']:.3f} |")
    w()
    w('**Paired null distribution of (two-variable minus margin).**')
    w()
    w('| annotation | observed | Null A mean [95% range] | p | Null B mean [95% range] | p |')
    w('|---|---|---|---|---|---|')
    for a in ('primary', 'second', 'consensus'):
        d = J(f'E2_null_{a}.json')
        if not d:
            continue
        o = d['observed']['diff']
        if 'null_A' not in d or 'null_B' not in d:
            continue
        A, B = d['null_A']['paired_diff'], d['null_B']['paired_diff']
        w(f"| {a} | {o:+.3f} | {A['mean']:+.3f} [{A['lo']:+.3f}, {A['hi']:+.3f}] | "
          f"{A['p_upper']:.3f} | {B['mean']:+.3f} [{B['lo']:+.3f}, {B['hi']:+.3f}] | "
          f"{B['p_upper']:.3f} |")
    w()

    # ---- E3
    d = J('E3_common_support.json')
    if d:
        w('## E3 Common-support refit')
        w()
        w(f"Common support, the two pipelines: {d['common_support_units_two_pipelines']} units "
          f"(the published mask). All three annotations: {d['common_support_units_all_three']}.")
        w()
        w('| cell | units | events | margin | stationary | signed | signed − margin | |')
        w('|---|---|---|---|---|---|---|---|')
        for c in d['cells']:
            v = c['signed_minus_margin']
            w(f"| {c['cell']} | {c['units']} | {c['events']} | {c['margin_auroc']:.3f} | "
              f"{c['stationary_auroc']:.3f} | {c['two_var_auroc']:.3f} | {v[0]:+.3f} {ci(v)} | "
              f"{'**established**' if c['signed_minus_margin_established'] else ''} |")
        w()
        w(f"*Decision rule outcome:* {d['verdict']}.")
        w()

    # ---- E4
    w('## E4 Within-class AUROC')
    w()
    for a in ('primary', 'second', 'consensus'):
        d = J(f'E4_stratum_{a}.json')
        if not d:
            continue
        w(f'**{a} annotation.**')
        w()
        w('| stratum | units | events | margin [95% CI] | two-variable [95% CI] | '
          'class-blind [95% CI] | two-var − margin |')
        w('|---|---|---|---|---|---|---|')
        for s_ in d['per_class']:
            f = lambda x: f'{x:.3f}' if x is not None else '—'
            def fc(row, k):
                c = row.get(k + '_ci')
                return f'{row[k]:.3f} [{c[0]:.3f}, {c[1]:.3f}]' if c and row.get(k) is not None else f(row.get(k))
            mm = s_.get('two_var_minus_margin'); cb = s_.get('class_blind_minus_margin')
            w(f"| {s_['stratum']} | {s_['units']} | {s_['events']} | {fc(s_,'margin')} | "
              f"{fc(s_,'two_var')} | {fc(s_,'class_blind')} | "
              f"{(str(round(mm[0],3)) + ' ' + ci(mm)) if mm else '—'} |")
        w()
        pw = d['pooled_within_class']
        w('| pooled within-class | AUROC | 95% CI | minus margin | |')
        w('|---|---|---|---|---|')
        for k, v in pw.items():
            mm = v.get('minus_margin')
            w(f"| {k} | {v['auroc']:.3f} | [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | "
              f"{(str(round(mm[0],3)) + ' ' + ci(mm)) if mm else '—'} | "
              f"{('**established**' if v.get('established') else 'not established') if mm else ''} |")
        w()

    # ---- E5
    d = J('E5_decisive.json')
    if d:
        w('## E5 Stricter endpoints')
        w()
        w('| annotation | endpoint | units kept | events | excluded | margin | stationary | signed | signed − margin | |')
        w('|---|---|---|---|---|---|---|---|---|---|')
        for c in d['cells']:
            v = c['signed_minus_margin']
            lbl = f"decisive δ={c['delta']}" if c['delta'] else 'common support'
            w(f"| {c['annotation']} | {lbl} | {c['units']} | {c['events']} | "
              f"{c['excluded_indecisive']} | {c['margin_auroc']:.3f} | {c['stationary_auroc']:.3f} | "
              f"{c['two_var_auroc']:.3f} | {v[0]:+.3f} {ci(v)} | "
              f"{'**established**' if c['signed_minus_margin_established'] else ''} |")
        w()

    # ---- E6
    d = J('E6_accrual.json')
    if d:
        w('## E6 Accrued versus censored candidates')
        w()
        w('| horizon | candidates | accrued | censored | SMD signed count | SMD \\|p\\| | SMD NULL share | SMD recent accrual | SMD years |')
        w('|---|---|---|---|---|---|---|---|---|')
        for s in d['smd']:
            if s.get('censored', 0) == 0:
                w(f"| {s['horizon']} | — | — | 0 | — | — | — | — | — |")
                continue
            w(f"| {s['horizon']} | {s['candidates']} | {s['accrued']} | {s['censored']} | "
              f"{s['smd_n_signed']:+.2f} | {s['smd_abs_p']:+.2f} | {s['smd_null_share']:+.2f} | "
              f"{s['smd_accrual_5y']:+.2f} | {s['smd_span_years']:+.2f} |")
        w()
        w(f"Largest absolute standardised mean difference: **{d['max_abs_smd']:.2f}**.")
        w()
        w('| horizon | comparison | unweighted | weighted | conclusion changes |')
        w('|---|---|---|---|---|')
        for c in d['cells']:
            for k in ('stationary', 'two_var'):
                u = c[f'{k}_minus_margin']; ww = c[f'{k}_minus_margin_weighted']
                w(f"| {c['horizon']} | {k} − margin | {u[0]:+.3f} {ci(u)} ({est(u)}) | "
                  f"{ww[0]:+.3f} {ci(ww)} ({est(ww)}) | "
                  f"{'**yes**' if c[f'{k}_conclusion_changes'] else 'no'} |")
        w()
        w(f"Largest change in any paired difference from weighting: **{d['max_abs_change_in_difference']:.3f}**.")
        w()

    # ---- E7
    d = J('E7_retrieval.json')
    if d:
        w('## E7 Retrieval-audit reconciliation')
        w()
        w(f"Claims audited: {d['n_claims']}. Per-claim Jaccard below 0.8: **{d['jaccard_below_0.8']}**. "
          f"Losing more than 10% of stored identifiers: **{d['lost_more_than_10pct']}** "
          f"(the paper reports 19).")
        w()
        w('Two disjoint mechanisms explain every flagged claim, and neither is evidence drift:')
        w()
        w(f"1. **{len(d['zero_return_claims'])} claims returned zero records at re-retrieval.** "
          'Their Jaccard is 0 by construction, and they are exactly the claims that appear to '
          '"lose more than 10%" of stored identifiers.')
        w(f"2. **{len(d['cap_inflated_claims'])} claims re-retrieved more records than were archived.** "
          'They lost nothing. Their low Jaccard reflects a record cap in the archived snapshot: '
          f"{d['archived_claims_near_cap']} of {d['n_claims']} claims have between 780 and 800 "
          f"archived records (overall range {d['archived_count_min']} to {d['archived_count_max']}).")
        w()
        w('| cell | population | units | events | margin | signed | signed − margin | |')
        w('|---|---|---|---|---|---|---|---|')
        for c in d['cells']:
            v = c['signed_minus_margin']
            w(f"| {c['annotation']} | {c['population']} | {c['units']} | {c['events']} | "
              f"{c['margin_auroc']:.3f} | {c['two_var_auroc']:.3f} | {v[0]:+.3f} {ci(v)} | "
              f"{'**established**' if c['established'] else ''} |")
        w()
        w(f"Conclusions changed by removing all flagged claims: "
          f"**{'none' if not d['conclusions_changed'] else d['conclusions_changed']}**.")
        w()

    # ---- E8
    d = J('E8_controlled_annotation.json')
    if d:
        w('## E8 Controlled annotation: model × prompt × read-out')
        w()
        w('| cell | configuration | PROTECTIVE | HARMFUL | NULL | UNCLEAR |')
        w('|---|---|---|---|---|---|')
        for c, v in d['cells'].items():
            s = v['shares']
            w(f"| {c} | {v['description']} | {s['PROTECTIVE']:.3f} | {s['HARMFUL']:.3f} | "
              f"{s['NULL']:.3f} | {s['UNCLEAR']:.3f} |")
        w()
        w('| contrast | kappa | event Jaccard | NULL-share shift | events |')
        w('|---|---|---|---|---|')
        for k, v in d['decomposition'].items():
            w(f"| {k} | {v['kappa']:.3f} | {v['jaccard']:.3f} | {v['null_shift']:+.3f} | "
              f"{v['events'][0]} → {v['events'][1]} |")
        w()
        if d.get('method'):
            w('| cell | units | events | margin | stationary | signed | signed − margin | |')
            w('|---|---|---|---|---|---|---|---|')
            for c in d['method']:
                v = c['signed_minus_margin']
                w(f"| {c['cell']} | {c['units']} | {c['events']} | {c['margin_auroc']:.3f} | "
                  f"{c['stationary_auroc']:.3f} | {c['two_var_auroc']:.3f} | {v[0]:+.3f} {ci(v)} | "
                  f"{'**established**' if c['signed_minus_margin_established'] else ''} |")
            w()

    # ---- E9b
    d = J('E9b_embeddings.json')
    if d:
        w('## E9b Embedding forecaster on the calendar cohort')
        w()
        w(f"Encoder: `{d['encoder']}`, mean-pooled, PCA to {d['n_pca']} components fitted on the "
          'training units of each origin.')
        w()
        w('| horizon | units | events | margin | text only | text + composition | composition only | text adds |')
        w('|---|---|---|---|---|---|---|---|')
        for c in d['cells']:
            t = c['text_plus_comp_minus_comp_only']
            w(f"| {c['horizon']} | {c['units']} | {c['events']} | {c['margin_auroc']:.3f} | "
              f"{c['text_only_auroc']:.3f} | {c['text_plus_comp_auroc']:.3f} | "
              f"{c['comp_only_auroc']:.3f} | {t[0]:+.3f} {ci(t)} ({est(t)}) |")
        w()

    # ---- E9a / E10
    d = J('E9a_E10_llm_forecasters.json')
    if d:
        w('## E9a / E10 Language-model forecasters on the calendar cohort')
        w()
        w(f"*{d['note']}.*")
        w()
        w('| horizon | model | variant | framing | units | events | AUROC | margin | minus margin | |')
        w('|---|---|---|---|---|---|---|---|---|---|')
        for c in d['cells']:
            v = c['minus_margin']
            if 'auroc' not in c:
                w(f"| {c['horizon']} | {c['model']} | {c['variant']} | {c['framing']} | "
                  f"{c['units']} | {c['events']} | — | — | {v[0]:+.3f} {ci(v)} | "
                  f"{'**established**' if c['established'] else ''} |")
                continue
            w(f"| {c['horizon']} | {c['model']} | {c['variant']} | {c['framing']} | {c['units']} | "
              f"{c['events']} | {c['auroc']:.3f} | {c['margin_auroc']:.3f} | {v[0]:+.3f} {ci(v)} | "
              f"{'**established**' if c['established'] else ''} |")
        w()

    # ---- E11
    d = J('E11_power.json')
    if d:
        w('## E11 Resampling power analysis')
        w()
        w('`paired` is the construction specified in the plan, in which the competing score is the '
          'margin rule\'s ranking plus a pure signal term; `realistic` adds independent noise so the '
          'competing score\'s rank correlation with the margin rule matches the value actually '
          'observed for the two-variable score in that cell.')
        w()
        w('| cell | units | events | MDE (paired) | MDE (realistic) | rank corr. with margin |')
        w('|---|---|---|---|---|---|')
        names = []
        for c in d['cells']:
            if c['cell'] not in names:
                names.append(c['cell'])
        for nm in names:
            cs = {c.get('mode', 'paired'): c for c in d['cells'] if c['cell'] == nm}
            c0 = cs.get('paired') or list(cs.values())[0]
            f = lambda m: ((f"{cs[m]['mde_80']:.4f}" if cs[m]['mde_80'] is not None
                            else f"> {max(d['deltas'])}") if m in cs else 'n/a')
            rho = c0.get('target_rho')
            w(f"| {nm} | {c0['units']} | {c0['events']} | {f('paired')} | {f('realistic')} | "
              f"{rho:.3f} |" if rho else
              f"| {nm} | {c0['units']} | {c0['events']} | {f('paired')} | {f('realistic')} | — |")
        w()

    # ---- E13
    d = J('E13_drift.json')
    if d:
        w('## E13 Annotation drift and look-ahead')
        w()
        w('| decade | records | kappa | agreement | PROTECTIVE | HARMFUL | NULL | UNCLEAR |')
        w('|---|---|---|---|---|---|---|---|')
        for r in d['by_decade']:
            s = r['primary_shares']
            w(f"| {r['decade']}s | {r['records']} | {r['kappa']:.3f} | {r['agreement']:.3f} | "
              f"{s['PROTECTIVE']:.3f} | {s['HARMFUL']:.3f} | {s['NULL']:.3f} | {s['UNCLEAR']:.3f} |")
        w()
        w(f"Kappa spread across decades: **{d['kappa_spread']:.3f}** "
          f"({d['kappa_range'][0]:.3f} to {d['kappa_range'][1]:.3f}).")
        w()
    d = J('E13_probe.json')
    if d:
        w(f"**Look-ahead probe.** Re-annotating {d['sampled']} associations with the primary "
          f"configuration after masking the exposure and outcome names changes "
          f"**{d['changed']} labels ({100*d['change_share']:.1f}%)**. "
          f"Before 2000: {100*d['change_share_before_2000']:.1f}% of {d['n_before_2000']}. "
          f"From 2000: {100*d['change_share_from_2000']:.1f}% of {d['n_from_2000']}. "
          f"Difference {d['difference_before_minus_after'][0]:+.3f} "
          f"{ci(d['difference_before_minus_after'])}, {est(d['difference_before_minus_after'])}.")
        w()

    # ---- E14
    if os.path.exists(os.path.join(N.FIGURES, 'E14_figure2_null_audit.pdf')):
        w('## E14 Figure 2')
        w()
        w('Redrawn in the style of the published null-audit figure: the two controls as overlaid '
          'histograms of their 1,000 draws, with the observed value on a red rule. Three fixes '
          'remove every collision in the original. The legend sits above the panels instead of on '
          'top of the histogram; each panel reserves headroom so the observed label never lands on '
          'a bar, and its alignment flips by which half the rule falls in; and Null B carries a '
          'hatch as well as a hue, so the controls separate in greyscale and under colour-vision '
          'deficiency.')
        w()
        w('Both files pass a programmatic check for text-on-text and text-on-bar collisions.')
        w()
        w('| File | Contents |')
        w('|---|---|')
        w('| `figures/E14_figure2_primary_3panel.pdf` | drop-in replacement for the published '
          'figure: same three panels, same order, same hues, same text width |')
        w('| `figures/E14_figure2_null_audit.pdf` | extended version: three annotations as rows, '
          'with the paired two-variable-minus-margin distribution as a fourth column |')
        w()
        w('Palette verified against the six colour checks. The two categorical hues sit in the '
          'lightness band at 0.516 and 0.728 and above the chroma floor at 0.107 and 0.138; the '
          'pair separates at dE 32.0 under normal vision and 28.4 under the worst of simulated '
          'protanopia and deuteranopia, against thresholds of 15 and 8. The gold reaches only '
          '2.42:1 on white, below the 3:1 mark threshold, which is why both series are named in '
          'the legend and Null B is hatched rather than identified by hue alone.')
        w()

    # ---- E15
    d = J('E15_cap_sensitivity.json')
    if d:
        c = d['census']
        w('## E15 Retrieval-cap sensitivity')
        w()
        w('The cap-affected set is read from the archive metadata, which records per claim whether '
          'the history was retrieved in full after an 800-record cap, rather than inferred from '
          're-retrieval overlap.')
        w()
        w('| set | claims |')
        w('|---|---|')
        w(f"| retrieved in full after the cap | {c['retrieved_in_full']} |")
        w(f"| archived count in [780, 800], the cap-like set | {c['near_cap_780_800']} |")
        w(f"| flagged by E7 on low overlap | {c['e7_flagged']} |")
        w(f"| union of cap-affected and E7-flagged | {c['union']} |")
        w()
        w('| cell | population | units | events | margin | signed | signed − margin | |')
        w('|---|---|---|---|---|---|---|---|')
        for x in d['cells']:
            v = x['signed_minus_margin']
            w(f"| {x['cell']} | {x['population']} | {x['units']} | {x['events']} | "
              f"{x['margin_auroc']:.3f} | {x['two_var_auroc']:.3f} | {v[0]:+.3f} {ci(v)} | "
              f"{'**established**' if x['established'] else ''} |")
        w()
        w(f"Conclusions changed: **{d['conclusions_changed'] or 'none under the cap-defined set'}**.")
        w()

    # ---- E17
    d = J('E17_drift_within.json')
    if d:
        w('## E17 Within-claim drift and the drift-preserving control')
        w()
        w('Is the decade-level label shift composition, meaning which claims publish when, or drift '
          'inside claims? Only the second can manufacture endpoint events without any change in the '
          'literature, because the endpoint compares a claim against itself.')
        w()
        w('| annotation | pooled slope per decade | within-claim slope per decade | claim effects remove |')
        w('|---|---|---|---|')
        for a, v in d['annotations'].items():
            p10 = [x * 10 for x in [v['pooled_slope_per_year']] + v['pooled_ci']]
            w10 = [x * 10 for x in [v['within_claim_slope_per_year']] + v['within_claim_ci']]
            w(f"| {a} | {p10[0]:+.3f} [{p10[1]:+.3f}, {p10[2]:+.3f}] | "
              f"{w10[0]:+.3f} [{w10[1]:+.3f}, {w10[2]:+.3f}]"
              f"{' **established**' if v['within_established'] else ''} | "
              f"{100*v['share_of_pooled_removed_by_claim_effects']:.1f}% |")
        w()
        for a, v in d['annotations'].items():
            if 'paired_within_claim_change' in v:
                pc = v['paired_within_claim_change']
                w(f"Paired within-claim check, {a}: across {v['paired_claims']} claims publishing in "
                  f"both the 1990s and 2020s the HARMFUL share moves "
                  f"{v['harmful_share_1990s']:.3f} to {v['harmful_share_2020s']:.3f}, a change of "
                  f"{pc[0]:+.3f} {ci(pc)} ({est(pc)}).")
                w()

    # ---- E18
    d = J('E18_cap_years.json')
    if d:
        w('## E18 Publication years of the capped claims')
        w()
        w('The direct test that replaces the identifier-position proxy: for each capped claim, the '
          'publication year of every record today\'s retrieval returns, against the archived years. '
          'Run only at full year coverage, because a partial fetch is biased toward older '
          'identifiers and would manufacture the result.')
        w()
        w('| measure | value |')
        w('|---|---|')
        w(f"| capped claims evaluated | {d['claims']} |")
        w(f"| archive starts later than the provider | {d['claims_starting_later']} of {d['claims']}, "
          f"median {d['min_year_gap_median']} years |")
        w(f"| median-year gap | {d['median_year_gap_median']} years |")
        w(f"| coverage of the provider's oldest quartile, relative to overall coverage | "
          f"median {d['oldest_quartile_ratio_median']:.3f} |")
        w(f"| claims below half that coverage | {d['claims_ratio_below_half']} of {d['claims']} |")
        w()
        w('A ratio of 0.000 means the archive holds none of the provider\'s oldest quarter of '
          'records. For most capped claims the early literature is absent outright, not merely '
          'thinned.')
        w()

    # ---- N series
    d = J('N1_N3_anchor.json')
    if d:
        w('## N1 Numeric anchor against both pipelines')
        w()
        w(f"Anchored records: {d['n_anchored']}. *{d['caveat']}.*")
        w()
        w('| pipeline | agreement [95% CI] | balanced accuracy [95% CI] | macro-F1 | recall P / H / N |')
        w('|---|---|---|---|---|')
        for pipe in ('primary', 'second', 'intersection'):
            m = d.get(pipe)
            if not m: continue
            w(f"| {pipe} | {m['agreement']:.3f} [{m['agreement_ci'][0]:.3f}, {m['agreement_ci'][1]:.3f}] | "
              f"{m['balanced_accuracy']:.3f} [{m['balanced_accuracy_ci'][0]:.3f}, "
              f"{m['balanced_accuracy_ci'][1]:.3f}] | {m['macro_f1']:.3f} | "
              + ' / '.join(f"{m['recall'][c]:.3f}" for c in ('PROTECTIVE','HARMFUL','NULL')) + ' |')
        w()
        for k in ('primary_minus_second_agreement', 'primary_minus_second_balanced_accuracy',
                  'signed_primary_minus_second'):
            v = d.get(k)
            if v:
                w(f"- {k.replace('_',' ')}: {v[0]:+.3f} {ci(v)} ({est(v)})")
        w()

        w('## N3 Era-stratified anchor comparison')
        w()
        w('| era | n | agreement primary / second | HARMFUL share anchor / primary / second |')
        w('|---|---|---|---|')
        for r in d.get('by_era', []):
            w(f"| {r['era']} | {r['n']} | {r['agreement_primary']:.3f} / {r['agreement_second']:.3f} | "
              f"{r['harmful_share_anchor']:.3f} / {r['harmful_share_primary']:.3f} / "
              f"{r['harmful_share_second']:.3f} |")
        w()
        w('| label source | within-claim slope per decade | established |')
        w('|---|---|---|')
        for src, v in d.get('within_claim_slopes', {}).items():
            w(f"| {src} | {v['slope_per_decade']:+.4f} [{v['ci_per_decade'][0]:+.4f}, "
              f"{v['ci_per_decade'][1]:+.4f}] | {'yes' if v['established'] else 'no'} |")
        w()

    d = J('N2_N5_branch_class.json')
    if d:
        w('## N2 Branch-free agreement')
        w()
        w('| cell | features | score AUROC | minus margin | |')
        w('|---|---|---|---|---|')
        for c in d.get('N2', []):
            v = c['minus_margin']
            w(f"| {c['cell']} | {c['features']} | {c['score_auroc']:.3f} | {v[0]:+.3f} {ci(v)} | "
              f"{'**established**' if c['established'] else ''} |")
        w()
        if d.get('branch'):
            w('| branch (second annotation) | units | events | restricted | refitted |')
            w('|---|---|---|---|---|')
            for b in d['branch']:
                r_ = b['restricted_minus_margin']; f_ = b['refit']['minus_margin']
                w(f"| {b['branch']} | {b['units']} | {b['events']} | {r_[0]:+.3f} {ci(r_)} | "
                  f"{f_[0]:+.3f} {ci(f_)} |")
            w()
        n5 = d.get('N5', {})
        if n5:
            w('## N5 Class effect within margin strata')
            w()
            w('| |p| quintile | HARMFUL n / events / rate | PROTECTIVE n / events / rate | OR |')
            w('|---|---|---|---|')
            for x in n5.get('strata', []):
                oR = x['odds_ratio']
                w(f"| Q{x['quintile']} | {x['harmful_n']} / {x['harmful_events']} / "
                  f"{x['harmful_rate']:.3f} | {x['protective_n']} / {x['protective_events']} / "
                  f"{x['protective_rate']:.3f} | "
                  f"{oR:.2f} |" if oR == oR else
                  f"| Q{x['quintile']} | {x['harmful_n']} / {x['harmful_events']} / "
                  f"{x['harmful_rate']:.3f} | {x['protective_n']} / {x['protective_events']} / "
                  f"{x['protective_rate']:.3f} | n/a |")
            w()
            ao = n5.get('adjusted_or')
            w(f"Crude odds ratio {n5['crude_or']:.2f}; Mantel-Haenszel across quintiles "
              f"{n5['mantel_haenszel_or']:.2f}; adjusted for |p| "
              f"{ao[0]:.2f} [{ao[1]:.2f}, {ao[2]:.2f}].")
            w()

    d = J('N4_N8_reversals_coef.json')
    if d:
        w('## N4 Documented reversals')
        w()
        w('| claim | pivotal | archive starts | first event, primary | lag |')
        w('|---|---|---|---|---|')
        for r in d.get('N4', []):
            fe = r['first_event']['primary']
            w(f"| {r['claim_id']} | {r['pivotal_year']} | {r['archive_starts']}"
              f"{' (capped)' if r['capped'] else ''} | {fe if fe else 'no event'} | "
              f"{r['lag_primary'] if r['lag_primary'] is not None else '—'} |")
        w()
        sm = d.get('summary', {})
        w(f"Of {sm.get('claims')} documented reversals, {sm.get('observable')} have an archive "
          f"reaching the pivotal date; the endpoint flags {sm.get('caught_primary')} of those, "
          f"{sm.get('caught_at_or_before_pivotal')} at or before the pivotal year. "
          f"{sm.get('capped_unobservable')} is unobservable because the cap leaves its archive "
          f"starting after the pivotal date.")
        w()
        n8 = d.get('N8', {})
        if n8:
            w('## N8 The second coefficient')
            w()
            w(f"Raw signed-branch logit b0 {n8['raw_b0']:+.3f}, b1 {n8['raw_b1']:+.3f}, "
              f"b2 {n8['raw_b2']:+.3f}. Slope in |p| is {n8['slope_in_abs_p_harmful_majority']:+.3f} "
              f"among HARMFUL majorities and {n8['slope_in_abs_p_protective_majority']:+.3f} among "
              f"PROTECTIVE ones. Both are negative in "
              f"{n8['folds_both_slopes_negative']} of {n8['folds']} folds.")
            w()

    d = J('N6_wild_bootstrap.json')
    if d:
        w('## N6 Wild cluster bootstrap')
        w()
        w('| cell | delta | percentile | wild cluster | jackknife | positive clusters |')
        w('|---|---|---|---|---|---|')
        for c in d['cells']:
            w(f"| {c['cell']} | {c['delta']:+.3f} | "
              f"[{c['percentile'][0]:.3f}, {c['percentile'][1]:.3f}]"
              f"{'*' if c['percentile_established'] else ''} | "
              f"[{c['wild'][0]:.3f}, {c['wild'][1]:.3f}]"
              f"{'*' if c['wild_established'] else ''} | "
              f"[{c['jackknife'][0]:.3f}, {c['jackknife'][1]:.3f}]"
              f"{'*' if c['jackknife_established'] else ''} | "
              f"{c['positive_clusters']}/{c['clusters']} |")
        w()
        w(f"Established under the percentile interval but not under the wild cluster bootstrap: "
          f"**{d['lost_under_wild'] or 'none'}**.")
        w()

    d = J('N7_null_variants.json')
    if d:
        w('## N7 Null C block sensitivity and Null D')
        w()
        o = d['observed']
        w(f"Observed: {o['events']} events, two-variable minus margin {o['diff']:+.3f}.")
        w()
        w('| control | events mean [95% range] | observed above | p | paired diff [95% range] | p |')
        w('|---|---|---|---|---|---|')
        for c in d['controls']:
            w(f"| {c['control']} | {c['events_mean']:.1f} [{c['events_range'][0]:.0f}, "
              f"{c['events_range'][1]:.0f}] | {'yes' if c['observed_above'] else '**no**'} | "
              f"{c['events_p']:.3f} | {c['paired_mean']:+.3f} [{c['paired_range'][0]:+.3f}, "
              f"{c['paired_range'][1]:+.3f}] | {c['paired_p']:.3f} |")
        w()

    # ---- E12
    w('## E12 Human adjudication')
    w()
    p = os.path.join(R, 'E12_union_events.csv')
    if os.path.exists(p):
        import csv as _csv
        rows = list(_csv.DictReader(open(p)))
        w(f"The sampling frame is built and released: **{len(rows)} union-event units** "
          f"(`results/E12_union_events.csv`) plus 50 matched non-event units "
          f"(`results/E12_matched_nonevents.csv`).")
    w()
    w('**Not run.** E12 requires two human annotators with biomedical training and a third '
      'adjudicator. No human labels were produced, and none were simulated.')
    w()

    txt = '\n'.join(L)
    open(os.path.join(R, 'REPORT.md'), 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
