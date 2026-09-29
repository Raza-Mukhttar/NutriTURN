"""Issue 4: audit the semantic orientation of every claim with deterministic rules over the claim id,
the exposure term and the outcome term, then re-evaluate on the orientation-clean benchmark.

Categories (a claim can carry several; any category except 'plain risk-factor exposure' sets the
ambiguity flag):
  deficiency_exposure        the exposure is an absence or deficiency state (deficiency, malnutrition, intolerance)
  therapeutic_or_preventive  the claim treats the exposure as a therapy or preventive intervention
  restriction_or_inversion   the exposure is a restriction, omission, substitution or timing manipulation
  benefit_valence_outcome    the outcome is a desirable quantity (increase = benefit), so the PROTECTIVE /
                             HARMFUL labels invert their everyday meaning
  non_disease_outcome_term   the outcome term is a query artefact ('prevention', 'control', 'duration', ...)
The rules are the whole method: no claim is re-labelled by hand. Canonicalisation by sign flip cannot
change the change target, because 1[sgn(p_post) != sgn(p_pre)] is invariant to flipping both signs, and
resolving within-claim mixtures would require reading records, so no canonicalised benchmark is built;
the sensitivity benchmark is exclusion. Namespace: orientation_clean.
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
import os, sys, json, re, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astra_common as ac
nm = ac.nm
RULES = {
    'deficiency_exposure': (r'deficien|malnutrition|intolerance', 'id_or_exposure'),
    'therapeutic_or_preventive': (r'prevention|treatment|therapy|remission|symptom|response|newborn|introduction|iodized|oral_iron|replacement|'
                                  r'^probiotics_|^prebiotics_|^magnesium_migraine|^vitamin_c_common_cold|^zinc_common_cold|^zinc_childhood|'
                                  r'^dash_diet|^gluten_free|^multivitamin_|^hormone_replacement', 'id'),
    'restriction_or_inversion': (r'skipping|replacing|low_fat|energy_deficit|restricted|fasting|late_night|alternate_day|low_intake|free_diet', 'id'),
    'benefit_valence_outcome': (r'^(weight loss|cognitive function|remission|growth|diversity|fertility|bioavailability|absorption|response|control|'
                                r'prevention|neurodevelopment|metabolism|hemoglobin)$', 'outcome'),
    'non_disease_outcome_term': (r'^(duration|intake|risk|progression|prevention|control|response|absorption|bioavailability|diversity|metabolism|growth|eye|cold)$', 'outcome'),
}


def main():
    Q = json.load(open(nm.QUERIES)); rows, keys = ac.load_units(); P, y, g = ac.prospective(rows)
    hand = set(json.load(open(nm.HAND_ADDED))['hand_added_claim_ids'])
    per_claim_units = collections.Counter(r['claim_id'] for r in rows); per_claim_prosp = collections.Counter(gi for gi in g)
    per_claim_events = collections.Counter(gi for gi, yi in zip(g, y) if yi)
    audit, amb = [], []
    for cid in Q:
        exp, out = Q[cid]['exposure'].lower(), Q[cid]['outcome'].lower()
        cats, reasons = [], []
        for cat, (pat, where) in RULES.items():
            hay = {'id': cid, 'exposure': exp, 'outcome': out, 'id_or_exposure': cid + ' ' + exp}[where]
            m = re.search(pat, hay)
            if m: cats.append(cat); reasons.append(f'{cat}: "{m.group(0)}" in {where}')
        nrec = len(ac.stream(cid)[0])
        a = {'claim_id': cid, 'exposure': Q[cid]['exposure'], 'outcome': Q[cid]['outcome'], 'hand_added': int(cid in hand),
             'orientation_category': ';'.join(cats) if cats else 'plain_risk_factor_exposure', 'ambiguity_flag': int(bool(cats)),
             'reason': ' | '.join(reasons), 'records': nrec, 'units': per_claim_units[cid], 'prospective_units': per_claim_prosp.get(cid, 0), 'events': per_claim_events.get(cid, 0)}
        audit.append(a)
        if cats: amb.append(a)
    ac.save_csv(audit, os.path.join(ac.DATA, 'claim_semantic_orientation_audit.csv'))
    ac.save_csv(amb, os.path.join(ac.DATA, 'orientation_ambiguous_claims.csv'))
    cat_counts = collections.Counter(c for a in amb for c in a['orientation_category'].split(';'))
    CORE = {'deficiency_exposure', 'therapeutic_or_preventive', 'restriction_or_inversion', 'non_disease_outcome_term'}
    tiers = {'core (spec categories: deficiency, therapeutic/preventive, restriction/inversion, non-disease outcome term)':
             {a['claim_id'] for a in amb if set(a['orientation_category'].split(';')) & CORE},
             'strict (core plus benefit-valence outcomes)': {a['claim_id'] for a in amb}}
    print(f'flagged {len(tiers[list(tiers)[1]])} of {len(Q)} claims (core {len(tiers[list(tiers)[0]])}); categories {dict(cat_counts)}', flush=True)

    # ---- reference on the full benchmark
    s7_all, t7_all = ac.loco(nm.X_of(P, nm.COMPACT7), y, g); s2_all, t2_all = ac.loco(np.column_stack([[r['pooled_direction'] for r in P], [r['agreement'] for r in P]]), y, g)
    st_all = np.array([r['maturity_state'] for r in P])
    def rates_of(yy, gg, st):
        return {s: {'units': int((st == s).sum()), 'events': int(yy[st == s].sum()), 'rate': ac.rate_ci(yy[st == s], gg[st == s]) if (st == s).sum() else (None, None)} for s in ('stable', 'still_forming', 'unstable')}
    res = {'category_counts': dict(cat_counts), 'tiers': {}, 'full_reference': {'units': len(P), 'events': int(y.sum()), 'positive_claims': int(len(set(g[y == 1]))),
           'compact7': ac.perf(y, s7_all, g, t7_all), 'two_variable': ac.perf(y, s2_all, g, t2_all), 'compact7_claim_level': ac.claim_level(y, s7_all, g), 'state_rates': rates_of(y, g, st_all)},
           'canonicalised_benchmark': 'not built: a sign flip of a claim leaves 1[sgn(p_post) != sgn(p_pre)] unchanged, and resolving within-claim orientation mixtures would need record-level reading, which is not a reproducible rule'}
    tab = [{'benchmark': 'full benchmark', 'units': len(P), 'claims': int(len(set(g))), 'events': int(y.sum()), 'positive_claims': res['full_reference']['positive_claims'],
            'compact7_auroc': res['full_reference']['compact7']['auroc'], 'compact7_auroc_ci': res['full_reference']['compact7']['auroc_ci'], 'compact7_auprc': res['full_reference']['compact7']['auprc'],
            'two_var_auroc': res['full_reference']['two_variable']['auroc'], 'two_var_auroc_ci': res['full_reference']['two_variable']['auroc_ci'],
            'claim_level_compact7_auroc': res['full_reference']['compact7_claim_level']['auroc'], 'claim_level_compact7_auroc_ci': res['full_reference']['compact7_claim_level']['auroc_ci'],
            **{f'{s}_{k}': res['full_reference']['state_rates'][s][k if k != 'rate_ci' else 'rate'][1 if k == 'rate_ci' else 0] if k in ('rate', 'rate_ci') else res['full_reference']['state_rates'][s][k] for s in ('stable', 'still_forming', 'unstable') for k in ('units', 'rate', 'rate_ci')}}]
    for tname, flagged in tiers.items():
        keep = np.array([gi not in flagged for gi in g]); Pc = [r for r, k in zip(P, keep) if k]; yc, gc = y[keep], g[keep]
        X7 = nm.X_of(Pc, nm.COMPACT7); pt = np.array([r['pooled_direction'] for r in Pc]); ag = np.array([r['agreement'] for r in Pc])
        s7, t7 = ac.loco(X7, yc, gc); s2, t2 = ac.loco(np.column_stack([pt, ag]), yc, gc); st = np.array([r['maturity_state'] for r in Pc]); rates = rates_of(yc, gc, st)
        d = {'flagged_claims': sorted(flagged), 'n_flagged': len(flagged), 'units': len(Pc), 'claims': int(len(set(gc))), 'events': int(yc.sum()), 'positive_claims': int(len(set(gc[yc == 1]))),
             'prevalence': round(float(yc.mean()), 4), 'compact7_refit': ac.perf(yc, s7, gc, t7), 'two_variable_refit': ac.perf(yc, s2, gc, t2), 'one_minus_abs_p_t': nm.perf(yc, 1 - np.abs(pt), gc),
             'compact7_claim_level': ac.claim_level(yc, s7, gc), 'compact7_full_model_restricted': nm.perf(yc, s7_all[keep], gc), 'state_rates': rates,
             'stable_vs_still_forming_gap': round(rates['still_forming']['rate'][0] - rates['stable']['rate'][0], 4), 'excluded_units': {'prospective': int((~keep).sum()), 'events': int(y[~keep].sum())}}
        res['tiers'][tname] = d
        tab.append({'benchmark': 'orientation-clean, ' + tname.split(' (')[0], 'units': len(Pc), 'claims': d['claims'], 'events': d['events'], 'positive_claims': d['positive_claims'],
                    'compact7_auroc': d['compact7_refit']['auroc'], 'compact7_auroc_ci': d['compact7_refit']['auroc_ci'], 'compact7_auprc': d['compact7_refit']['auprc'],
                    'two_var_auroc': d['two_variable_refit']['auroc'], 'two_var_auroc_ci': d['two_variable_refit']['auroc_ci'],
                    'claim_level_compact7_auroc': d['compact7_claim_level']['auroc'], 'claim_level_compact7_auroc_ci': d['compact7_claim_level']['auroc_ci'],
                    **{f'{s}_{k}': rates[s]['rate'][1 if k == 'rate_ci' else 0] if k in ('rate', 'rate_ci') else rates[s][k] for s in ('stable', 'still_forming', 'unstable') for k in ('units', 'rate', 'rate_ci')}})
        if tname.startswith('strict'):
            ac.save_frame([{k: r[k] for k in r if k != 'pmids_pre'} for r in Pc], os.path.join(ac.DATA, 'orientation_clean_benchmark.parquet'))
        else:
            ac.save_frame([{k: r[k] for k in r if k != 'pmids_pre'} for r in Pc], os.path.join(ac.ns('data', 'orientation_clean'), 'orientation_clean_benchmark_core.parquet'))
        print(f"  {tname[:6]:6s} units {len(Pc)} events {int(yc.sum())} C7 {d['compact7_refit']['auroc']} {d['compact7_refit']['auroc_ci']}  2v {d['two_variable_refit']['auroc']}  stable {rates['stable']['rate'][0]} still {rates['still_forming']['rate'][0]}", flush=True)
    ac.save_json(res, os.path.join(ac.RESULTS, 'orientation_clean_results.json'))
    ac.save_csv(tab, os.path.join(ac.TABLES, 'orientation_clean_results.csv'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
