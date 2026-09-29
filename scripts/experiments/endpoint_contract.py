"""Correction plan E1.1 / Group A-B tests: executed endpoint contract, per-unit audit record, and probability regression tests.
Writes results/guide/endpoint_contract.json and results/guide/unit_audit_source.csv."""

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
import os, sys, csv, json, numpy as np
from fractions import Fraction
from math import comb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_common as pc
ac, nm = pc.ac, pc.nm
out = {'contract': {
    'pre_cutoff': 'records with year < t and a resolved (non-UNCLEAR) annotation; signed = non-NULL; the frozen builder (direction_block / astra_common.direction_feats) returns pooled_direction = raw signed mean (h-l)/(h+l) when h+l >= 8 and 0.0 otherwise (fallback also sets agreement etc. to 0)',
    'operational_pre_sign': 'sgn(pooled_direction) with sgn(0)=0: 0 for an exact tie with >= 8 signed records AND for fewer than 8 signed records (sparse fallback); the endpoint uses the builder output, not a recomputed raw sign',
    'future_side': 'post_pooled_direction = raw mean of the signed post-cutoff records (no 8-record gate); 0.0 when no signed post-cutoff record exists; future sign = sgn of that mean',
    'endpoint': 'R = 1[sgn(post_pooled_direction) != sgn(pooled_direction)] (nutrimature_features / astra_common.load_units y_sub); an empty signed future gives sign 0, so it is a change iff the operational pre sign is non-zero',
    'eligibility': 'future-eligible = at least 20 post-cutoff records of any label (not a signed-record support rule); no source unit has zero signed post-cutoff records',
    'analytic_baselines': 'bb_change_prob / dm_change_prob use the posterior Beta(h+alpha, l+alpha) from the actual counts and compare the future signed majority with the OPERATIONAL pre sign (s0 argument); the raw-majority indicator is kept only as the labelled diagnostic columns *_rawsign; count draws are rounded and floored at 1, so the m=0 branch never occurs in the count-predicted implementation; the Dirichlet-multinomial draw may produce a future window with no signed record, which is a change iff s0 != 0 (same rule as the endpoint)',
    'soft_labels_and_rebuilds': 'issue6.unit_stats rebuilds the pre side with direction_feats (same gate) and the post side as the raw signed mean; identical indicator',
    'monte_carlo_check': 'guide_checks posterior-predictive check draws theta ~ Beta(h+1,l+1), K ~ Binomial(m, theta) and compares with the operational pre sign'}}
# ---- regression tests (alpha = 1 exact fractions from the plan, and the implementation)
def sign(v): return (v > 0) - (v < 0)
def ref(h, l, m, s0):
    den = comb(m + h + l + 1, h + l + 1); num = sum(comb(k + h, h) * comb(m - k + l, l) for k in range(m + 1) if sign(2 * k - m) != s0); return Fraction(num, den)
T = {}
T['pi_raw(30,30;2)=32/63'] = (pc.bb_change_prob(30, 30, 2), float(Fraction(32, 63)), float(ref(30, 30, 2, 0)))
T['pi_raw(31,29;2)=475/651'] = (pc.bb_change_prob(31, 29, 2), float(Fraction(475, 651)), float(ref(31, 29, 2, 1)))
T['pi_raw(8,0;1)=1/10'] = (pc.bb_change_prob(8, 0, 1), 0.1); T['symmetry (0,8;1)'] = (pc.bb_change_prob(0, 8, 1), 0.1)
T['pi_raw(4,4;3)=1'] = (pc.bb_change_prob(4, 4, 3), 1.0); T['pi_raw(7,0;1)=1/9'] = (pc.bb_change_prob(7, 0, 1), 1 / 9)
T['pi_op(7,0;1 | s0=0)=1'] = (pc.bb_change_prob(7, 0, 1, s0=0), 1.0); T['pi_op(7,0;2 | s0=0)=1-P(K=1)'] = (pc.bb_change_prob(7, 0, 2, s0=0), float(ref(7, 0, 2, 0)))
T['limit (8,0): pi(8,0;100000) vs 1/512'] = (pc.bb_change_prob(8, 0, 100000), 1 / 512); T['limit (8,0): m=100001 (odd)'] = (pc.bb_change_prob(8, 0, 100001), 1 / 512)
T['tied even m = 1-P(K=m/2)'] = (pc.bb_change_prob(5, 5, 2000), float(1 - ref(5, 5, 2000, 0).__class__(1) + ref(5, 5, 2000, 0)) if False else float(ref(5, 5, 2000, 0))); T['tied even m approaches 1 (m=2000 > 0.99)'] = (float(pc.bb_change_prob(5, 5, 2000) > 0.99), 1.0); T['tied odd m = 1'] = (pc.bb_change_prob(5, 5, 2001), 1.0)
# normalisation and bounds
from scipy.special import gammaln, betaln
def pmf_sum(h, l, m):
    k = np.arange(m + 1); lp = gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1) + betaln(k + h + 1, m - k + l + 1) - betaln(h + 1, l + 1); return float(np.exp(lp).sum())
T['pmf sums to 1 (max abs dev over grid)'] = (max(abs(pmf_sum(h, l, m) - 1) for h in (0, 3, 40) for l in (0, 5, 60) for m in (1, 2, 50, 500)), 0.0)
mono = 0
for n in (8, 20, 60):
    for m in (1, 2, 5, 50):
        vals = [pc.bb_change_prob(h, n - h, m) for h in range(n // 2 + 1, n + 1)]; mono += sum(1 for a, b in zip(vals, vals[1:]) if b > a + 1e-12)
T['non-tied monotonicity violations'] = (mono, 0)
out['regression_tests'] = {k: {'implementation': v[0], 'expected': v[1], 'pass': abs(v[0] - v[1]) < (2e-6 if 'limit' in k or 'tied even' in k else 1e-9)} for k, v in T.items()}
out['all_tests_pass'] = all(v['pass'] for v in out['regression_tests'].values())
# ---- per-unit audit record for the source cohort (deployed pipeline): archived event == reproduced operational event
U = pc.source_units(); P, y = U['P'], U['y']; rows = []; mism = 0; sparse = 0; raw_mism = 0
S1 = {(r['claim_id'], str(int(float(r['cutoff_year'])))): r for r in csv.DictReader(open(os.path.join(pc.RES, 'issue1', 'source_scores.csv')))}
for i, r in enumerate(P):
    yrs, codes, _ = ac.stream(r['claim_id']); t = int(r['cutoff_year']); pre = codes[yrs < t]; post = codes[yrs >= t]
    h = int((pre == 1).sum()); l = int((pre == -1).sum()); n0 = int((pre == 0).sum()); nu = int((pre == 99).sum()); ph = int((post == 1).sum()); pl = int((post == -1).sum()); p0 = int((post == 0).sum()); pu = int((post == 99).sum())
    raw_mean = (h - l) / (h + l) if h + l else None; builder = ac.direction_feats(yrs[yrs < t], pre, t, 'A')['pooled_direction']; op_sign = pc.sgn0(builder); raw_sign = pc.sgn0(h - l) if h + l else None
    fut_raw = pc.sgn0((ph - pl) / (ph + pl)) if ph + pl else 0; rep = int(fut_raw != op_sign); raw_ev = int(fut_raw != (raw_sign if raw_sign is not None else 0))
    mism += int(rep != y[i]); sparse += int(h + l < 8); raw_mism += int(raw_ev != y[i]); s1 = S1[(r['claim_id'], str(t))]
    rows.append({'claim_id': r['claim_id'], 'cutoff_year': t, 'pipeline_id': 'llama_first_token', 'protocol_id': 'claim_held_out', 'horizon': 'all_post', 'pre_harmful': h, 'pre_protective': l, 'pre_null': n0, 'pre_unclear': nu, 'post_harmful': ph, 'post_protective': pl, 'post_null': p0, 'post_unclear': pu,
                 'raw_pre_mean': raw_mean, 'builder_pre_mean': builder, 'raw_pre_sign': raw_sign, 'operational_pre_sign': op_sign, 'raw_future_sign': fut_raw, 'operational_future_sign': fut_raw, 'pre_support_status': 'sparse_fallback' if h + l < 8 else ('exact_tie' if h == l else 'supported'),
                 'future_support_status': 'empty' if ph + pl == 0 else ('sparse' if ph + pl < 8 else 'supported'), 'archived_event': int(y[i]), 'reproduced_event': rep, 'raw_majority_event': raw_ev, 'old_stationary_score_rawsign': s1.get('bb_deployable_rawsign'), 'endpoint_aligned_stationary_score': s1['bb_deployable'], 'count_model_id': 'loco_lognormal_400draws', 'probability_implementation_id': 'pro_common.bb_change_prob(s0=operational)'})
out['audit_source'] = {'units': len(rows), 'archived_vs_reproduced_mismatches': mism, 'sparse_pre_units': sparse, 'raw_majority_event_vs_archived_mismatches': raw_mism, 'assertion_archived_equals_reproduced': mism == 0}
pc.save_csv(rows, 'guide/unit_audit_source.csv')
# change log for the E1 correction (old raw-sign scores vs aligned scores)
old = {(r['claim_id'], str(int(float(r['cutoff_year'])))): r for r in csv.DictReader(open(os.path.join(pc.RES, 'historical_rawsign', 'issue1', 'source_scores.csv')))}
d = [abs(float(S1[k]['bb_deployable']) - float(old[k]['bb_deployable'])) for k in S1]; do = [abs(float(S1[k]['bb_oracle']) - float(old[k]['bb_oracle'])) for k in S1]
out['change_log_E1'] = {'correction_id': 'E1 operational current sign in the analytic baselines', 'affected_units_source': int(sum(1 for k in S1 if int(S1[k]['operational_pre_sign']) != int(S1[k]['raw_pre_sign']))), 'units_with_changed_count_predicted_score': int(sum(1 for x in d if x > 1e-9)), 'max_abs_change_count_predicted': float(max(d)), 'max_abs_change_realised_count': float(max(do)),
                        'target_changed': False, 'dependent_runs': 'issue1, issue9 addendum, issue5, issue2, guide_recalibrated_order, issue4, issue6, guide_checks', 'historical_copy': 'results/historical_rawsign/'}
pc.save_json(out, 'guide/endpoint_contract.json'); print(json.dumps({k: out[k] for k in ('all_tests_pass', 'audit_source', 'change_log_E1')}, indent=1)); print({k: v['pass'] for k, v in out['regression_tests'].items()})
