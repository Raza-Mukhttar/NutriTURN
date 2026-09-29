"""Verification checks and limited optional computations requested by NutriMATURE_Revision_Guide.md (sections 5, 9.2, 9.4).
Writes results/guide/checks.json. Nothing here changes an existing result; it audits them."""

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
import os, sys, json, csv, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_common as pc
from scipy.special import betainc
ac, nm = pc.ac, pc.nm
out = {}
# ---- 1. exact formula: counterexample of the guide, sign symmetry, bounds, non-tied monotonicity
bb = pc.bb_change_prob
out['counterexample'] = {'pi(30,30;2)': bb(30, 30, 2), 'expected_32_over_63': 32 / 63, 'pi(31,29;2)': bb(31, 29, 2), 'expected_475_over_651': 475 / 651}
sym = max(abs(bb(h, l, m) - bb(l, h, m)) for h in range(0, 40, 3) for l in range(0, 40, 5) for m in (1, 2, 7, 20, 101)); out['sign_symmetry_max_abs_diff'] = float(sym)
vals = [bb(h, l, m) for h in range(0, 60, 4) for l in range(0, 60, 4) for m in (1, 2, 10, 50, 300)]; out['bounds'] = {'min': float(min(vals)), 'max': float(max(vals)), 'all_finite': bool(np.all(np.isfinite(vals)))}
viol = 0; tests = 0
for n in (8, 20, 60, 120):
    for m in (1, 2, 5, 20, 100, 400):
        prev = None
        for h in range(n // 2 + 1, n + 1):               # strictly non-tied, majority h, |p| increasing
            v = bb(h, n - h, m); tests += 1
            if prev is not None and v > prev + 1e-12: viol += 1
            prev = v
out['non_tied_monotonicity'] = {'tests': tests, 'violations': viol}
# tied histories: pi = 1 for odd m, 1 - P(K = m/2) for even m
tied = {}
for m in (1, 2, 3, 10, 11):
    v = bb(30, 30, m); pk = None
    if m % 2 == 0:
        from scipy.special import gammaln, betaln
        k = m // 2; pk = float(np.exp(gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1) + betaln(k + 31, m - k + 31) - betaln(31, 31)))
    tied[f'm={m}'] = {'pi': v, 'expected': 1.0 if m % 2 else 1 - pk}
out['tied_history_branch'] = tied
# limiting statement with majority/minority ordering
h, l = 40, 20; lim = float(betainc(h + 1, l + 1, 0.5)); out['limit_check'] = {'pi(40,20;m=20000)': bb(h, l, 20000), 'pi(40,20;m=20001)': bb(h, l, 20001), 'I_half(max+a, min+a) = P(theta<1/2 | h,l)': lim, 'pi(8,0;m=100000)': bb(8, 0, 100000), 'expected_1_over_512': 1 / 512,
                                                   'note': 'the limit for h>l is P(theta<1/2 | h,l) = I_{1/2}(h+alpha, l+alpha), i.e. I_{1/2}(max+alpha, min+alpha); the manuscript text of the previous version printed the arguments in the wrong order; no prediction uses the limit (finite sums only)'}
# tied history: pi = 1 (odd m) or 1 - P(K = m/2) (even m) -> tends to 1
out['tied_limit'] = {'pi(30,30;1001)': bb(30, 30, 1001), 'pi(30,30;1000)': bb(30, 30, 1000)}
# ---- 2. endpoint construction: exact tie vs no signed evidence
S1 = list(csv.DictReader(open(os.path.join(pc.RES, 'issue1', 'source_scores.csv'))))
out['endpoint'] = {'units': len(S1), 'pre_cutoff_exact_ties': sum(1 for r in S1 if r['n_plus'] == r['n_minus']), 'events_among_tied': sum(int(r['y']) for r in S1 if r['n_plus'] == r['n_minus']),
                   'units_with_zero_future_signed_records': sum(1 for r in S1 if int(r['m_signed']) == 0), 'policy': 'future-eligible units require >= 20 later records; sgn(mean of an empty signed future) would be 0; no such unit exists in the source cohort, so the m=0 branch is never exercised'}
# ---- 3. count-model conversion: integers, zero handling (documented from pro_common.draw_counts / bb_change_prob)
rng = np.random.default_rng(0); b, sd = pc.fit_count_model(np.log1p(np.array([r['n_plus'] for r in S1], float) + np.array([r['n_minus'] for r in S1], float)), np.log1p(np.zeros(len(S1))), np.log1p(np.array([r['m_signed'] for r in S1], float)))
d = pc.draw_counts(b, sd, np.log1p(50.0), np.log1p(2.0), rng); out['count_model_draws'] = {'dtype': str(d.dtype), 'min_draw': int(d.min()), 'policy': 'round(exp(mu + sd*z)) to an integer; values below 1 are raised to 1 inside bb_change_prob (max(1, m)); the exact tail is evaluated for each distinct drawn m and averaged with the draw multiplicities'}
# ---- 4. posterior-predictive Monte-Carlo (theta ~ Beta(h+1, l+1), K ~ Binomial(m, theta)) versus the analytic probability, realised m
B = 2000; err = []; mc_se = []
for r in S1:
    h, l, m = int(r['n_plus']), int(r['n_minus']), int(r['m_signed'])
    if m == 0: continue
    th = rng.beta(h + 1, l + 1, size=B); K = rng.binomial(m, th); s0 = int(r['operational_pre_sign']); fr = np.mean(np.sign(2 * K - m) != s0); a = float(r['bb_oracle'])   # operational current sign, as scored
    err.append(abs(fr - a)); mc_se.append(np.sqrt(fr * (1 - fr) / B))
out['posterior_predictive_simulation'] = {'draws_per_unit': B, 'units': len(err), 'mean_abs_error': float(np.mean(err)), 'max_abs_error': float(np.max(err)), 'mean_mc_se': float(np.mean(mc_se)), 'share_within_2se': float(np.mean(np.array(err) <= 2 * np.maximum(np.array(mc_se), 1e-3)))}
# ---- 5. NULL-dominant branch counts and event retention by branch under each pipeline (from the saved re-annotation)
rows = []
for f in sorted(glob.glob(os.path.join(pc.RES, 'reextraction', 'qwen25_7b_shard*.jsonl'))):
    for line in open(f): rows.append(json.loads(line))
CODE = {'PROTECTIVE': -1, 'HARMFUL': 1, 'NULL': 0, 'UNCLEAR': 99}; qmap = {(r['claim_id'], r['pmid']): CODE[r['qwen_label_full']] for r in rows}
U = pc.source_units(); P, y, g = U['P'], U['y'], U['g']
def branch_of(codes):
    res = codes[codes != 99]; n0 = int((res == 0).sum()); ns = int((res != 0).sum())
    if ns < 8: return 'fallback'
    return 'null_dominant' if n0 / max(len(res), 1) >= 0.5 else 'signed'
br = {'llama': [], 'qwen': []}; yq = np.zeros(len(P), int); sparse_q = np.zeros(len(P), int); tie_q = np.zeros(len(P), int)
for i, r in enumerate(P):
    yrs, codes, pmids = ac.stream(r['claim_id']); t = int(r['cutoff_year']); cq = np.array([qmap.get((r['claim_id'], pm), 99) for pm in pmids])
    br['llama'].append(branch_of(codes[yrs < t])); br['qwen'].append(branch_of(cq[yrs < t]))
    pre = cq[yrs < t]; ps = pre[(pre != 99) & (pre != 0)]; post = cq[yrs >= t]; qs = post[(post != 99) & (post != 0)]
    pq = ac.direction_feats(yrs[yrs < t], pre, t, 'A')['pooled_direction']          # frozen builder: pooled direction is 0 when fewer than 8 signed records
    yq[i] = int(pc.sgn0(float(qs.mean()) if len(qs) else 0.0) != pc.sgn0(pq)); sparse_q[i] = int(len(ps) < 8); tie_q[i] = int(len(ps) >= 8 and (ps == 1).sum() == (ps == -1).sum())
import collections
out['branch_counts'] = {k: dict(collections.Counter(v)) for k, v in br.items()}
out['event_retention_by_branch'] = {}
for k in ('llama', 'qwen'):
    for bname in ('signed', 'null_dominant', 'fallback'):
        idx = [i for i in range(len(P)) if br[k][i] == bname]
        out['event_retention_by_branch'][f'{k}:{bname}'] = {'units': len(idx), 'llama_events': int(y[idx].sum()) if idx else 0, 'qwen_events': int(yq[idx].sum()) if idx else 0, 'both': int(((y[idx] == 1) & (yq[idx] == 1)).sum()) if idx else 0}
ct = {'both': int(((y == 1) & (yq == 1)).sum()), 'llama_only': int(((y == 1) & (yq == 0)).sum()), 'qwen_only': int(((y == 0) & (yq == 1)).sum()), 'neither': int(((y == 0) & (yq == 0)).sum())}; out['event_crosstab'] = ct
out['event_crosstab']['jaccard'] = ct['both'] / (ct['both'] + ct['llama_only'] + ct['qwen_only'])
sparse_l = np.array([int((U['npl'][i] + U['nmi'][i]) < 8) for i in range(len(P))]); tie_l = np.array([int((U['npl'][i] + U['nmi'][i]) >= 8 and U['npl'][i] == U['nmi'][i]) for i in range(len(P))])
out['tie_vs_sparse'] = {'llama': {'units_fewer_than_8_signed_pre': int(sparse_l.sum()), 'events_among_them': int(y[sparse_l == 1].sum()), 'exact_ties_ge8': int(tie_l.sum()), 'events_among_exact_ties': int(y[tie_l == 1].sum())},
                        'qwen': {'units_fewer_than_8_signed_pre': int(sparse_q.sum()), 'events_among_them': int(yq[sparse_q == 1].sum()), 'exact_ties_ge8': int(tie_q.sum()), 'events_among_exact_ties': int(yq[tie_q == 1].sum())},
                        'note': 'the frozen feature builder sets the pooled direction to 0 when fewer than 8 signed records precede the cutoff, so such units count as ties in the endpoint; events among them are floor-induced rather than exact ties'}
# Qwen first-token read-out event overlap (alternative read-out)
q1 = {(r['claim_id'], r['pmid']): CODE[r['qwen_label_first']] for r in rows}; y1 = np.zeros(len(P), int)
for i, r in enumerate(P):
    yrs, codes, pmids = ac.stream(r['claim_id']); t = int(r['cutoff_year']); c1 = np.array([q1.get((r['claim_id'], pm), 99) for pm in pmids])
    pre = c1[yrs < t]; post = c1[yrs >= t]; qs = post[(post != 99) & (post != 0)]
    y1[i] = int(pc.sgn0(float(qs.mean()) if len(qs) else 0.0) != pc.sgn0(ac.direction_feats(yrs[yrs < t], pre, t, 'A')['pooled_direction']))
out['qwen_first_token_readout'] = {'events': int(y1.sum()), 'jaccard_with_qwen_full': float(((y1 == 1) & (yq == 1)).sum() / max(((y1 == 1) | (yq == 1)).sum(), 1)), 'jaccard_with_llama': float(((y1 == 1) & (y == 1)).sum() / max(((y1 == 1) | (y == 1)).sum(), 1))}
# ---- 6. record-level consequences from the confusion matrix
E = json.load(open(os.path.join(pc.RES, 'issue6', 'extractor_agreement.json')))['record_level']['confusion_llama_rows_qwen_cols']
out['record_level_derived'] = {'signed_to_null': E['PROTECTIVE']['NULL'] + E['HARMFUL']['NULL'], 'direct_sign_switch': E['PROTECTIVE']['HARMFUL'] + E['HARMFUL']['PROTECTIVE'], 'total': sum(E[a][b] for a in E for b in E[a])}
# ---- 7. re-retrieval: unique PMID counting
snap = sorted(glob.glob(os.path.join(pc.RES, 'snapshot_*')))[-1]; new_ids = set(); stored_ids = set(); assoc_new = 0; assoc_stored = 0; Q = json.load(open(nm.QUERIES))
for line in open(os.path.join(snap, 'queries.jsonl')):
    d = json.loads(line); new_ids |= set(d['pmids']); assoc_new += len(d['pmids'])
for cid in Q:
    ids = [r['pmid'] for r in json.load(open(f'{ac.SNAP}/records/{cid}.json'))['records']]; stored_ids |= set(ids); assoc_stored += len(ids)
out['reretrieval_unique'] = {'stored_associations': assoc_stored, 'stored_unique_pmids': len(stored_ids), 'new_associations': assoc_new, 'new_unique_pmids': len(new_ids), 'stored_unique_recovered': len(stored_ids & new_ids), 'recovered_share_unique': len(stored_ids & new_ids) / len(stored_ids)}
comp = list(csv.DictReader(open(os.path.join(snap, 'comparison.csv')))); ib = sum(int(r['in_both']) for r in comp); st = sum(int(r['stored_ids']) for r in comp); out['reretrieval_unique']['association_recovered_share_exact'] = ib / st
# ---- 8. soft-label sampling unit (documented from issue6 code): one draw per (claim, record) per replicate, reused across every cutoff of the claim
out['soft_label_sampling_unit'] = 'one label per claim-record association per draw (probs[claim] indexed by record), reused by unit_stats for every cutoff of that claim; a PMID shared by two claims is drawn independently per claim'
# ---- 9. permutation script audit (documented from issue2 code)
out['order_randomisation_audit'] = {'unit': 'each claim-cutoff unit shuffles its own pre-cutoff prefix independently (rng.permutation over the prefix)', 'held_fixed': 'label multiset of the prefix, cutoff, endpoint, post-cutoff stream, publication years (features recomputed on the shuffled label order attached to the original dated positions)',
                                    'overlapping_prefixes_coherent': False, 'within_year_order': 'stable file order of the records (argsort kind=stable on year)', 'penalty': 'C = 1 fixed for the observed statistic and every replicate', 'null_hypothesis': 'the seven order features of an independently shuffled prefix carry no information beyond the offset',
                                    'disposition': 'prefixes of one claim are not shuffled coherently, so the replicates do not form coherent claim histories; the p-value is withdrawn from the scientific argument and the analysis is labelled superseded'}
pc.save_json(out, 'guide/checks.json'); print(json.dumps({k: v for k, v in out.items() if k not in ('tied_history_branch',)}, indent=1, default=float)[:6000])
