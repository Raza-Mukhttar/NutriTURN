"""Attribution cell analysis: Qwen2.5-7B under the ORIGINAL Llama prompt (cell Q0) compared with the deployed pipeline
(Llama, original prompt: L0) and the second pipeline (Qwen, explicit-definition prompt: Q1).
Record level: agreement / kappa / confusion of Q0 with L0 and with Q1; label shares.
Unit level (1,012 units): endpoint rebuilt under Q0 with the frozen builder; events, positive claims, Jaccard with L0 and Q1;
margin rule, count-predicted operational stationary baseline (LOCO count model) and signed two-variable score refitted LOCO;
paired signed-minus-margin; deployed score evaluated on the Q0 endpoint; common-support (>=8 signed under L0, Q0 and Q1).
Writes results/guide/cell_origprompt.json."""

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
import os, sys, json, glob, csv, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_common as pc
ac, nm = pc.ac, pc.nm
CODE = {'PROTECTIVE': -1, 'HARMFUL': 1, 'NULL': 0, 'UNCLEAR': 99}; LABELS = list(CODE)
def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(pc.RES, d, 'qwen25_7b_shard*.jsonl'))):
        for line in open(f): rows.append(json.loads(line))
    return {(r['claim_id'], r['pmid']): r for r in rows}
def kappa(a, b, labels):
    a = np.asarray(a); b = np.asarray(b); po = np.mean(a == b); pe = sum(np.mean(a == l) * np.mean(b == l) for l in labels); return float((po - pe) / (1 - pe))
Q0 = load('reextraction_origprompt'); Q1 = load('reextraction'); keys = sorted(set(Q0) & set(Q1)); print('records', len(Q0), len(Q1), 'joined', len(keys))
L = [Q1[k]['llama_label'] for k in keys]; q0 = [Q0[k]['qwen_label_full'] for k in keys]; q1 = [Q1[k]['qwen_label_full'] for k in keys]
rec = {'records': len(keys), 'Q0_vs_L0': {'agreement': float(np.mean([a == b for a, b in zip(L, q0)])), 'kappa': kappa(L, q0, LABELS)}, 'Q0_vs_Q1': {'agreement': float(np.mean([a == b for a, b in zip(q0, q1)])), 'kappa': kappa(q0, q1, LABELS)}, 'Q1_vs_L0': {'agreement': float(np.mean([a == b for a, b in zip(L, q1)])), 'kappa': kappa(L, q1, LABELS)},
       'label_shares': {n: {l: float(np.mean([x == l for x in v])) for l in LABELS} for n, v in (('L0', L), ('Q0', q0), ('Q1', q1))}}
conf = collections.Counter(zip(L, q0)); rec['confusion_L0_rows_Q0_cols'] = {a: {b: conf[(a, b)] for b in LABELS} for a in LABELS}
sm = [(a, b) for a, b in zip(L, q0) if a in ('PROTECTIVE', 'HARMFUL') and b in ('PROTECTIVE', 'HARMFUL')]; rec['Q0_vs_L0']['signed_only_agreement'] = float(np.mean([a == b for a, b in sm])); rec['Q0_vs_L0']['signed_only_n'] = len(sm)
print(json.dumps(rec, indent=1)[:1500], flush=True)
# ---- unit level
U = pc.source_units(); P, y, g = U['P'], U['y'], U['g']; rng = np.random.default_rng(3)
def codes_for(cid, cell):
    yrs, codes, pmids = ac.stream(cid)
    if cell == 'L0': return yrs, codes
    M = Q0 if cell == 'Q0' else Q1; return yrs, np.array([CODE[M[(cid, pm)]['qwen_label_full']] if (cid, pm) in M else 99 for pm in pmids])
def build(cell):
    p = np.zeros(len(P)); a = np.zeros(len(P)); yy = np.zeros(len(P), int); npl = np.zeros(len(P), int); nmi = np.zeros(len(P), int); r5 = np.zeros(len(P)); ms = np.zeros(len(P), int)
    for i, r in enumerate(P):
        yrs, codes = codes_for(r['claim_id'], cell); t = int(r['cutoff_year']); f = ac.direction_feats(yrs[yrs < t], codes[yrs < t], t, 'A'); p[i] = f['pooled_direction']; a[i] = f['agreement']
        pre = codes[yrs < t]; npl[i] = (pre == 1).sum(); nmi[i] = (pre == -1).sum(); r5[i] = (((yrs >= t - 5) & (yrs < t) & (codes != 99) & (codes != 0)).sum()) / 5.0
        post = codes[yrs >= t]; qs = post[(post != 99) & (post != 0)]; ms[i] = len(qs); yy[i] = int(pc.sgn0(float(qs.mean()) if len(qs) else 0.0) != pc.sgn0(p[i]))
    su = 1 - np.abs(p); s2 = ac.loco_scores(np.column_stack([p, a]), yy, g); dep = np.zeros(len(P)); ln_pre = np.log1p(npl + nmi); ln_rate = np.log1p(r5); ln_m = np.log1p(ms)
    for c in np.unique(g):
        tr = g != c; b, sd = pc.fit_count_model(ln_pre[tr], ln_rate[tr], ln_m[tr])
        for i in np.where(g == c)[0]: dep[i] = pc.bb_change_prob(npl[i], nmi[i], pc.draw_counts(b, sd, ln_pre[i], ln_rate[i], rng), rng=rng, s0=pc.sgn0(p[i]))
    return {'p': p, 'y': yy, 'nsig': npl + nmi, 'margin': su, 'two_var': s2, 'bb': dep}
cells = {c: build(c) for c in ('L0', 'Q0', 'Q1')}; assert (cells['L0']['y'] == y).all()
S1 = {(r['claim_id'], str(int(float(r['cutoff_year'])))): r for r in csv.DictReader(open(os.path.join(pc.RES, 'issue1', 'source_scores.csv')))}
cells['L0']['bb'] = np.array([float(S1[(r['claim_id'], str(int(r['cutoff_year'])))]['bb_deployable']) for r in P]); cells['L0']['two_var'] = np.array([float(S1[(r['claim_id'], str(int(r['cutoff_year'])))]['two_var']) for r in P])
out = {'protocol': __doc__, 'record_level': rec, 'unit_level': {}}
for c, C in cells.items():
    yy = C['y']; d = {'events': int(yy.sum()), 'positive_claims': int(len(set(g[yy == 1]))), 'sparse_units': int((C['nsig'] < 8).sum()), 'events_on_sparse': int(yy[C['nsig'] < 8].sum()), 'pooled_direction_corr_with_L0': float(np.corrcoef(cells['L0']['p'], C['p'])[0, 1])}
    for o in ('L0', 'Q1'):
        yo = cells[o]['y']; d[f'jaccard_with_{o}'] = float(((yy == 1) & (yo == 1)).sum() / max(((yy == 1) | (yo == 1)).sum(), 1)); d[f'shared_events_with_{o}'] = int(((yy == 1) & (yo == 1)).sum())
    for k in ('margin', 'bb', 'two_var'):
        d[f'{k}_auroc'] = float(pc.AUROC(yy, C[k])); d[f'{k}_auroc_ci'] = pc.ci(yy, C[k], g, pc.AUROC); d[f'{k}_auprc'] = float(pc.AUPRC(yy, C[k]))
    d['two_var_minus_margin'] = pc.paired_ci(yy, C['two_var'], C['margin'], g, pc.AUROC); d['two_var_minus_bb'] = pc.paired_ci(yy, C['two_var'], C['bb'], g, pc.AUROC)
    if c != 'L0': d['deployed_two_var_on_this_endpoint_auroc'] = float(pc.AUROC(yy, cells['L0']['two_var']))
    out['unit_level'][c] = d; print(c, json.dumps(d, default=float), flush=True)
mask = (cells['L0']['nsig'] >= 8) & (cells['Q0']['nsig'] >= 8) & (cells['Q1']['nsig'] >= 8); gm = g[mask]; cs = {'units': int(mask.sum())}
for c, C in cells.items():
    yy = C['y'][mask]; cs[c] = {'events': int(yy.sum()), 'margin_auroc': float(pc.AUROC(yy, C['margin'][mask])), 'bb_auroc': float(pc.AUROC(yy, C['bb'][mask])), 'two_var_auroc': float(pc.AUROC(yy, C['two_var'][mask])), 'two_var_minus_margin': pc.paired_ci(yy, C['two_var'][mask], C['margin'][mask], gm, pc.AUROC)}
out['common_support_three_cells'] = cs; print('common support', json.dumps(cs, default=float), flush=True)
pc.save_json(out, 'guide/cell_origprompt.json'); print('DONE')
