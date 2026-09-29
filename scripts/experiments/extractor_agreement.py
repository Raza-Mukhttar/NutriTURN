"""Issue 6: agreement between the deployed Llama-3.1-8B first-token annotation and the independent Qwen2.5-7B
re-annotation (complete verbalizer scoring), and the forecasting result under each annotator, under label
consensus, and under soft labels (Monte-Carlo draws from the Qwen label probabilities).
Record level: raw agreement, Cohen's kappa (4-way and signed-only), confusion matrix, first-token vs full-string
agreement within Qwen. Unit level (the 1,012 future-eligible source units): correlation of pre-cutoff pooled
direction, agreement of the target, event-set Jaccard; two-variable, margin and stationary-baseline AUROC under
Llama labels, Qwen labels, consensus (records where both agree; disagreements -> UNCLEAR) and 500 soft-label
draws. Writes results/issue6/."""

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
import os, sys, json, glob, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_common as pc
ac, nm = pc.ac, pc.nm
CODE = {'PROTECTIVE': -1, 'HARMFUL': 1, 'NULL': 0, 'UNCLEAR': 99}; LABELS = ['PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR']
NSOFT = 500


def kappa(a, b, labels):
    a = np.asarray(a); b = np.asarray(b); n = len(a); po = np.mean(a == b); pe = sum(np.mean(a == l) * np.mean(b == l) for l in labels); return float((po - pe) / (1 - pe))


def unit_stats(codes_by, P, y_ref):
    """codes_by: dict claim -> codes array aligned with ac.stream(claim) order. Returns p, y, n_plus, n_minus, rate5 per unit."""
    p = np.zeros(len(P)); y = np.zeros(len(P), int); npl = np.zeros(len(P), int); nmi = np.zeros(len(P), int); a = np.zeros(len(P)); r5 = np.zeros(len(P)); msig = np.zeros(len(P), int)
    for i, r in enumerate(P):
        yrs, _, _ = ac.stream(r['claim_id']); codes = codes_by[r['claim_id']]; t = int(r['cutoff_year'])
        f = ac.direction_feats(yrs[yrs < t], codes[yrs < t], t, 'A'); p[i] = f['pooled_direction']; a[i] = f['agreement']
        pre = codes[yrs < t]; npl[i] = (pre == 1).sum(); nmi[i] = (pre == -1).sum(); r5[i] = (((yrs >= t - 5) & (yrs < t) & (codes != 99) & (codes != 0)).sum()) / 5.0
        post = codes[yrs >= t]; qs = post[(post != 99) & (post != 0)]; msig[i] = len(qs)
        y[i] = int(pc.sgn0(float(qs.mean()) if len(qs) else 0.0) != pc.sgn0(p[i]))
    return p, a, y, npl, nmi, r5, msig


def scores(p, a, y, g, npl, nmi, r5, msig, rng):
    """Same procedure as issue 1: leave-one-claim-out count model (log future signed count on log pre-cutoff signed count and accrual),
    400 draws, exact Beta-Binomial tail averaged over the draws."""
    X2 = np.column_stack([p, a]); s2 = ac.loco_scores(X2, y, g); su = 1 - np.abs(p)
    ln_pre = np.log1p(npl + nmi); ln_rate = np.log1p(r5); ln_m = np.log1p(msig); dep = np.zeros(len(y))
    for c in np.unique(g):
        tr = g != c; b, sd = pc.fit_count_model(ln_pre[tr], ln_rate[tr], ln_m[tr])
        for i in np.where(g == c)[0]: dep[i] = pc.bb_change_prob(npl[i], nmi[i], pc.draw_counts(b, sd, ln_pre[i], ln_rate[i], rng), rng=rng, s0=pc.sgn0(p[i]))
    return {'two_var': s2, 'margin': su, 'bb_deployable': dep}


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(pc.RES, 'reextraction', 'qwen25_7b_shard*.jsonl'))):
        for l in open(f): rows.append(json.loads(l))
    print('records', len(rows), flush=True)
    L = [r['llama_label'] for r in rows]; Qf = [r['qwen_label_full'] for r in rows]; Q1 = [r['qwen_label_first'] for r in rows]
    conf = collections.Counter(zip(L, Qf))
    rec = {'records_scored': len(rows), 'raw_agreement': float(np.mean([a == b for a, b in zip(L, Qf)])), 'kappa_4way': kappa(L, Qf, LABELS),
           'qwen_first_vs_full_agreement': float(np.mean([a == b for a, b in zip(Q1, Qf)])), 'confusion_llama_rows_qwen_cols': {a: {b: conf[(a, b)] for b in LABELS} for a in LABELS},
           'llama_label_shares': {l: float(np.mean([x == l for x in L])) for l in LABELS}, 'qwen_label_shares': {l: float(np.mean([x == l for x in Qf])) for l in LABELS},
           'mean_first_token_mass': float(np.mean([r['first_token_mass'] for r in rows]))}
    sm = [(a, b) for a, b in zip(L, Qf) if a in ('PROTECTIVE', 'HARMFUL') and b in ('PROTECTIVE', 'HARMFUL')]
    rec['signed_only_records'] = len(sm); rec['signed_only_agreement'] = float(np.mean([a == b for a, b in sm])); rec['signed_only_kappa'] = kappa([a for a, _ in sm], [b for _, b in sm], ['PROTECTIVE', 'HARMFUL'])
    both_signed_or_null = [(a, b) for a, b in zip(L, Qf) if a != 'UNCLEAR' and b != 'UNCLEAR']; rec['resolved_both_agreement'] = float(np.mean([a == b for a, b in both_signed_or_null])); rec['resolved_both_kappa'] = kappa([a for a, _ in both_signed_or_null], [b for _, b in both_signed_or_null], ['PROTECTIVE', 'HARMFUL', 'NULL'])
    print(rec, flush=True)
    # ---- unit level
    U = pc.source_units(); P, y, g = U['P'], U['y'], U['g']; rng = np.random.default_rng(3)
    qmap = {(r['claim_id'], r['pmid']): r for r in rows}
    codes_llama, codes_qwen, codes_cons, probs = {}, {}, {}, {}
    missing = 0
    for c in set(g):
        yrs, codes, pmids = ac.stream(c); cq = codes.copy(); cc = codes.copy(); pr = np.zeros((len(codes), 4))
        for j, pm in enumerate(pmids):
            r = qmap.get((c, pm))
            if r is None: missing += 1; cq[j] = 99; cc[j] = 99; pr[j] = [0, 0, 0, 1]; continue
            cq[j] = CODE[r['qwen_label_full']]; cc[j] = codes[j] if CODE[r['qwen_label_full']] == codes[j] else 99; pr[j] = r['p_full']
        codes_llama[c] = codes; codes_qwen[c] = cq; codes_cons[c] = cc; probs[c] = pr
    print('missing re-annotations among stream records:', missing, flush=True)
    out = {'record_level': rec, 'unit_level': {}, 'protocol': __doc__}
    pl, al, yl, n1, n2, r5, _ = unit_stats(codes_llama, P, y); assert (yl == y).all()
    res_units = {}
    for name, cb in (('llama', codes_llama), ('qwen', codes_qwen), ('consensus', codes_cons)):
        p_, a_, y_, npl, nmi, r5_, ms_ = unit_stats(cb, P, y); S = scores(p_, a_, y_, g, npl, nmi, r5_, ms_, rng)
        if name == 'llama':   # identical procedure to issue 1; reuse its stored predictions so the deployed row is the same result identity as the source tables
            import csv as _csv
            S1 = {(r['claim_id'], str(int(float(r['cutoff_year'])))): r for r in _csv.DictReader(open(os.path.join(pc.RES, 'issue1', 'source_scores.csv')))}
            S['bb_deployable'] = np.array([float(S1[(r['claim_id'], str(int(r['cutoff_year'])))]['bb_deployable']) for r in P]); S['two_var'] = np.array([float(S1[(r['claim_id'], str(int(r['cutoff_year'])))]['two_var']) for r in P])
        d = None
        d = {'events': int(y_.sum()), 'positive_claims': int(len(set(g[y_ == 1]))), 'pooled_direction_corr_with_llama': float(np.corrcoef(pl, p_)[0, 1]), 'target_agreement_with_llama': float(np.mean(y_ == y)),
             'event_jaccard_with_llama': float(((y_ == 1) & (y == 1)).sum() / max(((y_ == 1) | (y == 1)).sum(), 1)), 'events_both': int(((y_ == 1) & (y == 1)).sum()),
             'events_with_fewer_than_8_signed_pre': int(y_[(npl + nmi) < 8].sum()), 'units_with_fewer_than_8_signed_pre': int(((npl + nmi) < 8).sum()),
             'identity': f'inputs and endpoint rebuilt under the {name} annotation; two-variable model refitted leave-one-claim-out on this target; count model leave-one-claim-out'}
        for k, s in S.items():
            d[f'{k}_auroc'] = float(pc.AUROC(y_, s)) if 0 < y_.sum() < len(y_) else None; d[f'{k}_auroc_ci'] = pc.ci(y_, s, g, pc.AUROC); d[f'{k}_auprc'] = float(pc.AUPRC(y_, s))
        d['two_var_minus_margin_auroc'] = pc.paired_ci(y_, S['two_var'], S['margin'], g, pc.AUROC); d['two_var_minus_bb_auroc'] = pc.paired_ci(y_, S['two_var'], S['bb_deployable'], g, pc.AUROC)
        # cross-annotator: Llama-trained score evaluated on Qwen targets and vice versa (same units)
        if name != 'llama': d['deployed_llama_two_var_score_evaluated_on_this_target_auroc'] = float(pc.AUROC(y_, res_units['llama']['s2'])) if 0 < y_.sum() < len(y_) else None   # different comparison: Llama inputs + Llama-trained score, evaluated against this annotation's endpoint
        res_units[name] = {'s2': S['two_var'], 'su': S['margin'], 'sdep': S['bb_deployable'], 'y': y_, 'nsig': npl + nmi, 'p': p_}; out['unit_level'][name] = d; print(' ', name, {k: v for k, v in d.items()}, flush=True)
        per_unit = [{'claim_id': r['claim_id'], 'cutoff_year': int(r['cutoff_year']), 'pipeline': name, 'y': int(y_[i]), 'n_signed_pre': int(npl[i] + nmi[i]), 'operational_pre_sign': int(pc.sgn0(p_[i])), 'raw_pre_sign': int(pc.sgn0(npl[i] - nmi[i])),
                     'margin': float(f'{S["margin"][i]:.10g}'), 'bb_deployable': float(f'{S["bb_deployable"][i]:.10g}'), 'two_var': float(f'{S["two_var"][i]:.10g}')} for i, r in enumerate(P)]
        pc.save_csv(per_unit, f'issue6/unit_scores_{name}.csv')
    # ---- E2: common-support sensitivity (>= 8 signed pre-cutoff records under BOTH pipelines); full-cohort-trained out-of-fold scores restricted to the mask
    L, Qn = res_units['llama'], res_units['qwen']; mask = (L['nsig'] >= 8) & (Qn['nsig'] >= 8); gm = g[mask]
    cs = {'definition': 'units with >= 8 signed pre-cutoff records under both the deployed and the second pipeline; each pipeline keeps its own operational endpoint; full-cohort-trained out-of-fold scores restricted to the mask (not refitted)',
          'units': int(mask.sum()), 'claims': int(len(set(gm))), 'excluded_units': int((~mask).sum()), 'excluded_deployed_sparse': int((L['nsig'] < 8).sum()), 'excluded_second_sparse': int((Qn['nsig'] < 8).sum()), 'deployed_sparse_unit_within_second_sparse': bool(np.all(Qn['nsig'][L['nsig'] < 8] < 8)),
          'mask_hash': __import__('hashlib').sha256(','.join(f"{r['claim_id']}:{int(r['cutoff_year'])}" for r, mm in zip(P, mask) if mm).encode()).hexdigest()}
    yl, yq = L['y'][mask], Qn['y'][mask]
    cs['deployed'] = {'events': int(yl.sum()), 'positive_claims': int(len(set(gm[yl == 1])))}; cs['second'] = {'events': int(yq.sum()), 'positive_claims': int(len(set(gm[yq == 1])))}
    cs['shared_events'] = int(((yl == 1) & (yq == 1)).sum()); cs['deployed_only_events'] = int(((yl == 1) & (yq == 0)).sum()); cs['second_only_events'] = int(((yl == 0) & (yq == 1)).sum()); cs['event_jaccard'] = cs['shared_events'] / max(cs['shared_events'] + cs['deployed_only_events'] + cs['second_only_events'], 1)
    for name, Rr, yy in (('deployed', L, yl), ('second', Qn, yq)):
        for k in ('margin', 'bb_deployable', 'two_var'):
            s = Rr[{'margin': 'su', 'bb_deployable': 'sdep', 'two_var': 's2'}[k]][mask]; cs[name][f'{k}_auroc'] = float(pc.AUROC(yy, s)); cs[name][f'{k}_auroc_ci'] = pc.ci(yy, s, gm, pc.AUROC); cs[name][f'{k}_auprc'] = float(pc.AUPRC(yy, s))
        cs[name]['two_var_minus_margin_auroc'] = pc.paired_ci(yy, Rr['s2'][mask], Rr['su'][mask], gm, pc.AUROC); cs[name]['two_var_minus_bb_auroc'] = pc.paired_ci(yy, Rr['s2'][mask], Rr['sdep'][mask], gm, pc.AUROC)
        # resamples in which a required class is absent: claim bootstrap over the restricted population
        _, _, nvalid = nm.claim_boot(gm, lambda i: pc.AUROC(yy[i], Rr['su'][mask][i]), n=2000, seed=3); cs[name]['resamples_evaluable_of_2000'] = int(nvalid)
    out['common_support'] = cs; print('  common support:', cs, flush=True)
    # ---- soft labels: draw each record's label from the Qwen probabilities, rebuild target and inputs, refit
    aur = collections.defaultdict(list); ev = []
    for d_ in range(NSOFT):
        cb = {}
        for c in probs:
            pr = probs[c]; u = rng.random(len(pr)); cum = np.cumsum(pr, 1); lab = (u[:, None] > cum).sum(1); lab = np.minimum(lab, 3); cb[c] = np.array([CODE[LABELS[k]] for k in lab])
        p_, a_, y_, npl, nmi, r5_, _ = unit_stats(cb, P, y)
        if 0 < y_.sum() < len(y_):
            s2 = ac.loco_scores(np.column_stack([p_, a_]), y_, g); aur['two_var'].append(float(pc.AUROC(y_, s2))); aur['margin'].append(float(pc.AUROC(y_, 1 - np.abs(p_)))); ev.append(int(y_.sum()))
        if d_ % 50 == 0: print(f'  soft draw {d_}: events {ev[-1] if ev else None} 2v {aur["two_var"][-1] if aur["two_var"] else None}', flush=True)
    out['soft_labels'] = {'draws': NSOFT, 'events_mean': float(np.mean(ev)), 'events_q025': float(np.quantile(ev, .025)), 'events_q975': float(np.quantile(ev, .975)),
                          **{f'{k}_auroc_mean': float(np.mean(v)) for k, v in aur.items()}, **{f'{k}_auroc_q025': float(np.quantile(v, .025)) for k, v in aur.items()}, **{f'{k}_auroc_q975': float(np.quantile(v, .975)) for k, v in aur.items()}, 'interval_type': 'variation across annotation draws (2.5-97.5 percentiles over draws); no claim-bootstrap layer', 'sampling_unit': 'one label per claim-record association per draw, reused across every cutoff of the claim'}
    print(out['soft_labels'], flush=True)
    pc.save_json(out, 'issue6/extractor_agreement.json'); print('DONE', flush=True)


if __name__ == '__main__':
    main()
