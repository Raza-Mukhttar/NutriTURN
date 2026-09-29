"""W4: corrected effect features from explicitly typed and relation-attributed 95% intervals (the anchor v2
candidate table), a Compact-7 variant on the corrected features, a parser-validation table (B3), a corrected
source-only numeric effect-reversal endpoint (I1) and a heterogeneity recomputation on typed ratio estimates.
The legacy parser inferred the scale from the sign of the limits; here the measure family is read from the
text and an interval is used only when attributed to the claim's exposure-outcome sentence."""

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
import os, sys, json, csv, math, collections
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astra_common as ac, plan_common as pc
nm, nf = ac.nm, ac.nf
CAND = os.path.join(ac.DATA, 'numeric_anchor_v2_all_candidates.csv')
H_NUM = 10


def dl_pool(y, v):
    y = np.asarray(y, float); v = np.asarray(v, float); w = 1 / v; mu = float((w * y).sum() / w.sum())
    Q = float((w * (y - mu) ** 2).sum()); k = len(y); C = w.sum() - (w ** 2).sum() / w.sum(); t2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    w2 = 1 / (v + t2); mu2 = float((w2 * y).sum() / w2.sum()); se2 = float(math.sqrt(1 / w2.sum())); i2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return mu2, se2, i2


def main():
    # typed, attributed intervals per (claim, pmid)
    typed = collections.defaultdict(list); status_est = collections.Counter(); status_rec = collections.defaultdict(set)
    for r in csv.DictReader(open(CAND)):
        status_est[(r['status'], r['effect_type'])] += 1; status_rec[(r['status'], r['effect_type'])].add((r['claim_id'], r['pmid']))
        if r['status'] == 'ATTRIBUTED':
            lo, hi = float(r['ci_lo']), float(r['ci_hi'])
            if r['effect_type'] == 'ratio' and lo > 0 and hi > lo:
                yv = (math.log(lo) + math.log(hi)) / 2; se = (math.log(hi) - math.log(lo)) / (2 * 1.96)
                if se > 1e-6 and abs(yv) < 10: typed[(r['claim_id'], r['pmid'])].append(('ratio', yv, se))
            elif r['effect_type'] == 'additive' and hi > lo:
                yv = (lo + hi) / 2; se = (hi - lo) / (2 * 1.96)
                if se > 1e-6 and abs(yv) < 10: typed[(r['claim_id'], r['pmid'])].append(('additive', yv, se))
    # legacy parser counts on the same records (for Table B3)
    Q = json.load(open(nm.QUERIES)); legacy_est = 0; legacy_rec = 0; rec_year = {}
    for cid in Q:
        recs = json.load(open(f'{nm.RECORDS}/{cid}.json'))['records']
        for rec in recs:
            e = nf.effects(rec); legacy_est += len(e); legacy_rec += int(len(e) > 0)
            if rec.get('year'): rec_year[(cid, rec['pmid'])] = int(rec['year'])
    b3 = [{'category': f'{s} / {t}', 'estimates': status_est[(s, t)], 'records': len(status_rec[(s, t)])} for (s, t) in sorted(status_est)]
    b3.append({'category': 'legacy parser: any interval, scale inferred from limit signs', 'estimates': legacy_est, 'records': legacy_rec})
    b3.append({'category': 'corrected features: attributed ratio intervals (log scale)', 'estimates': sum(1 for v in typed.values() for e in v if e[0] == 'ratio'), 'records': sum(1 for v in typed.values() if any(e[0] == 'ratio' for e in v))})
    b3.append({'category': 'corrected features: attributed additive intervals (linear scale, kept separate)', 'estimates': sum(1 for v in typed.values() for e in v if e[0] == 'additive'), 'records': sum(1 for v in typed.values() if any(e[0] == 'additive' for e in v))})
    ac.save_csv(b3, os.path.join(pc.PLAN_TABLES, 'Table_B3_parser_validation.csv'))
    # corrected per-unit effect features
    rows, keys = ac.load_units(); P, y, g = ac.prospective(rows); n = len(P)
    by_claim = collections.defaultdict(list)
    for (cid, pmid), v in typed.items():
        if (cid, pmid) in rec_year: by_claim[cid].append((rec_year[(cid, pmid)], v))
    feats = []
    for r in P:
        t = int(r['cutoff_year']); ratio = [e for yr, v in by_claim[r['claim_id']] if yr < t for e in v if e[0] == 'ratio']; add = [e for yr, v in by_claim[r['claim_id']] if yr < t for e in v if e[0] == 'additive']
        ys = [e[1] for e in ratio]
        feats.append({'val_mean_r': float(np.mean(ys)) if len(ys) >= 3 else 0.0, 'val_sd_r': float(np.std(ys)) if len(ys) >= 3 else 0.0, 'n_r': len(ys), 'n_add': len(add),
                      'i2_typed': dl_pool([e[1] for e in ratio], [e[2] ** 2 for e in ratio])[2] if len(ratio) >= 3 else float('nan')})
    X7 = nm.X_of(P, nm.COMPACT7); X2 = X7[:, :2]; su = 1 - np.abs(X2[:, 0])
    X7c = np.column_stack([X7[:, :4], [f['val_mean_r'] for f in feats], [f['val_sd_r'] for f in feats], [f['n_r'] for f in feats]])
    s7 = pc.loco_scores(X7, y, g); s7c = pc.loco_scores(X7c, y, g); s2 = pc.loco_scores(X2, y, g)
    res = {'protocol': __doc__, 'feature_coverage': {'units_with_>=3_attributed_ratio_estimates_pre_cutoff': int(sum(f['n_r'] >= 3 for f in feats)), 'units_with_any_attributed_ratio': int(sum(f['n_r'] > 0 for f in feats)),
                                                     'units_with_any_attributed_additive': int(sum(f['n_add'] > 0 for f in feats)), 'legacy_units_with_>=3_estimates': int(sum(int(r['n_rr']) >= 3 for r in P))}, 'source': {}}
    for k, s, lab in (('margin', su, 'margin rule 1-|p_t| (untrained)'), ('two_var', s2, 'two-variable score'), ('compact7_legacy', s7, 'Compact-7 (legacy effect features)'), ('compact7_corrected', s7c, 'Compact-7-corrected (typed, attributed ratio effects)')):
        d = pc.metrics(y, s, g, ref=None if k == 'margin' else su, label=lab); res['source'][k] = d
        print(f"  {lab:52s} AUROC {d['auroc']} {d['auroc_ci']} AUPRC {d['auprc']} {d['auprc_ci']} d_margin {d.get('delta_vs_margin_auroc','')} {d.get('delta_vs_margin_auroc_ci','')}", flush=True)
    d, ci = ac.paired(y, s7c, s7, g, 'auroc'); res['source']['corrected_minus_legacy_compact7_auroc'] = f'{d:+.3f} {ci}'
    d, ci = ac.paired(y, s7c, s2, g, 'auroc'); res['source']['corrected_compact7_minus_two_var_auroc'] = f'{d:+.3f} {ci}'
    # ---- corrected numeric endpoint (source only; externals have no record text) ----
    units = []
    for i, r in enumerate(P):
        t = int(r['cutoff_year']); cid = r['claim_id']
        if t + H_NUM - 1 > pc.LAST_COMPLETE_YEAR: continue
        pre = [e for yr, v in by_claim[cid] if yr < t for e in v if e[0] == 'ratio']; win = [e for yr, v in by_claim[cid] if t <= yr < t + H_NUM for e in v if e[0] == 'ratio']
        if len(pre) < 3 or len(win) < 3: continue
        mp, sp, _ = dl_pool([e[1] for e in pre], [e[2] ** 2 for e in pre]); mw, sw, _ = dl_pool([e[1] for e in win], [e[2] ** 2 for e in win])
        units.append({'i': i, 'claim_id': cid, 'cutoff_year': t, 'k_pre': len(pre), 'k_win': len(win), 'mu_pre': mp, 'mu_win': mw, 'event': int(np.sign(mp) != np.sign(mw))})
    if units:
        idx = np.array([u['i'] for u in units]); yn = np.array([u['event'] for u in units]); gn = g[idx]; mag = -np.abs(np.array([u['mu_pre'] for u in units]))
        num = {'units': len(units), 'claims': len(set(gn)), 'events': int(yn.sum()), 'positive_claims': int(len(set(gn[yn == 1]))), 'window': f'[t, t+{H_NUM}) complete (t+9 <= 2025); >= 3 attributed ratio estimates on each side', 'models': {}}
        if 0 < yn.sum() < len(yn):
            for k, s, lab in (('neg_abs_mu_pre', mag, 'untrained -|mu_pre| (magnitude reference)'), ('two_var_categorical_score', s2[idx], 'two-variable score trained on the categorical target, applied'),
                              ('compact7_corrected_categorical_score', s7c[idx], 'Compact-7-corrected trained on the categorical target, applied'), ('compact7_legacy_categorical_score', s7[idx], 'Compact-7 (legacy) trained on the categorical target, applied')):
                d = pc.metrics(yn, s, gn, ref=None if k == 'neg_abs_mu_pre' else mag, label=lab); num['models'][k] = d
                print(f"  numeric: {lab:60s} AUROC {d['auroc']} {d['auroc_ci']} AUPRC {d['auprc']} d_vs_mag {d.get('delta_vs_margin_auroc','')} {d.get('delta_vs_margin_auroc_ci','')}", flush=True)
            # refit to the numeric target (LOCO) with the two-variable inputs and with the corrected seven
            for k, X_, lab in (('two_var_refit_numeric', X2[idx], 'two-variable inputs refitted to the numeric target (LOCO)'), ('compact7_corrected_refit_numeric', X7c[idx], 'Compact-7-corrected inputs refitted to the numeric target (LOCO)')):
                s = pc.loco_scores(X_, yn, gn); d = pc.metrics(yn, s, gn, ref=mag, label=lab); num['models'][k] = d
                print(f"  numeric: {lab:60s} AUROC {d['auroc']} {d['auroc_ci']} AUPRC {d['auprc']} d_vs_mag {d.get('delta_vs_margin_auroc','')} {d.get('delta_vs_margin_auroc_ci','')}", flush=True)
        res['numeric_endpoint_corrected'] = num
        ac.save_csv([{k: v for k, v in u.items() if k != 'i'} for u in units], os.path.join(pc.PLAN_TABLES, 'Table_I1_numeric_units.csv'))
    else:
        res['numeric_endpoint_corrected'] = {'units': 0, 'note': 'no unit satisfies the corrected eligibility'}
    # ---- heterogeneity on typed ratio estimates ----
    i2 = np.array([f['i2_typed'] for f in feats]); het = {'units_with_i2_typed': int(np.isfinite(i2).sum())}
    for name, m in (('I2 not estimable (<3 typed ratio estimates)', ~np.isfinite(i2)), ('I2 <= 0.50', np.isfinite(i2) & (i2 <= .5)), ('0.50 < I2 <= 0.90', np.isfinite(i2) & (i2 > .5) & (i2 <= .9)), ('I2 > 0.90', np.isfinite(i2) & (i2 > .9))):
        if m.sum() and 0 < y[m].sum() < m.sum():
            het[name] = {'units': int(m.sum()), 'events': int(y[m].sum()), 'two_var_auroc': round(float(roc_auc_score(y[m], s2[m])), 4), 'margin_auroc': round(float(roc_auc_score(y[m], su[m])), 4), 'event_rate': round(float(y[m].mean()), 4)}
        elif m.sum(): het[name] = {'units': int(m.sum()), 'events': int(y[m].sum()), 'two_var_auroc': 'NA (single class)'}
    i2f = np.where(np.isfinite(i2), i2, 0.0); ind = np.isfinite(i2).astype(float)
    s2h = pc.loco_scores(np.column_stack([X2, i2f, ind]), y, g); d, ci = ac.paired(y, s2h, s2, g, 'auroc')
    het['two_var_plus_typed_I2_minus_two_var_auroc'] = f'{d:+.3f} {ci}'; het['typed_I2_alone_auroc_oriented_higher_means_more_change'] = round(float(roc_auc_score(y[np.isfinite(i2)], i2[np.isfinite(i2)])), 4) if 0 < y[np.isfinite(i2)].sum() < np.isfinite(i2).sum() else 'NA'
    res['heterogeneity_typed'] = het
    ac.save_json(res, os.path.join(pc.PLAN_RESULTS, 'effect_typing.json'))
    ac.save_csv([{'claim_id': r['claim_id'], 'cutoff_year': r['cutoff_year'], **f, 'legacy_n_rr': r['n_rr'], 'legacy_val_mean': r['val_mean']} for r, f in zip(P, feats)], os.path.join(pc.PLAN_RESULTS, 'corrected_effect_features.csv'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
