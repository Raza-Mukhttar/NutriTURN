"""Issue 4: one metric suite for every method and protocol. For each cohort and method: AUROC, AUPRC, log loss,
Brier, calibration slope/intercept, ECE (10 bins), precision/recall at review budgets 5/10/20 %, number needed to
review at 10 %, share reviewed and workload reduction at recall 0.8. Probabilities are calibrated with TRAINING-only
Platt scaling: leave-one-claim-out on the source cohort (calibrator fitted on the other claims), per-origin
training-fold calibrators on the forward cohorts (issue 5), and the FROZEN source calibrator (fitted once on all
source out-of-fold scores) on the external cohorts. Analytic Beta-Binomial probabilities are used as they are.
Cohorts: source (1,012 units, leave-one-claim-out), forward H3/H5/H10 complete-window origin cohorts (claim-disjoint),
External-1 and External-2 (frozen source fits). Writes results/issue4/full_metrics.{csv,json}."""

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_common as pc
sys.path.insert(0, _os.path.join(_NT, 'scripts', 'lib'))
import two_variable_transfer as tv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ac, nm = pc.ac, pc.nm
OOF = _os.path.join(_NT, 'data', 'protocol_snapshot', 'analyses_v2', 'tables', '08_oof_predictions.csv')
NAMES = {'margin': 'margin rule $1-|p_t|$', 'bb_deployable': 'Beta--Binomial, deployable', 'bb_oracle': 'Beta--Binomial, oracle count', 'dm_deployable': 'Dirichlet--multinomial, deployable', 'two_var': 'two-variable logistic',
         'compact7': 'Compact-7 logistic', 'random_forest': 'random forest (55 features)', 'xgboost': 'XGBoost (55 features)', 'offset_plus_order': 'stationary offset + order features', 'sign_invariant_full': 'sign-invariant logistic',
         'two_var_augmented': 'two-variable, flip-augmented'}


def rd(path): return list(csv.DictReader(open(os.path.join(pc.RES, path))))


def main():
    rows = []; res = {'protocol': __doc__, 'cohorts': {}}
    def add(cohort, method, y, s, p, g, extra=None):
        d = pc.full_metrics(y, s, p); d['auroc_ci'] = pc.ci(y, s, g, pc.AUROC); d['auprc_ci'] = pc.ci(y, s, g, pc.AUPRC); d['units'] = int(len(y)); d['events'] = int(y.sum())
        if extra: d.update(extra)
        res['cohorts'].setdefault(cohort, {})[method] = d; rows.append({'cohort': cohort, 'method': method, 'label': NAMES.get(method, method), **{k: (float(f'{v:.10g}') if isinstance(v, float) else v) for k, v in d.items()}})
        print(f"  {cohort:12s} {method:20s} n={len(y)} ev={int(y.sum())} AUROC {d['auroc']:.3f} {d['auroc_ci']} AUPRC {d['auprc']:.3f} LL {d['logloss']:.3f} Brier {d['brier']:.4f} slope {d['cal_slope']:.2f} ECE {d['ece']:.3f} R@10 {d['recall@10']:.3f} WR80 {d['workload_reduction_at_recall_80']:.3f}", flush=True)
    # ---------------- source
    S1 = rd('issue1/source_scores.csv'); S2 = {(r['claim_id'], r['cutoff_year']): r for r in rd('issue2/source_scores.csv')}; S3 = {(r['claim_id'], r['cutoff_year']): r for r in rd('issue3/source_scores.csv')}
    y = np.array([int(r['y']) for r in S1]); g = np.array([r['claim_id'] for r in S1]); key = [(r['claim_id'], r['cutoff_year']) for r in S1]
    oof = {(r['claim_id'], str(float(r['cutoff_year']))): r for r in csv.DictReader(open(OOF))}
    src = {'margin': np.array([float(r['margin']) for r in S1]), 'bb_deployable': np.array([float(r['bb_deployable']) for r in S1]), 'bb_oracle': np.array([float(r['bb_oracle']) for r in S1]), 'dm_deployable': np.array([float(r['dm_deployable']) for r in S1]),
           'two_var': np.array([float(r['two_var']) for r in S1]), 'compact7': np.array([float(r['compact7']) for r in S1]),
           'random_forest': np.array([float(oof[(k[0], str(float(k[1])))]['random_forest']) for k in key]), 'xgboost': np.array([float(oof[(k[0], str(float(k[1])))]['xgboost']) for k in key]),
           'offset_plus_order': np.array([float(S2[k]['offset_plus_order']) for k in key]), 'sign_invariant_full': np.array([float(S3[k]['sign_invariant_full']) for k in key]), 'two_var_augmented': np.array([float(S3[k]['two_var_augmented']) for k in key])}
    yo = np.array([int(oof[(k[0], str(float(k[1])))]['y']) for k in key]); assert (yo == y).all()
    cal_src = {}
    for k, s in src.items():
        p = s if k in ('bb_deployable', 'bb_oracle', 'dm_deployable', 'offset_plus_order') else pc.loco_calibrated(s, y, g); cal_src[k] = p
        ex = {} if k == 'margin' else dict(zip(('d_auroc_vs_margin', 'd_auroc_vs_margin_ci'), pc.paired_ci(y, s, src['margin'], g, pc.AUROC)))
        if k != 'margin': ex.update(dict(zip(('d_auprc_vs_margin', 'd_auprc_vs_margin_ci'), pc.paired_ci(y, s, src['margin'], g, pc.AUPRC)))); ex.update(dict(zip(('d_brier_vs_margin', 'd_brier_vs_margin_ci'), pc.paired_ci(y, p, cal_src['margin'], g, lambda yy, pp: -pc.BRIER(yy, pp)))))
        add('source', k, y, s, p, g, ex)
    # ---------------- forward complete-window cohorts (issue 5, claim-disjoint, per-origin training-only calibrators)
    for H in (3, 5, 10):
        F = rd(f'issue5/pooled_scores_H{H}.csv'); yf = np.array([int(r['y']) for r in F]); gf = np.array([r['claim_id'] for r in F])
        for k, pk in (('margin', 'margin_cal'), ('bb_deployable', 'bb_deployable'), ('two_var', 'two_var_cal'), ('compact7', 'compact7_cal')):
            s = np.array([float(r[k]) for r in F]); p = np.array([float(r[pk]) for r in F]); su = np.array([float(r['margin']) for r in F]); pu = np.array([float(r['margin_cal']) for r in F])
            ex = {} if k == 'margin' else {**dict(zip(('d_auroc_vs_margin', 'd_auroc_vs_margin_ci'), pc.paired_ci(yf, s, su, gf, pc.AUROC))), **dict(zip(('d_auprc_vs_margin', 'd_auprc_vs_margin_ci'), pc.paired_ci(yf, s, su, gf, pc.AUPRC))), **dict(zip(('d_brier_vs_margin', 'd_brier_vs_margin_ci'), pc.paired_ci(yf, p, pu, gf, lambda yy, pp: -pc.BRIER(yy, pp))))}
            add(f'forward_H{H}', k, yf, s, p, gf, ex)
    # ---------------- external cohorts: frozen source fits and frozen source calibrator
    rows_src, _ = ac.load_units(); P, ys, gs = ac.prospective(rows_src); X7 = nm.X_of(P, nm.COMPACT7); X2 = X7[:, :2]
    f2, m2, sc2 = tv.fit_frozen(X2, ys); f7, m7, sc7 = tv.fit_frozen(X7, ys)
    cal2 = pc.platt_fit(src['two_var'], y); cal7 = pc.platt_fit(src['compact7'], y); calu = pc.platt_fit(src['margin'], y)
    npl_s = np.array([tv.counts_of_row(r)[1] for r in P]); nmi_s = np.array([tv.counts_of_row(r)[0] for r in P])   # counts_of_row returns (nP, nH, nN); HARMFUL = +1
    rate_s = np.array([float(r.get('accrual_rate_5yr', 0)) for r in P]); m_s = []
    for r in P:
        yrs, codes, _ = ac.stream(r['claim_id']); t = int(r['cutoff_year']); w = codes[yrs >= t]; m_s.append(int(((w != 99) & (w != 0)).sum()))
    b_cnt, sd_cnt = pc.fit_count_model(np.log1p(npl_s + nmi_s), np.log1p(rate_s), np.log1p(np.array(m_s))); rng = np.random.default_rng(3)
    res['external_count_model'] = {'coef': [float(v) for v in b_cnt], 'resid_sd': sd_cnt, 'note': 'log(1+m_signed) ~ log(1+n_signed_pre) + log(1+accrual_rate_5yr), fitted on all source units; accrual_rate_5yr (all records) is the only accrual field shipped with the external units'}
    for k_ in (1, 2):
        E, _ = ac.load_external(k_); E = [r for r in E if float(r['n_post_t']) >= 20]; ye = np.array([r['y_sub'] for r in E]); ge = np.array([r['claim_id'] for r in E])   # future-eligible units only (322 / 1,257)
        Xe7 = nm.X_of(E, nm.COMPACT7); Xe2 = Xe7[:, :2]; su = 1 - np.abs(Xe2[:, 0])
        ag = np.array([tv.agreement_from_counts(*tv.counts_of_row(r))['A'] for r in E]); assert np.max(np.abs(ag - Xe2[:, 1])) < 1e-6
        s2 = f2(Xe2); s7 = f7(Xe7); dep = np.zeros(len(E))
        for i, r in enumerate(E):
            nP, nH, nN = tv.counts_of_row(r); dep[i] = pc.bb_change_prob(nH, nP, pc.draw_counts(b_cnt, sd_cnt, np.log1p(nP + nH), np.log1p(float(r.get('accrual_rate_5yr', 0))), rng), rng=rng, s0=pc.sgn0(float(r['pooled_direction'])))
        pu = pc.platt_apply(calu, su); p2 = pc.platt_apply(cal2, s2); p7 = pc.platt_apply(cal7, s7)
        for k, s, p in (('margin', su, pu), ('bb_deployable', dep, dep), ('two_var', s2, p2), ('compact7', s7, p7)):
            ex = {} if k == 'margin' else {**dict(zip(('d_auroc_vs_margin', 'd_auroc_vs_margin_ci'), pc.paired_ci(ye, s, su, ge, pc.AUROC))), **dict(zip(('d_auprc_vs_margin', 'd_auprc_vs_margin_ci'), pc.paired_ci(ye, s, su, ge, pc.AUPRC))), **dict(zip(('d_brier_vs_margin', 'd_brier_vs_margin_ci'), pc.paired_ci(ye, p, pu, ge, lambda yy, pp: -pc.BRIER(yy, pp))))}
            add(f'external_{k_}', k, ye, s, p, ge, ex)
    pc.save_csv(rows, 'issue4/full_metrics.csv'); pc.save_json(res, 'issue4/full_metrics.json'); print('DONE', flush=True)


if __name__ == '__main__':
    main()
