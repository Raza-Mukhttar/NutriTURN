"""Issue 1 (+ issue 9 validation): stationary posterior-predictive baselines on the source cohort.
  * oracle Beta-Binomial: realised future signed count m (structural baseline; not deployable)
  * deployable Beta-Binomial: m integrated over a count model fitted on the training fold (log m on log n_pre, log accrual)
  * ratio-deployable: m from the training-fold empirical future/past ratio distribution (earlier reference)
  * Dirichlet-multinomial sensitivity: NULL modelled, future total count from the count model (oracle and deployable)
Validation against the Monte-Carlo stationary null: 1,000 within-claim label permutations give each unit's simulated
change frequency, compared with the analytic oracle probability (calibration, Brier) -> issue 9 figure.
Writes results/issue1/*.json/csv and per-unit scores."""

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
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_common as pc
ac, nm = pc.ac, pc.nm


def main():
    U = pc.source_units(); y, g, P = U['y'], U['g'], U['P']; n = len(y); rng = np.random.default_rng(3)
    npl, nmi, nnull, m_sig, m_tot, rate5 = U['npl'], U['nmi'], U['nnull'], U['m_signed'], U['m_total'], U['rate5']
    ln_pre = np.log1p(npl + nmi); ln_rate = np.log1p(rate5); ln_m = np.log1p(m_sig); ln_mt = np.log1p(m_tot)
    su = 1 - np.abs(U['p']); S0 = np.array([pc.sgn0(v) for v in U['p']])        # operational current sign of the scored endpoint (0 when < 8 signed records)
    oracle = np.array([pc.bb_change_prob(npl[i], nmi[i], int(m_sig[i]), s0=S0[i]) for i in range(n)])
    oracle_raw = np.array([pc.bb_change_prob(npl[i], nmi[i], int(m_sig[i])) for i in range(n)])          # raw-majority sign: theoretical reference, diagnostic only
    dep = np.zeros(n); dep_raw = np.zeros(n); dep_ratio = np.zeros(n); dm_or = np.zeros(n); dm_dep = np.zeros(n)
    ratio = m_sig / np.maximum(npl + nmi, 1)
    for c in np.unique(g):
        tr = g != c; te = np.where(g == c)[0]
        b, sd = pc.fit_count_model(ln_pre[tr], ln_rate[tr], ln_m[tr]); bt, sdt = pc.fit_count_model(ln_pre[tr], ln_rate[tr], ln_mt[tr])
        for i in te:
            m = pc.draw_counts(b, sd, ln_pre[i], ln_rate[i], rng); dep[i] = pc.bb_change_prob(npl[i], nmi[i], m, rng=rng, s0=S0[i]); dep_raw[i] = pc.bb_change_prob(npl[i], nmi[i], m, rng=rng)
            dep_ratio[i] = pc.bb_change_prob(npl[i], nmi[i], np.maximum(1, np.rint(rng.choice(ratio[tr], pc.MC) * (npl[i] + nmi[i]))).astype(int), rng=rng, s0=S0[i])
            mt = pc.draw_counts(bt, sdt, ln_pre[i], ln_rate[i], rng); dm_dep[i] = pc.dm_change_prob(npl[i], nmi[i], nnull[i], mt, rng, s0=S0[i])
            dm_or[i] = pc.dm_change_prob(npl[i], nmi[i], nnull[i], np.full(pc.MC, int(m_tot[i])), rng, s0=S0[i])
    scores = {'margin': su, 'bb_oracle': oracle, 'bb_deployable': dep, 'bb_deployable_ratio': dep_ratio, 'dm_oracle': dm_or, 'dm_deployable': dm_dep, 'bb_oracle_rawsign': oracle_raw, 'bb_deployable_rawsign': dep_raw}
    # two-variable and Compact-7 for reference (uncalibrated ranking scores)
    X2 = U['X7'][:, :2]; scores['two_var'] = ac.loco_scores(X2, y, g); scores['compact7'] = ac.loco_scores(U['X7'], y, g)
    res = {'protocol': __doc__, 'n_units': int(n), 'events': int(y.sum()), 'source': {}}
    for k, s in scores.items():
        p = s if k.startswith(('bb', 'dm')) else pc.loco_calibrated(s, y, g)      # analytic baselines are probabilities already; others Platt-calibrated in-fold
        d = pc.full_metrics(y, s, p); d['auroc_ci'] = pc.ci(y, s, g, pc.AUROC); d['auprc_ci'] = pc.ci(y, s, g, pc.AUPRC)
        if k != 'margin':
            d['d_auroc_vs_margin'], d['d_auroc_vs_margin_ci'] = pc.paired_ci(y, s, su, g, pc.AUROC); d['d_auprc_vs_margin'], d['d_auprc_vs_margin_ci'] = pc.paired_ci(y, s, su, g, pc.AUPRC)
        if k not in ('margin', 'two_var'):
            d['d_auroc_vs_two_var'], d['d_auroc_vs_two_var_ci'] = pc.paired_ci(y, s, scores['two_var'], g, pc.AUROC)
        res['source'][k] = d
        print(f"  {k:22s} AUROC {d['auroc']:.4f} {d['auroc_ci']}  AUPRC {d['auprc']:.4f}  logloss {d['logloss']:.4f} brier {d['brier']:.4f} slope {d['cal_slope']:.2f} ece {d['ece']:.3f}  d_margin {d.get('d_auroc_vs_margin','')} {d.get('d_auroc_vs_margin_ci','')}", flush=True)
    # Brier/logloss of two-var vs bb_deployable vs prevalence-only (paired)
    prev = np.full(n, y.mean()); res['source']['prevalence_only'] = {'brier': float(pc.BRIER(y, prev)), 'logloss': float(pc.LOGLOSS(y, prev))}
    p2 = pc.loco_calibrated(scores['two_var'], y, g)
    res['source']['two_var_calibrated_minus_bb_deployable_brier'] = pc.paired_ci(y, p2, dep, g, lambda yy, pp: -pc.BRIER(yy, pp))
    res['source']['two_var_calibrated_minus_bb_deployable_logloss'] = pc.paired_ci(y, p2, dep, g, lambda yy, pp: -pc.LOGLOSS(yy, pp))
    # ---- issue 9 validation: analytic oracle probability vs Monte-Carlo permutation frequency, per unit ----
    freq = np.zeros(n); ND = 1000
    streams = {c: ac.stream(c) for c in set(g)}
    for d_ in range(ND):
        perm = {c: s_[1][rng.permutation(len(s_[1]))] for c, s_ in streams.items()}
        for i, r in enumerate(P):
            yrs, _, _ = streams[r['claim_id']]; t = int(r['cutoff_year']); codes = perm[r['claim_id']]
            pre = codes[(yrs < t)]; post = codes[(yrs >= t)]
            ps = pre[(pre != 99) & (pre != 0)]; qs = post[(post != 99) & (post != 0)]
            p_pre = ps.mean() if len(ps) >= 1 else 0.0; p_post = qs.mean() if len(qs) else 0.0
            freq[i] += (pc.sgn0(p_post) != pc.sgn0(p_pre)) if len(ps) >= 8 else 0
    freq /= ND
    val = {'draws': ND, 'mean_analytic_oracle_prob': float(oracle.mean()), 'mean_simulated_frequency': float(freq.mean()), 'expected_events_analytic': float(oracle.sum()), 'expected_events_simulated': float(freq.sum()),
           'pearson_r': float(np.corrcoef(oracle, freq)[0, 1]), 'mean_abs_diff': float(np.abs(oracle - freq).mean()), 'brier_analytic_vs_simulated_outcomes': float(np.mean((oracle - freq) ** 2)),
           'note': 'permutation null preserves realised counts; the oracle Beta-Binomial uses the realised future signed count, so the comparison tests the analytic approximation of the permutation risk'}
    res['issue9_validation'] = val; print('  validation:', val, flush=True)
    pc.save_json(res, 'issue1/stationary_baselines.json')
    rows = [{'claim_id': r['claim_id'], 'cutoff_year': r['cutoff_year'], 'y': int(y[i]), 'n_plus': int(npl[i]), 'n_minus': int(nmi[i]), 'n_null': int(nnull[i]), 'm_signed': int(m_sig[i]), 'operational_pre_sign': int(S0[i]), 'raw_pre_sign': int(pc.sgn0(npl[i] - nmi[i])), 'sim_freq': round(float(freq[i]), 4),
             **{k: float(f'{float(s[i]):.10g}') for k, s in scores.items()}} for i, r in enumerate(P)]
    pc.save_csv(rows, 'issue1/source_scores.csv')
    # figure: analytic vs simulated + margin curve
    plt = ac.plt_setup(); fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    a1.scatter(freq, oracle, s=6, alpha=.5, color='#2D6CA2'); a1.plot([0, 1], [0, 1], color='#B3402F', lw=1); a1.set_xlabel('simulated change frequency (1,000 permutations)', fontsize=8); a1.set_ylabel('analytic Beta-Binomial oracle probability', fontsize=8)
    a1.set_title(f'A. analytic vs Monte Carlo, r = {val["pearson_r"]:.3f}', fontsize=8.5, loc='left'); a1.tick_params(labelsize=7)
    absp = np.abs(U['p']); nn = npl + nmi
    for m_, col, lab in ((30, '#C9A227', 'm = 30'), (100, '#2D6CA2', 'm = 100'), (300, '#1A1A1A', 'm = 300')):
        xs = np.linspace(0, .95, 40); ys = [pc.bb_change_prob(int(round(60 * (1 + x) / 2)), int(round(60 * (1 - x) / 2)), m_) for x in xs]
        a2.plot(xs, ys, color=col, lw=1.3, label=f'analytic, n = 60, {lab}')
    a2.set_xlabel('pre-cutoff absolute margin $|p_t|$', fontsize=8); a2.set_ylabel('change probability', fontsize=8); a2.legend(fontsize=6.5, frameon=False); a2.tick_params(labelsize=7)
    a2.set_title('B. Beta-Binomial change probability vs margin', fontsize=8.5, loc='left')
    for ax in (a1, a2):
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=1.5); os.makedirs(os.path.join(pc.G, 'figures_pro'), exist_ok=True)
    fig.savefig(os.path.join(pc.G, 'figures_pro', 'fig_analytic_vs_simulation.pdf')); fig.savefig(os.path.join(pc.G, 'figures_pro', 'fig_analytic_vs_simulation.png'), dpi=200)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
