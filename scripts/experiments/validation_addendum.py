"""Issue 9 addendum: validate the analytic Beta-Binomial change probability against the i.i.d.-draw stationary null
(null B: future signed labels drawn i.i.d. with the pre-cutoff proportion, realised future signed count), per unit,
1,000 draws; and against the exact hypergeometric (permutation) probability conditional on the realised claim totals.
Extends results/issue1/stationary_baselines.json and redraws figures_pro/fig_analytic_vs_simulation.*"""

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
ac = pc.ac


def main():
    S1 = list(csv.DictReader(open(os.path.join(pc.RES, 'issue1', 'source_scores.csv')))); rng = np.random.default_rng(3); ND = 1000
    npl = np.array([int(r['n_plus']) for r in S1]); nmi = np.array([int(r['n_minus']) for r in S1]); m = np.array([int(r['m_signed']) for r in S1]); oracle = np.array([float(r['bb_oracle']) for r in S1]); simperm = np.array([float(r['sim_freq']) for r in S1])
    n = len(S1); s0 = np.array([int(r['operational_pre_sign']) for r in S1]); s0_raw = np.sign(npl - nmi); th = npl / np.maximum(npl + nmi, 1)   # operational event indicator, as scored
    freqB = np.zeros(n)
    for i in range(n):
        if m[i] == 0: continue
        k = rng.binomial(m[i], th[i], size=ND); freqB[i] = np.mean(np.sign(2 * k - m[i]) != s0[i])
    # exact plug-in binomial: k ~ Binomial(m, theta_hat)
    from scipy.stats import binom
    plug = np.zeros(n)
    for i in range(n):
        if m[i] == 0: continue
        k = np.arange(m[i] + 1); pk = binom.pmf(k, m[i], th[i]); plug[i] = float(pk[np.sign(2 * k - m[i]) != s0[i]].sum())
    plug_raw = np.zeros(n)
    for i in range(n):
        if m[i] == 0: continue
        k = np.arange(m[i] + 1); pk = binom.pmf(k, m[i], th[i]); plug_raw[i] = float(pk[np.sign(2 * k - m[i]) != s0_raw[i]].sum())
    oracle_raw = np.array([float(r['bb_oracle_rawsign']) for r in S1])
    out = {'event_indicator': 'operational sign s_t for the analytic, plug-in and simulated vectors (raw-majority vectors reported separately as *_raw)', 'mean_oracle_bb_raw': float(oracle_raw.mean()), 'expected_events_raw_bb': float(oracle_raw.sum()), 'expected_events_plugin_raw': float(plug_raw.sum()), 'expected_events_plugin_op': float(plug.sum()), 'expected_events_op_bb': float(oracle.sum()),
           'null_B_iid_draws': ND, 'mean_oracle_bb': float(oracle.mean()), 'mean_plugin_binomial': float(plug.mean()), 'mean_null_B_frequency': float(freqB.mean()), 'mean_permutation_frequency': float(simperm.mean()),
           'pearson_bb_vs_nullB': float(np.corrcoef(oracle, freqB)[0, 1]), 'pearson_plugin_vs_nullB': float(np.corrcoef(plug, freqB)[0, 1]), 'mean_abs_diff_plugin_nullB': float(np.abs(plug - freqB).mean()), 'mean_abs_diff_bb_nullB': float(np.abs(oracle - freqB).mean()),
           'pearson_bb_vs_permutation': float(np.corrcoef(oracle, simperm)[0, 1]),
           'note': 'the permutation null re-draws the pre-cutoff labels as well, so unit-level agreement with a formula that conditions on the observed pre-cutoff counts is necessarily weaker; the i.i.d. null conditions on them and is matched almost exactly by the plug-in formula, and closely by the Beta-Binomial with a flat prior (which adds prior mass toward 1/2)'}
    print(out, flush=True)
    p = os.path.join(pc.RES, 'issue1', 'stationary_baselines.json'); R = json.load(open(p)); R['issue9_validation_addendum'] = out; json.dump(R, open(p, 'w'), indent=1)
    plt = ac.plt_setup(); fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    a1.scatter(freqB, plug, s=7, alpha=.55, color='#2D6CA2', label=f'plug-in binomial, r = {out["pearson_plugin_vs_nullB"]:.3f}'); a1.scatter(freqB, oracle, s=7, alpha=.55, color='#C9A227', label=f'operational Beta-Binomial, r = {out["pearson_bb_vs_nullB"]:.3f}')
    a1.plot([0, 1], [0, 1], color='#B3402F', lw=.9); a1.set_xlabel('simulated change frequency, i.i.d. stationary null (1,000 draws)', fontsize=7.5); a1.set_ylabel('analytic change probability', fontsize=7.5); a1.legend(fontsize=6.5, frameon=False, loc='lower right'); a1.tick_params(labelsize=7)
    a1.set_title('A. analytic vs Monte Carlo, 1,012 source units', fontsize=8.5, loc='left')
    for m_, col, lab in ((30, '#C9A227', 'm = 30'), (100, '#2D6CA2', 'm = 100'), (300, '#1A1A1A', 'm = 300')):
        xs = np.linspace(0, .95, 40); ys = [pc.bb_change_prob(int(round(60 * (1 + x) / 2)), int(round(60 * (1 - x) / 2)), m_) for x in xs]; a2.plot(xs, ys, color=col, lw=1.3, label=f'n = 60, {lab}')
    for n_, ls in ((20, ':'), (200, '--')):
        xs = np.linspace(0, .95, 40); ys = [pc.bb_change_prob(int(round(n_ * (1 + x) / 2)), int(round(n_ * (1 - x) / 2)), 100) for x in xs]; a2.plot(xs, ys, color='#2D6CA2', lw=1, ls=ls, label=f'n = {n_}, m = 100')
    a2.set_xlabel('pre-cutoff absolute margin $|p_t|$', fontsize=7.5); a2.set_ylabel('P(sign change)', fontsize=7.5); a2.legend(fontsize=6.2, frameon=False); a2.tick_params(labelsize=7); a2.set_title('B. Beta-Binomial change probability', fontsize=8.5, loc='left')
    for ax in (a1, a2):
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=1.5); fig.savefig(os.path.join(pc.G, 'figures_pro', 'fig_analytic_vs_simulation.pdf')); fig.savefig(os.path.join(pc.G, 'figures_pro', 'fig_analytic_vs_simulation.png'), dpi=200); print('DONE')


if __name__ == '__main__':
    main()
