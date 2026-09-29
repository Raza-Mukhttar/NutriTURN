"""Combine the Null A and Null B draws into results/stationary_null_results.json,
tables/stationary_null_results.csv and figures/stationary_null_distributions.pdf."""

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
import astra_common as ac
KEYS = [('compact7_auroc', 'Compact-7 AUROC'), ('compact7_auprc', 'Compact-7 AUPRC'), ('two_var_auroc', '2-variable AUROC'), ('two_var_auprc', '2-variable AUPRC'),
        ('unfitted_1_minus_abs_p_auroc', '1-|p_t| AUROC'), ('prevalence', 'event prevalence'), ('events', 'events'), ('stable_rate', 'STABLE later-change rate'), ('still_forming_rate', 'STILL-FORMING later-change rate')]


def main():
    import stationary_null_permutation as snp
    d = ac.ns('results', 'stationary_null'); res, tab, dist = {}, [], {}
    for prefix, name in (('null_A_permutation_', 'Null A: within-claim permutation'), ('null_B_simulation_', 'Null B: stationary voting')):
        files = sorted(glob.glob(os.path.join(d, prefix + '*.json')))
        if not files: continue
        draws, obs, labels = [], None, []
        for f in files:
            J = json.load(open(f)); draws += J['draws']; obs = J['summary']['observed']; labels.append(J['summary']['label'])
        S = snp.summarise(obs, draws, name + ' (merged over ' + str(len(files)) + ' chunk files with distinct seeds: ' + '; '.join(labels) + ')')
        S['chunk_files'] = [os.path.basename(f) for f in files]
        res[name] = S; dist[name] = draws
        for k, lab in KEYS:
            if k not in S: continue
            s = S[k]; tab.append({'null': name, 'metric': lab, 'observed': round(s['observed'], 4), 'null_mean': round(s['null_mean'], 4), 'null_sd': round(s['null_sd'], 4),
                                  'null_95': f"[{s['null_2.5']:.3f}, {s['null_97.5']:.3f}]", 'empirical_p_upper': round(s['empirical_p_upper'], 4), 'empirical_p_lower': round(s['empirical_p_lower'], 4),
                                  'draws': S['draws_evaluable']})
    ac.save_json({'nulls': res, 'note': 'empirical p_upper = (#draws >= observed + 1)/(n + 1); rejection of a stationary null is not evidence of causal temporal dynamics'},
                 os.path.join(ac.RESULTS, 'stationary_null_results.json'))
    ac.save_csv(tab, os.path.join(ac.TABLES, 'stationary_null_results.csv'))
    plt = ac.plt_setup(); fig, axes = plt.subplots(2, 3, figsize=(10, 5.4))
    for i, (name, draws) in enumerate(dist.items()):
        ok = [x for x in draws if x.get('evaluable')]
        for j, (k, lab) in enumerate([('compact7_auroc', 'Compact-7 AUROC'), ('two_var_auroc', '2-variable AUROC'), ('prevalence', 'event prevalence')]):
            ax = axes[i, j]; v = np.array([x[k] for x in ok]); o = res[name][k]['observed']
            ax.hist(v, bins=40, color='#999', alpha=.9); ax.axvline(o, color='#B3402F', lw=1.6); ax.set_title(f'{name}\n{lab}', fontsize=8)
            ax.text(o, ax.get_ylim()[1] * .9, f' observed {o:.3f}', color='#B3402F', fontsize=7, ha='left' if o < v.mean() else 'right')
            ax.set_xlabel(lab, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(ac.FIGURES, 'stationary_null_distributions.pdf')); fig.savefig(os.path.join(ac.FIGURES, 'stationary_null_distributions.png'), dpi=180)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
