"""NutriMATURE system, step 4: the Compact-7 model and its prospective performance on the uncapped benchmark.

Compact-7: StandardScaler + L2 logistic regression (C = 1, balanced class weights) on seven pre-cutoff variables
(pooled direction p_t, agreement, directional and total record counts, mean and SD of log effect estimates, number of
ratio estimates). Target: later change of the categorical evidence direction (l2_sign_change) on prospective units
(>= 20 post-cutoff records). Leave-one-claim-out: every cutoff of the held-out claim is excluded from fitting;
the hard-decision threshold is chosen on the training claims only. Intervals: claim-clustered percentile bootstrap
(2,000 resamples, seed 3). The output is a ranking score for later change, not a calibrated probability.
Outputs: tables/04_system_performance.{csv,json}, tables/04_system_oof_predictions.csv, figures/04_system_roc_pr.png
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
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import nutrimature as nm
from sklearn.metrics import roc_curve, precision_recall_curve


met = nm.perf


def figure():
    import csv
    plt = nm.plt_setup(); T = {r['sample']: r for r in nm.table('04_system_performance')['rows']}
    P = list(csv.DictReader(open(f'{nm.TABLES}/04_system_oof_predictions.csv'))); y = np.array([int(r['y']) for r in P]); s = np.array([float(r['score']) for r in P])
    r0 = T['Complete uncapped benchmark']
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for a in ax: nm.style(a, 'both')
    fpr, tpr, _ = roc_curve(y, s); prec, rec, _ = precision_recall_curve(y, s)
    ax[0].plot([0, 1], [0, 1], color=nm.AXIS, lw=1, ls='--'); ax[0].plot(fpr, tpr, color=nm.BLUE, lw=2)
    ax[0].text(0.97, 0.05, f"AUROC {r0['auroc']:.3f} {r0['auroc_ci']}", ha='right', color=nm.INK, fontsize=8.5, transform=ax[0].transAxes)
    ax[1].axhline(r0['base_rate'], color=nm.AXIS, lw=1, ls='--'); ax[1].step(rec, prec, where='post', color=nm.BLUE, lw=2)
    ax[1].text(0.97, 0.93, f"AUPRC {r0['auprc']:.3f} {r0['auprc_ci']}", ha='right', color=nm.INK, fontsize=8.5, transform=ax[1].transAxes)
    ax[1].text(0.99, r0['base_rate'] + 0.015, f"event prevalence {r0['base_rate']:.3f}", ha='right', color=nm.INK2, fontsize=7.5, transform=ax[1].get_yaxis_transform())
    ax[0].set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1.02))
    ax[1].set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
    fig.suptitle(f"NutriMATURE Compact-7, leave-one-claim-out: later direction change on {r0['units']:,} prospective units "
                 f"({r0['events']} events in {r0['positive_claims']} claims)", fontsize=10, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94)); nm.save_fig(fig, '04_system_roc_pr')


if __name__ == '__main__':
    if '--figures-only' in sys.argv: figure(); sys.exit(0)
    rows, keys = nm.load('complete'); R, y, g = nm.prospective(rows); X = nm.X_of(R, nm.COMPACT7)
    s, thr = nm.loco_binary(X, y, g)
    nm.save_preds('04_system_oof_predictions', [(r['claim_id'], int(r['cutoff_year'])) for r in R], y, s, {'threshold_train_fold': thr.tolist()})
    srows, _ = nm.load('systematic'); Rs, ys, gs = nm.prospective(srows); ss, _ = nm.loco_binary(nm.X_of(Rs, nm.COMPACT7), ys, gs)
    out = [{'sample': 'Complete uncapped benchmark', **met(y, s, g)}, {'sample': 'Systematic-only claims (hand-added claims removed)', **met(ys, ss, gs)}]
    nm.save_table(out, '04_system_performance', {'model': 'Compact-7: StandardScaler + LogisticRegression(C=1, L2, class_weight=balanced)',
                  'features': nm.COMPACT7, 'target': 'l2_sign_change (later categorical direction change)', 'sample': 'prospective units, n_post_t >= 20',
                  'evaluation': 'leave-one-claim-out; claim-clustered percentile bootstrap, 2000 resamples, seed 3'})
    for r in out: print(r, flush=True)
    figure()
