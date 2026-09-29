"""Issue 6: calendar-forward evaluation, as distinct from retrospective claim-held-out backtesting.

For an origin year Y and a fixed horizon H:
  * the calendar cutoff grid is every second year, independent of any claim's last publication year;
  * a training unit (c, t) has t <= Y - H, at least 40 records before t, and at least 10 signed
    records in [t, t+H); its target 1[sgn0(p_window) != sgn0(p_pre)] therefore depends on no record
    published at or after Y;
  * the test units are (c, Y) for every claim with at least 40 records before Y and at least 10
    signed records in [Y, Y+H); their targets use only the window [Y, Y+H), which must lie inside the
    observed data;
  * the models (2-variable; Compact-7 inputs) are fitted once on the training units and applied to
    the test units; nothing is refitted on test outcomes; the same claim may appear in training at
    earlier cutoffs, which is what a calendar deployment would see.
Text-derived inputs are computed from the pre-cutoff records with the frozen parser. The result is a
calendar-time forward test, not a claim-held-out one, and is named as such. Namespace: calendar_forward.

  python calendar_forward_eval.py --defn B
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
import os, sys, json, argparse, collections, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astra_common as ac
nm, nf = ac.nm, ac.nf
HORIZONS, MIN_WIN, MIN_PRE, STEP = (3, 5, 10), 10, 40, 2
ORIGINS = [2004, 2007, 2010, 2013, 2016, 2019, 2022]
LAST_YEAR = 2026


def text_feats_cache(cid):
    """Per record: (year, [(y, se)...]) from the frozen effects() parser, so cutoff aggregation is cheap."""
    recs = json.load(open(f'{nm.RECORDS}/{cid}.json'))['records']; Dj = {d['pmid'] for d in json.load(open(f'{nm.DIRECTIONS}/{cid}.json'))['directions']}
    return [(int(r['year']), nf.effects(r)) for r in recs if r['pmid'] in Dj and r.get('year')]


def unit(cid, t, H, yrs, codes, tf, defn):
    m = yrs < t
    if m.sum() < MIN_PRE: return None
    f = ac.direction_feats(yrs[m], codes[m], t, defn)
    w = codes[(yrs >= t) & (yrs < t + H)]; ws = w[(w != 99) & (w != 0)]
    if len(ws) < MIN_WIN: return None
    eff = [e for yr, es in tf if yr < t for e in es]
    f.update(claim_id=cid, cutoff_year=t, horizon=H, n_pre_t=int(m.sum()), n_rr=len(eff),
             val_mean=float(np.mean([e[0] for e in eff])) if len(eff) >= 3 else 0.0, val_sd=float(np.std([e[0] for e in eff])) if len(eff) >= 3 else 0.0,
             n_window_signed=int(len(ws)), y=int(ac.sgn0(float(ws.mean())) != ac.sgn0(f['pooled_direction'])))
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--defn', default='B'); a = ap.parse_args()
    Q = json.load(open(nm.QUERIES)); cids = list(Q)
    streams = {c: ac.stream(c) for c in cids}; tf = {c: text_feats_cache(c) for c in cids}
    print('caches built', flush=True)
    units, feas, res, tab = [], [], {'protocol': __doc__, 'definition': a.defn, 'origins': {}}, []
    X7keys = nm.COMPACT7
    for H in HORIZONS:
        pooled = {'y': [], 's2': [], 's7': [], 'su': [], 'g': [], 'origin': []}
        for Y in ORIGINS:
            if Y + H > LAST_YEAR + 1: feas.append({'horizon': H, 'origin': Y, 'status': 'window not fully observable'}); continue
            train, test = [], []
            for c in cids:
                yrs, codes, _ = streams[c]
                for t in range(1975 + (Y % STEP), Y - H + 1, STEP):
                    u = unit(c, t, H, yrs, codes, tf[c], a.defn)
                    if u: u['role'] = 'train'; u['origin'] = Y; train.append(u)
                u = unit(c, Y, H, yrs, codes, tf[c], a.defn)
                if u: u['role'] = 'test'; u['origin'] = Y; test.append(u)
            units += train + test
            ytr = np.array([u['y'] for u in train]); yte = np.array([u['y'] for u in test])
            row = {'horizon': H, 'origin': Y, 'train_units': len(train), 'train_claims': len({u['claim_id'] for u in train}), 'train_events': int(ytr.sum()),
                   'test_units': len(test), 'test_claims': len({u['claim_id'] for u in test}), 'test_events': int(yte.sum())}
            if len(test) < 20 or yte.sum() < 3 or ytr.sum() < 10 or len(np.unique(ytr)) < 2:
                row['status'] = 'insufficient support (need >= 20 test units, >= 3 test events, >= 10 training events)'; feas.append(row); continue
            def X(U, keys): return np.array([[u[k] for k in keys] for u in U], float)
            fits = {}
            for name, keys in (('two_var', ['pooled_direction', 'agreement']), ('compact7', X7keys)):
                sc = StandardScaler().fit(X(train, keys)); m = LogisticRegression(max_iter=3000, class_weight='balanced').fit(sc.transform(X(train, keys)), ytr)
                fits[name] = m.predict_proba(sc.transform(X(test, keys)))[:, 1]
            su = 1 - np.abs(np.array([u['pooled_direction'] for u in test])); g = np.array([u['claim_id'] for u in test])
            e = {**row, 'status': 'evaluated', 'two_var_auroc': float(roc_auc_score(yte, fits['two_var'])), 'two_var_auprc': float(average_precision_score(yte, fits['two_var'])),
                 'compact7_auroc': float(roc_auc_score(yte, fits['compact7'])), 'compact7_auprc': float(average_precision_score(yte, fits['compact7'])),
                 'unfitted_1_minus_abs_p_auroc': float(roc_auc_score(yte, su)), 'test_prevalence': float(yte.mean())}
            lo, hi, _ = nm.claim_boot(g, lambda i: roc_auc_score(yte[i], fits['two_var'][i])); e['two_var_auroc_ci'] = nm.fmt_ci(lo, hi)
            lo, hi, _ = nm.claim_boot(g, lambda i: roc_auc_score(yte[i], fits['compact7'][i])); e['compact7_auroc_ci'] = nm.fmt_ci(lo, hi)
            feas.append(e); tab.append(e); res['origins'][f'H{H}_Y{Y}'] = e
            for k, v in (('y', yte), ('s2', fits['two_var']), ('s7', fits['compact7']), ('su', su), ('g', g), ('origin', np.full(len(test), Y))): pooled[k] += list(v)
            print(f"  H={H:2d} Y={Y}: train {len(train)} ({int(ytr.sum())} ev)  test {len(test)} ({int(yte.sum())} ev)  2-var {e['two_var_auroc']:.3f} {e['two_var_auroc_ci']}  C7 {e['compact7_auroc']:.3f}  1-|p| {e['unfitted_1_minus_abs_p_auroc']:.3f}", flush=True)
        if pooled['y'] and 0 < sum(pooled['y']) < len(pooled['y']):
            y = np.array(pooled['y']); g = np.array(pooled['g']); s2 = np.array(pooled['s2']); s7 = np.array(pooled['s7']); su = np.array(pooled['su'])
            p = {'horizon': H, 'origin': 'pooled over origins', 'test_units': len(y), 'test_claims': len(set(g)), 'test_events': int(y.sum()), 'status': 'evaluated',
                 'two_var_auroc': float(roc_auc_score(y, s2)), 'two_var_auprc': float(average_precision_score(y, s2)), 'compact7_auroc': float(roc_auc_score(y, s7)),
                 'compact7_auprc': float(average_precision_score(y, s7)), 'unfitted_1_minus_abs_p_auroc': float(roc_auc_score(y, su)), 'test_prevalence': float(y.mean())}
            lo, hi, _ = nm.claim_boot(g, lambda i: roc_auc_score(y[i], s2[i])); p['two_var_auroc_ci'] = nm.fmt_ci(lo, hi)
            lo, hi, _ = nm.claim_boot(g, lambda i: roc_auc_score(y[i], s7[i])); p['compact7_auroc_ci'] = nm.fmt_ci(lo, hi)
            lo, hi, _ = nm.claim_boot(g, lambda i: average_precision_score(y[i], s2[i])); p['two_var_auprc_ci'] = nm.fmt_ci(lo, hi)
            p['paired_two_var_minus_1_minus_abs_p_auroc'] = ac.paired(y, s2, su, g, 'auroc'); p['paired_two_var_minus_compact7_auroc'] = ac.paired(y, s2, s7, g, 'auroc')
            res[f'pooled_H{H}'] = p; tab.append(p)
            print(f"  H={H:2d} POOLED: {len(y)} test units, {int(y.sum())} events, 2-var {p['two_var_auroc']:.3f} {p['two_var_auroc_ci']}  C7 {p['compact7_auroc']:.3f}  1-|p| {p['unfitted_1_minus_abs_p_auroc']:.3f}", flush=True)
    res['feasibility'] = feas
    ac.save_frame(units, os.path.join(ac.DATA, 'calendar_forward_units.parquet'))
    ac.save_json(res, os.path.join(ac.RESULTS, 'calendar_forward_results.json')); ac.save_csv(tab, os.path.join(ac.TABLES, 'calendar_forward_results.csv'))
    ac.save_csv(feas, os.path.join(ac.ns('results', 'calendar_forward'), 'feasibility_by_origin.csv'))
    plt = ac.plt_setup(); fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, H in zip(axes, HORIZONS):
        E = [t for t in tab if t['horizon'] == H and t['origin'] != 'pooled over origins']
        if not E: ax.set_title(f'H = {H} y: not evaluable', fontsize=9); continue
        xs = [t['origin'] for t in E]
        ax.plot(xs, [t['two_var_auroc'] for t in E], 'o-', color='#D97528', label='2-variable'); ax.plot(xs, [t['compact7_auroc'] for t in E], 's-', color='#2D6CA2', label='Compact-7')
        ax.plot(xs, [t['unfitted_1_minus_abs_p_auroc'] for t in E], 'x--', color='#888', label='1-|p_t|')
        for t in E: ax.text(t['origin'], .42, f"{t['test_events']}/{t['test_units']}", ha='center', fontsize=6, color='#666')
        ax.axhline(.5, color='#bbb', lw=.7); ax.set_ylim(.35, 1); ax.set_title(f'H = {H} years', fontsize=9); ax.set_xlabel('origin year Y'); ax.grid(alpha=.25, lw=.5)
        ax.set_xticks(xs); ax.set_xticklabels([str(int(x)) for x in xs], fontsize=7)
        pl = next((t for t in tab if t['horizon'] == H and t['origin'] == 'pooled over origins'), None)
        if pl: ax.axhline(pl['two_var_auroc'], color='#D97528', lw=.8, ls=':'); ax.axhline(pl['compact7_auroc'], color='#2D6CA2', lw=.8, ls=':')
    axes[0].set_ylabel('test AUROC at origin'); axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle('Calendar-forward evaluation: fitted on outcomes observable by Y, tested at Y (events/units shown)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(ac.FIGURES, 'calendar_forward_performance.pdf')); fig.savefig(os.path.join(ac.FIGURES, 'calendar_forward_performance.png'), dpi=200)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
