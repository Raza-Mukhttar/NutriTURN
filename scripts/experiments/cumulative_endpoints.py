"""Issue 3: the subsequent-evidence endpoint (the paper's target) beside a cumulative updated-direction
endpoint and a NULL-sensitive three-state endpoint, on the same units.

  R_sub(c,t) = 1[sgn0(p_post) != sgn0(p_pre)]      p_post over post-cutoff signed records only
  R_cum(c,t) = 1[sgn0(p_cum)  != sgn0(p_pre)]      p_cum over pre and post signed records together
  R3_delta   = 1[s(p_post) != s(p_pre)],  s(p) = 0 if |p| < delta else sgn(p)   (registers movement to/from
               no predominant direction), reported for delta = 0.10 and 0.20, in subsequent and cumulative forms
Fixed-horizon variants use the window [t, t+H) with at least 10 signed records and a literature that
extends to t+H-1. Namespace: cumulative_target.
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
import os, sys, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astra_common as ac
nm = ac.nm
HORIZONS, MIN_WIN = (3, 5, 10), 10


def s3(p, delta): return 0 if abs(p) < delta else ac.sgn0(p)


def main():
    rows, keys = ac.load_units(); P, y_ship, g = ac.prospective(rows)
    print(f'{len(rows)} units, {len(P)} prospective', flush=True)
    T = []
    for r in rows:
        cid, cut = r['claim_id'], int(r['cutoff_year']); py, pc, qy, qc = ac.split(cid, cut)
        pre_s = pc[(pc != 99) & (pc != 0)]; post_s = qc[(qc != 99) & (qc != 0)]
        p_pre = float(pre_s.mean()) if len(pre_s) >= 1 else 0.0
        p_pre_bench = r['pooled_direction']                      # the benchmark zeroes p_pre when n_signed < 8
        p_post = float(post_s.mean()) if len(post_s) else 0.0
        assert abs(p_post - r['post_pooled_direction']) < 1e-9 and (len(pre_s) < 8 or abs(p_pre - p_pre_bench) < 1e-9)
        cum = np.concatenate([pre_s, post_s]); p_cum = float(cum.mean()) if len(cum) else 0.0
        t = {'claim_id': cid, 'cutoff_year': cut, 'n_post_t': r['n_post_t'], 'prospective': int(r['n_post_t'] >= nm.POST20),
             'n_signed_pre': len(pre_s), 'n_signed_post': len(post_s), 'p_pre': round(p_pre_bench, 4), 'p_post': round(p_post, 4), 'p_cum': round(p_cum, 4),
             'R_sub': int(ac.sgn0(p_post) != ac.sgn0(p_pre_bench)), 'R_cum': int(ac.sgn0(p_cum) != ac.sgn0(p_pre_bench))}
        assert t['R_sub'] == r['y_sub']
        for d in (0.10, 0.20):
            t[f'R3_sub_{d:.2f}'] = int(s3(p_post, d) != s3(p_pre_bench, d)); t[f'R3_cum_{d:.2f}'] = int(s3(p_cum, d) != s3(p_pre_bench, d))
        last = int(ac.stream(cid)[0].max())
        for H in HORIZONS:
            w = qc[(qy >= cut) & (qy < cut + H)]; ws = w[(w != 99) & (w != 0)]
            ok = len(ws) >= MIN_WIN and last >= cut + H - 1
            t[f'eligible_{H}y'] = int(ok)
            if ok:
                pw = float(ws.mean()); pcw = float(np.concatenate([pre_s, ws]).mean())
                t[f'R_sub_{H}y'] = int(ac.sgn0(pw) != ac.sgn0(p_pre_bench)); t[f'R_cum_{H}y'] = int(ac.sgn0(pcw) != ac.sgn0(p_pre_bench))
            else:
                t[f'R_sub_{H}y'] = ''; t[f'R_cum_{H}y'] = ''
        T.append(t)
    out = ac.save_frame(T, os.path.join(ac.DATA, 'cumulative_targets.parquet'))

    # ---- evaluate every endpoint on the same prospective units
    idx = {(t['claim_id'], t['cutoff_year']): t for t in T}
    TP = [idx[(r['claim_id'], int(r['cutoff_year']))] for r in P]
    X7 = nm.X_of(P, nm.COMPACT7); pt = np.array([r['pooled_direction'] for r in P]); ag = np.array([r['agreement'] for r in P])
    X2 = np.column_stack([pt, ag]); unf = 1 - np.abs(pt)
    res, tab = {}, []

    def evaluate(name, yv, mask=None, note=''):
        m = np.ones(len(P), bool) if mask is None else mask
        yy = np.asarray(yv)[m].astype(int); gg = g[m]
        if yy.sum() == 0 or yy.sum() == len(yy):
            res[name] = {'units': int(m.sum()), 'events': int(yy.sum()), 'note': 'single class; not evaluable'}; return
        s7, t7 = ac.loco(X7[m], yy, gg); s2, t2 = ac.loco(X2[m], yy, gg)
        e = {'note': note, 'units': int(m.sum()), 'events': int(yy.sum()), 'positive_claims': int(len(set(gg[yy == 1]))), 'prevalence': round(float(yy.mean()), 4),
             'compact7': ac.perf(yy, s7, gg, t7), 'two_variable': ac.perf(yy, s2, gg, t2), 'one_minus_abs_p_t': nm.perf(yy, unf[m], gg),
             'compact7_claim_level': ac.claim_level(yy, s7, gg), 'two_variable_claim_level': ac.claim_level(yy, s2, gg)}
        res[name] = e
        tab.append({'endpoint': name, 'units': e['units'], 'events': e['events'], 'positive_claims': e['positive_claims'], 'prevalence': e['prevalence'],
                    'compact7_auroc': e['compact7']['auroc'], 'compact7_auroc_ci': e['compact7']['auroc_ci'], 'compact7_auprc': e['compact7']['auprc'], 'compact7_auprc_ci': e['compact7']['auprc_ci'],
                    'two_var_auroc': e['two_variable']['auroc'], 'two_var_auroc_ci': e['two_variable']['auroc_ci'], 'two_var_auprc': e['two_variable']['auprc'],
                    'unfitted_1_minus_abs_p_auroc': e['one_minus_abs_p_t']['auroc'], 'claim_level_compact7_auroc': e['compact7_claim_level']['auroc'],
                    'claim_level_compact7_auroc_ci': e['compact7_claim_level']['auroc_ci'], 'note': note})
        print(f"  {name:18s} n={e['units']:5d} ev={e['events']:3d} C7 {e['compact7']['auroc']} {e['compact7']['auroc_ci']}  2v {e['two_variable']['auroc']}  1-|p| {e['one_minus_abs_p_t']['auroc']}", flush=True)

    evaluate('R_sub (primary)', [t['R_sub'] for t in TP], note='subsequent-evidence directional change; the paper target')
    evaluate('R_cum', [t['R_cum'] for t in TP], note='cumulative updated-direction change: pre and post evidence pooled')
    for d in (0.10, 0.20):
        evaluate(f'R3_sub delta={d:.2f}', [t[f'R3_sub_{d:.2f}'] for t in TP], note='three-state: transitions to or from |p|<delta count')
        evaluate(f'R3_cum delta={d:.2f}', [t[f'R3_cum_{d:.2f}'] for t in TP], note='three-state, cumulative')
    for H in HORIZONS:
        m = np.array([t[f'eligible_{H}y'] == 1 for t in TP])
        evaluate(f'R_sub {H}y', [t[f'R_sub_{H}y'] if t[f'R_sub_{H}y'] != '' else 0 for t in TP], m, note=f'window [t,t+{H}), >= {MIN_WIN} signed records in the window')
        evaluate(f'R_cum {H}y', [t[f'R_cum_{H}y'] if t[f'R_cum_{H}y'] != '' else 0 for t in TP], m, note=f'cumulative through t+{H}-1')
    # overlap and examples
    ys = np.array([t['R_sub'] for t in TP]).astype(bool); yc = np.array([t['R_cum'] for t in TP]).astype(bool)
    ov = {'both': int((ys & yc).sum()), 'sub_only': int((ys & ~yc).sum()), 'cum_only': int((~ys & yc).sum()), 'neither': int((~ys & ~yc).sum()),
          'jaccard': round(float((ys & yc).sum() / max((ys | yc).sum(), 1)), 4)}
    ex_sub_only = sorted([t for t in TP if t['R_sub'] == 1 and t['R_cum'] == 0], key=lambda t: -abs(t['p_pre']))
    ex_cum_only = [t for t in TP if t['R_sub'] == 0 and t['R_cum'] == 1]
    res['event_set_overlap_R_sub_vs_R_cum'] = ov; res['examples_sub_change_but_cum_unchanged'] = ex_sub_only[:15]; res['examples_cum_change_but_sub_unchanged'] = ex_cum_only[:15]
    res['n_examples'] = {'sub_only': len(ex_sub_only), 'cum_only': len(ex_cum_only)}
    ac.save_json({'results': res, 'targets_file': os.path.relpath(out, ac.ROOT), 'definitions': __doc__}, os.path.join(ac.RESULTS, 'cumulative_endpoint_results.json'))
    ac.save_csv(tab, os.path.join(ac.TABLES, 'cumulative_endpoint_results.csv'))
    ac.save_csv(ex_sub_only, os.path.join(ac.ns('data', 'cumulative_target'), 'examples_sub_only.csv'))
    plt = ac.plt_setup(); fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
    names = [t['endpoint'] for t in tab]; v7 = [t['compact7_auroc'] for t in tab]; v2 = [t['two_var_auroc'] for t in tab]; vu = [t['unfitted_1_minus_abs_p_auroc'] for t in tab]
    xs = np.arange(len(names)); W = .27
    ax[0].bar(xs - W, v7, W, label='Compact-7', color='#2D6CA2'); ax[0].bar(xs, v2, W, label='2-variable', color='#D97528'); ax[0].bar(xs + W, vu, W, label='1-|p_t|', color='#999999')
    ax[0].set_xticks(xs); ax[0].set_xticklabels(names, rotation=60, ha='right', fontsize=6.5); ax[0].set_ylim(.5, 1); ax[0].set_ylabel('LOCO AUROC'); ax[0].legend(fontsize=7, frameon=False)
    ax[1].bar(['both', 'subsequent only', 'cumulative only'], [ov['both'], ov['sub_only'], ov['cum_only']], color=['#1A1A1A', '#2D6CA2', '#D97528'])
    ax[1].set_ylabel('prospective units'); ax[1].set_title('Event sets of R_sub and R_cum', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(ac.FIGURES, 'subsequent_vs_cumulative_target.pdf')); fig.savefig(os.path.join(ac.FIGURES, 'subsequent_vs_cumulative_target.png'), dpi=200)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
