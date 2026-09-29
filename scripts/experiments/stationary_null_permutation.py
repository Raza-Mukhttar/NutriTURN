"""Issue 7, Null A: within-claim temporal permutation of the direction labels.

For each permutation, every claim's direction labels (PROTECTIVE / HARMFUL / NULL / UNCLEAR) are
shuffled across its records while the publication years, the record count, the overall label counts
and the cutoff schedule are kept. The direction-derived variables, the state rule and the target are
rebuilt for every benchmark unit; the text-derived Compact-7 inputs (n_pre_t, val_mean, val_sd, n_rr)
do not depend on the labels and are kept. The leave-one-claim-out pipeline is then rerun for the
2-variable model and for Compact-7. The observed pipeline is run first through the same code and
asserted to reproduce the shipped result. Namespace: stationary_null.

  python stationary_null_permutation.py --n 1000 --defn B
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
import os, sys, json, argparse, time, collections
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astra_common as ac
nm = ac.nm


def unit_index(rows):
    return [(r['claim_id'], int(r['cutoff_year'])) for r in rows]


def rebuild(rows, codes_by_claim, defn):
    """Direction features, states and targets for every unit under a given per-claim code array."""
    F = []
    for r in rows:
        cid, cut = r['claim_id'], int(r['cutoff_year']); yrs = ac.stream(cid)[0]; c = codes_by_claim[cid]
        m = yrs < cut; f = ac.direction_feats(yrs[m], c[m], cut, defn); f['state'] = ac.state_of(f)
        post = c[~m]; ps = post[(post != 99) & (post != 0)]; p_post = float(ps.mean()) if len(ps) else 0.0
        f['y'] = int(ac.sgn0(p_post) != ac.sgn0(f['pooled_direction'])); F.append(f)
    return F


def evaluate(rows, P_idx, F, defn, X7_text, y_state=True):
    y = np.array([F[i]['y'] for i in P_idx]); g = np.array([rows[i]['claim_id'] for i in P_idx])
    if y.sum() < 2 or y.sum() > len(y) - 2: return {'events': int(y.sum()), 'evaluable': False}
    pt = np.array([F[i]['pooled_direction'] for i in P_idx]); ag = np.array([F[i]['agreement'] for i in P_idx])
    s2 = ac.loco_scores(np.column_stack([pt, ag]), y, g)   # identical fits to ac.loco; threshold search skipped (AUROC/AUPRC only)
    X7 = X7_text.copy(); X7[:, nm.COMPACT7.index('pooled_direction')] = pt; X7[:, nm.COMPACT7.index('agreement')] = ag
    X7[:, nm.COMPACT7.index('n_directional')] = [F[i]['n_directional'] for i in P_idx]
    s7 = ac.loco_scores(X7, y, g)
    out = {'evaluable': True, 'events': int(y.sum()), 'prevalence': float(y.mean()), 'positive_claims': int(len(set(g[y == 1]))),
           'two_var_auroc': float(roc_auc_score(y, s2)), 'two_var_auprc': float(average_precision_score(y, s2)),
           'compact7_auroc': float(roc_auc_score(y, s7)), 'compact7_auprc': float(average_precision_score(y, s7)),
           'unfitted_1_minus_abs_p_auroc': float(roc_auc_score(y, 1 - np.abs(pt)))}
    if y_state:
        st = np.array([F[i]['state'] for i in P_idx])
        for s in ('stable', 'still_forming'):
            k = st == s; out[f'{s}_units'] = int(k.sum()); out[f'{s}_rate'] = float(y[k].mean()) if k.sum() else float('nan')
    return out


def setup(defn):
    rows, keys = ac.load_units(); P, yP, gP = ac.prospective(rows)
    P_idx = [i for i, r in enumerate(rows) if r['n_post_t'] >= nm.POST20]
    X7_text = nm.X_of([rows[i] for i in P_idx], nm.COMPACT7)
    codes = {cid: ac.stream(cid)[1].copy() for cid in sorted({r['claim_id'] for r in rows})}   # sorted: dict order (and so every seeded draw) is process-independent
    obs = evaluate(rows, P_idx, rebuild(rows, codes, defn), defn, X7_text)
    return rows, P_idx, X7_text, codes, obs


def summarise(obs, draws, label):
    keys = ['two_var_auroc', 'two_var_auprc', 'compact7_auroc', 'compact7_auprc', 'unfitted_1_minus_abs_p_auroc', 'prevalence', 'events', 'stable_rate', 'still_forming_rate']
    ok = [d for d in draws if d.get('evaluable')]
    out = {'label': label, 'draws_requested': len(draws), 'draws_evaluable': len(ok), 'observed': obs}
    for k in keys:
        v = np.array([d[k] for d in ok if k in d and d[k] == d[k]]); o = obs.get(k)
        if len(v) == 0 or o is None: continue
        out[k] = {'null_mean': float(v.mean()), 'null_sd': float(v.std()), 'null_2.5': float(np.percentile(v, 2.5)), 'null_97.5': float(np.percentile(v, 97.5)),
                  'observed': float(o), 'empirical_p_upper': float((np.sum(v >= o) + 1) / (len(v) + 1)), 'empirical_p_lower': float((np.sum(v <= o) + 1) / (len(v) + 1))}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=1000); ap.add_argument('--defn', default='B'); ap.add_argument('--seed', type=int, default=3); ap.add_argument('--tag', default=''); a = ap.parse_args()
    rows, P_idx, X7_text, codes, obs = setup(a.defn)
    print(f"observed: C7 AUROC {obs['compact7_auroc']:.4f}  2-var {obs['two_var_auroc']:.4f}  events {obs['events']}", flush=True)
    if a.defn == 'A':
        assert abs(obs['compact7_auroc'] - 0.9456) < 5e-4, obs['compact7_auroc']   # the shipped result under the same pipeline
    rng = np.random.default_rng(a.seed); draws = []; t0 = time.time()
    for k in range(a.n):
        perm = {cid: c[rng.permutation(len(c))] for cid, c in codes.items()}
        draws.append(evaluate(rows, P_idx, rebuild(rows, perm, a.defn), a.defn, X7_text))
        if (k + 1) % 25 == 0:
            ok = [d for d in draws if d.get('evaluable')]
            print(f'  perm {k + 1}/{a.n}  null C7 AUROC mean {np.mean([d["compact7_auroc"] for d in ok]):.3f}  2-var {np.mean([d["two_var_auroc"] for d in ok]):.3f}  '
                  f'events mean {np.mean([d["events"] for d in draws]):.1f}  ({(time.time() - t0) / (k + 1):.1f}s/perm)', flush=True)
    out = summarise(obs, draws, f'Null A: within-claim temporal permutation (agreement {a.defn}, n={a.n}, seed {a.seed})')
    d = ac.ns('results', 'stationary_null'); ac.save_json({'summary': out, 'draws': draws}, os.path.join(d, f'null_A_permutation_{a.defn}{a.tag}.json'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
