"""Issue 7, Null B: stationary categorical-voting simulation.

For each claim a stationary distribution over PROTECTIVE / HARMFUL / NULL / UNCLEAR is estimated from
the claim's own label frequencies (all its records; a stationary i.i.d. process carries no temporal
structure, so these marginal frequencies cannot encode the sign-change signal). Labels are then drawn
i.i.d. from that distribution over the observed publication-time structure, and the benchmark,
states and targets are rebuilt and re-evaluated exactly as in Null A. Namespace: stationary_null.

  python stationary_null_simulation.py --n 1000 --defn B
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
import os, sys, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astra_common as ac
import stationary_null_permutation as snp
nm = ac.nm
LAB = np.array([-1, 1, 0, 99])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=1000); ap.add_argument('--defn', default='B'); ap.add_argument('--seed', type=int, default=3); ap.add_argument('--tag', default=''); a = ap.parse_args()
    rows, P_idx, X7_text, codes, obs = snp.setup(a.defn)
    probs = {cid: np.array([(c == l).mean() for l in LAB]) for cid, c in codes.items()}
    rng = np.random.default_rng(a.seed + 1000); draws = []; t0 = time.time()
    for k in range(a.n):
        sim = {cid: LAB[rng.choice(4, size=len(c), p=probs[cid])] for cid, c in codes.items()}
        draws.append(snp.evaluate(rows, P_idx, snp.rebuild(rows, sim, a.defn), a.defn, X7_text))
        if (k + 1) % 25 == 0:
            ok = [d for d in draws if d.get('evaluable')]
            print(f'  sim {k + 1}/{a.n}  null C7 AUROC mean {np.mean([d["compact7_auroc"] for d in ok]):.3f}  2-var {np.mean([d["two_var_auroc"] for d in ok]):.3f}  '
                  f'events mean {np.mean([d["events"] for d in draws]):.1f}  ({(time.time() - t0) / (k + 1):.1f}s/draw)', flush=True)
    out = snp.summarise(obs, draws, f'Null B: stationary categorical voting (agreement {a.defn}, n={a.n}, seed {a.seed + 1000})')
    out['stationary_distribution_source'] = 'each claim\'s own label frequencies over all of its records'
    d = ac.ns('results', 'stationary_null'); ac.save_json({'summary': out, 'draws': draws}, os.path.join(d, f'null_B_simulation_{a.defn}{a.tag}.json'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
