"""Paired within-claim slope differences between the anchor and each pipeline.

R4 reported three slopes separately. The quantity that decides whether the drift is the literature
or the annotator is the DIFFERENCE, resampled on the same claims so the comparison is paired:
anchor minus primary and anchor minus second, fitted on the records each source calls signed.
A difference whose interval covers zero means the annotator's drift is indistinguishable from the
drift already present in the reported effect estimates.
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
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N
from within_claim_label_drift import slopes

AD = _os.path.join(_NT, 'data', 'derived')
SIGNED = ('PROTECTIVE', 'HARMFUL')


def load():
    A = {(r['claim_id'], r['pmid']): r for r in N.assoc()}
    out = []
    for r in csv.DictReader(open(os.path.join(AD, 'numeric_anchor_v2_high_confidence.csv'))):
        a = A.get((r['claim_id'], r['pmid']))
        if a:
            out.append(dict(claim_id=r['claim_id'], year=int(r['year']), anchor=r['anchor'],
                            primary=a['label_primary'], second=a['label_second']))
    return out


def slope_of(rows, src):
    sel = [r for r in rows if r[src] in SIGNED]
    if len(sel) < 50:
        return np.nan
    yr = np.array([r['year'] for r in sel], float)
    sg = np.array([1 if r[src] == 'HARMFUL' else -1 for r in sel], float)
    cl = np.array([r['claim_id'] for r in sel])
    return slopes(yr, sg, cl, True) * 10          # per decade


def main():
    rows = load()
    g = np.array([r['claim_id'] for r in rows])
    cl = np.unique(g); by = {c: np.where(g == c)[0] for c in cl}
    rng = np.random.default_rng(N.SEED)
    print('Paired within-claim slope differences, per decade, on the same anchored records\n')
    out = {}
    pairs = [('anchor', 'primary'), ('anchor', 'second'), ('primary', 'second')]
    obs = {s: slope_of(rows, s) for s in ('anchor', 'primary', 'second')}
    for s, v in obs.items():
        print(f'   {s:8s} slope {v:+.4f} per decade')
    print()
    draws = {p: [] for p in pairs}
    for _ in range(N.BOOT):
        pick = rng.choice(cl, size=len(cl), replace=True)
        ix = np.concatenate([by[c] for c in pick])
        rr = [rows[i] for i in ix]
        s_ = {s: slope_of(rr, s) for s in ('anchor', 'primary', 'second')}
        for a, b in pairs:
            if np.isfinite(s_[a]) and np.isfinite(s_[b]):
                draws[(a, b)].append(s_[a] - s_[b])
    for a, b in pairs:
        v = np.array(draws[(a, b)])
        d = obs[a] - obs[b]
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        out[f'{a}_minus_{b}'] = [float(d), lo, hi, N.established(lo, hi)]
        print(f'   {a} minus {b}: {d:+.4f} per decade {N.fmt_ci(lo, hi, 4)}'
              f"{'  ESTABLISHED' if N.established(lo, hi) else '  not established'}")
    out['slopes_per_decade'] = {k: float(v) for k, v in obs.items()}
    print('\n   reading: an interval covering zero means the pipeline drifts at a rate')
    print('   indistinguishable from the reported effect estimates themselves')
    print('\nwritten', N.save_json(out, 'R4b_anchor_slope_differences.json'))


if __name__ == '__main__':
    main()
