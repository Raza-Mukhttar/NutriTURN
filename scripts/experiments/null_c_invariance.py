"""R2 (their E2, part A): how much of the benchmark can Null C actually alter?

Null C permutes labels inside each claim-by-block period. A unit can only be affected by blocks
that contain records on BOTH sides of its cutoff: a block lying wholly before or wholly after the
cutoff contributes the same counts to the pre and post windows under every permutation.

Reported separately, as their specification requires:
  structural invariance  no occupied block straddles the cutoff, or every straddling block holds a
                         single label, so the permitted permutations cannot alter the pre/post
                         count vectors at all
  straddling evidence    how much of each unit's signed evidence sits in straddling blocks

Observed invariance across sampled draws is a different and weaker statement, and is measured by
the draw runs rather than here.

  python analyses/nfx_R2_invariance.py
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
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

GRID = [('C10-original', 10, 0), ('C10-shifted', 10, 5), ('C5', 5, 0), ('C20', 20, 0)]


def main():
    out = {'configurations': []}
    print('R2 Null C invariance audit\n')
    print(f"{'annotation':12s} {'config':14s} {'units':>6s} {'structurally invariant':>23s} "
          f"{'of which events':>16s} {'median straddling share':>24s}")
    for ann in ('primary', 'second', 'consensus'):
        rows = N.unit_table(ann)
        S = N.streams(ann)
        for name, width, offset in GRID:
            inv = 0; inv_ev = 0; shares = []
            per_unit = []
            for r in rows:
                yrs, codes = S[r['claim_id']]
                t = r['cutoff']
                blk = ((yrs - offset) // width) * width
                # Null C permutes EVERY label type inside a block, UNCLEAR included, so an
                # UNCLEAR sitting in a straddling block can exchange with a signed label and move
                # signed evidence across the cutoff. The invariance condition therefore ranges over
                # all records, not only the resolved ones.
                allr = np.ones(len(codes), bool)
                straddle = set()
                for b in np.unique(blk):
                    m = (blk == b) & allr
                    if (yrs[m] < t).any() and (yrs[m] >= t).any():
                        straddle.add(b)
                effective = []
                for b in straddle:
                    m = (blk == b) & allr
                    if len(np.unique(codes[m])) > 1:      # a single-label block cannot change counts
                        effective.append(b)
                res = codes != 99
                sig = res & (codes != 0)
                n_sig = int(sig.sum())
                in_str = int(sum(((blk == b) & sig).sum() for b in effective))
                sh = in_str / n_sig if n_sig else 0.0
                shares.append(sh)
                is_inv = len(effective) == 0
                inv += is_inv; inv_ev += is_inv and r['y']
                per_unit.append(dict(claim_id=r['claim_id'], cutoff=t,
                                     straddling_blocks=len(effective),
                                     signed_in_straddling=in_str,
                                     share_of_signed=round(sh, 4),
                                     structurally_invariant=int(is_inv), event=r['y']))
            sh = np.array(shares)
            rec = dict(annotation=ann, config=name, width=width, offset=offset,
                       units=len(rows), structurally_invariant=int(inv),
                       invariant_events=int(inv_ev),
                       median_straddling_share=float(np.median(sh)),
                       mean_straddling_share=float(sh.mean()))
            out['configurations'].append(rec)
            if ann == 'primary' and name == 'C10-original':
                N.save_csv(per_unit, 'R2_invariance_units_primary_C10.csv')
            print(f"{ann:12s} {name:14s} {len(rows):6d} {inv:10d} ({100*inv/len(rows):5.1f}%)   "
                  f"{inv_ev:10d}       {np.median(sh):24.3f}")
    print('\nreading: a structurally invariant unit cannot have its pre/post counts changed by the')
    print('control, so any reproduction of its endpoint is arithmetic, not evidence about drift.')
    print('\nwritten', N.save_json(out, 'R2_nullc_invariance.json'))


if __name__ == '__main__':
    main()
