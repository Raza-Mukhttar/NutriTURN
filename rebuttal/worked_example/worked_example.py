"""Worked example for AqvV C3: one claim from query to event label, under both core annotators.

Reads data_release/02_records/cohort_tables/{claims,associations,units}.csv (labels, PMIDs, years only),
recomputes the endpoint of the paper (Section 3, eq. 1) for every unit, checks the totals against the paper
(1,012 units; 71 primary events, 52 second, 27 shared) and prints the units of one claim.
Usage: python worked_example.py [DATA_RELEASE] [CLAIM_ID]
"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# default: data_release/ beside this folder, or one level higher (this folder inside rebuttal/)
_DR = [p for p in (os.path.join(HERE, '..', 'data_release'), os.path.join(HERE, '..', '..', 'data_release'))
       if os.path.isdir(p)]
DR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('NUTRITURN_DATA_RELEASE', (_DR or ['data_release'])[0])
CLAIM = sys.argv[2] if len(sys.argv) > 2 else 'coffee_cvd'
T = os.path.join(DR, '02_records', 'cohort_tables')
a = pd.read_csv(os.path.join(T, 'associations.csv'), keep_default_na=False)
u = pd.read_csv(os.path.join(T, 'units.csv'))
c = pd.read_csv(os.path.join(T, 'claims.csv'), keep_default_na=False)
LABELS = ['HARMFUL', 'PROTECTIVE', 'NULL', 'UNCLEAR']

def direction(lab, gate):
    h = int((lab == 'HARMFUL').sum()); l = int((lab == 'PROTECTIVE').sum())
    if h + l == 0 or (gate and h + l < 8):
        return 0.0
    return (h - l) / (h + l)

rows = []
for _, r in u.iterrows():
    g = a[a.claim_id == r.claim_id]; pre = g[g.year < r.cutoff]; post = g[g.year >= r.cutoff]
    o = {'claim_id': r.claim_id, 'cutoff': int(r.cutoff), 'n_pre': len(pre), 'n_post': len(post)}
    for k in ['primary', 'second']:
        p = direction(pre['label_' + k], True); q = direction(post['label_' + k], False)
        o[k] = {'pre_counts': {x: int((pre['label_' + k] == x).sum()) for x in LABELS},
                'post_counts': {x: int((post['label_' + k] == x).sum()) for x in LABELS},
                'p_t': round(p, 3), 'p_t_plus': round(q, 3), 'event': int(np.sign(p) != np.sign(q))}
    rows.append(o)

ev1 = sum(o['primary']['event'] for o in rows); ev2 = sum(o['second']['event'] for o in rows)
both = sum(o['primary']['event'] and o['second']['event'] for o in rows)
check = {'units': len(rows), 'events_primary': ev1, 'events_second': ev2, 'shared': both,
         'matches_paper': (len(rows), ev1, ev2, both) == (1012, 71, 52, 27)}
cl = c[c.claim_id == CLAIM].iloc[0]
ex = [o for o in rows if o['claim_id'] == CLAIM]
out = {'check': check, 'claim': {'claim_id': CLAIM, 'exposure': cl.exposure, 'outcome': cl.outcome, 'query': cl.query,
       'associations': int((a.claim_id == CLAIM).sum())}, 'units': ex}
json.dump(out, open(os.path.join(HERE, 'worked_example.json'), 'w'), indent=1)

print('Totals:', check)
print(f"Claim {CLAIM}: exposure '{cl.exposure}', outcome '{cl.outcome}', {out['claim']['associations']} associations")
print('Query:', cl.query)
print('cutoff  n_pre/n_post | primary: H/P before, p_t -> p_t+, event | second: H/P before, p_t -> p_t+, event')
for o in ex:
    P, S = o['primary'], o['second']
    print(f"{o['cutoff']}    {o['n_pre']:>3}/{o['n_post']:<3}   | {P['pre_counts']['HARMFUL']:>3}/{P['pre_counts']['PROTECTIVE']:<3} {P['p_t']:+.3f} -> {P['p_t_plus']:+.3f}  {P['event']} "
          f"| {S['pre_counts']['HARMFUL']:>3}/{S['pre_counts']['PROTECTIVE']:<3} {S['p_t']:+.3f} -> {S['p_t_plus']:+.3f}  {S['event']}")
sys.exit(0 if check['matches_paper'] else 1)
