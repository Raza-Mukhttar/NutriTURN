"""R1: fresh-directory smoke test of the advertised rebuild levels using only released per-unit / association-level files.
Level 1: recompute reported AUROC/AUPRC from released per-unit prediction+outcome files (no access to text or code beyond sklearn).
Level 2: rebuild counts, pooled direction (with the 8-record gate), operational endpoint and the realised-count stationary probability
from association-level identifiers, years and labels; compare with the archived unit table."""

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
import os, sys, csv, json, shutil, tempfile, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
G = _NT; R = os.path.join(G, 'results'); SNAP = _os.path.join(_NT, 'data', 'protocol_snapshot')
tmp = tempfile.mkdtemp(prefix='nutri_smoke_'); out = {'fresh_directory': tmp}
# ---- level 1: metrics from released per-unit files
for f in ('issue1/source_scores.csv', 'issue6/unit_scores_qwen.csv', 'issue5/pooled_scores_H5.csv'):
    shutil.copy(os.path.join(R, f), tmp)
rows = list(csv.DictReader(open(os.path.join(tmp, 'source_scores.csv')))); y = np.array([int(r['y']) for r in rows])
lvl1 = {k: round(float(roc_auc_score(y, [float(r[k]) for r in rows])), 4) for k in ('margin', 'bb_deployable', 'two_var')}
ref = json.load(open(os.path.join(R, 'issue1', 'stationary_baselines.json')))['source']
out['level1_metric_recomputation'] = {'recomputed': lvl1, 'reported': {k: round(ref[k]['auroc'], 4) for k in lvl1}, 'match': all(abs(lvl1[k] - ref[k]['auroc']) < 5e-4 for k in lvl1)}
rq = list(csv.DictReader(open(os.path.join(tmp, 'unit_scores_qwen.csv')))); yq = np.array([int(r['y']) for r in rq]); refq = json.load(open(os.path.join(R, 'issue6', 'extractor_agreement.json')))['unit_level']['qwen']
out['level1_second_pipeline'] = {'recomputed_two_var': round(float(roc_auc_score(yq, [float(r['two_var']) for r in rq])), 4), 'reported': round(refq['two_var_auroc'], 4)}
# ---- level 2: rebuild from association-level labels + years (no text)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pro_common as pc
Q = json.load(open(f'{SNAP}/claim_queries.json')); CODE = {'PROTECTIVE': -1, 'HARMFUL': 1, 'NULL': 0, 'UNCLEAR': 99}
mism = 0; n = 0; prob_mism = 0
for r in rows:
    cid = r['claim_id']; t = int(float(r['cutoff_year']))
    D = {d['pmid']: CODE[d['direction']] for d in json.load(open(f'{SNAP}/directions/{cid}.json'))['directions']}
    recs = [(int(x['year']), D[x['pmid']]) for x in json.load(open(f'{SNAP}/records/{cid}.json'))['records'] if x['pmid'] in D and x.get('year')]
    pre = [c for yr, c in recs if yr < t and c != 99]; post = [c for yr, c in recs if yr >= t and c != 99]
    h = sum(1 for c in pre if c == 1); l = sum(1 for c in pre if c == -1); ph = sum(1 for c in post if c == 1); pl = sum(1 for c in post if c == -1)
    s0 = pc.sgn0((h - l) / (h + l)) if h + l >= 8 else 0; fut = pc.sgn0((ph - pl) / (ph + pl)) if ph + pl else 0
    ev = int(fut != s0); n += 1; mism += int(ev != int(r['y'])) + int(h != int(r['n_plus'])) + int(l != int(r['n_minus']))
    prob_mism += int(abs(pc.bb_change_prob(h, l, ph + pl, s0=s0) - float(r['bb_oracle'])) > 1e-6)
out['level2_rebuild_from_labels'] = {'units': n, 'count_or_endpoint_mismatches': mism, 'realised_count_probability_mismatches': prob_mism, 'note': 'uses identifiers, years and labels only; numeric/design features need abstract text and are not rebuilt here'}
out['level3_text_stage'] = 'not supported from the release: abstract text is not redistributed'
json.dump(out, open(os.path.join(R, 'guide', 'smoke_test.json'), 'w'), indent=1); shutil.rmtree(tmp); print(json.dumps(out, indent=1))
