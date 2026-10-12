"""T5 steps 1, 2 and 4: reconstruct the model-ready inputs, hash them, and ship the rebuild script.

A hash-MATCH RATE against the original annotation run is not computable: the archived external
directions files carry the pmid, year, label and probability vector and a `model` field, but no hash
of the text that was scored, and the abstracts files carry no hash field either. Nothing in the
release records what text the first run saw, so there is no second set of digests to match against.
What IS deliverable, and is what this writes, is a forward manifest: the sha256 of every model-ready
input as reconstructed here, so any future re-execution can be checked against THIS run byte for
byte. That is the provenance the appendix can honestly claim.

The model-ready input is the exact string the annotator builds: the 3,000-character truncation of
title + '. ' + abstract, which is what enters the prompt. Hashing the prompt itself would fold in the
system text and chat template and so would not isolate the input.
"""

# ---- NutriTURN repository path bootstrap ----
import os as _os, sys as _sys
_NT = _os.path.abspath(__file__)
for _ in range(6):
    _NT = _os.path.dirname(_NT)
    if _os.path.isfile(_os.path.join(_NT, '.nutriturn_root')):
        break
else:
    raise SystemExit('cannot locate the repository root (.nutriturn_root marker) from ' + __file__)
for _p in ('audit_pipeline/lib', 'data_preprocessing/lib', 'audit_experiments',
           'audit_experiments/figures', 'data_annotation', 'data_preparation', ''):
    _q = _os.path.join(_NT, *_p.split('/')) if _p else _NT
    if _q not in _sys.path:
        _sys.path.insert(0, _q)
# ---------------------------------------------

import hashlib, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_BUDGET = 3000

def model_ready(title, abstract):
    return ((title or '') + '. ' + (abstract or ''))[:TEXT_BUDGET]

def main():
    items = [json.loads(l) for l in open(os.path.join(HERE, 'data', 'items.jsonl')) if l.strip()]
    rows, agg = [], hashlib.sha256()
    n_ab = 0
    for it in sorted(items, key=lambda x: (int(x['domain']), x['claim_id'], str(x['pmid']))):
        t = model_ready(it.get('title'), it.get('abstract'))
        h = hashlib.sha256(t.encode('utf-8')).hexdigest()
        n_ab += bool((it.get('abstract') or '').strip())
        rows.append((int(it['domain']), it['claim_id'], str(it['pmid']), len(t), h))
        agg.update(h.encode())
    out = os.path.join(HERE, 'validation', 'T5_input_manifest.tsv')
    with open(out, 'w') as f:
        f.write('domain\tclaim_id\tpmid\tn_chars\tsha256_model_ready_input\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    meta = dict(n_inputs=len(rows), with_abstract=n_ab,
                pct_with_abstract=round(100 * n_ab / max(len(rows), 1), 2),
                text_budget=TEXT_BUDGET,
                definition="sha256 of (title + '. ' + abstract)[:3000], the string the annotator scores",
                aggregate_sha256=agg.hexdigest(),
                manifest_sha256=hashlib.sha256(open(out, 'rb').read()).hexdigest(),
                manifest_path='external_reannotation/validation/T5_input_manifest.tsv',
                rebuild_script='external_reannotation/scripts/build_items.py',
                hash_script='external_reannotation/scripts/build_input_manifest.py',
                probe_script='external_reannotation/scripts/probe_primary_reconstruction.py',
                cross_run_match_rate=None,
                cross_run_match_note=('Not computable: the archived external directions and abstracts '
                                      'files record no hash of the scored text, so there is no prior '
                                      'digest set to compare against.'))
    json.dump(meta, open(os.path.join(HERE, 'validation', 'T5_input_manifest_meta.json'), 'w'), indent=1)
    for k in ('n_inputs', 'pct_with_abstract', 'aggregate_sha256', 'manifest_sha256'):
        print(f'{k:22s} {meta[k]}')
    print(f'\nwritten {out}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
