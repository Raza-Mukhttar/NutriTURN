"""NutriMATURE system, step 2 (optional, needs a GPU): study-level direction extraction.

For every record of a claim, the corrected NutriMATURE extractor asks meta-llama/Llama-3.1-8B-Instruct (bfloat16) for the
direction of the exposure-outcome association (PROTECTIVE / HARMFUL / NULL / UNCLEAR). There is no text generation: the
logits at the answer position are restricted to the first token of each label, and the argmax is taken (softmax kept
as confidence). Title + abstract are truncated to 3,000 characters and the prompt to 1,600 tokens; batch size 16. Every
prompt is checked to contain the claim's own outcome term before scoring (the guard against the prompt defect that the
corrected extractor fixed). Records that already have a label keep it; only unlabelled records are scored.

  python 02_extract_directions.py --gpu 0 --shard 0 --nshard 1     # label unlabelled records (one process per GPU)
  python 02_extract_directions.py --summary-only                   # no GPU: label summary table from data/directions
Outputs: data/directions/<claim>.json (updated), tables/02_direction_labels.{csv,json}
Model revision used for the shipped labels: 0e9e39f249a16976918f6564b8830bc894c89659.
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

import os, sys, json, time, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import nutrimature as nm

LABELS = ['PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR']
BATCH, MAX_LEN, TEXT_BUDGET = 16, 1600, 3000
SYS = ('You classify the direction of association reported between an exposure and an outcome. '
       'Answer with exactly one word: PROTECTIVE if the exposure is reported to reduce the '
       'outcome, HARMFUL if it increases it, NULL if no association is reported, UNCLEAR if the '
       'study does not report this association.')


def summary():
    Q = json.load(open(nm.QUERIES)); rows = []; tot = collections.Counter()
    for cid in Q:
        recs = json.load(open(f'{nm.RECORDS}/{cid}.json'))['records']; D = json.load(open(f'{nm.DIRECTIONS}/{cid}.json'))['directions']
        c = collections.Counter(d['direction'] for d in D); tot.update(c)
        rows.append({'claim': cid, 'records': len(recs), 'labelled': len(D), **{l.lower(): c.get(l, 0) for l in LABELS},
                     'history_retrieved_in_full_after_800_cap': Q[cid]['history_retrieved_in_full_after_800_cap']})
    n = sum(tot.values())
    nm.save_table(rows, '02_direction_labels', {'labels_total': n, 'label_shares': {l: round(tot[l] / n, 4) for l in LABELS},
                  'extractor': 'meta-llama/Llama-3.1-8B-Instruct, first-token logit argmax over four labels, no generation'})
    print(f'{len(rows)} claims, {n} labelled records; shares ' + ', '.join(f'{l} {tot[l] / n:.3f}' for l in LABELS), flush=True)


def extract(gpu, shard, nshard, model_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id); tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
    ids = [tok.encode(l, add_special_tokens=False)[0] for l in LABELS]; assert len(set(ids)) == 4, ids

    def prompt(exposure, outcome, title, abstract):
        text = (title + '. ' + abstract)[:TEXT_BUDGET]
        return tok.apply_chat_template([{'role': 'system', 'content': SYS},
                                        {'role': 'user', 'content': f'Exposure: {exposure}\nOutcome: {outcome}\n\nStudy: {text}\n\nDirection:'}],
                                       tokenize=False, add_generation_prompt=True)
    Q = json.load(open(nm.QUERIES)); mine = [c for i, c in enumerate(Q) if i % nshard == shard]; t0 = time.time(); ndone = 0
    for cid in mine:
        recs = json.load(open(f'{nm.RECORDS}/{cid}.json'))['records']; dp = f'{nm.DIRECTIONS}/{cid}.json'
        D = json.load(open(dp))['directions'] if os.path.exists(dp) else []
        have = {d['pmid'] for d in D}; todo = [r for r in recs if r['pmid'] not in have]
        if not todo: continue
        exposure, outcome = Q[cid]['exposure'], Q[cid]['outcome']
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            prompts = [prompt(exposure, outcome, r.get('title', ''), r.get('abstract', '')) for r in chunk]
            assert all(f'Outcome: {outcome}\n' in p for p in prompts), f'{cid}: malformed Outcome field'
            enc = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN, add_special_tokens=False).to('cuda:0')
            with torch.no_grad():
                try: o = model(**enc, num_logits_to_keep=1)
                except TypeError:
                    try: o = model(**enc, logits_to_keep=1)
                    except TypeError: o = model(**enc)
                sel = o.logits[:, -1, :].float()[:, ids]
            prob = torch.softmax(sel, -1).tolist(); pick = sel.argmax(-1).tolist()
            for r, p, pr in zip(chunk, pick, prob):
                D.append({'pmid': r['pmid'], 'year': r['year'], 'direction': LABELS[p], 'p': {l: round(x, 4) for l, x in zip(LABELS, pr)}})
            ndone += len(chunk)
        json.dump({'claim_id': cid, 'n': len(D), 'directions': D}, open(dp, 'w'))
        print(f'[gpu{gpu}] {cid:44s} +{len(todo)} labels  ({ndone / max(time.time() - t0, 1):.0f}/s)', flush=True)
    print(f'[gpu{gpu}] done: {ndone} records labelled', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--summary-only', action='store_true'); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct'); a = ap.parse_args()
    if not a.summary_only: extract(a.gpu, a.shard, a.nshard, a.model)
    summary()
