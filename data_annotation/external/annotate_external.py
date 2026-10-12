"""Independent setting at scale, step 2 -- re-annotate External-1/2 with the SECOND pipeline.

Qwen2.5-7B-Instruct on the frozen read-out, byte-identical to the shipped pipeline: same system
prompt, same user-prompt template, 3,000-character text budget, chat template with
add_generation_prompt, left padding, truncation at 1,600, batch 8, a single forward pass with no
sampling, and p_full as the softmax over the four summed label log-probabilities.

Nothing about the method changes between the diet corpus and these two non-diet corpora; only the
records change. That is the point of the experiment.
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

import argparse, json, math, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ['PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR']
BATCH, MAX_LEN, TEXT_BUDGET = 8, 1600, 3000
SYS = ('You classify the direction of association reported between an exposure and an outcome. '
       'Answer with exactly one word: PROTECTIVE if the exposure is reported to reduce the outcome, '
       'HARMFUL if it increases it, NULL if the abstract directly reports no association for the named '
       'exposure and outcome, UNCLEAR if the pair is absent from the abstract or its direction cannot be determined.')
REPO = 'Qwen/Qwen2.5-7B-Instruct'


def load_items():
    return [json.loads(l) for l in open(os.path.join(HERE, 'data', 'items.jsonl')) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    outd = os.path.join(HERE, 'data', 'annot_second')
    os.makedirs(outd, exist_ok=True)
    out_path = os.path.join(outd, f'shard{a.shard}of{a.nshards}.jsonl')
    items = load_items()
    total = len(items)
    items = items[a.shard::a.nshards]
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                d = json.loads(line); done.add((d['claim_id'], d['pmid']))
            except Exception:
                pass
    items = [it for it in items if (it['claim_id'], it['pmid']) not in done]
    print(f'[second shard {a.shard}/{a.nshards} gpu {a.gpu}] corpus {total:,}  '
          f'this shard {len(items):,} to score ({len(done):,} done)  model {REPO}', flush=True)
    if not items:
        print('nothing to do', flush=True); return 0

    tok = AutoTokenizer.from_pretrained(REPO); tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16,
                                                device_map='cuda:0').eval()
    lab_ids = [tok.encode(l, add_special_tokens=False) for l in LABELS]
    first_ids = [x[0] for x in lab_ids]
    print('label tokenisations:', dict(zip(LABELS, lab_ids)),
          'first tokens distinct:', len(set(first_ids)) == 4, flush=True)
    maxlab = max(len(x) for x in lab_ids)

    def build(it):
        text = (it['title'] + '. ' + it['abstract'])[:TEXT_BUDGET]
        msg = [{'role': 'system', 'content': SYS},
               {'role': 'user', 'content': f"Exposure: {it['exposure']}\nOutcome: {it['outcome']}\n\nStudy: {text}\n\nDirection:"}]
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    fh = open(out_path, 'a'); t0 = time.time(); n = 0
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        enc = tok([build(it) for it in chunk], return_tensors='pt', padding=True, truncation=True,
                  max_length=MAX_LEN, add_special_tokens=False)
        P, M = enc['input_ids'], enc['attention_mask']; L = P.shape[1]
        seqs, masks, labpos = [], [], []
        for b in range(P.shape[0]):
            for lt in lab_ids:
                pad = maxlab - len(lt)
                seqs.append(torch.cat([torch.full((pad,), tok.pad_token_id), P[b], torch.tensor(lt)]))
                masks.append(torch.cat([torch.zeros(pad, dtype=M.dtype), M[b],
                                        torch.ones(len(lt), dtype=M.dtype)]))
                labpos.append((pad + L, len(lt)))
        S = torch.stack(seqs).to('cuda:0'); MM = torch.stack(masks).to('cuda:0')
        T = S.shape[1]; keep = maxlab + 1
        with torch.no_grad():
            h = model.model(input_ids=S, attention_mask=MM).last_hidden_state[:, -keep:, :]
            logp = torch.log_softmax(model.lm_head(h).float(), -1)
        off = T - keep
        full_lp = []
        for k, (start, ln) in enumerate(labpos):
            lp = 0.0
            for j in range(ln):
                lp += logp[k, start + j - 1 - off, S[k, start + j]].item()
            full_lp.append(lp)
        for b, it in enumerate(chunk):
            rows = full_lp[b * 4:(b + 1) * 4]
            mx = max(rows); ws = [math.exp(x - mx) for x in rows]; z = sum(ws)
            p_full = [w / z for w in ws]
            k0 = b * 4; start = labpos[k0][0]
            ft = logp[k0, start - 1 - off, :]
            p_first = torch.softmax(ft[first_ids], -1).tolist()
            fh.write(json.dumps({
                'domain': it['domain'], 'claim_id': it['claim_id'], 'pmid': it['pmid'],
                'year': it['year'],
                'label_full': LABELS[max(range(4), key=lambda j: p_full[j])],
                'p_full': [round(x, 5) for x in p_full],
                'label_first': LABELS[max(range(4), key=lambda j: p_first[j])],
                'p_first': [round(x, 5) for x in p_first]}) + '\n')
        n += len(chunk)
        if (i // BATCH) % 100 == 0:
            fh.flush()
            el = max(time.time() - t0, 1e-9)
            print(f'  {n:,}/{len(items):,}  {n/el:.2f} prompts/s  '
                  f'eta {(len(items)-n)/max(n/el,1e-9)/3600:.2f} h', flush=True)
    fh.close()
    print(f'DONE {n:,} prompts in {round(time.time()-t0)} s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
