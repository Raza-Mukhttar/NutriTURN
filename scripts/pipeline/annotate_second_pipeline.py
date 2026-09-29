"""Issue 6: independent re-extraction of the full source corpus with a second model family
(Qwen2.5-7B-Instruct), same prompt, truncation and batch settings as the deployed Llama extractor, but
with COMPLETE verbalizer scoring: the probability of each label is the teacher-forced probability of the
whole label string (all its tokens) given the prompt, normalised over the four labels. The deployed
first-token read-out is recorded alongside for comparison. Label definitions are made explicit in the
system prompt (NULL = the abstract directly reports no association for the named pair; UNCLEAR = the pair
is absent or its direction cannot be determined).

  python reextract_qwen.py --gpu 2 --shard 0 --nshards 3
Writes results/reextraction/qwen25_7b_shard<k>.jsonl (one line per (claim, pmid)).
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
import argparse, json, os, sys, time, math
HERE = _os.path.join(_NT, 'scripts'); G = _NT
SNAP = _os.path.join(_NT, 'data', 'protocol_snapshot')
LABELS = ['PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR']
BATCH, MAX_LEN, TEXT_BUDGET = 8, 1600, 3000
REPO = 'Qwen/Qwen2.5-7B-Instruct'
SYS = ('You classify the direction of association reported between an exposure and an outcome. '
       'Answer with exactly one word: PROTECTIVE if the exposure is reported to reduce the outcome, '
       'HARMFUL if it increases it, NULL if the abstract directly reports no association for the named '
       'exposure and outcome, UNCLEAR if the pair is absent from the abstract or its direction cannot be determined.')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gpu', required=True); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    Q = json.load(open(f'{SNAP}/claim_queries.json'))
    items = []
    for cid, q in Q.items():
        Dj = {d['pmid']: d['direction'] for d in json.load(open(f'{SNAP}/directions/{cid}.json'))['directions']}
        for r in json.load(open(f'{SNAP}/records/{cid}.json'))['records']:
            if r['pmid'] in Dj and r.get('year'):
                items.append((cid, r['pmid'], int(r['year']), q['exposure'], q['outcome'], r.get('title') or '', r.get('abstract') or '', Dj[r['pmid']]))
    items = items[a.shard::a.nshards]
    if a.limit: items = items[:a.limit]
    out_dir = f'{G}/results/reextraction'; os.makedirs(out_dir, exist_ok=True); out_path = f'{out_dir}/qwen25_7b_shard{a.shard}.jsonl'
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path): d = json.loads(line); done.add((d['claim_id'], d['pmid']))
    items = [it for it in items if (it[0], it[1]) not in done]
    print(f'[shard {a.shard}/{a.nshards}] {len(items)} prompts to score ({len(done)} already done)', flush=True)
    tok = AutoTokenizer.from_pretrained(REPO); tok.padding_side = 'left'
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
    lab_ids = [tok.encode(l, add_special_tokens=False) for l in LABELS]; first_ids = [x[0] for x in lab_ids]
    print('label tokenisations:', dict(zip(LABELS, lab_ids)), 'first tokens distinct:', len(set(first_ids)) == 4, flush=True)

    def build(it):
        cid, pmid, yr, exp, outc, title, abstract, _ = it
        text = (title + '. ' + abstract)[:TEXT_BUDGET]
        msg = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': f"Exposure: {exp}\nOutcome: {outc}\n\nStudy: {text}\n\nDirection:"}]
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    fh = open(out_path, 'a'); t0 = time.time(); n = 0
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]; prompts = [build(it) for it in chunk]
        enc = tok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN, add_special_tokens=False)
        P = enc['input_ids']; M = enc['attention_mask']; L = P.shape[1]
        # sequences: prompt + label tokens, for each of the 4 labels (left-padded prompts keep the label at the end)
        maxlab = max(len(x) for x in lab_ids)
        seqs, masks, labpos = [], [], []
        for b in range(P.shape[0]):
            for li, lt in enumerate(lab_ids):
                pad = maxlab - len(lt)
                seqs.append(torch.cat([torch.full((pad,), tok.pad_token_id), P[b], torch.tensor(lt)]))
                masks.append(torch.cat([torch.zeros(pad, dtype=M.dtype), M[b], torch.ones(len(lt), dtype=M.dtype)]))
                labpos.append((pad + L, len(lt)))
        S = torch.stack(seqs).to('cuda:0'); MM = torch.stack(masks).to('cuda:0')
        T = S.shape[1]; keep = maxlab + 1                      # only the last (maxlab+1) positions can predict label tokens
        with torch.no_grad():
            h = model.model(input_ids=S, attention_mask=MM).last_hidden_state[:, -keep:, :]
            logp = torch.log_softmax(model.lm_head(h).float(), -1)   # [rows, keep, vocab]
        off = T - keep
        full_lp = []
        for k, (start, ln) in enumerate(labpos):
            lp = 0.0
            for j in range(ln):
                lp += logp[k, start + j - 1 - off, S[k, start + j]].item()
            full_lp.append(lp)
        # first-token read-out from the prompt-only position (row of label 0 has the prompt ending at index start-1)
        for b, it in enumerate(chunk):
            rows = full_lp[b * 4:(b + 1) * 4]; mx = max(rows); ws = [math.exp(x - mx) for x in rows]; z = sum(ws); p_full = [w / z for w in ws]
            k0 = b * 4; start = labpos[k0][0]; ft = logp[k0, start - 1 - off, :]
            sel = ft[first_ids]; p_first = torch.softmax(sel, -1).tolist(); mass = float(torch.exp(ft[first_ids]).sum())
            cid, pmid, yr, exp, outc, title, abstract, llama = it
            fh.write(json.dumps({'claim_id': cid, 'pmid': pmid, 'year': yr, 'llama_label': llama,
                                 'qwen_label_full': LABELS[max(range(4), key=lambda j: p_full[j])], 'p_full': [round(x, 5) for x in p_full],
                                 'qwen_label_first': LABELS[max(range(4), key=lambda j: p_first[j])], 'p_first': [round(x, 5) for x in p_first],
                                 'first_token_mass': round(mass, 5), 'label_logprob_full': [round(x, 4) for x in rows]}) + '\n')
        n += len(chunk)
        if (i // BATCH) % 50 == 0:
            fh.flush(); print(f'  {n}/{len(items)}  ({n / max(time.time() - t0, 1e-9):.1f} prompts/s)', flush=True)
    fh.close(); print('DONE', n, 'prompts in', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main()
