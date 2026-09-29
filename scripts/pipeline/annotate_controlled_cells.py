"""E8: controlled annotation cells (model x prompt x read-out).

Cells of the design. A and D already exist in the release; C exists from the attribution run.
  A      Llama-3.1-8B  P1  first token   (the deployed pipeline)
  Aprime Llama-3.1-8B  P1  full string   <- computed here
  B      Llama-3.1-8B  P2  full string   <- computed here
  C      Qwen2.5-7B    P1  full string   (results/reextraction_origprompt)
  D      Qwen2.5-7B    P2  full string   (results/reextraction, the second pipeline)

Identical text budget (3,000 characters), 1,600-token right truncation, bf16, no sampling, as in
the deployed configuration.

Full-string scoring is exact and matches the released implementation, but reuses the prompt's
key/value cache so the four label continuations cost a few token positions instead of four full
prompt passes. `--verify` rescores a slice of an existing cell and compares, to prove equivalence.

  python analyses/nfx_annotate.py --cell Aprime --gpu 0 --shard 0 --nshards 5
  python analyses/nfx_annotate.py --cell D --gpu 0 --limit 400 --verify
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

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = _NT
SNAP = _os.path.join(_NT, 'data', 'protocol_snapshot')
GPTPRO = _NT
LABELS = ['PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR']
MAX_LEN, TEXT_BUDGET = 1600, 3000

P1 = ('You classify the direction of association reported between an exposure and an outcome. '
      'Answer with exactly one word: PROTECTIVE if the exposure is reported to reduce the outcome, '
      'HARMFUL if it increases it, NULL if no association is reported, UNCLEAR if the study does not report '
      'this association.')
P2 = ('You classify the direction of association reported between an exposure and an outcome. '
      'Answer with exactly one word: PROTECTIVE if the exposure is reported to reduce the outcome, '
      'HARMFUL if it increases it, NULL if the abstract directly reports no association for the named '
      'exposure and outcome, UNCLEAR if the pair is absent from the abstract or its direction cannot be determined.')

LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'
QWEN = 'Qwen/Qwen2.5-7B-Instruct'
CELLS = {'Aprime': (LLAMA, P1), 'B': (LLAMA, P2), 'C': (QWEN, P1), 'D': (QWEN, P2)}
REF = {'C': os.path.join(GPTPRO, 'results', 'reextraction_origprompt'),
       'D': os.path.join(GPTPRO, 'results', 'reextraction')}


def load_items():
    Q = json.load(open(f'{SNAP}/claim_queries.json'))
    items = []
    for cid, q in Q.items():
        Dj = {d['pmid']: d['direction'] for d in json.load(open(f'{SNAP}/directions/{cid}.json'))['directions']}
        for r in json.load(open(f'{SNAP}/records/{cid}.json'))['records']:
            if r['pmid'] in Dj and r.get('year'):
                items.append((cid, r['pmid'], int(r['year']), q['exposure'], q['outcome'],
                              r.get('title') or '', r.get('abstract') or '', Dj[r['pmid']]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, choices=list(CELLS))
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--batch', type=int, default=16)   # the deployed batch size
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--verify', action='store_true', help='compare against the released cell instead of writing')
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    repo, SYS = CELLS[a.cell]
    items = load_items()
    items = items[a.shard::a.nshards]
    if a.limit:
        items = items[:a.limit]

    out_dir = os.path.join(BASE, 'annotations', 'raw')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'cell{a.cell}_shard{a.shard}.jsonl')
    done = set()
    if os.path.exists(out_path) and not a.verify:
        for line in open(out_path):
            d = json.loads(line); done.add((d['claim_id'], d['pmid']))
        items = [it for it in items if (it[0], it[1]) not in done]
    print(f'[cell {a.cell} shard {a.shard}/{a.nshards}] {len(items)} prompts ({len(done)} already done) '
          f'model {repo}', flush=True)
    if not items:
        return

    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
    lab_ids = [tok.encode(l, add_special_tokens=False) for l in LABELS]
    first_ids = [x[0] for x in lab_ids]
    maxlab = max(len(x) for x in lab_ids)
    print('label tokenisations:', dict(zip(LABELS, lab_ids)),
          '| first tokens distinct:', len(set(first_ids)) == 4, flush=True)

    # label continuation tensor, right-padded
    cont = torch.full((4, maxlab), tok.pad_token_id, dtype=torch.long)
    cmask = torch.zeros((4, maxlab), dtype=torch.long)
    for i, lt in enumerate(lab_ids):
        cont[i, :len(lt)] = torch.tensor(lt); cmask[i, :len(lt)] = 1
    cont = cont.to('cuda:0'); cmask = cmask.to('cuda:0')
    lab_len = torch.tensor([len(x) for x in lab_ids], device='cuda:0')

    def build(it):
        cid, pmid, yr, exp, outc, title, abstract, _ = it
        text = (title + '. ' + abstract)[:TEXT_BUDGET]
        msg = [{'role': 'system', 'content': SYS},
               {'role': 'user', 'content': f"Exposure: {exp}\nOutcome: {outc}\n\nStudy: {text}\n\nDirection:"}]
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    fh = None if a.verify else open(out_path, 'a')
    recs = []
    t0 = time.time(); n = 0
    for i in range(0, len(items), a.batch):
        chunk = items[i:i + a.batch]
        enc = tok([build(it) for it in chunk], return_tensors='pt', padding=True,
                  truncation=True, max_length=MAX_LEN, add_special_tokens=False)
        P = enc['input_ids'].to('cuda:0'); M = enc['attention_mask'].to('cuda:0')
        B, L = P.shape
        with torch.no_grad():
            # apply the language-model head only where a label token is predicted; materialising
            # logits over the whole prompt would cost tens of GB at these batch sizes.
            o = model.model(input_ids=P, attention_mask=M, use_cache=True)
            last = torch.log_softmax(model.lm_head(o.last_hidden_state[:, -1, :]).float(), -1)
            cache = o.past_key_values
            cache.batch_repeat_interleave(4)
            S = cont.repeat(B, 1)                                             # [B*4, maxlab]
            CM = cmask.repeat(B, 1)
            Mrep = M.repeat_interleave(4, 0)
            full_mask = torch.cat([Mrep, CM], 1)
            # The deployed pipeline scores a left-padded batch without explicit position ids, so the
            # prompt occupies positions 0..L-1 (RoPE is relative, so the per-row padding offset is
            # harmless). Continue that same numbering for the label tokens.
            pos = (L + torch.arange(maxlab, device=S.device)).unsqueeze(0).expand(B * 4, -1)
            lp2 = None
            if maxlab > 1:
                o2 = model.model(input_ids=S, attention_mask=full_mask, position_ids=pos,
                                 past_key_values=cache, use_cache=False)
                lp2 = torch.log_softmax(
                    model.lm_head(o2.last_hidden_state[:, :maxlab - 1, :]).float(), -1)
            del cache
        # log p(label) = log p(t0 | prompt) + sum_{j>=1} log p(tj | prompt, t<j)
        first_lp = last.repeat_interleave(4, 0).gather(1, S[:, :1]).squeeze(1)  # [B*4]
        rest = torch.zeros_like(first_lp)
        if maxlab > 1:
            tgt = S[:, 1:]                                                     # token j predicted at position j-1
            g = lp2.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            keep = (torch.arange(1, maxlab, device=S.device).unsqueeze(0)
                    < lab_len.repeat(B).unsqueeze(1)).float()
            rest = (g * keep).sum(1)
        total = (first_lp + rest).view(B, 4)
        p_full = torch.softmax(total, -1)
        sel = last[:, first_ids]
        p_first = torch.softmax(sel, -1)
        mass = last[:, first_ids].exp().sum(-1)
        for b, it in enumerate(chunk):
            cid, pmid, yr, exp, outc, title, abstract, llama = it
            pf = p_full[b].tolist(); p1 = p_first[b].tolist(); tl = total[b].tolist()
            rec = {'claim_id': cid, 'pmid': pmid, 'year': yr, 'cell': a.cell,
                   'primary_label': llama,
                   'label_full': LABELS[max(range(4), key=lambda j: pf[j])],
                   'p_full': [round(x, 5) for x in pf],
                   'label_first': LABELS[max(range(4), key=lambda j: p1[j])],
                   'p_first': [round(x, 5) for x in p1],
                   'first_token_mass': round(float(mass[b]), 5),
                   'label_logprob_full': [round(x, 4) for x in tl]}
            if a.verify:
                recs.append(rec)
            else:
                fh.write(json.dumps(rec) + '\n')
        n += len(chunk)
        if (i // a.batch) % 100 == 0:
            if fh:
                fh.flush()
            el = time.time() - t0
            print(f'  {n}/{len(items)}  {n/max(el,1e-9):.1f} prompts/s  '
                  f'eta {(len(items)-n)/max(n/max(el,1e-9),1e-9)/60:.0f} min', flush=True)
    if fh:
        fh.close()
    el = time.time() - t0
    print(f'DONE cell {a.cell} shard {a.shard}: {n} prompts in {el:.0f} s ({n/max(el,1e-9):.1f}/s)', flush=True)

    if a.verify and a.cell in ('Aprime', 'B'):
        # the strict test for a Llama cell: its FIRST-TOKEN read-out must reproduce the
        # deployed labels exactly, because the deployed pipeline is that same read-out.
        n_ = sum(1 for r in recs)
        same = sum(1 for r in recs if r['label_first'] == r['primary_label'])
        agree_full = sum(1 for r in recs if r['label_full'] == r['primary_label'])
        print(f'VERIFY cell {a.cell}: {n_} records | first-token read-out equals the deployed label '
              f'{same}/{n_} ({100*same/max(n_,1):.1f}%) | full-string equals deployed '
              f'{agree_full}/{n_} ({100*agree_full/max(n_,1):.1f}%)', flush=True)
        return
    if a.verify:
        ref = {}
        for p in sorted(os.listdir(REF[a.cell])):
            for line in open(os.path.join(REF[a.cell], p)):
                d = json.loads(line); ref[(d['claim_id'], d['pmid'])] = d
        same_full = same_first = both = 0
        maxdp = 0.0
        for r in recs:
            k = (r['claim_id'], r['pmid'])
            if k not in ref:
                continue
            both += 1
            q = ref[k]
            same_full += int(r['label_full'] == q['qwen_label_full'])
            same_first += int(r['label_first'] == q['qwen_label_first'])
            maxdp = max(maxdp, max(abs(x - y) for x, y in zip(r['p_full'], q['p_full'])))
        print(f'VERIFY cell {a.cell}: compared {both} records | full-string label identical '
              f'{same_full}/{both} | first-token label identical {same_first}/{both} | '
              f'max |delta p_full| {maxdp:.5f}', flush=True)


if __name__ == '__main__':
    main()
