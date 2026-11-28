"""Full-precision 70B cell: Llama-3.1-70B-Instruct, bfloat16 weights, on all source associations.

The 70B-class cells of the paper were run from 4-bit NF4 checkpoints. This script repeats the
Llama-3.1-70B cell from the official checkpoint in bfloat16, with everything else unchanged, so a
difference between the two cells is attributable to weight precision alone.

Configuration, identical to the 4-bit cell except for the weights:
  prompt        P2 (label definitions in the system prompt)
  read-out      full-string: summed log-probability of each label's tokens appended to the prompt,
                softmax over the four sums, not length-normalised; first-token is stored as well
  text budget   3,000 characters of title + abstract
  truncation    1,600 tokens, left padding
  batch         1 association (its four label continuations form one forward pass)
  decoding      none: a single forward pass, no sampling
  weights       official checkpoint, bfloat16, spread over several GPUs (about 141 GB)

    python annotate_fp70.py --repo /path/to/NutriTURN-Repository --gpus 1,2,3,4

Resumable: associations already in the output file are skipped. --limit N scores only the first N
remaining associations (smoke test). One JSON line per association, same fields as the 4-bit cell.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = 'meta-llama/Llama-3.1-70B-Instruct'
REVISION = '1605565b47bb9346c5515c34102e054115b4f98b'
LABELS = ['PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR']
BATCH, MAX_LEN, TEXT_BUDGET = 1, 1600, 3000
P2 = ('You classify the direction of association reported between an exposure and an outcome. '
      'Answer with exactly one word: PROTECTIVE if the exposure is reported to reduce the outcome, '
      'HARMFUL if it increases it, NULL if the abstract directly reports no association for the named '
      'exposure and outcome, UNCLEAR if the pair is absent from the abstract or its direction cannot be determined.')


def load_items(repo, text_path):
    rel = os.path.join(repo, 'data_release')
    txt = json.load(open(text_path))
    queries = json.load(open(os.path.join(rel, '01_claims', 'claim_queries.json')))
    items = []
    for r in csv.DictReader(open(os.path.join(rel, '02_records', 'cohort_tables', 'associations.csv'))):
        c, p = r['claim_id'], str(r['pmid'])
        if not r.get('year'):
            continue
        q = queries.get(c) or {}
        t = txt.get(f'{c}|{p}')
        if t is None or not (q.get('exposure') and q.get('outcome')):
            continue
        items.append((c, p, int(r['year']), q['exposure'], q['outcome'],
                      t.get('title') or '', t.get('abstract') or ''))
    items.sort(key=lambda x: (x[0], x[1]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, help='path to NutriTURN-Repository')
    ap.add_argument('--model', default=os.path.join(HERE, 'model'))
    ap.add_argument('--text', default=os.path.join(HERE, 'inputs', 'text_full.json'))
    ap.add_argument('--out', default=os.path.join(HERE, 'output', 'cell_llama70_bf16.jsonl'))
    ap.add_argument('--gpus', default='1,2,3,4')
    ap.add_argument('--max-mem-gib', default='',
                    help='comma-separated per-GPU cap in GiB, in the order of --gpus')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--scoring', choices=('prefix', 'full4'), default='prefix',
                    help='prefix: the prompt is run once and the four label continuations are scored '
                         'from its key-value cache; full4: prompt + label is run as four sequences, '
                         'the code of the 4-bit cell. Mathematically identical; prefix is ~3x faster')
    ap.add_argument('--keys', help='file of claim_id<TAB>pmid lines: score only these (validation)')
    a = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpus
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    items = load_items(a.repo, a.text)
    total = len(items)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
                done.add((d['claim_id'], d['pmid']))
            except Exception:
                pass
    if a.keys:
        want = {tuple(l.rstrip('\n').split('\t')) for l in open(a.keys) if l.strip()}
        items = [it for it in items if (it[0], it[1]) in want]
    items = [it for it in items if (it[0], it[1]) not in done]
    if a.limit:
        items = items[:a.limit]
    print(f'[llama70 bf16, gpus {a.gpus}] corpus {total:,}  to score {len(items):,} '
          f'({len(done):,} already present)', flush=True)
    if not items:
        print('nothing to do', flush=True)
        return 0

    n_gpu = len(a.gpus.split(','))
    max_memory = None
    if a.max_mem_gib:
        caps = [x.strip() for x in a.max_mem_gib.split(',')]
        assert len(caps) == n_gpu, 'one cap per GPU'
        max_memory = {i: f'{c}GiB' for i, c in enumerate(caps)}
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t_load = time.time()
    device_map = 'auto'
    if max_memory:
        # Explicit placement: embeddings on the first GPU, decoder layers filled in order up to each
        # cap, final norm and output head with the last layer. Nothing may fall back to CPU or disk.
        from transformers import AutoConfig
        cfg_m = AutoConfig.from_pretrained(a.model)
        hsz, isz, voc = cfg_m.hidden_size, cfg_m.intermediate_size, cfg_m.vocab_size
        kv = hsz * cfg_m.num_key_value_heads // cfg_m.num_attention_heads
        layer_gib = 2 * (2 * hsz * hsz + 2 * hsz * kv + 3 * hsz * isz + 2 * hsz) / 2 ** 30
        emb_gib = 2 * voc * hsz / 2 ** 30
        caps = [float(c) for c in a.max_mem_gib.split(',')]
        used = [0.0] * n_gpu
        device_map = {'model.embed_tokens': 0}
        used[0] += emb_gib
        dev = 0
        for li in range(cfg_m.num_hidden_layers):
            last = li == cfg_m.num_hidden_layers - 1
            need = layer_gib + (emb_gib if last else 0)
            while dev < n_gpu and used[dev] + need > caps[dev]:
                dev += 1
            if dev >= n_gpu:
                raise SystemExit(f'the caps {caps} GiB cannot hold the model (layer {li} unplaced)')
            device_map[f'model.layers.{li}'] = dev
            used[dev] += layer_gib
        device_map['model.norm'] = dev
        device_map['model.rotary_emb'] = dev
        device_map['lm_head'] = dev
        print('placement GiB per GPU:', [round(u + (emb_gib if i == dev else 0), 1) for i, u in enumerate(used)],
              flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 device_map=device_map).eval()
    placed = set(model.hf_device_map.values())
    assert all(isinstance(d, int) for d in placed), f'part of the model is not on a GPU: {placed}'
    dtypes = {str(p.dtype) for p in model.parameters()}
    n_par = sum(p.numel() for p in model.parameters())
    quantised = getattr(model, 'is_quantized', False) or hasattr(model.config, 'quantization_config')
    print(f'loaded in {round(time.time() - t_load)} s  parameters {n_par / 1e9:.2f}B  '
          f'dtypes {sorted(dtypes)}  quantised {bool(quantised)}', flush=True)
    assert dtypes == {'torch.bfloat16'} and not quantised, 'weights are not plain bfloat16'
    in_dev = model.get_input_embeddings().weight.device
    head_dev = model.lm_head.weight.device

    cfg = dict(cell='llama70_bf16', model=MODEL_ID, revision=REVISION, prompt_id='P2', prompt=P2,
               readout='full_string', scoring=a.scoring, weights='bfloat16, official checkpoint, not quantised',
               batch=BATCH, max_len=MAX_LEN, text_budget=TEXT_BUDGET, labels=LABELS,
               decoding='single forward pass, no sampling', scope='all source associations with a year')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cfg_path = os.path.join(os.path.dirname(a.out), 'config_record.json')
    if not a.limit or not os.path.exists(cfg_path):
        json.dump(dict(config=cfg,
                       config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                       text_full_sha256=hashlib.sha256(open(a.text, 'rb').read()).hexdigest(),
                       parameters=n_par, parameter_dtypes=sorted(dtypes)),
                  open(cfg_path, 'w'), indent=1)

    lab_ids = [tok.encode(l, add_special_tokens=False) for l in LABELS]
    first_ids = [x[0] for x in lab_ids]
    print('label tokenisations:', dict(zip(LABELS, lab_ids)),
          'first tokens distinct:', len(set(first_ids)) == 4, flush=True)
    maxlab = max(len(x) for x in lab_ids)

    def build(it):
        c, p, yr, exp, outc, title, abstract = it
        text = (title + '. ' + abstract)[:TEXT_BUDGET]
        msg = [{'role': 'system', 'content': P2},
               {'role': 'user', 'content': f"Exposure: {exp}\nOutcome: {outc}\n\nStudy: {text}\n\nDirection:"}]
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fh = open(a.out, 'a')
    t0 = time.time()
    n = 0
    from transformers import DynamicCache
    conts = [lt[:-1] for lt in lab_ids]            # tokens fed after the prompt, per label
    need_c = [j for j, c in enumerate(conts) if c]  # labels with more than one token
    cmax = max((len(conts[j]) for j in need_c), default=0)

    def score_prefix(it):
        """One association: prompt once with cache, then all multi-token continuations in one batch.

        Right padding of the continuations needs no mask: under causal attention a padded position
        comes after every real token and cannot influence it."""
        enc = tok([build(it)], return_tensors='pt', truncation=True, max_length=MAX_LEN,
                  add_special_tokens=False)
        P = enc['input_ids'].to(in_dev)
        L = P.shape[1]
        cache = DynamicCache()
        with torch.no_grad():
            h = model.model(input_ids=P, past_key_values=cache, use_cache=True).last_hidden_state[:, -1:, :]
            lp0 = torch.log_softmax(model.lm_head(h.to(head_dev)).float(), -1)[0, 0].cpu()
            lpc = None
            if need_c:
                cache.batch_repeat_interleave(len(need_c))
                X = torch.full((len(need_c), cmax), tok.pad_token_id, dtype=P.dtype)
                for r, j in enumerate(need_c):
                    X[r, :len(conts[j])] = torch.tensor(conts[j])
                pos = torch.arange(L, L + cmax, device=in_dev)
                hc = model.model(input_ids=X.to(in_dev), past_key_values=cache, use_cache=True,
                                 cache_position=pos, position_ids=pos.unsqueeze(0).expand(len(need_c), -1)
                                 ).last_hidden_state
                lpc = torch.log_softmax(model.lm_head(hc.to(head_dev)).float(), -1).cpu()
        full_lp = []
        for j, lt in enumerate(lab_ids):
            v = lp0[lt[0]].item()
            if len(lt) > 1:
                r = need_c.index(j)
                for q in range(1, len(lt)):
                    v += lpc[r, q - 1, lt[q]].item()
            full_lp.append(v)
        p_first = torch.softmax(lp0[first_ids], -1).tolist()
        return full_lp, p_first

    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        if a.scoring == 'prefix':
            for it in chunk:
                rows, p_first = score_prefix(it)
                mx = max(rows)
                ws = [math.exp(x - mx) for x in rows]
                z = sum(ws)
                p_full = [w / z for w in ws]
                c, p, yr, exp, outc, title, abstract = it
                fh.write(json.dumps({
                    'claim_id': c, 'pmid': p, 'year': yr, 'cell': 'llama70_bf16',
                    'label_full': LABELS[max(range(4), key=lambda j: p_full[j])],
                    'p_full': [round(x, 5) for x in p_full],
                    'label_first': LABELS[max(range(4), key=lambda j: p_first[j])],
                    'p_first': [round(x, 5) for x in p_first]}) + '\n')
            n += len(chunk)
            if (i // BATCH) % 100 == 0:
                fh.flush()
                el = max(time.time() - t0, 1e-9)
                print(f'  {n:,}/{len(items):,}  {n / el:.3f} prompts/s  '
                      f'eta {(len(items) - n) / max(n / el, 1e-9) / 3600:.1f} h', flush=True)
            continue
        enc = tok([build(it) for it in chunk], return_tensors='pt', padding=True,
                  truncation=True, max_length=MAX_LEN, add_special_tokens=False)
        P, M = enc['input_ids'], enc['attention_mask']
        L = P.shape[1]
        seqs, masks, labpos = [], [], []
        for b in range(P.shape[0]):
            for lt in lab_ids:
                pad = maxlab - len(lt)
                seqs.append(torch.cat([torch.full((pad,), tok.pad_token_id), P[b], torch.tensor(lt)]))
                masks.append(torch.cat([torch.zeros(pad, dtype=M.dtype), M[b],
                                        torch.ones(len(lt), dtype=M.dtype)]))
                labpos.append((pad + L, len(lt)))
        S = torch.stack(seqs).to(in_dev)
        MM = torch.stack(masks).to(in_dev)
        T = S.shape[1]
        keep = maxlab + 1
        with torch.no_grad():
            h = model.model(input_ids=S, attention_mask=MM).last_hidden_state[:, -keep:, :]
            logp = torch.log_softmax(model.lm_head(h.to(head_dev)).float(), -1).cpu()
        S = S.cpu()
        off = T - keep
        full_lp = []
        for k, (start, ln) in enumerate(labpos):
            lp = 0.0
            for j in range(ln):
                lp += logp[k, start + j - 1 - off, S[k, start + j]].item()
            full_lp.append(lp)
        for b, it in enumerate(chunk):
            rows = full_lp[b * 4:(b + 1) * 4]
            mx = max(rows)
            ws = [math.exp(x - mx) for x in rows]
            z = sum(ws)
            p_full = [w / z for w in ws]
            k0 = b * 4
            start = labpos[k0][0]
            ft = logp[k0, start - 1 - off, :]
            p_first = torch.softmax(ft[first_ids], -1).tolist()
            c, p, yr, exp, outc, title, abstract = it
            fh.write(json.dumps({
                'claim_id': c, 'pmid': p, 'year': yr, 'cell': 'llama70_bf16',
                'label_full': LABELS[max(range(4), key=lambda j: p_full[j])],
                'p_full': [round(x, 5) for x in p_full],
                'label_first': LABELS[max(range(4), key=lambda j: p_first[j])],
                'p_first': [round(x, 5) for x in p_first]}) + '\n')
        n += len(chunk)
        if (i // BATCH) % 100 == 0:
            fh.flush()
            el = max(time.time() - t0, 1e-9)
            print(f'  {n:,}/{len(items):,}  {n / el:.3f} prompts/s  '
                  f'eta {(len(items) - n) / max(n / el, 1e-9) / 3600:.1f} h', flush=True)
    fh.close()
    print(f'DONE {n:,} prompts in {round(time.time() - t0)} s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
