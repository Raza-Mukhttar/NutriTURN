"""T5 -- does the reconstructed external input reproduce the ARCHIVED deployed-pipeline labels?

The external re-annotation ran the SECOND pipeline over a freshly built item set, and the unit-level
rebuild already reproduces the shipped benchmark columns 322/322 and 1260/1260. That validates the
feature contract, not the text: the rebuild consumes the archived per-association directions, so a
reconstructed abstract that differs from the one the deployed pipeline actually saw would pass it
untouched. This probe closes that gap by re-running the DEPLOYED pipeline on the reconstructed text
and comparing label for label against the archive.

The archive makes this possible: external1/2 ship per-association `directions` files carrying both
the label and the full four-way probability vector, stamped model meta-llama/Llama-3.1-8B-Instruct.
So the comparison is exact, not inferred from unit features.

DEPLOYED CONFIGURATION, reproduced rather than re-chosen:
    model      meta-llama/Llama-3.1-8B-Instruct
    prompt     P1, verbatim from the shipped generator
    read-out   FIRST TOKEN (the deployed pipeline is that read-out; full string is recorded too,
               but the archive's labels are the first-token ones and that is what kappa is on)
    text       3,000-character title + abstract budget, 1,600-token truncation, left padding, bf16,
               batch 16, one forward pass, no sampling
Nothing here is new: P1, the budget and the read-out are read out of the release's own generator, and
the probe's only job is to put the reconstructed text through them.

SAMPLE, fixed before any outcome is seen: 1,000 associations from External-1 and 2,000 from
External-2, seed 3, stratified by the ARCHIVED label so all four classes are represented in
proportion-free quotas -- an UNCLEAR-heavy or HARMFUL-heavy sample would otherwise hide a class the
reconstruction breaks. Quotas are filled from the shuffled pool and any shortfall in a rare class is
redistributed, so the domain totals hold exactly.

THRESHOLDS, fixed in advance:
    kappa >= 0.95   the reconstruction is equivalent to the archived input
    0.85 - 0.95     a minor deviation, reported alongside the external numbers
    kappa < 0.85    a stated limitation on the external results
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

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np

import nfx_common as N

LABELS = ['PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR']
MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
MAX_LEN, TEXT_BUDGET, BATCH = 1600, 3000, 16
# P1, verbatim from data_annotation/04_annotate_controlled_cells.py
P1 = ('You classify the direction of association reported between an exposure and an outcome. '
      'Answer with exactly one word: PROTECTIVE if the exposure is reported to reduce the outcome, '
      'HARMFUL if it increases it, NULL if no association is reported, UNCLEAR if the study does not report '
      'this association.')
# External-1/2 corpora are not redistributed. Point these at a local copy, or set
# NUTRITURN_EXTERNAL1_DIR / NUTRITURN_EXTERNAL2_DIR. Each root holds abstracts/, directions/
# and claim_queries.json. See config/external_sources.json.
EXT_ROOT = {1: os.environ.get('NUTRITURN_EXTERNAL1_DIR', os.path.join(_NT, 'data', 'external', 'external1')),
            2: os.environ.get('NUTRITURN_EXTERNAL2_DIR', os.path.join(_NT, 'data', 'external', 'external2'))}
DIRS = {k: os.path.join(v, 'directions') for k, v in EXT_ROOT.items()}
QUOTA = {1: 1000, 2: 2000}
SEED = N.SEED
THRESH = dict(equivalent=0.95, minor=0.85)


def archive():
    """{(domain, claim_id, pmid): {'label':..., 'p':[...]}} from the archived directions files."""
    out = {}
    for which, d in DIRS.items():
        for f in sorted(glob.glob(d + '/*.json')):
            c = os.path.basename(f)[:-5]
            j = json.load(open(f))
            for r in j['directions']:
                pv = r.get('p') or {}
                out[(which, c, str(r['pmid']))] = dict(
                    label=r['direction'],
                    p=[pv.get(l) for l in LABELS] if pv else None,
                    model=j.get('model'))
    return out


def build_sample():
    """The frozen 3,000-association sample, stratified by archived label within each domain."""
    A = archive()
    items = [json.loads(l) for l in open(os.path.join(HERE, 'data', 'items.jsonl')) if l.strip()]
    pool = {1: {l: [] for l in LABELS}, 2: {l: [] for l in LABELS}}
    for it in items:
        k = (int(it['domain']), it['claim_id'], str(it['pmid']))
        a = A.get(k)
        if a is None or a['label'] not in LABELS:
            continue
        it = dict(it, domain=int(it['domain']), archived=a['label'], archived_p=a['p'])
        pool[int(it['domain'])][a['label']].append(it)
    rng = random.Random(SEED)
    sample, meta = [], {}
    for which, n in QUOTA.items():
        for l in LABELS:
            pool[which][l].sort(key=lambda x: (x['claim_id'], x['pmid']))
            rng.shuffle(pool[which][l])
        per = n // len(LABELS)
        take = {l: min(per, len(pool[which][l])) for l in LABELS}
        # redistribute any shortfall in a rare class so the domain total holds exactly
        short = n - sum(take.values())
        while short > 0:
            room = [l for l in LABELS if len(pool[which][l]) > take[l]]
            if not room:
                break
            for l in room:
                if short == 0:
                    break
                take[l] += 1
                short -= 1
        picked = []
        for l in LABELS:
            picked += pool[which][l][:take[l]]
        picked.sort(key=lambda x: (x['claim_id'], x['pmid']))
        sample += picked
        meta[f'external{which}'] = dict(requested=n, drawn=len(picked),
                                        available={l: len(pool[which][l]) for l in LABELS},
                                        drawn_by_archived_label=take)
    return sample, meta


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float('nan'), po


def boot_kappa(a, b, groups, n=N.BOOT, seed=SEED):
    g = np.array(groups)
    claims = np.unique(g)
    by = {c: np.where(g == c)[0] for c in claims}
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n):
        pick = rng.choice(claims, size=len(claims), replace=True)
        ix = np.concatenate([by[c] for c in pick])
        k, _po = kappa([a[i] for i in ix], [b[i] for i in ix])
        if not math.isnan(k):
            ks.append(k)
    return (float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))) if ks else (float('nan'),) * 2


def freeze(sample, meta):
    cfg = dict(model=MODEL, prompt_id='P1', prompt=P1, readout='first_token (the deployed read-out)',
               dtype='bfloat16', batch=BATCH, max_len=MAX_LEN, text_budget=TEXT_BUDGET,
               padding='left', decoding='single forward pass, no sampling',
               quota=QUOTA, seed=SEED, stratified_by='archived label, within domain',
               thresholds=THRESH,
               reference='archived per-association directions (external1/2), '
                         'model meta-llama/Llama-3.1-8B-Instruct')
    keys = '\n'.join(f"{it['domain']}\t{it['claim_id']}\t{it['pmid']}" for it in sample)
    rec = dict(config=cfg, sample=meta, n=len(sample),
               config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
               sample_sha256=hashlib.sha256(keys.encode()).hexdigest(),
               frozen_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    os.makedirs(os.path.join(HERE, 'validation'), exist_ok=True)
    p = os.path.join(HERE, 'validation', 'T5_probe_freeze.json')
    if os.path.exists(p):
        old = json.load(open(p))
        if old['sample_sha256'] != rec['sample_sha256'] or old['config_sha256'] != rec['config_sha256']:
            raise SystemExit(f'the frozen probe does not match this one; refusing to proceed\n  {p}')
        return old
    json.dump(rec, open(p, 'w'), indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--score-only', action='store_true')
    # execution-level only: a smaller batch when the card is shared with another tenant. Each
    # sequence is scored independently under left padding and an attention mask, so the batch
    # size changes no per-item value.
    ap.add_argument('--batch', type=int, default=BATCH)
    a = ap.parse_args()

    sample, meta = build_sample()
    rec = freeze(sample, meta)
    print(f'T5 probe   n={len(sample):,}   config {rec["config_sha256"][:16]}   '
          f'sample {rec["sample_sha256"][:16]}')
    for d, m in meta.items():
        print(f'  {d}: drawn {m["drawn"]:,} of {m["requested"]:,}  by archived label '
              f'{m["drawn_by_archived_label"]}  available {m["available"]}')

    outd = os.path.join(HERE, 'data', 'probe_primary')
    os.makedirs(outd, exist_ok=True)

    if not a.score_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        mine = sample[a.shard::a.nshards]
        out_path = os.path.join(outd, f'shard{a.shard}of{a.nshards}.jsonl')
        done = set()
        for f in sorted(glob.glob(os.path.join(outd, '*.jsonl'))):
            for line in open(f):
                try:
                    d = json.loads(line)
                    done.add((d['domain'], d['claim_id'], d['pmid']))
                except Exception:
                    pass
        mine = [it for it in mine
                if (it['domain'], it['claim_id'], str(it['pmid'])) not in done]
        print(f'[shard {a.shard}/{a.nshards} gpu {a.gpu}] {len(mine):,} to score '
              f'({len(done):,} already present)', flush=True)
        if mine:
            tok = AutoTokenizer.from_pretrained(MODEL)
            tok.padding_side = 'left'
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                         device_map='cuda:0').eval()
            lab_ids = [tok.encode(l, add_special_tokens=False) for l in LABELS]
            first_ids = [x[0] for x in lab_ids]
            print('label tokenisations:', dict(zip(LABELS, lab_ids)),
                  '| first tokens distinct:', len(set(first_ids)) == 4, flush=True)

            def build(it):
                text = ((it['title'] or '') + '. ' + (it['abstract'] or ''))[:TEXT_BUDGET]
                msg = [{'role': 'system', 'content': P1},
                       {'role': 'user', 'content': f"Exposure: {it['exposure']}\n"
                                                   f"Outcome: {it['outcome']}\n\n"
                                                   f"Study: {text}\n\nDirection:"}]
                return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

            fh = open(out_path, 'a')
            t0 = time.time()
            n = 0
            for i in range(0, len(mine), a.batch):
                chunk = mine[i:i + a.batch]
                enc = tok([build(it) for it in chunk], return_tensors='pt', padding=True,
                          truncation=True, max_length=MAX_LEN, add_special_tokens=False)
                P = enc['input_ids'].to('cuda:0')
                M = enc['attention_mask'].to('cuda:0')
                with torch.no_grad():
                    o = model.model(input_ids=P, attention_mask=M)
                    last = torch.log_softmax(
                        model.lm_head(o.last_hidden_state[:, -1, :]).float(), -1)
                p_first = torch.softmax(last[:, first_ids], -1)
                mass = last[:, first_ids].exp().sum(-1)
                for b, it in enumerate(chunk):
                    pv = p_first[b].tolist()
                    fh.write(json.dumps({
                        'domain': it['domain'], 'claim_id': it['claim_id'], 'pmid': str(it['pmid']),
                        'year': it['year'], 'archived': it['archived'], 'archived_p': it['archived_p'],
                        'label_first': LABELS[max(range(4), key=lambda j: pv[j])],
                        'p_first': [round(x, 5) for x in pv],
                        'first_token_mass': round(float(mass[b]), 5),
                        'has_abstract': bool((it['abstract'] or '').strip())}) + '\n')
                n += len(chunk)
                if (i // a.batch) % 20 == 0:
                    fh.flush()
                    el = max(time.time() - t0, 1e-9)
                    print(f'  {n:,}/{len(mine):,}  {n/el:.1f} prompts/s  '
                          f'eta {(len(mine)-n)/max(n/el,1e-9)/60:.1f} min', flush=True)
            fh.close()
            print(f'DONE {n:,} prompts in {round(time.time()-t0)} s', flush=True)

    # ---------------- scoring ----------------
    recs = []
    for f in sorted(glob.glob(os.path.join(outd, '*.jsonl'))):
        for line in open(f):
            if line.strip():
                recs.append(json.loads(line))
    if len(recs) < len(sample):
        print(f'\n{len(recs):,}/{len(sample):,} scored -- not scoring the probe yet')
        return 0
    seen = {}
    for r in recs:
        seen[(r['domain'], r['claim_id'], r['pmid'])] = r
    recs = [seen[(it['domain'], it['claim_id'], str(it['pmid']))] for it in sample
            if (it['domain'], it['claim_id'], str(it['pmid'])) in seen]

    out = dict(config_sha256=rec['config_sha256'], sample_sha256=rec['sample_sha256'],
               thresholds=THRESH, n=len(recs))
    print(f'\n{"="*96}\nT5 PROBE -- re-run deployed pipeline vs archived labels   {len(recs):,} associations\n{"="*96}')
    for tag, sub in (('External-1', [r for r in recs if r['domain'] == 1]),
                     ('External-2', [r for r in recs if r['domain'] == 2]),
                     ('POOLED', recs)):
        if not sub:
            continue
        arch = [r['archived'] for r in sub]
        new = [r['label_first'] for r in sub]
        k, po = kappa(arch, new)
        lo, hi = boot_kappa(arch, new, [r['claim_id'] for r in sub])
        dp = [max(abs(x - y) for x, y in zip(r['p_first'], r['archived_p']))
              for r in sub if r['archived_p'] and all(v is not None for v in r['archived_p'])]
        verdict = ('equivalent' if k >= THRESH['equivalent']
                   else 'minor deviation' if k >= THRESH['minor'] else 'LIMITATION')
        print(f'\n  {tag}   n={len(sub):,}')
        print(f'    raw agreement  {po:.4f} ({sum(x==y for x,y in zip(arch,new)):,}/{len(sub):,})')
        print(f"    Cohen's kappa  {k:.4f}  [{lo:.4f}, {hi:.4f}]   -> {verdict}")
        if dp:
            print(f'    max |delta p| over {len(dp):,} with an archived vector: '
                  f'median {np.median(dp):.5f}  p95 {np.percentile(dp,95):.5f}  max {max(dp):.5f}')
        hdr = 'archived \\ re-run'
        print('    ' + f'{hdr:20s}' + ''.join(f'{l[:7]:>9s}' for l in LABELS))
        for la in LABELS:
            row = [sum(1 for r in sub if r['archived'] == la and r['label_first'] == lb)
                   for lb in LABELS]
            print(f'    {la:20s}' + ''.join(f'{v:9,d}' for v in row))
        out[tag] = dict(n=len(sub), raw_agreement=po, kappa=k, kappa_ci=[lo, hi], verdict=verdict,
                        confusion={la: {lb: sum(1 for r in sub if r['archived'] == la
                                                and r['label_first'] == lb) for lb in LABELS}
                                   for la in LABELS},
                        max_delta_p=dict(median=float(np.median(dp)), p95=float(np.percentile(dp, 95)),
                                         max=float(max(dp))) if dp else None,
                        n_with_abstract=sum(1 for r in sub if r['has_abstract']))
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'T5_probe.json'), 'w'), indent=1, default=float)
    print('\nwritten results/T5_probe.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
