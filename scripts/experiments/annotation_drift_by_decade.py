"""E13: annotation drift by decade, and a look-ahead probe.

drift  Cohen's kappa between the primary and second pipelines by publication decade; the label
       distribution by decade and by abstract-length quartile. A strong trend would mean the
       annotator behaves differently on older abstracts, which is a candidate source of the
       excess reversals.

probe  Re-annotate a random sample of associations with the PRIMARY configuration after masking
       every exposure and outcome term. The terms are replaced by EXPOSURE and OUTCOME
       consistently, in the abstract AND in the prompt's Exposure/Outcome fields, so the task
       stays well posed: the model must still locate the relation in the text, but cannot use
       knowledge of which exposure and outcome they are. Masking only the abstract would leave the
       prompt asking about a term the text no longer contains, which would inflate the change rate
       for a trivial reason. Report the share of labels that change, overall and for records
       published before versus after 2000. A larger change for older records would suggest that
       named-claim knowledge affects historical labels.

  python analyses/nfx_drift.py drift
  python analyses/nfx_drift.py probe --gpu 0 --n 2000
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
import os, sys, json, argparse, glob, re, time, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

LABELS = ['PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR']
SYS_PRIMARY = ('You classify the direction of association reported between an exposure and an outcome. '
               'Answer with exactly one word: PROTECTIVE if the exposure is reported to reduce the outcome, '
               'HARMFUL if it increases it, NULL if no association is reported, UNCLEAR if the study does not report '
               'this association.')
LLAMA = 'meta-llama/Llama-3.1-8B-Instruct'
TEXT_BUDGET, MAX_LEN = 3000, 1600


def kappa(a, b, cats):
    a = list(a); b = list(b); n = len(a)
    if n == 0:
        return float('nan')
    idx = {c: i for i, c in enumerate(cats)}
    M = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    po = np.trace(M) / n
    pe = (M.sum(0) / n * M.sum(1) / n).sum()
    return float((po - pe) / (1 - pe)) if pe < 1 else float('nan')


def texts():
    out = {}
    for p in sorted(glob.glob(os.path.join(N.SNAPD, 'records', '*.json'))):
        d = json.load(open(p))
        for r in d['records']:
            out[(d['claim_id'], str(r['pmid']))] = ((r.get('title') or '').strip(),
                                                    (r.get('abstract') or '').strip())
    return out


def cmd_drift(args):
    A = N.assoc()
    T = texts()
    print('E13 annotation drift\n')
    out = {'by_decade': [], 'by_abstract_length': []}
    dec = {}
    for r in A:
        d = (r['year'] // 10) * 10
        dec.setdefault(d, []).append(r)
    print('kappa between the primary and second pipelines, by publication decade')
    print(f"{'decade':>8s} {'records':>8s} {'kappa':>7s} {'agree':>7s}  primary label shares (P/H/N/U)")
    for d in sorted(dec):
        rows = [r for r in dec[d] if r['label_second']]
        if len(rows) < 200:
            continue
        a = [r['label_primary'] for r in rows]; b = [r['label_second'] for r in rows]
        k = kappa(a, b, LABELS)
        ag = float(np.mean([x == y for x, y in zip(a, b)]))
        sh = [round(a.count(l) / len(a), 3) for l in LABELS]
        out['by_decade'].append(dict(decade=int(d), records=len(rows), kappa=k, agreement=ag,
                                     primary_shares=dict(zip(LABELS, sh))))
        print(f'{d:>8d} {len(rows):>8d} {k:>7.3f} {ag:>7.3f}  {sh}')
    ks = [x['kappa'] for x in out['by_decade']]
    out['kappa_range'] = [min(ks), max(ks)]
    out['kappa_spread'] = max(ks) - min(ks)
    print(f'\nkappa range across decades: {min(ks):.3f} to {max(ks):.3f} (spread {max(ks)-min(ks):.3f})')

    L = []
    for r in A:
        t = T.get((r['claim_id'], r['pmid']), ('', ''))
        L.append(len(t[1]))
    q = np.percentile([x for x in L if x > 0], [25, 50, 75])
    print(f'\nabstract length quartile cut points: {q.round(0)}')
    print(f"{'quartile':>10s} {'records':>8s} {'kappa':>7s} {'agree':>7s}  primary label shares (P/H/N/U)")
    bins = [(-1, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 10 ** 9)]
    for bi, (lo, hi) in enumerate(bins, 1):
        rows = [r for r, n_ in zip(A, L) if lo < n_ <= hi and r['label_second']]
        if len(rows) < 200:
            continue
        a = [r['label_primary'] for r in rows]; b = [r['label_second'] for r in rows]
        k = kappa(a, b, LABELS); ag = float(np.mean([x == y for x, y in zip(a, b)]))
        sh = [round(a.count(l) / len(a), 3) for l in LABELS]
        out['by_abstract_length'].append(dict(quartile=bi, records=len(rows), kappa=k,
                                              agreement=ag, primary_shares=dict(zip(LABELS, sh))))
        print(f'{bi:>10d} {len(rows):>8d} {k:>7.3f} {ag:>7.3f}  {sh}')
    print('\nwritten', N.save_json(out, 'E13_drift.json'))


def cmd_probe(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    A = N.assoc()
    T = texts()
    Q = json.load(open(os.path.join(N.SNAPD, 'claim_queries.json')))
    pool = [r for r in A if T.get((r['claim_id'], r['pmid']), ('', ''))[1]]
    rng = random.Random(N.SEED)
    sample = rng.sample(pool, min(args.n, len(pool)))
    print(f'E13 look-ahead probe: {len(sample)} associations, primary configuration, names masked',
          flush=True)
    tok = AutoTokenizer.from_pretrained(LLAMA)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.bfloat16,
                                                 device_map='cuda:0').eval()
    first_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LABELS]

    def mask(text, exp, outc):
        for term, tag in ((exp, 'EXPOSURE'), (outc, 'OUTCOME')):
            for w in sorted({term} | set(term.split()), key=len, reverse=True):
                if len(w) >= 4:
                    text = re.sub(re.escape(w), tag, text, flags=re.I)
        return text

    recs = []
    B = 16
    t0 = time.time()
    for i in range(0, len(sample), B):
        chunk = sample[i:i + B]
        prompts = []
        for r in chunk:
            title, abstract = T[(r['claim_id'], r['pmid'])]
            exp, outc = Q[r['claim_id']]['exposure'], Q[r['claim_id']]['outcome']
            text = mask((title + '. ' + abstract)[:TEXT_BUDGET], exp, outc)
            msg = [{'role': 'system', 'content': SYS_PRIMARY},
                   {'role': 'user', 'content': f'Exposure: EXPOSURE\nOutcome: OUTCOME\n\nStudy: {text}\n\nDirection:'}]
            prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors='pt', padding=True, truncation=True,
                  max_length=MAX_LEN, add_special_tokens=False).to('cuda:0')
        with torch.no_grad():
            h = model.model(**enc).last_hidden_state[:, -1, :]
            lg = model.lm_head(h).float()
        sel = lg[:, first_ids]
        lab = sel.argmax(-1).tolist()
        for r, k in zip(chunk, lab):
            recs.append(dict(claim_id=r['claim_id'], pmid=r['pmid'], year=r['year'],
                             label_primary=r['label_primary'], label_masked=LABELS[k]))
        if (i // B) % 20 == 0:
            print(f'  {i+len(chunk)}/{len(sample)}  '
                  f'{(i+len(chunk))/max(time.time()-t0,1e-9):.1f}/s', flush=True)
    ch = [r for r in recs if r['label_masked'] != r['label_primary']]
    pre = [r for r in recs if r['year'] < 2000]; post = [r for r in recs if r['year'] >= 2000]
    f_all = len(ch) / len(recs)
    f_pre = np.mean([r['label_masked'] != r['label_primary'] for r in pre]) if pre else float('nan')
    f_post = np.mean([r['label_masked'] != r['label_primary'] for r in post]) if post else float('nan')
    # claim-clustered interval on the pre-minus-post difference
    g = np.array([r['claim_id'] for r in recs])
    d_ = np.array([r['label_masked'] != r['label_primary'] for r in recs], float)
    yr = np.array([r['year'] for r in recs])
    vals = []
    for ix in N.claim_boot_pairs(np.zeros(len(recs), int), g):
        a_ = d_[ix][yr[ix] < 2000]; b_ = d_[ix][yr[ix] >= 2000]
        if len(a_) and len(b_):
            vals.append(a_.mean() - b_.mean())
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    out = dict(sampled=len(recs), changed=len(ch), change_share=f_all,
               change_share_before_2000=float(f_pre), n_before_2000=len(pre),
               change_share_from_2000=float(f_post), n_from_2000=len(post),
               difference_before_minus_after=[float(f_pre - f_post), lo, hi],
               established=N.established(lo, hi))
    print(f'\nlabels changed when names are masked: {len(ch)}/{len(recs)} ({100*f_all:.1f}%)')
    print(f'  published before 2000: {100*f_pre:.1f}% of {len(pre)}')
    print(f'  published from 2000  : {100*f_post:.1f}% of {len(post)}')
    print(f'  difference (before minus after): {f_pre-f_post:+.3f} {N.fmt_ci(lo,hi)}'
          f"{'  ESTABLISHED' if out['established'] else '  not established'}")
    N.save_csv(recs, 'E13_probe_records.csv')
    print('written', N.save_json(out, 'E13_probe.json'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('drift')
    s = sub.add_parser('probe'); s.add_argument('--gpu', default='0'); s.add_argument('--n', type=int, default=2000)
    a = ap.parse_args()
    {'drift': cmd_drift, 'probe': cmd_probe}[a.cmd](a)
