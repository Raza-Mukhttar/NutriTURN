"""E9a / E10: language-model forecasters on the origin-defined calendar cohort.

Temporal validity: the prompt shows only records published before the origin; the evaluated units
are the accrued candidates of the calendar cohort; nothing is fitted, so there is no training set
to keep claim-disjoint.

Variants
  digest   pre-origin label counts and year-by-year history only          (E10)
  text     the same digest plus the 8 most recent pre-origin abstracts    (E9a)
  masked   the text variant with every exposure and outcome term in the prompt and in the
           abstracts replaced by EXPOSURE and OUTCOME                     (hindsight control)
Each variant runs named and blinded, under both option orders, and the reported probability is the
average over the two orders.

  python analyses/llm_text_forecaster.py score --model llama --variant text --gpu 0
  python analyses/llm_text_forecaster.py eval

The 70B-class models named in the plan are not present in the local cache and cannot be fetched
here, so the forecasters run on the two 7-8B models the project already holds. Their results bound
that model class only.
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
import os, sys, json, argparse, re, time, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

MODELS = {'llama': 'meta-llama/Llama-3.1-8B-Instruct',
          'qwen': 'Qwen/Qwen2.5-7B-Instruct',
          # 70B-class checkpoints, pre-quantised to 4-bit (bitsandbytes NF4). Full-precision
          # weights are 141-145 GB and do not fit on this machine; the paper's own 70B-class
          # digest forecasters also used pre-quantised 4-bit checkpoints.
          'llama70': 'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit',
          'qwen72': 'unsloth/Qwen2.5-72B-Instruct-bnb-4bit'}
QUANTISED = {'llama70', 'qwen72'}
K_ABS = 8
ABS_CHARS = 900
OUT = os.path.join(N.BASE, 'annotations', 'forecast')

SYS = ('You forecast how a body of scientific evidence will develop after a historical moment. '
       'You see only studies published before the cutoff year. When the studies published after the '
       'cutoff are pooled, will the predominant direction of their protective and harmful findings be '
       'the opposite of the predominant direction before the cutoff? Answer with exactly one letter: '
       '{A} if the predominant direction will reverse; {B} if the predominant direction will stay the '
       'same. Answer with the letter only.')


def digest(rows, t):
    pre = [r for r in rows if r['year'] < t]
    lab = [r['label_primary'] for r in pre]
    n = len(pre)
    c = {k: lab.count(k) for k in ('PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR')}
    yrs = sorted({r['year'] for r in pre})
    per = {}
    for r in pre:
        d = per.setdefault(r['year'], [0, 0, 0])
        if r['label_primary'] == 'PROTECTIVE': d[0] += 1
        elif r['label_primary'] == 'HARMFUL': d[1] += 1
        elif r['label_primary'] == 'NULL': d[2] += 1
    show = yrs if len(yrs) <= 28 else yrs[:8] + yrs[-20:]
    lines = [f'  {y}: protective {per.get(y,[0,0,0])[0]}, harmful {per.get(y,[0,0,0])[1]}, '
             f'null {per.get(y,[0,0,0])[2]}' for y in show]
    sg = c['PROTECTIVE'] + c['HARMFUL']
    share = max(c['PROTECTIVE'], c['HARMFUL']) / sg if sg else 0.0
    rec = [r for r in pre if r['year'] >= t - 6]
    rs = sum(1 for r in rec if r['label_primary'] in ('PROTECTIVE', 'HARMFUL'))
    rshare = (max(sum(1 for r in rec if r['label_primary'] == 'PROTECTIVE'),
                  sum(1 for r in rec if r['label_primary'] == 'HARMFUL')) / rs) if rs else 0.0
    return (f'Studies before the cutoff: {n}\n'
            f'  protective {c["PROTECTIVE"]}, harmful {c["HARMFUL"]}, null {c["NULL"]}, '
            f'unclear {c["UNCLEAR"]}\n'
            f'Years covered: {yrs[0] if yrs else "?"} to {yrs[-1] if yrs else "?"}\n'
            f'Largest directional share overall: {share:.2f}\n'
            f'Largest directional share in the last six years: {rshare:.2f}\n'
            f'Year-by-year counts:\n' + '\n'.join(lines))


def mask_terms(text, exp, outc):
    for term, tag in ((exp, 'EXPOSURE'), (outc, 'OUTCOME')):
        for w in sorted({term} | set(term.split()), key=len, reverse=True):
            if len(w) >= 4:
                text = re.sub(re.escape(w), tag, text, flags=re.I)
    return text


def build_prompt(variant, named, order, cid, t, rows, exp, outc, abstracts):
    a_lab, b_lab = ('A', 'B') if order == 0 else ('B', 'A')
    sysmsg = SYS.format(A=a_lab, B=b_lab)
    claim = (f'Claim: does {exp} affect {outc}?' if named
             else 'Claim: does an exposure (name withheld) affect a health outcome (name withheld)?')
    body = digest(rows, t)
    if variant in ('text', 'masked'):
        parts = []
        for r, txt in abstracts:
            tt = txt[:ABS_CHARS]
            if variant == 'masked':
                tt = mask_terms(tt, exp, outc)
            parts.append(f'  ({r["year"]}) {tt}')
        body += '\n\nMost recent studies before the cutoff:\n' + '\n'.join(parts)
    if variant == 'masked' and named:
        claim = 'Claim: does EXPOSURE affect OUTCOME?'
    user = (f'{claim}\nCutoff year: {t}\n\n{body}\n\nWill the predominant direction reverse after {t}?')
    return sysmsg, user, a_lab


def cmd_score(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    repo = MODELS[args.model]
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, f'{args.model}_{args.variant}_shard{args.shard}.jsonl')
    A = N.assoc()
    by = {}
    for r in A:
        by.setdefault(r['claim_id'], []).append(r)
    Q = json.load(open(os.path.join(N.SNAPD, 'claim_queries.json')))
    abst = {}
    if args.variant in ('text', 'masked'):
        for p in sorted(glob.glob(os.path.join(N.SNAPD, 'records', '*.json'))):
            d = json.load(open(p))
            abst[d['claim_id']] = {str(r['pmid']): ((r.get('title') or '') + '. ' + (r.get('abstract') or '')).strip()
                                   for r in d['records']}
    co = N.build_cohort('primary')
    units = [r for r in co if r['accrued']]
    units = units[args.shard::args.nshards]
    if args.limit:
        units = units[:args.limit]
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            d = json.loads(line); done.add((d['claim_id'], d['origin'], d['horizon']))
        units = [u for u in units if (u['claim_id'], u['origin'], u['horizon']) not in done]
    print(f'[{args.model} {args.variant} shard {args.shard}] {len(units)} units ({len(done)} done)', flush=True)
    if not units:
        return
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.model in QUANTISED:
        # the checkpoint carries its own quantization_config; do not override the dtype
        model = AutoModelForCausalLM.from_pretrained(repo, device_map='cuda:0').eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16,
                                                     device_map='cuda:0').eval()
    idA = tok.encode('A', add_special_tokens=False)[0]
    idB = tok.encode('B', add_special_tokens=False)[0]
    fh = open(out_path, 'a'); t0 = time.time()
    for i, u in enumerate(units):
        cid, t = u['claim_id'], u['origin']
        rows = by[cid]; exp, outc = Q[cid]['exposure'], Q[cid]['outcome']
        pre = sorted([r for r in rows if r['year'] < t], key=lambda r: r['year'])[-K_ABS:]
        pairs = [(r, abst.get(cid, {}).get(r['pmid'], '')) for r in pre] if args.variant != 'digest' else []
        rec = {'claim_id': cid, 'origin': t, 'horizon': u['horizon'], 'y': u['y'],
               'model': args.model, 'variant': args.variant}
        for named in (True, False):
            ps = []
            for order in (0, 1):
                sysmsg, user, a_lab = build_prompt(args.variant, named, order, cid, t, rows,
                                                   exp, outc, pairs)
                msg = [{'role': 'system', 'content': sysmsg}, {'role': 'user', 'content': user}]
                s = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                enc = tok(s, return_tensors='pt', truncation=True, max_length=3500,
                          add_special_tokens=False).to('cuda:0')
                with torch.no_grad():
                    h = model.model(**enc).last_hidden_state[:, -1, :]
                    lg = model.lm_head(h).float()[0]
                pa, pb = torch.softmax(torch.stack([lg[idA], lg[idB]]), 0).tolist()
                # a_lab is the letter meaning "will reverse" in this order
                p_rev = pa if a_lab == 'A' else pb
                ps.append(p_rev)
                rec[f'mass_{"named" if named else "blind"}_{order}'] = round(
                    float(torch.exp(lg[idA]) + torch.exp(lg[idB])), 5)
            rec[f'p_reverse_{"named" if named else "blind"}'] = float(np.mean(ps))
            rec[f'p_order_spread_{"named" if named else "blind"}'] = float(abs(ps[0] - ps[1]))
        fh.write(json.dumps(rec) + '\n')
        if i % 50 == 0:
            fh.flush()
            el = time.time() - t0
            print(f'  {i+1}/{len(units)}  {(i+1)/max(el,1e-9):.2f} units/s  '
                  f'eta {(len(units)-i-1)/max((i+1)/max(el,1e-9),1e-9)/60:.0f} min', flush=True)
    fh.close()
    print(f'DONE {args.model} {args.variant} shard {args.shard}: {len(units)} units '
          f'in {time.time()-t0:.0f} s', flush=True)


def cmd_eval(args):
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, '*.jsonl'))):
        for line in open(p):
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit('no forecast files; run "score" first')
    co = {(r['claim_id'], r['origin'], r['horizon']): r for r in N.build_cohort('primary') if r['accrued']}
    out = {'cells': [], 'note': 'Llama-3.1-8B and Qwen2.5-7B in bf16; Llama-3.1-70B and Qwen2.5-72B as pre-quantised 4-bit checkpoints, since full-precision 70B weights (141-145 GB) exceed the available disk. The paper used 4-bit checkpoints for its own 70B-class forecasters'}
    print('E9a / E10 language-model forecasters on the calendar cohort\n')
    keys = sorted({(r['model'], r['variant']) for r in rows})
    for H in (3, 5, 10):
        base = [r for r in rows if r['horizon'] == H]
        if not base:
            continue
        ref = {}
        for m, v in keys:
            sel = [r for r in base if r['model'] == m and r['variant'] == v]
            if len(sel) < 30:
                continue
            key = [(r['claim_id'], r['origin']) for r in sel]
            y = np.array([r['y'] for r in sel]); g = np.array([r['claim_id'] for r in sel])
            sm = np.array([1 - abs(co[(r['claim_id'], r['origin'], H)]['p']) for r in sel])
            for named in ('named', 'blind'):
                s = np.array([r[f'p_reverse_{named}'] for r in sel])
                d, lo, hi, _ = N.paired_ci(y, s, sm, g)
                cell = dict(horizon=H, model=m, variant=v, framing=named, units=len(y),
                            events=int(y.sum()), auroc=N.auroc(y, s), auprc=N.auprc(y, s),
                            margin_auroc=N.auroc(y, sm), minus_margin=[d, lo, hi],
                            established=N.established(lo, hi),
                            mean_order_spread=float(np.mean([r[f'p_order_spread_{named}'] for r in sel])))
                out['cells'].append(cell)
                ref[(m, v, named)] = (y, g, s, sm, key)
                print(f"  H{H} {m:6s} {v:7s} {named:6s} units {len(y):4d} events {int(y.sum()):3d}  "
                      f"AUROC {cell['auroc']:.3f} (margin {cell['margin_auroc']:.3f})  "
                      f"minus margin {d:+.3f} {N.fmt_ci(lo,hi)}"
                      f"{'  ESTABLISHED' if cell['established'] else ''}  "
                      f"order spread {cell['mean_order_spread']:.3f}")
        # text minus digest, on the shared units
        for m in {k[0] for k in keys}:
            for named in ('named', 'blind'):
                a = ref.get((m, 'text', named)); b = ref.get((m, 'digest', named))
                if not a or not b:
                    continue
                ka = {k: i for i, k in enumerate(a[4])}; kb = {k: i for i, k in enumerate(b[4])}
                sh = sorted(set(ka) & set(kb))
                if len(sh) < 30:
                    continue
                ia = np.array([ka[k] for k in sh]); ib = np.array([kb[k] for k in sh])
                y = a[0][ia]; g = a[1][ia]
                d, lo, hi, _ = N.paired_ci(y, a[2][ia], b[2][ib], g)
                out['cells'].append(dict(horizon=H, model=m, variant='text minus digest',
                                         framing=named, units=len(sh), events=int(y.sum()),
                                         minus_margin=[d, lo, hi], established=N.established(lo, hi)))
                print(f"  H{H} {m:6s} text minus digest {named:6s}: {d:+.3f} {N.fmt_ci(lo,hi)}"
                      f"{'  ESTABLISHED' if N.established(lo,hi) else ''}")
    print('written', N.save_json(out, 'E9a_E10_llm_forecasters.json'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    s.add_argument('--model', required=True, choices=list(MODELS))
    s.add_argument('--variant', required=True, choices=('digest', 'text', 'masked'))
    s.add_argument('--gpu', default='0'); s.add_argument('--shard', type=int, default=0)
    s.add_argument('--nshards', type=int, default=1); s.add_argument('--limit', type=int, default=0)
    sub.add_parser('eval')
    a = ap.parse_args()
    {'score': cmd_score, 'eval': cmd_eval}[a.cmd](a)
