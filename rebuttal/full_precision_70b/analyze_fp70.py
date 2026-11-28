"""Full-precision 70B cell: every 70B-class quantity of the paper, recomputed for bfloat16 weights.

The same functions are applied to two Llama-3.1-70B cells:
  4-bit   data_release/03_annotations/large_cells/cell_llama70.jsonl  (the cell of the paper)
  bf16    output/cell_llama70_bf16.jsonl                              (annotate_fp70.py)
The 4-bit numbers are first checked against the values printed in the paper; the bf16 numbers are
then the like-for-like replacements. A final block compares the two cells directly, which isolates
the effect of weight precision (model, prompt, read-out, text and truncation are identical).

  python analyze_fp70.py /path/to/NutriTURN-Repository [bf16_cell.jsonl]

CPU only, a few minutes. Writes output/fp70_numbers.json and output/fp70_log.txt.
Intervals: paired claim-clustered percentile bootstrap, 2,000 resamples, seed 3, as in the paper.
"""
import collections
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('NUTRITURN_ROOT', '.'))
if not os.path.isfile(os.path.join(ROOT, '.nutriturn_root')):
    raise SystemExit('pass the repository root (the folder holding .nutriturn_root)')
for p in ('audit_pipeline/lib', 'data_preprocessing/lib', 'audit_experiments',
          'audit_experiments/figures', 'data_annotation', 'data_preparation', ''):
    sys.path.insert(0, os.path.join(ROOT, *p.split('/')) if p else ROOT)
import nfx_common as N
import core_reanalyses as C

HERE = os.path.dirname(os.path.abspath(__file__))
BF16 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'output', 'cell_llama70_bf16.jsonl')
REL = os.path.join(ROOT, 'data_release', '03_annotations')
ANCHOR = os.path.join(ROOT, 'data', 'derived', 'numeric_anchor_v2_high_confidence.csv')
GATE, SEED, BOOT = N.SPARSE_GATE, N.SEED, N.BOOT
LABELS = ('PROTECTIVE', 'HARMFUL', 'NULL', 'UNCLEAR')
N_ASSOC = 112453

LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def load_cell(path, field='label_full'):
    """First occurrence of a key wins, the duplicate rule of the paper."""
    lab, prob = {}, {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        k = (r['claim_id'], str(r['pmid']))
        if k not in lab:
            lab[k] = r[field]
            prob[k] = r.get('p_full')
    return lab, prob


def load_released(name):
    out = {}
    for line in open(os.path.join(REL, 'controlled_cells', f'cell{name}.jsonl')):
        r = json.loads(line)
        out[(r['claim_id'], str(r['pmid']))] = r['label']
    return out


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[c] / n * cb[c] / n for c in set(ca) | set(cb))
    return (po - pe) / (1 - pe), po


def unit_table(lab):
    """nfx_common.unit_table for an arbitrary label column: same 1,012 units, only labels change."""
    per = {}
    for r in N.assoc():
        k = (r['claim_id'], r['pmid'])
        per.setdefault(r['claim_id'], []).append((r['year'], N.CODE.get(lab.get(k, ''), 99)))
    S = {c: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for c, v in per.items()}
    rows = []
    for u in csv.DictReader(open(os.path.join(N.DATA, 'units.csv'))):
        cid, t = u['claim_id'], int(u['cutoff'])
        yrs, codes = S[cid]
        pre = (yrs < t) & (codes != 99)
        post = (yrs >= t) & (codes != 99)
        h = int((codes[pre] == 1).sum())
        l = int((codes[pre] == -1).sum())
        n0 = int((codes[pre] == 0).sum())
        n = h + l
        r_t = (h - l) / n if n else 0.0
        p_t = r_t if n >= GATE else 0.0
        s_t = N.sgn0(p_t)
        resolved = h + l + n0
        null_share = n0 / resolved if resolved else 0.0
        a_t = 0.0 if n < GATE else (null_share if null_share >= 0.5 else (1 + abs(p_t)) / 2)
        ph = int((codes[post] == 1).sum())
        pl = int((codes[post] == -1).sum())
        m = ph + pl
        p_post = (ph - pl) / m if m else 0.0
        rows.append(dict(claim_id=cid, cutoff=t, h=h, l=l, n0=n0, n_signed=n, r_t=r_t, p=p_t,
                         s=s_t, a=a_t, null_share=null_share, m_signed=m, p_post=p_post,
                         y=int(N.sgn0(p_post) != s_t), sparse=int(n < GATE),
                         c=0.0 if n < GATE else (1 + abs(p_t)) / 2))
    for r, acc in zip(rows, N.build_cohort_accrual5(rows)):
        r['accrual_5y'] = acc
    return rows


KEY = lambda r: (r['claim_id'], r['cutoff'])


def overlap(ta, tb, mask=None, gate_non_event=False):
    def ev(t):
        return {KEY(r) for r in t if r['y'] and not (gate_non_event and r['sparse'])
                and (mask is None or KEY(r) in mask)}
    ea, eb = ev(ta), ev(tb)
    keys = [KEY(r) for r in ta if mask is None or KEY(r) in mask]
    ek, _ = kappa([k in ea for k in keys], [k in eb for k in keys])
    i, u = len(ea & eb), len(ea | eb)
    return dict(shared=i, union=u, jaccard=i / max(u, 1), event_kappa=ek)


def ci(d, lo, hi):
    return dict(delta=float(d), ci=[float(lo), float(hi)], established=bool(lo > 0 or hi < 0))


def fmt(c):
    return f"{c['delta']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]" + (' *' if c['established'] else '')


def summarise(name, lab, others, anchor_rows):
    """Every 70B-class quantity of the paper for one Llama-70B label column."""
    out = {}
    keys = sorted(lab)
    cnt = collections.Counter(lab.values())
    out['n_records'] = len(lab)
    out['label_share_pct'] = {x: 100 * cnt[x] / len(lab) for x in LABELS}

    t = unit_table(lab)
    out['units'] = len(t)
    out['events'] = int(sum(r['y'] for r in t))
    out['gated_units'] = int(sum(r['sparse'] for r in t))
    out['events_from_gated_units'] = int(sum(r['y'] for r in t if r['sparse']))

    sel = [(k, a) for k, a in anchor_rows if k in lab]
    y = [a for _, a in sel]
    q = [lab[k] for k, _ in sel]
    res = [(a, b) for a, b in zip(y, q) if b != 'UNCLEAR']
    rec = []
    for c in ('PROTECTIVE', 'HARMFUL', 'NULL'):
        tp = sum(1 for a, b in zip(y, q) if a == c and b == c)
        fn = sum(1 for a, b in zip(y, q) if a == c and b != c)
        rec.append(tp / (tp + fn) if tp + fn else float('nan'))
    out['anchor'] = dict(n=len(sel), agree_all=float(np.mean([a == b for a, b in zip(y, q)])),
                         resolved_pct=100 * len(res) / len(sel),
                         agree_resolved=float(np.mean([a == b for a, b in res])),
                         balanced_accuracy=float(np.nanmean(rec)))

    tabs = {nm: unit_table(l) for nm, l in others.items()}
    four = [tabs['primary'], tabs['second'], t, tabs['qwen72']]
    cs = set(KEY(r) for r in t)
    for tt in four:
        cs &= {KEY(r) for r in tt if r['n_signed'] >= GATE}
    out['common_support_units'] = len(cs)
    out['events_on_common_support'] = int(sum(r['y'] for r in t if KEY(r) in cs))
    pairs = {}
    for nm, l in others.items():
        ks = [k for k in keys if k in l]
        kp, po = kappa([lab[k] for k in ks], [l[k] for k in ks])
        pairs[nm] = dict(label_kappa=kp, raw_agreement=po, full=overlap(tabs[nm], t),
                         gate_non_event=overlap(tabs[nm], t, gate_non_event=True),
                         common_support=overlap(tabs[nm], t, mask=cs))
    out['pairs'] = pairs

    con = {}
    stat_full = C._stationary_all(t)
    for scope, mask in (('all', None), ('common_support', np.array([KEY(r) in cs for r in t]))):
        iA, SA, yy, g = N.loco_two_var(t, features=('p', 'a'), subset=mask)
        iC, SC, _, _ = N.loco_two_var(t, features=('p', 'c'), subset=mask)
        assert list(iA) == list(iC)
        marg = np.array([1 - abs(t[i]['p']) for i in iA])
        stat = stat_full if mask is None else stat_full[mask]
        con[scope] = dict(units=int(len(yy)), events=int(yy.sum()),
                          auroc_A=float(N.auroc(yy, SA)), auroc_C=float(N.auroc(yy, SC)),
                          auroc_margin=float(N.auroc(yy, marg)), auroc_stationary=float(N.auroc(yy, stat)),
                          A_minus_margin=ci(*N.paired_ci(yy, SA, marg, g, n=BOOT, seed=SEED)[:3]),
                          A_minus_stationary=ci(*N.paired_ci(yy, SA, stat, g, n=BOOT, seed=SEED)[:3]),
                          C_minus_margin=ci(*N.paired_ci(yy, SC, marg, g, n=BOOT, seed=SEED)[:3]),
                          C_minus_stationary=ci(*N.paired_ci(yy, SC, stat, g, n=BOOT, seed=SEED)[:3]),
                          C_minus_A=ci(*N.paired_ci(yy, SC, SA, g, n=BOOT, seed=SEED)[:3]))
    out['contrasts'] = con

    say(f'\n=== Llama-3.1-70B, {name} ===')
    say('records', f"{out['n_records']:,}", ' label share %',
        ' '.join(f"{x} {out['label_share_pct'][x]:.1f}" for x in LABELS))
    say(f"events {out['events']} of {out['units']} units; gated units {out['gated_units']}, "
        f"events from gated units {out['events_from_gated_units']}; "
        f"common support {out['common_support_units']} units, {out['events_on_common_support']} events")
    a = out['anchor']
    say(f"anchor (n {a['n']:,}): agree all {a['agree_all']:.4f}  resolved {a['resolved_pct']:.1f}%  "
        f"agree resolved {a['agree_resolved']:.4f}  balanced accuracy {a['balanced_accuracy']:.4f}")
    say(f"{'pair':28s}{'label k':>9s}{'raw':>8s}{'full':>16s}{'ev k':>7s}{'gate non-ev':>16s}{'common sup':>16s}")
    for nm, p in pairs.items():
        f_, g_, c_ = p['full'], p['gate_non_event'], p['common_support']
        say(f"{nm + ' -- Llama-70B':28s}{p['label_kappa']:9.4f}{p['raw_agreement']:8.4f}"
            f"{f_['shared']:6d}/{f_['union']:<3d}={f_['jaccard']:.3f}{f_['event_kappa']:7.3f}"
            f"{g_['shared']:6d}/{g_['union']:<3d}={g_['jaccard']:.3f}"
            f"{c_['shared']:6d}/{c_['union']:<3d}={c_['jaccard']:.3f}")
    for scope, c in con.items():
        say(f"contrasts, {scope} ({c['units']} units, {c['events']} events): AUROC A {c['auroc_A']:.4f}  "
            f"C {c['auroc_C']:.4f}  margin {c['auroc_margin']:.4f}  stationary {c['auroc_stationary']:.4f}")
        for k in ('A_minus_margin', 'A_minus_stationary', 'C_minus_margin', 'C_minus_stationary', 'C_minus_A'):
            say(f"   {k:20s} {fmt(c[k])}")
    return out, t


# values printed in the paper for the 4-bit cell: (path into the summary, published value, tolerance)
PUBLISHED = [
    (('label_share_pct', 'PROTECTIVE'), 36.7, .05), (('label_share_pct', 'HARMFUL'), 18.1, .05),
    (('label_share_pct', 'NULL'), 14.1, .05), (('label_share_pct', 'UNCLEAR'), 31.1, .05),
    (('events',), 55, 0), (('gated_units',), 21, 0), (('events_from_gated_units',), 21, 0),
    (('common_support_units',), 987, 0), (('events_on_common_support',), 33, 0),
    (('anchor', 'n'), 1841, 0), (('anchor', 'agree_all'), .6855, 5e-5),
    (('anchor', 'resolved_pct'), 90.4, .05), (('anchor', 'agree_resolved'), .7584, 5e-5),
    (('anchor', 'balanced_accuracy'), .6553, 5e-5),
    (('pairs', 'primary', 'label_kappa'), .4056, 5e-5), (('pairs', 'primary', 'raw_agreement'), .5626, 5e-5),
    (('pairs', 'primary', 'full', 'shared'), 24, 0), (('pairs', 'primary', 'full', 'union'), 102, 0),
    (('pairs', 'primary', 'full', 'event_kappa'), .341, 5e-4),
    (('pairs', 'primary', 'common_support', 'shared'), 17, 0), (('pairs', 'primary', 'common_support', 'union'), 79, 0),
    (('pairs', 'second', 'label_kappa'), .3843, 5e-5), (('pairs', 'second', 'raw_agreement'), .5316, 5e-5),
    (('pairs', 'second', 'full', 'shared'), 26, 0), (('pairs', 'second', 'full', 'union'), 81, 0),
    (('pairs', 'second', 'common_support', 'shared'), 16, 0), (('pairs', 'second', 'common_support', 'union'), 57, 0),
    (('pairs', 'qwen72', 'label_kappa'), .6148, 5e-5), (('pairs', 'qwen72', 'raw_agreement'), .7242, 5e-5),
    (('pairs', 'qwen72', 'full', 'shared'), 39, 0), (('pairs', 'qwen72', 'full', 'union'), 74, 0),
    (('pairs', 'qwen72', 'full', 'event_kappa'), .672, 5e-4),
    (('pairs', 'qwen72', 'gate_non_event', 'shared'), 20, 0), (('pairs', 'qwen72', 'gate_non_event', 'union'), 51, 0),
    (('pairs', 'qwen72', 'common_support', 'shared'), 20, 0), (('pairs', 'qwen72', 'common_support', 'union'), 49, 0),
    (('pairs', 'qwen72', 'common_support', 'event_kappa'), .565, 5e-4),
    (('pairs', 'llama8_P2', 'label_kappa'), .456, 5e-4),
    (('pairs', 'llama8_P2', 'full', 'shared'), 26, 0), (('pairs', 'llama8_P2', 'full', 'union'), 104, 0),
    (('pairs', 'llama8_P2', 'gate_non_event', 'shared'), 19, 0), (('pairs', 'llama8_P2', 'gate_non_event', 'union'), 85, 0),
    (('contrasts', 'all', 'auroc_A'), .8985, 5e-5), (('contrasts', 'all', 'auroc_C'), .9651, 5e-5),
    (('contrasts', 'all', 'A_minus_margin', 'delta'), -.0642, 5e-5),
    (('contrasts', 'all', 'A_minus_stationary', 'delta'), -.0782, 5e-5),
    (('contrasts', 'all', 'A_minus_stationary', 'ci', 0), -.140, 5e-4),
    (('contrasts', 'all', 'A_minus_stationary', 'ci', 1), -.013, 5e-4),
    (('contrasts', 'all', 'C_minus_margin', 'delta'), .0023, 5e-5),
    (('contrasts', 'all', 'C_minus_stationary', 'delta'), -.0116, 5e-5),
    (('contrasts', 'all', 'C_minus_A', 'delta'), .0666, 5e-5),
    (('contrasts', 'all', 'C_minus_A', 'ci', 0), -.0038, 5e-5), (('contrasts', 'all', 'C_minus_A', 'ci', 1), .1337, 5e-5),
    (('contrasts', 'common_support', 'auroc_A'), .8472, 5e-5), (('contrasts', 'common_support', 'auroc_C'), .9408, 5e-5),
    (('contrasts', 'common_support', 'C_minus_A', 'delta'), .0936, 5e-5),
]


def get(d, path):
    for p in path:
        d = d[p]
    return d


def main():
    primary, second = N.label_column('primary'), N.label_column('second')
    q72, _ = load_cell(os.path.join(REL, 'large_cells', 'cell_qwen72.jsonl'))
    l70_4, p4 = load_cell(os.path.join(REL, 'large_cells', 'cell_llama70.jsonl'))
    others = collections.OrderedDict(primary=primary, second=second, qwen72=q72,
                                     llama8_P2=load_released('B'))
    assoc = {(r['claim_id'], r['pmid']) for r in N.assoc()}
    anchor_rows = [((r['claim_id'], r['pmid']), r['anchor'])
                   for r in csv.DictReader(open(ANCHOR)) if (r['claim_id'], r['pmid']) in assoc]

    res = {}
    res['4bit'], t4 = summarise('4-bit NF4 weights (the cell of the paper)', l70_4, others, anchor_rows)
    bad = []
    for path, pub, tol in PUBLISHED:
        v = get(res['4bit'], path)
        if abs(v - pub) > tol + 1e-12:
            bad.append((path, pub, v))
    say(f'\n4-bit cell against the paper: {len(PUBLISHED) - len(bad)} of {len(PUBLISHED)} published values reproduced')
    for path, pub, v in bad:
        say('   MISMATCH', '.'.join(map(str, path)), 'published', pub, 'recomputed', round(v, 5))
    res['published_check'] = dict(n=len(PUBLISHED), reproduced=len(PUBLISHED) - len(bad),
                                  mismatches=[['.'.join(map(str, p)), a, float(b)] for p, a, b in bad])

    if not os.path.exists(BF16):
        say(f'\nno full-precision cell at {BF16}; only the 4-bit check was run')
        return finish(res, 2)
    l70_f, pf = load_cell(BF16)
    missing = len(assoc - set(l70_f))
    say(f'\nfull-precision cell: {len(l70_f):,} records, {missing:,} of {N_ASSOC:,} associations missing')
    res['bf16_complete'] = missing == 0 and len(l70_f) == N_ASSOC
    res['bf16'], tf = summarise('bfloat16 weights (official checkpoint)', l70_f, others, anchor_rows)

    # precision effect: the two Llama-70B cells against each other
    ks = [k for k in sorted(l70_f) if k in l70_4]
    kp, po = kappa([l70_4[k] for k in ks], [l70_f[k] for k in ks])
    conf = collections.Counter((l70_4[k], l70_f[k]) for k in ks)
    dp = np.array([np.abs(np.array(p4[k]) - np.array(pf[k])).max() for k in ks if p4[k] and pf[k]])
    cs = {KEY(r) for r in t4 if r['n_signed'] >= GATE} & {KEY(r) for r in tf if r['n_signed'] >= GATE}
    res['precision_effect'] = dict(
        n=len(ks), label_kappa=kp, raw_agreement=po, labels_changed=int(sum(l70_4[k] != l70_f[k] for k in ks)),
        confusion_4bit_rows_bf16_cols=[[conf[(a, b)] for b in LABELS] for a in LABELS],
        max_abs_prob_diff=dict(mean=float(dp.mean()), median=float(np.median(dp)),
                               p95=float(np.quantile(dp, .95)), share_above_0_1=float((dp > .1).mean())),
        events=overlap(t4, tf), events_gate_non_event=overlap(t4, tf, gate_non_event=True),
        events_both_ungated=overlap(t4, tf, mask=cs), units_both_ungated=len(cs))
    pe = res['precision_effect']
    say('\n=== Precision effect: 4-bit against bfloat16, same model, prompt and read-out ===')
    say(f"records {pe['n']:,}: label kappa {kp:.4f}, raw agreement {po:.4f}, labels changed {pe['labels_changed']:,}")
    say('confusion (rows 4-bit, columns bf16; order ' + ', '.join(LABELS) + ')')
    for a, row in zip(LABELS, pe['confusion_4bit_rows_bf16_cols']):
        say(f'   {a:11s}' + ''.join(f'{v:9,d}' for v in row))
    m = pe['max_abs_prob_diff']
    say(f"largest probability change per record: mean {m['mean']:.4f}, median {m['median']:.4f}, "
        f"95th pct {m['p95']:.4f}, above 0.1 in {100 * m['share_above_0_1']:.1f}% of records")
    for k, lbl in (('events', 'all 1,012 units'), ('events_gate_non_event', 'gated history a non-event'),
                   ('events_both_ungated', f"{len(cs)} units ungated under both")):
        e = pe[k]
        say(f"events, {lbl}: shared {e['shared']}/{e['union']} = {e['jaccard']:.3f}, event kappa {e['event_kappa']:.3f}")
    say('reference from the paper: batch-size rerun of the 8B annotator, event Jaccard 0.846, label kappa 0.980')
    # scoring path: the first rows were also scored with the four-sequence code of the 4-bit cell
    ref = os.path.join(HERE, 'output', 'validation', 'full4_first6569.jsonl')
    if os.path.exists(ref):
        l70_r, pr = load_cell(ref)
        kk = [k for k in sorted(l70_r) if k in l70_f and k in l70_4]
        kpp, pop = kappa([l70_r[k] for k in kk], [l70_f[k] for k in kk])
        k4r, _ = kappa([l70_4[k] for k in kk], [l70_r[k] for k in kk])
        k4f, _ = kappa([l70_4[k] for k in kk], [l70_f[k] for k in kk])
        dq = np.array([np.abs(np.array(pr[k]) - np.array(pf[k])).max() for k in kk])
        res['scoring_path'] = dict(
            n=len(kk), bf16_full4_vs_bf16_prefix=dict(label_kappa=kpp, raw_agreement=pop,
                                                      labels_changed=int(sum(l70_r[k] != l70_f[k] for k in kk)),
                                                      max_abs_prob_diff_median=float(np.median(dq)),
                                                      max_abs_prob_diff_p95=float(np.quantile(dq, .95))),
            kappa_4bit_vs_bf16_full4=k4r, kappa_4bit_vs_bf16_prefix=k4f)
        sp = res['scoring_path']
        say(f"\n=== Scoring path, on the first {len(kk):,} associations scored both ways (bf16) ===")
        say(f"four-sequence vs cached-prefix scoring: label kappa {kpp:.4f}, labels changed "
            f"{sp['bf16_full4_vs_bf16_prefix']['labels_changed']:,}, largest probability change median "
            f"{np.median(dq):.4f}, 95th pct {np.quantile(dq, .95):.4f}")
        say(f"4-bit against bf16 on the same records: kappa {k4r:.4f} (four-sequence), {k4f:.4f} (cached prefix)")
    return finish(res, 0 if (not bad and res['bf16_complete']) else 1)


def finish(res, code):
    os.makedirs(os.path.join(HERE, 'output'), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, 'output', 'fp70_numbers.json'), 'w'), indent=1, default=float)
    open(os.path.join(HERE, 'output', 'fp70_log.txt'), 'w').write('\n'.join(LOG) + '\n')
    say(f'written output/fp70_numbers.json and output/fp70_log.txt (exit {code})')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
