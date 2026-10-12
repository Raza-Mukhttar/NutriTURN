"""Which pipeline is wrong outside diet? The numeric anchor applied to the external records.

The anchor is LLM-independent: every 95% CI is located, typed by the effect-measure phrase beside
it, attributed to the claim only when BOTH the exposure and outcome terms appear in the same
sentence (or a 220-character window), and a record is high-confidence only when every attributed
interval agrees in direction. All of that machinery is imported unchanged from
NutriTURN/scripts/pipeline/build_numeric_anchor.py -- the regexes CI, MEASURE, SENT, the STOP list,
the LEX lexicon and the functions patterns/design_of/measure_before.

MATERIAL CAVEAT, measured and reported below rather than assumed. The attribution lexicon LEX is
hand-built for the DIET corpus: its exposure keys are dietary (sugars, alcohol, coffee, vitamins,
dairy, meat, ...) and its outcome keys are the source corpus's outcomes. For an external claim such
as "ace inhibitors x blood pressure" or "air pollution x asthma", patterns() falls through to a
generic rule -- tokens of >= 4 characters that are not stop words, truncated to 5-character
prefixes. That is a weaker and blunter attribution than the source anchor enjoyed, so the external
anchor's YIELD and PRECISION are not comparable to the source's by construction. The script counts
exactly how many external terms get curated variants against generic prefixes, so the reader can
see the limitation instead of inferring it.
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

import collections, csv, glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NT = os.path.join(os.path.dirname(HERE), 'NutriTURN')
for p in ('lib', 'experiments', 'pipeline'):
    sys.path.insert(0, os.path.join(NT, 'scripts', p))
import nfx_common as N
import build_numeric_anchor as BA     # frozen regexes, lexicon and helpers

SRC = {1: ('/home/raza/gpt-nutri/revision_v2/data/external_benchmark/abstracts',
           '/home/raza/gpt-nutri/revision_v2/data/external_benchmark/directions',
           '/home/raza/gpt-nutri/revision_v2/data/external_benchmark/claim_queries.json'),
       2: ('/home/raza/gpt-nutri/revision_v2/revision2_round2/data/external2/abstracts',
           '/home/raza/gpt-nutri/revision_v2/revision2_round2/data/external2/directions',
           '/home/raza/gpt-nutri/revision_v2/revision2_round2/data/external2/claim_queries.json')}
CLS = ['PROTECTIVE', 'HARMFUL', 'NULL']


def second_labels():
    out = {}
    for f in glob.glob(os.path.join(HERE, 'data', 'annot_second', '*.jsonl')):
        for l in open(f):
            if l.strip():
                d = json.loads(l)
                out[(d['claim_id'], str(d['pmid']))] = d['label_full']
    return out


def build(which, claims):
    adir, ddir, qp = SRC[which]
    Q = json.load(open(qp))
    recs, reasons = [], collections.Counter()
    lex_hit = collections.Counter()
    n_rec = 0
    for cid in claims:
        q = Q.get(cid) or {}
        if not (q.get('exposure') and q.get('outcome')):
            continue
        for fld in ('exposure', 'outcome'):
            lex_hit[f'{fld}_curated' if q[fld].lower().strip() in BA.LEX[fld]
                    else f'{fld}_generic_prefix'] += 1
        ep = BA.patterns('exposure', q['exposure'])
        op = BA.patterns('outcome', q['outcome'])
        dp = os.path.join(ddir, f'{cid}.json')
        ap = os.path.join(adir, f'{cid}.json')
        if not (os.path.exists(dp) and os.path.exists(ap)):
            continue
        Dj = {str(d['pmid']): d for d in json.load(open(dp))['directions']}
        for r in json.load(open(ap)).get('records') or []:
            pm = str(r['pmid'])
            if pm not in Dj:
                continue
            n_rec += 1
            text = (r.get('title') or '') + '. ' + (r.get('abstract') or '')
            low = text.lower()
            bounds = [0] + [m.end() for m in BA.SENT.finditer(text)] + [len(text)]
            cis = list(BA.CI.finditer(text))
            rec = dict(claim_id=cid, pmid=pm, year=r.get('year'),
                       design=BA.design_of(r.get('pub_types')),
                       extractor=Dj[pm]['direction'], n_ci=len(cis), n_typed=0, n_attributed=0)
            if not cis:
                reasons['NO_CI'] += 1
                continue
            labels = []
            for m in cis:
                lo, hi = float(m.group(1)), float(m.group(2))
                typ, grp, conflict = BA.measure_before(text, m.start(), m.end())
                if hi <= lo or typ is None or conflict or (typ == 'ratio' and lo <= 0):
                    continue
                rec['n_typed'] += 1
                si = max(i for i, b in enumerate(bounds) if b <= m.start())
                s0, s1 = bounds[si], bounds[min(si + 1, len(bounds) - 1)]
                if s1 - s0 > 600 or s1 - s0 < 20:
                    s0, s1 = max(0, m.start() - 220), min(len(text), m.end() + 220)
                w = low[s0:s1]
                if not (any(p in w for p in ep) and any(p in w for p in op)):
                    continue
                rec['n_attributed'] += 1
                null = 1.0 if typ == 'ratio' else 0.0
                labels.append('NULL' if lo <= null <= hi else ('PROTECTIVE' if hi < null else 'HARMFUL'))
            if not labels:
                reasons['ANCHOR_AMBIGUOUS_RELATION' if rec['n_typed'] else 'NOT_TYPED'] += 1
                continue
            if len(set(labels)) > 1:
                reasons['MULTIPLE_ATTRIBUTED_INTERVALS_DISAGREE'] += 1
                continue
            rec['anchor'] = labels[0]
            rec['unanimous_n'] = len(labels)
            recs.append(rec)
    return recs, reasons, lex_hit, n_rec


def metrics(y, p):
    agree = float(np.mean([a == b for a, b in zip(y, p)]))
    rec = {}
    for c in CLS:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        n = sum(1 for a in y if a == c)
        rec[c] = tp / n if n else float('nan')
    return agree, float(np.nanmean([rec[c] for c in CLS])), rec


def boot(keys, y, p, n=2000, seed=3):
    g = np.array([k[0] for k in keys]); cl = np.unique(g)
    by = {c: np.where(g == c)[0] for c in cl}
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        pick = rng.choice(cl, size=len(cl), replace=True)
        ix = np.concatenate([by[c] for c in pick])
        out.append(float(np.mean([y[i] == p[i] for i in ix])))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    SEC = second_labels()
    res = {}
    for which in (1, 2):
        bench = list(csv.DictReader(open(os.path.join(
            NT, 'data', 'protocol_snapshot', 'external', f'external{which}_benchmark.csv'))))
        claims = sorted({r['claim_id'] for r in bench})
        recs, reasons, lex, n_rec = build(which, claims)
        print(f'\n{"="*84}\nEXTERNAL-{which}  numeric anchor\n{"="*84}')
        print(f'  attribution lexicon coverage: {dict(lex)}')
        print(f'  records scanned {n_rec:,}   high-confidence anchors {len(recs):,} '
              f'({100*len(recs)/max(n_rec,1):.2f}%)   [source: 1,841 of 112,453 = 1.64%]')
        print(f'  exclusions: {dict(reasons.most_common())}')
        keys = [(r['claim_id'], r['pmid']) for r in recs if (r['claim_id'], r['pmid']) in SEC]
        if len(keys) < 30:
            print(f'  only {len(keys)} anchored records also scored by the second pipeline '
                  f'-> NOT reported')
            res[f'external{which}'] = dict(n_scanned=n_rec, n_anchor=len(recs),
                                           lexicon=dict(lex), reasons=dict(reasons),
                                           n_comparable=len(keys), reported=False)
            continue
        A = {(r['claim_id'], r['pmid']): r['anchor'] for r in recs}
        P = {(r['claim_id'], r['pmid']): r['extractor'] for r in recs}
        y = [A[k] for k in keys]
        print(f'\n  {len(keys):,} anchored records scored by both pipelines, '
              f'{len({k[0] for k in keys})} claims')
        print(f'  anchor label mix: {dict(collections.Counter(y).most_common())}')
        print(f'\n  {"pipeline":12s} {"agree":>7s} {"[95% CI]":>18s} {"bal.acc":>8s}  recall P / H / N')
        cell = {}
        for nm, lab in (('primary', P), ('second', SEC)):
            p = [lab[k] for k in keys]
            ag, ba, rc = metrics(y, p)
            lo, hi = boot(keys, y, p)
            cell[nm] = dict(agreement=ag, ci=[lo, hi], balanced_accuracy=ba, recall=rc,
                            unclear=float(np.mean([x == 'UNCLEAR' for x in p])))
            print(f'  {nm:12s} {ag:7.4f} [{lo:.4f}, {hi:.4f}] {ba:8.4f}  '
                  + ' / '.join(f'{rc[c]:.3f}' for c in CLS)
                  + f"   UNCLEAR {cell[nm]['unclear']:.3f}")
        pa = [P[k] for k in keys]; sa = [SEC[k] for k in keys]
        d = float(np.mean([y[i] == sa[i] for i in range(len(y))])
                  - np.mean([y[i] == pa[i] for i in range(len(y))]))
        g = np.array([k[0] for k in keys]); cl = np.unique(g)
        by = {c: np.where(g == c)[0] for c in cl}
        rng = np.random.default_rng(3); dr = []
        for _ in range(2000):
            pick = rng.choice(cl, size=len(cl), replace=True)
            ix = np.concatenate([by[c] for c in pick])
            dr.append(np.mean([y[i] == sa[i] for i in ix]) - np.mean([y[i] == pa[i] for i in ix]))
        lo, hi = float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))
        print(f'\n  second minus primary: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]'
              + ('  <-- interval excludes 0' if not (lo <= 0 <= hi) else ''))
        res[f'external{which}'] = dict(n_scanned=n_rec, n_anchor=len(recs), lexicon=dict(lex),
                                       reasons=dict(reasons), n_comparable=len(keys),
                                       claims=len({k[0] for k in keys}), reported=True,
                                       anchor_mix=dict(collections.Counter(y)), cells=cell,
                                       second_minus_primary=dict(delta=d, ci=[lo, hi]))
    json.dump(res, open(os.path.join(HERE, 'results', 'external_anchor.json'), 'w'),
              indent=1, default=float)
    print('\nwritten results/external_anchor.json')


if __name__ == '__main__':
    main()
