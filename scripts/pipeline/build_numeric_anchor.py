"""Issue 1: an LLM-independent numeric anchor with explicit effect-measure identification and
claim-relation attribution.

For every record in the corpus, every 95% confidence interval is located. For each interval the
effect-measure phrase is read from the text immediately before it (or, failing that, immediately
after it); ratio measures (OR, RR, HR, rate ratio, prevalence ratio, ...) have null 1 and are placed
on the log scale, additive measures (mean difference, risk difference, beta, ...) have null 0 and are
left on the linear scale. The scale is never inferred from the sign of the limits. An interval is
attributed to the claim only when the exposure term and the outcome term both occur in the same
sentence as the interval (or within a 220-character window when sentences cannot be delimited),
using a fixed lexicon of term variants that is printed in the report. A record enters the
high-confidence set when at least one interval is both typed and attributed and every such interval
yields the same direction. Everything else is excluded with a recorded reason. No language model
and no human label is used anywhere in the reference. Namespace: anchor_v2.
"""
# NOTE (release): this stage REBUILDS data/derived/numeric_anchor_v2_*.csv from the abstracts.
# It is a construction step, not a reproduction step. Without the abstracts it writes empty
# tables and destroys the shipped anchor, so it is deliberately NOT part of scripts/run_all.sh.
# Run scripts/pipeline/fetch_abstracts.py first if you really mean to rebuild the anchor.


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
import os, sys, re, json, math, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import astra_common as ac
nm = ac.nm
CI = re.compile(r'95\s*%\s*(?:CI|confidence\s+intervals?|C\.I\.)[^0-9\-−]{0,14}(-?\d+(?:\.\d+)?)\s*(?:to|[-–−,;])\s*(-?\d+(?:\.\d+)?)', re.I)
RATIO_LONG = r'odds\s+ratios?|relative\s+risks?|risk\s+ratios?|hazard\s+ratios?|(?:incidence\s+)?rate\s+ratios?|prevalence\s+ratios?|standardi[sz]ed\s+(?:incidence|mortality)\s+ratios?|relative\s+hazard'
RATIO_SHORT = r'a?OR|a?RR|a?HR|IRR|SIR|SMR|PR'
ADD_LONG = r'mean\s+differences?|standardi[sz]ed\s+mean\s+differences?|weighted\s+mean\s+differences?|risk\s+differences?|absolute\s+(?:risk\s+)?differences?|differences?\s+in\s+means?|mean\s+changes?|regression\s+coefficients?|β\s*-?\s*coefficients?|beta\s+coefficients?'
ADD_SHORT = r'MD|WMD|SMD|RD|β|beta|coefficient'
MEASURE = re.compile(r'\b(?P<rl>' + RATIO_LONG + r')\b|\b(?P<rs>' + RATIO_SHORT + r')\b(?=\s*[\(\[=:,]?\s*(?:of\s+|was\s+|were\s+|=\s*)?-?\d)|\b(?P<al>' + ADD_LONG + r')\b|(?<![A-Za-z])(?P<as>' + ADD_SHORT + r')(?![A-Za-z])(?=\s*[\(\[=:,]?\s*(?:of\s+|was\s+|were\s+|=\s*)?-?\d)', re.I)
SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z(\[])')
STOP = {'dietary', 'products', 'acid', 'acids', 'disease', 'diseases', 'mellitus', 'type', 'neoplasms', 'drinks', 'drinking', 'events', 'function', 'fatty', 'diet', 'based', 'plant', 'free', 'food', 'foods'}
LEX = {  # deterministic term variants; substring, case-insensitive
    'exposure': {'dietary sugars': ['sugar'], 'alcohol drinking': ['alcohol', 'drink', 'ethanol'], 'alcoholic drinks': ['alcohol', 'drink'], 'coffee': ['coffee', 'caffein'],
                 'tea': ['tea', 'catechin'], 'eggs': ['egg'], 'fish products': ['fish', 'omega', 'n-3'], 'fatty acids omega-3': ['omega', 'n-3', 'fish oil', 'epa', 'dha', 'pufa'],
                 'folic acid': ['folic', 'folate'], 'ascorbic acid': ['ascorbic', 'vitamin c'], 'vitamin D': ['vitamin d', 'cholecalciferol', '25-hydroxy', '25(oh)'],
                 'vitamin E': ['vitamin e', 'tocopherol'], 'vitamin A': ['vitamin a', 'retinol', 'carotene'], 'vitamin b12': ['b12', 'cobalamin'], 'beta carotene': ['carotene'],
                 'niacin': ['niacin', 'nicotinic'], 'thiamin deficiency': ['thiamin', 'vitamin b1'], 'vitamin k deficiency': ['vitamin k'], 'vitamin k newborn bleeding': ['vitamin k'],
                 'iron dietary': ['iron'], 'oral iron hemoglobin': ['iron'], 'calcium dietary': ['calcium'], 'sodium dietary': ['sodium', 'salt'], 'iodine deficiency': ['iodine'],
                 'iodized salt iodine deficiency': ['iodi', 'salt'], 'magnesium': ['magnesium'], 'selenium': ['selenium'], 'zinc': ['zinc'], 'multivitamin': ['multivitamin', 'vitamin'],
                 'dairy products': ['dairy', 'milk', 'cheese', 'yogurt'], 'milk': ['milk', 'dairy'], 'cheese': ['cheese', 'dairy'], 'yogurt': ['yogurt', 'yoghurt', 'dairy'], 'butter': ['butter'],
                 'meat products': ['meat'], 'red meat': ['meat'], 'fabaceae': ['legume', 'bean', 'lentil', 'pulse', 'soy'], 'soybeans': ['soy', 'isoflavone'], 'nuts': ['nut', 'almond', 'walnut'],
                 'olive oil': ['olive'], 'coconut oil': ['coconut'], 'coconut oil ldl': ['coconut'], 'cocoa': ['cocoa', 'chocolate'], 'dark chocolate': ['chocolate', 'cocoa'],
                 'dietary fiber': ['fiber', 'fibre'], 'soluble fiber ldl': ['fiber', 'fibre'], 'whole grains': ['whole grain', 'whole-grain', 'grain'], 'fruit veg': ['fruit', 'vegetable'],
                 'sugar-sweetened beverages': ['sugar-sweetened', 'sugar sweetened', 'soft drink', 'soda', 'ssb', 'beverage'], 'food processed': ['processed'], 'food additives': ['additive', 'emulsifier'],
                 'trans fatty acids': ['trans fat', 'trans-fat', 'trans fatty'], 'dietary cholesterol': ['cholesterol'], 'replacing saturated fat polyunsaturated fat': ['saturated', 'polyunsaturated'],
                 'plant sterols ldl': ['sterol', 'stanol'], 'probiotics': ['probiotic', 'lactobacill', 'bifidobacter'], 'prebiotics': ['prebiotic', 'inulin', 'oligosaccharide'],
                 'fermented foods': ['fermented'], 'fermented foods microbiome': ['fermented'], 'intermittent fasting': ['fasting', 'time-restricted', 'time restricted', 'alternate-day'],
                 'caloric restriction': ['caloric restriction', 'calorie restriction', 'energy restriction', 'energy deficit'], 'diet ketogenic': ['ketogenic', 'low-carbohydrate', 'low carbohydrate'],
                 'diet fat-restricted': ['low-fat', 'low fat', 'fat-restricted'], 'dash diet': ['dash'], 'gluten free diet celiac symptom': ['gluten'], 'plant based diet': ['plant-based', 'plant based', 'vegetarian', 'vegan'],
                 'plant based diet bone': ['plant-based', 'plant based', 'vegetarian', 'vegan'], 'dietary pattern': ['dietary pattern', 'diet'], 'breakfast': ['breakfast'], 'late night eating': ['night', 'evening'],
                 'obesity': ['obes', 'body mass', 'bmi', 'overweight', 'adiposity'], 'obesity esophageal': ['obes', 'body mass', 'bmi'], 'obesity postmenopausal': ['obes', 'body mass', 'bmi'],
                 'physical activity': ['physical activity', 'exercise', 'activity'], 'physical activity colon': ['physical activity', 'exercise'], 'physical activity postmenopausal': ['physical activity', 'exercise'],
                 'hormone replacement': ['hormone', 'estrogen', 'oestrogen', 'hrt', 'menopausal hormone'], 'breast feeding': ['breastfe', 'breast-fe', 'breast fe', 'lactation'],
                 'aflatoxins': ['aflatoxin'], 'peanut early introduction allergy': ['peanut'], 'lactose intolerance lactose': ['lactose'], 'protein energy malnutrition child': ['malnutrition', 'protein'],
                 'vitamin b12 vegan deficiency': ['b12', 'cobalamin', 'vegan'], 'niacin deficiency': ['niacin', 'pellagra'], 'iodine deficiency fetal': ['iodine']},
    'outcome': {'cancer': ['cancer', 'carcinoma', 'neoplas', 'tumor', 'tumour', 'malignan'], 'breast neoplasms': ['breast'], 'colorectal neoplasms': ['colorectal', 'colon', 'rectal'], 'colorectal': ['colorectal', 'colon', 'rectal'],
                'liver neoplasms': ['liver', 'hepatocellular', 'hepat'], 'esophageal neoplasms': ['esophag', 'oesophag'], 'laryngeal neoplasms': ['laryn'], 'mouth neoplasms': ['oral', 'mouth'], 'pharyngeal neoplasms': ['pharyn'],
                'lung neoplasms': ['lung'], 'prostatic neoplasms': ['prostat'], 'endometrial neoplasms': ['endometri'], 'kidney neoplasms': ['kidney', 'renal'], 'pancreatic neoplasms': ['pancrea'], 'adenocarcinoma': ['adenocarcinoma', 'esophag'],
                'cardiovascular disease': ['cardiovascular', 'coronary', 'cvd', 'heart', 'stroke', 'myocardial', 'chd'], 'cardiovascular events': ['cardiovascular', 'coronary', 'cvd', 'heart', 'stroke', 'myocardial', 'chd'],
                'diabetes mellitus type 2': ['diabet', 't2d'], 'diabetes': ['diabet'], 'mortality': ['mortality', 'death', 'died', 'survival'], 'all-cause mortality': ['mortality', 'death', 'died'],
                'obesity': ['obes', 'body mass', 'bmi', 'weight', 'adiposity'], 'weight gain': ['weight'], 'weight loss': ['weight'], 'blood pressure': ['blood pressure', 'hypertens', 'systolic', 'diastolic', 'bp'],
                'hypertension': ['hypertens', 'blood pressure'], 'cholesterol': ['cholesterol', 'ldl', 'lipid'], 'hypercholesterolemia': ['cholesterol', 'lipid'], 'triglycerides': ['triglycerid'],
                'cognitive function': ['cognit', 'dementia', 'memory', 'mmse'], 'dementia': ['dementia', 'cognit', 'alzheimer'], 'depression': ['depress'], 'sclerosis': ['sclerosis'],
                'fracture': ['fractur', 'bone'], 'osteoporosis': ['osteopor', 'bone', 'fractur'], 'migraine': ['migraine', 'headache'], 'caries': ['caries', 'dental'], 'ibd': ['inflammatory bowel', 'crohn', 'colitis', 'ibd'],
                'ibs': ['irritable bowel', 'ibs'], 'cold': ['cold', 'respiratory'], 'infection': ['infect'], 'preterm': ['preterm', 'necrotizing', 'enterocolitis'], 'enterocolitis necrotizing': ['enterocolitis', 'nec'],
                'clostridium infections': ['clostridi', 'diarrh'], 'dermatitis': ['dermatitis', 'eczema', 'atopic'], 'inflammation': ['inflammat', 'crp', 'interleukin', 'cytokine'], 'microbiome': ['microbio'],
                'diversity': ['diversity', 'microbio'], 'arrhythmia': ['arrhythm', 'fibrillation'], 'atrial fibrillation': ['fibrillation'], 'goiter': ['goit'], 'beriberi': ['beriberi'], 'pellagra': ['pellagra'],
                'scurvy': ['scurvy'], 'rickets': ['rickets'], 'night blindness': ['night blindness', 'xerophthalm'], 'bleeding': ['bleed', 'hemorrhag', 'haemorrhag'], 'neural tube defects': ['neural tube', 'spina bifida'],
                'constipation': ['constipat'], 'menopause': ['menopaus', 'hot flash', 'vasomotor'], 'fertility': ['fertil', 'sperm', 'semen'], 'eye': ['dry eye', 'ocular'], 'symptoms': ['symptom'],
                'metabolism': ['metabol'], 'control': ['glyc', 'hba1c', 'glucose', 'symptom'], 'prevention': ['prevent', 'incidence', 'risk'], 'remission': ['remission', 'hba1c'], 'growth': ['growth', 'height', 'stunting', 'weight'],
                'response': ['hemoglobin', 'haemoglobin', 'ferritin'], 'absorption': ['absorption', 'iron'], 'bioavailability': ['bioavailab', 'absorption', 'iron'], 'duration': ['duration', 'days'],
                'intake': ['preeclampsia', 'pre-eclampsia', 'hypertens'], 'risk': ['deficien', 'b12', 'cobalamin'], 'progression': ['alzheimer', 'cognit', 'progression'], 'neurodevelopment': ['neurodevelop', 'iq', 'cognit']}}
SYN_TYPES = {'Meta-Analysis', 'Systematic Review'}


def patterns(field, term):
    t = term.lower().strip()
    if t in LEX[field]: return LEX[field][t]
    toks = [w for w in re.findall(r'[a-z0-9\-]+', t) if len(w) >= 4 and w not in STOP]
    return [w[:5] for w in toks] or [t]


def design_of(pub_types):
    pts = pub_types if isinstance(pub_types, list) else (eval(pub_types) if pub_types else [])
    s = set(pts)
    return 'synthesis' if SYN_TYPES & s else ('trial' if any('Trial' in x for x in s) else 'other')


def measure_before(text, start, end):
    """Nearest effect-measure token in the 120 characters before the interval, else the 40 after."""
    win = text[max(0, start - 120):start]; found = [(m.end(), m.lastgroup) for m in MEASURE.finditer(win)]
    if not found:
        aft = text[end:end + 40]; found = [(-m.start(), m.lastgroup) for m in MEASURE.finditer(aft)]
        if not found: return None, None, False
    found.sort(); pos, grp = found[-1]
    typ = 'ratio' if grp in ('rl', 'rs') else 'additive'
    conflict = any(abs(pos - p2) <= 25 and (('r' in g2) != ('r' in grp)) for p2, g2 in found[:-1])
    return typ, grp, conflict


def main():
    Q = json.load(open(nm.QUERIES)); cands, records, excl = [], [], []
    reasons = collections.Counter(); n_rec = 0
    for cid in Q:
        Dj = {d['pmid']: d for d in json.load(open(f'{nm.DIRECTIONS}/{cid}.json'))['directions']}
        ep, op = patterns('exposure', Q[cid]['exposure']), patterns('outcome', Q[cid]['outcome'])
        for r in json.load(open(f'{nm.RECORDS}/{cid}.json'))['records']:
            if r['pmid'] not in Dj: continue
            n_rec += 1; text = (r.get('title') or '') + '. ' + (r.get('abstract') or ''); low = text.lower()
            sents = list(SENT.finditer(text)); bounds = [0] + [m.end() for m in sents] + [len(text)]
            cis = list(CI.finditer(text)); rec = {'claim_id': cid, 'pmid': r['pmid'], 'year': r.get('year'), 'design': design_of(r.get('pub_types')), 'extractor': Dj[r['pmid']]['direction'],
                                                 'n_ci': len(cis), 'n_typed': 0, 'n_attributed': 0}
            if not cis:
                excl.append({**rec, 'reason': 'NO_CI'}); reasons['NO_CI'] += 1; continue
            labels = []
            for m in cis:
                lo, hi = float(m.group(1)), float(m.group(2)); typ, grp, conflict = measure_before(text, m.start(), m.end())
                c = {'claim_id': cid, 'pmid': r['pmid'], 'ci_lo': lo, 'ci_hi': hi, 'measure_token': grp or '', 'effect_type': typ or 'unidentified', 'type_conflict': int(conflict),
                     'context': text[max(0, m.start() - 90):m.end() + 20].replace('\n', ' ')}
                if hi <= lo: c['status'] = 'INVALID_INTERVAL'
                elif typ is None: c['status'] = 'EFFECT_TYPE_UNIDENTIFIED'
                elif conflict: c['status'] = 'EFFECT_TYPE_CONFLICT'
                elif typ == 'ratio' and lo <= 0: c['status'] = 'INVALID_INTERVAL_FOR_RATIO'
                else:
                    rec['n_typed'] += 1
                    si = max(i for i, b in enumerate(bounds) if b <= m.start()); s0, s1 = bounds[si], bounds[min(si + 1, len(bounds) - 1)]
                    if s1 - s0 > 600 or s1 - s0 < 20: s0, s1 = max(0, m.start() - 220), min(len(text), m.end() + 220)
                    w = low[s0:s1]
                    c['exposure_in_window'] = int(any(p in w for p in ep)); c['outcome_in_window'] = int(any(p in w for p in op))
                    if not (c['exposure_in_window'] and c['outcome_in_window']): c['status'] = 'ANCHOR_AMBIGUOUS_RELATION'
                    else:
                        rec['n_attributed'] += 1; c['status'] = 'ATTRIBUTED'
                        null = 1.0 if typ == 'ratio' else 0.0
                        c['anchor'] = 'NULL' if lo <= null <= hi else ('PROTECTIVE' if hi < null else 'HARMFUL')
                        c['y_mid'] = round((math.log(lo) + math.log(hi)) / 2 if typ == 'ratio' else (lo + hi) / 2, 4)
                        labels.append(c['anchor'])
                cands.append(c)
            if not labels:
                reason = 'ANCHOR_AMBIGUOUS_RELATION' if rec['n_typed'] else ('EFFECT_TYPE_CONFLICT' if any(c.get('type_conflict') for c in cands[-len(cis):]) else 'EFFECT_TYPE_UNIDENTIFIED')
                excl.append({**rec, 'reason': reason}); reasons[reason] += 1; continue
            if len(set(labels)) > 1:
                excl.append({**rec, 'reason': 'MULTIPLE_ATTRIBUTED_INTERVALS_DISAGREE'}); reasons['MULTIPLE_ATTRIBUTED_INTERVALS_DISAGREE'] += 1; continue
            rec['anchor'] = labels[0]; rec['unanimous_n'] = len(labels); records.append(rec)
    ns_ = ac.ns('data', 'anchor_v2')
    ac.save_csv(cands, os.path.join(ac.DATA, 'numeric_anchor_v2_all_candidates.csv'))
    ac.save_csv(records, os.path.join(ac.DATA, 'numeric_anchor_v2_high_confidence.csv'))
    ac.save_csv(excl, os.path.join(ac.DATA, 'numeric_anchor_v2_exclusions.csv'))

    # ---- metrics -----------------------------------------------------------------------------
    CLS = ['PROTECTIVE', 'HARMFUL', 'NULL']
    def conf(a, e):
        return np.array([[int(((a == c) & (e == d)).sum()) for d in CLS + ['UNCLEAR']] for c in CLS])
    def scores(a, e):
        m = conf(a, e); tp = np.array([m[i, i] for i in range(3)], float)
        rec_ = np.divide(tp, m.sum(1), out=np.zeros(3), where=m.sum(1) > 0); pred = np.array([(e == c).sum() for c in CLS], float)
        prec = np.divide(tp, pred, out=np.zeros(3), where=pred > 0); f1 = np.divide(2 * prec * rec_, prec + rec_, out=np.zeros(3), where=(prec + rec_) > 0)
        return {'agreement': float((a == e).mean()), 'balanced_accuracy': float(rec_.mean()), 'macro_f1': float(f1.mean()),
                **{f'recall_{c}': float(rec_[i]) for i, c in enumerate(CLS)}, **{f'precision_{c}': float(prec[i]) for i, c in enumerate(CLS)}, **{f'f1_{c}': float(f1[i]) for i, c in enumerate(CLS)}}
    def block(name, R, note=''):
        a = np.array([r['anchor'] for r in R]); e = np.array([r['extractor'] for r in R]); g = np.array([r['claim_id'] for r in R])
        s = scores(a, e); ci = {}
        for k in ('agreement', 'balanced_accuracy', 'macro_f1', 'f1_PROTECTIVE', 'f1_HARMFUL', 'f1_NULL'):
            lo, hi, _ = nm.claim_boot(g, lambda i, k=k: scores(a[i], e[i])[k]); ci[k] = nm.fmt_ci(lo, hi)
        m = conf(a, e)
        d = {'evaluation': name, 'records': len(R), 'claims': int(len(set(g))), 'note': note, **{k: round(v, 4) for k, v in s.items()}, **{f'{k}_ci': v for k, v in ci.items()},
             'anchor_dist': dict(collections.Counter(a)), 'extractor_unclear': int((e == 'UNCLEAR').sum()), 'confusion': m.tolist()}
        print(f"  {name:34s} n={len(R):5d} agree {s['agreement']:.3f} {ci['agreement']} bal {s['balanced_accuracy']:.3f} mF1 {s['macro_f1']:.3f}", flush=True)
        return d
    res = {'coverage': {'records_with_extractor_label': n_rec, 'records_with_any_ci': n_rec - reasons['NO_CI'], 'records_with_typed_interval': sum(1 for r in records + excl if r.get('n_typed', 0) > 0),
                        'records_with_attributed_interval': sum(1 for r in records + excl if r.get('n_attributed', 0) > 0), 'high_confidence_records': len(records),
                        'claims_covered': len({r['claim_id'] for r in records}), 'share_of_corpus': round(len(records) / n_rec, 4), 'exclusion_reasons': dict(reasons),
                        'candidate_intervals': len(cands), 'candidate_status': dict(collections.Counter(c['status'] for c in cands)),
                        'measure_tokens_in_attributed': dict(collections.Counter(c['measure_token'].lower() for c in cands if c['status'] == 'ATTRIBUTED'))},
           'evaluations': {}}
    res['evaluations']['high_confidence'] = block('high-confidence anchor v2', records, 'typed measure + attributed relation; unanimous when several intervals')
    res['evaluations']['exactly_one_attributed'] = block('exactly one attributed interval', [r for r in records if r['unanimous_n'] == 1], 'stricter subset')
    res['evaluations']['excluding_extractor_unclear'] = block('excluding extractor-UNCLEAR', [r for r in records if r['extractor'] != 'UNCLEAR'])
    for typ in ('ratio', 'additive'):
        pm = {c['pmid'] for c in cands if c['status'] == 'ATTRIBUTED' and c['effect_type'] == typ}
        res['evaluations'][f'measure_{typ}'] = block(f'{typ}-scale intervals only', [r for r in records if r['pmid'] in pm])
    by_design = []
    for d in ('synthesis', 'other', 'trial'):
        R = [r for r in records if r['design'] == d]
        if len(R) >= 30: e = block(f'design: {d}', R); res['evaluations'][f'design_{d}'] = e; by_design.append(e)
    # legacy comparison
    leg = list(__import__('csv').DictReader(open(os.path.join(ac.SNAP, 'analyses_v2', 'legacy_anchor_records.csv'))))
    res['evaluations']['legacy_anchor_sign_inferred'] = block('legacy anchor (scale inferred from limits)', leg, 'the earlier reference, for transparency only')
    v2 = {(r['claim_id'], r['pmid']): r['anchor'] for r in records}; ov = [(l, v2[(l['claim_id'], l['pmid'])]) for l in leg if (l['claim_id'], l['pmid']) in v2]
    res['legacy_vs_v2'] = {'legacy_records': len(leg), 'v2_records': len(records), 'overlap': len(ov), 'legacy_not_in_v2': len(leg) - len(ov), 'v2_not_in_legacy': len(records) - len(ov),
                           'anchor_label_agreement_on_overlap': round(float(np.mean([l['anchor'] == a for l, a in ov])), 4) if ov else None,
                           'legacy_excluded_reasons': dict(collections.Counter(next((x['reason'] for x in excl if x['pmid'] == l['pmid'] and x['claim_id'] == l['claim_id']), 'not excluded') for l in leg if (l['claim_id'], l['pmid']) not in v2))}
    ac.save_json({'results': res, 'lexicon': LEX, 'regex': {'ci': CI.pattern, 'measure': MEASURE.pattern}, 'protocol': 'claim-clustered percentile bootstrap, 2,000 resamples, seed 3'}, os.path.join(ac.RESULTS, 'numeric_anchor_v2_metrics.json'))
    main_rows = [{k: v for k, v in res['evaluations'][n].items() if k not in ('confusion', 'anchor_dist')} for n in ('high_confidence', 'exactly_one_attributed', 'excluding_extractor_unclear', 'measure_ratio', 'measure_additive', 'legacy_anchor_sign_inferred')]
    ac.save_csv(main_rows, os.path.join(ac.TABLES, 'numeric_anchor_v2_main.csv'))
    ac.save_csv([{k: v for k, v in e.items() if k not in ('confusion', 'anchor_dist')} for e in by_design], os.path.join(ac.TABLES, 'numeric_anchor_v2_by_design.csv'))
    confr = []
    for n in ['high_confidence'] + [f'design_{d}' for d in ('synthesis', 'other', 'trial') if f'design_{d}' in res['evaluations']]:
        m = np.array(res['evaluations'][n]['confusion'])
        for i, c in enumerate(CLS):
            confr.append({'evaluation': n, 'anchor': c, 'n': int(m[i].sum()), **{f'pred_{d}': int(m[i, j]) for j, d in enumerate(CLS + ['UNCLEAR'])},
                          **{f'share_{d}': round(float(m[i, j] / max(m[i].sum(), 1)), 3) for j, d in enumerate(CLS + ['UNCLEAR'])}})
    ac.save_csv(confr, os.path.join(ac.TABLES, 'numeric_anchor_v2_confusion.csv'))
    print(json.dumps(res['coverage'], indent=1)); print('DONE', flush=True)


if __name__ == '__main__':
    main()
