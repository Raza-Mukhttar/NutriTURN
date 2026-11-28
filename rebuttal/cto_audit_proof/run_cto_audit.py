"""Annotator-substitution audit of CTO (Gao et al., 2024), one self-contained script.

CTO labels clinical-trial outcomes with fifteen weak labelling functions aggregated by a random
forest. One function, `gpt`, is LLM-derived. This script takes

  inputs/cto_released/  CTO's own released files (three phase prediction files, human-label subset)
  inputs/annotations/  the `gpt` function re-derived with Llama-3.1-8B-Instruct and
                       Qwen2.5-7B-Instruct on the abstracts CTO released (74,200 per annotator)

and recomputes every number reported for the audit. Everything is evaluated on distinct trials
(one row per trial) and every interval resamples trials.

    python run_cto_audit.py                       # recompute, write output/, compare to expected
    python run_cto_audit.py --jobs 8 --tol 1e-6

Exit status is 0 only if every recomputed number matches expected_numbers.json within --tol.
No GPU is needed; the run takes a few minutes on CPU.
"""
import argparse, collections, csv, glob, hashlib, json, math, os, sys
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
csv.field_size_limit(10 ** 9)
SEED, BOOT, FOLDS, TREES = 3, 2000, 5, 300
LFS = ['status', 'gpt', 'linkage', 'stock_price', 'results_reported', 'new_headlines', 'pvalues',
       'update_more_recent', 'sites', 'serious_ae', 'patient_drop', 'num_patients', 'death_ae',
       'amendments', 'all_ae']
GI = LFS.index('gpt')
CODE = {'SUCCESS': 1, 'FAILURE': 0, 'UNCLEAR': -1}
MODELS = ('llama8b', 'qwen7b')
JOBS = 8
N = {}          # flat name -> recomputed number, compared with expected_numbers.json
LOG = []


def say(s=''):
    print(s, flush=True); LOG.append(s)


def put(key, value):
    N[key] = value
    return value


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if not npos or not nneg:
        return float('nan')
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def boot_diff(y, a, b, seed=SEED, reps=BOOT):
    """Percentile interval for AUROC(a) - AUROC(b), resampling trials."""
    rng = np.random.default_rng(seed); n = len(y); out = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, n, n); yy = y[idx]
        if (yy == 1).sum() == 0 or (yy == 0).sum() == 0:
            out[i] = np.nan; continue
        out[i] = auroc(yy, a[idx]) - auroc(yy, b[idx])
    return tuple(float(v) for v in np.nanpercentile(out, [2.5, 97.5]))


def kappa(a, b):
    """Cohen's kappa on the trials where both functions commit (neither is -1)."""
    pair = [(x, y) for x, y in zip(a, b) if x != -1 and y != -1]
    if not pair:
        return float('nan'), 0
    n = len(pair); po = sum(x == y for x, y in pair) / n
    ca = collections.Counter(x for x, _ in pair); cb = collections.Counter(y for _, y in pair)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (float('nan') if pe == 1 else (po - pe) / (1 - pe)), n


def phase_rows(data):
    for p in (1, 2, 3):
        with open(f'{data}/cto_released/phase{p}_CTO_rf.csv', newline='') as f:
            for r in csv.DictReader(f):
                r['_phase'] = p
                yield r


def dedup(data, rule='lowest'):
    """One row per trial: the lowest (default) or highest phase file in which it appears."""
    out = {}
    rows = list(phase_rows(data))
    if rule == 'highest':
        rows = sorted(rows, key=lambda r: -r['_phase'])
    for r in rows:
        out.setdefault(r['nct_id'], r)
    return out


def load_annotations(data):
    out, stats = {}, {}
    for m in MODELS:
        d = collections.defaultdict(dict); n = 0; keys = set()
        for f in sorted(glob.glob(f'{data}/annotations/{m}/*.jsonl')):
            for line in open(f):
                r = json.loads(line); n += 1
                keys.add((r['nctid'], r['slot'])); d[r['nctid']][r['slot']] = r['label']
        out[m] = d; stats[m] = (n, len(keys), len(d))
    return out, stats


def combine(slots):
    """Trial label from up to two abstract slots: the first slot that commits, else abstain."""
    for s in sorted(slots):
        if slots[s] in ('SUCCESS', 'FAILURE'):
            return CODE[slots[s]]
    return -1


def forest(seed=SEED):
    return RandomForestClassifier(n_estimators=TREES, random_state=seed, n_jobs=JOBS,
                                  min_samples_leaf=2)


def oof(X, y, seed=SEED):
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    return cross_val_predict(forest(seed), X, y, cv=cv, method='predict_proba', n_jobs=1)[:, 1]


def contrast(tag, label, y, p, p0):
    a = auroc(y, p); d = a - auroc(y, p0); lo, hi = boot_diff(y, p, p0)
    est = bool(lo > 0 or hi < 0)
    put(f'{tag}.auroc', a); put(f'{tag}.delta', d); put(f'{tag}.ci_lo', lo); put(f'{tag}.ci_hi', hi)
    put(f'{tag}.established', est)
    say(f'   {label:30s} {a:7.4f}   {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  '
        f'{"established" if est else "not established"}')


def main():
    global JOBS
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=f'{HERE}/inputs')
    ap.add_argument('--out', default=f'{HERE}/output')
    ap.add_argument('--expected', default=f'{HERE}/expected_numbers.json')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--tol', type=float, default=1e-6)
    a = ap.parse_args(); JOBS = a.jobs; D = a.data
    os.makedirs(a.out, exist_ok=True)
    import sklearn, scipy
    say(f'python {sys.version.split()[0]}  numpy {np.__version__}  scipy {scipy.__version__}  '
        f'scikit-learn {sklearn.__version__}')

    # ------------------------------------------------------------------ 0. inputs
    say('\n0. INPUTS (SHA-256)')
    files = sorted(glob.glob(f'{D}/cto_released/*.csv') + glob.glob(f'{D}/annotations/*/*.jsonl'))
    manifest = {os.path.relpath(f, D): sha256(f) for f in files}
    for k, v in manifest.items():
        say(f'   {v}  {k}')
    ours, astats = load_annotations(D)
    for m in MODELS:
        n, k, t = astats[m]
        say(f'   {m}: {n:,} scored prompts, {k:,} distinct (trial, slot) keys, {t:,} trials')
        put(f'annotation.{m}.prompts', n); put(f'annotation.{m}.distinct_keys', k)
    put('annotation.total_prompts', sum(astats[m][0] for m in MODELS))

    # ------------------------------------------------------------------ 1. deduplication
    rows = [r for r in phase_rows(D) if r['hint_train'] in ('0', '1')]
    copies = collections.defaultdict(list)
    for r in rows:
        copies[r['nct_id']].append(tuple(r[c] for c in LFS))
    rep = {n: v for n, v in copies.items() if len(v) > 1}
    differ = sum(1 for v in rep.values() if len(set(v)) > 1)
    mult = collections.Counter(len(v) for v in copies.values())
    say('\n1. DEDUPLICATION')
    say(f'   {len(rows):,} labelled rows in the three phase files cover {len(copies):,} distinct trials')
    say(f'   repeated trials {len(rep):,}: identical on all 15 functions {len(rep) - differ:,}, '
        f'differing {differ:,}')
    say(f'   copies per trial: {dict(sorted(mult.items()))}')
    put('dedup.rows', len(rows)); put('dedup.distinct_trials', len(copies))
    put('dedup.repeated', len(rep)); put('dedup.identical', len(rep) - differ)
    put('dedup.differing', differ)
    put('dedup.trials_16_copies', mult.get(16, 0)); put('dedup.trials_64_copies', mult.get(64, 0))

    byid = dedup(D, 'lowest')
    ev = [r for r in byid.values() if r['hint_train'] in ('0', '1')]
    y = np.array([int(r['hint_train']) for r in ev])
    X0 = np.array([[float(r[c]) for c in LFS] for r in ev])
    shipped = np.array([float(r['pred_proba']) for r in ev])
    prior = float(y.mean())
    gpt = X0[:, GI].astype(int)
    col = {m: np.array([combine(ours[m].get(r['nct_id'], {})) for r in ev]) for m in MODELS}
    fill = lambda v, p=prior: np.where(v == -1, p, v).astype(float)
    put('weak.n_trials', len(ev)); put('weak.base_rate', prior)

    # ------------------------------------------------------------------ 2. the labelling functions
    say(f'\n2. THE LLM LABELLING FUNCTION, three versions, on {len(ev):,} distinct trials '
        f'(weak target hint_train, base rate {prior:.4f})')
    say(f'   {"function":22s} {"coverage":>9s} {"commits":>8s} {"AUROC":>7s} {"kappa vs CTO gpt":>17s}')
    for name, v in [('cto_gpt', gpt)] + [(m, col[m]) for m in MODELS]:
        k, nb = (float('nan'), 0) if name == 'cto_gpt' else kappa(v.tolist(), gpt.tolist())
        put(f'lf.{name}.coverage', float((v != -1).mean())); put(f'lf.{name}.commits', int((v != -1).sum()))
        put(f'lf.{name}.auroc', auroc(y, fill(v)))
        if name != 'cto_gpt':
            put(f'lf.{name}.kappa_vs_cto_gpt', k)
        say(f'   {name:22s} {N[f"lf.{name}.coverage"]:9.4f} {N[f"lf.{name}.commits"]:8,d} '
            f'{N[f"lf.{name}.auroc"]:7.4f} {k:17.4f}')
    kq, nq = kappa(col['llama8b'].tolist(), col['qwen7b'].tolist())
    put('lf.kappa_llama_qwen', kq); put('lf.kappa_llama_qwen_n', nq)
    say(f'   kappa(Llama, Qwen) = {kq:.4f} on {nq:,} co-committed trials')
    say('   confusion, ours (Llama) x CTO gpt:      gpt=1    gpt=0   gpt=-1')
    for av in (1, 0, -1):
        cells = [int(((col['llama8b'] == av) & (gpt == b)).sum()) for b in (1, 0, -1)]
        for b, c in zip((1, 0, -1), cells):
            put(f'confusion.ours={av}.gpt={b}', c)
        say(f'   {"ours=" + str(av):36s} {cells[0]:8,d} {cells[1]:8,d} {cells[2]:8,d}')
    say('   single-function AUROC against the weak target (all fifteen released functions):')
    single = {c: auroc(y, fill(X0[:, i])) for i, c in enumerate(LFS)}
    for c in sorted(single, key=lambda k: -single[k]):
        say(f'     {c:22s} {single[c]:.4f}')
    best = max(single, key=single.get)
    put('weak.best_single_lf_auroc', single[best])

    # ------------------------------------------------------------------ 3. rebuilding the aggregate
    say('\n3. REBUILDING THE AGGREGATE (out-of-fold random forest, 5 folds, seed 3; '
        f'{BOOT:,} resamples of trials)')
    V = collections.OrderedDict()
    V['R0'] = ('R0 original 15 functions', X0)
    V['R1'] = ('R1 drop gpt', np.delete(X0, GI, axis=1))
    for m in MODELS:
        X = X0.copy(); X[:, GI] = col[m]; V[f'R2_{m}'] = (f'R2 gpt <- {m}', X)
    X = X0.copy(); X[:, GI] = np.where(col['llama8b'] == col['qwen7b'], col['llama8b'], -1)
    V['R3'] = ('R3 gpt <- consensus', X)
    V['R4'] = ('R4 add both, keep gpt', np.column_stack([X0, col['llama8b'], col['qwen7b']]))
    X = X0.copy(); X[:, GI] = (col['llama8b'] != -1).astype(float)
    V['L1'] = ('L1 coverage indicator only', X)
    X = X0.copy(); X[:, GI] = np.where(gpt == -1, -1, col['llama8b'])
    V['L2'] = ('L2 ours under CTO abstention', X)
    rng = np.random.default_rng(SEED); com = np.where(col['llama8b'] != -1)[0]
    sh = col['llama8b'].copy(); perm = sh[com].copy(); rng.shuffle(perm); sh[com] = perm
    X = X0.copy(); X[:, GI] = sh
    V['L3'] = ('L3 ours shuffled among commits', X)
    pr = {k: oof(Xk, y) for k, (_, Xk) in V.items()}
    put('rebuild.shipped.auroc', auroc(y, shipped)); put('rebuild.R0.auroc', auroc(y, pr['R0']))
    say(f'   {"CTO shipped pred_proba":30s} {N["rebuild.shipped.auroc"]:7.4f}')
    say(f'   {V["R0"][0]:30s} {N["rebuild.R0.auroc"]:7.4f}   (reference)')
    for k, (label, _) in V.items():
        if k != 'R0':
            contrast(f'rebuild.{k}', label, y, pr[k], pr['R0'])
    put('weak.aggregate_minus_best_single', N['rebuild.R0.auroc'] - single[best])
    say(f'   aggregate R0 minus best single released function ({best}): '
        f'{N["weak.aggregate_minus_best_single"]:+.4f}')

    say('\n   seed sensitivity of the forest (noise floor):')
    seeds = [3, 17, 101, 2027]
    for k in ('R0', 'R2_llama8b'):
        vals = [auroc(y, oof(V[k][1], y, seed=s)) for s in seeds]
        put(f'seeds.{k}.spread', max(vals) - min(vals))
        say(f'   {V[k][0]:30s} {", ".join(f"{v:.4f}" for v in vals)}   spread {max(vals) - min(vals):.4f}')
    hb = dedup(D, 'highest')
    e2 = [r for r in hb.values() if r['hint_train'] in ('0', '1')]
    y2 = np.array([int(r['hint_train']) for r in e2])
    X2 = np.array([[float(r[c]) for c in LFS] for r in e2])
    X2r = X2.copy(); X2r[:, GI] = [combine(ours['llama8b'].get(r['nct_id'], {})) for r in e2]
    b2, s2 = auroc(y2, oof(X2, y2)), auroc(y2, oof(X2r, y2))
    put('dedup.highest_phase.R0', b2); put('dedup.highest_phase.R2', s2)
    put('dedup.highest_phase.delta', s2 - b2)
    say(f'   deduplication keeping the highest phase instead: R0 {b2:.4f}  R2 {s2:.4f}  '
        f'delta {s2 - b2:+.4f}')

    # ------------------------------------------------------------------ 4. what the gpt column encodes
    with open(f'{D}/cto_released/human_labels_2020_2024.csv', newline='') as f:
        hrows = [r for r in csv.DictReader(f) if r['labels'] not in ('', None)]
    hum = {r['nct_id']: int(float(r['labels'])) for r in hrows}
    status = {r['nct_id']: r['overall_status'] for r in hrows}
    hv = [r for r in byid.values() if r['nct_id'] in hum]
    yh = np.array([hum[r['nct_id']] for r in hv]); ph = float(yh.mean())
    Xh = np.array([[float(r[c]) for c in LFS] for r in hv])
    gh = Xh[:, GI].astype(int)
    say('\n4. WHAT THE RELEASED gpt COLUMN ENCODES: P(target = 1 | gpt)')
    say(f'   {"target":12s} {"n":>7s}  {"gpt=1":>6s} {"gpt=0":>6s} {"gpt=-1":>7s}  AUROC')
    for nm, yy, vv, pp in (('hint_train', y, gpt, prior), ('human', yh, gh, ph)):
        r = {b: float(yy[vv == b].mean()) for b in (1, 0, -1)}
        for b in (1, 0, -1):
            put(f'gpt_rates.{nm}.gpt={b}', r[b])
        put(f'gpt_rates.{nm}.auroc', auroc(yy, fill(vv, pp)))
        say(f'   {nm:12s} {len(yy):7,d}  {r[1]:6.3f} {r[0]:6.3f} {r[-1]:7.3f}  '
            f'{N[f"gpt_rates.{nm}.auroc"]:.4f}')
    shared = [r for r in ev if r['nct_id'] in hum]
    agree = sum(int(r['hint_train']) == hum[r['nct_id']] for r in shared)
    put('human.shared_with_weak', len(shared)); put('human.weak_agreement', agree / len(shared))
    say(f'   the weak target and the human labels share {len(shared)} trials and agree on '
        f'{agree} ({agree / len(shared):.1%})')

    # ------------------------------------------------------------------ 5. the human subset
    say(f'\n5. THE HUMAN-REVIEWED SUBSET ({len(hrows):,} trials)')
    say(f'   {"overall_status":14s} {"n":>6s} {"success":>8s} {"failure":>8s}')
    tab = collections.defaultdict(lambda: [0, 0])
    for r in hrows:
        tab[r['overall_status']][hum[r['nct_id']]] += 1
    for s, (f0, f1) in sorted(tab.items(), key=lambda kv: -sum(kv[1])):
        put(f'human.status.{s}.n', f0 + f1); put(f'human.status.{s}.success', f1)
        say(f'   {s:14s} {f0 + f1:6,d} {f1:8,d} {f0:8,d}')
    right = sum((status[n] == 'COMPLETED') == bool(v) for n, v in hum.items())
    put('human.n', len(hum)); put('human.status_rule_correct', right)
    put('human.status_rule_accuracy', right / len(hum))
    put('human.majority_baseline', max(np.mean(list(hum.values())), 1 - np.mean(list(hum.values()))))
    say(f'   "success iff COMPLETED" is right on {right:,} / {len(hum):,} = {right / len(hum):.1%} '
        f'(majority baseline {N["human.majority_baseline"]:.1%})')

    say(f'   scored on the {len(hv):,} human-labelled trials present in the phase files '
        f'(base rate {ph:.4f}):')
    put('human.n_in_phase_files', len(hv))
    put('human.auroc_status', auroc(yh, fill(Xh[:, LFS.index('status')], ph)))
    put('human.auroc_shipped', auroc(yh, np.array([float(r['pred_proba']) for r in hv])))
    train = [r for r in ev if r['nct_id'] not in hum]
    ytr = np.array([int(r['hint_train']) for r in train])
    Xtr = np.array([[float(r[c]) for c in LFS] for r in train])
    pt = forest().fit(Xtr, ytr).predict_proba(Xh)[:, 1]
    put('human.transfer_train_n', len(train)); put('human.auroc_forest', auroc(yh, pt))
    put('human.forest_minus_status', N['human.auroc_forest'] - N['human.auroc_status'])
    say(f'     status alone                                   {N["human.auroc_status"]:.4f}')
    say(f'     fifteen-function forest, refit on {len(train):,} weak-target trials '
        f'with the human set held out   {N["human.auroc_forest"]:.4f}  '
        f'({N["human.forest_minus_status"]:+.4f} over status)')
    say(f'     CTO shipped pred_proba                         {N["human.auroc_shipped"]:.4f}')

    comp = [r for r in hv if status[r['nct_id']] == 'COMPLETED']
    yc = np.array([hum[r['nct_id']] for r in comp]); pc = float(yc.mean())
    Xc = np.array([[float(r[c]) for c in LFS] for r in comp])
    rk = {c: auroc(yc, fill(Xc[:, i], pc)) for i, c in enumerate(LFS)}
    cov = {}
    for m in MODELS:
        v = np.array([combine(ours[m].get(r['nct_id'], {})) for r in comp])
        rk[f'ours:{m}'] = auroc(yc, fill(v, pc)); cov[m] = float((v != -1).mean())
    rk['CTO shipped pred_proba'] = auroc(yc, np.array([float(r['pred_proba']) for r in comp]))
    put('completed.n', len(comp)); put('completed.failures', int((yc == 0).sum()))
    put('completed.base_rate', pc)
    say(f'\n   COMPLETED trials only, where status is constant: {len(comp):,} trials, '
        f'{int(yc.sum()):,} success / {int((yc == 0).sum()):,} failure')
    order = sorted(rk, key=lambda k: -rk[k])
    for i, k in enumerate(order, 1):
        put(f'completed.auroc.{k}', rk[k])
        say(f'     {i:2d}. {k:24s} {rk[k]:.4f}' + (f'   coverage {cov[k[5:]]:.3f}' if k.startswith('ours:') else ''))
    put('completed.rank.gpt', order.index('gpt') + 1); put('completed.n_ranked', len(order))

    # ------------------------------------------------------------------ write and compare
    json.dump({'inputs_sha256': manifest, 'numbers': N}, open(f'{a.out}/cto_audit_numbers.json', 'w'),
              indent=1, default=float)
    ok = True
    if os.path.exists(a.expected):
        exp = json.load(open(a.expected))['numbers']
        say(f'\nCHECK against {os.path.basename(a.expected)} (tolerance {a.tol:g})')
        bad = []
        for k, e in exp.items():
            g = N.get(k)
            if g is None:
                bad.append((k, e, 'missing'))
            elif isinstance(e, bool) or isinstance(e, int):
                if g != e: bad.append((k, e, g))
            elif not (abs(float(g) - float(e)) <= a.tol or (math.isnan(g) and math.isnan(e))):
                bad.append((k, e, g))
        for k, e, g in bad:
            say(f'   MISMATCH {k}: expected {e}, recomputed {g}')
        ok = not bad
        say(f'   {len(exp) - len(bad)} of {len(exp)} numbers reproduced' + ('' if ok else '  <-- FAILED'))
    open(f'{a.out}/cto_audit_log.txt', 'w').write('\n'.join(LOG) + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
