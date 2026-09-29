"""Issue 5: origin-defined calendar cohort. At each origin Y (2004, 2007, ..., 2022) and horizon H in {3, 5, 10}
every claim with >= 40 records before Y enters the cohort (no future-based removal). Accrual A = 1 if at least ten
signed records arrive in [Y, Y+H); the change outcome R is defined only when A = 1 (hurdle representation).
Only windows completely observed before the freeze (Y+H-1 <= 2025) are used. Training is CLAIM-DISJOINT: for each
test claim every earlier unit of that claim is removed from the training set before refitting (one fit per test
claim per origin). Reports the census (candidates, accrued, censored), the accrual model, the conditional change
model and a combined 'accrual x change' risk, for the margin rule, the two-variable score, Compact-7 and the
deployable Beta-Binomial baseline, with paired intervals. Writes results/issue5/."""

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
import os, sys, json, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pro_common as pc
sys.path.insert(0, _os.path.join(_NT, 'scripts', 'lib'))
import calendar_forward_evaluation as cf
ac, nm = pc.ac, pc.nm
ORIGINS = [2004, 2007, 2010, 2013, 2016, 2019, 2022]; HORIZONS = (3, 5, 10); STEP, MIN_PRE, MIN_WIN = 2, 40, 10


def unit_full(cid, t, H, yrs, codes, tf):
    """Unit with accrual indicator (never excluded for low accrual)."""
    m = yrs < t
    if m.sum() < MIN_PRE: return None
    f = ac.direction_feats(yrs[m], codes[m], t, 'A')
    w = codes[(yrs >= t) & (yrs < t + H)]; ws = w[(w != 99) & (w != 0)]
    eff = [e for yr, es in tf if yr < t for e in es]
    pre = codes[m]; f.update(claim_id=cid, cutoff_year=t, horizon=H, n_pre_t=int(m.sum()), n_rr=len(eff),
             val_mean=float(np.mean([e[0] for e in eff])) if len(eff) >= 3 else 0.0, val_sd=float(np.std([e[0] for e in eff])) if len(eff) >= 3 else 0.0,
             n_plus=int((pre == 1).sum()), n_minus=int((pre == -1).sum()), rate5=int(((yrs >= t - 5) & (yrs < t) & (codes != 99) & (codes != 0)).sum()) / 5.0,
             m_signed=int(len(ws)), accrued=int(len(ws) >= MIN_WIN), y=int(pc.sgn0(float(ws.mean())) != pc.sgn0(f['pooled_direction'])) if len(ws) >= MIN_WIN else -1)
    return f


def main():
    Q = json.load(open(nm.QUERIES)); cids = list(Q); streams = {c: ac.stream(c) for c in cids}; tf = {c: cf.text_feats_cache(c) for c in cids}
    rng = np.random.default_rng(3); print('caches built', flush=True)
    census, rows, res = [], [], {'protocol': __doc__, 'pooled': {}}
    X = lambda U, keys: np.array([[u[k] for k in keys] for u in U], float)
    for H in HORIZONS:
        pooled = collections.defaultdict(list)
        for Y in ORIGINS:
            complete = (Y + H - 1) <= pc.LAST_COMPLETE_YEAR
            train_all, test = [], []
            for c in cids:
                yrs, codes, _ = streams[c]
                for t in range(1975 + (Y % STEP), Y - H + 1, STEP):
                    u = unit_full(c, t, H, yrs, codes, tf[c])
                    if u and u['accrued']: train_all.append(u)
                u = unit_full(c, Y, H, yrs, codes, tf[c])
                if u: test.append(u)
            acc = np.array([u['accrued'] for u in test]); ev = np.array([u['y'] for u in test])
            c_row = {'horizon': H, 'origin': Y, 'window': f'[{Y}, {Y + H})', 'complete': complete, 'candidate_claims': len(test), 'accrued': int(acc.sum()), 'censored_low_accrual': int((acc == 0).sum()),
                     'events_among_accrued': int((ev == 1).sum()), 'accrual_rate': round(float(acc.mean()), 4) if len(test) else None,
                     'event_rate_among_accrued': round(float((ev == 1).sum() / max(acc.sum(), 1)), 4), 'unconditional_event_rate_lower_bound': round(float((ev == 1).sum() / max(len(test), 1)), 4)}
            census.append(c_row)
            if not complete or acc.sum() < 3 or (ev == 1).sum() < 3 or len(train_all) < 30: c_row['status'] = 'excluded (incomplete window)' if not complete else 'not evaluable'; continue
            c_row['status'] = 'primary'
            # claim-disjoint fits: for each test claim drop its own training units, refit, score that claim
            keys2 = ['pooled_direction', 'agreement']; keys7 = nm.COMPACT7
            s2 = np.zeros(len(test)); s7 = np.zeros(len(test)); sacc = np.zeros(len(test)); sdep = np.zeros(len(test)); s2_nd = None
            ytr_all = np.array([u['y'] for u in train_all]); gtr = np.array([u['claim_id'] for u in train_all])
            ln_pre = np.log1p(np.array([u['n_plus'] + u['n_minus'] for u in train_all])); ln_rate = np.log1p(np.array([u['rate5'] for u in train_all])); ln_m = np.log1p(np.array([u['m_signed'] for u in train_all]))
            # accrual model: logistic on log n_pre, log accrual rate -> P(A) using ALL earlier cohort units (accrued or not)
            train_cohort = []
            for c in cids:
                yrs, codes, _ = streams[c]
                for t in range(1975 + (Y % STEP), Y - H + 1, STEP):
                    u = unit_full(c, t, H, yrs, codes, tf[c])
                    if u: train_cohort.append(u)
            Xa_tr = np.array([[np.log1p(u['n_plus'] + u['n_minus']), np.log1p(u['rate5']), np.log1p(u['n_pre_t'])] for u in train_cohort]); ya_tr = np.array([u['accrued'] for u in train_cohort]); ga_tr = np.array([u['claim_id'] for u in train_cohort])
            for j, u in enumerate(test):
                keep = gtr != u['claim_id']
                pass
            # pc has no fit_predict; do it inline
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            def fp(Xtr, ytr, Xte):
                sc = StandardScaler().fit(Xtr); m = LogisticRegression(max_iter=3000, class_weight='balanced').fit(sc.transform(Xtr), ytr); return m.predict_proba(sc.transform(Xte))[:, 1]
            for j, u in enumerate(test):
                keep = gtr != u['claim_id']; TR = [tt for tt, k in zip(train_all, keep) if k]
                s2[j] = fp(X(TR, keys2), ytr_all[keep], X([u], keys2))[0]; s7[j] = fp(X(TR, keys7), ytr_all[keep], X([u], keys7))[0]
                b, sd = pc.fit_count_model(ln_pre[keep], ln_rate[keep], ln_m[keep])
                sdep[j] = pc.bb_change_prob(u['n_plus'], u['n_minus'], pc.draw_counts(b, sd, np.log1p(u['n_plus'] + u['n_minus']), np.log1p(u['rate5']), rng), rng=rng, s0=pc.sgn0(u['pooled_direction']))
                keepa = ga_tr != u['claim_id']
                sacc[j] = fp(Xa_tr[keepa], ya_tr[keepa], np.array([[np.log1p(u['n_plus'] + u['n_minus']), np.log1p(u['rate5']), np.log1p(u['n_pre_t'])]]))[0] if len(np.unique(ya_tr[keepa])) > 1 else float(ya_tr[keepa].mean())
            su = 1 - np.abs(np.array([u['pooled_direction'] for u in test])); g = np.array([u['claim_id'] for u in test])
            m = acc == 1; yy = ev[m]
            row = {**{k: c_row[k] for k in ('horizon', 'origin', 'window', 'candidate_claims', 'accrued', 'censored_low_accrual', 'events_among_accrued', 'accrual_rate')}}
            row['accrual_model_auroc'] = round(float(pc.AUROC(acc, sacc)), 4) if 0 < acc.sum() < len(acc) else 'NA'
            for name, s in (('margin', su), ('two_var', s2), ('compact7', s7), ('bb_deployable', sdep)):
                row[f'{name}_auroc_cond'] = round(float(pc.AUROC(yy, s[m])), 4); row[f'{name}_auprc_cond'] = round(float(pc.AUPRC(yy, s[m])), 4)
                # combined risk on the full cohort, censored units scored but not evaluated as outcomes: report ranking of accrued-and-changed vs everything else among ALL candidates
                yall = (ev == 1).astype(int); comb = sacc * s
                row[f'{name}_auroc_accrual_x_change_all_candidates'] = round(float(pc.AUROC(yall, comb)), 4)
            for name, s in (('two_var', s2), ('compact7', s7), ('bb_deployable', sdep)):
                d, c_ = pc.paired_ci(yy, s[m], su[m], g[m], pc.AUROC, n=500); row[f'{name}_minus_margin_auroc_cond'] = f'{d:+.3f} {c_}'
            rows.append(row)
            # training-only per-origin calibrators: Platt on the in-sample scores of one fit on all accrued training units
            cal = {}
            for name, keys, s in (('two_var', keys2, s2), ('compact7', keys7, s7)):
                p_tr = fp(X(train_all, keys), ytr_all, X(train_all, keys)); cal[name] = pc.platt_apply(pc.platt_fit(p_tr, ytr_all), s)
            cal['margin'] = pc.platt_apply(pc.platt_fit(1 - np.abs(np.array([u['pooled_direction'] for u in train_all])), ytr_all), su); cal['bb_deployable'] = sdep
            for k, v in (('y', yy), ('s2', s2[m]), ('s7', s7[m]), ('su', su[m]), ('sdep', sdep[m]), ('g', g[m]), ('yall', (ev == 1).astype(int)), ('comb2', sacc * s2), ('combu', sacc * su), ('gall', g),
                         ('p2', cal['two_var'][m]), ('p7', cal['compact7'][m]), ('pu', cal['margin'][m]), ('origin', [Y] * int(m.sum())), ('cid', g[m]), ('nsig', [u['n_plus'] + u['n_minus'] for u, mm in zip(test, m) if mm])): pooled[k] += list(v)
            print(f"  H={H} Y={Y}: candidates {len(test)} accrued {int(acc.sum())} censored {int((acc==0).sum())} events {int((ev==1).sum())} | cond AUROC margin {row['margin_auroc_cond']} 2v {row['two_var_auroc_cond']} C7 {row['compact7_auroc_cond']} bb {row['bb_deployable_auroc_cond']} | accrual AUROC {row['accrual_model_auroc']}", flush=True)
        if pooled['y']:
            yy = np.array(pooled['y']); g = np.array(pooled['g']); su = np.array(pooled['su']); P = {'horizon': H, 'accrued_units': len(yy), 'events': int(yy.sum()), 'positive_claims': int(len(set(g[yy == 1]))), 'candidates_all': len(pooled['yall']), 'censored': int(len(pooled['yall']) - len(yy))}
            for name, key in (('margin', 'su'), ('two_var', 's2'), ('compact7', 's7'), ('bb_deployable', 'sdep')):
                s = np.array(pooled[key]); P[f'{name}_auroc'] = round(float(pc.AUROC(yy, s)), 4); P[f'{name}_auroc_ci'] = pc.ci(yy, s, g, pc.AUROC); P[f'{name}_auprc'] = round(float(pc.AUPRC(yy, s)), 4); P[f'{name}_auprc_ci'] = pc.ci(yy, s, g, pc.AUPRC)
                if name != 'margin':
                    P[f'{name}_minus_margin_auroc'] = pc.paired_ci(yy, s, su, g, pc.AUROC); P[f'{name}_minus_margin_auprc'] = pc.paired_ci(yy, s, su, g, pc.AUPRC)
            yall = np.array(pooled['yall']); gall = np.array(pooled['gall'])
            P['two_var_x_accrual_auroc_all_candidates'] = round(float(pc.AUROC(yall, np.array(pooled['comb2']))), 4); P['margin_x_accrual_auroc_all_candidates'] = round(float(pc.AUROC(yall, np.array(pooled['combu']))), 4)
            P['two_var_x_accrual_minus_margin_x_accrual_auroc'] = pc.paired_ci(yall, np.array(pooled['comb2']), np.array(pooled['combu']), gall, pc.AUROC)
            pc.save_csv([{'claim_id': pooled['cid'][i], 'origin': pooled['origin'][i], 'y': int(pooled['y'][i]), 'n_signed_pre': int(pooled['nsig'][i]), 'margin': pooled['su'][i], 'two_var': pooled['s2'][i], 'compact7': pooled['s7'][i], 'bb_deployable': pooled['sdep'][i],
                          'margin_cal': pooled['pu'][i], 'two_var_cal': pooled['p2'][i], 'compact7_cal': pooled['p7'][i]} for i in range(len(yy))], f'issue5/pooled_scores_H{H}.csv')
            res['pooled'][f'H{H}'] = P; print(f"  H={H} POOLED (claim-disjoint, complete): accrued {len(yy)}/{int(yy.sum())} of {len(yall)} candidates | margin {P['margin_auroc']} 2v {P['two_var_auroc']} {P['two_var_minus_margin_auroc']} C7 {P['compact7_auroc']} bb {P['bb_deployable_auroc']} | all-candidate 2v x acc {P['two_var_x_accrual_auroc_all_candidates']} margin x acc {P['margin_x_accrual_auroc_all_candidates']}", flush=True)
    res['census'] = census; res['by_origin'] = rows
    pc.save_json(res, 'issue5/origin_cohort.json'); pc.save_csv(census, 'issue5/census.csv'); pc.save_csv(rows, 'issue5/by_origin.csv'); print('DONE', flush=True)


if __name__ == '__main__':
    main()
