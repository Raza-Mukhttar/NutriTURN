"""E9b: embedding baseline that reads the abstracts, on the origin-defined calendar cohort.

Temporal validity: only records with year < origin enter a unit's features; training units are
those whose horizon outcome is observable by the origin, on the same two-year grid as the
composition models; training is claim-disjoint; the evaluated cells are those of the calendar
cohort.

  python analyses/nfx_embed.py embed --gpu 0          # one vector per distinct PubMed record
  python analyses/nfx_embed.py eval                   # features, models, cells

Encoder: a biomedical BERT already present in the local cache; mean-pooled over the title and
abstract, L2-normalised. Unit features from the pre-origin records: the mean embedding of all of
them, the mean of the most recent 20, their difference, and separate means over the records the
primary pipeline labelled HARMFUL and PROTECTIVE. Dimensionality is reduced by a PCA fitted on the
training units of each origin only (unsupervised, no outcome involved).

Models: class-balanced L2 logistic regression on [PCA(64), p, a] (text + composition), on the PCA
block alone (text only), and on (p, a) alone (composition only, i.e. the two-variable score).
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
import os, sys, json, argparse, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

ENCODER = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
EMB_PATH = os.path.join(N.DATA, 'record_embeddings.npz')
MAXTOK = 320
RECENT = 20
NPCA = 64


def texts_by_pmid():
    out = {}
    for p in sorted(glob.glob(os.path.join(N.SNAPD, 'records', '*.json'))):
        for r in json.load(open(p))['records']:
            k = str(r['pmid'])
            if k not in out:
                out[k] = ((r.get('title') or '') + '. ' + (r.get('abstract') or '')).strip()
    return out


def cmd_embed(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from transformers import AutoTokenizer, AutoModel
    T = texts_by_pmid()
    pmids = sorted(T)
    print(f'embedding {len(pmids)} distinct records with {ENCODER}', flush=True)
    tok = AutoTokenizer.from_pretrained(ENCODER)
    mdl = AutoModel.from_pretrained(ENCODER, torch_dtype=torch.float16).to('cuda:0').eval()
    B = args.batch
    V = np.zeros((len(pmids), mdl.config.hidden_size), dtype=np.float16)
    import time
    t0 = time.time()
    for i in range(0, len(pmids), B):
        batch = [T[p] for p in pmids[i:i + B]]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=MAXTOK)
        enc = {k: v.to('cuda:0') for k, v in enc.items()}
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state
        m = enc['attention_mask'].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        v = torch.nn.functional.normalize(v.float(), dim=-1)
        V[i:i + len(batch)] = v.half().cpu().numpy()
        if (i // B) % 200 == 0:
            el = time.time() - t0
            print(f'  {i+len(batch)}/{len(pmids)}  {(i+len(batch))/max(el,1e-9):.0f} rec/s', flush=True)
    np.savez_compressed(EMB_PATH, pmids=np.array(pmids), V=V)
    print('written', EMB_PATH, V.shape, flush=True)


def unit_text_features(cid, t, S, IDX, V, assoc_by_claim):
    """Mean embedding of pre-origin records, of the most recent 20, their difference, and the
    per-label means over HARMFUL and PROTECTIVE records."""
    rows = assoc_by_claim[cid]
    pre = [r for r in rows if r['year'] < t]
    if not pre:
        d = V.shape[1]
        return np.zeros(5 * d, dtype=np.float32)
    ix = np.array([IDX[r['pmid']] for r in pre])
    E = V[ix].astype(np.float32)
    yrs = np.array([r['year'] for r in pre])
    o = np.argsort(yrs, kind='stable')
    allm = E.mean(0)
    rec = E[o[-RECENT:]].mean(0)
    early = E[o[:-RECENT]].mean(0) if len(o) > RECENT else allm
    lab = np.array([r['label_primary'] for r in pre])
    hm = E[lab == 'HARMFUL'].mean(0) if (lab == 'HARMFUL').any() else np.zeros_like(allm)
    pm = E[lab == 'PROTECTIVE'].mean(0) if (lab == 'PROTECTIVE').any() else np.zeros_like(allm)
    return np.concatenate([allm, rec, rec - early, hm, pm]).astype(np.float32)


def cmd_eval(args):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if not os.path.exists(EMB_PATH):
        raise SystemExit('run "embed" first')
    z = np.load(EMB_PATH, allow_pickle=True)
    pmids = list(z['pmids']); V = z['V']
    IDX = {str(p): i for i, p in enumerate(pmids)}
    A = N.assoc()
    by = {}
    for r in A:
        by.setdefault(r['claim_id'], []).append(r)
    S = N.streams('primary')
    co = N.build_cohort('primary')
    out = {'encoder': ENCODER, 'n_pca': NPCA, 'cells': []}
    print(f'E9b embedding forecaster on the calendar cohort (encoder {ENCODER})\n')
    cache = {}

    def feats(cid, t):
        k = (cid, t)
        if k not in cache:
            cache[k] = unit_text_features(cid, t, S, IDX, V, by)
        return cache[k]

    for H in (3, 5, 10):
        d = [r for r in co if r['horizon'] == H]
        preds = {m: [] for m in ('text_only', 'text_plus_comp', 'comp_only')}
        ys, gs, sms = [], [], []
        for Y in sorted({r['origin'] for r in d}):
            te = [r for r in d if r['origin'] == Y and r['accrued'] == 1]
            if not te:
                continue
            tr = N.cohort_training_units(H, Y, S)
            if len(tr) < N.TRAIN_MIN:
                continue
            Xtr_t = np.stack([feats(u['claim_id'], u['origin']) for u in tr])
            # unsupervised PCA on the training units of this origin only
            npc = min(NPCA, Xtr_t.shape[0] - 1, Xtr_t.shape[1])
            pca = PCA(n_components=npc, random_state=N.SEED).fit(Xtr_t)
            Ztr = pca.transform(Xtr_t)
            Ctr = np.array([[u['p'], u['a']] for u in tr], float)
            ytr_all = np.array([u['y'] for u in tr])
            gtr = np.array([u['claim_id'] for u in tr])
            for r in te:
                keep = gtr != r['claim_id']
                yk = ytr_all[keep]
                zt = pca.transform(feats(r['claim_id'], r['origin'])[None, :])
                ct = np.array([[r['p'], r['a']]], float)
                for name, Xt, Xe in (('text_only', Ztr[keep], zt),
                                     ('text_plus_comp', np.hstack([Ztr[keep], Ctr[keep]]),
                                      np.hstack([zt, ct])),
                                     ('comp_only', Ctr[keep], ct)):
                    if len(np.unique(yk)) < 2 or len(yk) < N.TRAIN_MIN:
                        preds[name].append(0.5); continue
                    sc = StandardScaler().fit(Xt)
                    m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000)
                    m.fit(sc.transform(Xt), yk)
                    preds[name].append(float(m.predict_proba(sc.transform(Xe))[0, 1]))
                ys.append(r['y']); gs.append(r['claim_id']); sms.append(1 - abs(r['p']))
        y = np.array(ys); g = np.array(gs); sm = np.array(sms)
        cell = dict(horizon=H, units=len(y), events=int(y.sum()))
        for name in preds:
            s = np.array(preds[name])
            cell[f'{name}_auroc'] = N.auroc(y, s)
            cell[f'{name}_auprc'] = N.auprc(y, s)
            dd, lo, hi, _ = N.paired_ci(y, s, sm, g)
            cell[f'{name}_minus_margin'] = [dd, lo, hi]
            cell[f'{name}_established'] = N.established(lo, hi)
        st = np.array(preds['text_plus_comp']); sc_ = np.array(preds['comp_only'])
        dd, lo, hi, _ = N.paired_ci(y, st, sc_, g)
        cell['text_plus_comp_minus_comp_only'] = [dd, lo, hi]
        cell['text_adds_established'] = N.established(lo, hi)
        cell['margin_auroc'] = N.auroc(y, sm)
        out['cells'].append(cell)
        print(f"H{H} ({len(y)} units, {int(y.sum())} events)  margin {cell['margin_auroc']:.3f}")
        for name in ('text_only', 'text_plus_comp', 'comp_only'):
            dd, lo, hi = cell[f'{name}_minus_margin']
            print(f"   {name:16s} AUROC {cell[f'{name}_auroc']:.3f}  AUPRC {cell[f'{name}_auprc']:.3f}  "
                  f"minus margin {dd:+.3f} {N.fmt_ci(lo,hi)}"
                  f"{'  ESTABLISHED' if cell[f'{name}_established'] else ''}")
        dd, lo, hi = cell['text_plus_comp_minus_comp_only']
        print(f"   text adds over composition alone: {dd:+.3f} {N.fmt_ci(lo,hi)}"
              f"{'  ESTABLISHED' if cell['text_adds_established'] else '  not established'}\n")
    print('written', N.save_json(out, 'E9b_embeddings.json'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('embed'); s.add_argument('--gpu', default='0'); s.add_argument('--batch', type=int, default=128)
    sub.add_parser('eval')
    a = ap.parse_args()
    {'embed': cmd_embed, 'eval': cmd_eval}[a.cmd](a)
