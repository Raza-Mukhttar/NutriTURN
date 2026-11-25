"""Noise-calibrated event overlap: how much overlap should two annotators share by construction?

The endpoint is a threshold on a vote-count ratio, so label disagreement near a zero margin flips
events whether or not the disagreement is systematic. This asks, for a pair (X, Y): given the
record-level confusion between them, what event overlap would a RANDOM annotator with the same
confusion produce?

Synthetic X': for every association, redraw the label from P(Y | X's label). X' has the same label
shift and the same record-level agreement with X as Y does, but its disagreements are placed at
random. Rebuilding the endpoint for X' and comparing Jaccard(X, X') with the observed Jaccard(X, Y)
separates "the endpoint amplifies ordinary disagreement" from "the disagreement is structured".

A ladder of nulls locates any structure, mirroring the paper's Nulls A-D:
  N1  confusion estimated globally          preserves overall label shift
  N2  per claim                             preserves claim-specific annotator bias
  N3  per claim x decade                    preserves claim- and era-specific bias
A stratum with fewer than MIN_STRAT records falls back to the global matrix, recorded per null.

The batch-size rerun is a POSITIVE CONTROL: its disagreement is run-to-run noise, so its observed
overlap should sit inside the null range. If it does not, the method is wrong, not the pipelines.

Endpoint: both rules are reported. 'original' is the endpoint of the paper; 'gate-corrected' treats
a sparse-gated history as a non-event, as in the sparse-gate sensitivity.

Reads only data_release/ (labels, identifiers, years); needs numpy and nothing else.

  python noise_null_calibration/noise_null_overlap.py        # CPU only; REPS=1000 by default
"""
import collections, csv, glob, json, os, sys, zlib
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
REL=os.environ.get('NUTRITURN_DATA_RELEASE',os.path.join(os.path.dirname(HERE),'data_release'))
if not os.path.isdir(REL):
    raise SystemExit(f'data_release/ not found at {REL}; set NUTRITURN_DATA_RELEASE to its path')
TABLES=f'{REL}/02_records/cohort_tables'
CONTROLLED=f'{REL}/03_annotations/controlled_cells'
RAW=f'{REL}/03_annotations/controlled_cells_raw'
LARGE=f'{REL}/03_annotations/large_cells'
SEED,GATE=3,8      # the study seed; signed pre-cutoff records needed before p_t is non-zero
ASSOC=list(csv.DictReader(open(f'{TABLES}/associations.csv')))
REPS=int(os.environ.get('REPS','1000')); MIN_STRAT=40
LAB=['PROTECTIVE','HARMFUL','NULL','UNCLEAR']           # codes +1,-1,0,99 -> 0,1,2,3 internally
C2I={'PROTECTIVE':0,'HARMFUL':1,'NULL':2,'UNCLEAR':3}
SGN=np.array([1,-1,0,99])

def load(d,field='label_full'):
    o={}
    for f in sorted(glob.glob(d)):
        for l in open(f):
            if l.strip():
                r=json.loads(l); k=(r['claim_id'],str(r['pmid']))
                if k not in o: o[k]=r[field]
    return o
def released(name,field='label'):
    o={}
    p=f'{CONTROLLED}/cell{name}.jsonl'
    for l in open(p):
        d=json.loads(l); o[(d['claim_id'],str(d['pmid']))]=d[field]
    return o

# ---------- association index, shared by every rebuild ----------
A=[(r['claim_id'],r['pmid'],int(r['year'])) for r in ASSOC]
AIDX={(c,p):i for i,(c,p,_) in enumerate(A)}
claim_of=np.array([c for c,_,_ in A]); year_of=np.array([y for _,_,y in A],dtype=int)
units=[(u['claim_id'],int(u['cutoff'])) for u in csv.DictReader(open(f'{TABLES}/units.csv'))]
by_claim=collections.defaultdict(list)
for i,(c,_,_) in enumerate(A): by_claim[c].append(i)
pre_u=[];pre_r=[];post_u=[];post_r=[]
for ui,(cid,t) in enumerate(units):
    for i in by_claim[cid]:
        (pre_u if year_of[i]<t else post_u).append(ui)
        (pre_r if year_of[i]<t else post_r).append(i)
pre_u=np.array(pre_u);pre_r=np.array(pre_r);post_u=np.array(post_u);post_r=np.array(post_r)
NU=len(units)
print(f'{len(A):,} associations   {NU:,} units   pre entries {len(pre_r):,}  post {len(post_r):,}')

def events(code,gate_corrected=True):
    """code: int array over associations, values 0..3. Returns a boolean event vector of length NU."""
    bc=lambda u,r,v: np.bincount(u[code[r]==v],minlength=NU)
    h=bc(pre_u,pre_r,0); l=bc(pre_u,pre_r,1)
    ph=bc(post_u,post_r,0); pl=bc(post_u,post_r,1)
    n=h+l; m=ph+pl; gated=n<GATE
    d=h-l; s=np.sign(d); s[gated]=0
    sp=np.sign(ph-pl)
    y=(sp!=s)
    if gate_corrected: y=y&~gated
    return y
def jac(a,b):
    i=int((a&b).sum()); u=int((a|b).sum()); return i,u,i/max(u,1)
def codes_of(lab):
    out=np.full(len(A),3,dtype=np.int8)
    for k,v in lab.items():
        i=AIDX.get(k)
        if i is not None: out[i]=C2I.get(v,3)
    return out

def strata(level):
    if level=='N1': return np.zeros(len(A),dtype=int),1
    if level=='N2':
        u={c:i for i,c in enumerate(sorted(set(claim_of)))}
        return np.array([u[c] for c in claim_of]),len(u)
    dec=(year_of//10)*10
    keys=sorted({(c,d) for c,d in zip(claim_of,dec)})
    u={k:i for i,k in enumerate(keys)}
    return np.array([u[(c,d)] for c,d in zip(claim_of,dec)]),len(u)

def null_dist(cx,cy,level):
    """Per-stratum cumulative P(Y|X); strata below MIN_STRAT fall back to global."""
    st,ns=strata(level)
    G=np.zeros((4,4))
    for x in range(4):
        m=cx==x
        G[x]=np.bincount(cy[m],minlength=4) if m.sum() else np.ones(4)
        G[x]=G[x]/max(G[x].sum(),1)
    CUM=np.tile(G.cumsum(1),(ns,1,1))
    fell=0
    for s in range(ns):
        ms=st==s
        if ms.sum()<MIN_STRAT: fell+=1; continue
        for x in range(4):
            m=ms&(cx==x)
            if m.sum()<5: continue
            row=np.bincount(cy[m],minlength=4).astype(float); row/=row.sum()
            CUM[s,x]=row.cumsum()
    return st,CUM,fell

def run_null(cx,cy,level,reps,rng,gate_corrected=True):
    st,CUM,fell=null_dist(cx,cy,level)
    # CUM[st, cx] is constant across repetitions -- materialise it ONCE. Recomputing this fancy
    # index per draw dominated the runtime (measured: ~0.3 s/rebuild vs ~0.03 s after hoisting).
    ROW=CUM[st,cx]                      # (n_assoc, 4) cumulative P(Y | X label), per record
    ex=events(cx,gate_corrected)
    out=np.empty(reps)
    for i in range(reps):
        u=rng.random(len(A))
        draw=(u[:,None]>ROW).sum(1).astype(np.int8)
        out[i]=jac(ex,events(draw,gate_corrected))[2]
    return out,fell

P={}
P['primary']={(r['claim_id'],r['pmid']):r['label_primary'] for r in ASSOC}
P['second']={(r['claim_id'],r['pmid']):r['label_second'] for r in ASSOC}
P['Llama-8B P2']=load(f'{RAW}/cellB_shard*.jsonl')
P['Qwen-7B P2']=load(f'{RAW}/cellD_shard*.jsonl')
P['Llama-70B P2']=load(f'{LARGE}/cell_llama70.jsonl')
P['Qwen-72B P2']=load(f'{LARGE}/cell_qwen72.jsonl')
P['batch-size rerun']=released('A')
CODE={k:codes_of(v) for k,v in P.items()}
PAIRS=[('primary','second','headline pair'),
       ('Llama-70B P2','Qwen-72B P2','70B-class pair'),
       ('Llama-8B P2','Llama-70B P2','scale at fixed prompt'),
       ('Llama-8B P2','Qwen-7B P2','family swap at fixed prompt'),
       ('primary','batch-size rerun','POSITIVE CONTROL')]
res={}
for gc,tag in ((True,'gate-corrected'),(False,'original')):
    print(f'\n{"="*108}\nENDPOINT: {tag}\n{"="*108}')
    print(f'{"pair":42s}{"obs J":>8s}  ' + ''.join(f'{n+" null mean [95%]":>30s}' for n in ('N1','N2','N3')))
    for x,y,lab in PAIRS:
        ex,ey=events(CODE[x],gc),events(CODE[y],gc)
        i,u,obs=jac(ex,ey)
        row={'observed':obs,'shared':i,'union':u,'events_x':int(ex.sum()),'events_y':int(ey.sum())}
        line=f'{lab+" ["+x+"~"+y+"]":42s}{obs:8.3f}  '
        for lev in ('N1','N2','N3'):
            # seed from a stable checksum of the cell name, so draws do not depend on PYTHONHASHSEED
            rng=np.random.default_rng([SEED,zlib.crc32(f'{x}|{y}|{lev}|{int(gc)}'.encode())])
            a,fa=run_null(CODE[x],CODE[y],lev,REPS,rng,gc)
            b,fb=run_null(CODE[y],CODE[x],lev,REPS,rng,gc)
            allv=np.concatenate([a,b])
            lo,hi=np.percentile(allv,[2.5,97.5])
            inside=bool(lo<=obs<=hi)
            row[lev]=dict(mean=float(allv.mean()),lo=float(lo),hi=float(hi),
                          observed_inside=inside,fallback_strata=[fa,fb],
                          mean_xy=float(a.mean()),mean_yx=float(b.mean()))
            line+=f'{allv.mean():8.3f} [{lo:.3f},{hi:.3f}]{"  in" if inside else " OUT":>5s}'
        res[f'{tag}|{lab}']=row
        print(line)
json.dump(dict(reps=REPS,seed=SEED,min_stratum=MIN_STRAT,results=res),
          open(f'{HERE}/noise_null_overlap.json','w'),indent=1,default=float)
print('\nwritten noise_null_calibration/noise_null_overlap.json')
