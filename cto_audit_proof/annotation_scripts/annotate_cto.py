"""Re-run CTO's LLM labelling function with our pipelines, under NutriTURN's own configuration.

CTO used GPT-3.5-turbo-0125 on the top-2 PubMed abstracts per trial. This substitutes Llama-3.1-8B
and Qwen2.5-7B -- the two pipelines the paper audits -- holding everything else at NutriTURN's
frozen settings: full-string read-out (summed label-token log-probabilities, softmax over the label
set), 3,000-character text budget, 1,600-token truncation, left padding, bfloat16, one forward pass,
no sampling. That makes the substitution a clean swap of annotator, not of method.

TRUNCATION SIDE. The shipped NutriTURN annotator leaves truncation_side at the HuggingFace default
('right'), which drops the assistant header on over-long prompts. CTO abstracts are longer than
PubMed nutrition abstracts, so that failure mode is likelier here; truncation_side is set to 'left'
so the header always survives, and the count of prompts that would have exceeded the limit is logged.

LABEL SET. CTO's gpt column is binary with abstention (1 success / 0 failure / -1 abstain), so the
label set is SUCCESS / FAILURE / UNCLEAR, the last playing the role of the released -1.
"""
import argparse, glob, hashlib, json, math, os, sys, time
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS=['SUCCESS','FAILURE','UNCLEAR']
MAX_LEN,BATCH=1600,8
SYS=('You assess whether a clinical trial met its primary objective, based on a related publication. '
     'Answer with exactly one word: SUCCESS if the publication reports the trial met its primary '
     'endpoint or showed a statistically significant benefit, FAILURE if it reports the trial did not '
     'meet its primary endpoint or showed no significant benefit, UNCLEAR if the publication does not '
     'report the trial outcome or its direction cannot be determined.')
MODELS={'llama8b':'meta-llama/Llama-3.1-8B-Instruct','qwen7b':'Qwen/Qwen2.5-7B-Instruct'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,choices=list(MODELS))
    ap.add_argument('--gpu',required=True); ap.add_argument('--keyfile',required=True)
    ap.add_argument('--tag',required=True); ap.add_argument('--batch',type=int,default=BATCH)
    a=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    REPO=MODELS[a.model]
    outd=f'{HERE}/data/annot_{a.model}'; os.makedirs(outd,exist_ok=True)
    out_path=f'{outd}/{a.tag}.jsonl'
    want=set()
    for l in open(a.keyfile):
        l=l.rstrip('\n')
        if l: n,s=l.split('\t'); want.add((n,int(s)))
    items=[json.loads(l) for l in open(f'{HERE}/data/items.jsonl') if l.strip()]
    items=[x for x in items if (x['nctid'],x['slot']) in want]
    done=set()
    for f in sorted(glob.glob(f'{outd}/*.jsonl')):
        for l in open(f):
            try: d=json.loads(l); done.add((d['nctid'],d['slot']))
            except Exception: pass
    items=[x for x in items if (x['nctid'],x['slot']) not in done]
    print(f'[{a.model} {a.tag} gpu {a.gpu}] {len(items):,} to score ({len(done):,} present)  {REPO}',flush=True)
    if not items: return 0
    tok=AutoTokenizer.from_pretrained(REPO)
    tok.padding_side='left'; tok.truncation_side='left'
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(REPO,torch_dtype=torch.bfloat16,device_map='cuda:0').eval()
    lab_ids=[tok.encode(l,add_special_tokens=False) for l in LABELS]
    maxlab=max(len(x) for x in lab_ids); first=[x[0] for x in lab_ids]
    print('label tokenisations:',dict(zip(LABELS,lab_ids)),'distinct first:',len(set(first))==len(LABELS),flush=True)
    def build(x):
        u=f"Trial: {x['official_title']}\n\nPublication: {x['text']}\n\nOutcome:"
        return tok.apply_chat_template([{'role':'system','content':SYS},{'role':'user','content':u}],
                                       tokenize=False,add_generation_prompt=True)
    fh=open(out_path,'a'); t0=time.time(); n=0; over=0
    for i in range(0,len(items),a.batch):
        ch=items[i:i+a.batch]
        raw=[build(x) for x in ch]
        over+=sum(1 for s in raw if len(tok(s,add_special_tokens=False)['input_ids'])>MAX_LEN)
        enc=tok(raw,return_tensors='pt',padding=True,truncation=True,max_length=MAX_LEN,
                add_special_tokens=False)
        P,M=enc['input_ids'],enc['attention_mask']; L=P.shape[1]
        seqs,masks,pos=[],[],[]
        for b in range(P.shape[0]):
            for lt in lab_ids:
                pad=maxlab-len(lt)
                seqs.append(torch.cat([torch.full((pad,),tok.pad_token_id),P[b],torch.tensor(lt)]))
                masks.append(torch.cat([torch.zeros(pad,dtype=M.dtype),M[b],torch.ones(len(lt),dtype=M.dtype)]))
                pos.append((pad+L,len(lt)))
        S=torch.stack(seqs).to('cuda:0'); MM=torch.stack(masks).to('cuda:0')
        T=S.shape[1]; keep=maxlab+1
        with torch.no_grad():
            h=model.model(input_ids=S,attention_mask=MM).last_hidden_state[:,-keep:,:]
            lp=torch.log_softmax(model.lm_head(h).float(),-1)
        off=T-keep; full=[]
        for k,(st,ln) in enumerate(pos):
            v=0.0
            for j in range(ln): v+=lp[k,st+j-1-off,S[k,st+j]].item()
            full.append(v)
        nl=len(LABELS)
        for b,x in enumerate(ch):
            r=full[b*nl:(b+1)*nl]; mx=max(r); w=[math.exp(z-mx) for z in r]; Z=sum(w)
            p=[z/Z for z in w]
            fh.write(json.dumps({'nctid':x['nctid'],'slot':x['slot'],'pmid':x['pmid'],
                'label':LABELS[max(range(nl),key=lambda j:p[j])],
                'p':[round(z,5) for z in p],'model':a.model})+'\n')
        n+=len(ch)
        if (i//a.batch)%200==0:
            fh.flush(); el=max(time.time()-t0,1e-9)
            print(f'  {n:,}/{len(items):,}  {n/el:.1f} p/s  eta {(len(items)-n)/max(n/el,1e-9)/60:.1f} min',flush=True)
    fh.close()
    print(f'DONE {n:,} in {round(time.time()-t0)} s   prompts over {MAX_LEN} tokens: {over}',flush=True)
    return 0
if __name__=='__main__': sys.exit(main())
