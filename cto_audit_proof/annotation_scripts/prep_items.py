"""Build the annotation item list from CTO's released GPT-labelling-function file.

CTO's `gpt` labelling function prompted GPT-3.5-turbo-0125 on the top-2 PubMed abstracts by title
similarity and asked it to "summarize important trial-related statistical tests and predict the
outcome". The released file ships those abstracts inline, so our pipelines see EXACTLY the text
GPT-3.5 saw -- no refetch, no reconstruction caveat.

One item per (trial, abstract slot). Per-abstract labels are stored and aggregated at analysis time,
because CTO does not document how it combined the two slots into one `gpt` value; keeping them
separate lets the aggregation rule be stated rather than assumed.
"""
import csv, json, os, sys
csv.field_size_limit(sys.maxsize)
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET=3000
out=[]
with open(f'{HERE}/data/pubmed_gpt_outcomes.csv') as f:
    for r in csv.DictReader(f):
        for slot in (1,2):
            ab=(r.get(f'top_{slot}_similar_article_abstract') or '').strip()
            if not ab: continue
            ti=(r.get(f'top_{slot}_similar_article_title') or '').strip()
            out.append(dict(nctid=r['nctid'],slot=slot,
                            pmid=(r.get(f'top_{slot}_similar_article_PMID') or '').strip(),
                            official_title=(r.get('official_title') or '').strip()[:400],
                            text=((ti+'. '+ab)[:BUDGET]).replace('\n',' '),
                            cto_outcome=r.get('outcome','')))
with open(f'{HERE}/data/items.jsonl','w') as f:
    for x in out: f.write(json.dumps(x)+'\n')
print(f'items {len(out):,}  trials {len({x["nctid"] for x in out}):,}')
import collections
print('  by slot:',dict(collections.Counter(x['slot'] for x in out)))
print(f'  text length: median {sorted(len(x["text"]) for x in out)[len(out)//2]:,}  max {max(len(x["text"]) for x in out):,}')
