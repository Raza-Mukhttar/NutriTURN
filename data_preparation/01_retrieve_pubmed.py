"""NutriMATURE system, step 1 (optional, needs network): complete PubMed retrieval for a claim, with no record cap.

For each claim, the retrieval query (data/claim_queries.json: 'retrieval_query') is sent to NCBI ESearch with
usehistory=y. The whole server-side result set is then fetched with EFetch in blocks of 500 (retstart = 0, 500, ...)
and parsed with the frozen NutriMATURE PubMed parser. A claim is "fully retrieved" when the unique PMIDs retrieved
reach PubMed's hit count. The new retrieval is merged with the claim's existing history: records no longer returned by
PubMed are kept, so a history never loses a record.

By default the 30 claims whose original retrieval hit the historical 800-record cap are retrieved (the
'history_retrieved_in_full_after_800_cap' claims). This is how the shipped data/records were completed.
  python 01_retrieve_pubmed.py                    # retrieve, report completeness; does not touch data/records
  python 01_retrieve_pubmed.py --overwrite        # also write merged histories into data/records
  python 01_retrieve_pubmed.py --all              # every benchmark claim
  python 01_retrieve_pubmed.py --figures-only     # redraw the figure from tables/01_retrieval_completeness.json
PubMed keeps growing, so a new retrieval returns more records than the shipped September 2026 retrieval.
Afterwards run 02_extract_directions.py (label new records) and 03_build_benchmark.py.
Outputs: tables/01_retrieval_completeness.{csv,json}, figures/01_retrieval_completeness.png
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

import os, sys, json, re, time, argparse, urllib.parse, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import nutrimature as nm

EUT = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
BLOCK, RATE = 500, 0.36                     # EFetch block size; seconds between requests (NCBI: <= 3/s without an API key)
ART = re.compile(r'<(PubmedArticle|PubmedBookArticle)>.*?</\1>', re.S)
PMID = re.compile(r'<PMID[^>]*>(\d+)</PMID>')
_last = [0.0]


def parse(xml):
    """Frozen NutriMATURE PubMed XML parser (unchanged)."""
    recs = []
    for art in re.findall(r'<PubmedArticle>.*?</PubmedArticle>', xml, re.S):
        pm = re.search(r'<PMID[^>]*>(\d+)</PMID>', art)
        ti = re.search(r'<ArticleTitle[^>]*>(.*?)</ArticleTitle>', art, re.S)
        yr = re.search(r'<PubDate>.*?<Year>(\d{4})</Year>', art, re.S) or re.search(r'<Year>(\d{4})</Year>', art)
        abs_parts = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', art, re.S)
        pts = re.findall(r'<PublicationType[^>]*>(.*?)</PublicationType>', art, re.S)
        def clean(s): return re.sub(r'<[^>]+>', '', s or '').replace('&lt;','<').replace('&gt;','>').replace('&amp;','&').strip()
        if not (pm and yr): continue
        recs.append({'pmid': pm.group(1), 'year': int(yr.group(1)),
                     'title': clean(ti.group(1) if ti else ''),
                     'abstract': ' '.join(clean(a) for a in abs_parts),
                     'pub_types': [clean(p) for p in pts]})
    return recs


def get(url, tries=6):
    for a in range(tries):
        dt = time.time() - _last[0]
        if dt < RATE: time.sleep(RATE - dt)
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(url + ('&' if '?' in url else '?') + 'tool=nutrimature', timeout=120) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception:
            if a == tries - 1: raise
            time.sleep(3 * (a + 1))


def fetch_history(q):
    d = json.loads(get(EUT + 'esearch.fcgi?db=pubmed&usehistory=y&retmax=0&retmode=json&term=' + urllib.parse.quote(q)))['esearchresult']
    n, we, qk = int(d['count']), d.get('webenv'), d.get('querykey')
    pm, recs, failed = [], [], 0
    for start in range(0, n, BLOCK):
        try: xml = get(EUT + f'efetch.fcgi?db=pubmed&query_key={qk}&WebEnv={we}&retstart={start}&retmax={BLOCK}&retmode=xml')
        except Exception: failed += 1; continue
        pm += [PMID.search(a.group(0)).group(1) for a in ART.finditer(xml) if PMID.search(a.group(0))]; recs += parse(xml)
    return n, list(dict.fromkeys(pm)), recs, failed


def figure():
    plt = nm.plt_setup(); T = nm.table('01_retrieval_completeness'); rows = sorted(T['rows'], key=lambda r: r['provider_total_count'])
    y = np.arange(len(rows)); fig, ax = plt.subplots(figsize=(7.2, 0.2 * len(rows) + 1.4)); nm.style(ax, 'x')
    ax.hlines(y, [r['historical_capped_count'] for r in rows], [r['retrieved_unique_pmids'] for r in rows], color=nm.GRID, lw=2)
    ax.plot([r['historical_capped_count'] for r in rows], y, 'o', color=nm.ORANGE, ms=5, mec=nm.SURF, mew=1, label='records before (original 800-record cap)')
    ax.plot([r['retrieved_unique_pmids'] for r in rows], y, 'o', color=nm.BLUE, ms=5, mec=nm.SURF, mew=1, label='records retrieved in full')
    ax.set_yticks(y); ax.set_yticklabels([r['claim'].replace('_', ' ') for r in rows], fontsize=7, color=nm.INK)
    ax.set_xlabel('PubMed records per claim'); ax.legend(loc='lower right', fontsize=7.5)
    full = sum(bool(r['fully_uncapped']) for r in rows)
    ax.set_title(f'Complete retrieval of the {len(rows)} capped claims: {full}/{len(rows)} reach PubMed\'s hit count', loc='left', fontsize=9.5)
    nm.save_fig(fig, '01_retrieval_completeness')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--overwrite', action='store_true'); ap.add_argument('--all', action='store_true')
    ap.add_argument('--figures-only', action='store_true'); a = ap.parse_args()
    if a.figures_only: figure(); sys.exit(0)
    Q = json.load(open(nm.QUERIES))
    todo = [c for c in Q if a.all or Q[c]['history_retrieved_in_full_after_800_cap']]
    rows = []
    for i, cid in enumerate(todo):
        t0 = time.time(); prev = json.load(open(f'{nm.RECORDS}/{cid}.json'))['records']
        n, upm, recs, failed = fetch_history(Q[cid]['retrieval_query'])
        seen, urecs = set(), []
        for r in recs:
            if r['pmid'] not in seen: seen.add(r['pmid']); urecs.append(r)
        full = len(upm) >= n
        rows.append({'claim': cid, 'historical_capped_count': len(prev), 'provider_total_count': n, 'strategy': 'A', 'retrieved_unique_pmids': len(upm),
                     'records_parsed': len(urecs), 'missing_count': max(0, n - len(upm)), 'completeness_pct': round(100 * len(upm) / n, 2) if n else 100.0,
                     'fully_uncapped': bool(full), 'reason_if_incomplete': '' if full else (f'{failed} EFetch block(s) failed' if failed else 'provider returned fewer PMIDs than its count'),
                     'seconds': round(time.time() - t0, 1)})
        if a.overwrite:
            kept = [r for r in prev if r['pmid'] not in seen]                        # never drop a record from the history
            hist = urecs + kept
            json.dump({'claim_id': cid, 'n': len(hist), 'records': hist}, open(f'{nm.RECORDS}/{cid}.json', 'w'))
        print(f'[{i + 1}/{len(todo)}] {cid:44s} PubMed {n:6d} retrieved {len(upm):6d}  {"FULL" if full else "INCOMPLETE"}', flush=True)
    nm.save_table(rows, '01_retrieval_completeness', {'method': 'ESearch usehistory + EFetch from the history server, blocks of 500, no cap',
                  'rule': 'fully retrieved iff unique PMIDs >= PubMed hit count', 'records_written': bool(a.overwrite)})
    figure()
