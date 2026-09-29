"""Rebuild level 3: restore PubMed abstract text into the shipped record files.

The repository ships records without abstract text (identifiers, years, titles and publication types only), so
this script refetches the abstracts from PubMed EFetch for the archived identifiers and writes them back in
place. It is needed only to re-run the annotation models or to recompute the text-derived numeric and design
features; every analysis reported in the paper runs without it, because the parsed effect estimates are shipped
as a derived cache.

  python3 scripts/fetch_abstracts.py                 # all 165 claims
  python3 scripts/fetch_abstracts.py --claims a b     # selected claims
  python3 scripts/fetch_abstracts.py --dry-run        # report what is missing, fetch nothing

PubMed content retrieved now can differ from the archived snapshot: records are revised and removed over time.
results/snapshot_<timestamp>/ documents the identifier-level drift measured for this corpus.
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
import os as _os
_REPO = _NT
_PROTOCOL = _os.path.join(_REPO, 'protocol')

import argparse, json, os, sys, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET

DATA = os.path.join(_PROTOCOL, 'data', 'protocol_snapshot')
EFETCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
BATCH = 200
TOOL = {'tool': 'nutrimature_release', 'email': os.environ.get('NCBI_EMAIL', 'anonymous@example.org')}


def efetch(pmids):
    q = dict(TOOL, db='pubmed', id=','.join(pmids), retmode='xml')
    if os.environ.get('NCBI_API_KEY'):
        q['api_key'] = os.environ['NCBI_API_KEY']
    url = EFETCH + '?' + urllib.parse.urlencode(q)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(3 + 4 * attempt)


def abstracts_from(xml):
    out = {}
    root = ET.fromstring(xml)
    for art in root.iter('PubmedArticle'):
        pid = art.findtext('.//PMID')
        parts = []
        for ab in art.iter('AbstractText'):
            lab = ab.get('Label')
            txt = ''.join(ab.itertext()).strip()
            if txt:
                parts.append(f'{lab}: {txt}' if lab else txt)
        if pid and parts:
            out[pid] = ' '.join(parts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--claims', nargs='*', default=None)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    claims = a.claims or list(json.load(open(os.path.join(DATA, 'claim_queries.json'))))
    need_total = have_total = got_total = 0
    for i, cid in enumerate(claims):
        p = os.path.join(DATA, 'records', cid + '.json')
        d = json.load(open(p))
        recs = d['records']
        need = [r['pmid'] for r in recs if not (r.get('abstract') or '').strip()]
        need_total += len(need)
        have_total += len(recs) - len(need)
        if a.dry_run:
            print(f'  {cid}: {len(need)} of {len(recs)} records without abstract text', flush=True)
            continue
        if not need:
            continue
        got = {}
        for j in range(0, len(need), BATCH):
            got.update(abstracts_from(efetch(need[j:j + BATCH])))
            time.sleep(0.4 if os.environ.get('NCBI_API_KEY') else 0.8)
        for r in recs:
            if r['pmid'] in got:
                r['abstract'] = got[r['pmid']]
        got_total += len(got)
        d.pop('abstract_text', None)
        json.dump(d, open(p, 'w'))
        print(f'  {i + 1}/{len(claims)} {cid}: restored {len(got)} of {len(need)}', flush=True)
    print(f'DONE claims {len(claims)} | already present {have_total} | missing {need_total} | restored {got_total}')
    if not a.dry_run and got_total:
        print('Rebuild the parsed effect estimates with: python3 scripts/build_effects_cache.py')


if __name__ == '__main__':
    main()
