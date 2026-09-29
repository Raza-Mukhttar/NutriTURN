"""Issue 7: a new, timestamped PubMed retrieval of every source query (ESearch only: counts, query
translation and the full PMID list; abstracts are not re-fetched, so the extraction and benchmark are
NOT rebuilt on the new snapshot). Records one UTC timestamp per query, the returned PMID list, its SHA-256
over the sorted normalised identifiers, and a version-to-version comparison with the stored snapshot.
External-1/2 query strings were never released, so they cannot be re-run and are marked non-regenerable.

  python reretrieve_pubmed.py            -> results/snapshot_<UTCdate>/{queries.jsonl, comparison.csv, summary.json}
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
import json, os, sys, time, hashlib, csv, datetime, urllib.parse, urllib.request, xml.etree.ElementTree as ET
HERE = _os.path.join(_NT, 'scripts'); G = _NT
SNAP = _os.path.join(_NT, 'data', 'protocol_snapshot')
BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'


def esearch(term, retmax):
    url = BASE + '?' + urllib.parse.urlencode({'db': 'pubmed', 'term': term, 'retmax': retmax, 'retmode': 'xml', 'tool': 'nutrimature_snapshot', 'email': 'anonymous@example.org'})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r: return r.read()
        except Exception as e:
            time.sleep(2 + attempt * 3)
    raise RuntimeError('esearch failed: ' + term[:60])


def main():
    Q = json.load(open(f'{SNAP}/claim_queries.json'))
    stamp = datetime.datetime.now(datetime.timezone.utc); out = f'{G}/results/snapshot_{stamp.strftime("%Y%m%dT%H%M%SZ")}'; os.makedirs(out, exist_ok=True)
    fh = open(f'{out}/queries.jsonl', 'w'); comp = []
    for i, (cid, q) in enumerate(Q.items()):
        term = q['retrieval_query']; t = datetime.datetime.now(datetime.timezone.utc).isoformat()
        xml = esearch(term, 200000); root = ET.fromstring(xml)
        count = int(root.findtext('Count') or 0); ids = sorted(e.text for e in root.iter('Id'))
        trans = root.findtext('QueryTranslation') or ''
        stored = sorted(r['pmid'] for r in json.load(open(f'{SNAP}/records/{cid}.json'))['records'])
        sha_new = hashlib.sha256('\n'.join(ids).encode()).hexdigest(); sha_old = hashlib.sha256('\n'.join(stored).encode()).hexdigest()
        s_new, s_old = set(ids), set(stored)
        row = {'claim_id': cid, 'timestamp_utc': t, 'pubmed_count': count, 'returned_ids': len(ids), 'stored_ids': len(stored), 'in_both': len(s_new & s_old),
               'new_only': len(s_new - s_old), 'stored_only': len(s_old - s_new), 'jaccard': round(len(s_new & s_old) / max(len(s_new | s_old), 1), 4),
               'sha256_new_ids': sha_new, 'sha256_stored_ids': sha_old}
        comp.append(row)
        fh.write(json.dumps({'claim_id': cid, 'query': term, 'timestamp_utc': t, 'pubmed_count': count, 'query_translation': trans, 'pmids': ids, 'sha256_ids': sha_new}) + '\n'); fh.flush()
        print(f"  {i + 1}/{len(Q)} {cid}: count {count} returned {len(ids)} stored {len(stored)} both {row['in_both']} new-only {row['new_only']} stored-only {row['stored_only']}", flush=True)
        time.sleep(0.4)
    fh.close()
    with open(f'{out}/comparison.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0])); w.writeheader(); w.writerows(comp)
    tot = {'snapshot_timestamp_utc': stamp.isoformat(), 'queries': len(comp), 'stored_ids_total': sum(r['stored_ids'] for r in comp), 'returned_ids_total': sum(r['returned_ids'] for r in comp),
           'in_both_total': sum(r['in_both'] for r in comp), 'new_only_total': sum(r['new_only'] for r in comp), 'stored_only_total': sum(r['stored_only'] for r in comp),
           'note': 'ESearch-only re-retrieval at one UTC timestamp; abstracts not re-fetched; benchmark not rebuilt on this snapshot; external corpora queries unavailable (non-regenerable)'}
    json.dump(tot, open(f'{out}/summary.json', 'w'), indent=1); print('DONE', tot, flush=True)


if __name__ == '__main__':
    main()
