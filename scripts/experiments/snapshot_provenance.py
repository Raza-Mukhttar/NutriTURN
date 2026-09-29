"""Post hoc provenance bounds on the original PubMed harvest.

The archived snapshot stores no retrieval timestamp (verified here: no timestamp-like key occurs in
any claim-query, record or annotation file). Two independent observations nevertheless bound when
the harvest ran.

  Lower bound. The highest archived PubMed identifiers entered PubMed on a date that PubMed itself
  still reports in each record's history block (the 'entrez' PubMedPubDate). The harvest cannot
  predate the arrival of a record it contains, so that date is a floor. This argument does not
  assume identifiers are assigned in order; it needs only that these records are in the archive.

  Upper bound. The earliest on-disk write of the 165 archived record files. A copy cannot precede
  what it copies, so this is a ceiling whether the timestamps belong to the harvest or to a later
  duplication of it.

The upper bound is an observation about the authoring filesystem and is therefore recorded here
rather than recomputed: copying the release rewrites file times. Re-running this script outside the
original tree keeps the archived observation and says so in 'upper_bound_source'.

Writes results/guide/snapshot_provenance.json.
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
import os, sys, json, glob, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET

G = _NT
RES = os.path.join(G, 'results', 'guide')
OUT = os.path.join(RES, 'snapshot_provenance.json')
DATA = _os.path.join(_NT, 'data', 'protocol_snapshot')

# The directory whose file times were measured for the upper bound, and the measurement itself.
MTIME_TREE = _os.path.join(_NT, 'data', 'protocol_snapshot', 'records')
ARCHIVED_UPPER = {'utc': '2026-09-10 13:53:35', 'tree': MTIME_TREE, 'files': 165,
                  'note': 'earliest write of the 165 archived record files; two bulk writes, 135 files at 13:53:35 and 30 at 14:58:53'}

TIMESTAMP_KEYS = ('retrieval_date', 'retrieved_at', 'retrieved_utc', 'timestamp', 'timestamp_utc',
                  'fetch_date', 'fetched_date', 'download_date', 'harvested_at', 'date_created',
                  'date_revised', 'entrez_date', 'edat', 'crdt', 'snapshot_timestamp_utc')


def scan_keys():
    """Confirm no timestamp-like key occurs anywhere in the archived snapshot."""
    seen, files = set(), 0

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.lower() in TIMESTAMP_KEYS:
                    seen.add(k)
                walk(v)
        elif isinstance(o, list):
            for v in o[:200]:
                walk(v)

    for sub in ('claim_queries.json', 'records', 'directions'):
        p = os.path.join(DATA, sub)
        for f in ([p] if p.endswith('.json') else sorted(glob.glob(os.path.join(p, '*.json')))):
            files += 1
            try:
                walk(json.load(open(f)))
            except Exception:
                pass
    return {'files_scanned': files, 'timestamp_keys_found': sorted(seen)}


def top_pmids(k=3):
    ids = set()
    for f in glob.glob(os.path.join(DATA, 'records', '*.json')):
        for r in json.load(open(f))['records']:
            try:
                ids.add(int(r['pmid']))
            except (KeyError, TypeError, ValueError):
                pass
    return sorted(ids)[-k:][::-1], len(ids)


def entrez_dates(pmids):
    q = {'db': 'pubmed', 'id': ','.join(str(p) for p in pmids), 'retmode': 'xml',
         'tool': 'nutrimature_release', 'email': os.environ.get('NCBI_EMAIL', 'anonymous@example.org')}
    if os.environ.get('NCBI_API_KEY'):
        q['api_key'] = os.environ['NCBI_API_KEY']
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?' + urllib.parse.urlencode(q)
    root = ET.fromstring(urllib.request.urlopen(url, timeout=90).read())
    out = {}
    for a in root.iter('PubmedArticle'):
        pid = a.findtext('.//PMID')
        e = a.find('.//PubmedData/History/PubMedPubDate[@PubStatus="entrez"]')
        if pid and e is not None:
            out[pid] = '%s-%02d-%02d' % (e.findtext('Year'), int(e.findtext('Month')), int(e.findtext('Day')))
    return out


def main():
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    o = {'archive_has_retrieval_timestamp': False, 'key_scan': scan_keys()}
    o['archive_has_retrieval_timestamp'] = bool(o['key_scan']['timestamp_keys_found'])

    pmids, n_ids = top_pmids()
    o['distinct_archived_identifiers'] = n_ids
    o['highest_archived_identifiers'] = pmids
    try:
        d = entrez_dates(pmids)
        o['entrez_dates'] = d
        o['lower_bound_source'] = 'PubMed EFetch, queried at run time'
    except Exception as e:                                   # offline: keep the archived observation
        d = prev.get('entrez_dates', {})
        o['entrez_dates'] = d
        o['lower_bound_source'] = f'archived observation retained; live lookup failed ({type(e).__name__})'
    o['lower_bound_utc'] = min(d.values()) if d else None

    if os.path.isdir(MTIME_TREE):
        fs = glob.glob(os.path.join(MTIME_TREE, '*.json'))
        t = min(os.path.getmtime(f) for f in fs)
        o['upper_bound_utc'] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t))
        o['upper_bound_detail'] = dict(ARCHIVED_UPPER, files=len(fs))
        o['upper_bound_source'] = 'measured on the authoring filesystem'
    else:
        o['upper_bound_utc'] = ARCHIVED_UPPER['utc']
        o['upper_bound_detail'] = ARCHIVED_UPPER
        o['upper_bound_source'] = 'archived observation; the measured tree is not present here (file times do not survive copying)'

    o['window'] = f"{o['lower_bound_utc']} to {o['upper_bound_utc']} UTC"
    o['exact_timestamp_recorded'] = False
    o['regenerable'] = False
    o['note'] = ('The bounds constrain when the harvest ran. They do not make the archived identifier '
                 'sets regenerable: re-running the stored queries returns a different identifier set '
                 '(see results/snapshot_*/ and the dataset card).')
    os.makedirs(RES, exist_ok=True)
    json.dump(o, open(OUT, 'w'), indent=1)
    print('timestamp keys in archive:', o['key_scan']['timestamp_keys_found'] or 'none',
          f"({o['key_scan']['files_scanned']} files scanned)")
    print('highest archived identifiers:', pmids, '->', o['entrez_dates'])
    print('lower bound', o['lower_bound_utc'], '|', o['lower_bound_source'])
    print('upper bound', o['upper_bound_utc'], '|', o['upper_bound_source'])
    print('WINDOW', o['window'])
    print('wrote', OUT)


if __name__ == '__main__':
    main()
