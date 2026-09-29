"""E18: do the capped claims' archived streams start later than today's retrieval?

App. C.3 currently argues from identifier position, a proxy for accession date. This runs the
direct test. For each of the 48 claims holding 780-800 archived records, it fetches the publication
year of every identifier today's retrieval returns, and compares that year distribution with the
archived one.

If the cap kept the most recently accessioned records, the archived stream's earliest year will sit
later than the provider's, and the archived share of the provider's pre-1990 (or pre-2000) records
will be far below its overall share. If the cap was arbitrary with respect to date, the two
distributions will agree.

  python analyses/nfx_capyears.py fetch     # ESummary years for the re-retrieved identifiers
  python analyses/nfx_capyears.py compare   # the comparison, no network

Fetched years are cached in data/reretrieved_years.json so the comparison is reproducible offline.
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
import os, sys, json, glob, csv, time, urllib.parse, urllib.request, argparse
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

CACHE = os.path.join(N.DATA, 'reretrieved_years.json')
ESUM = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi'
BATCH = 200


def capped_claims():
    rows = list(csv.DictReader(open(os.path.join(N.RESULTS, 'E15_cap_census.csv'))))
    return [r['claim_id'] for r in rows if 780 <= int(r['archived']) <= 800]


def reretrieved():
    adir = sorted(glob.glob(os.path.join(N.GPTPRO, 'results', 'snapshot_*')))[-1]
    out = {}
    for line in open(os.path.join(adir, 'queries.jsonl')):
        d = json.loads(line)
        out[d['claim_id']] = [str(x) for x in d.get('pmids', [])]
    return out


def cmd_fetch(a):
    want = set()
    R = reretrieved()
    for c in capped_claims():
        want |= set(R.get(c, []))
    have = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = sorted(want - set(have))
    print(f'{len(want):,} identifiers across {len(capped_claims())} capped claims; '
          f'{len(have):,} cached, {len(todo):,} to fetch', flush=True)
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        ids = todo[i:i + BATCH]
        q = {'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml',
             'tool': 'nutrimature_capyears', 'email': os.environ.get('NCBI_EMAIL', 'anonymous@example.org')}
        if os.environ.get('NCBI_API_KEY'):
            q['api_key'] = os.environ['NCBI_API_KEY']
        # NCBI rejects long GET URLs, so an identifier list of this size must go by POST
        data = urllib.parse.urlencode(q).encode()
        root = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(ESUM, data=data)
                with urllib.request.urlopen(req, timeout=120) as r:
                    root = ET.fromstring(r.read())
                break
            except Exception as e:
                if attempt == 3:
                    print(f'  giving up on batch {i}: {type(e).__name__} '
                          f'{getattr(e, "code", "")} {str(e)[:120]}', flush=True)
                else:
                    time.sleep(3 + 3 * attempt)
        if root is None:
            continue
        for doc in root.iter('DocSum'):
            pid = doc.findtext('Id')
            pd = None
            for it in doc.iter('Item'):
                if it.get('Name') in ('PubDate', 'EPubDate') and it.text and pd is None:
                    pd = it.text
            if pid and pd:
                y = ''.join(ch for ch in pd[:4] if ch.isdigit())
                if len(y) == 4:
                    have[pid] = int(y)
        if (i // BATCH) % 20 == 0:
            json.dump(have, open(CACHE, 'w'))
            el = time.time() - t0
            print(f'  {i+len(ids):,}/{len(todo):,}  {(i+len(ids))/max(el,1e-9):.0f} ids/s', flush=True)
        time.sleep(0.15 if os.environ.get('NCBI_API_KEY') else 0.35)
    json.dump(have, open(CACHE, 'w'))
    print(f'cached {len(have):,} identifier years -> {CACHE}', flush=True)


def cmd_compare(a):
    if not os.path.exists(CACHE):
        raise SystemExit('run "fetch" first')
    years = {k: int(v) for k, v in json.load(open(CACHE)).items()}
    R = reretrieved()
    arch = {}
    for p in sorted(glob.glob(os.path.join(N.SNAPD, 'records', '*.json'))):
        d = json.load(open(p))
        arch[d['claim_id']] = {str(r['pmid']): int(r['year']) for r in d['records'] if r.get('year')}
    # Refuse to run on a partial cache. The fetch queue is sorted, so an incomplete cache is
    # biased toward older identifiers and would manufacture an "archive starts later" result.
    R_all = {c: R.get(c, []) for c in capped_claims()}
    tot = sum(len(v) for v in R_all.values())
    cov = sum(1 for v in R_all.values() for p in v if p in years)
    print(f'year coverage across the capped claims: {cov:,} of {tot:,} '
          f'({100*cov/max(tot,1):.1f}%)')
    if cov / max(tot, 1) < 0.97:
        raise SystemExit('year cache is incomplete; rerun "fetch" before comparing '
                         '(a partial cache is biased toward older identifiers)')
    print()
    print('E18 publication years of the capped claims: archive versus today\'s retrieval\n')
    rows = []
    for cid in capped_claims():
        ids = R.get(cid, [])
        new = [p for p in ids if p in years]
        if len(ids) < 100 or len(new) / max(len(ids), 1) < 0.97:
            continue
        ny = np.array([years[p] for p in new])
        ay = np.array(list(arch[cid].values()))
        # share of the provider's oldest records that the archive holds
        cut = int(np.percentile(ny, 25))
        prov_old = (ny <= cut).sum()
        arch_old = sum(1 for p in new if p in arch[cid] and years[p] <= cut)
        rows.append(dict(claim_id=cid, archived=len(ay), provider=len(ny),
                         archived_min_year=int(ay.min()), provider_min_year=int(ny.min()),
                         archived_median_year=int(np.median(ay)), provider_median_year=int(np.median(ny)),
                         min_year_gap=int(ay.min() - ny.min()),
                         median_year_gap=int(np.median(ay) - np.median(ny)),
                         provider_oldest_quartile=int(prov_old),
                         archived_share_of_oldest_quartile=round(arch_old / max(prov_old, 1), 4),
                         archived_overall_share=round(len(ay) / max(len(ny), 1), 4)))
    N.save_csv(rows, 'E18_cap_years.csv')
    mg = np.array([r['min_year_gap'] for r in rows])
    dg = np.array([r['median_year_gap'] for r in rows])
    ratio = np.array([r['archived_share_of_oldest_quartile'] / max(r['archived_overall_share'], 1e-9)
                      for r in rows])
    print(f'{len(rows)} capped claims with at least 100 dated identifiers today\n')
    print(f'   earliest archived year minus earliest provider year: median {int(np.median(mg))}, '
          f'range [{mg.min()}, {mg.max()}]')
    print(f'      claims whose archive starts LATER than the provider: {(mg > 0).sum()} of {len(rows)}')
    print(f'   median archived year minus median provider year     : median {int(np.median(dg))}, '
          f'range [{dg.min()}, {dg.max()}]')
    print(f'      claims whose archive is centred LATER            : {(dg > 0).sum()} of {len(rows)}')
    print()
    print('   coverage of the provider\'s oldest quartile, relative to the claim\'s overall coverage')
    print('   (1.0 means the oldest records are kept at the same rate as the rest; below 1 means')
    print('    the oldest are under-represented, which is what date-ordered truncation produces)')
    print(f'      median {np.median(ratio):.3f}   range [{ratio.min():.3f}, {ratio.max():.3f}]')
    print(f'      claims below 0.5: {(ratio < 0.5).sum()} of {len(rows)}')
    print(f'      claims below 0.9: {(ratio < 0.9).sum()} of {len(rows)}')
    worst = sorted(rows, key=lambda r: r['archived_share_of_oldest_quartile'] / max(r['archived_overall_share'], 1e-9))[:5]
    print('\n   most date-skewed claims:')
    for r in worst:
        print(f"      {r['claim_id']:42s} archive {r['archived']:4d}/{r['provider']:5d}  "
              f"starts {r['archived_min_year']} vs {r['provider_min_year']}  "
              f"oldest-quartile coverage ratio "
              f"{r['archived_share_of_oldest_quartile']/max(r['archived_overall_share'],1e-9):.3f}")
    out = dict(claims=len(rows),
               min_year_gap_median=int(np.median(mg)),
               claims_starting_later=int((mg > 0).sum()),
               median_year_gap_median=int(np.median(dg)),
               oldest_quartile_ratio_median=float(np.median(ratio)),
               claims_ratio_below_half=int((ratio < 0.5).sum()),
               rows=rows)
    print('\nwritten', N.save_json(out, 'E18_cap_years.json'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('fetch'); sub.add_parser('compare')
    a = ap.parse_args()
    {'fetch': cmd_fetch, 'compare': cmd_compare}[a.cmd](a)
