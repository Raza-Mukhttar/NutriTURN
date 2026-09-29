"""Resolve the eight pivotal-source citations to real PubMed records.

The dates used in the reversal tabulation must be externally verifiable, so each source is
looked up against PubMed rather than recalled. The script prints the returned title, journal and
year so every match can be checked by eye before it is used; nothing is asserted from memory.
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
import os, sys, json, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nfx_common as N

ESEARCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
EFETCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
TOOL = {'tool': 'nutrimature_sources', 'email': os.environ.get('NCBI_EMAIL', 'anonymous@example.org')}

QUERIES = [
    # (label, role, search phrase, expected publication year, words the returned title must contain)
    ('WHI 2002', 'hormone replacement -> CVD',
     'Risks and benefits of estrogen plus progestin in healthy postmenopausal women '
     'principal results Women Health Initiative randomized controlled trial', 2002,
     ['estrogen', 'progestin']),
    ('ATBC 1994', 'beta-carotene -> cancer',
     'effect of vitamin E and beta carotene on the incidence of lung cancer and other cancers '
     'in male smokers', 1994, ['beta carotene', 'lung cancer']),
    ('CARET 1996', 'beta-carotene -> cancer',
     'Effects of a combination of beta carotene and vitamin A on lung cancer and '
     'cardiovascular disease', 1996, ['beta carotene', 'lung cancer']),
    ('HOPE 2000', 'vitamin E -> CVD',
     'Vitamin E supplementation and cardiovascular events in high-risk patients '
     'Heart Outcomes Prevention Evaluation', 2000, ['vitamin e', 'cardiovascular']),
    ('HPS 2002', 'vitamin E -> CVD',
     'MRC/BHF Heart Protection Study of antioxidant vitamin supplementation in 20536 '
     'high-risk individuals randomised placebo-controlled trial', 2002,
     ['heart protection study', 'antioxidant']),
    ('SELECT 2009', 'selenium -> prostate cancer',
     'Effect of selenium and vitamin E on risk of prostate cancer and other cancers '
     'Selenium and Vitamin E Cancer Prevention Trial SELECT', 2009,
     ['selenium', 'prostate cancer']),
    ('Cole 2007', 'folic acid -> colorectal',
     'Folic acid for the prevention of colorectal adenomas a randomized clinical trial',
     2007, ['folic acid', 'adenomas']),
    ('AIM-HIGH 2011', 'niacin -> CVD',
     'Niacin in patients with low HDL cholesterol levels receiving intensive statin therapy',
     2011, ['niacin', 'hdl']),
    ('HPS2-THRIVE 2014', 'niacin -> CVD',
     'Effects of extended-release niacin with laropiprant in high-risk patients', 2014,
     ['niacin', 'laropiprant']),
    ('Prasad 2013', 'source of the pivotal-date list',
     'A decade of reversal an analysis of 146 contradicted medical practices', 2013,
     ['decade of reversal']),
]


def esearch(term):
    q = dict(TOOL, db='pubmed', term=term, retmax=10, retmode='json')
    with urllib.request.urlopen(ESEARCH + '?' + urllib.parse.urlencode(q), timeout=60) as r:
        return json.loads(r.read())['esearchresult'].get('idlist', [])


def efetch(pmids):
    q = dict(TOOL, db='pubmed', id=','.join(pmids), retmode='xml')
    with urllib.request.urlopen(EFETCH + '?' + urllib.parse.urlencode(q), timeout=90) as r:
        root = ET.fromstring(r.read())
    out = []
    for a in root.iter('PubmedArticle'):
        out.append(dict(pmid=a.findtext('.//PMID'),
                        title=(a.findtext('.//ArticleTitle') or '').strip(),
                        journal=(a.findtext('.//Journal/ISOAbbreviation') or '').strip(),
                        year=(a.findtext('.//JournalIssue/PubDate/Year') or
                              a.findtext('.//JournalIssue/PubDate/MedlineDate') or '')[:4]))
    return out


def main():
    out = []
    print('Pivotal sources resolved against PubMed (verify each title before use)\n')
    for label, role, term, want_year, want_words in QUERIES:
        try:
            ids = esearch(term)
            recs = efetch(ids) if ids else []
        except Exception as e:
            print(f'{label:18s} lookup failed: {type(e).__name__}'); time.sleep(1); continue
        ok = [r for r in recs
              if str(r['year']) == str(want_year)
              and all(w in r['title'].lower() for w in want_words)]
        if not ok:
            print(f'{label:18s} NO VERIFIED MATCH  (year must be {want_year} and the title must '
                  f'contain {want_words})')
            for r in recs[:2]:
                print(f"{'':18s}   rejected: {r['year']} PMID {r['pmid']} {r['title'][:80]}")
            out.append(dict(label=label, role=role, pmid=None, verified=False))
            time.sleep(0.4); continue
        r = ok[0]
        out.append(dict(label=label, role=role, pmid=r['pmid'], title=r['title'],
                        journal=r['journal'], year=r['year'], query=term, verified=True))
        print(f"{label:18s} PMID {r['pmid']:>9s}  {r['year']}  {r['journal']}")
        print(f"{'':18s} {r['title'][:110]}")
        time.sleep(0.4)
    print('\nwritten', N.save_json({'sources': out}, 'R4c_pivotal_sources.json'))


if __name__ == '__main__':
    main()
