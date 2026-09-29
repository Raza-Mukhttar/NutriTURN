"""Write MANIFEST.json: sha256 and size of every shipped file, plus seeds and library versions.

  python scripts/build_file_manifest.py
"""
import os, sys, json, hashlib, platform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {'MANIFEST.json'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def versions():
    out = {'python': platform.python_version(), 'platform': platform.platform()}
    for m in ('numpy', 'scipy', 'sklearn', 'pandas', 'statsmodels', 'xgboost', 'matplotlib'):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = 'not installed'
    return out


def main():
    files, total = {}, 0
    for r, ds, fs in os.walk(ROOT):
        ds[:] = [d for d in ds if d != '__pycache__']
        for f in sorted(fs):
            p = os.path.join(r, f)
            rel = os.path.relpath(p, ROOT)
            if rel in SKIP:
                continue
            n = os.path.getsize(p)
            files[rel] = {'sha256': sha(p), 'bytes': n}
            total += n
    by = {}
    for rel, v in files.items():
        by.setdefault(rel.split(os.sep)[0], [0, 0])
        by[rel.split(os.sep)[0]][0] += 1
        by[rel.split(os.sep)[0]][1] += v['bytes']
    man = {
        'release': 'NutriTURN',
        'files': len(files),
        'bytes': total,
        'by_directory': {k: {'files': v[0], 'bytes': v[1]} for k, v in sorted(by.items())},
        'seeds': {'shared_seed': 3, 'percentile_bootstrap_resamples': 2000,
                  'wild_cluster_draws': 9999, 'null_control_draws': 1000,
                  'null_D_seeds': '130000-130999 under PYTHONHASHSEED=0'},
        'required_environment': {'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1'},
        'library_versions': versions(),
        'withheld': ['abstract text (records ship identifiers, years, titles, publication types)',
                     'external corpora document inputs (unit-level label counts only)',
                     'model weights (downloaded from the Hugging Face Hub at pinned revisions)'],
        'sha256_of_file_list': '',
    }
    man['sha256_of_file_list'] = hashlib.sha256(
        '\n'.join(f'{k} {v["sha256"]}' for k, v in sorted(files.items())).encode()).hexdigest()
    man['file_hashes'] = dict(sorted(files.items()))
    out = os.path.join(ROOT, 'MANIFEST.json')
    json.dump(man, open(out, 'w'), indent=1)
    print(f'{len(files)} files, {total/1e6:.1f} MB')
    for k, v in man['by_directory'].items():
        print(f'   {k:20s} {v["files"]:6d} files {v["bytes"]/1e6:9.1f} MB')
    print(f'\nfile-list digest {man["sha256_of_file_list"][:32]}')
    print('written', out)


if __name__ == '__main__':
    main()
