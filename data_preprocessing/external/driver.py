"""Launch the External re-annotation shards as GPUs free up (E2 is still finishing).

One shard per GPU, never targeting a GPU that already hosts one of our processes, waiting for each
launch to establish before considering the next device. 5 shards over GPUs 0-4; a shard that cannot
start yet is retried. Output is append-only and keyed by (claim_id, pmid), so retries resume.
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

import os, subprocess, sys, time

PY = sys.executable
SCRIPT = os.path.join(_NT, 'data_annotation', 'external', 'annotate_external.py')
NSHARDS = 5
NEED_MIB = 30000
CANDIDATES = [0, 1, 3, 4, 2]


def free_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    return {int(l.split(', ')[0]): int(l.split(', ')[2]) - int(l.split(', ')[1])
            for l in out.strip().splitlines()}


def mine():
    out = subprocess.run(['ps', '-eo', 'args', '--no-headers'], capture_output=True, text=True).stdout
    gpus, shards = set(), set()
    for line in out.splitlines():
        if 'annotate_external.py' not in line or '--gpu' not in line or 'driver' in line:
            continue
        p = line.split()
        try:
            gpus.add(int(p[p.index('--gpu') + 1])); shards.add(int(p[p.index('--shard') + 1]))
        except (ValueError, IndexError):
            pass
    return gpus, shards


def busy_other():
    """GPUs occupied by our OTHER experiment (E2's 70B chunks)"""
    out = subprocess.run(['ps', '-eo', 'args', '--no-headers'], capture_output=True, text=True).stdout
    g = set()
    for line in out.splitlines():
        if 'annotate_70b.py' in line and '--gpu' in line:
            p = line.split()
            try:
                g.add(int(p[p.index('--gpu') + 1]))
            except (ValueError, IndexError):
                pass
    return g


def lines(sh):
    p = os.path.join(HERE, 'data', 'annot_second', f'shard{sh}of{NSHARDS}.jsonl')
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def main():
    target = 166548
    while True:
        gpus, running = mine()
        tot = sum(lines(s) for s in range(NSHARDS))
        if tot >= target and not running:
            print(f'{time.strftime("%FT%TZ", time.gmtime())} complete: {tot:,} rows', flush=True)
            return 0
        pending = [s for s in range(NSHARDS) if s not in running]
        busy = gpus | busy_other()
        fm = free_mib()
        placed = 0
        for s in pending:
            cand = [g for g in CANDIDATES if g not in busy and fm.get(g, 0) >= NEED_MIB]
            if not cand:
                break
            g = cand[0]
            log = os.path.join(HERE, 'logs', f'shard{s}.log')
            print(f'{time.strftime("%FT%TZ", time.gmtime())} launching shard {s} on GPU {g} '
                  f'({fm[g]} MiB free)', flush=True)
            with open(log, 'a') as lf:
                subprocess.Popen([PY, SCRIPT, '--gpu', str(g), '--shard', str(s),
                                  '--nshards', str(NSHARDS)], stdout=lf,
                                 stderr=subprocess.STDOUT, env=dict(
                                     os.environ, PYTHONDONTWRITEBYTECODE='1',
                                     PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'),
                                 start_new_session=True)
            busy.add(g); placed += 1
            time.sleep(200)
            _, now = mine()
            print(f'   shard {s} {"established" if s in now else "did NOT survive warm-up"} on GPU {g}',
                  flush=True)
            if s not in now:
                busy.discard(g)
            fm = free_mib()
        print(f'{time.strftime("%FT%TZ", time.gmtime())} rows {tot:,}/{target:,}  '
              f'running shards {sorted(running)}  our-70B GPUs {sorted(busy_other())}', flush=True)
        if not placed:
            time.sleep(120)


if __name__ == '__main__':
    sys.exit(main())
