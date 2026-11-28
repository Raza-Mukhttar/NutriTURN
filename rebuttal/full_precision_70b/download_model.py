"""Download the official full-precision Llama-3.1-70B-Instruct checkpoint (safetensors only).

    python download_model.py [target_dir]        # default: ./model

About 141 GB. The repository is gated: the Hugging Face token in use needs access to
meta-llama/Llama-3.1-70B-Instruct and gated-repository read scope. The revision is pinned.
"""
import os
import sys

from huggingface_hub import snapshot_download

REPO = 'meta-llama/Llama-3.1-70B-Instruct'
REVISION = '1605565b47bb9346c5515c34102e054115b4f98b'
HERE = os.path.dirname(os.path.abspath(__file__))
target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'model')
path = snapshot_download(REPO, revision=REVISION, local_dir=target, max_workers=8,
                         allow_patterns=['*.safetensors', '*.json'],
                         ignore_patterns=['original/*'])
size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
print(f'DOWNLOADED {REPO}@{REVISION} to {path}: {size / 1e9:.1f} GB', flush=True)
