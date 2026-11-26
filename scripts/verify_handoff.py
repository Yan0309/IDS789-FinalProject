"""Verify portable data/artifact integrity; --create records a completed handoff."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.full_population import data_bytes, MAX_BYTES

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def run(create=False, modeling_only=False):
    manifest = ROOT/'data/reference/handoff_integrity.json'
    if create:
        if modeling_only: raise ValueError('--modeling-only is a verification option')
        files = []
        for directory in ['data/raw_full','data/processed/full_population','data/modeling','data/reference','src','scripts','notebooks','tests']:
            files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p != manifest)
        files.extend([ROOT/'README.md', ROOT/'requirements.txt', ROOT/'.gitignore'])
        entries = [{'path':str(p.relative_to(ROOT)), 'bytes':p.stat().st_size, 'sha256':digest(p)} for p in sorted(files)]
        manifest.write_text(json.dumps({'files':entries},indent=2)+'\n')
        print(f'Recorded {len(entries)} file hashes.')
    entries = json.loads(manifest.read_text())['files']
    if modeling_only: entries = [x for x in entries if x['path'].startswith('data/modeling/')]
    failures = []
    for item in entries:
        path = ROOT/item['path']
        if not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            failures.append(item['path'])
    if failures: raise RuntimeError('Missing or changed files: '+', '.join(failures))
    contract = json.loads((ROOT/'data/modeling/modeling_contract.json').read_text())
    assert all(contract['invariants'].values()), 'Failed modeling invariants'
    size = data_bytes(ROOT)
    assert size <= MAX_BYTES, 'Project data exceeds 10 GB'
    print(f'PASS: {len(entries)} files verified; {contract["pd_ready_rows"]:,} labeled rows; {size/1e9:.3f} GB total data.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', action='store_true')
    parser.add_argument('--modeling-only', action='store_true')
    args = parser.parse_args()
    run(args.create,args.modeling_only)
