#!/usr/bin/env python3
"""Check pinned snapshot files. --full additionally rehashes all weight bytes."""
import argparse,hashlib,json
from pathlib import Path
def verify(root,manifest,full=False):
 root=root.resolve();m=json.loads(manifest.read_text());checked=0
 for entry in m['files']:
  relative=Path(entry['path'])
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid manifest path')
  path=root/relative
  assert path.is_file() and path.stat().st_size==entry['bytes'],f'Missing or incomplete file: {relative}'
  if not full and path.suffix=='.safetensors':continue
  h=hashlib.sha256() if entry['algorithm']=='sha256' else hashlib.sha1()
  if entry['algorithm']=='git-blob-sha1':h.update(b'blob '+str(entry['bytes']).encode()+b'\0')
  with path.open('rb') as f:
   while chunk:=f.read(16<<20):h.update(chunk)
  assert h.hexdigest()==entry['hash'],f'Hash mismatch: {relative}'
  checked+=1
 print(f"{m['repo']}@{m['revision']}: {len(m['files'])} file sizes and {checked} hashes checked; full={full}")
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('model',type=Path);p.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parent.parent/'target-manifest.json');p.add_argument('--full',action='store_true');a=p.parse_args();verify(a.model,a.manifest,a.full)
