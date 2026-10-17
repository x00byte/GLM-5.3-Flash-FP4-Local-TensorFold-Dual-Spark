#!/usr/bin/env python3
"""Download pinned target and drafter snapshots to real directories."""
import argparse,json
from pathlib import Path
from huggingface_hub import snapshot_download
TARGET=('nvidia/GLM-5.3-Flash-FP4-Local','76348ddb2be2e35e39ecfc00e75dc07703f4f3d5')
DRAFT=('incoai/GLM-5.3-Flash-DFlash2','bf582e4eacc1810f76656d1811693ff6c6737d2a')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/models'));p.add_argument('--target-only',action='store_true');p.add_argument('--draft-only',action='store_true');a=p.parse_args()
 if a.target_only and a.draft_only:p.error('Choose only one filter')
 for repo,revision in ([TARGET] if a.target_only else [DRAFT] if a.draft_only else [TARGET,DRAFT]):
  dest=a.root/repo.split('/')[-1]
  snapshot_download(repo_id=repo,revision=revision,local_dir=str(dest),max_workers=8)
  if repo==TARGET[0]:
   cfg=json.loads((dest/'config.json').read_text());assert cfg['quantization_config']['quant_algo']=='MIXED_PRECISION'
   index=json.loads((dest/'model.safetensors.index.json').read_text())['weight_map'];assert len(set(index.values()))==42
   assert all((dest/f).is_file() for f in set(index.values()))
  (dest/'.recipe-snapshot.json').write_text(json.dumps({'repo':repo,'revision':revision},indent=2)+'\n')
  print(repo,revision,'ready')
