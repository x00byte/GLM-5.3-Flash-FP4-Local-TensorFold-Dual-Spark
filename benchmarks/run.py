#!/usr/bin/env python3
"""TensorFold-only C1-C8 warm-prefix decode and separate cold-prefill sweep."""
import argparse,csv,json,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
MODEL='nvidia/GLM-5.3-Flash-FP4-Local'
def get(base,route):
 with urllib.request.urlopen(base.rstrip('/')+route,timeout=15) as r:return json.load(r)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',default='http://127.0.0.1:8888');p.add_argument('--out',type=Path,default=ROOT/'local-results');p.add_argument('--concurrency',default='1,2,3,4,5,6,7,8');p.add_argument('--repeats',type=int,default=1);a=p.parse_args()
 cs=[int(c) for c in a.concurrency.split(',')]
 if not cs or len(set(cs))!=len(cs) or any(c<1 or c>8 for c in cs) or a.repeats<1:p.error('Distinct concurrency values 1-8 and positive repeats required')
 assert MODEL in {m['id'] for m in get(a.base,'/v1/models')['data']}
 health=get(a.base,'/health');assert health.get('backend')=='tensorfold' and health.get('ok') and not health.get('busy'),health
 assert health.get('decode_active') is not None,'This recipe requires the active-decode counter patch'
 a.out.mkdir(parents=True,exist_ok=True);rows=[];records=[]
 before=get(a.base,'/health');(a.out/'health-before.json').write_text(json.dumps(before,indent=2))
 for phase,cap in (('decode',64),('prefill',1)):
  for mode in ('off','on'):
   case=f'{phase}-thinking-{mode}';dest=a.out/case
   cmd=[sys.executable,'-u',str(ROOT/'benchmarks/client.py'),'--model',MODEL,'--label','glm53-fp4-local-tensorfold','--base',a.base,'--policy','fnc7:0.3','--thinking',mode,'--concurrency',a.concurrency,'--repeats',str(a.repeats),'--input-lines','27','--output-tokens',str(cap),'--skip-warmup','--out',str(dest)]
   if phase=='decode':cmd+=['--warm-prefix']
   print('Running',case,flush=True)
   with (a.out/(case+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
   rec=[json.loads(line) for line in (dest/'requests.jsonl').read_text().splitlines()]
   assert len(rec)==sum(cs)*a.repeats and all(r['completion_tokens']==cap and r['finish_reason']=='length' for r in rec)
   if mode=='off':assert all(r['reasoning_chunks']==0 for r in rec)
   elif phase=='decode':assert all(r['reasoning_chunks']>0 for r in rec)
   records.extend(rec)
   with (dest/'summary.csv').open() as f:
    for item in csv.DictReader(f):
     if phase=='decode':
      assert int(item['prefill_active_tokens'])==0,'Decode request did uncached prefill'
     else:
      assert int(item['cached_tokens'])==0 and int(item['prefill_active_tokens'])==int(item['prompt_tokens'])
     rows.append({'phase':phase,**item})
 after=get(a.base,'/health');(a.out/'health-after.json').write_text(json.dumps(after,indent=2))
 assert after['requests_total']-before['requests_total']==sum(cs)*a.repeats*6,'Other requests overlapped this benchmark'
 with (a.out/'metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n");w.writeheader();w.writerows(rows)
 (a.out/'complete.json').write_text(json.dumps({'model':MODEL,'backend':'TensorFold','target_revision':'76348ddb2be2e35e39ecfc00e75dc07703f4f3d5','policy':'fnc7:0.3','timed_requests':len(records),'prefix_seeds':sum(cs)*a.repeats*2,'completed':time.time(),'repeats':a.repeats,'decode':'warm-prefix; actual cache coverage and zero prefill work verified; full-burst client and active-round rates retained','prefill':'cold, one output token, no prefix-cache reuse'},indent=2)+'\n')
 print('All workload, cache and request-accounting checks passed.',flush=True)
if __name__=='__main__':main()
