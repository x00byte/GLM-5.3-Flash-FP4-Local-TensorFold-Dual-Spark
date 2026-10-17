#!/usr/bin/env python3
"""Write an explicit decode/prefill table from a completed recipe sweep."""
import argparse,csv,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);a=p.parse_args()
 assert (a.results/'complete.json').is_file(),'A completed sweep is required'
 info=json.loads((a.results/'complete.json').read_text());workload=info.get('workload','ledger');cap=info.get('decode_output_tokens',64)
 with (a.results/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
 clean=[]
 for row in rows:
  decode=row['phase']=='decode'
  clean.append({'phase':row['phase'],'thinking':row['thinking'],'concurrency':int(row['concurrency']),'repeat':int(row['repeat']),
   'streamed_decode_tps':row['decode_observed_aggregate_tps'] if decode else '',
   'decode_round_tps':row['decode_active_aggregate_tps'] if decode else '',
   'steady_concurrency_decode_round_tps':row['decode_exact_concurrency_tps'] if decode else '',
   'steady_concurrency_rounds':row['decode_exact_concurrency_rounds'] if decode else '',
   'steady_concurrency_tokens':row['decode_exact_concurrency_tokens'] if decode else '',
   'cold_prefill_endpoint_tps':row['prefill_effective_aggregate_tps'] if not decode else '',
   'cold_prefill_work_tps':row['prefill_active_tps'] if not decode else '',
   'prompt_tokens':row['prompt_tokens'],'completion_tokens':row['completion_tokens'],'cached_tokens':row['cached_tokens']})
 with (a.results/'reported-metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=clean[0].keys(),lineterminator="\n");w.writeheader();w.writerows(clean)
 index={(r['phase'],r['thinking'],r['concurrency']):r for r in clean if r['repeat']==0}
 lines=['# GLM-5.3-Flash-FP4-Local TensorFold measurements','','Decode uses fully cached prompts and excludes time to first token. Prefill uses separate uncached requests with a one-token output cap. C2-C8 decode is aggregate. Values are tokens per second.','','| Concurrent requests | Decode off | Decode high | Cold prefill off | Cold prefill high |','|---:|---:|---:|---:|---:|']
 for c in sorted({r['concurrency'] for r in clean}):
  vals=[index[(phase,mode,c)][key] for phase,mode,key in [('decode','off','streamed_decode_tps'),('decode','on','streamed_decode_tps'),('prefill','off','cold_prefill_endpoint_tps'),('prefill','on','cold_prefill_endpoint_tps')]]
  lines.append('| '+str(c)+' | '+' | '.join(f'{float(v):.1f}' for v in vals)+' |')
 lines+=[f'','Workload: '+workload+f'; decode cap: {cap} tokens.','','These are full-burst streamed decode rates after prefix warmup. The separate `steady_concurrency_decode_round_tps` column in [reported-metrics.csv](reported-metrics.csv) counts only engine rounds with exactly C decoding streams; it excludes scheduling gaps and lower-concurrency tails. It must not be labeled as a client full-burst rate. Cached token counts divided by warm-request TTFT are not reported as prefill throughput.','','The complete sweep, individual request timing/usage records and health-counter deltas are retained beside this report. This is a short workload-specific benchmark; the exact workload and decode cap are recorded in `complete.json`.','','## Credits','','Measured using the [TensorFold](https://github.com/ashhart/TensorFold) Spark runtime, [NVIDIA target checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local), and [IncoAI DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).']
 (a.results/'README.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
