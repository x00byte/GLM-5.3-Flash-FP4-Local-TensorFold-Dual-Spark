"""Quick C1/C4/C8 decode screen using prose, code and repetitive edits."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def validate(folder):
    rows=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
    timed=[r for r in rows if not r.get('warmup')]
    assert len(timed)==13, 'Expected 13 timed requests'
    assert all(r['completion_tokens']==128 for r in timed), 'Incomplete output'
    assert all(r['backend']['tensorfold']['token_sha'] for r in timed), 'Missing token hashes'
    for row in csv.DictReader((folder/'summary.csv').open()):
        assert int(row['prefill_active_tokens'])==0, 'Timed decode included prefill'
        h=json.loads((folder/f"c{row['concurrency']}-r0-health.json").read_text())
        assert not h['before']['busy'] and not h['after']['busy'], 'Outside traffic'
        assert h['after']['requests_total']-h['before']['requests_total']==int(row['concurrency']), 'Outside requests'
    return {(r['concurrency'],r['stream'],r['prompt_sha256']):r['backend']['tensorfold']['token_sha'] for r in timed}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--label',required=True)
    ap.add_argument('--base',default='http://127.0.0.1:8888')
    ap.add_argument('--policy',help='Optional request-level draft policy')
    ap.add_argument('--no-copy',action='store_true')
    ap.add_argument('--reference',type=Path,help='Compare token hashes with a previous screen')
    a=ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):ap.error('Use an empty output directory')
    checks={}
    for workload in ('general','code','edit'):
        folder=a.out/workload
        cmd=[sys.executable,str(Path(__file__).with_name('client.py')),
                        '--model','nvidia/GLM-5.3-Flash-FP4-Local',
                        '--base',a.base,'--label',a.label,'--workload',workload,
                        '--warm-prefix','--concurrency','1,4,8','--repeats','1',
                        '--input-lines','32','--output-tokens','128','--out',str(folder)]
        if a.policy:cmd+=['--policy',a.policy]
        if a.no_copy:cmd+=['--no-copy']
        subprocess.run(cmd,check=True)
        hashes=validate(folder)
        if a.reference:
            assert hashes==validate(a.reference/workload), f'{workload}: output hashes differ'
        checks[workload]={'timed_requests':len(hashes),'zero_timed_prefill':True,
                          'token_hashes_match_reference':True if a.reference else None}
    (a.out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')


if __name__=='__main__':main()
