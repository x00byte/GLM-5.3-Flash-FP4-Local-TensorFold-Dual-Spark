#!/usr/bin/env python3
"""Compare request-level DFlash2 policies without changing the running model."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

MODEL = 'nvidia/GLM-5.3-Flash-FP4-Local'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8888')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--policies', default='fnc7:0.3,fnc7:0.1,fnc7:0.5')
    parser.add_argument('--concurrency', default='1,4,8')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--tokens', type=int, default=64)
    args = parser.parse_args()
    levels = [int(c) for c in args.concurrency.split(',')]
    if (args.repeats < 1 or args.tokens < 2 or not levels
            or len(set(levels)) != len(levels) or any(c < 1 or c > 8 for c in levels)):
        parser.error('Use distinct concurrency values 1–8, positive repeats, and at least two output tokens')
    policies = args.policies.split(',')
    if len(set(policies)) != len(policies) or any(not p for p in policies):
        parser.error('Use distinct nonempty policies')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Choose an empty output directory so separate runs cannot mix')

    def get(route):
        with urllib.request.urlopen(args.base.rstrip('/') + route, timeout=15) as response:
            return json.load(response)

    assert MODEL in {m['id'] for m in get('/v1/models')['data']}
    args.out.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    for number, policy in enumerate(policies):
        for thinking in ('off', 'on'):
            destination = args.out / f'policy-{number}-thinking-{thinking}'
            before = get('/health')
            assert before['backend'] == 'tensorfold' and before['ok'] and not before['busy']
            command = [sys.executable, '-u', str(Path(__file__).with_name('client.py')),
                       '--model', MODEL, '--label', 'decode-policy-tuning', '--base', args.base,
                       '--policy', policy, '--thinking', thinking, '--concurrency', args.concurrency,
                       '--repeats', str(args.repeats), '--input-lines', '27',
                       '--output-tokens', str(args.tokens), '--skip-warmup', '--warm-prefix',
                       '--out', str(destination)]
            with Path(str(destination) + '.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            after = get('/health')
            expected = sum(levels) * args.repeats
            assert after['requests_total'] - before['requests_total'] == expected * 2, 'Outside traffic'
            requests = [json.loads(line) for line in (destination / 'requests.jsonl').read_text().splitlines()]
            assert len(requests) == expected
            assert all(r['completion_tokens'] == args.tokens and r['finish_reason'] == 'length' for r in requests)
            with (destination / 'summary.csv').open() as source:
                summaries = list(csv.DictReader(source))
            assert all(int(r['prefill_active_tokens']) == 0 for r in summaries), 'Uncached prefill in decode'
            for row in summaries:
                rows.append({'policy': policy, 'thinking': thinking, 'concurrency': row['concurrency'],
                             'repeat': row['repeat'], 'streamed_decode_tps': row['decode_observed_aggregate_tps'],
                             'decode_round_tps': row['decode_active_aggregate_tps'],
                             'steady_concurrency_decode_round_tps': row['decode_exact_concurrency_tps'],
                             'cached_tokens': row['cached_tokens'], 'prompt_tokens': row['prompt_tokens']})
            checks.append({'policy': policy, 'thinking': thinking, 'requests': expected,
                           'zero_prefill': True, 'outside_requests': 0})
            with (args.out / 'summary.csv').open('w') as target:
                writer = csv.DictWriter(target, fieldnames=rows[0].keys(), lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
            (args.out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
            print(json.dumps({'policy': policy, 'thinking': thinking,
                              'decode_tps': [round(float(r['decode_observed_aggregate_tps']), 2)
                                             for r in summaries]}), flush=True)


if __name__ == '__main__':
    main()
