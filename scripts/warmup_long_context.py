#!/usr/bin/env python3
"""Load the >131k selection tile in the serving process before production.

Addresses the late module-load exposure in issue #150. Does not claim to fix
the underlying driver bug. A non-drafted request leaves no kept prompt state.
"""
import json
import time
import urllib.request

BASE = 'http://127.0.0.1:8888'
MODEL = 'nvidia/GLM-5.3-Flash-FP4-Local'
TARGET = 145000

def post(route, body):
    request = urllib.request.Request(BASE + route, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)

def main():
    with urllib.request.urlopen(BASE + '/v1/models', timeout=10) as response:
        models = json.load(response)['data']
    if not any(model['id'] == MODEL for model in models):
        raise SystemExit('Warmup refused: the GLM model is not serving. The served model was not changed.')
    line = 'GET /api/item?id=17 HTTP/1.1\nHost: example.invalid\nHTTP/1.1 200 OK\nContent-Type: text/plain\nBody: synthetic calibration record\n'
    lines = 2000
    for _ in range(5):
        messages = [{'role': 'user', 'content': 'Read these synthetic HTTP records. Reply OK.\n' + line * lines}]
        result = post('/tokenize', {'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}})
        tokens = len(result['tokens'])
        if TARGET <= tokens <= 180000:
            break
        lines = max(1, int(lines * (TARGET + 512) / max(1, tokens)))
    else:
        raise RuntimeError('Could not size the long-context warmup safely')
    started = time.monotonic()
    result = post('/v1/chat/completions', {'model': MODEL, 'messages': messages,
        'max_tokens': 1, 'temperature': 0, 'draft': False,
        'chat_template_kwargs': {'enable_thinking': False}})
    assert result.get('choices'), result
    with urllib.request.urlopen(BASE + '/health', timeout=10) as response:
        health = json.load(response)
    assert health.get('ok') is not False, health
    print(f'GLM serving-process warmup passed: {tokens} prompt tokens, {time.monotonic() - started:.1f}s; no drafted cache state retained.')

if __name__ == '__main__':
    main()
