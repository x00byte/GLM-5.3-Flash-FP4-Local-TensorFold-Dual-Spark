# GLM-5.3-Flash-FP4-Local TensorFold measurements

Decode uses fully cached prompts and excludes time to first token. Prefill uses separate uncached requests with a one-token output cap. C2-C8 decode is aggregate. Values are tokens per second.

| Concurrent requests | Decode off | Decode high | Cold prefill off | Cold prefill high |
|---:|---:|---:|---:|---:|
| 1 | 50.9 | 40.9 | 828.8 | 819.6 |
| 2 | 80.0 | 71.3 | 818.1 | 896.6 |
| 3 | 89.1 | 71.4 | 920.6 | 866.1 |
| 4 | 98.0 | 88.8 | 896.6 | 894.5 |
| 5 | 96.4 | 95.0 | 868.3 | 907.4 |
| 6 | 122.8 | 107.5 | 920.2 | 919.6 |
| 7 | 123.0 | 113.9 | 907.0 | 863.9 |
| 8 | 128.0 | 118.2 | 894.2 | 894.1 |

These are full-burst streamed decode rates after prefix warmup. The separate `steady_concurrency_decode_round_tps` column in [reported-metrics.csv](reported-metrics.csv) counts only engine rounds with exactly C decoding streams; it excludes scheduling gaps and lower-concurrency tails. It must not be labeled as a client full-burst rate. Cached token counts divided by warm-request TTFT are not reported as prefill throughput.

The complete sweep, individual request timing/usage records and health-counter deltas are retained beside this report. This is a short benchmark of a fixed ledger workload, not a general model-quality or long-output throughput evaluation.

The tuned profile allocates 1,415,168 FP8 pool tokens. The separate [long-context validation](long-context-validation.json) and post-warmup C1/C4/C8 decode checks passed. All 72 sweep hashes and 26 post-warmup hashes match the original profile. [Memory observations](memory-validation.json), [configuration changes](optimization.json), and policy/graph experiments under `tuning/` are retained.

The runtime sources and Dockerfile are unchanged from v0.1.0. The [original real-weight GPU checks](../2026-10-17/gpu-checks.json) and [source-matching build receipt](../2026-10-17/build-validation.json) still apply.

## Credits

Measured using the [TensorFold](https://github.com/ashhart/TensorFold) Spark runtime, [NVIDIA target checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local), and [IncoAI DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
