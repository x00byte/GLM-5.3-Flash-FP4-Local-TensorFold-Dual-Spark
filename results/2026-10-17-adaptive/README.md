# GLM-5.3-Flash-FP4-Local TensorFold measurements

Decode uses fully cached prompts and excludes time to first token. Prefill uses separate uncached requests with a one-token output cap. C2-C8 decode is aggregate. Values are tokens per second.

| Concurrent requests | Decode off | Decode high | Cold prefill off | Cold prefill high |
|---:|---:|---:|---:|---:|
| 1 | 94.7 | 87.7 | 830.8 | 866.0 |
| 2 | 152.5 | 150.9 | 872.9 | 863.8 |
| 3 | 199.7 | 155.5 | 881.8 | 836.6 |
| 4 | 239.6 | 212.3 | 857.3 | 860.6 |
| 5 | 257.7 | 232.3 | 863.0 | 887.5 |
| 6 | 300.9 | 293.4 | 876.7 | 873.2 |
| 7 | 338.7 | 275.9 | 896.1 | 891.7 |
| 8 | 328.5 | 280.3 | 890.5 | 885.9 |

Workload: edit; decode cap: 128 tokens.

These are full-burst streamed decode rates after prefix warmup. The separate `steady_concurrency_decode_round_tps` column in [reported-metrics.csv](reported-metrics.csv) counts only engine rounds with exactly C decoding streams; it excludes scheduling gaps and lower-concurrency tails. It must not be labeled as a client full-burst rate. Cached token counts divided by warm-request TTFT are not reported as prefill throughput.

The complete sweep, individual request timing/usage records and health-counter deltas are retained beside this report. This is a short workload-specific benchmark; the exact workload and decode cap are recorded in `complete.json`.

The shared FP8 pool held 1,286,144 tokens. [Exact configuration](configuration.json), [source-matching Docker build](build-validation.json) and [token checks](token-validation.json) are included. [Controlled comparisons](../2026-10-17-draft-tests/README.md) include prose/code, original settings, copy on/off repeats and a separate 512-token test.

The [145,485-token context check](long-context-validation.json) passed in 146.7 seconds. All nine post-context decode hashes matched the original profile, with zero timed prefill and no outside timed requests. [Final validation](final-validation.json).

## Credits

Measured using the [TensorFold](https://github.com/ashhart/TensorFold) Spark runtime, [NVIDIA target checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local), and [IncoAI DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
