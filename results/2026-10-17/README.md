# GLM-5.3-Flash-FP4-Local TensorFold measurements

Decode uses fully cached prompts and excludes time to first token. Prefill uses separate uncached requests with a one-token output cap. C2-C8 decode is aggregate. Values are tokens per second.

| Concurrent requests | Decode off | Decode high | Cold prefill off | Cold prefill high |
|---:|---:|---:|---:|---:|
| 1 | 52.8 | 40.8 | 822.9 | 811.6 |
| 2 | 90.5 | 71.4 | 901.5 | 896.5 |
| 3 | 88.6 | 74.9 | 916.2 | 922.9 |
| 4 | 92.1 | 88.5 | 894.3 | 894.1 |
| 5 | 102.4 | 93.4 | 916.0 | 914.0 |
| 6 | 119.0 | 106.9 | 926.5 | 921.0 |
| 7 | 122.6 | 114.6 | 909.0 | 908.5 |
| 8 | 131.5 | 115.4 | 921.1 | 897.6 |

These are full-burst streamed decode rates after prefix warmup. The separate `steady_concurrency_decode_round_tps` column in [reported-metrics.csv](reported-metrics.csv) counts only engine rounds with exactly C decoding streams; it excludes scheduling gaps and lower-concurrency tails. It must not be labeled as a client full-burst rate. Cached token counts divided by warm-request TTFT are not reported as prefill throughput.

The complete sweep, individual request timing/usage records and health-counter deltas are retained beside this report. This is a short benchmark of a fixed ledger workload, not a general model-quality or long-output throughput evaluation.

The [long-context check](long-context-validation.json) passed with 145,485 uncached input tokens. All 72 decode token hashes matched between the 512- and 2,048-token prefill configurations, and all 13 C1/C4/C8 outputs matched after the long-context check; see [token validation](token-validation.json).

## Credits

Measured using the [TensorFold](https://github.com/ashhart/TensorFold) Spark runtime, [NVIDIA target checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local), and [IncoAI DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
