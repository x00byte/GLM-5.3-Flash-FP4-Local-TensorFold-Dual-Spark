# Drafting and serving comparison

Same target and prompts, 128 output tokens, thinking off. C4/C8 values are aggregate decode tok/s. One burst per cell; repeat checks are separate.

| Workload | C | Original | MSE draft | Basic copy | New neural-only | Adaptive |
|---|---:|---:|---:|---:|---:|---:|
| general | 1 | 38.2 | 39.1 | 38.2 | 38.0 | 37.9 |
| general | 4 | 84.1 | 81.7 | 84.5 | 98.4 | 98.5 |
| general | 8 | 109.8 | 109.6 | 108.9 | 139.9 | 140.0 |
| code | 1 | 49.4 | 46.4 | 49.1 | 49.1 | 49.2 |
| code | 4 | 99.4 | 100.5 | 98.4 | 114.0 | 114.1 |
| code | 8 | 131.6 | 132.8 | 131.1 | 151.8 | 151.7 |
| edit | 1 | 81.0 | 80.4 | 63.3 | 80.4 | 94.8 |
| edit | 4 | 156.7 | 156.5 | 125.6 | 209.2 | 241.2 |
| edit | 8 | 213.0 | 211.9 | 172.8 | 295.3 | 341.1 |

The new profile combines cached-request scheduling, wider verification and adaptive span reuse. Its neural-only column disables copying in that same process. [Configuration and measurement scope](configuration.json), [raw comparison CSV](comparison.csv), and request records accompany the table. All rates exclude prompt work; cached-input/TTFT ratios are omitted.

The MSE setting affects both the proposal head and other DFlash matrices. It did not establish a repeatable improvement. The original target head remains MXFP8; proposals use a separate Q4 head.

Native TensorFold 1.0.12 could not replace this mixed FP4-Local loader. The [binary smoke check](native-binary-check.json) stopped at the deliberately limited memory budget; the separate [CPU format-admission check](../2026-10-17-tuned/native-admission-check.json) rejects the actual mixed format. No native full-model speed is claimed.

## Credits

[TensorFold](https://github.com/ashhart/TensorFold) prompt lookup and inference engine; [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) Spark runtime; NVIDIA target and IncoAI drafter. See [NOTICE](../../NOTICE).
