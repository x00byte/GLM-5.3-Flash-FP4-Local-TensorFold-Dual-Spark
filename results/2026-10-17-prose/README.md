# Prose decode measurements

GLM-5.3-Flash-FP4-Local on Dual Spark, TP2, patched Python TensorFold with DFlash2. Current adaptive serving profile; target precision and runtime unchanged. Temperature zero, warmed prefixes, one burst per cell. Decode excludes prefill and TTFT; C2–C8 is aggregate output tokens/s.

## Thinking off and high

128 output tokens per request. High-effort rates include generated reasoning tokens.

| Concurrent requests | Thinking off | Thinking high |
|---:|---:|---:|
| 1 | 38.0 | 46.8 |
| 2 | 66.3 | 69.1 |
| 3 | 78.1 | 97.9 |
| 4 | 98.7 | 102.8 |
| 5 | 105.0 | 120.3 |
| 6 | 136.8 | 122.5 |
| 7 | 125.5 | 138.0 |
| 8 | 140.0 | 144.1 |

## Draft-policy screen

Same 128-token thinking-off prompts. All six policies produced matching output hashes; no candidate improved on the existing default in this screen.

| Policy | C1 | C8 aggregate |
|---|---:|---:|
| `fnc7:0.3` | 38.1 | 140.0 |
| `fnc7:0.1` | 36.4 | 127.3 |
| `f7` | 28.0 | 94.0 |
| `f3` | 33.7 | 129.0 |
| `fnc7:0.6` | 37.8 | 137.7 |
| `fcost7:noisy` | 36.8 | 134.3 |

Separate **512-token** thinking-off check using the default: **37.2 tok/s C1**, **111.3 tok/s aggregate C8**. Different output caps must not be used to claim a tuning gain.

The earlier serving-profile change improved the 128-token prose screen from 109.8 to 140.0 tok/s at C8; C1 remained approximately 38 tok/s. The default screen accepted about 2.3 output tokens per verification round. Longer chains added verification work with little acceptance gain. Repetitive-edit copy gains do not transfer to this prose sample.

[Raw policy comparison](screen-results.json), [full-sweep/long-output summary](final-results.json), [configuration](configuration.json), and [validation](validation.json). Request records and backend counter deltas accompany each run. Reproduce with `benchmarks/client.py --workload general --warm-prefix --skip-warmup --policy fnc7:0.3 --output-tokens 128 --thinking off` plus the model, label, concurrency and output-directory arguments; use `--thinking on` for native high effort.

## Credits

[TensorFold](https://github.com/ashhart/TensorFold) inference engine, [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) Spark runtime, NVIDIA target checkpoint and IncoAI DFlash2 drafter. See [NOTICE](../../NOTICE).
