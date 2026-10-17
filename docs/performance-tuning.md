# Serving profile and cache

The target remains the pinned `nvidia/GLM-5.3-Flash-FP4-Local` checkpoint. Target verification retains its original MXFP8 head. The selected profile uses minmax Q4 proposal weights, DFlash2 `fnc7:0.3`, adaptive copy lookup, packed sampling, async rank messages and batched verification. Lone-stream graphs remain off.

## Decode

The [adaptive extension](adaptive-drafting.md) learns repeated token-span edits from verified output. The shared verification window is 128 rows, with a 2 MiB RoCE threshold. Complete cache hits start before expensive-prefill share accounting, with bounded priority and existing partial-fill scheduling preserved.

The [controlled comparisons](../results/2026-10-17-draft-tests/README.md) separate the original serving profile, MSE proposal quantization, basic copying, the new neural-only profile and adaptive copying. The same-process copy on/off measurements isolate copying. Other improvements are measured as a combined serving profile.

## Cache

| Profile | Shared FP8 pool | Shared verification rows |
|---|---:|---:|
| Initial | 610,304 | 64 |
| Previous cache tuning | 1,415,168 | 64 |
| Current measured launch | 1,286,144 | 128 |

Each request retains a 262,144-token context limit; eight slots share the pool. These are observed allocations, which also depend on free memory at startup. Wider verification consumes buffer memory. This profile prioritizes decode throughput over the previous maximum pool allocation.

The startup reserve is 14.5 GiB per rank. `TF_GLM_CACHE_GIB=8` caps extra kept-prompt budget; it is not the entire KV pool. The reserve is an admission estimate, not a guarantee of physically free memory. [Fresh configuration and checks](../results/2026-10-17-adaptive/README.md) and [previous memory observations](../results/2026-10-17-tuned/memory-validation.json) are retained. Validation covers short bursts and a separate 145k-token prompt, rather than eight simultaneous maximum-context requests.

## Native engine

The official native TensorFold 1.0.12 binary was checksum-verified and inspected. Its loader is not a replacement for this mixed FP4-Local reader. The executed CPU format check rejected `MIXED_PRECISION`; the bounded binary serve check stopped at its imposed memory ceiling. No native full-model GPU speed is claimed. [Evidence](../results/2026-10-17-draft-tests/configuration.json).

## Credits

[TensorFold](https://github.com/ashhart/TensorFold) engine and prompt lookup; [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) Spark runtime and backports. See [NOTICE](../NOTICE).
