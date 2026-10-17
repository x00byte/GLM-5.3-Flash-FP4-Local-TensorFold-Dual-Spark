# Decode and FP8 cache tuning

The target revision, FP4-Local weight reader, target verification head and runtime kernels stay the same. This tuning changes serving configuration and repeats the measurements against the original deployment. Checkpoint weights stay read-only.

## Cache allocation

The launcher's startup admission reserve is 14.5 GiB per rank instead of 20 GiB. `TF_GLM_CACHE_GIB=8` caps the extra budget for kept prompts, including their recurrent states and draft windows; it is not the total size of the KV pool. The engine combines the request window and the available extra budget into one FP8 token pool. Both ranks agree on the smaller available budget.

The pool is shared across eight request slots. Each request still has a 262,144-token context limit. Pool capacity depends on available memory at startup, so another machine may allocate a different number of tokens. The admission reserve is an estimate rather than a guaranteed amount of physically free memory after CUDA initialization. Monitor Linux `MemAvailable` and swap activity during long-context use.

During this validation, minimum `MemAvailable` was 8.87 GiB on the head and 9.39 GiB on the worker. Head swap use rose from 1.28 to 1.88 GiB; worker swap use stayed at 0.09 GiB. The larger cache therefore has a host-memory cost even though these inference checks passed. Raise the reserve when more headroom is needed. This run checks short C1–C8 bursts and one 145k-token prompt, not eight simultaneous full-window requests or long-duration reliability.

`node.env.example` exposes the measured settings. To reproduce the earlier conservative configuration, set `TENSORFOLD_MEMORY_RESERVE_GIB=20`, `TF_GLM_MULTI_SAMPLER=streams`, `TF_GLM_MULTI_ASYNC=0`, and `TF_GLM_MULTI_LONE=0` on both ranks, then restart both recipe containers.

## Decode dispatch

`TF_GLM_MULTI_SAMPLER=packed` gathers candidate rows for multiple streams together, including nucleus-sampling streams. The original sampler already batches greedy/top-k gathers, so this is not a new communication saving in the temperature-zero benchmark. `TF_GLM_MULTI_ASYNC=1` sends rank messages through pinned staging buffers so the head can continue its host work. Target verification and sampling still use the original model. This configuration has not established a repeatable decode speedup; the confirmed improvement is cache capacity.

The single-stream graph option was also measured. With a 16 GiB reserve, packed sampling and async dispatch, `TF_GLM_MULTI_LONE=1` produced 51.6 tok/s at C1 and 126.9 aggregate tok/s at C8 with thinking off. A fresh run of the original configuration produced 53.4 and 136.3 respectively. These short runs vary, but they gave no reason to enable the lone-stream path for this checkpoint. The comparison changes multiple settings and does not isolate the graph option’s cost. The final recipe keeps `TF_GLM_MULTI_LONE=0`.

## Draft policies

Three request-level policies were measured: `fnc7:0.3`, `fnc7:0.1`, and `fnc7:0.5`, with thinking off/high at C1, C4 and C8. Each profile made 78 timed requests and 78 prefix seeds. All timed requests performed zero prefill work, and their token hashes matched the original configuration.

Lowering the confidence threshold helped some thinking-high cases and hurt thinking-off cases. The default remains `fnc7:0.3`; the experiments do not establish one policy as fastest for arbitrary workloads. The published main table uses that same policy in both thinking modes.

To run a quick policy comparison against this model:

```bash
python3 benchmarks/tune_decode.py --out local-results/policies
```

The script runs requests against the serving model without switching it. It checks token counts, zero prefill work and backend request totals. `summary.csv` separates full-burst streamed decode from engine-round throughput. Cached-input/TTFT ratios are omitted from warm-prefix decode records at their source.

The complete C1–C8 sweep and separate uncached prefill test remain:

```bash
python3 benchmarks/run.py --out local-results/tuned
python3 benchmarks/report.py local-results/tuned
```

## Upstream work to watch

As of 2026-10-17, TensorFold's [native GLM CUDA tracker](https://github.com/ashhart/TensorFold/issues/593) places the NVFP4 GLM family in the 1.0.12 candidate, with the full-checkpoint serving gate still pending. Our CPU admission check against pinned native source `6ee828c24bf802cdd73507d6fc60bb27a82c0d66` refused this target's `MIXED_PRECISION` format. Upgrading the binary alone does not replace this recipe's FP4-Local loader; no native GPU speedup is claimed.

[MiaAI-Lab's full GLM three-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold) includes a 4-bit KV-cache implementation. It is a separate full-GLM family and deployment. Porting it to this Flash engine would require checking its attention kernels, stored states, speculative decoding and model quality. These measurements continue to use FP8 KV.

## Credits

These options and their serving implementation come from [TensorFold](https://github.com/ashhart/TensorFold) and [MiaAI-Lab's Spark runtime](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold). See [NOTICE](../NOTICE) for retained credits and terms.
