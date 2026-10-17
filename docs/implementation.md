# FP4-Local TensorFold support

The target is exactly `nvidia/GLM-5.3-Flash-FP4-Local`, revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`. This is a patch of the Python TensorFold 0.6.0 Spark runtime. It is separate from TensorFold's newer native runtime.

The checkpoint uses ModelOpt `MIXED_PRECISION`. Routed experts store NVFP4 nibbles with E4M3 scales compressed as `csf_32x2_v1`; dense projections use MXFP8 or block-FP8, with BF16 tensors elsewhere. Forty numbered shards plus the input-scale sidecar and native-QB file form 42 indexed weight files. Download the entire pinned snapshot.

The loader expands CSF32 scale palettes and exceptions losslessly, splits weights and scales in the same TP coordinates, and packs the original codes into TensorFold's NVFP4 expert layout. It keeps dense quantization codes and scales as stored. Compatible adjacent projections are combined to reduce launches and intermediate copies. MLA absorption matrices are decoded to the BF16 form used by this TensorFold engine. FP32 rank partials are preserved until the existing TP reduction.

DFlash2 proposes tokens. A Q4 copy of the output head serves only those proposals; the original MXFP8 target head verifies them. This does not convert the target checkpoint to Q4 or EXL3. The served KV cache uses FP8, as recorded in the benchmark configuration.

New counters accumulate decode-round wall time and emitted tokens by actual active concurrency. They exclude prompt forwards and do not add GPU synchronizations. They measure the decode loop, including its internal CPU dispatch and sampling, rather than isolated CUDA kernels. `prefill_active_*` counters separately measure prompt work. Full-burst streamed output rates remain in the results because they include scheduling and tail effects that the steady-concurrency rate omits.

Validation includes real-checkpoint scale-plane comparisons against FlashInfer's reference, actual-weight projection/expert checks, fused-projection checks, speculative/non-speculative token hashes, C1-C8 request accounting, exact cache coverage, and separate uncached prefill. These checks are not a general model-quality evaluation. Text inference is the measured scope; vision is not enabled by this launcher.

Two experimental kernels were rejected after measurement: scale-folded expert MMA helped heavily shared experts but offered little benefit across the full 288-expert routing geometry; a block-FP8 Triton prefill experiment was numerically sound but slower than the existing CUDA path. Neither experiment is enabled in the published recipe.

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold), [MiaAI-Lab's Spark runtime](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold), and the [FlashInfer/b12x](https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/experimental/b12x) CSF32 format. See [NOTICE](../NOTICE) for retained notices and model terms.
