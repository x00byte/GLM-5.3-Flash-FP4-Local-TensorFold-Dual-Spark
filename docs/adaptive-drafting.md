# Adaptive copy drafting

The extension reuses repeated text and edits while the original GLM target verifies every proposed token. It changes neither the target prompt nor checkpoint weights. General prose and novel code may have few reusable spans.

The lookup chooses the longest matching suffix and abstains when equally strong matches disagree. A verified mismatch and a later unique prompt anchor identify a possible rewrite. Two distinct prompt positions must show the same edit before it enters the private lookup index. Edits may insert or delete tokens. Each request learns at most one rule, with spans bounded to 64 tokens. Three later copy misses disable copying for that request. DFlash2 supplies the remaining drafts.

The batched initializer uses the same factory as the single-stream path. Unused prelaunched proposals are discarded before recording neural or short-round output. Response statistics include `copy_rounds`, `copy_drafted`, `copy_accepted` and `copy_learned_edits`.

## Configuration

Use the same settings on both ranks:

```bash
TF_GLM_COPY_DRAFTS=1
TF_GLM_COPY_SMART=1
TF_GLM_COPY_MATCH=8
TF_GLM_COPY_MAX=15
TF_GLM_COPY_REPLY_MATCH=32
TF_GLM_COPY_MISS_MAX=0
TF_GLM_MULTI_WINDOW=128
TF_ROCE_MAX_KB=2048
```

The wider shared window allows eight 16-row verification segments. Buffers and admission estimates include its size. The larger window consumes memory that would otherwise be available for cache. Target and KV precision remain unchanged.

## Cached requests and comparison

A complete prompt-cache hit samples a saved head row. It now starts before applying the expensive-prefill share rule. Priority is bounded by admission order; existing partial prefills retain their scheduling rules.

The eight-slot server accepts `"tf_copy": false` to disable copy proposals for one request while retaining DFlash2, prefix reuse and the cache scheduling fix. The setting travels to both ranks. This allows a comparison in one serving process:

```bash
python3 benchmarks/workloads.py --label neural --no-copy --out local-results/neural
python3 benchmarks/workloads.py --label adaptive --reference local-results/neural --out local-results/adaptive
```

The Docker build runs CPU tests for span edits, conflicts, ambiguity, the real batched initializer, request transport, cache scheduling and a recorded tokenization regression. GPU measurements are reported separately under `results/`.

## Credits

Extends [TensorFold's](https://github.com/ashhart/TensorFold) prompt lookup and [MiaAI-Lab's](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) Spark runtime. Retained terms are in [NOTICE](../NOTICE).
