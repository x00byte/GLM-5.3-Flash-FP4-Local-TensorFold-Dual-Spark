# GLM-5.3-Flash-FP4-Local · TensorFold · Dual Spark

![GLM-5.3-Flash FP4 Local on Dual Spark with TensorFold](assets/cover.png)

Run [`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) on two 128 GB GB10 Spark-class nodes with TensorFold. This recipe adds support for NVIDIA's mixed FP4-Local storage and losslessly compressed expert scales to the patched Python TensorFold Spark runtime.

The target snapshot, TensorFold base image and DFlash2 drafter are pinned in [provenance.json](provenance.json). The measured deployment uses TP2 over RoCE, eight request slots, FP8 KV, a 262,144-token per-request context limit and adaptive DFlash2 `fnc7:0.3` proposals. The shared KV pool limits the total live context across requests.

## Measured performance

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

Measured on 2026-10-17. Rates are tokens/s. C1 is single-stream decode; C2-C8 decode is aggregate. Thinking high uses GLM's native high effort. Decode is measured after full prompt-cache warmup with zero prefill tokens verified; cold prefill comes from separate uncached requests. The table uses full-burst streamed decode, including scheduling and tail effects after the first token. [Raw records, counters and exact configuration](results/2026-10-17/README.md) are included.

The 2,048-token prefill chunks improved this workload's C1 cold prefill by **24.1%** over the initial 512-token configuration. Best cold prefill in the final sweep was **926.5 input tok/s**. The earlier three-run C1/C4/C8 decode confirmation is retained under a separately labeled 512-token tuning configuration. Steady-concurrency engine-round rates are available in the detailed CSV; they are distinct from the full-burst client rates above.

## Build

Each node needs Arm64 Linux, Docker with NVIDIA Container Toolkit, a working NVIDIA driver, RDMA devices and a configured direct RoCE link. The measured driver was 580.142; the pinned runtime supplies PyTorch 2.13.0a0 from NVIDIA 26.07 and CUDA 13.3. Build on both nodes:

```bash
git clone https://github.com/x00byte/GLM-5.3-Flash-FP4-Local-TensorFold-Dual-Spark.git
cd GLM-5.3-Flash-FP4-Local-TensorFold-Dual-Spark
docker build -t local/glm53-fp4-local-tensorfold:20261017 .
```

The Dockerfile applies the recorded Spark backports and the FP4-Local patch to the pinned public base. Its 429 runtime source files were compared with the measured deployment; [the build receipt](results/2026-10-17/build-validation.json) records the matching fingerprint.

## Download the pinned checkpoints

The target snapshot is 165.3 GiB. Allow additional space for the runtime, cache and drafter. Download to real directories on the head, and download the drafter on the worker. The worker can use a full local target copy or the [read-only NFS setup](docs/nfs.md) used for these measurements.

```bash
mkdir -p "$HOME/models"
docker run --rm --user "$(id -u):$(id -g)" \
  -e HF_HOME=/tmp/hf-cache \
  -v "$HOME/models:/models" \
  -v "$PWD/scripts:/recipe:ro" \
  --entrypoint python \
  local/glm53-fp4-local-tensorfold:20261017 \
  /recipe/download.py
```

On a worker using NFS for the target, add `--draft-only` to that download command. The pinned target contains 42 indexed weight files, including the input-scale sidecar and `tensors/native-QB.safetensors`. Download the entire snapshot. File sizes and small-file hashes are checked before serving; a one-time full weight verification is available:

```bash
python3 scripts/verify_snapshot.py "$HOME/models/GLM-5.3-Flash-FP4-Local" --full
```

The default measured configuration uses [IncoAI DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), licensed **CC BY-NC-ND 4.0 for non-commercial use**. The NVIDIA target uses the NVIDIA Open Model Agreement. Weights are downloaded from their owners and are not included here.

## Start the two ranks

Copy `node.env.example` to `node.local.env` on each node. Set its model and cache paths, fabric address, interface, RoCE devices and GID index for that node. Use `RANK=1` on the worker and `RANK=0` on the head. Directories must be writable where downloads and caches are stored; serving mounts both checkpoints read-only.

On each node:

```bash
set -a
source node.local.env
set +a
python3 scripts/node.py validate
```

Start the worker first, then the head, using the following command on each:

```bash
python3 scripts/node.py start
```

Rank 0 waits for API readiness on port 8888. Kernel compilation is cached between starts. `scripts/node.py stop` stops only this recipe's named container on the node where it runs. The launcher does not stop other GPU workloads. For a run without draft proposals, add `--plain` to both start commands; the published rates use DFlash2.

## Reproduce the benchmark

Pause other inference clients and warm the serving process before measuring. The suite checks the exact served model and TensorFold backend, verifies every request's token cap and cache use, and checks backend request totals for outside traffic:

```bash
python3 benchmarks/run.py --out local-results
python3 benchmarks/report.py local-results
```

C1-C8 decode requests first seed their own prompt state, then generate 64 output tokens. Counter deltas must show zero prefill tokens during each measured decode burst. Separate one-output-token requests measure uncached prefill. Temperature is zero; thinking uses GLM's native off/high modes. The full sweep makes 144 timed requests and 72 prefix seeds.

The [measurement report](results/2026-10-17/README.md) separates full-burst client decode, decode-round throughput at exactly C active streams, and cold prefill. Warm-prefix TTFT is never presented as prefill throughput. Short ledger prompts and 64-token outputs keep the test quick; throughput varies with prompt content, draft acceptance, output length and context.

For the retained runtime's long-context module warmup:

```bash
python3 scripts/warmup_long_context.py
```

The measured warmup passed at **145,485 input tokens in 150.8 s** (964.8 input tok/s). This is a separate long-context validation workload. It exercises the selection tile above 131k tokens in the serving process. It takes a separate uncached 145k-token prompt, uses no drafts, and is excluded from the throughput table.

## Implementation and validation

The [implementation notes](docs/implementation.md) describe the lossless CSF32 decoder, stored MXFP8/block-FP8 paths, fused projections and decode counters. An additional Q4 output-head copy serves draft proposals; target verification retains the original MXFP8 head. The target checkpoint files remain unchanged.

The published scope is text inference. Real-weight GPU checks, speculative/non-speculative token hashes, C1-C8 measurements and the long-context warmup are recorded with the results. These checks do not establish general model quality. The runtime here is the patched **Python TensorFold 0.6.0** Spark engine; the newer native TensorFold engine has a different loader.

## Credits

- [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart and contributors supplies the inference engine, NVFP4/MXFP8/FP8 CUDA kernels and logo reference.
- [MiaAI-Lab's GLM Spark runtime](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) supplies the pinned base image and backports, with prior work by [Jay Leaton](https://github.com/jayleaton/glm53-tensorfold-spark) and other contributors recorded in [the retained credits](third_party/MiaAI-CREDITS.md).
- [FlashInfer/b12x](https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/experimental/b12x) supplies the reference CSF32 scale layout and the underlying RoCE transport lineage.
- [NVIDIA](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local), [Z.AI](https://huggingface.co/zai-org/GLM-5.3-Flash), and [IncoAI](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) supply the target/base model and drafter.

See [NOTICE](NOTICE) for retained third-party terms. The cover uses TensorFold's logo as its reference for this independent recipe and implies no endorsement.
