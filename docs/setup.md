# Setup and operation

## Build

Each node needs Arm64 Linux, Docker with NVIDIA Container Toolkit, a working NVIDIA driver, RDMA devices and a configured direct RoCE link. The measured driver was 580.142; the pinned runtime supplies PyTorch 2.13.0a0 from NVIDIA 26.07 and CUDA 13.3. Build on both nodes:

```bash
git clone https://github.com/x00byte/GLM-5.3-Flash-FP4-Local-TensorFold-Dual-Spark.git
cd GLM-5.3-Flash-FP4-Local-TensorFold-Dual-Spark
docker build -t local/glm53-fp4-local-tensorfold:20261017-v3 .
```

The Dockerfile applies the recorded Spark backports and the FP4-Local patch to the pinned public base. Its 430 runtime source files were compared with the measured deployment; [the build receipt](../results/2026-10-17-adaptive/build-validation.json) records the matching fingerprint.

## Download the pinned checkpoints

The target snapshot is 165.3 GiB. Allow additional space for the runtime, cache and drafter. Download to real directories on the head, and download the drafter on the worker. The worker can use a full local target copy or the [read-only NFS setup](../docs/nfs.md) used for these measurements.

```bash
mkdir -p "$HOME/models"
docker run --rm --user "$(id -u):$(id -g)" \
  -e HF_HOME=/tmp/hf-cache \
  -v "$HOME/models:/models" \
  -v "$PWD/scripts:/recipe:ro" \
  --entrypoint python \
  local/glm53-fp4-local-tensorfold:20261017-v3 \
  /recipe/download.py
```

On a worker using NFS for the target, add `--draft-only` to that download command. The pinned target contains 42 indexed weight files, including the input-scale sidecar and `tensors/native-QB.safetensors`. Download the entire snapshot. File sizes and small-file hashes are checked before serving; a one-time full weight verification is available:

```bash
python3 scripts/verify_snapshot.py "$HOME/models/GLM-5.3-Flash-FP4-Local" --full
```

The default measured configuration uses [IncoAI DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), licensed **CC BY-NC-ND 4.0 for non-commercial use**. The NVIDIA target uses the NVIDIA Open Model Agreement. Weights are downloaded from their owners and are not included here.

## Start the two ranks

Copy `node.env.example` to `node.local.env` on each node. Set its model and cache paths, fabric address, interface, RoCE devices and GID index for that node. Use `RANK=1` on the worker and `RANK=0` on the head. The measured performance defaults include a 14.5 GiB startup admission reserve and an 8 GiB cap on extra kept-prompt memory; see [cache tuning](performance-tuning.md). Directories must be writable where downloads and caches are stored; serving mounts both checkpoints read-only.

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

Pause other inference clients and warm the serving process before measuring. Use an empty output directory for each run. The suite checks the exact served model and TensorFold backend, verifies every request's token cap and cache use, and checks backend request totals for outside traffic:

```bash
python3 benchmarks/run.py --workload edit --input-lines 32 --output-tokens 128 --out local-results/sweep
python3 benchmarks/report.py local-results/sweep
```

C1-C8 decode requests first seed their own prompt state, then generate 128 output tokens. Counter deltas must show zero prefill tokens during each measured decode burst. Separate one-output-token requests measure uncached prefill. Temperature is zero; thinking uses GLM's native off/high modes. The full sweep makes 144 timed requests and 72 prefix seeds.

The [measurement report](../results/2026-10-17-adaptive/README.md) separates full-burst client decode, decode-round throughput at exactly C active streams, and cold prefill. Warm-prefix TTFT is never presented as prefill throughput. Synthetic record rewrites and 128-token outputs keep the test quick; throughput varies with prompt content, draft acceptance, output length and context.

For the retained runtime's long-context module warmup:

```bash
python3 scripts/warmup_long_context.py
```

The [145,485-token context check](../results/2026-10-17-adaptive/long-context-validation.json) passed in 146.7 seconds. It exercises the selection tile above 131k tokens in the serving process. It takes a separate uncached 145k-token prompt, uses no drafts, and is excluded from the throughput table.

## Implementation and validation

The [implementation notes](../docs/implementation.md) describe the lossless CSF32 decoder, stored MXFP8/block-FP8 paths, fused projections and decode counters. [Adaptive span drafting](adaptive-drafting.md) reuses repeated verified rewrites. An additional Q4 output-head copy serves draft proposals; target verification retains the original MXFP8 head. The target checkpoint files remain unchanged.

The published scope is text inference. Real-weight GPU checks, speculative/non-speculative token hashes, C1-C8 measurements and the long-context warmup are recorded with the results. These checks do not establish general model quality. The runtime here is the patched **Python TensorFold 0.6.0** Spark engine; the newer native TensorFold engine has a different loader.

## Credits

Based on [TensorFold](https://github.com/ashhart/TensorFold) and [MiaAI-Lab’s Spark runtime](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold). See [NOTICE](../NOTICE).
