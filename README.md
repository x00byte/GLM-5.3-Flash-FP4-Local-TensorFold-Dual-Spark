# GLM-5.3-Flash-FP4-Local · TensorFold · Dual Spark

![GLM-5.3-Flash FP4 Local on Dual Spark](assets/cover.png)

[`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) on two 128 GB GB10 Sparks. TP2 over RoCE, patched Python TensorFold 0.6.0, DFlash2 `fnc7:0.3`, FP8 KV.

- Shared KV pool: **1,415,168 tokens** across eight slots.
- Per-request context limit: **262,144 tokens**.
- Target snapshot: **165.3 GiB**. [Pinned revisions](provenance.json).

## Measurements

| Concurrent requests | Decode off | Decode high | Cold prefill off | Cold prefill high |
|---:|---:|---:|---:|---:|
| 1 | 50.9 | 40.9 | 828.8 | 819.6 |
| 2 | 80.0 | 71.3 | 818.1 | 896.6 |
| 3 | 89.1 | 71.4 | 920.6 | 866.1 |
| 4 | 98.0 | 88.8 | 896.6 | 894.5 |
| 5 | 96.4 | 95.0 | 868.3 | 907.4 |
| 6 | 122.8 | 107.5 | 920.2 | 919.6 |
| 7 | 123.0 | 113.9 | 907.0 | 863.9 |
| 8 | 128.0 | 118.2 | 894.2 | 894.1 |

Tokens/s, measured 2026-10-17. C2–C8 decode is aggregate. Thinking uses native off/high modes. Decode excludes prefill and TTFT; prefill uses separate uncached requests. Short synthetic prompts, 64-token decode output. [Raw results and method](results/2026-10-17-tuned/README.md).

Cache capacity increased 2.32× from the [initial configuration](results/2026-10-17/README.md). No repeatable decode speedup was established. The 145,485-token context check passed. [Tuning and memory observations](docs/performance-tuning.md).

## Run

```bash
docker build -t local/glm53-fp4-local-tensorfold:20261017 .
```

[Download, configure and start both ranks](docs/setup.md). [Worker NFS setup](docs/nfs.md).

```bash
python3 benchmarks/run.py --out local-results/sweep
python3 benchmarks/report.py local-results/sweep
```

Text inference is tested. The measured DFlash2 drafter is **CC BY-NC-ND 4.0**; the target uses the NVIDIA Open Model Agreement. Weights are downloaded separately.

## Credits

[TensorFold](https://github.com/ashhart/TensorFold) engine and CUDA kernels; [MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) Spark base and backports; [FlashInfer/b12x](https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/experimental/b12x) scale format and RoCE lineage. Models from NVIDIA, Z.AI and IncoAI. [Full credits](third_party/MiaAI-CREDITS.md) · [Notices](NOTICE).
