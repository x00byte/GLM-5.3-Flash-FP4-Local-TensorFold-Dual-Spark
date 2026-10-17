FROM ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14b5072e8e9918f1080153f57321415a387bc7d78cf9c9b44a7aeae2d4399ccf
COPY patches/bespoke/ /opt/glm-fp4-local/bespoke/
RUN cd "$(python3 -c 'import pathlib,tensorfold; print(pathlib.Path(tensorfold.__file__).parent.parent)')" && \
    for p in /opt/glm-fp4-local/bespoke/*.patch; do patch -p0 --forward --fuzz=0 < "$p" || exit 1; done
COPY patches/fp4-local.patch /opt/glm-fp4-local/fp4-local.patch
RUN cd "$(python3 -c 'import pathlib,tensorfold; print(pathlib.Path(tensorfold.__file__).parent.parent)')" && \
    patch -p1 --forward --fuzz=0 < /opt/glm-fp4-local/fp4-local.patch && \
    python3 -c 'import tensorfold.families.glm5_next.cuda.fp4_local,tensorfold.families.glm5_next.cuda.multi,tensorfold.server.metrics'
ENV MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=true
LABEL org.opencontainers.image.title="GLM-5.3-Flash-FP4-Local TensorFold" \
      org.opencontainers.image.source="https://github.com/x00byte/GLM-5.3-Flash-FP4-Local-TensorFold-Dual-Spark"

COPY target-manifest.json scripts/verify_snapshot.py /opt/glm-fp4-local/
