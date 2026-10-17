# Read-only target weights over the fabric

The measured worker read the 165.3 GiB target checkpoint over read-only NFS 4.2 on the direct ConnectX link. The weights are uploaded into each GPU at startup; inference does not stream expert weights from NFS. This avoids requiring another full target copy on the worker. The small DFlash2 checkpoint is local to each node.

Use an existing NFS server when one is already exporting model storage. Add this checkpoint to its existing export layout; keep the other exports intact. A new Ubuntu server can use `nfs-kernel-server`. The exported directory must be a real filesystem directory, with readable checkpoint files and traversable directories.

For example, with `/srv/models` exported as the NFSv4 root and the checkpoint at `/srv/models/GLM-5.3-Flash-FP4-Local`, restrict the read-only export to the worker's fabric IP. Adapt the addresses and existing NFS root to your system. The worker volume is then:

```bash
docker volume create --driver local \
  --opt type=nfs \
  --opt 'o=addr=10.10.10.1,nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600' \
  --opt 'device=:/GLM-5.3-Flash-FP4-Local' \
  glm53-fp4-local-weights
```

Set `MODEL_SOURCE=glm53-fp4-local-weights` in the worker's `node.local.env`. The head can keep a normal directory in `MODEL_SOURCE`. Run `scripts/node.py validate` on both nodes before starting either GPU rank.

The recipe does not create, replace or stop an NFS server. A slow or unavailable export affects startup; verify the fabric path and mount before launching.

## Credits

Uses Linux NFS and Docker's local NFS volume driver. The shared-weight deployment follows the existing MiaAI-Lab dual-Spark runtime pattern; see the [upstream recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).
