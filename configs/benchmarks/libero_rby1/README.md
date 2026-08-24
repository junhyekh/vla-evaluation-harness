---
smoke_config: smoke_test.yaml
---

# LIBERO RBY1-right

Cross-embodiment LIBERO using the fixed RBY1-A body, stock right arm, and stock
parallel-gripper meshes from the pinned MuJoCo Menagerie model. The second jaw
transform and motion limits match Rainbow Robotics' current RBY1-A URDF.

These results are a separate RBY1 protocol and are not official Panda LIBERO
scores. `object.yaml` preserves LIBERO's task and 50-state order while changing
only the embodiment.

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/libero-rby1:latest`

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `smoke_test.yaml` | RBY1-right proof gate | 1 | 1 |
| `object.yaml` | Complete LIBERO-Object order | 10 | 50 |

The image also contains a deterministic motion proof that first rejects
non-mesh gripper geometry, then must satisfy a task inside the standard
280-action LIBERO-Object horizon:

```bash
mkdir -p results/rby1_repair
docker run --rm --gpus all \
  -v "$PWD/results/rby1_repair:/workspace/results" \
  --entrypoint conda \
  ghcr.io/allenai/vla-evaluation-harness/libero-rby1:latest \
  run --no-capture-output -n libero python /app/prove_libero_rby1.py \
  --video-dir /workspace/results
```

The proof writes an MP4 plus a JSON sidecar containing each motion stage,
tool axis, object pose, and jaw positions.
