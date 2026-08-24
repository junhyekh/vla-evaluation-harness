---
smoke_config: smoke_test.yaml
---

# LIBERO UR5e + Robotiq85

Cross-embodiment LIBERO using robosuite 1.4's native UR5e and Robotiq85
geometry. The adapter keeps LIBERO's 7-D Cartesian-delta action contract,
transcodes the official Panda initial states, and maps Robotiq closure to the
2-D finger state expected by LIBERO-trained policies.

These results are not official Panda LIBERO scores. The Robotiq actuator speed
is set to Panda's `0.2` command rate so one binary gripper action has the same
temporal meaning while preserving the native Robotiq geometry and contacts.

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/libero-ur5:latest`

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `smoke_test.yaml` | UR5e + Robotiq85 gate | 1 | 1 |
| `object.yaml` | Complete LIBERO-Object order | 10 | 50 |

Run the deterministic controller/gripper proof with recording:

```bash
mkdir -p results/ur5_proof
docker run --rm --gpus all \
  -v "$PWD/results/ur5_proof:/workspace/results" \
  --entrypoint conda \
  ghcr.io/allenai/vla-evaluation-harness/libero-ur5:latest \
  run --no-capture-output -n libero python /app/prove_libero_ur5.py \
  --video-dir /workspace/results
```
