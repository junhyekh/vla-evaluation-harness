---
smoke_config: smoke_test.yaml
---

# LIBERO-Pro UR5e + Robotiq85

The four released LIBERO-Pro Object perturbations evaluated with the UR5e and
Robotiq85 cross-embodiment adapter. These results are not official Panda
LIBERO-Pro scores.

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/libero-pro-ur5:latest`

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `smoke_test.yaml` | One perturbed task and initial state | 1 | 1 |
| `object.yaml` | Four Object-suite perturbations | 40 | 50 |
