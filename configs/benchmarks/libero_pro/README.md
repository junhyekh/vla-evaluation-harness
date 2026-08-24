---
smoke_config: smoke_test.yaml
---

# LIBERO-Pro

Robustness evaluation over LIBERO object, initial-position, language, and task
perturbations (MuJoCo/robosuite).
[Paper](https://arxiv.org/abs/2510.03827) | [GitHub](https://github.com/Zxy-MLlab/LIBERO-PRO) | [Dataset](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/libero-pro:latest`

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `smoke_test.yaml` | One official task and initial state | 1 | 1 |
| `object.yaml` | Object suite x 4 released perturbations | 40 | 50 |
| `eval.yaml` | 4 suites × 4 released perturbations | 160 | 50 |

The image pins both the official code and dataset revisions. Environment
perturbations are not in the released pre-generated dataset, so they are not
part of this reproducible order.

See also: [RBY1-right](../libero_pro_rby1/) and
[UR5e + Robotiq85](../libero_pro_ur5/) cross-embodiment LIBERO-Pro.
