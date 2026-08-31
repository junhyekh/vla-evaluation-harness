---
smoke_config: eval.yaml
---

# VLABench

Language-conditioned manipulation benchmark built on dm_control and MuJoCo.
[Paper](https://arxiv.org/abs/2412.18194) | [GitHub](https://github.com/OpenMOSS/VLABench)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/vlabench:latest`

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | Official Track 1 (`track_1_in_distribution`) | 10 | 50 |

The adapter reads the fixed episodes from
`$VLABENCH_ROOT/configs/evaluation/tracks/track_1_in_distribution.json` and
reports success rate, intention score, and progress score.
