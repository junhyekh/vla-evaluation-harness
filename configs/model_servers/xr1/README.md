---
smoke_config: robocasa365.yaml
---

# Xiaomi Robotics XR-1

Xiaomi's Qwen3-VL/DiT policy. [Paper](https://arxiv.org/abs/2607.15330) | [GitHub](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1)

| File | Benchmark | Checkpoint |
|------|-----------|------------|
| `robocasa.yaml` | RoboCasa | `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa` |
| `robocasa365.yaml` | RoboCasa365 | `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365` |
| `vlabench.yaml` | VLABench | `XiaomiRobotics/Xiaomi-Robotics-1-VLABench` |

Each config pins the tested Hugging Face revision. Download it before launching, for example:

```bash
hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 \
  --revision 3a6d0293bfa90759d34a7fc48c2c62413cd7bcf4
vla-eval serve configs/model_servers/xr1/robocasa365.yaml
```

The server preserves the released processor prompts, state padding, normalization,
RoboCasa365 observation history, and benchmark-specific replanning horizons.
