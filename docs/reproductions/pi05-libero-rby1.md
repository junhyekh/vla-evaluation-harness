# pi0.5 LIBERO cross-embodiment diagnostic rerun

This report compares untouched pi0.5 base weights and pi0.5-LIBERO on the
complete LIBERO-Object task order, using the default Panda embodiment, the
fixed-base RBY1 right arm, and a UR5e with a Robotiq 85 gripper. It also
evaluates the four released
LIBERO-Pro Object perturbations: object, swap, language (`lan`), and task.

The run is a diagnostic slice: initial-state index 0 for every task. It is not
the official 50-initial-state LIBERO score, and RBY1 results remain non-official
cross-embodiment results. UR5 was added as a more Panda-like industrial-arm and
parallel-jaw control: it tests whether embodiment similarity recovers useful
behavior without model finetuning.

## Environment and pinned inputs

- Date: 2026-08-24/25 KST
- Hardware: 1x NVIDIA RTX A6000 (48 GiB)
- Harness revision at evaluation start: `e1ee9ad`
- Source state: that tracked revision plus the RBY1, UR5, and LIBERO-Pro working-tree
  changes in this checkout; the exact evaluated runtimes are identified by the
  Docker image hashes below
- OpenPI source revision: `981483dca0fd9acba698fea00aa6e52d56a66c58`
- Base model: `gs://openpi-assets/checkpoints/pi05_base`
- Base-model normalization statistics: `gs://openpi-assets/checkpoints/pi05_libero`
- Finetuned model: `gs://openpi-assets/checkpoints/pi05_libero`
- Normal RBY1 image: `ghcr.io/allenai/vla-evaluation-harness/libero-rby1:latest`,
  `sha256:453113e75925bdb591a2adb1f84be210efb9e1b2a59da1bbc488f97abf095378`
- Pro RBY1 image: `ghcr.io/allenai/vla-evaluation-harness/libero-pro-rby1:latest`,
  `sha256:c929ffe4a35241de93bf61e9eedf7a219c6aa34568ce38a972f1025dc293990f`
- Normal UR5 image: `ghcr.io/allenai/vla-evaluation-harness/libero-ur5:latest`,
  `sha256:a0534329c0d48ddc7136381618847d57fa8276e51b60891259c2d1490c1a6ca7`
- Pro UR5 image: `ghcr.io/allenai/vla-evaluation-harness/libero-pro-ur5:latest`,
  `sha256:15a860daca793d65a8fb1331e38e345e0434d22446817b04f04ad32bd0b98d55`
- UR5 stack: robosuite 1.4 native `UR5e` with stock `Robotiq85Gripper`
  geometry, joints, actuator, and contacts; only gripper speed is set to `0.2`
  to match the Panda binary-gripper response time
- RBY1 model source: MuJoCo Menagerie
  `da76818e269b82289eba39808e2fb91d679d6994`
- LIBERO-Pro source: `eafdb809426b13153aa1e4c42d6601844217dfec`

Each model was served once while four benchmark clients ran concurrently. Each
trial recorded step data and an MP4. Shard `0/50` deterministically selects
episode index 0 for every task: 10 normal Panda episodes, 10 normal RBY1
episodes, 40 Pro Panda episodes, and 40 Pro RBY1 episodes per model.

UR5 used the same one-state-per-task protocol. Pi0.5-LIBERO completed all 50
UR5 cells. For the untouched base model, normal and Pro object completed, but
the Pro run was stopped after three swap tasks once repeated failures made the
fine-tuned comparison the higher-value test. Base Pro swap is therefore 0/3
partial; language and task were not run.

## Reproduction

From the repository root, build the repaired RBY1 images:

```bash
docker/build.sh libero_rby1
docker/build.sh libero_pro_rby1
docker/build.sh libero_ur5
docker/build.sh libero_pro_ur5
```

Start the base-weight server in one terminal:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync vla-eval serve \
  -c configs/model_servers/pi0/libero_base.yaml -v
```

In a second terminal, run the four cells concurrently:

```bash
RERUN_ROOT=results/matrix_rby1_fixed_reproduction
RERUN_MODEL=pi05_base

uv run --no-sync vla-eval run -c configs/benchmarks/libero/object.yaml \
  --shard-id 0 --num-shards 50 --eval-id rerun-rby1fix-${RERUN_MODEL}-panda-normal-e0 \
  --output-dir ${RERUN_ROOT}/${RERUN_MODEL}/panda_normal --record-video -y -v &
uv run --no-sync vla-eval run -c configs/benchmarks/libero_rby1/object.yaml \
  --shard-id 0 --num-shards 50 --eval-id rerun-rby1fix-${RERUN_MODEL}-rby1-normal-e0 \
  --output-dir ${RERUN_ROOT}/${RERUN_MODEL}/rby1_normal --record-video -y -v &
uv run --no-sync vla-eval run -c configs/benchmarks/libero_pro/object.yaml \
  --shard-id 0 --num-shards 50 --eval-id rerun-rby1fix-${RERUN_MODEL}-panda-pro-e0 \
  --output-dir ${RERUN_ROOT}/${RERUN_MODEL}/panda_pro --record-video -y -v &
uv run --no-sync vla-eval run -c configs/benchmarks/libero_pro_rby1/object.yaml \
  --shard-id 0 --num-shards 50 --eval-id rerun-rby1fix-${RERUN_MODEL}-rby1-pro-e0 \
  --output-dir ${RERUN_ROOT}/${RERUN_MODEL}/rby1_pro --record-video -y -v &
wait
```

Stop the server, start the finetuned server, set `RERUN_MODEL=pi05_libero`, and
repeat the same four commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync vla-eval serve \
  -c configs/model_servers/pi0/libero.yaml -v
```

For UR5, substitute these benchmark configs and use a separate output root:

```bash
uv run --no-sync vla-eval run -c configs/benchmarks/libero_ur5/object.yaml \
  --shard-id 0 --num-shards 50 --eval-id ur5-${RERUN_MODEL}-normal-e0 \
  --output-dir results/matrix_ur5_reproduction/${RERUN_MODEL}/ur5_normal \
  --record-video -y -v

uv run --no-sync vla-eval run -c configs/benchmarks/libero_pro_ur5/object.yaml \
  --shard-id 0 --num-shards 50 --eval-id ur5-${RERUN_MODEL}-pro-e0 \
  --output-dir results/matrix_ur5_reproduction/${RERUN_MODEL}/ur5_pro \
  --record-video -y -v
```

The controller/gripper proof is independent of model inference:

```bash
docker run --rm --gpus all \
  -v "${PWD}/results/ur5_proof:/workspace/results" \
  --entrypoint conda \
  ghcr.io/allenai/vla-evaluation-harness/libero-ur5:latest \
  run --no-capture-output -n libero python /app/prove_libero_ur5.py \
  --video-dir /workspace/results
```

Docker creates the episode directories as an unmapped user. Restore ownership,
then materialize per-episode JSONL and aggregate JSON from every SQLite file:

```bash
docker run --rm -v "${PWD}/${RERUN_ROOT}:/data" --entrypoint chown \
  ghcr.io/allenai/vla-evaluation-harness/base:latest \
  -R "$(id -u):$(id -g)" /data

find ${RERUN_ROOT} -name 'recording-*.sqlite' -print0 \
  | xargs -0 -n1 uv run --no-sync vla-eval merge --db
```

The model samples actions stochastically. Concurrent-client scheduling changes
the sampling sequence, so these commands reproduce the protocol and artifact
structure but are not expected to reproduce every binary success outcome.

## Results

| Model | Embodiment | Normal | Pro object | Pro swap | Pro language | Pro task |
|---|---|---:|---:|---:|---:|---:|
| pi0.5 base | Panda | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| pi0.5 base | RBY1 right | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| pi0.5 base | UR5e + Robotiq 85 | 0/10 | 0/10 | 0/3 partial | not run | not run |
| pi0.5-LIBERO | Panda | 10/10 | 10/10 | 2/10 | 9/10 | 1/10 |
| pi0.5-LIBERO | RBY1 right | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| pi0.5-LIBERO | UR5e + Robotiq 85 | 2/10 | 0/10 | 2/10 | 0/10 | 2/10 |

All 200 episodes completed without runtime errors. The local artifact set at
`results/matrix_rby1_fixed_20260824/` passed SQLite integrity and artifact-count
checks: 8 SQLite databases, 200 episode rows, 200 JSONL trajectories, 200 MP4s,
and 20 aggregate JSON files. The ignored artifact bundle is 77,190,921 bytes.

The added UR5 bundle at `results/matrix_ur5_20260825/` contains four SQLite
databases, 73 completed episode rows, 73 JSONL trajectories, 73 finalized MP4s,
and eight aggregate JSON files. It also retains the interrupted base-Pro
recorder temporary MP4 as evidence that the 13-episode slice is partial. The
bundle is 40,442,457 bytes.

## Interpretation

### Base pi0.5 does not provide usable zero-shot LIBERO control

The untouched base checkpoint scored 0/100 across the complete Panda/RBY1
matrix and 0/23 in the completed UR5 slice. A representative Panda failure approached and knocked the red
carton instead of manipulating the requested green salad-dressing bottle. The
LIBERO normalization statistics make the action scale executable, but they do
not supply the task grounding learned by the finetuned checkpoint.

### The repaired RBY1 gripper is not the remaining bottleneck

Pi0.5-LIBERO scored 10/10 on normal Panda and 0/10 on the matched RBY1 task
order; it scored 22/40 on Panda Pro and 0/40 on RBY1 Pro. All 50 RBY1 failures
reached the 280-step horizon without simulator or controller errors.

The RBY1 controller was active rather than frozen. Across the 10 normal RBY1
episodes, median end-effector path length was 0.998 m, close commands occupied
36.8-60.4% of steps, and measured jaw gap moved from about 85.6 mm open to
33.6 mm closed. Video review showed the arm reaching near objects or moving
toward the basket without establishing a stable pickup. The dominant failure is
therefore Panda-to-RBY1 viewpoint, state, pre-grasp, and action-dynamics
transfer—not incorrect jaw shape or absent gripper actuation.

### Panda Pro exposes spatial and task-binding shortcuts

- **Object appearance, 10/10:** appearance perturbation did not hurt this slice.
- **Language, 9/10:** paraphrased instructions mostly transferred. The single
  failure grasped the correct bottle and carried it over the basket but never
  satisfied the placement predicate.
- **Swap, 2/10:** a representative failure picked the swapped red/green can and
  deposited it in the basket while leaving the requested bottle untouched.
  This is consistent with a positional shortcut.
- **Task, 1/10:** a representative task requested tomato sauce, while the policy
  picked the green salad-dressing bottle associated with the original task
  index. This indicates original scene/task binding overriding the perturbed
  instruction.

The dominant Panda Pro weakness in this diagnostic is spatial and task
rebinding, not generic object appearance or language paraphrase robustness.

### The closer UR5 setup recovers real manipulation, but not Panda-level performance

Pi0.5-LIBERO improves from **0/50 on RBY1 to 6/50 on UR5**: 2/10 normal and
4/40 Pro. The successful normal tasks are milk (164 steps) and chocolate
pudding (161 steps). Pro successes are alphabet soup and salad dressing under
swap, and tomato sauce and BBQ sauce under task rebinding. The independently
scripted controller proof also grasps, lifts, transports, and releases salad
dressing in 210/280 actions; measured jaw gap changes from 111.7 mm open to
36.8 mm grasped and back to 112.7 mm released.

This supports the intended hypothesis: a conventional tabletop industrial arm
with a parallel-jaw Robotiq gripper is meaningfully easier for the LIBERO-tuned
policy than RBY1. It does **not** close the embodiment gap: Panda remains 32/50
in the same diagnostic while UR5 reaches 6/50. A representative normal UR5
failure stably grasps and deposits the red/green can instead of the requested
salad-dressing bottle, showing that the remaining failure can be object
grounding rather than gripper actuation. Pro object and language remain 0/10,
so appearance/paraphrase robustness observed on Panda does not survive the
combined visual and morphology shift in this slice.

## Limits

These results cover one initial state per task. They are useful for diagnosing
concrete failures, not for publication-grade success estimates. The next score
claim should use all 50 initial states per task, explicitly control or report
model sampling, and continue to report RBY1 separately from official Panda
LIBERO. UR5 is also non-official and must remain a separate cross-embodiment
diagnostic. The incomplete base-Pro UR5 slice must not be interpreted as a
full 0/40 result.
