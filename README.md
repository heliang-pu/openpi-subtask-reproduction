# openpi-subtask

This repository is an OpenPI fork for experimenting with subtask-aware pi0.5 policies.
The original OpenPI README is preserved in [README_OPENPI.md](README_OPENPI.md).

## What Changed From OpenPI

The upstream OpenPI policy predicts action chunks directly from a high-level task
prompt, images, and robot state:

```text
Task + observation -> actions
```

This fork adds a subtask language step for pi0.5:

```text
Task -> Subtask -> actions
```

The goal is to let the model first predict a short low-level stage such as
`grasp the yellow camera`, then condition action generation on both the original
task and the generated subtask.

## Main Additions

### 1. New pi0.5 Subtask Model Type

This fork adds `PI05_SUBTASK` and `Pi05SubtaskConfig`.

Key files:

- `src/openpi/models/model.py`
- `src/openpi/models/pi0_config.py`
- `src/openpi/models/pi0.py`

`Pi05Subtask` keeps the pi0.5 architecture but changes the training objective.
It adds a language-modeling cross-entropy loss on the subtask tokens while
keeping the original flow-matching action loss.

```text
total loss = subtask CE loss + action flow-matching loss
```

### 2. Subtask Prompt Tokenization

The tokenizer now supports a high-level task plus low-level subtask format:

```text
[BOS] Task: <high-level task>. Subtask: <low-level subtask> [EOS] [PAD...]
```

During training:

- `Task: ... Subtask: ` is treated as the prefix.
- Tokens after `Subtask:` are autoregressive.
- Cross-entropy loss is applied only to the subtask portion.

During inference:

```text
[BOS] Task: <high-level task>. Subtask: [PAD...]
```

The model autoregressively fills in the subtask.

Key files:

- `src/openpi/models/tokenizer.py`
- `src/openpi/transforms.py`

### 3. Two-Stage Inference

The policy can now run:

```text
1. generate_subtask(observation)
2. build_full_observation(observation, generated_subtask)
3. sample_actions(full_observation)
```

The generated subtask is also returned in the policy output as:

```text
generated_subtask
```

Subtask generation is cached by prompt in `Policy.infer()` so repeated calls for
the same high-level task do not regenerate the subtask every frame.

Key files:

- `src/openpi/models/pi0.py`
- `src/openpi/policies/policy.py`

### 4. Training and Smoke-Test Configs

This fork adds configs for subtask training and inference experiments.

Important configs:

- `pi05_subtask_libero`
- `pi05_subtask_libero_infer`
- `pi05_subtask_pickup_round1_50ep_lora`
- `debug_pi05_subtask`

The pickup LoRA config is intended as a small smoke-test style setup for
checking that subtask generation can be trained and visualized before scaling up.

Key file:

- `src/openpi/training/config.py`

### 5. Subtask Validation and Visualization Scripts

This fork adds helper scripts for checking subtask predictions on real dataset
samples and rendering visual summaries.

Scripts:

- `scripts/validate_subtask_on_dataset.py`
- `scripts/visualize_subtask_predictions.py`
- `scripts/visualize_subtask_episode_timeline.py`
- `scripts/visualize_subtask_prediction_video.py`

These scripts load a trained subtask checkpoint, run `generate_subtask()`, and
save text/image/video views of predicted subtasks.

## Dataset Format

The base dataset format is still LeRobot.

For standard OpenPI training, each frame or episode provides:

```text
image / video observations
state
actions
task_index -> meta/tasks.parquet
```

For true subtask training, the dataset should also provide:

```text
subtask_index -> meta/subtasks.parquet
```

Recommended structure:

```text
dataset/
  data/
    chunk-000/
      file-000.parquet          # includes subtask_index per frame
  meta/
    info.json
    stats.json
    tasks.parquet               # task_index -> task text
    subtasks.parquet            # subtask_index -> subtask text
    subtask_segments.csv        # optional human-readable segment summary
    episodes/
      chunk-000/
        file-000.parquet
  videos/
    ...
```

Example labels:

```text
Task: Pick up the yellow camera and put it in the box.

Subtasks:
  move the gripper to the yellow camera
  grasp the yellow camera
  move the yellow camera toward the box
  place the yellow camera in the box
  release the yellow camera and move away
```

## Current Important Caveat

The current `TokenizeSubtaskTraining` implementation still defaults to identity
subtask supervision:

```python
high_prompt = prompt
low_prompt = prompt
```

That is enough to smoke-test the model path, loss, checkpointing, and inference
scripts, but it is not yet full real subtask supervision.

To train on real subtask labels, add a data transform that maps:

```text
subtask_index -> meta/subtasks.parquet -> data["subtask"]
```

Then change `TokenizeSubtaskTraining` to consume:

```python
high_prompt = data.pop("prompt")
low_prompt = data.pop("subtask")
```

## Suggested Development Flow

1. Create or collect a small LeRobot dataset.
2. Add `subtask_index` and `meta/subtasks.parquet`.
3. Validate that the dataset loads and that each frame has both task and subtask labels.
4. Train a short LoRA smoke-test run with `Pi05SubtaskConfig`.
5. Run the visualization scripts to inspect generated subtasks.
6. Only then scale to more episodes or longer training.

For the current robot-pickup experiments, the intended direction is:

```text
Evo-RL data collection
-> value model / progress scoring
-> subtask annotation
-> OpenPI-subtask fine-tuning
-> rollout and visualization
```

## Relationship to Upstream OpenPI

This repository keeps the upstream OpenPI model and training stack as the base.
Most existing OpenPI configs, policies, and examples are still present. The main
fork-specific work is isolated around:

- subtask tokenizer paths
- `PI05_SUBTASK`
- subtask CE loss
- two-stage inference
- subtask smoke-test configs
- subtask prediction visualizers

For upstream installation, checkpoints, base model notes, and general OpenPI
usage, see [README_OPENPI.md](README_OPENPI.md).
