import collections
import dataclasses
import logging
import math
import os
import pathlib
import sys
import textwrap

# Set LIBERO_CONFIG_PATH and PYTHONPATH before importing libero.
_DEFAULT_LIBERO_CONFIG_PATH = pathlib.Path("/workspace/dataset/libero_config")
_DEFAULT_LIBERO_CONFIG_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LIBERO_CONFIG_PATH", str(_DEFAULT_LIBERO_CONFIG_PATH))
_LIBERO_ROOT_CANDIDATES = (
    pathlib.Path(__file__).resolve().parents[2] / "third_party/libero",
    pathlib.Path("/workspace/shared/openpi_jax/third_party/libero"),
)
_LIBERO_ROOT = next((path for path in _LIBERO_ROOT_CANDIDATES if (path / "libero/libero").exists()), None)
if _LIBERO_ROOT is not None:
    sys.path.insert(0, str(_LIBERO_ROOT))
    config_path = pathlib.Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    if not config_path.exists():
        benchmark_root = _LIBERO_ROOT / "libero/libero"
        config_path.write_text(
            "\n".join(
                [
                    f"benchmark_root: {benchmark_root}",
                    f"bddl_files: {benchmark_root / 'bddl_files'}",
                    f"init_states: {benchmark_root / 'init_files'}",
                    f"datasets: {_LIBERO_ROOT / 'libero/datasets'}",
                    f"assets: {benchmark_root / 'assets'}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import torch
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_compat


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    task_ids: str | None = None  # Comma-separated task ids to evaluate, e.g. "0,1,2,3,4".
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_steps_override: int | None = None  # Optional short-rollout cap for quick video generation.
    random_init: bool = False  # Use simulator reset states instead of LIBERO benchmark init states.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_all_videos: bool = False

    seed: int = 7  # Random Seed (for reproducibility)


def _parse_task_ids(task_ids: str | None, num_tasks: int) -> list[int]:
    if task_ids is None:
        return list(range(num_tasks))
    ids = [int(task_id) for task_id in task_ids.split(",") if task_id.strip()]
    invalid = [task_id for task_id in ids if task_id < 0 or task_id >= num_tasks]
    if invalid:
        raise ValueError(f"task_ids out of range for suite with {num_tasks} tasks: {invalid}")
    return ids


def _compose_replay_frame(
    image: np.ndarray,
    wrist_image: np.ndarray,
    *,
    task: str,
    subtask: str,
    step: int,
    done: bool,
) -> np.ndarray:
    image_pil = Image.fromarray(np.asarray(image)).convert("RGB")
    wrist_pil = Image.fromarray(np.asarray(wrist_image)).convert("RGB")
    panel_w = 420
    gap = 12
    pad = 12
    canvas_w = image_pil.width + wrist_pil.width + panel_w + gap * 2 + pad * 2
    canvas_h = max(image_pil.height, wrist_pil.height) + pad * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 247, 250))
    canvas.paste(image_pil, (pad, pad))
    canvas.paste(wrist_pil, (pad + image_pil.width + gap, pad))
    draw = ImageDraw.Draw(canvas, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = small_font = ImageFont.load_default()

    lines = [f"step {step} | {'success' if done else 'running'}"]
    lines.extend(textwrap.wrap("Task: " + task, width=48)[:3])
    if subtask:
        lines.extend(textwrap.wrap("Subtask: " + subtask, width=48)[:3])
    text_x = pad + image_pil.width + gap + wrist_pil.width + gap
    draw.text((pad, 2), "agentview", font=small_font, fill=(255, 255, 255, 220))
    draw.text((pad + image_pil.width + gap, 2), "wrist", font=small_font, fill=(255, 255, 255, 220))
    y = pad
    for i, line in enumerate(lines):
        fill = (16, 24, 39, 255)
        if line.startswith("Subtask:"):
            fill = (37, 99, 235, 255)
        draw.text((text_x, y), line, font=font if i == 0 else small_font, fill=fill)
        y += 22 if i == 0 else 18
    return np.asarray(canvas)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_ids = _parse_task_ids(args.task_ids, num_tasks_in_suite)
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Task ids: {task_ids}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    if args.max_steps_override is not None:
        max_steps = args.max_steps_override

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states unless explicitly using randomized simulator resets.
        initial_states = None if args.random_init else task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        saved_success_videos, saved_failure_videos = 0, 0
        max_success_videos, max_failure_videos = 3, 6
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            action_plan = collections.deque()

            if args.random_init:
                random_seed = int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
                env.seed(random_seed)
                obs = env.reset()
                logging.info(f"Using random simulator reset seed: {random_seed}")
            else:
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            current_subtask = ""

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(
                        _compose_replay_frame(
                            img,
                            wrist_img,
                            task=task_description,
                            subtask=current_subtask,
                            step=t,
                            done=False,
                        )
                    )

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        result = client.infer(element)
                        action_chunk = result["actions"]
                        current_subtask = str(result.get("generated_subtask", "")).strip()
                        if current_subtask:
                            logging.info(f"Generated subtask: {current_subtask}")
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        if replay_images:
                            replay_images[-1] = _compose_replay_frame(
                                img,
                                wrist_img,
                                task=task_description,
                                subtask=current_subtask,
                                step=t,
                                done=True,
                            )
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode (up to 3 successes and 6 failures per task)
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            should_save = args.save_all_videos or (done and saved_success_videos < max_success_videos) or (
                not done and saved_failure_videos < max_failure_videos
            )
            if should_save:
                vid_idx = saved_success_videos if done else saved_failure_videos
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_{vid_idx}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                if done:
                    saved_success_videos += 1
                else:
                    saved_failure_videos += 1

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        # Explicitly close environment to avoid EGL cleanup errors
        try:
            env.close()
        except Exception as e:
            logging.debug(f"Error closing environment (non-critical): {e}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
