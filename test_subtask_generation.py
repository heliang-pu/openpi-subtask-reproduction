"""Minimal script to load a pi0.5 subtask checkpoint and print the generated subtask.

Usage:
    cd /storages/liweile/Robo/openpi
    uv run python scripts/test_subtask_generation.py \
        --checkpoint_dir /storages/liweile/Robo/openpi/checkpoints/pi05_subtask_libero/subtask-libero-0221/29999 \
        --prompt "pick up the black bowl on the stove and place it on the plate"
"""

import argparse
import logging
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test pi0.5 subtask generation")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/storages/liweile/Robo/openpi/checkpoints/pi05_subtask_libero/subtask-libero-0221/29999",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="pick up the black bowl on the stove and place it on the plate",
    )
    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()

    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    from openpi.models import model as _model
    from openpi.models import pi0_config
    from openpi.models import tokenizer as _tokenizer

    config = pi0_config.Pi05SubtaskConfig(action_horizon=10, discrete_state_input=True)

    logger.info(f"Loading params from {checkpoint_dir / 'params'}...")
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)

    logger.info("Creating model and loading weights...")
    model = config.load(params)
    logger.info("Model loaded successfully.")

    logger.info(f"Tokenizing prompt: '{args.prompt}'")
    tok = _tokenizer.PaligemmaTokenizer(max_len=config.max_token_len)
    dummy_state = np.zeros((1, config.action_dim), dtype=np.float32)
    prefix_tokens, prefix_mask = tok.tokenize_high_level_prefix(args.prompt, state=dummy_state[0])

    logger.info(f"  prefix_tokens shape: {prefix_tokens.shape}, non-pad count: {prefix_mask.sum()}")
    logger.info(f"  prefix text (decoded): {tok.detokenize(prefix_tokens)}")

    B = 1
    dummy_images = {
        "base_0_rgb": np.zeros((B, 224, 224, 3), dtype=np.float32),
        "left_wrist_0_rgb": np.zeros((B, 224, 224, 3), dtype=np.float32),
        "right_wrist_0_rgb": np.zeros((B, 224, 224, 3), dtype=np.float32),
    }
    dummy_image_masks = {
        "base_0_rgb": np.array([True]),
        "left_wrist_0_rgb": np.array([True]),
        "right_wrist_0_rgb": np.array([False]),
    }
    observation = _model.Observation(
        images=dummy_images,
        image_masks=dummy_image_masks,
        state=dummy_state,
        tokenized_prompt=prefix_tokens[None, ...],
        tokenized_prompt_mask=prefix_mask[None, ...],
    )

    logger.info(f"Running subtask generation (max_tokens={args.max_tokens})...")
    subtask_tokens = model.generate_subtask(observation, max_tokens=args.max_tokens)
    subtask_tokens_np = np.asarray(subtask_tokens[0])

    logger.info(f"Generated {subtask_tokens.shape[1]} tokens: {subtask_tokens_np.tolist()}")

    subtask_text = tok.detokenize(subtask_tokens_np)

    print("\n" + "=" * 60)
    print(f"Prompt:           {args.prompt}")
    print(f"Generated Subtask: {subtask_text}")
    print(f"Token count:       {subtask_tokens.shape[1]}")
    print(f"Token IDs:         {subtask_tokens_np.tolist()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
