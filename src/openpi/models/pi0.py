import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# PaliGemma EOS token ID (same as in pi0_fast.py)
PALIGEMMA_EOS_TOKEN = 1


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.discrete_state_input = config.discrete_state_input # 是否使用离散状态输入
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            ) 
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))   # 第一个“动作序列”的起点为True，其他都是False（双向），意味着将后续的49个Action Token视为与第一个Token同一级的“并列块”
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _compute_subtask_ce_loss(
        self,
        prefix_out: at.Float[at.Array, "b s d"],
        observation: _model.Observation,
        num_image_tokens: int,
    ) -> at.Float[at.Array, " b"]:
        """Compute cross-entropy loss on the subtask portion of the prefix.

        Uses next-token prediction: hidden[i] predicts token[i+1].
        Loss is masked to only the subtask region via observation.token_loss_mask.

        Args:
            prefix_out: VLM hidden states [B, prefix_S, vlm_width].
            observation: Must have tokenized_prompt and token_loss_mask.
            num_image_tokens: Number of image tokens before the text tokens.

        Returns:
            Per-sample CE loss [B].
        """
        num_text = self.max_token_len
        text_hidden = prefix_out[:, num_image_tokens : num_image_tokens + num_text - 1, :]
        logits = self.PaliGemma.llm(text_hidden, method="decode_to_logits")

        targets = observation.tokenized_prompt[:, 1:]
        loss_mask = observation.token_loss_mask[:, 1:].astype(jnp.float32)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_nll = -jnp.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)

        masked_nll = token_nll * loss_mask
        return jnp.sum(masked_nll, axis=-1) / jnp.clip(jnp.sum(loss_mask, axis=-1), 1)


class Pi05Subtask(Pi0):
    """Pi0.5 with joint subtask language-modeling loss + flow-matching action loss.

    Architecture is identical to Pi0(pi05=True).  The only difference is
    compute_loss, which adds a cross-entropy term for subtask prediction.

    The per-token AR mask (observation.token_ar_mask) produced by
    TokenizeSubtaskTraining enforces *causal* attention on the subtask
    portion of the text tokens, while keeping bidirectional attention for
    the high-level prompt and image tokens.
    """

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, _ = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask_1d, adarms_cond = self.embed_suffix(observation, x_t, time)

        B = prefix_tokens.shape[0]
        prefix_S = prefix_tokens.shape[1]
        num_text_tokens = self.max_token_len
        num_image_tokens = prefix_S - num_text_tokens

        img_ar = jnp.zeros((B, num_image_tokens), dtype=jnp.bool_)
        if observation.token_ar_mask is not None:
            text_ar = observation.token_ar_mask.astype(jnp.bool_)
        else:
            text_ar = jnp.zeros((B, num_text_tokens), dtype=jnp.bool_)
        prefix_ar_mask = jnp.concatenate([img_ar, text_ar], axis=1)

        suffix_ar_mask = jnp.broadcast_to(
            suffix_ar_mask_1d[None, :], (B, suffix_tokens.shape[1])
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # Paper-faithful low-level policy: the action tokens condition on o_t and the
        # subtask l_hat only, NOT the high-level task l. Forbid the action suffix from
        # attending to the high-level task text columns. Presence-gated: no effect
        # when token_highlevel_mask is absent (base model / other configs / ablation).
        if observation.token_highlevel_mask is not None:
            hl_text = observation.token_highlevel_mask.astype(jnp.bool_)  # [B, num_text]
            hl_col = jnp.concatenate(
                [
                    jnp.zeros((B, num_image_tokens), dtype=jnp.bool_),
                    hl_text,
                    jnp.zeros((B, suffix_tokens.shape[1]), dtype=jnp.bool_),
                ],
                axis=1,
            )  # [B, S] — True over high-level task columns
            seq_len = input_mask.shape[1]
            is_action_row = (jnp.arange(seq_len) >= prefix_S)[None, :]  # [1, S]
            forbid = is_action_row[:, :, None] & hl_col[:, None, :]  # [B, S, S]
            attn_mask = attn_mask & ~forbid

        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        # --- flow-matching loss (same as Pi0) ---
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # --- subtask cross-entropy loss ---
        if observation.token_loss_mask is not None:
            subtask_loss = self._compute_subtask_ce_loss(prefix_out, observation, num_image_tokens)
        else:
            subtask_loss = jnp.zeros(B)

        return subtask_loss[:, None] + flow_loss

    def generate_subtask(
        self,
        observation: _model.Observation,
        *,
        max_tokens: int = 50,
    ) -> jnp.ndarray:
        """Stage 1 of pi0.5 inference: autoregressively generate subtask tokens.

        Uses greedy decoding (argmax). Runs in eager Python (not JIT) because
        the loop length is dynamic and KV cache grows each step.

        Args:
            observation: Batched observation with prefix-only tokenized_prompt.
            max_tokens: Maximum number of subtask tokens to generate.

        Returns:
            int32[B, gen_len] — generated token IDs including EOS.
        """
        tokens, _, _, _ = self._generate_subtask_with_cache(observation, max_tokens=max_tokens)
        return tokens

    def _generate_subtask_with_cache(
        self,
        observation: _model.Observation,
        *,
        max_tokens: int = 50,
    ) -> tuple[jnp.ndarray, _gemma.KVCache, jnp.ndarray, jnp.ndarray]:
        """Core of stage-1 inference, retaining the KV cache for stage-2 reuse.

        Autoregressively (greedy) decodes the subtask while building up a KV cache
        that covers ``[images][high-level prompt][subtask tokens]``. Returning the
        cache lets ``sample_actions_hierarchical`` predict the action chunk without
        re-encoding the images, and have the action expert attend to the subtask
        under the same *causal* encoding it was trained with.

        Returns:
            tokens:      int32[B, gen_len] — generated token IDs (including EOS).
            kv_cache:    KV cache over ``[images][prompt][subtask tokens]``.
            cache_mask:  bool[B, cache_len] — True for real tokens (False for the
                         padding inside the prompt). Used for suffix RoPE positions.
            action_cache_mask: bool[B, cache_len] — cache_mask with the high-level
                         task columns removed; what the action expert may attend to
                         (paper: pi(a | o_t, l_hat)). Equals cache_mask when no
                         token_highlevel_mask is present.
        """
        observation = jax.tree.map(jnp.asarray, observation)
        observation = _model.preprocess_observation(None, observation, train=False)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        B = prefix_tokens.shape[0]
        prefix_S = prefix_tokens.shape[1]

        # High-level (task) columns over the prefix — excluded from action attention.
        num_text = self.max_token_len
        num_image = prefix_S - num_text
        if observation.token_highlevel_mask is not None:
            hl_text = observation.token_highlevel_mask.astype(jnp.bool_)  # [B, num_text]
        else:
            hl_text = jnp.zeros((B, num_text), dtype=jnp.bool_)
        prefix_hl = jnp.concatenate([jnp.zeros((B, num_image), dtype=jnp.bool_), hl_text], axis=1)  # [B, prefix_S]

        # Find the actual last real token INDEX in the sequence (not count-based).
        # sum(mask) is wrong as an index when masked-out regions (e.g. unused camera)
        # sit before real text tokens.
        seq_indices = jnp.arange(prefix_S)[None, :]  # [1, S]
        last_pos = jnp.max(
            jnp.where(prefix_mask, seq_indices, -1), axis=1
        ).astype(jnp.int32)  # [B]
        last_hidden = prefix_out[jnp.arange(B), last_pos, :]  # [B, d]
        logits = self.PaliGemma.llm(
            last_hidden[:, None, :], method="decode_to_logits"
        )  # [B, 1, V]

        generated: list[jnp.ndarray] = []
        # Position encoding uses cumulative count (not sequence index).
        num_real = jnp.sum(prefix_mask, axis=1).astype(jnp.int32)  # [B]

        for i in range(max_tokens):
            next_token = jnp.argmax(logits[:, -1, :], axis=-1)  # [B]
            generated.append(next_token)

            # Append this token to the KV cache (incl. EOS, to match training where
            # the action expert attends to "...<subtask> [EOS]").
            next_emb = self.PaliGemma.llm(next_token[:, None], method="embed")  # [B,1,d]
            gen_count = i + 1
            gen_mask = jnp.ones((B, gen_count), dtype=jnp.bool_)
            full_mask = jnp.concatenate([prefix_mask, gen_mask], axis=1)  # [B, prefix_S+gen_count]
            attn_mask = full_mask[:, None, :]  # [B, 1, prefix_S+gen_count]
            new_positions = (num_real + i)[:, None]  # [B, 1]
            (new_out, _), kv_cache = self.PaliGemma.llm(
                [next_emb, None],
                mask=attn_mask,
                positions=new_positions,
                kv_cache=kv_cache,
            )

            if jnp.all(next_token == PALIGEMMA_EOS_TOKEN):
                break

            logits = self.PaliGemma.llm(new_out, method="decode_to_logits")  # [B, 1, V]

        if not generated:
            action_prefix_mask = prefix_mask & ~prefix_hl
            return jnp.zeros((B, 0), dtype=jnp.int32), kv_cache, prefix_mask, action_prefix_mask

        tokens = jnp.stack(generated, axis=1)  # [B, gen_len]
        gen_ones = jnp.ones((B, tokens.shape[1]), dtype=jnp.bool_)
        cache_mask = jnp.concatenate([prefix_mask, gen_ones], axis=1)
        # Action attends to o_t (images + state) + subtask, but not the task columns.
        highlevel_cache_mask = jnp.concatenate(
            [prefix_hl, jnp.zeros((B, tokens.shape[1]), dtype=jnp.bool_)], axis=1
        )
        action_cache_mask = cache_mask & ~highlevel_cache_mask
        return tokens, kv_cache, cache_mask, action_cache_mask

    def sample_actions_hierarchical(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_tokens: int = 50,
        num_steps: int = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, jnp.ndarray]:
        """Two-stage pi0.5 inference in a single pass (stage-1 cache reused by stage-2).

        Stage 1 predicts the subtask and keeps its KV cache. Stage 2 runs
        flow-matching for the action chunk while attending to that same cache, so
        (a) the images are encoded only once, and (b) the action expert reads the
        subtask under the *causal* encoding it was trained with — instead of
        re-encoding the full prompt bidirectionally as a plain prefix would.

        Runs eagerly (the subtask length, hence the cache length, is dynamic).

        Returns:
            actions:        the predicted action chunk.
            subtask_tokens: int32[B, gen_len] generated subtask tokens (incl. EOS).
        """
        observation = jax.tree.map(jnp.asarray, observation)
        subtask_tokens, kv_cache, cache_mask, action_cache_mask = self._generate_subtask_with_cache(
            observation, max_tokens=max_tokens
        )

        # Stage 2: flow matching, reusing the prefix + subtask KV cache.
        # pi05 embed_suffix only consumes the noisy actions + timestep, so the raw
        # observation (just used for the batch dim) is sufficient here.
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        real_count = jnp.sum(cache_mask, axis=1)  # [B]
        dt = -1.0 / num_steps
        x_t = noise
        time = 1.0
        for _ in range(num_steps):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(jnp.asarray(time, jnp.float32), batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # Action attends to o_t (images + state) + subtask, but NOT the high-level
            # task columns (action_cache_mask drops them; cache_mask keeps them for
            # RoPE positions, so the task still occupies sequence positions).
            prefix_attn_mask = einops.repeat(action_cache_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            # suffix positions follow the real cache tokens.
            positions = real_count[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t, subtask_tokens

    def build_full_observation(
        self,
        observation: _model.Observation,
        subtask_tokens: jnp.ndarray,
    ) -> _model.Observation:
        """Insert generated subtask tokens into the padded prompt for Stage 2.

        The original prompt is "Task: X. Subtask: [PAD...]". This method fills
        the padding region with the generated subtask tokens so that
        sample_actions sees the complete prompt.

        Args:
            observation: Original observation with prefix-only prompt.
            subtask_tokens: int32[B, gen_len] from generate_subtask.

        Returns:
            New Observation with updated tokenized_prompt and mask.
        """
        observation = jax.tree.map(jnp.asarray, observation)
        B, max_len = observation.tokenized_prompt.shape
        gen_len = subtask_tokens.shape[1]

        if gen_len == 0:
            return observation

        prefix_len = jnp.sum(
            observation.tokenized_prompt_mask, axis=1
        )  # [B]

        idx = jnp.arange(max_len)[None, :]  # [1, max_len]
        offset = idx - prefix_len[:, None]  # [B, max_len]
        in_gen = (offset >= 0) & (offset < gen_len)  # [B, max_len]
        offset_clamped = jnp.clip(offset, 0, gen_len - 1).astype(jnp.int32)
        gen_vals = subtask_tokens[
            jnp.arange(B)[:, None], offset_clamped
        ]  # [B, max_len]

        new_tokens = jnp.where(in_gen, gen_vals, observation.tokenized_prompt)
        new_mask = observation.tokenized_prompt_mask | in_gen

        return _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=new_tokens,
            tokenized_prompt_mask=new_mask,
        )
