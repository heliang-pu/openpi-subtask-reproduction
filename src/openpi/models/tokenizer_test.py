import numpy as np

from openpi.models import tokenizer as _tokenizer


class _FakeSentencePiece:
    def __init__(self):
        self.encoded_texts = []

    def encode(self, text: str, add_bos: bool = False):
        self.encoded_texts.append(text)
        tokens = [ord(char) for char in text]
        if add_bos:
            return [2, *tokens]
        return tokens


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_tokenize_subtask_prefix_includes_discretized_state():
    tokenizer = object.__new__(_tokenizer.PaligemmaTokenizer)
    tokenizer._max_len = 128
    tokenizer._tokenizer = _FakeSentencePiece()

    state = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    tokens, mask = tokenizer.tokenize_high_level_prefix("Pick_up_block", state=state)

    encoded_prefix = tokenizer._tokenizer.encoded_texts[0]
    assert encoded_prefix == "Task: pick up block, State: 0 128 255;\nSubtask: "
    assert tokens.shape == (128,)
    assert mask.sum() == len(encoded_prefix) + 1


def test_tokenize_subtask_training_includes_discretized_state_before_subtask_target():
    tokenizer = object.__new__(_tokenizer.PaligemmaTokenizer)
    tokenizer._max_len = 128
    tokenizer._tokenizer = _FakeSentencePiece()

    state = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    tokens, _mask, _ar_mask, loss_mask = tokenizer.tokenize_high_low_prompt(
        "Pick_up_block", "move to block", state=state
    )

    encoded_prefix, encoded_suffix = tokenizer._tokenizer.encoded_texts
    assert encoded_prefix == "Task: pick up block, State: 0 128 255;\nSubtask: "
    assert encoded_suffix == "move to block"
    assert tokens.shape == (128,)
    assert not loss_mask[: len(encoded_prefix) + 1].any()
    assert loss_mask[len(encoded_prefix) + 1 : len(encoded_prefix) + 1 + len(encoded_suffix) + 1].all()


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)
