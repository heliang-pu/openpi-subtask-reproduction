import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


class _RecordingSubtaskTokenizer:
    def __init__(self):
        self.training_calls = []
        self.inference_calls = []

    def tokenize_high_low_prompt(self, high_prompt, low_prompt, state=None):
        self.training_calls.append((high_prompt, low_prompt, state))
        return (
            np.asarray([1, 2, 3], dtype=np.int32),
            np.asarray([True, True, True]),
            np.asarray([0, 1, 1], dtype=np.int32),
            np.asarray([False, True, True]),
        )

    def tokenize_high_level_prefix(self, high_prompt, state=None):
        self.inference_calls.append((high_prompt, state))
        return np.asarray([1, 2, 0], dtype=np.int32), np.asarray([True, True, False])

    def highlevel_task_token_len(self, high_prompt, state=None):
        # Pretend the first token is the high-level task region.
        return 1


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_tokenize_subtask_training_uses_subtask_and_state():
    tokenizer = _RecordingSubtaskTokenizer()
    transform = _transforms.TokenizeSubtaskTraining(tokenizer, discrete_state_input=True)
    state = np.asarray([0.1, -0.2], dtype=np.float32)

    data = transform({"prompt": "Pick up object", "subtask": "move to object", "state": state})

    high_prompt, low_prompt, recorded_state = tokenizer.training_calls[0]
    assert high_prompt == "Pick up object"
    assert low_prompt == "move to object"
    assert recorded_state is state
    assert "subtask" not in data
    assert np.all(data["tokenized_prompt"] == np.asarray([1, 2, 3]))
    assert np.all(data["token_ar_mask"] == np.asarray([0, 1, 1]))
    assert np.all(data["token_loss_mask"] == np.asarray([False, True, True]))
    assert np.all(data["token_highlevel_mask"] == np.asarray([True, False, False]))


def test_tokenize_subtask_training_requires_state_when_enabled():
    tokenizer = _RecordingSubtaskTokenizer()
    transform = _transforms.TokenizeSubtaskTraining(tokenizer, discrete_state_input=True)

    with pytest.raises(ValueError, match="State is required"):
        transform({"prompt": "Pick up object", "subtask": "move to object"})


def test_tokenize_subtask_inference_uses_state():
    tokenizer = _RecordingSubtaskTokenizer()
    transform = _transforms.TokenizeSubtaskInference(tokenizer, discrete_state_input=True)
    state = np.asarray([0.1, -0.2], dtype=np.float32)

    data = transform({"prompt": "Pick up object", "state": state})

    high_prompt, recorded_state = tokenizer.inference_calls[0]
    assert high_prompt == "Pick up object"
    assert recorded_state is state
    assert np.all(data["tokenized_prompt"] == np.asarray([1, 2, 0]))
    assert np.all(data["tokenized_prompt_mask"] == np.asarray([True, True, False]))
    assert np.all(data["token_highlevel_mask"] == np.asarray([True, False, False]))


def test_tokenize_subtask_inference_requires_state_when_enabled():
    tokenizer = _RecordingSubtaskTokenizer()
    transform = _transforms.TokenizeSubtaskInference(tokenizer, discrete_state_input=True)

    with pytest.raises(ValueError, match="State is required"):
        transform({"prompt": "Pick up object"})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_extract_prompt_from_lerobot_v3_task():
    transform = _transforms.PromptFromLeRobotTask({})

    data = transform({"task": "Pick the cup", "task_index": 3})

    assert data["prompt"] == "Pick the cup"
    assert "task" not in data


def test_extract_subtask_from_lerobot_v3_subtask_index():
    transform = _transforms.SubtaskFromLeRobotSubtask({2: "Grasp the cup"})

    data = transform({"subtask_index": 2})

    assert data["subtask"] == "Grasp the cup"

    with pytest.raises(ValueError, match="subtask_index=3 not found in subtask mapping"):
        transform({"subtask_index": 3})


def test_extract_subtask_ignores_unannotated_index():
    transform = _transforms.SubtaskFromLeRobotSubtask({2: "Grasp the cup"})

    data = {"subtask_index": -1}

    assert transform(data) is data
    assert "subtask" not in data
