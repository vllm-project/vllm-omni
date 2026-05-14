# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import ast
import os
import pathlib

import pytest

from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    _TASK_PRESETS,
    HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS,
    available_bot_tasks,
    available_tasks,
    build_prompt,
    build_prompt_tokens,
    resolve_stop_token_ids,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeTokenizer:
    SPECIAL = {
        "<|startoftext|>": 1,
        "<img>": 2,
        "<think>": 3,
        "<recaption>": 4,
        "<|endoftext|>": 5,
        "</recaption>": 6,
        "</answer>": 7,
        "<boi>": 8,
        "</think>": 9,
        **{f"<img_ratio_{i}>": 1000 + i for i in range(33)},
    }

    def __init__(self) -> None:
        self.encode_calls: list[str] = []
        self.eos_token_id = self.SPECIAL["<|endoftext|>"]

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self.SPECIAL.get(tok, 0)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls.append(text)
        return list(range(100, 100 + len(text)))


def test_available_tasks_covers_all_modalities():
    assert set(available_tasks()) == {"t2t", "i2t", "it2i", "t2i"}


def test_available_bot_tasks_covers_all_modes():
    assert set(available_bot_tasks()) == {None, "think", "recaption", "think_recaption", "vanilla"}


def test_legacy_task_presets_still_available():
    assert {
        "it2i_think",
        "it2i_recaption",
        "it2i_think_recaption",
        "t2i_think",
        "t2i_recaption",
        "t2i_vanilla",
    } <= set(_TASK_PRESETS)


def test_legacy_base_task_omitted_bot_task_keeps_plain_mode():
    prompt = build_prompt("HELLO", task="i2t")
    assert prompt.endswith("Assistant: ")
    assert not prompt.endswith("<think>")

    result = build_prompt_tokens("hi", FakeTokenizer(), task="i2t")
    assert result.system_prompt_type == "en_unified"
    assert result.token_ids[-1] not in {
        FakeTokenizer.SPECIAL["<think>"],
        FakeTokenizer.SPECIAL["<recaption>"],
    }


def test_legacy_composite_task_with_none_bot_task_keeps_encoded_mode():
    prompt = build_prompt("HELLO", task="it2i_think", bot_task=None)
    assert prompt.endswith("Assistant: <think>")

    result = build_prompt_tokens("hi", FakeTokenizer(), task="it2i_recaption", bot_task=None)
    assert result.token_ids[-1] == FakeTokenizer.SPECIAL["<recaption>"]


def test_default_prompt_still_uses_it2i_think_mode():
    prompt = build_prompt("HELLO")
    assert prompt.endswith("Assistant: <think>")

    result = build_prompt_tokens("hi", FakeTokenizer())
    assert result.system_prompt_type == "en_unified"
    assert result.token_ids[-1] == FakeTokenizer.SPECIAL["<think>"]


def test_resolve_stop_token_ids_image_tasks_stop_on_ratio_range():
    """Image-output tasks stop on any ``<img_ratio_*>`` token.

    Mirrors upstream ``modeling_hunyuan_image_3.py::generate_image``
    (line 3289-3303): when ``need_ratio`` is true,
    ``final_stop_tokens = list(range(start_ratio, end_ratio + 1)) +
    ratio_token_other_slices``. AR stops AT the ratio token sampled
    after ``<img_size_*>``; the bridge then strips the trailing ratio
    token before passing the cot to DiT.
    """
    tok = FakeTokenizer()

    start = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_0>"]
    end = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_32>"]
    other_start = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_33>"]
    other_end = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<img_ratio_36>"]
    expected = list(range(start, end + 1)) + list(range(other_start, other_end + 1))

    # Image-output: t2i / it2i stop on the full ratio token range.
    for bot in ("think", "recaption", "think_recaption", "vanilla"):
        assert resolve_stop_token_ids(task="t2i", bot_task=bot, tokenizer=tok) == expected
        assert resolve_stop_token_ids(task="it2i", bot_task=bot, tokenizer=tok) == expected

    # Text-output: i2t / t2t comprehension stops on <answer> (response sits inside).
    answer_id = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<answer>"]
    assert resolve_stop_token_ids(task="i2t", bot_task=None, tokenizer=tok) == [answer_id]
    assert resolve_stop_token_ids(task="t2t", bot_task=None, tokenizer=tok) == [answer_id]


@pytest.mark.parametrize(
    "task,bot_task",
    [
        ("t2t", None),
        ("i2t", None),
        ("it2i", "think"),
        ("it2i", "recaption"),
        ("it2i", "think_recaption"),
        ("t2i", "think"),
        ("t2i", "recaption"),
        ("t2i", "think_recaption"),
    ],
)
def test_build_prompt_string_structure_chat_template(task: str, bot_task: str | None):
    s = build_prompt("HELLO", task=task, bot_task=bot_task)
    assert s.startswith("<|startoftext|>")
    assert "User: " in s
    assert "Assistant: " in s
    assert s.index("User: ") < s.index("HELLO") < s.index("Assistant: ")

    if task in ("i2t", "it2i"):
        assert s.index("User: ") < s.index("<img>") < s.index("HELLO")
    else:
        assert "<img>" not in s

    if bot_task in ("think", "think_recaption"):
        assert s.endswith("Assistant: <think>")
    elif bot_task == "recaption":
        assert s.endswith("Assistant: <recaption>")
    elif bot_task is None:
        assert s.endswith("Assistant: ")


def test_build_prompt_vanilla_uses_pretrain_template():
    s = build_prompt("HELLO", task="t2i", bot_task="vanilla")
    assert s.startswith("<|startoftext|>")
    assert "User: " not in s
    assert "Assistant: " not in s
    assert s.endswith("HELLO")


def test_build_prompt_vanilla_rejects_non_t2i_task():
    with pytest.raises(ValueError, match="bot_task='vanilla'"):
        build_prompt("x", task="it2i", bot_task="vanilla")
    with pytest.raises(ValueError, match="bot_task='vanilla'"):
        build_prompt_tokens("x", FakeTokenizer(), task="i2t", bot_task="vanilla")


def test_build_prompt_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        build_prompt("x", task="bogus")
    with pytest.raises(ValueError, match="Unknown task"):
        build_prompt_tokens("x", FakeTokenizer(), task="bogus")


def test_build_prompt_unknown_bot_task_raises():
    with pytest.raises(ValueError, match="Unknown bot_task"):
        build_prompt("x", task="t2i", bot_task="bogus")
    with pytest.raises(ValueError, match="Unknown bot_task"):
        build_prompt_tokens("x", FakeTokenizer(), task="t2i", bot_task="bogus")


def test_build_prompt_tokens_segments_each_boundary():
    tok = FakeTokenizer()
    build_prompt_tokens("写诗。", tok, task="i2t", bot_task=None)
    assert "User: " in tok.encode_calls
    assert "写诗。" in tok.encode_calls
    assert "\n\nAssistant: " in tok.encode_calls
    for call in tok.encode_calls:
        if call != "写诗。":
            assert "写诗。" not in call


def test_build_prompt_tokens_image_placeholder_present_for_image_tasks():
    tok = FakeTokenizer()
    result = build_prompt_tokens("hi", tok, task="i2t", bot_task=None)
    ids = result.token_ids
    assert ids[0] == FakeTokenizer.SPECIAL["<|startoftext|>"]
    assert FakeTokenizer.SPECIAL["<img>"] in ids


def test_build_prompt_tokens_no_image_for_text_only_tasks():
    tok = FakeTokenizer()
    result = build_prompt_tokens("hi", tok, task="t2t", bot_task=None)
    ids = result.token_ids
    assert FakeTokenizer.SPECIAL["<img>"] not in ids


@pytest.mark.parametrize(
    "task,bot_task,trigger_id",
    [
        ("it2i", "think", FakeTokenizer.SPECIAL["<think>"]),
        ("t2i", "think", FakeTokenizer.SPECIAL["<think>"]),
        ("t2i", "think_recaption", FakeTokenizer.SPECIAL["<think>"]),
        ("it2i", "recaption", FakeTokenizer.SPECIAL["<recaption>"]),
        ("t2i", "recaption", FakeTokenizer.SPECIAL["<recaption>"]),
        ("it2i_think", None, FakeTokenizer.SPECIAL["<think>"]),
        ("it2i_recaption", None, FakeTokenizer.SPECIAL["<recaption>"]),
    ],
)
def test_build_prompt_tokens_trigger_is_last_token(task: str, bot_task: str | None, trigger_id: int):
    tok = FakeTokenizer()
    result = build_prompt_tokens("hi", tok, task=task, bot_task=bot_task)
    assert result.token_ids[-1] == trigger_id


def test_build_prompt_tokens_no_trigger_for_plain_tasks():
    tok = FakeTokenizer()
    result = build_prompt_tokens("hi", tok, task="t2t", bot_task=None)
    assert result.token_ids[-1] not in {
        FakeTokenizer.SPECIAL["<think>"],
        FakeTokenizer.SPECIAL["<recaption>"],
    }


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def test_end2end_routes_through_shared_prompt_utils():
    end2end_path = _repo_root() / "examples" / "offline_inference" / "hunyuan_image3" / "end2end.py"
    tree = ast.parse(end2end_path.read_text(encoding="utf-8"))

    local_func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert not (local_func_names & {"build_prompt", "build_prompt_tokens"})

    imported_from_prompt_utils: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("hunyuan_image3.prompt_utils"):
            imported_from_prompt_utils.update(alias.name for alias in node.names)
    expected_imports = {"build_prompt_tokens", "resolve_stop_token_ids", "resolve_sys_type"}
    assert expected_imports <= imported_from_prompt_utils


_HUNYUAN_MODEL_ID = "tencent/HunyuanImage-3.0-Instruct"


def _hf_cached(model_id: str) -> bool:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    snap_dir = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}", "snapshots")
    return os.path.isdir(snap_dir) and any(os.scandir(snap_dir))


@pytest.mark.skipif(not _hf_cached(_HUNYUAN_MODEL_ID), reason=f"{_HUNYUAN_MODEL_ID} tokenizer not in HF cache")
def test_segment_tokenize_diverges_from_full_string_encode():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_HUNYUAN_MODEL_ID, trust_remote_code=True)
    user_prompt = "写一首关于夜的诗。"
    result = build_prompt_tokens(user_prompt, tok, task="i2t", bot_task=None)
    seg_ids = result.token_ids
    full_ids = tok.encode(build_prompt(user_prompt, task="i2t", bot_task=None), add_special_tokens=False)
    assert seg_ids != full_ids
    assert len(seg_ids) >= len(full_ids)
