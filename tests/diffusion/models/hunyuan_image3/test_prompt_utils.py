# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for HunyuanImage3 prompt construction (PR #3243).

Two layers:
  1. Pure-logic tests with a recording fake tokenizer -- protect the
     prompt template structure (BOS, User:/Assistant: framing, trigger
     placement, image placeholder position) and protect the segment-
     by-segment tokenization contract (each segment must hit
     `tokenizer.encode` in isolation).
  2. Real-tokenizer regression -- run when the HunyuanImage3-Instruct
     tokenizer is in the local HF cache. Asserts the segment-tokenized
     output diverges from the naive full-string encode, which is the
     bug-tripping fixture for the cross-segment BPE merge fix
     (commit 7bd429ed).
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS,
    available_tasks,
    build_prompt,
    build_prompt_tokens,
    resolve_stop_token_ids,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# -------------------- Pure-logic structural tests --------------------


class FakeTokenizer:
    """Minimal tokenizer stub that records every encode() call.

    Returns deterministic ids from convert_tokens_to_ids while
    encode() returns one id per character starting at 100. This lets
    tests both verify segmentation (by inspecting `encode_calls`) and
    locate substrings inside the returned id list.
    """

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
    tasks = set(available_tasks())
    assert tasks >= {
        "t2t",
        "i2t",
        "it2i_think",
        "it2i_recaption",
        "it2i_think_recaption",
        "t2i_think",
        "t2i_recaption",
        "t2i_vanilla",
    }


def test_resolve_stop_token_ids_uses_answer_for_generation_tasks():
    tok = FakeTokenizer()

    answer_id = HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS["<answer>"]
    assert resolve_stop_token_ids(task="t2i_think", tokenizer=tok) == [answer_id]
    assert resolve_stop_token_ids(task="t2i_recaption", tokenizer=tok) == [answer_id]


@pytest.mark.parametrize(
    "task",
    [
        "t2t",
        "i2t",
        "it2i_think",
        "it2i_recaption",
        "it2i_think_recaption",
        "t2i_think",
        "t2i_recaption",
    ],
)
def test_build_prompt_string_structure_chat_template(task: str):
    """Chat-template tasks must produce <|startoftext|>...User: ...Assistant: ...
    with image placeholder (when applicable) and trigger tag AFTER `Assistant: `."""
    s = build_prompt("HELLO", task=task)

    assert s.startswith("<|startoftext|>")
    assert "User: " in s
    assert "Assistant: " in s
    assert s.index("User: ") < s.index("HELLO") < s.index("Assistant: ")

    if task.startswith(("i2t", "it2i")):
        assert s.index("User: ") < s.index("<img>") < s.index("HELLO"), (
            "<img> placeholder must sit between `User: ` and the user prompt"
        )
    else:
        assert "<img>" not in s

    # Trigger tag must be the FINAL token of the prompt (after `Assistant: `).
    # Note: the system prompt itself mentions <think>/<recaption> as mode
    # documentation, so substring index() catches the wrong occurrence -- use
    # endswith() which directly captures "trigger is at the tail" (the Part A
    # fix: trigger goes AFTER `Assistant: `, not before user_prompt).
    if task in ("it2i_think", "t2i_think", "it2i_think_recaption"):
        assert s.endswith("Assistant: <think>"), (
            f"Trigger <think> must be appended right after `Assistant: ` (Part A fix). Got tail: ...{s[-40:]!r}"
        )
    if task in ("it2i_recaption", "t2i_recaption"):
        assert s.endswith("Assistant: <recaption>"), (
            f"Trigger <recaption> must be appended right after `Assistant: ` (Part A fix). Got tail: ...{s[-40:]!r}"
        )
    if task in ("t2t", "i2t"):
        assert s.endswith("Assistant: "), "Plain (no-trigger) task must end at `Assistant: ` with no trailing tag."


def test_build_prompt_vanilla_uses_pretrain_template():
    """t2i_vanilla is the only task that bypasses chat structure -- direct
    text->image generation driven by the vanilla system prompt."""
    s = build_prompt("HELLO", task="t2i_vanilla")
    assert s.startswith("<|startoftext|>")
    assert "User: " not in s
    assert "Assistant: " not in s
    assert "<think>" not in s
    assert "<recaption>" not in s
    assert s.endswith("HELLO")


def test_build_prompt_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        build_prompt("x", task="bogus")
    with pytest.raises(ValueError, match="Unknown task"):
        build_prompt_tokens("x", FakeTokenizer(), task="bogus")


def test_build_prompt_tokens_segments_each_boundary():
    """Regression for cross-segment BPE merge bug (commit 7bd429ed):
    each template segment must hit tokenizer.encode() independently;
    user_prompt MUST NOT be concatenated with the following separator
    in the same encode() call."""
    tok = FakeTokenizer()
    build_prompt_tokens("写诗。", tok, task="i2t")

    # Each canonical segment is encoded in its own call.
    assert "User: " in tok.encode_calls
    assert "写诗。" in tok.encode_calls, (
        "user_prompt must be encoded alone -- if it is concatenated with the "
        "trailing separator, BPE will merge across the boundary (the PR-#3243 bug)."
    )
    assert "\n\nAssistant: " in tok.encode_calls

    # No call must contain user_prompt glued to neighboring text.
    for call in tok.encode_calls:
        if call != "写诗。":
            assert "写诗。" not in call, f"user_prompt leaked into a multi-segment encode call: {call!r}"


def test_build_prompt_tokens_image_placeholder_present_for_image_tasks():
    tok = FakeTokenizer()
    result = build_prompt_tokens("hi", tok, task="i2t")
    ids = result.token_ids
    assert ids[0] == FakeTokenizer.SPECIAL["<|startoftext|>"], "BOS (<|startoftext|>) must be the first token"
    assert FakeTokenizer.SPECIAL["<img>"] in ids, "<img> placeholder must be present for i2t/it2i tasks"


def test_build_prompt_tokens_no_image_for_text_only_tasks():
    tok = FakeTokenizer()
    result = build_prompt_tokens("hi", tok, task="t2t")
    ids = result.token_ids
    assert FakeTokenizer.SPECIAL["<img>"] not in ids, "<img> must NOT appear for text-only tasks"


@pytest.mark.parametrize(
    "task,trigger_id",
    [
        ("it2i_think", FakeTokenizer.SPECIAL["<think>"]),
        ("t2i_think", FakeTokenizer.SPECIAL["<think>"]),
        ("it2i_recaption", FakeTokenizer.SPECIAL["<recaption>"]),
        ("t2i_recaption", FakeTokenizer.SPECIAL["<recaption>"]),
    ],
)
def test_build_prompt_tokens_trigger_is_last_token(task: str, trigger_id: int):
    """Trigger tag id must be the LAST token (after `Assistant: ` segment)."""
    tok = FakeTokenizer()
    result = build_prompt_tokens("hi", tok, task=task)
    ids = result.token_ids
    assert ids[-1] == trigger_id


def test_build_prompt_tokens_no_trigger_for_plain_tasks():
    """Tasks without trigger_tag (t2t / i2t) must NOT append a trigger id."""
    tok = FakeTokenizer()
    result = build_prompt_tokens("hi", tok, task="t2t")
    ids = result.token_ids
    assert ids[-1] not in {
        FakeTokenizer.SPECIAL["<think>"],
        FakeTokenizer.SPECIAL["<recaption>"],
    }


# -------------------- end2end.py wiring guard --------------------


def _repo_root() -> pathlib.Path:
    # tests/diffusion/models/hunyuan_image3/test_prompt_utils.py -> repo root
    return pathlib.Path(__file__).resolve().parents[4]


def test_end2end_routes_through_shared_prompt_utils():
    """Regression for the *delivery vector* of PR #3243.

    Background: the wrong-template bug that PR #3243 fixes was introduced
    when end2end.py grew its own hand-rolled prompt builder that diverged
    from the canonical instruct chat template. To prevent that exact
    failure mode from recurring, end2end.py MUST:
      1. Import the prompt builders from the shared prompt_utils module.
      2. NOT redefine `build_prompt` or `build_prompt_tokens` locally.

    A local redefinition is precisely how a future merge can silently
    re-introduce a pretrain-style template (trigger BEFORE user_prompt,
    no User:/Assistant: framing, etc.) without touching prompt_utils,
    bypassing every other test in this file.
    """
    end2end_path = _repo_root() / "examples" / "offline_inference" / "hunyuan_image3" / "end2end.py"
    assert end2end_path.is_file(), f"end2end.py not found at {end2end_path}"

    tree = ast.parse(end2end_path.read_text(encoding="utf-8"))

    local_func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    forbidden = {"build_prompt", "build_prompt_tokens"}
    redefined = local_func_names & forbidden
    assert not redefined, (
        f"end2end.py defines {sorted(redefined)} locally. This is exactly how "
        "the wrong prompt template re-entered the example before PR #3243. "
        "Use the shared `vllm_omni.diffusion.models.hunyuan_image3.prompt_utils` "
        "helpers instead."
    )

    imported_from_prompt_utils: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("hunyuan_image3.prompt_utils"):
            imported_from_prompt_utils.update(alias.name for alias in node.names)
    expected_imports = {
        "_TASK_PRESETS",
        "build_prompt_tokens",
        "resolve_stop_token_ids",
    }
    assert expected_imports <= imported_from_prompt_utils, (
        "end2end.py must import the HunyuanImage3 prompt and stop-token helpers from "
        "vllm_omni.diffusion.models.hunyuan_image3.prompt_utils -- the shared "
        "module is the single source of truth for the AR-prefill template and "
        "bot_task-derived AR stop token ids."
    )


# -------------------- Real-tokenizer regression --------------------


_HUNYUAN_MODEL_ID = "tencent/HunyuanImage-3.0-Instruct"


def _hf_cached(model_id: str) -> bool:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    snap_dir = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}", "snapshots")
    return os.path.isdir(snap_dir) and any(os.scandir(snap_dir))


@pytest.mark.skipif(
    not _hf_cached(_HUNYUAN_MODEL_ID),
    reason=f"{_HUNYUAN_MODEL_ID} tokenizer not in HF cache",
)
def test_segment_tokenize_diverges_from_full_string_encode():
    """Regression for PR #3243 segment-tokenization fix.

    The naive `tokenizer.encode(build_prompt(...))` lets BPE merge tokens
    across segment boundaries (notably `。\\n\\n` -> a single id), which
    drifts the AR prefill away from HF's apply_chat_template output. The
    segment-by-segment build_prompt_tokens must produce a STRICTLY
    DIFFERENT id sequence on a prompt that triggers the merge.

    If someone "simplifies" build_prompt_tokens to call encode() on the
    full string, this assertion fires.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_HUNYUAN_MODEL_ID, trust_remote_code=True)

    user_prompt = "写一首关于夜的诗。"
    result = build_prompt_tokens(user_prompt, tok, task="i2t")
    seg_ids = result.token_ids
    full_ids = tok.encode(build_prompt(user_prompt, task="i2t"), add_special_tokens=False)

    assert seg_ids != full_ids, (
        "build_prompt_tokens output equals naive full-string encode -- "
        "the BPE-merge-bypass behavior is no longer exercised. This means "
        "the segment-by-segment fix from PR #3243 has been silently undone."
    )

    # Segmenting prevents merges, so the segment id list should have AT LEAST
    # as many tokens as the merged version (a merge consumes 2+ ids -> 1).
    assert len(seg_ids) >= len(full_ids), (
        f"segment-encoded length ({len(seg_ids)}) shorter than full-string "
        f"merged length ({len(full_ids)}) -- impossible if segmentation is "
        f"genuinely bypassing merges."
    )
