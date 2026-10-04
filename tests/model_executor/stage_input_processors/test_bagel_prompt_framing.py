# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Bagel AR stage must frame text like the reference ``prepare_prompts``.

The reference tokenizes every text segment as ``[<|im_start|>] + encode(text) + [<|im_end|>]``;
a think-mode system prompt and the user text are separate segments around the image.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.stage_input_processors.bagel import (
    BOS,
    EOS,
    IMG2IMG_PLACEHOLDER,
    expand_cfg_prompts,
    expand_cfg_prompts_think,
    frame_prompt,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PROMPT = "a red vintage bicycle leaning against a white brick wall"
BOS_ID, EOS_ID, PLACEHOLDER_ID = 151644, 151645, 151660


def _fake_encode(text: str) -> list[int]:
    ids: list[int] = []
    specials = {BOS: BOS_ID, EOS: EOS_ID, IMG2IMG_PLACEHOLDER: PLACEHOLDER_ID}
    pos = 0
    while pos < len(text):
        for tok, tid in specials.items():
            if text.startswith(tok, pos):
                ids.append(tid)
                pos += len(tok)
                break
        else:
            end = min([text.find(t, pos) for t in specials if text.find(t, pos) != -1] + [len(text)])
            ids.extend(hash(w) % 100_000 for w in text[pos:end].split())
            pos = end
    return ids


def _reference_ids(text: str) -> list[int]:
    return [BOS_ID] + _fake_encode(text) + [EOS_ID]


def test_t2i_prompt_gets_the_reference_framing() -> None:
    framed = frame_prompt({"prompt": PROMPT, "modalities": ["image"]}, [])

    assert framed["prompt"] == f"{BOS}{PROMPT}{EOS}"
    assert _fake_encode(framed["prompt"]) == _reference_ids(PROMPT)


def test_framing_is_idempotent_and_keeps_img2img_placeholders_in_front() -> None:
    already = frame_prompt({"prompt": f"{BOS}{PROMPT}{EOS}", "modalities": ["image"]}, [])
    assert already["prompt"] == f"{BOS}{PROMPT}{EOS}"

    edit = frame_prompt({"prompt": f"{IMG2IMG_PLACEHOLDER}make the circle green", "modalities": ["img2img"]}, [])
    assert edit["prompt"] == f"{IMG2IMG_PLACEHOLDER}{BOS}make the circle green{EOS}"
    assert _fake_encode(edit["prompt"]) == [PLACEHOLDER_ID] + _reference_ids("make the circle green")


def test_text_only_and_string_prompts_are_left_alone() -> None:
    text_only = {"prompt": "hello", "modalities": ["text"]}
    assert frame_prompt(text_only, []) is text_only
    assert frame_prompt("plain", []) == "plain"


def test_prompt_dict_is_not_mutated() -> None:
    original = {"prompt": PROMPT, "modalities": ["image"], "negative_prompt": "blurry"}
    framed = frame_prompt(original, [])

    assert original["prompt"] == PROMPT
    assert framed["negative_prompt"] == "blurry"


@pytest.mark.parametrize("expand", [expand_cfg_prompts, expand_cfg_prompts_think])
def test_cfg_companions_are_framed_and_empty_negative_stays_empty(expand) -> None:
    prompt = {"prompt": PROMPT, "modalities": ["image"]}

    assert expand(prompt, SimpleNamespace(extra_args={"negative_prompt": ""})) == []
    (cfg_text,) = expand(prompt, SimpleNamespace(extra_args={"negative_prompt": "blurry"}))
    assert cfg_text.role == "cfg_text"
    assert cfg_text.prompt["prompt"] == f"{BOS}blurry{EOS}"


def test_img2img_companions_match_the_reference_contexts() -> None:
    prompt = {
        "prompt": f"{IMG2IMG_PLACEHOLDER}make the circle green",
        "modalities": ["img2img"],
        "multi_modal_data": {"img2img": [object()]},
    }
    cfg_text, cfg_img = expand_cfg_prompts(prompt, SimpleNamespace(extra_args={"negative_prompt": ""}))

    assert cfg_text.prompt["prompt"] == IMG2IMG_PLACEHOLDER
    assert cfg_img.prompt["prompt"] == f"{BOS}make the circle green{EOS}"
    assert "multi_modal_data" not in cfg_img.prompt
    assert cfg_text.prompt["multi_modal_data"] is prompt["multi_modal_data"]


SYSTEM = "think first"


def test_think_prompt_frames_the_system_prompt_and_the_text_separately() -> None:
    raw = frame_prompt({"prompt": f"{SYSTEM}{IMG2IMG_PLACEHOLDER}make it green", "modalities": ["img2img"]}, [])
    hand_framed = frame_prompt(
        {"prompt": f"{BOS}{SYSTEM}{EOS}{IMG2IMG_PLACEHOLDER}make it green", "modalities": ["img2img"]}, []
    )

    expected = f"{BOS}{SYSTEM}{EOS}{IMG2IMG_PLACEHOLDER}{BOS}make it green{EOS}"
    assert raw["prompt"] == expected
    assert hand_framed["prompt"] == expected
    assert _fake_encode(expected) == _reference_ids(SYSTEM) + [PLACEHOLDER_ID] + _reference_ids("make it green")


def test_think_img2img_companions_keep_the_system_segment() -> None:
    prompt = {"prompt": f"{SYSTEM}{IMG2IMG_PLACEHOLDER}make it green", "modalities": ["img2img"]}

    cfg_text, cfg_img = expand_cfg_prompts_think(prompt, SimpleNamespace(extra_args={"negative_prompt": "blurry"}))

    assert cfg_text.prompt["prompt"] == f"{BOS}{SYSTEM}{EOS}{IMG2IMG_PLACEHOLDER}{BOS}blurry{EOS}"
    assert cfg_img.prompt["prompt"] == f"{BOS}{SYSTEM}{EOS}{BOS}make it green{EOS}"
