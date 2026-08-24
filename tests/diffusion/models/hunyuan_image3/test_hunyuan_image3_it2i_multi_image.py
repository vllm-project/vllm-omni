# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-image input regression for HunyuanImage3 IT2I prompt construction.

The official HunyuanImage-3.0-Instruct supports up to 3 reference images
per IT2I request ("Multi-Image Fusion"; see hunyuan3.0_ins/README.md
section 200-216 + line 500). Each cond image becomes its own user-role
message and `apply_general_template` concatenates successive user
messages back-to-back inside ONE user_prefix/user_suffix wrap (see
hunyuan3.0_ins/tokenization_hunyuan_image_3.py:1399-1400, 1499-1515).
The lightweight `<img>` + `multi_modal_data` builder used by the example
flow must match that contract: N consecutive `<img>` placeholders sit
between `User: ` and the user prompt, with no separator between them.

This file pins:
  1. N consecutive `<img>` placeholders for N=1/2/3 across both the
     string builder (`build_prompt`) and the token builder
     (`build_prompt_tokens`).
  2. The N=1 path stays bit-identical to the legacy single-image builder
     (regression guard so default callers don't notice).
  3. N=2 / N=3 token sequences differ from N=1 by exactly (N-1) extra
     `<img>` ids inserted between `User: ` and `user_prompt`.
  4. Validation: N<1 and N>3 raise ValueError (hard cap N<=3 mirrors
     official upstream).
  5. Text-only tasks ignore `num_images` (no validation, no extra ids).
"""

from __future__ import annotations

import os

import pytest

from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import (
    MAX_IMAGES_PER_REQUEST,
    build_prompt,
    build_prompt_tokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeTokenizer:
    """Recording fake tokenizer mirroring the one in test_prompt_utils.

    Special token ids: `<|startoftext|>`=1, `<img>`=2, `<think>`=3,
    `<recaption>`=4. encode() returns one id per character starting at
    100, so substring-position assertions are stable.
    """

    SPECIAL = {
        "<|startoftext|>": 1,
        "<img>": 2,
        "<think>": 3,
        "<recaption>": 4,
    }

    def __init__(self) -> None:
        self.encode_calls: list[str] = []

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self.SPECIAL.get(tok, 0)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls.append(text)
        return list(range(100, 100 + len(text)))


_IMAGE_TASK_COMBOS = (
    ("i2t", None),
    ("it2i", "think"),
    ("it2i", "recaption"),
)
_TEXT_ONLY_TASK_COMBOS = (("t2t", None),)


# -------------------- string builder --------------------


@pytest.mark.parametrize("task,bot_task", _IMAGE_TASK_COMBOS)
@pytest.mark.parametrize("num_images", [1, 2, 3])
def test_build_prompt_emits_N_consecutive_img_placeholders(task: str, bot_task: str | None, num_images: int):
    """N=1/2/3 -> exactly N `<img>` substrings appear consecutively
    between `User: ` and the user prompt, with no separator between them."""
    s = build_prompt("HELLO", task=task, bot_task=bot_task, num_images=num_images)
    assert s.count("<img>") == num_images, (
        f"task={task} bot_task={bot_task} num_images={num_images}: expected {num_images} <img> "
        f"placeholders, found {s.count('<img>')} -- prompt was: {s!r}"
    )

    # All `<img>` placeholders must form one contiguous run "<img><img>..."
    # immediately after `User: ` and before HELLO.
    user_idx = s.index("User: ") + len("User: ")
    hello_idx = s.index("HELLO")
    between = s[user_idx:hello_idx]
    assert between == "<img>" * num_images, (
        f"region between `User: ` and prompt must be exactly N <img> placeholders; got {between!r}"
    )


def test_build_prompt_default_num_images_matches_legacy():
    """num_images default = 1 must produce a string bit-identical to the
    pre-multi-image behavior (single `<img>` placeholder)."""
    legacy = build_prompt("HELLO", task="it2i", bot_task="think")
    explicit = build_prompt("HELLO", task="it2i", bot_task="think", num_images=1)
    assert legacy == explicit, "default num_images=1 must match legacy single-image output"


# -------------------- token builder --------------------


@pytest.mark.parametrize("task,bot_task", _IMAGE_TASK_COMBOS)
def test_build_prompt_tokens_inserts_N_img_ids(task: str, bot_task: str | None):
    """N=1/2/3 -> the resulting id sequence contains exactly N copies of
    img_id (=2) sitting consecutively after the `User: ` segment."""
    tok = FakeTokenizer()
    ids_n1 = build_prompt_tokens("hi", tok, task=task, bot_task=bot_task, num_images=1).token_ids
    tok = FakeTokenizer()
    ids_n2 = build_prompt_tokens("hi", tok, task=task, bot_task=bot_task, num_images=2).token_ids
    tok = FakeTokenizer()
    ids_n3 = build_prompt_tokens("hi", tok, task=task, bot_task=bot_task, num_images=3).token_ids

    assert ids_n1.count(2) == 1
    assert ids_n2.count(2) == 2
    assert ids_n3.count(2) == 3

    # Each additional image must extend the sequence by exactly one img_id,
    # not shift other tokens around.
    assert len(ids_n2) == len(ids_n1) + 1
    assert len(ids_n3) == len(ids_n1) + 2

    # The img_ids must be CONSECUTIVE (no other token between successive
    # `<img>` placeholders -- mirrors the official `process_successive_message`
    # wrapping where successive user messages share one user_prefix/suffix).
    for ids, n in [(ids_n2, 2), (ids_n3, 3)]:
        first = ids.index(2)
        for k in range(n):
            assert ids[first + k] == 2, (
                f"img_ids must be consecutive starting at position {first} for n={n}; got {ids[first : first + n]!r}"
            )


def test_build_prompt_tokens_default_num_images_matches_legacy():
    """num_images default = 1 must produce the same id sequence as
    omitting the parameter (regression guard for existing single-image
    callers)."""
    tok_a = FakeTokenizer()
    legacy = build_prompt_tokens("hi", tok_a, task="it2i", bot_task="think").token_ids
    tok_b = FakeTokenizer()
    explicit = build_prompt_tokens("hi", tok_b, task="it2i", bot_task="think", num_images=1).token_ids
    assert legacy == explicit
    # Also: encode() must have been called on the same set of segments,
    # so segment boundaries are preserved.
    assert tok_a.encode_calls == tok_b.encode_calls


# -------------------- validation --------------------


@pytest.mark.parametrize("task,bot_task", _IMAGE_TASK_COMBOS)
@pytest.mark.parametrize("bad", [0, -1, MAX_IMAGES_PER_REQUEST + 1, 99])
def test_build_prompt_rejects_out_of_range_num_images(task: str, bot_task: str | None, bad: int):
    with pytest.raises(ValueError, match="num_images must be in"):
        build_prompt("hi", task=task, bot_task=bot_task, num_images=bad)
    with pytest.raises(ValueError, match="num_images must be in"):
        build_prompt_tokens("hi", FakeTokenizer(), task=task, bot_task=bot_task, num_images=bad)


@pytest.mark.parametrize("task,bot_task", _TEXT_ONLY_TASK_COMBOS)
@pytest.mark.parametrize("num_images", [0, 1, 2, 99])
def test_text_only_tasks_ignore_num_images(task: str, bot_task: str | None, num_images: int):
    """Validation only kicks in for image-input tasks; t2t et al. accept
    any num_images and emit zero `<img>` placeholders."""
    s = build_prompt("hi", task=task, bot_task=bot_task, num_images=num_images)
    assert "<img>" not in s
    ids = build_prompt_tokens("hi", FakeTokenizer(), task=task, bot_task=bot_task, num_images=num_images).token_ids
    assert 2 not in ids


# -------------------- real HF tokenizer regression --------------------

_HUNYUAN_MODEL_ID = "tencent/HunyuanImage-3.0-Instruct"


def _hf_cached(model_id: str) -> bool:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    snap_dir = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}", "snapshots")
    return os.path.isdir(snap_dir) and any(os.scandir(snap_dir))


@pytest.mark.skipif(not _hf_cached(_HUNYUAN_MODEL_ID), reason=f"{_HUNYUAN_MODEL_ID} tokenizer not in HF cache")
@pytest.mark.parametrize("num_images", [1, 2, 3])
def test_real_tokenizer_emits_n_consecutive_img_ids(num_images: int):
    """Real `AutoTokenizer.from_pretrained(...)` (the production path) must
    encode N=1/2/3 prompts to a sequence with exactly N consecutive `<img>`
    token-ids in the right place — proves the placeholder layout from
    `build_prompt_tokens` survives a real BPE tokenizer, not just FakeTokenizer.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_HUNYUAN_MODEL_ID, trust_remote_code=True)
    img_id = tok.convert_tokens_to_ids("<img>")
    assert img_id is not None and img_id >= 0, f"<img> not in tokenizer vocab; got id={img_id}"

    ids = build_prompt_tokens("hi", tok, task="it2i", bot_task="think", num_images=num_images).token_ids

    # Exactly N copies of <img> id, all consecutive.
    img_positions = [i for i, x in enumerate(ids) if x == img_id]
    assert len(img_positions) == num_images, (
        f"expected {num_images} <img> ids, got {len(img_positions)} at positions {img_positions}"
    )
    assert img_positions == list(range(img_positions[0], img_positions[0] + num_images)), (
        f"<img> ids must be contiguous; got positions {img_positions}"
    )


@pytest.mark.skipif(not _hf_cached(_HUNYUAN_MODEL_ID), reason=f"{_HUNYUAN_MODEL_ID} tokenizer not in HF cache")
def test_real_tokenizer_n_plus_one_extends_by_exactly_one_img_id():
    """Going from N to N+1 images must extend the encoded id sequence by
    exactly one extra `<img>` token-id and shift nothing else. Catches
    accidental separator tokens between successive `<img>` placeholders
    that a FakeTokenizer (deterministic encode) can't surface."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_HUNYUAN_MODEL_ID, trust_remote_code=True)
    img_id = tok.convert_tokens_to_ids("<img>")

    ids_n1 = build_prompt_tokens("hi", tok, task="it2i", bot_task="think", num_images=1).token_ids
    ids_n2 = build_prompt_tokens("hi", tok, task="it2i", bot_task="think", num_images=2).token_ids
    ids_n3 = build_prompt_tokens("hi", tok, task="it2i", bot_task="think", num_images=3).token_ids

    assert len(ids_n2) == len(ids_n1) + 1, f"N=2 should be N=1 + 1 token; got {len(ids_n2)} vs {len(ids_n1)}"
    assert len(ids_n3) == len(ids_n1) + 2, f"N=3 should be N=1 + 2 tokens; got {len(ids_n3)} vs {len(ids_n1)}"

    # Insert one img_id at the existing position; everything else unchanged.
    p1 = ids_n1.index(img_id)
    assert ids_n2[: p1 + 1] == ids_n1[: p1 + 1] + [], "prefix before extra <img> must match N=1"
    assert ids_n2[p1] == img_id and ids_n2[p1 + 1] == img_id, "two consecutive <img> ids at the insertion point"
    assert ids_n2[p1 + 2 :] == ids_n1[p1 + 1 :], "tail after the extra <img> must match N=1's tail"
    # N=3 same pattern, three in a row.
    assert ids_n3[p1 : p1 + 3] == [img_id, img_id, img_id]
    assert ids_n3[p1 + 3 :] == ids_n1[p1 + 1 :]


@pytest.mark.skipif(not _hf_cached(_HUNYUAN_MODEL_ID), reason=f"{_HUNYUAN_MODEL_ID} tokenizer not in HF cache")
def test_real_tokenizer_default_n1_byte_identical_to_legacy():
    """Default `num_images=1` must produce the exact same id sequence as
    omitting the parameter — pins the legacy single-image regression
    against the real tokenizer (not just FakeTokenizer)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_HUNYUAN_MODEL_ID, trust_remote_code=True)
    legacy = build_prompt_tokens("hi", tok, task="it2i", bot_task="think").token_ids
    explicit = build_prompt_tokens("hi", tok, task="it2i", bot_task="think", num_images=1).token_ids
    assert legacy == explicit, "real tokenizer: default num_images=1 must be byte-identical to legacy"
