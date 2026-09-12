# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the SenseNova-Vision tokenizer vocab sizing.

The SenseNova-Vision-7B-MoT LLM has ``vocab_size = 152064`` (``llm_config.json``)
and its ``ema.safetensors`` embedding/head weights have exactly 152064 rows.
Every token id the model embeds must therefore be ``< 152064``.

The checkpoint's stock ``AutoTokenizer`` path renumbers the 2033 added tokens
(ids 149632-151664) to 151643-153675, pushing the four BAGEL control tokens past
the embedding rows.  ``vllm-omni`` ships ``VLLMSenseNovaVisionTokenizer`` (see
``vllm_omni/diffusion/models/sensenova_vision/tokenization_sensenova_vision.py``)
which preserves ids verbatim.  ``SenseNovaVisionPipeline`` now constructs it
**in-process** and injects it into the BAGEL core via the ``tokenizer`` kwarg, so
**no file in the checkpoint directory is ever created or modified**.

These tests are CPU-only and require no model weights: they load the tokenizer
directly from the checkpoint path (via ``VLLMSenseNovaVisionTokenizer.from_pretrained``,
the same call the pipeline uses) when the checkpoint is cached locally (or via
``SENSENOVA_VISION_MODEL_PATH``).  They are skipped when the checkpoint is
unavailable.
"""

from __future__ import annotations

import glob
import hashlib
import os
from pathlib import Path

import pytest

from vllm_omni.diffusion.models.sensenova_vision.tokenization_sensenova_vision import (
    VLLMSenseNovaVisionTokenizer,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]

# The LLM's embedding/head row count (``llm_config.json`` vocab_size).
EMBEDDING_SIZE = 152064

# The four control tokens BAGEL relies on (via ``BagelPipeline.add_special_tokens``).
_CONTROL_TOKENS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
}


def _cached_checkpoint() -> str | None:
    """Return the checkpoint root (HF cache snapshot) if it is cached locally.

    Falls back to ``SENSENOVA_VISION_MODEL_PATH`` (useful for CI / Modal runs
    that keep the checkpoint in a directory).
    """
    env_path = os.environ.get("SENSENOVA_VISION_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    snapshot = os.path.expanduser("~/.cache/huggingface/hub/models--sensenova--SenseNova-Vision-7B-MoT/snapshots/*")
    matches = sorted(glob.glob(snapshot))
    return matches[-1] if matches else None


@pytest.fixture()
def sensenova_vision_checkpoint() -> str:
    """The local SenseNova-Vision checkpoint root, skipped if unavailable."""
    checkpoint = _cached_checkpoint()
    if checkpoint is None:
        pytest.skip("SenseNova-Vision-7B-MoT not cached and SENSENOVA_VISION_MODEL_PATH is unset")
    return checkpoint


@pytest.fixture()
def tokenizer(sensenova_vision_checkpoint: str) -> VLLMSenseNovaVisionTokenizer:
    """The tokenizer exactly as ``SenseNovaVisionPipeline.__init__`` builds it."""
    return VLLMSenseNovaVisionTokenizer.from_pretrained(
        sensenova_vision_checkpoint,
        local_files_only=True,
        trust_remote_code=True,
    )


def test_tokenizer_is_custom_class(tokenizer: VLLMSenseNovaVisionTokenizer) -> None:
    """The tokenizer constructed by the pipeline is our id-preserving class."""
    assert type(tokenizer).__name__ == "VLLMSenseNovaVisionTokenizer"
    # `len` must be the true used id span (max_id + 1), not the key count.
    assert len(tokenizer) == 151665


def test_control_token_ids_are_in_vocab(tokenizer: VLLMSenseNovaVisionTokenizer) -> None:
    """The four BAGEL control tokens keep their checkpoint ids (< 152064)."""
    for token, expected_id in _CONTROL_TOKENS.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        assert actual_id == expected_id, f"{token} renumbered to {actual_id}"
        assert actual_id < EMBEDDING_SIZE, f"{token} id {actual_id} >= {EMBEDDING_SIZE} rows"


def test_control_ids_match_reference_order(tokenizer: VLLMSenseNovaVisionTokenizer) -> None:
    """bos/eos = 151644/151645 match the reference repo's appended order."""
    ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_start|>", "<|im_end|>")]
    assert ids == [151644, 151645]


def test_encode_renders_control_ids(tokenizer: VLLMSenseNovaVisionTokenizer) -> None:
    """Encoding a prompt renders the *verbatim* control ids (no renumbering)."""
    encoded = tokenizer("<|im_start|>Hello<|im_end|>", add_special_tokens=False)["input_ids"]
    assert encoded[0] == 151644, f"expected im_start first, got {encoded}"
    assert encoded[-1] == 151645, f"expected im_end last, got {encoded}"
    assert all(0 <= tid < EMBEDDING_SIZE for tid in encoded)


def test_non_special_added_tokens_stay_verbatim(tokenizer: VLLMSenseNovaVisionTokenizer) -> None:
    """Added tokens (special and not) keep ids within the 152064 rows."""
    for token in ("<|file_sep|>", "<100>", "<-999>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        assert tid < EMBEDDING_SIZE, f"{token} id {tid} >= {EMBEDDING_SIZE}"
        assert tid >= 149632, f"{token} id {tid} unexpectedly renumbered to {tid}"


def test_tokenizer_len_stays_within_vocab(tokenizer: VLLMSenseNovaVisionTokenizer) -> None:
    """len(tokenizer) must not push vocab_size above the 152064 rows."""
    assert len(tokenizer) <= EMBEDDING_SIZE, f"len {len(tokenizer)} > {EMBEDDING_SIZE}"


def test_vocab_size_matches_embedding_rows(tokenizer: VLLMSenseNovaVisionTokenizer) -> None:
    """max(vocab, len(tok), max_ctl+1) == 152064 matches the checkpoint weights."""
    inflate = max(
        EMBEDDING_SIZE,
        len(tokenizer),
        max(tokenizer.convert_tokens_to_ids(t) for t in _CONTROL_TOKENS) + 1,
    )
    assert inflate == EMBEDDING_SIZE, f"vocab inflated to {inflate}"


def test_load_does_not_modify_checkpoint_files(sensenova_vision_checkpoint: str) -> None:
    """Loading the tokenizer must not create or modify any checkpoint file.

    The pipeline no longer writes a patched ``tokenizer_config.json`` (nor a
    copied source file) anywhere; the load happens entirely in-process.
    """
    snap = Path(sensenova_vision_checkpoint)
    before_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.iterdir() if p.is_file()}
    before_symlinks = {p.name: os.readlink(p) for p in snap.iterdir() if p.is_symlink()}

    _ = VLLMSenseNovaVisionTokenizer.from_pretrained(str(snap), local_files_only=True, trust_remote_code=True)

    after_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.iterdir() if p.is_file()}
    after_symlinks = {p.name: os.readlink(p) for p in snap.iterdir() if p.is_symlink()}
    assert before_files == after_files, "checkpoint file(s) modified by tokenizer load"
    assert before_symlinks == after_symlinks, "checkpoint symlinks modified by tokenizer load"
