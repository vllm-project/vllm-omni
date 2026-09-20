# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The talker embedding dtype must follow the engine dtype, not a bf16 literal.

``_embedding_dtype`` types every embedding the Qwen3-TTS talker hands back from
``preprocess`` / ``preprocess_decode_batch``, and ``OmniGPUModelRunner``
``index_copy_``s those into an ``inputs_embeds`` buffer allocated in the
engine's configured dtype. A hardcoded ``torch.bfloat16`` there kills every
request on a GPU with no bfloat16 support, where ``--dtype float16`` is the
only option (Turing / sm75 and older):

    RuntimeError: index_copy_(): self and source expected to have the same
    dtype, but got (self) Half and (source) BFloat16

Per-PR CI has no fp16-only GPU, so the invariant is pinned twice, CPU-only:

* :class:`Qwen3TTSPromptEmbedsBuilder.__init__` is cheap (pure attribute
  assignment, tokenizer loaded lazily), so it is called for real and asserted
  to follow its ``embedding_dtype`` argument — and to have no default;
* ``Qwen3TTSTalkerForConditionalGeneration.__init__`` is *not* cheap (it reads
  configs, the speech tokenizer and a feature extractor from disk), so it is
  checked statically: its AST must derive ``_embedding_dtype`` from the engine
  dtype rather than from a ``torch.<dtype>`` literal, and must pass that same
  name into the builder. Same source-level-contract approach as
  ``tests/worker/test_capture_replay_contract.py``.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    Qwen3TTSPromptEmbedsBuilder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TALKER_SRC = (
    pathlib.Path(__file__).resolve().parents[4]
    / "vllm_omni"
    / "model_executor"
    / "models"
    / "qwen3_tts"
    / "qwen3_tts_talker.py"
)


# -------------------- builder: the real __init__ --------------------


def _builder_kwargs() -> dict:
    """Minimal stub dependencies for the real builder constructor.

    ``__init__`` only stores these; nothing here is called during
    construction, so plain namespaces / lambdas are enough.
    """
    return dict(
        config=SimpleNamespace(),
        talker_config=SimpleNamespace(),
        model_path="",
        text_embedding=SimpleNamespace(),
        text_projection=lambda x: x,
        codec_embed=lambda ids: ids,
        residual_code_embeddings=lambda: [],
        speaker_encoder=None,
        tts_pad_embed=torch.zeros((1, 4)),
        encode_ref_audio_batch=lambda wavs, sr, *, device: [],
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["fp16", "bf16", "fp32"],
)
def test_builder_init_follows_embedding_dtype_argument(dtype):
    builder = Qwen3TTSPromptEmbedsBuilder(**_builder_kwargs(), embedding_dtype=dtype)

    assert builder._embedding_dtype is dtype, (
        "Qwen3TTSPromptEmbedsBuilder.__init__ no longer follows its embedding_dtype "
        "argument — prefill embeddings would disagree with the engine's inputs_embeds "
        "buffer under --dtype float16."
    )


def test_builder_init_requires_embedding_dtype():
    """No default: a bf16 default is exactly the footgun the argument removes."""
    with pytest.raises(TypeError, match="embedding_dtype"):
        Qwen3TTSPromptEmbedsBuilder(**_builder_kwargs())


# -------------------- talker: source-level contract --------------------


def _talker_init() -> ast.FunctionDef:
    tree = ast.parse(_TALKER_SRC.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3TTSTalkerForConditionalGeneration"
    )
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")


def _assigned_names(init: ast.FunctionDef, attr: str) -> list[ast.expr]:
    """Values assigned to ``self.<attr>`` anywhere in the constructor."""
    values: list[ast.expr] = []
    for node in ast.walk(init):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == attr and isinstance(target.value, ast.Name):
                if target.value.id == "self":
                    values.append(node.value)
    return values


def test_talker_init_takes_embedding_dtype_from_engine_dtype():
    init = _talker_init()
    values = _assigned_names(init, "_embedding_dtype")

    assert values, "Qwen3TTSTalkerForConditionalGeneration.__init__ no longer sets _embedding_dtype"
    for value in values:
        assert isinstance(value, ast.Name), (
            "_embedding_dtype is assigned a literal "
            f"({ast.unparse(value)}) instead of the engine dtype — this is the "
            "hardcode that crashes every request on fp16-only GPUs."
        )

    # ...and the name it is assigned from must itself come from the engine config.
    dtype_name = values[0].id
    sources = [
        node.value
        for node in ast.walk(init)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == dtype_name for t in node.targets)
    ]
    assert sources, f"{dtype_name} is not assigned inside __init__"
    assert any("model_config" in ast.unparse(src) for src in sources), (
        f"{dtype_name} no longer reads the engine's model_config dtype"
    )


def test_talker_passes_engine_dtype_into_prompt_embeds_builder():
    init = _talker_init()
    calls = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Qwen3TTSPromptEmbedsBuilder"
    ]
    assert calls, "the talker no longer constructs Qwen3TTSPromptEmbedsBuilder in __init__"

    dtype_name = _assigned_names(init, "_embedding_dtype")[0].id
    for call in calls:
        passed = {kw.arg: kw.value for kw in call.keywords}
        assert "embedding_dtype" in passed, (
            "Qwen3TTSPromptEmbedsBuilder is constructed without embedding_dtype — "
            "prefill embeddings would fall back to a dtype the engine did not choose."
        )
        value = passed["embedding_dtype"]
        assert isinstance(value, ast.Name) and value.id == dtype_name, (
            "embedding_dtype is passed as "
            f"{ast.unparse(value)} instead of the engine dtype {dtype_name} — the "
            "prefill and decode paths would disagree."
        )
