# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Alpamayo multimodal processor.

Validates the two essentials without a full vLLM processing context:
1. it reuses Qwen3-VL's preprocess (subclass relationship);
2. ``get_hf_processor`` adds the new trajectory tokens to the tokenizer.
"""

from __future__ import annotations

import os
import types

import pytest
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)

from vllm_omni.model_executor.models.alpamayo.processor import (
    AlpamayoDummyInputsBuilder,
    AlpamayoMultiModalProcessor,
    AlpamayoProcessingInfo,
)

_R1_DIR = "/data/models/Alpamayo-R1-10B"


def test_reuses_qwen3vl_preprocess_by_subclassing():
    # Essential #2: reuse Qwen3-VL preprocess.
    assert issubclass(AlpamayoMultiModalProcessor, Qwen3VLMultiModalProcessor)
    assert issubclass(AlpamayoProcessingInfo, Qwen3VLProcessingInfo)
    assert issubclass(AlpamayoDummyInputsBuilder, Qwen3VLDummyInputsBuilder)


def test_model_registered_with_alpamayo_processor():
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

    from vllm_omni.model_executor.models.alpamayo.alpamayo import (
        Alpamayo15ForConditionalGeneration,
    )

    # Importing the decorated model class exercises register_processor with the
    # Alpamayo processor/info/dummy-inputs; it must succeed and stay a Qwen3-VL.
    assert issubclass(Alpamayo15ForConditionalGeneration, Qwen3VLForConditionalGeneration)


@pytest.mark.skipif(not os.path.isdir(_R1_DIR), reason="Alpamayo-R1 tokenizer not present locally")
def test_get_hf_processor_adds_traj_tokens():
    # Essential #1: add the new tokens. Drive _ensure_traj_tokens with a
    # fake processor wrapping the real base tokenizer + a fake ctx providing the
    # Alpamayo config fields it reads.
    from transformers import AutoTokenizer

    import vllm_omni.transformers_utils.configs.alpamayo  # noqa: F401 (register)

    tokenizer = AutoTokenizer.from_pretrained(_R1_DIR, trust_remote_code=True)
    assert "<i0>" not in tokenizer.get_vocab()

    fake_processor = types.SimpleNamespace(tokenizer=tokenizer)
    hf_config = types.SimpleNamespace(traj_vocab_size=4000, add_special_tokens=True)

    info = AlpamayoProcessingInfo.__new__(AlpamayoProcessingInfo)
    info.get_hf_config = lambda: hf_config  # type: ignore[method-assign]
    info._ensure_traj_tokens(fake_processor)

    # New tokens present with the checkpoint's ids.
    assert tokenizer.convert_tokens_to_ids("<i0>") == 151669
    assert tokenizer.convert_tokens_to_ids("<|traj_future_start|>") == 155681
    assert len(tokenizer) == 155697
    assert fake_processor._alpamayo_tokens_added is True

    # Idempotent: a second call is a no-op.
    info._ensure_traj_tokens(fake_processor)
    assert len(tokenizer) == 155697
