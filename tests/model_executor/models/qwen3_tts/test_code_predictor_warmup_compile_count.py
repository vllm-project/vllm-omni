# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression guard: _warmup_buckets()'s plain branch must not recompile per bucket.

Drives the real _setup_compile()/_warmup_buckets() chain with torch.compile
patched to count Dynamo frames, so dropping mark_dynamic or adding a
shape-dependent branch to forward() fails this test instead of silently
costing the warmup time back.

Uses plain imports rather than test_code_predictor_dtype.py's mocked-import
harness: mocking current_omni_platform there makes Dynamo hit a graph break
and wrongly specialize the batch dim, unrelated to this fix.
"""

import types

import pytest
import torch
from torch._dynamo.testing import CompileCounter

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm import (
    Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM,
)

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA")
def test_plain_warmup_compiles_exactly_two_graphs(mocker) -> None:
    """Real _setup_compile()/_warmup_buckets() plain branch; asserts == 2 Dynamo frames."""
    torch._dynamo.reset()
    try:
        cp_config = Qwen3TTSTalkerCodePredictorConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_code_groups=4,
            rms_norm_eps=1e-6,
        )
        talker_config = Qwen3TTSTalkerConfig(hidden_size=64, num_code_groups=4)

        # Only the attributes _setup_compile() reads; unlike a bare MagicMock(),
        # a new attribute read later raises instead of silently flipping the branch.
        vllm_config = types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(max_num_seqs=16),
            model_config=types.SimpleNamespace(stage_connector_config=None),
            additional_config={},
        )

        counter = CompileCounter()
        real_compile = torch.compile

        def counting_compile(fn, *args, **kwargs):
            kwargs["backend"] = counter
            # "options" is Inductor-specific; CompileCounter.__call__ doesn't accept it.
            kwargs.pop("options", None)
            return real_compile(fn, *args, **kwargs)

        mocker.patch("torch.compile", counting_compile)

        predictor = Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        ).to(device="cuda", dtype=torch.bfloat16)

        # _setup_compile() derives its own cache_size_limit from bucket count
        # alone (2 for this branch), independent of whether mark_dynamic runs.
        # Left as-is, a regression that drops mark_dynamic would hit that
        # limit after 2 real compiles and fall back to eager for the rest --
        # invisible to CompileCounter, which only sees compile attempts. A
        # generous override here isolates what this test actually checks.
        mocker.patch.object(
            predictor, "_compile_cache_size_limit", return_value=len(predictor._batch_bucket_sizes()) + 1
        )

        # Pins the branch: the frame-local KV path returns before calling the
        # base _warmup_buckets(), which would otherwise fail below with a
        # confusing frame_count mismatch instead of a clear assertion here.
        assert not predictor._kv_requested
        assert predictor._frame_cache is None

        predictor._setup_compile()  # drives the real plain-branch _warmup_buckets()

        assert not predictor._kv_cache_enabled
        assert not predictor._prefix_reprefill_enabled
        # 1 static graph (bsz==1) + 1 dynamic graph reused for bsz>1. A graph
        # break in forward() also raises frame_count, so this catches that too.
        assert counter.frame_count == 2, (
            f"expected 2 Dynamo frames, got {counter.frame_count} for buckets "
            f"{predictor._bucket_sizes} -- per-bucket specialization returned, "
            "or forward() gained a new graph break"
        )
    finally:
        torch._dynamo.reset()
