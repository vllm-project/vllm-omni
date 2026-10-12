# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cache isolation and graph replay for the opt-in TTS predictor."""

import pytest
import torch

from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorBaseModel
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_frames_match_causal_prefill_after_graph_replay(dtype):
    from vllm_omni.model_executor.models.qwen3_tts.cached_code_predictor import FrameLocalKVCache

    config = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_code_groups=4,
    )
    torch.manual_seed(171)
    model = CodePredictorBaseModel(config).to(device="cuda", dtype=dtype).eval()
    inputs = torch.randn(3, 5, 64, device="cuda", dtype=dtype)
    positions = torch.arange(5, device="cuda").expand(3, -1)
    cache = FrameLocalKVCache(model, max_batch=3)
    with torch.inference_mode():
        for batch in (1, 3):
            for _ in range(2):
                for step in (1, 2, 3):
                    cache(inputs, batch, step)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = torch.cat([cache(inputs, 3, step) for step in (1, 2, 3)], dim=1)
        for _ in range(3):
            inputs.normal_()
            # No previous frame (including a smaller batch) may leak through.
            cache(inputs, 1, 1)
            cache.buffer.fill_(torch.nan)
            graph.replay()
            expected = model(inputs, positions)[:, :4]
            tolerance = 1e-5 if dtype == torch.float32 else 0.04
            torch.testing.assert_close(output, expected, atol=tolerance, rtol=tolerance)
