# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio chunk planning and captured full-tower numerical equivalence."""

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.qwen3_omni.test_vision_encoder_cudagraph_cuda import _parallel_state  # noqa: F401
from vllm_omni.model_executor.models.qwen3_omni.audio_encoder_cudagraph import (
    Qwen3OmniAudioEncoderCudaGraphs,
    audio_chunk_metadata,
)

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"})]


def test_audio_chunk_metadata_preserves_clip_boundaries():
    chunks, indices, boundaries, lengths = audio_chunk_metadata([100, 101, 205], 100, 800)
    assert chunks == [100, 100, 1, 100, 100, 5]
    assert lengths == [13, 14, 27]
    assert boundaries == [0, 13, 27, 54]
    assert indices == list(range(27)) + list(range(39, 66))


@pytest.mark.parametrize("backend", [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TRITON_ATTN])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.usefixtures("_parallel_state")
def test_audio_graph_matches_eager_and_preserves_cached_outputs(monkeypatch, backend):
    import vllm.model_executor.models.vision as vision_module
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.multimodal import MultiModalConfig
    from vllm.distributed.parallel_state import graph_capture
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import Qwen3OmniMoeAudioEncoder

    monkeypatch.setattr(
        vision_module, "get_multimodal_config", lambda: MultiModalConfig(mm_encoder_attn_backend=backend)
    )
    config = Qwen3OmniMoeAudioEncoderConfig(
        d_model=128,
        encoder_attention_heads=2,
        encoder_layers=2,
        encoder_ffn_dim=256,
        downsample_hidden_size=8,
        output_dim=64,
        num_mel_bins=80,
        max_source_positions=1500,
        n_window=50,
        n_window_infer=800,
        conv_chunksize=2,
    )
    with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        tower = Qwen3OmniMoeAudioEncoder(config)
    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        for name, param in tower.named_parameters():
            if "norm" in name or "ln_post" in name:
                param.fill_(1 if name.endswith("weight") else 0)
            else:
                param.copy_(torch.randn(param.shape, generator=generator, device="cuda") * 0.02)
    graphs = Qwen3OmniAudioEncoderCudaGraphs(
        tower, budgets=tuple(range(1, 17)), extra_shapes=((5, 47), (9, 105), (10, 118), (15, 195))
    )
    retained = []
    with torch.inference_mode(), graph_capture(device=torch.device("cuda")):
        set_cudagraph_capturing_enabled(True)
        try:
            graphs.capture(torch.cuda.graph_pool_handle())
        finally:
            set_cudagraph_capturing_enabled(False)
    with torch.inference_mode():
        for lengths in ([100], [101], [199], [201], [150, 205], [801], [1500], [801, 100], [1601]):
            features = torch.randn(80, sum(lengths), device="cuda", dtype=torch.bfloat16, generator=generator)
            lens = torch.tensor(lengths, device="cuda")
            output_lens = audio_chunk_metadata(list(lengths), 100, 800)[-1]
            expected = tower(features, lens, torch.tensor(output_lens, device="cuda")).split(output_lens)
            actual = graphs.execute(features, list(lengths))
            if sum((n + 99) // 100 for n in lengths) > 16:
                assert actual is None
                continue
            assert [x.shape for x in actual] == [x.shape for x in expected]
            for output, reference in zip(actual, expected):
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                retained.append((output, output.clone()))
        assert graphs.execute(torch.zeros(80, 99, device="cuda", dtype=torch.bfloat16), [99]) is None
        for output, saved in retained:
            torch.testing.assert_close(output, saved, rtol=0, atol=0)
