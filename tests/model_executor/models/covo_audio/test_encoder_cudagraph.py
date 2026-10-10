# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config import CUDAGraphMode, ModelConfig, VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from vllm_omni.model_executor.models.covo_audio.covo_audio import CovoAudioForConditionalGeneration
from vllm_omni.model_executor.models.covo_audio.covo_audio_llm import CovoAudioLLMForConditionalGeneration

pytestmark = [pytest.mark.core_model]


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, features: torch.Tensor) -> SimpleNamespace:
        hidden = features[:, :4, ::2].transpose(1, 2) * self.scale
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyAudioAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = nn.Module()
        layer.linear2 = nn.Linear(4, 6, bias=False)
        self.downsample_layers = nn.ModuleList([layer])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.downsample_layers[0].linear2(hidden[:, ::8])


def _stage0_model(device: str) -> CovoAudioForConditionalGeneration:
    # Keep the production Stage-0 wrapper and graph protocol; only the costly
    # Whisper and adapter weights are replaced with small deterministic modules.
    inner = CovoAudioLLMForConditionalGeneration.__new__(CovoAudioLLMForConditionalGeneration)
    nn.Module.__init__(inner)
    inner.encoder = _TinyEncoder()
    inner.audio_adapter = _TinyAudioAdapter()

    model = CovoAudioForConditionalGeneration.__new__(CovoAudioForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "fused_thinker_talker"
    model.fused_thinker_talker = inner
    return model.eval().to(device)


def _inputs(lengths: list[int], device: str, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "audio_features": torch.randn(len(lengths), 128, 3000, generator=generator).to(device),
        "audio_num_tokens": torch.tensor(lengths, device=device),
    }


def _runner(model: CovoAudioForConditionalGeneration, config: VllmConfig, device: str):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.compilation_config = config.compilation_config
    runner.supports_mm_inputs = True
    runner.get_model = lambda: model
    runner.vllm_config = config
    runner.device = torch.device(device)
    runner.dtype = torch.float32
    return runner


@pytest.mark.cpu
@torch.inference_mode()
def test_stage0_eager_audio_embeddings_use_actual_lengths() -> None:
    model = _stage0_model("cpu")
    inputs = _inputs([35, 161], "cpu", seed=0)

    outputs = model.embed_multimodal(**inputs)

    assert [output.shape for output in outputs] == [(35, 6), (161, 6)]
    specs = model.get_encoder_cudagraph_item_specs(inputs)
    assert [spec.output_tokens for spec in specs] == [188, 188]
    selected = model.select_encoder_cudagraph_items(inputs, [1])
    assert selected["audio_features"].shape == (1, 128, 3000)
    assert selected["audio_num_tokens"].tolist() == [161]


@pytest.mark.cpu
@pytest.mark.parametrize("enabled", [False, True])
@torch.inference_mode()
def test_stage0_stays_eager_without_cuda_or_opt_in(enabled: bool) -> None:
    model = _stage0_model("cpu")
    config = VllmConfig()
    config.compilation_config.cudagraph_mm_encoder = enabled
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    runner = _runner(model, config, "cpu")
    runner.encoder_cudagraph_manager = None
    if enabled:
        assert runner.capture_model() == 0
    else:
        assert runner._create_encoder_cudagraph_manager() is None
    assert runner.encoder_cudagraph_manager is None

    assert model.embed_multimodal(**_inputs([35], "cpu", seed=0))[0].shape == (35, 6)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_stage0_encoder_graph_matches_eager_and_splits_large_batches() -> None:
    model = _stage0_model("cuda")
    config = VllmConfig()
    config.model_config = ModelConfig.__new__(ModelConfig)
    config.model_config.multimodal_config = MultiModalConfig()
    config.compilation_config.cudagraph_mm_encoder = True
    config.compilation_config.encoder_cudagraph_token_budgets = [188, 376]
    config.compilation_config.encoder_cudagraph_max_vision_items_per_batch = 2
    manager = _runner(model, config, "cuda")._create_encoder_cudagraph_manager()
    assert isinstance(manager, EncoderCudaGraphManager)
    manager.capture(None)

    assert manager.get_cumulative_stats()["num_budgets"] == 2
    retained_output = None
    for lengths, seed in [([35], 0), ([35, 161], 1), ([35, 161, 50], 2)]:
        inputs = _inputs(lengths, "cuda", seed)
        expected = model.embed_multimodal(**inputs)
        actual = manager.execute(inputs)
        assert len(actual) == len(expected)
        for graph_output, eager_output in zip(actual, expected, strict=True):
            torch.testing.assert_close(graph_output, eager_output)
        if retained_output is None:
            retained_output = actual[0]
            retained_expected = expected[0].clone()

    torch.testing.assert_close(retained_output, retained_expected)
    assert set(manager.budget_graphs["default"]) == {188, 376}

    # The three-item input exceeds the graph's two-item limit and is split.
    stats = manager.get_cumulative_stats()
    assert stats["graph_hits"] == 6
    assert stats["graph_misses"] == 0
    manager.clear()
