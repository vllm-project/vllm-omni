# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA manager integration with small production encoders, not serving E2E."""

import pytest
import torch
from vllm.config import CompilationConfig, MultiModalConfig
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

from vllm_omni.model_executor.models.minicpmo_4_5.encoder_cudagraph import bind_minicpmo_encoder_cudagraph
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)

from .test_encoder_cudagraph import _EncoderModel, _EncoderTestModelConfig, _EncoderTestVllmConfig

pytestmark = [pytest.mark.core_model, pytest.mark.gpu, pytest.mark.cuda]


@pytest.mark.skipif(
    "capture_axes" not in EncoderCudaGraphConfig.__dataclass_fields__,
    reason="Requires vLLM containing capture_axes (#42785)",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("dtype, grid", [(torch.float32, (2, 3)), (torch.bfloat16, (32, 32))])
def test_manager_graph_replay_matches_encoder_entry_point(modality: str, dtype, grid, monkeypatch) -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    torch.manual_seed(42)
    model = _EncoderModel().eval().cuda().to(dtype)
    mm_config = MultiModalConfig(
        media_io_kwargs={"video": {"num_frames": 2}}, limit_per_prompt={"image": 2, "video": 2, "audio": 2}
    )
    model.multimodal_config = mm_config
    compilation = CompilationConfig(
        cudagraph_mm_encoder=True,
        encoder_cudagraph_token_budgets=[16, 32],
        encoder_cudagraph_max_vision_items_per_batch=2,
        encoder_cudagraph_max_frames_per_batch=2,
    )
    config = _EncoderTestVllmConfig(
        compilation_config=compilation,
        model_config=_EncoderTestModelConfig(multimodal_config=mm_config),
    )
    model.vllm_config = config
    runner = GPUModelRunner.__new__(GPUModelRunner)
    serving_model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(serving_model)
    serving_model.thinker = model
    serving_model.model = model
    bind_minicpmo_encoder_cudagraph(serving_model, model)
    runner.model = serving_model
    runner.compilation_config = compilation
    runner.supports_mm_inputs = True
    runner.vllm_config = config
    runner.device = torch.device("cuda")
    runner.dtype = dtype
    manager = GPUModelRunner._create_encoder_cudagraph_manager(runner)
    assert manager is not None
    assert manager.model is serving_model
    budgets_used = set()
    replay = manager._run_budget_graph

    def record_budget(mm_kwargs, token_budget, *args, **kwargs):
        budgets_used.add(token_budget)
        return replay(mm_kwargs, token_budget, *args, **kwargs)

    monkeypatch.setattr(manager, "_run_budget_graph", record_budget)
    with torch.inference_mode():
        manager.capture(torch.cuda.graph_pool_handle())
        assert set(manager.budget_graphs["default"]) == {
            (budget, (("vision", patches),)) for budget in (16, 32) for patches in (1024, 1152, 2048)
        }
        for iteration in range(8):
            prefix = "video_" if modality == "video" else ""
            counts = (5, 3) if iteration % 2 else (2, 1)
            kwargs = {
                prefix + "pixel_values": [
                    [torch.randn(3, 2, grid[0] * grid[1] * 2, device="cuda", dtype=dtype) for _ in range(count)]
                    for count in counts
                ],
                prefix + "tgt_sizes": [torch.tensor([list(grid)] * count) for count in counts],
            }
            expected = model.get_multimodal_embeddings(**kwargs)
            actual = manager.execute(kwargs)
            assert len(actual) == len(expected) == 2
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(
                    result,
                    reference,
                    atol=1e-2 if dtype == torch.bfloat16 else 1e-5,
                    rtol=1e-2 if dtype == torch.bfloat16 else 1e-4,
                )
        assert manager.get_cumulative_stats()["graph_hits"] == 16
        assert manager.get_cumulative_stats()["graph_misses"] == 0
        assert budgets_used == {16, 32}

        # One item beyond the largest token budget must use the original
        # encoder and still preserve all slices/chunks and output ownership.
        oversized = {
            prefix + "pixel_values": [
                [torch.randn(3, 2, grid[0] * grid[1] * 2, device="cuda", dtype=dtype) for _ in range(9)]
            ],
            prefix + "tgt_sizes": [torch.tensor([list(grid)] * 9)],
        }
        expected = model.get_multimodal_embeddings(**oversized)
        actual = manager.execute(oversized)
        assert len(actual) == len(expected) == 1
        torch.testing.assert_close(
            actual[0],
            expected[0],
            atol=1e-2 if dtype == torch.bfloat16 else 1e-5,
            rtol=1e-2 if dtype == torch.bfloat16 else 1e-4,
        )
        assert manager.get_cumulative_stats()["graph_hits"] == 16
        assert manager.get_cumulative_stats()["graph_misses"] == 1
        compilation.cudagraph_mm_encoder = False
        assert GPUModelRunner._create_encoder_cudagraph_manager(runner) is None
    manager.clear()
