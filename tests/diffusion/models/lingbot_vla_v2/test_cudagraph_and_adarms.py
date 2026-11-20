# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU and CUDA parity tests for LingBot-VLA 2.0 CUDA-graph action loop and AdaRMSNorm fusion."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, fields

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.lingbot_vla_v2.config import LingBotVLAV2Config
from vllm_omni.diffusion.models.lingbot_vla_v2.cudagraph import LingBotActionLoopCUDAGraph
from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingBotVLAV2ForActionPrediction
from vllm_omni.diffusion.models.lingbot_vla_v2.processor_lingbot_vla_v2 import LingBotVLAV2Processor
from vllm_omni.diffusion.models.lingbot_vla_v2.qwen2_action_expert import AdaRMSNorm

from .conftest import observation, tiny_backbone, tiny_config

pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
def test_config_does_not_expose_checkpoint_cuda_graph_knob() -> None:
    assert "use_cuda_graph" not in {f.name for f in fields(LingBotVLAV2Config)}


@pytest.mark.cpu
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_adarms_fusion_drops_originals_and_preserves_parity_cpu(input_dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    norm = AdaRMSNorm(hidden_size=32, cond_dim=16).float().eval()
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
        norm.gamma.weight.uniform_(-0.2, 0.2)
        norm.gamma.bias.uniform_(-0.1, 0.1)
        norm.beta.weight.uniform_(-0.2, 0.2)
        norm.beta.bias.uniform_(-0.1, 0.1)

    unfused = copy.deepcopy(norm)
    x = torch.randn(2, 5, 32, dtype=input_dtype)
    cond = torch.randn(2, 16, dtype=input_dtype)
    expected = unfused(x, cond)

    assert norm.fuse_weights() is True
    assert norm.fuse_weights() is False
    assert norm.gamma is None and norm.beta is None
    assert norm.fused_gb is not None
    actual = norm(x, cond)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.cpu
def test_model_fuse_adarms_preserves_sample_actions_cpu(model: LingBotVLAV2ForActionPrediction, assets) -> None:
    proc = LingBotVLAV2Processor(tiny_config(), assets)
    inputs = proc.build_model_inputs(observation())
    noise = torch.randn(1, 4, 55, generator=torch.Generator().manual_seed(3))
    expected = model.sample_actions(**inputs, noise=noise)

    fused = model.fuse_adarms()
    assert fused
    actual = model.sample_actions(**inputs, noise=noise)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.cpu
def test_cudagraph_runner_cpu_fallback_and_shape_guard() -> None:
    model = LingBotVLAV2ForActionPrediction(tiny_config(), vlm_config=tiny_backbone()).float().eval()
    with pytest.raises(ValueError, match="max_graphs"):
        LingBotActionLoopCUDAGraph(model, device="cpu", max_graphs=0)

    runner = LingBotActionLoopCUDAGraph(model, device="cpu", max_graphs=2)
    assert runner.enabled is False

    state = torch.zeros(1, 55)
    masks = torch.ones(1, 10, dtype=torch.bool)
    positions = torch.zeros(3, 1, 10, dtype=torch.long)
    noise = torch.zeros(1, 4, 55)
    kv = {0: {"key_states": torch.zeros(1, 10, 2, 16), "value_states": torch.zeros(1, 10, 2, 16)}}
    assert runner.run(num_steps=2, state=state, masks=masks, kv=kv, positions=positions, noise=noise) is None


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("enforce_eager", "enable_cuda_graph_decode"),
    [(True, True), (False, False), (False, True)],
)
def test_pipeline_gating_respects_od_config(
    assets,
    model: LingBotVLAV2ForActionPrediction,
    monkeypatch: pytest.MonkeyPatch,
    enforce_eager: bool,
    enable_cuda_graph_decode: bool,
) -> None:
    pytest.importorskip("vllm")
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2 import LingBotVLAV2Pipeline

    cfg = tiny_config()
    backbone = assets / "backbone"
    tiny_backbone().save_pretrained(backbone)
    cfg.tokenizer_path = str(backbone)
    (assets / "config.json").write_text(json.dumps(asdict(cfg)))
    save_file({"model." + k: v.clone() for k, v in model.state_dict().items()}, assets / "model.safetensors")
    monkeypatch.setattr("vllm_omni.diffusion.distributed.utils.get_local_device", lambda: torch.device("cpu"))

    runtime = OmniDiffusionConfig(
        model=str(assets),
        dtype=torch.float32,
        num_gpus=1,
        enforce_eager=enforce_eager,
        enable_cuda_graph_decode=enable_cuda_graph_decode,
    )
    pipeline = LingBotVLAV2Pipeline(od_config=runtime)
    # CPU never enables CUDA graph even when flags allow it.
    assert pipeline.enable_cuda_graph is False
    pipeline.release_captured_graphs()

    empty_pipe = LingBotVLAV2Pipeline.__new__(LingBotVLAV2Pipeline)
    torch.nn.Module.__init__(empty_pipe)
    empty_pipe.release_captured_graphs()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture and GPU parity tests")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_adarms_fusion_parity_cuda(input_dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    norm = AdaRMSNorm(hidden_size=64, cond_dim=64).float().cuda().eval()
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
        norm.gamma.weight.uniform_(-0.2, 0.2)
        norm.gamma.bias.uniform_(-0.1, 0.1)
        norm.beta.weight.uniform_(-0.2, 0.2)
        norm.beta.bias.uniform_(-0.1, 0.1)

    unfused = copy.deepcopy(norm)
    x = torch.randn(1, 8, 64, device="cuda", dtype=input_dtype)
    cond = torch.randn(1, 64, device="cuda", dtype=input_dtype)
    expected = unfused(x, cond)

    assert norm.fuse_weights() is True
    assert norm.gamma is None and norm.beta is None
    actual = norm(x, cond)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture and replay")
@torch.no_grad()
def test_action_cudagraph_replay_matches_eager_and_handles_steps_and_shapes_cuda(
    model: LingBotVLAV2ForActionPrediction, assets
) -> None:
    cuda_model = copy.deepcopy(model).cuda().eval()
    cuda_model.fuse_adarms()

    proc = LingBotVLAV2Processor(tiny_config(), assets, device="cuda")
    inputs_a = proc.build_model_inputs(observation())
    altered_obs = observation()
    altered_obs["state"] = altered_obs["state"] + 0.25
    altered_obs["prompt"] = "cube"
    inputs_b = proc.build_model_inputs(altered_obs)

    noise_a = torch.randn(1, 4, 55, device="cuda", generator=torch.Generator(device="cuda").manual_seed(11))
    noise_b = torch.randn(1, 4, 55, device="cuda", generator=torch.Generator(device="cuda").manual_seed(22))

    short_inputs = copy.deepcopy(inputs_a)
    short_inputs["lang_tokens"] = short_inputs["lang_tokens"][:, :8]
    short_inputs["lang_masks"] = short_inputs["lang_masks"][:, :8]

    # Eager baselines for steps=2, steps=3, and shorter prefix.
    cuda_model.configure_action_cuda_graph(False)
    eager_a_2 = cuda_model.sample_actions(**inputs_a, noise=noise_a, num_steps=2)
    eager_b_2 = cuda_model.sample_actions(**inputs_b, noise=noise_b, num_steps=2)
    eager_a_3 = cuda_model.sample_actions(**inputs_a, noise=noise_a, num_steps=3)
    eager_short = cuda_model.sample_actions(**short_inputs, noise=noise_a, num_steps=2)

    # Graphed path with bounded cache.
    cuda_model.configure_action_cuda_graph(True)
    graph_a_2 = cuda_model.sample_actions(**inputs_a, noise=noise_a, num_steps=2)
    runner = cuda_model._action_graph_runner
    assert runner is not None and len(runner._cache) == 1

    # Different observation/state/noise with same prefix length reuses the same graph entry.
    graph_b_2 = cuda_model.sample_actions(**inputs_b, noise=noise_b, num_steps=2)
    assert len(runner._cache) == 1

    # Different step count gets its own cached entry rather than thrashing.
    graph_a_3 = cuda_model.sample_actions(**inputs_a, noise=noise_a, num_steps=3)
    assert len(runner._cache) == 2

    torch.testing.assert_close(graph_a_2, eager_a_2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(graph_b_2, eager_b_2, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(graph_a_3, eager_a_3, atol=1e-5, rtol=1e-5)
    assert not torch.equal(graph_a_2, graph_b_2)

    # Mismatched prefix length does not crash replay and falls back cleanly / captures new shape.
    graph_short = cuda_model.sample_actions(**short_inputs, noise=noise_a, num_steps=2)
    assert len(runner._cache) == 3
    torch.testing.assert_close(graph_short, eager_short, atol=1e-5, rtol=1e-5)

    # Direct replay() with uncached shape/step returns None instead of copy_ RuntimeError.
    first_entry = next(iter(runner._cache.values()))
    assert (
        runner.replay(
            num_steps=99,
            state=first_entry.static_state,
            masks=first_entry.static_masks,
            kv=first_entry.static_kv,
            positions=first_entry.static_positions,
            noise=first_entry.static_noise,
        )
        is None
    )
    assert (
        runner.replay(
            num_steps=first_entry.num_steps,
            state=first_entry.static_state,
            masks=first_entry.static_masks[:, :2],
            kv=first_entry.static_kv,
            positions=first_entry.static_positions,
            noise=first_entry.static_noise,
        )
        is None
    )

    released = cuda_model.release_captured_graphs()
    assert released == 3 and len(runner._cache) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pipeline startup capture test")
def test_pipeline_startup_capture_enforce_eager_and_release_cuda(
    assets,
    model: LingBotVLAV2ForActionPrediction,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm")
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2 import LingBotVLAV2Pipeline
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    cfg = tiny_config()
    backbone = assets / "backbone"
    tiny_backbone().save_pretrained(backbone)
    cfg.tokenizer_path = str(backbone)
    (assets / "config.json").write_text(json.dumps(asdict(cfg)))
    save_file({"model." + k: v.clone() for k, v in model.state_dict().items()}, assets / "model.safetensors")
    monkeypatch.setattr("vllm_omni.diffusion.distributed.utils.get_local_device", lambda: torch.device("cuda:0"))

    eager_cfg = OmniDiffusionConfig(model=str(assets), dtype=torch.float32, num_gpus=1, enforce_eager=True)
    graph_cfg = OmniDiffusionConfig(model=str(assets), dtype=torch.float32, num_gpus=1, enforce_eager=False)

    with torch.inference_mode():
        eager_pipe = LingBotVLAV2Pipeline(od_config=eager_cfg)
        graph_pipe = LingBotVLAV2Pipeline(od_config=graph_cfg)

    assert eager_pipe.enable_cuda_graph is False
    assert eager_pipe.model._action_graph_runner is None

    assert graph_pipe.enable_cuda_graph is True
    runner = graph_pipe.model._action_graph_runner
    # Startup precaptured the default num_steps graph before any user request ran.
    assert runner is not None and len(runner._cache) == 1

    obs = observation()
    eager_params = OmniDiffusionSamplingParams(num_inference_steps=cfg.num_steps, extra_args={"robot_obs": obs})
    eager_params.generator = torch.Generator(device="cuda").manual_seed(5)
    graph_params = OmniDiffusionSamplingParams(num_inference_steps=cfg.num_steps, extra_args={"robot_obs": obs})
    graph_params.generator = torch.Generator(device="cuda").manual_seed(5)

    eager_res = eager_pipe(
        DiffusionRequestBatch(
            requests=[OmniDiffusionRequest(prompt="pick", sampling_params=eager_params, request_id="e")]
        )
    )
    graph_res = graph_pipe(
        DiffusionRequestBatch(
            requests=[OmniDiffusionRequest(prompt="pick", sampling_params=graph_params, request_id="g")]
        )
    )
    assert len(runner._cache) == 1
    torch.testing.assert_close(
        torch.from_numpy(graph_res.output["actions"]),
        torch.from_numpy(eager_res.output["actions"]),
        atol=1e-5,
        rtol=1e-5,
    )

    graph_pipe.release_captured_graphs()
    assert len(runner._cache) == 0
