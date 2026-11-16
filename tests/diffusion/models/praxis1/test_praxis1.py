# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts of the native Praxis-1 port: config, masks, rotary, sampler, loading and worker state."""

import hashlib
import time
from collections import OrderedDict
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.tokenization_utils_base import BatchEncoding

from vllm_omni.diffusion.models.praxis1 import layers, pipeline_praxis1, processing
from vllm_omni.diffusion.models.praxis1.action_expert import trunk_masks
from vllm_omni.diffusion.models.praxis1.backbone import observed_context_mask
from vllm_omni.diffusion.models.praxis1.configuration_praxis1 import _REQUIRED_VALUES, CheckpointError, validate_config
from vllm_omni.diffusion.models.praxis1.pipeline_praxis1 import Praxis1Pipeline
from vllm_omni.diffusion.models.praxis1.praxis1_transformer import (
    Praxis1Transformer,
    guidance_combine,
    seeded_normal,
    window_time_grid,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras.praxis1 import PraxisSessionOptions, SamplerSettings

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def minimal_config(embodiment: str = "single_arm") -> dict:
    """The smallest config ``validate_config`` accepts for one embodiment."""
    config: dict = {}
    for path, value in _REQUIRED_VALUES.items():
        node = config
        *parents, key = path.split(".")
        for parent in parents:
            node = node.setdefault(parent, {})
        node[key] = value
    dim = 20 if embodiment == "bimanual" else 10
    config["text_encoder"]["blank_caption"] = " "
    config["normalization"] = {
        "action_q01": [0.0] * dim,
        "action_q99": [1.0] * dim,
        "proprio_mean": [0.0] * dim,
        "proprio_std": [1.0] * dim,
    }
    if embodiment == "bimanual":
        config.update(embodiment="bimanual", pose_frame="rig")
        config["video"].update(width=1248, height=256, tile_height=256, views=["top_left", "wrist_left", "wrist_right"])
        config["action_space"]["dim"] = 20
        config["action_expert"].update(action_patch_dim=80, proprio_patch_dim=20)
    return config


def _worker(transformer=None):
    """Build a worker shell; full component construction is covered by the GPU E2E."""
    worker = Praxis1Pipeline.__new__(Praxis1Pipeline)
    nn.Module.__init__(worker)
    worker.transformer = transformer if transformer is not None else nn.Linear(2, 2, bias=False)
    worker.text_encoder = nn.Module()
    worker.text_cache_max_bytes = pipeline_praxis1.DEFAULT_TEXT_CACHE_MAX_BYTES
    worker.text_cache_max_entries = pipeline_praxis1.DEFAULT_TEXT_CACHE_MAX_ENTRIES
    worker._text_cache = OrderedDict()
    worker._text_cache_bytes = 0
    worker._sessions = {}
    worker.max_sessions = 4
    worker.max_num_steps = 16
    worker.session_idle_timeout = 300
    worker.startup_warmup = False
    worker.sampler = PraxisSessionOptions()
    worker.embodiment = "single_arm"
    return worker


@pytest.mark.parametrize("embodiment", ["single_arm", "bimanual"])
def test_checkpoint_config_contract(embodiment):
    config = minimal_config(embodiment)
    validate_config(config)
    config["video"].update(height=256, tile_height=256, antialias=False) if embodiment == "single_arm" else None
    if embodiment == "single_arm":
        with pytest.raises(CheckpointError, match="video"):
            validate_config(config)
    else:
        config["action_expert"]["action_patch_dim"] = 40
        with pytest.raises(CheckpointError, match="action_patch_dim"):
            validate_config(config)
    bad = minimal_config(embodiment)
    bad["normalization"]["proprio_std"][0] = 0.0
    with pytest.raises(CheckpointError, match="proprio_std"):
        validate_config(bad)


def test_native_checkpoint_resolves_to_generic_action_stage(monkeypatch, tmp_path):
    import yaml

    from vllm_omni.config import config_factory
    from vllm_omni.config.resolver import resolve_omni_config
    from vllm_omni.diffusion.registry import DiffusionModelRegistry
    from vllm_omni.diffusion.utils import hf_utils

    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        yaml.safe_dump(
            {
                "async_chunk": False,
                "dtype": "bfloat16",
                "stages": [
                    {
                        "stage_id": 0,
                        "devices": "0",
                        "max_num_seqs": 1,
                        "enforce_eager": True,
                        "model_class_name": "Praxis1Pipeline",
                        "model_config": {"max_context_chunks": 1},
                    }
                ],
            }
        )
    )

    def read(filename, *_args, **_kwargs):
        return {"model_type": "praxis", "torch_dtype": "bfloat16"} if filename == "config.json" else None

    monkeypatch.setattr("vllm.transformers_utils.config.get_hf_file_to_dict", read)
    monkeypatch.setattr(hf_utils, "get_hf_file_to_dict", read)
    monkeypatch.setattr(config_factory, "get_hf_file_to_dict", read)
    monkeypatch.setattr(config_factory.StageConfigFactory, "get_hf_config", lambda **_: None)
    hf_utils.is_diffusion_model.cache_clear()
    try:
        assert "Praxis1Pipeline" in DiffusionModelRegistry.get_supported_archs()
        assert hf_utils.is_diffusion_model(str(tmp_path))
        resolved = resolve_omni_config(
            str(tmp_path),
            trust_remote_code=False,
            deploy_config_path=str(deploy),
            cli_overrides={},
            stage_overrides=None,
            strategy_config_path=None,
        )
        assert resolved.pipeline_config.model_type == "generic_diffusion"
        assert resolved.pipeline_config.stages[0].final_output_type == "actions"
        assert len(resolved.stage_configs) == 1
        assert resolved.stage_configs[0].diffusion_config.model_class_name == "Praxis1Pipeline"
    finally:
        hf_utils.is_diffusion_model.cache_clear()


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_video_and_expert_masks_hide_future_or_noised_history(depth):
    chunk, globals_ = 6, 3
    frames = (depth + 1) * chunk
    video = observed_context_mask(globals_, frames, depth * chunk, 1, chunk, torch.device("cpu")).dense[0, 0]
    assert video.shape == (globals_ + frames + depth * chunk,) * 2
    assert video[:, :globals_].all() and not video[:globals_, globals_:].any()
    assert video[globals_, globals_ : globals_ + chunk].all()
    cross, self_mask = trunk_masks(
        num_action_tokens=frames,
        num_future_per_chunk=2,
        chunk_size=chunk,
        num_clean_frames=depth * chunk,
        num_noised_frames=frames,
        tokens_per_frame=1,
        window_size=frames,
        num_global_tokens=globals_,
        device=torch.device("cpu"),
    )
    cross, self_mask = cross.dense[0, 0], self_mask.dense[0, 0]
    assert cross.shape == ((depth + 1) * 3 + frames, globals_ + frames + depth * chunk)
    assert cross[:, :globals_].all()
    # Keys are [globals, clean frames, noised frames]: the last action token sees its own noised chunk only.
    assert cross[-1, -chunk:].all()
    assert depth == 0 or not cross[-1, -frames:-chunk].any()
    if depth:
        assert not video[globals_, globals_ + chunk :].any()
        assert video[globals_ + chunk, globals_ + frames : globals_ + frames + chunk].all()
        assert not video[globals_ + chunk, globals_ + frames + chunk :].any()
        assert not video[globals_ + chunk, globals_ : globals_ + chunk].any()
        assert not cross[0, globals_ : globals_ + chunk].any()
        assert cross[1, globals_ : globals_ + chunk].all()
        assert not self_mask[0, 1] and self_mask[0, 0]


def test_block_mask_matches_dense_rule():
    mask = observed_context_mask(5, 12, 6, 7, 6, torch.device("cpu"))
    block = mask.block
    assert block.BLOCK_SIZE == (layers.MASK_BLOCK_SIZE, layers.MASK_BLOCK_SIZE)
    assert block.seq_lengths == (mask.q_len, mask.k_len)
    qi, ki = torch.arange(mask.q_len)[:, None], torch.arange(mask.k_len)[None]
    torch.testing.assert_close(block.mask_mod(None, None, qi, ki), mask.dense[0, 0])


def test_rotary_table_and_adapter():
    positions = torch.tensor([[0, 15, 16, 1000031]])
    angles = positions[..., None] * torch.tensor([1, 26000000**-0.5, 0, 0], dtype=torch.float64)
    expected = torch.stack([angles.cos(), angles.sin()], dim=-1).float()
    table = layers.RotaryTable(dim=8, effective_dim=4, base=26000000, table_size=16)(positions)
    torch.testing.assert_close(table, expected, rtol=1e-5, atol=1e-5)
    q = torch.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]], [[2, 4, 6, 8], [10, 12, 14, 16]]]], dtype=torch.float32)
    table = torch.tensor([[[[0.6, 0.8], [1, 0]], [[-1, 0], [0, -1]]]])
    rotated = torch.tensor([[[[-1.8, 2, 2.6, 4], [-5, 8, -7, -6]], [[-3.6, 4, 5.2, 8], [-10, 16, -14, -12]]]])
    torch.testing.assert_close(layers.apply_rope(q, table), rotated, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(layers.apply_rope(q[:, :1], table), rotated[:, :1], rtol=1e-6, atol=1e-6)


def test_attention_layout_and_grouped_heads_preserve_mask(monkeypatch):
    class CPUAttention(nn.Module):
        def forward(self, q, k, v, metadata):
            mask = metadata.extra["block_mask"]
            qi, ki = torch.arange(q.shape[1])[:, None], torch.arange(k.shape[1])[None]
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=mask.mask_mod(None, None, qi, ki),
                enable_gqa=True,
            ).transpose(1, 2)

    monkeypatch.setattr(layers, "OmniAttention", lambda **_: CPUAttention())
    q, k = torch.zeros(1, 2, 4, 1), torch.zeros(1, 1, 4, 1)
    mask = layers.SparseMask(lambda b, h, q, k: q >= k, 4, 4, torch.device("cpu"))
    actual = layers.PraxisAttention(2, 1, 1)(q, k, torch.arange(4.0).reshape(1, 1, 4, 1), mask)
    torch.testing.assert_close(actual, torch.tensor([0.0, 0.5, 1.0, 1.5]).reshape(1, 1, 4, 1).expand_as(q))
    with pytest.raises(ValueError, match="attention_impl"):
        layers.PraxisAttention(2, 1, 1, attention_impl="dense")


def test_sampler_noise_grid_and_guidance():
    before = torch.random.get_rng_state()
    video = seeded_normal((4, 8), 7, 0)
    torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
    torch.testing.assert_close(video, seeded_normal((4, 8), 7, 0), rtol=0, atol=0)
    assert not torch.equal(video, seeded_normal((4, 8), 7, 1))
    assert window_time_grid(2) == [1.0, 0.5, 0.0]
    assert window_time_grid(2, 8.0) == [1.0, 0.8888888888888888, 0.0]
    torch.testing.assert_close(guidance_combine(torch.tensor([2.0]), torch.tensor([1.0]), 3), torch.tensor([4.0]))
    for shift in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="t_shift"):
            SamplerSettings(t_shift=shift)
    assert not SamplerSettings(cfg_scale=1.0, cfg_video_scale=1.0).guided and SamplerSettings().guided


@pytest.mark.parametrize("missing_shard", [None, "k", "b"])
def test_checkpoint_maps_packed_projections_and_transposes_experts(missing_shard):
    model = nn.ModuleDict(
        {
            "attn": nn.ModuleDict({"in_proj": nn.Linear(2, 8, bias=False)}),
            "backbone": nn.ModuleDict(
                {
                    "input_projections": nn.ModuleDict(
                        {"TextEncodingTensor": nn.Sequential(nn.Linear(2, 2, bias=False))}
                    ),
                    "output_projections": nn.ModuleDict({"VideoTensor": nn.Linear(2, 2, bias=False)}),
                }
            ),
            "ffn": nn.ModuleDict(
                {
                    "0": nn.ModuleDict({"linear": nn.Linear(2, 6, bias=False)}),
                    "shared": nn.ModuleDict(
                        {"gate_up": nn.Linear(2, 6, bias=False), "down": nn.Linear(3, 2, bias=False)}
                    ),
                    "experts": nn.ModuleDict(
                        {
                            "routed_experts": nn.ParameterDict(
                                {
                                    "w13_weight": nn.Parameter(torch.empty(2, 4, 3)),
                                    "w2_weight": nn.Parameter(torch.empty(2, 3, 2)),
                                }
                            )
                        }
                    ),
                }
            ),
        }
    )
    swiglu_loader = Mock()
    model["ffn"]["0"]["linear"].weight.weight_loader = swiglu_loader
    qkv_loader = Mock()
    model["attn"]["in_proj"].weight.weight_loader = qkv_loader
    gate, down = torch.arange(24.0).reshape(2, 3, 4), torch.arange(12.0).reshape(2, 2, 3)
    weights = [
        (f"transformer.attn.in_proj.{key}_weight", torch.ones(rows, 2))
        for key, rows in [("q", 4), ("k", 2), ("v", 2)]
        if key != missing_shard
    ]
    weights += [("transformer.ffn.routed.gate_up_weight", gate), ("transformer.ffn.routed.down_weight", down)]
    weights += [
        (f"transformer.ffn.0.linear.{key}_weight", torch.full((3, 2), value))
        for key, value in [("a", 1.0), ("b", 2.0)]
        if key != missing_shard
    ]
    projection = torch.arange(4.0).reshape(2, 2)
    weights += [
        ("transformer.backbone.input_projections.TextEncodingTensor.proj.0.weight", projection),
        ("transformer.backbone.output_projections.VideoTensor.proj.weight", projection),
    ]
    shared_up, shared_down = torch.arange(12.0).reshape(2, 6), torch.arange(6.0).reshape(3, 2)
    weights += [("transformer.ffn.shared.gate_up", shared_up), ("transformer.ffn.shared.down", shared_down)]
    load_weights = Praxis1Transformer.load_weights
    if missing_shard is not None:
        with pytest.raises(ValueError, match="Incomplete Praxis-1 QKV/SwiGLU"):
            load_weights(model, iter(weights))
        return
    assert load_weights(model, iter(weights)) == set(model.state_dict())
    assert [call.args[2] for call in qkv_loader.call_args_list] == ["q", "k", "v"]
    assert [call.args[2] for call in swiglu_loader.call_args_list] == [0, 1]
    torch.testing.assert_close(swiglu_loader.call_args_list[0].args[1], torch.ones(3, 2))
    torch.testing.assert_close(swiglu_loader.call_args_list[1].args[1], torch.full((3, 2), 2.0))
    torch.testing.assert_close(model["ffn"]["shared"]["gate_up"].weight, shared_up.T)
    torch.testing.assert_close(model["ffn"]["shared"]["down"].weight, shared_down.T)
    torch.testing.assert_close(model["backbone"]["input_projections"]["TextEncodingTensor"][0].weight, projection)
    torch.testing.assert_close(model["backbone"]["output_projections"]["VideoTensor"].weight, projection)
    experts = model["ffn"]["experts"]["routed_experts"]
    torch.testing.assert_close(experts["w13_weight"], gate.transpose(1, 2))
    torch.testing.assert_close(experts["w2_weight"], down.transpose(1, 2))


def test_checkpoint_rejects_missing_and_unknown_weights():
    model = nn.Linear(2, 2, bias=False)
    with pytest.raises(ValueError, match="Missing Praxis-1"):
        Praxis1Transformer.load_weights(model, iter(()))
    with pytest.raises(ValueError, match="Unexpected Praxis-1"):
        Praxis1Transformer.load_weights(model, iter([("transformer.unknown", torch.zeros(1))]))
    worker = _worker(Mock(load_weights=Mock(return_value={"weight"})))
    assert worker.load_weights(iter([("transformer.weight", torch.ones(2, 2))])) == {"transformer.weight"}


def test_router_selects_with_bias_and_weights_without_it(monkeypatch):
    class Experts(nn.Module):
        def forward(self, x, router_logits):
            return torch.full_like(x, 2.0)

    factory = Mock(return_value=Experts())
    monkeypatch.setattr(layers, "FusedMoE", factory)
    monkeypatch.setattr(layers, "ReplicatedLinear", lambda *args, return_bias, **kw: nn.Linear(*args, **kw))
    layer = layers.MoEFeedForward(dim=4, hidden=4, num_experts=3, top_k=2, routed_scaling_factor=2.5)
    layer.expert_bias.copy_(torch.tensor([0.6, 0.0, 0.0]))
    weights, ids = layer.route(torch.zeros(1, 4), torch.log(torch.tensor([[1.0, 3.0, 9.0]])), 2, False)
    torch.testing.assert_close(ids, torch.tensor([[0, 2]], dtype=torch.int32))
    torch.testing.assert_close(weights, torch.tensor([[5 / 14, 9 / 14]]) * 2.5)
    assert factory.call_args.kwargs["top_k"] == 2 and factory.call_args.kwargs["routed_scaling_factor"] == 1.0
    assert factory.call_args.kwargs["custom_routing_function"] == layer.route
    for parameter in layer.shared.parameters():
        parameter.data.fill_(0.5)
    x = torch.ones(1, 2, 4)
    expected = 2.0 + layer.shared(x.reshape(-1, 4)).reshape(1, 2, 4)
    torch.testing.assert_close(layer(x), expected)
    assert layer.bfloat16().expert_bias.dtype == torch.float32


def test_decoding_clamps_actions_and_restores_client_pose_frame():
    stats = processing.NormStats(
        np.array([-0.2] * 9 + [0], np.float32),
        np.array([0.2] * 9 + [1], np.float32),
        np.zeros(10, np.float32),
        np.ones(10, np.float32),
    )
    rows: np.ndarray = np.zeros((24, 10), np.float32)
    rows[:, 3:9] = [1, 0, 0, 0, 1, 0]
    rows[::2, [0, 1, 2, 9]], rows[1::2, [0, 1, 2, 9]] = 2, -2
    tokens = torch.from_numpy(rows.reshape(1, 6, 40))
    norm, rel = processing.decode_actions(tokens, stats, depth=1)
    np.testing.assert_array_equal(norm, rows)
    np.testing.assert_allclose(rel[::2, :3], 0.2)
    np.testing.assert_allclose(rel[1::2, :3], -0.2)
    np.testing.assert_array_equal(rel[:, 9], np.tile([1, 0], 12))
    assert len(processing.decode_actions(tokens, stats, depth=0)[0]) == 21
    pose = np.array([0.5, 0.1, 0.3, 1, 0, 0, 0, 0.5], np.float32)
    frame = processing.RIG_FRAMES[""]
    model_pose = processing.pose8_to_model(pose, frame)
    commands = processing.rel10_to_wire(rel, model_pose, frame=frame)
    np.testing.assert_allclose(commands[0]["robot_command"]["pose"][:3], pose[:3] + 0.2, atol=1e-6)
    np.testing.assert_allclose(np.asarray(commands[0]["robot_command"]["pose"])[3:].reshape(3, 3), np.eye(3), atol=1e-6)


def test_tiles_fit_native_views_and_trim_client_fit_frames():
    rng = np.random.default_rng(7)
    ext, wrist = [rng.integers(0, 256, (3, 234, 416, 3), dtype=np.uint8) for _ in range(2)]
    obs = {"image.exterior": ext, "image.wrist": wrist}
    expected = (
        torch.from_numpy(np.concatenate((ext[:, 5:229], wrist[:, 5:229]), axis=2)).permute(3, 0, 1, 2)[None].float()
    )
    current = processing.current_clip(obs, tile_size=(224, 416))
    executed, _ = processing.executed_clip(obs, 2, tile_size=(224, 416))
    torch.testing.assert_close(current, expected[:, :, -1:], rtol=0, atol=0)
    torch.testing.assert_close(executed, expected[:, :, :2], rtol=0, atol=0)
    native = [rng.integers(0, 256, (2, 720, 1280, 3), dtype=np.uint8) for _ in range(2)]
    assert processing.tile_clip(*native, 224, 416).shape == (1, 3, 2, 224, 832)


@pytest.mark.parametrize(
    "size,digest",
    [
        ((720, 1280), "e9a050a35e7c8d6fff1940109412f0375d5772b33293327c5cbf3c14dc47db48"),
        ((224, 416), "56eeed2897141aa05a9b0dea3d702206b6f21e587104b2b369659eacce4239a2"),
        ((234, 416), "8046acf194e1b67ebedc19e5169035f9a3bf772192986c362ac706cadfbd5495"),
        ((288, 512), "cbec75a15f3995b104e092cc460303d5f4a3bdaa893e15db8aa01912106017ad"),
    ],
)
def test_canvas_matches_reference_preprocessing_hash(size, digest):
    # Exact bytes of the reference implementation's tiled canvas for these synthetic views: the fit
    # resize, centre trim and uint8 rounding must stay bit-identical to the policy's training input.
    rng = np.random.default_rng(0)
    ext, wrist = [rng.integers(0, 256, (2, *size, 3), dtype=np.uint8) for _ in range(2)]
    clip = processing.tile_clip(ext, wrist, 224, 416)
    assert clip.shape == (1, 3, 2, 224, 832)
    assert hashlib.sha256(clip.to(torch.uint8).numpy().tobytes()).hexdigest() == digest


@pytest.mark.parametrize("max_context,horizon", [(0, 24), (1, 6), (3, 24)])
def test_observed_context_continues_or_restarts_with_pose_history_fallback(max_context, horizon):
    stats = processing.NormStats(np.zeros(10), np.ones(10), np.zeros(10), np.ones(10))

    def encode(clip):
        return torch.zeros(1, 192, (clip.shape[2] - 1) // 4 + 1, 1, 1)

    state = processing.ServeSession(
        encode, stats, "cpu", max_context_chunks=max_context, exec_horizon=horizon, tile_size=(2, 4)
    )
    rows = torch.zeros(24, 10)
    rows[:, 3:9] = torch.tensor([1, 0, 0, 0, 1, 0])
    tokens = rows.reshape(1, 6, 40)
    for call in range(5):
        count = 1 if call == 0 else state.stream.expected_next_rows + 1
        obs = {
            "image.exterior": np.zeros((count, 2, 4, 3), np.uint8),
            "image.wrist": np.zeros((count, 2, 4, 3), np.uint8),
            # Current pose only: clean-chunk proprio must use the boundary fallback.
            "ee_pose": np.array([0.4 + call * 0.01, 0, 0.3, 1, 0, 0, 0], np.float32),
            "grip": np.array([0.5]),
            "descriptor": "praxis.sim",
            "reset_context": call == 4,
        }
        inputs = state.prepare(obs)
        depth = 0 if state.stateless or call == 4 else min(call, max_context)
        assert inputs.depth == depth and inputs.chunk_index == call
        assert inputs.proprio_all.shape == (1, depth + 1, 10)
        commands, normalized, relative = state.finish(inputs, tokens)
        assert normalized.shape == relative.shape == (21 if depth == 0 else 24, 10)
        assert len(commands) == min(horizon, len(normalized))
        assert state.chunk_index == call + 1


def test_failed_inference_discards_episode_context():
    worker = _worker()
    session = Mock(spec=pipeline_praxis1.Session, last_used=time.monotonic())
    session.step.side_effect = RuntimeError("failed inference")
    worker._sessions["episode"] = session
    req = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="task",
                request_id="request",
                sampling_params=OmniDiffusionSamplingParams(extra_args={"session_id": "episode", "robot_obs": {}}),
            )
        ]
    )
    with pytest.raises(RuntimeError, match="failed inference"):
        worker._forward(req)
    assert not worker._sessions


@pytest.mark.parametrize("embodiment,width", [("single_arm", 13), ("bimanual", 26)])
@pytest.mark.parametrize("session_id", ["a", "missing"])
def test_worker_cleanup_releases_only_named_session_without_inference(session_id, embodiment, width):
    worker = _worker()
    worker.embodiment = embodiment
    worker.max_sessions = 2
    worker._sessions = {"a": Mock(spec=pipeline_praxis1.Session), "b": Mock(spec=pipeline_praxis1.Session)}
    request = OmniDiffusionRequest(
        prompt="",
        request_id="close-a",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"session_id": session_id, "close_session": True}),
    )
    output = worker._forward(DiffusionRequestBatch(requests=[request]))
    assert set(worker._sessions) == ({"b"} if session_id == "a" else {"a", "b"})
    assert output.output["payload"]["actions"].shape == (0, width)
    worker._forward(DiffusionRequestBatch(requests=[request]))  # idempotent
    worker._sessions["b"].step.assert_not_called()
    worker.drop_session_state("b")
    assert "b" not in worker._sessions


def test_vae_normalizes_pixels_and_folds_latent_patches():
    vae = pipeline_praxis1.Praxis1VAE.__new__(pipeline_praxis1.Praxis1VAE)
    nn.Module.__init__(vae)
    vae.register_buffer("mean", torch.zeros(48))
    vae.register_buffer("inv_std", torch.full((48,), 2.0))
    vae.model = nn.Module()
    vae.model.encode = Mock()
    vae.model.encode.return_value.latent_dist.mode.return_value = torch.ones(1, 48, 1, 4, 4)
    vae.model.decode = Mock(side_effect=lambda z: Mock(sample=z[:, :3].float() / 2))
    pixels = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 1, 16, 16).expand(1, 3, 1, 16, 16)
    latent = vae.encode(pixels)
    normalized = vae.model.encode.call_args.args[0]
    assert normalized.dtype == torch.bfloat16
    torch.testing.assert_close(normalized.float(), (pixels.float() / 255 * 2 - 1).bfloat16().float(), rtol=0, atol=0)
    assert latent.shape == (1, 192, 1, 2, 2) and latent.dtype == torch.bfloat16
    assert torch.all(latent == 2)
    frames = vae.decode(latent)
    decoded = vae.model.decode.call_args.args[0]
    assert decoded.shape == (1, 48, 1, 4, 4) and torch.all(decoded == 1)
    assert frames.shape == (1, 3, 4, 4) and frames.dtype == torch.uint8 and torch.all(frames == 191)


def test_text_conditioning_selects_layer_and_trims_padding(monkeypatch):
    class TinyText(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, input_ids, **_):
            return BaseModelOutputWithPast(
                hidden_states=tuple(torch.full((*input_ids.shape, 4), float(i)) for i in range(5))
            )

    tokenizer = Mock(
        return_value=BatchEncoding(
            {
                "input_ids": torch.zeros(1, 544, dtype=torch.long),
                "attention_mask": torch.tensor([[1] * 512 + [0] * 32]),
            }
        )
    )
    monkeypatch.setattr(pipeline_praxis1.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer))
    monkeypatch.setattr(pipeline_praxis1.AutoConfig, "from_pretrained", Mock(return_value=Mock()))
    factory = Mock(return_value=TinyText())
    monkeypatch.setattr(pipeline_praxis1, "create_transformers_model_with_vllm_linears", factory)
    monkeypatch.setattr(
        pipeline_praxis1,
        "safetensors_weights_iterator",
        lambda *_: iter(
            [("model.weight", torch.ones(1)), ("mtp.fc.weight", torch.ones(1)), ("lm_head.weight", torch.ones(1))]
        ),
    )
    encoder = pipeline_praxis1.Praxis1TextEncoder("/local/text", torch.device("cpu"))
    result = encoder.encode(["pick"])[0]
    assert factory.call_args.args[0] is pipeline_praxis1.AutoModel
    assert factory.call_args.kwargs["prefix"] == "text_encoder.model"
    assert result.shape == (1, 512, 4) and torch.all(result == 2)
    assert tokenizer.call_args.kwargs["pad_to_multiple_of"] == 32
    assert tokenizer.call_args.kwargs["max_length"] == 544
    assert not encoder.model.training and not encoder.model.weight.requires_grad
    tokenizer.return_value["attention_mask"] = torch.tensor([[1] * 513 + [0] * 31])
    encoder.model.forward = Mock(side_effect=AssertionError("oversized task reached the model"))
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        encoder.encode(["oversized"])
    tokenizer.side_effect = AssertionError("oversized task reached tokenization")
    for prompt in ("x" * 65537, "é" * 32769):
        with pytest.raises(ValueError, match="64 KiB"):
            encoder.encode([prompt])


def test_model_construction_rejects_invalid_shapes():
    from vllm_omni.diffusion.models.praxis1.action_expert import ActionExpert

    with pytest.raises(ValueError, match="effective_dim"):
        layers.RotaryTable(4, 8, 100, 16)
    expert = {
        "dim": 8,
        "num_layers": 3,
        "proprio_patch_dim": 10,
        "trunk_dim": 8,
        "rope": {"effective_dim": 8, "base": 100, "table_size": 16},
        "head_dim": 8,
    }
    with pytest.raises(ValueError, match="num_layers must be even"):
        ActionExpert(expert)
    assert Praxis1Transformer.from_config.__func__ is not None
