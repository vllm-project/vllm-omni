# SPDX-License-Identifier: Apache-2.0
"""Exercise the real admission/rollout methods with CPU transformer/decoder doubles.

Extracting the methods avoids loading CUDA-only import-time dependencies on CPU
hosts. The generation loop itself is compiled unchanged from the pipeline file.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_inputs import prepare_action_values
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.geometry import (
    Cosmos3NanoSimBimanualResolutionPolicy,
    resolve_cosmos3_nano_sim_bimanual_geometry,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.normalizer import ActionAffineNormalizer
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.state_cosmos3_nano_sim_bimanual import (
    Cosmos3NanoSimBimanualSessionFingerprint,
    Cosmos3NanoSimBimanualSessionState,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.tick_adapter import (
    build_cosmos3_nano_sim_bimanual_action_control,
    parse_cosmos3_nano_sim_bimanual_tick,
)
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionRequestRejectedError
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionChunkMetadata, ARDiffusionTickRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "vllm_omni/diffusion/models/cosmos3_nano_sim_bimanual"


def pipeline_methods() -> type:
    namespace = {
        "__name__": __name__,
        "torch": torch,
        "time": time,
        "math": math,
        "hashlib": hashlib,
        "json": json,
        "Mapping": Mapping,
        "dataclass": dataclass,
        "Any": Any,
        "ARDiffusionRequestRejectedError": ARDiffusionRequestRejectedError,
        "ARDiffusionTickRequest": ARDiffusionTickRequest,
        "ARDiffusionChunkMetadata": ARDiffusionChunkMetadata,
        "Cosmos3NanoSimBimanualSessionFingerprint": Cosmos3NanoSimBimanualSessionFingerprint,
        "resolve_cosmos3_nano_sim_bimanual_geometry": resolve_cosmos3_nano_sim_bimanual_geometry,
        "parse_cosmos3_nano_sim_bimanual_tick": parse_cosmos3_nano_sim_bimanual_tick,
        "prepare_action_values": prepare_action_values,
        "DiffusionMediaOutput": DiffusionMediaOutput,
        "VideoMediaOutput": VideoMediaOutput,
        "VideoTensorSpec": VideoTensorSpec,
        "VideoTensorLayout": VideoTensorLayout,
        "VideoTensorEncoding": VideoTensorEncoding,
        "VideoValueRange": VideoValueRange,
        "DiffusionOutput": SimpleNamespace,
    }
    tree = ast.parse((SOURCE / "pipeline_cosmos3_nano_sim_bimanual.py").read_text())
    body = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    helpers = {"_first_not_none", "_admission_int", "_admission_float", "_ActionRequestContract", "_ActionConditioning"}
    body += [node for node in tree.body if getattr(node, "name", None) in helpers]
    pipeline = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Cosmos3NanoSimBimanualPipeline"
    )
    methods = {
        "forward",
        "_forward_impl",
        "_parse_tick",
        "_resolve_request_fps",
        "_resolve_request_geometry",
        "_resolve_requested_pixel_frames",
        "_validate_conditioning_request",
        "_prepare_conditioning",
        "_conditioning_fingerprint",
        "_build_prompt_tokens",
        "_run_chunk",
        "_prefill_first_frame",
        "_denoise_chunk",
        "_validate_session_mode",
        "_prepare_raw_action",
        "_fingerprint",
        "_resolve_action_layout",
        "_actions_for_frames",
        "_build_output",
        "_commit_clean_chunk",
        "_prompt_token_limit",
    }
    selected = [node for node in pipeline.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    body.append(ast.ClassDef(name="Pipeline", bases=[], keywords=[], decorator_list=[], body=selected))
    utils = ast.parse((SOURCE / "utils.py").read_text())
    body += [
        node
        for node in utils.body
        if getattr(node, "name", None) in ("iter_ar_chunk_ranges", "iter_clean_commit_frames", "prompt_token_hash")
    ]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace["Pipeline"]


@pytest.fixture(autouse=True)
def guardrail_gate(monkeypatch):
    # Execute the real gate without importing GPU guardrail implementations.
    path = SOURCE.parent / "cosmos3/guardrails.py"
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    body += [node for node in tree.body if getattr(node, "name", None) == "is_guardrails_enabled"]
    module = ModuleType("vllm_omni.diffusion.models.cosmos3.guardrails")
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)


def fake_pipeline(*, prefix: bool = True) -> tuple[Any, Any]:
    from test_cookbook import manifest

    pipe = pipeline_methods()()
    pipe.manifest = manifest()
    pipe._states = {}
    pipe._bound_session_id = None
    pipe._ar_diffusion_kv_state = None
    pipe._decode_queue = None
    pipe._distilled_num_steps = 4
    pipe.default_domain_id = 15
    pipe.default_fps = 30
    pipe.checkpoint_id = pipe.manifest.checkpoint_id
    pipe._MAIN_BRANCH = "main"
    pipe.od_config = SimpleNamespace(model_config={"guardrails": True})
    pipe.resolution_policy = Cosmos3NanoSimBimanualResolutionPolicy(default_resolution=(32, 32))
    pipe.action_normalizers = {
        name: ActionAffineNormalizer.from_contract(contract.normalizer)
        for name, contract in pipe.manifest.require_action_schema().embodiments.items()
    }
    pipe.transformer = SimpleNamespace(latent_channel_size=3)
    pipe.device = "cpu"
    pipe.dtype = torch.float32
    pipe._get_sp_param = lambda sp, key, default=None: sp.extra_args.get(key, getattr(sp, key, default))
    pipe._tokenize_prompt = Mock(return_value=(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2)))
    pipe._initial_condition_latent = Mock(return_value=torch.zeros(1, 3, 1, 2, 2) if prefix else None)
    pipe._sync_for_tick_timing = Mock()
    pipe._drop_session = Mock(side_effect=lambda name: pipe._states.pop(name, None))
    state = Cosmos3NanoSimBimanualSessionState("test")

    def create_state(name: str) -> Any:
        pipe._states[name] = state
        return state

    def decode(state: Any, latent: torch.Tensor) -> torch.Tensor:
        frames = latent.shape[2] * 4 - (0 if state.vae_decoder_initialized else 3)
        state.record_incremental_decode(input_frames=latent.shape[2], feature_cache=[])
        return torch.zeros(1, 3, frames, 32, 32)

    pipe._get_or_create_state = create_state
    pipe._ensure_text_kv = Mock(return_value=[])
    pipe._resolve_seed = Mock(return_value=42)
    pipe._transformer_forward = Mock(side_effect=lambda _state, latent, *_args, **_kw: SimpleNamespace(video=latent))
    pipe._sample_distilled = Mock(side_effect=lambda velocity, noise, **_kw: velocity(noise, torch.tensor([1.0])))
    pipe._commit_clean_frame = Mock()
    pipe._decode_live_latents = Mock(side_effect=decode)
    pipe._decode_latents = Mock(side_effect=lambda latent: torch.zeros(1, 3, (latent.shape[2] - 1) * 4 + 1, 32, 32))
    pipe._timed_tick_stage = lambda *_args, **_kwargs: nullcontext()
    return pipe, state


def request(frames: int = 61, *, output_type: str | None = None, **extra: Any) -> Any:
    return SimpleNamespace(
        prompts=[{"prompt": "test"}],
        sampling_params=SimpleNamespace(
            extra_args={"session_id": "test", "reset": True, "close_session": True, "guardrails": False, **extra},
            output_type=output_type,
            num_frames=frames,
            num_inference_steps=4,
            generator=None,
        ),
    )


@pytest.mark.parametrize("frames", [5, 9, 13, 17, 61, 901, 1801])
@pytest.mark.parametrize("prefix", [True, False])
def test_full_video_decodes_incrementally_without_retaining_latents(frames: int, prefix: bool) -> None:
    pipe, state = fake_pipeline(prefix=prefix)
    media = pipe.forward(request(frames)).media.video
    media.validate()
    output = media.tensor
    assert media.spec.layout == VideoTensorLayout.BCTHW
    assert media.spec.value_range == VideoValueRange.NEGATIVE_ONE_TO_ONE
    assert output.shape == (1, 3, frames, 32, 32) and output.device.type == "cpu"
    assert state.latents == []
    assert state.max_vae_decode_input_frames <= 4
    assert state.next_frame_idx == (frames - 1) // 4 + 1
    # The terminal latent has no downstream reader and must not enter KV history.
    assert [call.kwargs["frame_idx"] for call in pipe._commit_clean_frame.call_args_list] == list(
        range(state.next_frame_idx - 1)
    )
    pipe._decode_latents.assert_not_called()
    assert state.terminal


def test_latent_output_keeps_existing_path() -> None:
    pipe, state = fake_pipeline()
    output = pipe.forward(request(output_type="latent")).output["video"]
    assert output.shape == (1, 3, 16, 2, 2) and state.latents
    pipe._decode_live_latents.assert_not_called()
    pipe._decode_latents.assert_not_called()


@pytest.mark.parametrize("frames", [5, 17, 21, 61])
@pytest.mark.parametrize("prefix", [True, False])
def test_batched_clean_commit_preserves_terminal_frame_and_action_slices(frames, prefix):
    pipe, state = fake_pipeline(prefix=prefix)
    pipe.clean_commit_mode = "batched"
    pipe._set_mixed_precision_step = Mock()
    pipe._reset_mixed_precision = Mock()
    pipe.forward(request(frames))
    committed = [call.kwargs["frame_idx"] for call in pipe._commit_clean_frame.call_args_list]
    for call in pipe._transformer_forward.call_args_list:
        if not call.kwargs.get("frame_causal"):
            continue
        count = call.args[1].shape[2]
        start = call.kwargs["frame_start"]
        committed.extend(range(start, start + count))
        assert call.kwargs["condition_vision"] and call.kwargs["commit_current"]
        assert (
            call.kwargs["conditioning_kwargs"]["action_latents"].shape[1]
            == count * pipe.manifest.action_tokens_per_frame
        )
        assert call.kwargs["conditioning_kwargs"]["null_action_frame_indexes"] == tuple(range(count))
    assert committed == list(range(state.next_frame_idx - 1))
    assert pipe._set_mixed_precision_step.call_count == pipe._reset_mixed_precision.call_count


def test_batched_clean_commit_resets_precision_on_failure():
    pipe, _ = fake_pipeline()
    pipe.clean_commit_mode = "batched"
    pipe._set_mixed_precision_step = Mock()
    pipe._reset_mixed_precision = Mock()

    def forward(_state, latent, *_args, **kwargs):
        if kwargs.get("frame_causal"):
            raise RuntimeError("clean refresh failed")
        return SimpleNamespace(video=latent)

    pipe._transformer_forward.side_effect = forward
    with pytest.raises(RuntimeError, match="clean refresh failed"):
        pipe.forward(request(17))
    pipe._reset_mixed_precision.assert_called_once()


def test_full_video_keeps_guardrail_output_route() -> None:
    pipe, _ = fake_pipeline()
    assert pipe.forward(request(guardrails=True)).output["video"].shape[2] == 61
    pipe._decode_live_latents.assert_called()


def test_terminal_tick_keeps_incremental_path() -> None:
    pipe, state = fake_pipeline()
    output = pipe.forward(request(chunk_only=True, num_latent_frames=4)).media.video.tensor
    assert output.shape[2] == 17 and state.next_frame_idx == 5
    assert state.latents == []
    pipe._decode_live_latents.assert_called_once()


def test_unaligned_request_does_not_mutate_session() -> None:
    pipe, _ = fake_pipeline()
    with pytest.raises(ARDiffusionRequestRejectedError, match="temporal_compression_factor"):
        pipe.forward(request(62))
    assert pipe._states == {}
    pipe._drop_session.assert_not_called()
    pipe._sample_distilled.assert_not_called()


def test_failed_decoder_releases_session() -> None:
    pipe, _ = fake_pipeline()
    pipe._decode_live_latents.side_effect = RuntimeError("decoder failure")
    with pytest.raises(RuntimeError, match="decoder failure"):
        pipe.forward(request())
    assert pipe._states == {}


@pytest.mark.parametrize("space", ["raw", "model"])
def test_action_space_reaches_transformer(space: str) -> None:
    pipe, state = fake_pipeline()
    action = torch.arange(16 * 29, dtype=torch.float32).reshape(16, 29) / 100
    pipe.forward(request(17, action=action, action_space=space, action_mode="forward_dynamics"))
    expected = pipe.action_normalizers["agibotworld"].normalize(action) if space == "raw" else action
    actual = pipe._transformer_forward.call_args.kwargs["conditioning_kwargs"]["action_latents"]
    torch.testing.assert_close(actual[0, :, :29], expected)
    assert not actual[0, :, 29:].count_nonzero()
    assert state.fingerprint.action_space == space


def test_action_space_change_rejects_continuation_without_mutation() -> None:
    pipe, state = fake_pipeline()
    pipe.forward(request(chunk_only=True, close_session=False, action_space="raw"))
    pipe._initial_condition_latent.return_value = None
    decode_count = pipe._decode_live_latents.call_count
    with pytest.raises(ARDiffusionRequestRejectedError, match="action_space"):
        pipe.forward(request(chunk_only=True, reset=False, close_session=False, action_space="model"))
    assert pipe._states["test"] is state and state.next_frame_idx == 5
    assert pipe._decode_live_latents.call_count == decode_count


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"action_space": "unknown"}, "action_space"),
        ({"action_mode": "forward_dynamics"}, "requires action"),
        ({"action": [[True] * 29] * 60}, "booleans"),
        ({"action": [[0.0] * 29] * 16}, "action length"),
    ],
)
def test_invalid_actions_do_not_mutate_session(extra: dict, match: str) -> None:
    pipe, _ = fake_pipeline()
    with pytest.raises(ARDiffusionRequestRejectedError, match=match):
        pipe.forward(request(**extra))
    assert pipe._states == {}
    pipe._drop_session.assert_not_called()
    pipe._sample_distilled.assert_not_called()


def test_typed_tick_still_uses_raw_actions_and_media_metadata() -> None:
    pipe, _ = fake_pipeline()
    tick = ARDiffusionTickRequest(
        session_id="test",
        request_id="tick-0",
        chunk_index=0,
        applied_event_ids=(1,),
        reset=True,
        controls=(build_cosmos3_nano_sim_bimanual_action_control(torch.zeros(16, 29)),),
    )
    sp = request().sampling_params
    typed_inputs = pipe._parse_tick(tick)
    conditioning = pipe._validate_conditioning_request(sp, typed_inputs)
    assert conditioning.action_space == "raw" and typed_inputs.action.shape == (16, 29)
    output = pipe._build_output(torch.zeros(1, 3, 17, 32, 32), sampling_params=sp, typed_tick=tick, stage_durations={})
    assert output.media.metadata["ar_diffusion"] == ARDiffusionChunkMetadata.from_tick(tick).to_dict()
    sp.extra_args["action_space"] = "model"
    with pytest.raises(ARDiffusionRequestRejectedError, match="typed ticks require raw"):
        pipe._validate_conditioning_request(sp, typed_inputs)


def test_prompt_override_does_not_raise_bimanual_checkpoint_limit() -> None:
    pipe, _ = fake_pipeline()
    limit = pipe.manifest.text_cache_max_len
    pipe._tokenize_prompt.return_value = (torch.ones(1, limit + 1, dtype=torch.long), torch.ones(1, limit + 1))
    with pytest.raises(ARDiffusionRequestRejectedError, match="prompt exceeds token limit"):
        pipe.forward(request(max_prompt_tokens=limit + 100))
    assert pipe._states == {}
    pipe._ensure_text_kv.assert_not_called()
    pipe._sample_distilled.assert_not_called()
