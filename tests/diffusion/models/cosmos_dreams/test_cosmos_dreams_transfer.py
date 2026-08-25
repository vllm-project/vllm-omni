# SPDX-License-Identifier: Apache-2.0
"""Fixture-independent Phase T1 contracts for Cosmos-Dreams-Transfer."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

from examples.offline_inference.cosmos_dreams.cosmos_dreams_transfer import (
    _resolve_fps,
    _resolve_num_frames,
    _unwrap_video,
)
from tests.diffusion.models.cosmos_dreams.test_cosmos_dreams_core import _artifact
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import COSMOS3_TRANSFER_SYSTEM_PROMPT
from vllm_omni.diffusion.models.cosmos_dreams import pipeline_cosmos_dreams_transfer as transfer_pipeline_module
from vllm_omni.diffusion.models.cosmos_dreams.conditioning_control import (
    build_shared_vision_mrope_position_ids,
    pack_pure_vision_tokens,
    unpack_pure_vision_tokens,
)
from vllm_omni.diffusion.models.cosmos_dreams.config import CosmosDreamsManifest
from vllm_omni.diffusion.models.cosmos_dreams.control_contract import (
    CosmosDreamsActionConditioning,
    CosmosDreamsControlVideoConditioning,
)
from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams import CosmosDreamsPipeline
from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams_transfer import (
    CosmosDreamsTransferPipeline,
    _TransferConditioning,
    _TransferRequestContract,
    format_cosmos_dreams_transfer_prompt,
)
from vllm_omni.diffusion.models.cosmos_dreams.state_cosmos_dreams import CosmosDreamsSessionState
from vllm_omni.diffusion.models.cosmos_dreams.transformer_cosmos_dreams_transfer import (
    CosmosDreamsTransferTransformer,
)
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionRequestRejectedError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _control_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "mode": "control_video",
        "hints": ["edge", "blur", "depth", "seg"],
        "transfer_control_attention_mode": "causal_control_with_rgb_history",
        "share_vision_temporal_positions": True,
        "system_prompt_id": "cosmos3_transfer_v1",
        "emphasize_control_in_prompt": True,
        "no_eviction": True,
    }
    payload.update(overrides)
    return payload


def _v3_artifact(**conditioning_overrides: Any) -> dict[str, Any]:
    artifact = deepcopy(_artifact())
    artifact["schema_version"] = 3
    artifact["deploy_resolution"] = [480, 832]
    artifact.pop("action_schema")
    artifact["conditioning"] = _control_payload(**conditioning_overrides)
    return artifact


def _config(artifact: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        model_config={},
        tf_model_config={"cosmos_dreams": artifact},
        custom_pipeline_args={},
    )


def _transfer_manifest(**overrides: Any) -> CosmosDreamsManifest:
    values = {
        "schema_version": 3,
        "height": 480,
        "width": 832,
        "window_frames": 64,
        "sink_frames": 0,
        "conditioning": CosmosDreamsControlVideoConditioning.model_validate(_control_payload()),
    }
    values.update(overrides)
    return CosmosDreamsManifest(**values)


def _pipeline_stub(manifest: CosmosDreamsManifest | None = None) -> SimpleNamespace:
    stub = SimpleNamespace(manifest=manifest or _transfer_manifest())

    def get_sp_param(sp: Any, key: str, default: Any = None) -> Any:
        extra = getattr(sp, "extra_args", None)
        if isinstance(extra, dict) and extra.get(key) is not None:
            return extra[key]
        value = getattr(sp, key, None)
        return default if value is None else value

    stub._get_sp_param = get_sp_param
    stub._request_param = MethodType(CosmosDreamsTransferPipeline._request_param, stub)
    return stub


def _sampling_params(*, num_frames: int = 97, extra_args: dict[str, Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(num_frames=num_frames, extra_args={"edge": True, **(extra_args or {})})


def test_manifest_v3_is_additive_and_v2_remains_byte_compatible() -> None:
    v2_artifact = _artifact()
    v2 = CosmosDreamsManifest.from_od_config(_config(v2_artifact), require_explicit=True)
    v3 = CosmosDreamsManifest.from_od_config(_config(_v3_artifact()), require_explicit=True)

    assert v2.schema_version == 2
    assert v2.action_schema is v2.conditioning
    assert v2.tokens_per_frame == 924
    assert v3.schema_version == 3
    assert v3.action_schema is None
    assert v3.require_control_video_conditioning().mode == "control_video"
    assert v3.vision_tokens_per_frame == 390
    assert v3.conditioning_tokens_per_frame == 0
    assert v3.tokens_per_frame == 390


def test_manifest_v3_rejects_missing_unknown_and_wrong_discriminator_fields() -> None:
    missing = _v3_artifact()
    missing.pop("conditioning")
    with pytest.raises(ValueError, match="missing conditioning"):
        CosmosDreamsManifest.from_od_config(_config(missing), require_explicit=True)

    unknown = _v3_artifact()
    unknown["action_schema"] = _artifact()["action_schema"]
    with pytest.raises(ValueError, match="unknown fields.*action_schema"):
        CosmosDreamsManifest.from_od_config(_config(unknown), require_explicit=True)

    with pytest.raises(ValueError, match="share_vision_temporal_positions"):
        CosmosDreamsManifest.from_od_config(
            _config(_v3_artifact(share_vision_temporal_positions=False)),
            require_explicit=True,
        )
    with pytest.raises(ValueError, match="JSON boolean true"):
        CosmosDreamsManifest.from_od_config(
            _config(_v3_artifact(no_eviction=1)),
            require_explicit=True,
        )
    with pytest.raises(ValueError, match="hints must exactly match"):
        CosmosDreamsManifest.from_od_config(
            _config(_v3_artifact(hints=["edge", "depth"])),
            require_explicit=True,
        )

    with pytest.raises(ValueError, match="mode"):
        CosmosDreamsManifest.from_od_config(
            _config(_v3_artifact(mode="unknown")),
            require_explicit=True,
        )


def test_manifest_v3_dispatches_the_td4_action_branch_without_changing_v2() -> None:
    artifact = _v3_artifact()
    artifact["conditioning"] = {"mode": "action", **_artifact()["action_schema"]}

    manifest = CosmosDreamsManifest.from_od_config(_config(artifact), require_explicit=True)

    assert isinstance(manifest.conditioning, CosmosDreamsActionConditioning)
    assert manifest.action_schema is manifest.conditioning
    assert manifest.tokens_per_frame == 394


@pytest.mark.parametrize(
    ("artifact_factory", "forbidden_field", "forbidden_value"),
    [
        (_v3_artifact, "action_schema", _artifact()["action_schema"]),
        (_artifact, "conditioning", _control_payload()),
    ],
)
@pytest.mark.parametrize(
    ("config_attr", "nested"),
    [
        ("model_config", False),
        ("model_config", True),
        ("custom_pipeline_args", False),
        ("custom_pipeline_args", True),
    ],
)
def test_deployment_rejects_all_conditioning_artifact_field_aliases(
    artifact_factory: Any,
    forbidden_field: str,
    forbidden_value: Any,
    config_attr: str,
    nested: bool,
) -> None:
    artifact = artifact_factory()
    override = {forbidden_field: forbidden_value}
    if nested:
        override = {"cosmos_dreams": override}
    config_values = {
        "model_config": {},
        "tf_model_config": {"cosmos_dreams": artifact},
        "custom_pipeline_args": {},
    }
    config_values[config_attr] = override

    with pytest.raises(ValueError, match=rf"{forbidden_field} may only come from transformer/config.json"):
        CosmosDreamsManifest.from_od_config(SimpleNamespace(**config_values), require_explicit=True)


def test_transfer_init_requires_eager_unquantized_dense_execution() -> None:
    manifest = _transfer_manifest()
    stub = SimpleNamespace(manifest=manifest)
    with pytest.raises(ValueError, match="enforce_eager=True"):
        CosmosDreamsTransferPipeline._init_conditioning(
            stub,
            SimpleNamespace(enforce_eager=False, diffusion_kv_cache_dtype=None),
        )
    with pytest.raises(ValueError, match="quantized"):
        CosmosDreamsTransferPipeline._init_conditioning(
            stub,
            SimpleNamespace(enforce_eager=True, diffusion_kv_cache_dtype="fp8"),
        )
    CosmosDreamsTransferPipeline._init_conditioning(
        stub,
        SimpleNamespace(enforce_eager=True, diffusion_kv_cache_dtype=None),
    )


def test_control_and_rgb_mrope_ids_are_byte_identical() -> None:
    kwargs = {
        "frame_start": 5,
        "num_frames": 4,
        "grid_h": 2,
        "grid_w": 3,
        "text_temporal_offset": 17,
        "temporal_modality_margin": 15_000,
        "fps": 15.0,
        "base_fps": 24.0,
        "temporal_compression_factor": 4,
    }
    control_ids = build_shared_vision_mrope_position_ids(**kwargs)
    rgb_ids = build_shared_vision_mrope_position_ids(**kwargs)

    assert torch.equal(control_ids, rgb_ids)
    assert control_ids.numpy().tobytes() == rgb_ids.numpy().tobytes()
    assert tuple(control_ids.shape) == (3, 24)


def test_transfer_transformer_needs_no_action_modules_or_weights() -> None:
    manifest = _transfer_manifest()
    CosmosDreamsTransferTransformer._validate_conditioning_config(SimpleNamespace(manifest=manifest, action_gen=False))
    with pytest.raises(ValueError, match="action_gen=False"):
        CosmosDreamsTransferTransformer._validate_conditioning_config(
            SimpleNamespace(manifest=manifest, action_gen=True)
        )

    transformer = CosmosDreamsTransferTransformer.__new__(CosmosDreamsTransferTransformer)
    nn.Module.__init__(transformer)
    transformer.register_parameter("vision_weight", nn.Parameter(torch.ones(1)))
    transformer.validate_loaded_weights({"transformer.vision_weight"})
    assert CosmosDreamsTransferPipeline._is_action_weight("transformer.action_proj_in.weight")
    assert not CosmosDreamsTransferPipeline._is_action_weight("transformer.proj_in.weight")


def test_transfer_token_pack_is_pure_vision() -> None:
    vision = torch.arange(2 * 3 * 4, dtype=torch.float32).view(1, 2, 3, 4)
    packed = pack_pure_vision_tokens(vision)
    unpacked = unpack_pure_vision_tokens(packed, num_frames=2, vision_tokens_per_frame=3)

    assert tuple(packed.shape) == (1, 6, 4)
    torch.testing.assert_close(unpacked, vision)


def test_transfer_prompt_matches_reference_full_clip_format_and_forces_system_prompt() -> None:
    expected = (
        "A city street. The video is 6.5 seconds long and is of 15 FPS. "
        "This video is of 480x832 resolution. Follow the edge control video precisely: shape, contour, "
        "silhouette, position, and motion of every visible structure must align with the edge signal at every frame."
    )
    assert (
        format_cosmos_dreams_transfer_prompt(
            "A city street.",
            hint="edge",
            num_frames=97,
            fps=15.0,
            height=480,
            width=832,
        )
        == expected
    )
    assert (
        format_cosmos_dreams_transfer_prompt(
            "  A city street.   ",
            hint="edge",
            num_frames=97,
            fps=15.0,
            height=480,
            width=832,
        )
        == expected
    )

    captured: dict[str, Any] = {}
    request = _TransferRequestContract("edge", {}, None, 97)
    stub = SimpleNamespace(
        manifest=_transfer_manifest(),
        _validate_conditioning_request=lambda *args, **kwargs: request,
        _tokenize_prompt=lambda text, max_sequence_length, use_system_prompt, system_prompt: (
            captured.update(
                text=text,
                max_sequence_length=max_sequence_length,
                use_system_prompt=use_system_prompt,
                system_prompt=system_prompt,
            )
            or (torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.long))
        ),
    )
    CosmosDreamsTransferPipeline._build_prompt_tokens(
        stub,
        "A city street.",
        sampling_params=SimpleNamespace(),
        prompt_data={},
        fps=15.0,
    )
    assert captured["text"] == expected
    assert captured["use_system_prompt"] is True
    assert captured["system_prompt"] == COSMOS3_TRANSFER_SYSTEM_PROMPT
    assert captured["max_sequence_length"] == 1 << 30

    assert (
        format_cosmos_dreams_transfer_prompt(
            "A city street.",
            hint="edge",
            num_frames=97,
            fps=14.6,
            height=480,
            width=832,
        )
        == expected
    )


def test_transfer_example_null_sampling_fields_fall_back_by_precedence() -> None:
    assert _resolve_num_frames(None, {"num_frames": None}) == 97
    assert _resolve_num_frames(113, {"num_frames": None}) == 113
    assert _resolve_fps(None, {"fps": None, "frame_rate": 12.0}) == 12.0
    assert _resolve_fps(None, {"fps": None, "frame_rate": None}) == 15.0
    assert _resolve_fps(24.0, {"fps": None, "frame_rate": 12.0}) == 24.0


def test_transfer_example_rejects_explicit_zero_sampling_values() -> None:
    with pytest.raises(ValueError, match="num_frames must be positive"):
        _resolve_num_frames(0, {"num_frames": 113})
    with pytest.raises(ValueError, match="FPS must be positive"):
        _resolve_fps(0.0, {"fps": 12.0})


def test_transfer_example_unwraps_valid_envelopes_and_rejects_malformed_ones() -> None:
    video = object()
    assert _unwrap_video({"payload": {"video": video}}) is video

    with pytest.raises(ValueError, match="unsupported output mapping"):
        _unwrap_video({"metadata": {"request_id": "test"}})
    with pytest.raises(ValueError, match="empty video output list"):
        _unwrap_video([])

    cyclic: dict[str, Any] = {}
    cyclic["payload"] = cyclic
    with pytest.raises(ValueError, match="cyclic video output envelope"):
        _unwrap_video(cyclic)


def test_transfer_input_video_fps_overrides_request_fps_for_the_full_run_contract() -> None:
    stub = _pipeline_stub()
    sp = SimpleNamespace(fps=30.0, extra_args={"edge": True})
    prompt_data = {"additional_information": {"transfer_input_fps": 12.0}}

    fps = CosmosDreamsTransferPipeline._resolve_request_fps(stub, sp, prompt_data)

    assert fps == 12.0
    assert "8.1 seconds long and is of 12 FPS" in format_cosmos_dreams_transfer_prompt(
        "A city street.",
        hint="edge",
        num_frames=97,
        fps=fps,
        height=480,
        width=832,
    )


def test_humanoid_fps_and_num_frames_resolution_remain_sampling_param_owned() -> None:
    def get_sp_param(sp: Any, key: str, default: Any = None) -> Any:
        extra = getattr(sp, "extra_args", {})
        if extra.get(key) is not None:
            return extra[key]
        value = getattr(sp, key, None)
        return default if value is None else value

    stub = SimpleNamespace(default_fps=15.0, _get_sp_param=get_sp_param)
    sp = SimpleNamespace(num_frames=97, fps=24.0, extra_args={})

    assert CosmosDreamsPipeline._resolve_request_fps(stub, sp, {}) == 24.0
    assert CosmosDreamsPipeline._resolve_requested_pixel_frames(stub, sp, {"num_frames": 113}, object()) == 97


@pytest.mark.parametrize("num_frames", [93, 98, 100, 101])
def test_transfer_rejects_invalid_frame_counts_without_trimming(num_frames: int) -> None:
    stub = _pipeline_stub()
    with pytest.raises(ARDiffusionRequestRejectedError, match=r"F >= 17.*\(F - 1\) % 16"):
        CosmosDreamsTransferPipeline._validate_conditioning_request(
            stub,
            _sampling_params(num_frames=num_frames),
            None,
        )


def test_transfer_rejects_non_integer_frame_count_without_coercion() -> None:
    stub = _pipeline_stub()
    with pytest.raises(ARDiffusionRequestRejectedError, match="without coercion"):
        CosmosDreamsTransferPipeline._validate_conditioning_request(
            stub,
            _sampling_params(num_frames=97.5),
            None,
        )


@pytest.mark.parametrize("num_frames", [17, 97, 113])
def test_transfer_accepts_canonical_frame_counts(num_frames: int) -> None:
    stub = _pipeline_stub()
    request = CosmosDreamsTransferPipeline._validate_conditioning_request(
        stub,
        _sampling_params(num_frames=num_frames),
        None,
    )
    assert request.num_pixel_frames == num_frames


def test_transfer_uses_one_authoritative_num_frames_value() -> None:
    stub = _pipeline_stub()
    sp = _sampling_params(num_frames=17, extra_args={"num_frames": 113})

    request = CosmosDreamsTransferPipeline._validate_conditioning_request(stub, sp, None)
    resolved = CosmosDreamsTransferPipeline._resolve_requested_pixel_frames(stub, sp, {}, request)

    assert request.num_pixel_frames == 113
    assert resolved == 113


def test_transfer_enforces_hint_exclusivity_and_supported_hints() -> None:
    stub = _pipeline_stub()
    with pytest.raises(ARDiffusionRequestRejectedError, match="exactly one"):
        CosmosDreamsTransferPipeline._validate_conditioning_request(
            stub,
            _sampling_params(extra_args={"blur": True}),
            None,
        )
    with pytest.raises(ARDiffusionRequestRejectedError, match="Unsupported.*wsm"):
        CosmosDreamsTransferPipeline._validate_conditioning_request(
            stub,
            SimpleNamespace(num_frames=97, extra_args={"wsm": True}),
            None,
        )

    generic = CosmosDreamsTransferPipeline._validate_conditioning_request(
        stub,
        SimpleNamespace(
            num_frames=97,
            extra_args={"control_hint": "depth", "control_video": torch.zeros(3, 97, 4, 4)},
        ),
        None,
    )
    assert generic.hint == "depth"


def test_transfer_generic_hint_routes_vision_video_through_transfer_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_extra: list[dict[str, Any]] = []

    def fake_cosmos3_pre_process(
        _od_config: Any,
        *,
        transfer_target_size: tuple[int, int] | None = None,
    ):
        assert transfer_target_size == (480, 832)

        def preprocess(request: Any) -> Any:
            seen_extra.append(dict(request.sampling_params.extra_args))
            assert request.sampling_params.extra_args["edge"] is True
            request.prompt.setdefault("additional_information", {})["preprocessed_transfer_video"] = torch.zeros(
                1, 3, 17, 1, 1
            )
            return request

        return preprocess

    monkeypatch.setattr(
        transfer_pipeline_module,
        "get_cosmos3_pre_process_func",
        fake_cosmos3_pre_process,
    )
    preprocess = transfer_pipeline_module.get_cosmos_dreams_transfer_pre_process_func(_config(_v3_artifact()))
    request = SimpleNamespace(
        prompt={"prompt": "Transfer.", "multi_modal_data": {"video": object()}},
        sampling_params=SimpleNamespace(
            height=None,
            width=None,
            extra_args={"control_hint": "edge", "num_frames": 17},
        ),
    )

    processed = preprocess(request)

    assert seen_extra[0]["edge"] is True
    assert "preprocessed_transfer_video" in processed.prompt["additional_information"]
    assert "edge" not in processed.sampling_params.extra_args


def test_transfer_preprocessor_center_crops_non_16_9_vision_video_to_manifest_geometry() -> None:
    preprocess = transfer_pipeline_module.get_cosmos_dreams_transfer_pre_process_func(_config(_v3_artifact()))
    request = SimpleNamespace(
        prompt={
            "prompt": "Transfer.",
            "multi_modal_data": {
                "video": [Image.new("RGB", (40, 30), "red")],
            },
        },
        sampling_params=SimpleNamespace(
            height=None,
            width=None,
            extra_args={"control_hint": "edge", "num_frames": 17},
        ),
    )

    processed = preprocess(request)

    assert (processed.sampling_params.height, processed.sampling_params.width) == (480, 832)
    transfer_video = processed.prompt["additional_information"]["preprocessed_transfer_video"]
    assert tuple(transfer_video.shape) == (1, 3, 1, 480, 832)
    assert "preprocessed_video" not in processed.prompt["additional_information"]


def test_transfer_generic_hint_accepts_a_path_valued_control_video() -> None:
    stub = _pipeline_stub()
    request = CosmosDreamsTransferPipeline._validate_conditioning_request(
        stub,
        SimpleNamespace(
            num_frames=97,
            extra_args={"control_hint": "seg", "control_video": "/controls/seg.mp4"},
        ),
        None,
    )

    assert request.hint == "seg"
    assert request.control_video == "/controls/seg.mp4"


@pytest.mark.parametrize(
    ("hint", "config", "error"),
    [
        ("depth", {"preset_edge_threshold": "low"}, "Unsupported.*depth.*preset_edge_threshold"),
        ("seg", {"preset_blur_strength": "low"}, "Unsupported.*seg.*preset_blur_strength"),
        ("edge", {"preset_edge_threshold": "invalid"}, "Unsupported Cosmos3 edge preset"),
        ("blur", {"preset_blur_strength": "invalid"}, "Unsupported Cosmos3 blur preset"),
    ],
)
def test_transfer_reuses_per_hint_fields_and_preset_validation(
    hint: str,
    config: dict[str, Any],
    error: str,
) -> None:
    stub = _pipeline_stub()
    with pytest.raises(ARDiffusionRequestRejectedError, match=error):
        CosmosDreamsTransferPipeline._validate_conditioning_request(
            stub,
            SimpleNamespace(num_frames=97, extra_args={hint: config}),
            None,
        )


def test_transfer_rejects_seed_images_and_initial_latents() -> None:
    stub = _pipeline_stub()
    with pytest.raises(ValueError, match="does not accept initial_latent"):
        CosmosDreamsTransferPipeline._initial_condition_latent(
            stub,
            {},
            SimpleNamespace(extra_args={"initial_latent": torch.zeros(1)}),
        )
    with pytest.raises(ValueError, match="does not accept initial_latent"):
        CosmosDreamsTransferPipeline._initial_condition_latent(
            stub,
            {"multi_modal_data": {"image": object()}},
            SimpleNamespace(extra_args={}),
        )
    with pytest.raises(ValueError, match="does not accept initial_latent"):
        CosmosDreamsTransferPipeline._initial_condition_latent(
            stub,
            {"initial_latent": torch.zeros(1)},
            SimpleNamespace(extra_args={}),
        )


def test_transfer_encodes_the_complete_control_clip_once(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _transfer_manifest(height=32, width=32)
    request = _TransferRequestContract("edge", {}, None, 17)
    encoded_inputs: list[torch.Tensor] = []

    monkeypatch.setattr(
        transfer_pipeline_module,
        "load_or_compute_control_frames",
        lambda *args, **kwargs: torch.zeros(3, 17, 32, 32, dtype=torch.uint8),
    )

    def encode(video: torch.Tensor) -> torch.Tensor:
        encoded_inputs.append(video)
        return torch.zeros(1, 48, 5, 2, 2)

    stub = SimpleNamespace(
        manifest=manifest,
        transformer=SimpleNamespace(latent_channel_size=48),
        _encode_video_tensor=encode,
    )
    conditioning = CosmosDreamsTransferPipeline._prepare_conditioning(
        stub,
        SimpleNamespace(),
        typed_inputs=None,
        request=request,
        start_frame=0,
        target_frame=5,
        prompt_data={},
    )

    assert len(encoded_inputs) == 1
    assert tuple(encoded_inputs[0].shape) == (1, 3, 17, 32, 32)
    assert tuple(conditioning.control_latents.shape) == (1, 48, 5, 2, 2)


def test_control_chunk_commits_once_before_rgb_denoise_and_refresh() -> None:
    events: list[Any] = []
    control_latents = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1, 1)
    conditioning = _TransferConditioning(
        request=_TransferRequestContract("edge", {}, None, 17),
        control_latents=control_latents,
    )

    def transformer_forward(_state, hidden_states, _timestep, **kwargs):
        events.append(("control", hidden_states.clone(), kwargs))
        return SimpleNamespace()

    def denoise(_state, **kwargs):
        events.append(("denoise", kwargs))
        return torch.full((1, 1, 4, 1, 1), 7.0)

    def commit(_state, _latent, *, frame_idx, **kwargs):
        events.append(("rgb", frame_idx, kwargs))

    stub = SimpleNamespace(
        dtype=torch.float32,
        device=torch.device("cpu"),
        _timed_tick_stage=lambda *args, **kwargs: nullcontext(),
        _transformer_forward=transformer_forward,
        _denoise_chunk=denoise,
        _commit_clean_frame=commit,
    )
    CosmosDreamsTransferPipeline._run_chunk(
        stub,
        CosmosDreamsSessionState("session"),
        chunk_start=1,
        chunk_end=5,
        target_frame=5,
        terminal_request=False,
        request_start_frame=0,
        seed=42,
        text_kv=[],
        real_text_kv_len=3,
        fps=15.0,
        conditioning=conditioning,
        tick_durations={},
        measure_tick_latency=False,
    )

    assert [event[0] for event in events] == ["control", "denoise", "rgb", "rgb", "rgb", "rgb"]
    assert tuple(events[0][1].shape) == (1, 1, 4, 1, 1)
    assert events[0][2]["condition_vision"] is True
    assert events[0][2]["commit_current"] is True
    assert [event[1] for event in events[2:]] == [1, 2, 3, 4]
