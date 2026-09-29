# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import subprocess
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.data_entry_keys import flatten_payload
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MINIMAX_H3_ENCODER_LAYOUT_KEY,
    STAGE_SCHEMA_VERSION,
    MiniMaxH3EncoderConditioning,
    MiniMaxH3EncoderMediaConditioning,
    MiniMaxH3EncoderMediaInput,
)
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import (
    _canonical_audio_edit_mask,
    _canonical_video_edit_mask,
)
from vllm_omni.model_executor.stage_input_processors.minimax_h3 import encoder2diffusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_edit_mask_boundary_canonicalizes_supported_request_shapes() -> None:
    video_token = torch.arange(168, dtype=torch.float32).reshape(7, 4, 6) / 168
    video_full = video_token.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    video_kwargs = {"latent_t": 7, "latent_h": 8, "latent_w": 12}
    for value in (video_token.flatten()[None, None], video_token[None], video_full[None]):
        torch.testing.assert_close(_canonical_video_edit_mask(value, **video_kwargs), video_full)
    torch.testing.assert_close(
        _canonical_video_edit_mask(torch.tensor([[[0.25]]]), **video_kwargs),
        torch.full((7, 8, 12), 0.25),
    )

    audio_temporal = torch.arange(37, dtype=torch.float32) / 37
    audio_full = audio_temporal.repeat(2, 1)
    for value in (audio_temporal[None, None], audio_full.flatten()[None], audio_full[None]):
        torch.testing.assert_close(_canonical_audio_edit_mask(value, audio_t=37), audio_full)
    torch.testing.assert_close(
        _canonical_audio_edit_mask(torch.tensor([[[0.25]]]), audio_t=37),
        torch.full((2, 37), 0.25),
    )


@pytest.mark.parametrize(
    ("canonicalize", "value", "kwargs", "message"),
    [
        (_canonical_video_edit_mask, [True], {"latent_t": 7, "latent_h": 8, "latent_w": 12}, "booleans"),
        (_canonical_audio_edit_mask, float("nan"), {"audio_t": 37}, "finite"),
        (_canonical_audio_edit_mask, 10**400, {"audio_t": 37}, "finite"),
        (_canonical_audio_edit_mask, 1.01, {"audio_t": 37}, r"\[0, 1\]"),
        (_canonical_video_edit_mask, [0.0, 0.5], {"latent_t": 7, "latent_h": 8, "latent_w": 12}, "shape"),
    ],
)
def test_edit_mask_boundary_rejects_invalid_values(canonicalize, value, kwargs, message) -> None:
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError, match=message):
        canonicalize(value, **kwargs)


def test_encoder_elides_all_generate_masks_but_requires_sources(monkeypatch) -> None:
    from vllm_omni.errors import OmniClientError
    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing

    monkeypatch.setattr(processing, "resolve_minimax_h3_shape", lambda *_args: (64, 96, 22, 7, 37))
    sampling = SimpleNamespace(extra_args={"task": "t2va"})
    prepared = processing.prepare_encoder_inputs(
        {
            "prompt": "generate",
            "multi_modal_data": {"video_noise_mask": 1.0, "audio_noise_mask": 1.0},
        },
        sampling,
    )
    assert prepared.media.video_edit is None
    assert prepared.media.audio_edit is None

    with pytest.raises(OmniClientError, match="video_noise_mask requires source_video"):
        processing.prepare_encoder_inputs(
            {"prompt": "edit", "multi_modal_data": {"video_noise_mask": 0.5}},
            sampling,
        )


def test_pipeline_owns_three_encoders_then_dit_and_decoders() -> None:
    from vllm_omni.model_executor.models.minimax_h3.pipeline import MINIMAX_H3_PIPELINE

    encoder, diffusion = MINIMAX_H3_PIPELINE.stages
    assert encoder.model_arch == "MiniMaxH3Encoder"
    assert encoder.custom_process_next_stage_input_func == (
        "vllm_omni.model_executor.stage_input_processors.minimax_h3.encoder2diffusion_full_payload"
    )
    assert diffusion.model_arch == "MiniMaxH3Pipeline"
    assert diffusion.stage_input_payload_keys == ("encoder_output",)
    assert diffusion.requires_multimodal_data is False
    assert diffusion.requires_full_payload_input is False


def _patch_encoder_constructors(monkeypatch, *, rank: int):
    from vllm_omni.model_executor.models.minimax_h3 import encoder as encoder_module

    group = SimpleNamespace(rank_in_group=rank, world_size=2, device_group=object())

    class FakeVideoVAE(torch.nn.Module):
        def __init__(self, path, *, device, encode_only, trust_remote_code):
            super().__init__()
            self.path = path
            self.parallel_args = None
            self.trust_remote_code = trust_remote_code

        def set_parallel_size(self, size, *, process_group):
            self.parallel_args = (size, process_group)

    class FakeAudioVAE(torch.nn.Module):
        def __init__(self, path, *, device, encode_only, trust_remote_code):
            super().__init__()
            self.path = path
            self.trust_remote_code = trust_remote_code

    def init_backbone(self, *, vllm_config, prefix):
        torch.nn.Module.__init__(self)

    monkeypatch.setattr(encoder_module.MiniMaxH3TextEncoderBackbone, "__init__", init_backbone)
    monkeypatch.setattr(encoder_module, "get_tp_group", lambda: group)
    monkeypatch.setattr(encoder_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(encoder_module, "MiniMaxH3VideoVAE", FakeVideoVAE)
    monkeypatch.setattr(encoder_module, "MiniMaxH3AudioVAE", FakeAudioVAE)
    return encoder_module, group


def _component_root(tmp_path):
    root = tmp_path / "FL2VA"
    for name in ("text_encoder", "video_vae", "audio_vae"):
        (root / name).mkdir(parents=True)
    return root


def _encoder_config(root, *, video_mode="patch", roles=None, trust_remote_code=True):
    components = (
        {
            "text_encoder": {"parallel_mode": "tp"},
            "video_vae": {"parallel_mode": video_mode},
            "audio_vae": {"parallel_mode": "leader"},
        }
        if roles is None
        else roles
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=str(root / "text_encoder"),
            hf_config=SimpleNamespace(minimax_h3_encoder_components=components),
            trust_remote_code=trust_remote_code,
        )
    )


def test_encoder_requires_exactly_three_components(monkeypatch, tmp_path) -> None:
    encoder_module, _ = _patch_encoder_constructors(monkeypatch, rank=0)
    root = tmp_path / "FL2VA"
    (root / "text_encoder").mkdir(parents=True)
    (root / "video_vae").mkdir()
    config = _encoder_config(root)

    with pytest.raises(RuntimeError, match=r"requires all three encoders.*audio_vae"):
        encoder_module.MiniMaxH3Encoder(vllm_config=config)


@pytest.mark.parametrize(("rank", "has_audio_vae"), [(0, True), (1, False)])
@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_encoder_uses_tp_patch_and_leader_on_the_same_rank_set(
    monkeypatch,
    tmp_path,
    rank,
    has_audio_vae,
    trust_remote_code,
) -> None:
    encoder_module, group = _patch_encoder_constructors(monkeypatch, rank=rank)
    root = _component_root(tmp_path)
    config = _encoder_config(root, trust_remote_code=trust_remote_code)

    model = encoder_module.MiniMaxH3Encoder(vllm_config=config)

    assert model.component_config.text_parallel_mode == "tp"
    assert model.component_config.video_parallel_mode == "patch"
    assert model.component_config.audio_parallel_mode == "leader"
    assert model.video_vae.parallel_args == (2, group.device_group)
    assert model.video_vae.trust_remote_code is trust_remote_code
    assert (model.audio_vae is not None) is has_audio_vae
    if has_audio_vae:
        assert model.audio_vae.trust_remote_code is trust_remote_code


@pytest.mark.parametrize("component", ["video", "audio"])
@pytest.mark.parametrize("trusted", [False, True])
def test_encoder_only_vae_requires_remote_code_trust(monkeypatch, component, trusted) -> None:
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    config = {"auto_map": {"AutoModel": "modeling.RemoteVAE"}, "sample_rate": 32000}
    monkeypatch.setattr(vae_module, "_load_component_config", lambda _path: config)
    remote = torch.nn.Module()
    remote.model = torch.nn.Module()
    calls = []

    def load_encoder(path, config_dict):
        calls.append((path, config_dict))
        return remote

    monkeypatch.setattr(vae_module, f"_load_{component}_vae_encoder", load_encoder)
    vae_cls = vae_module.MiniMaxH3VideoVAE if component == "video" else vae_module.MiniMaxH3AudioVAE
    if trusted:
        model = vae_cls("unused", device=torch.device("cpu"), encode_only=True, trust_remote_code=True)
        assert model.remote is remote
        assert calls == [("unused", config)]
    else:
        with pytest.raises(ValueError, match="trust-remote-code"):
            vae_cls("unused", device=torch.device("cpu"), encode_only=True)
        assert calls == []


def test_encoder_requires_all_role_policies(monkeypatch, tmp_path) -> None:
    encoder_module, _ = _patch_encoder_constructors(monkeypatch, rank=0)
    root = _component_root(tmp_path)
    config = _encoder_config(
        root,
        roles={
            "text_encoder": {"parallel_mode": "tp"},
            "video_vae": {"parallel_mode": "patch"},
        },
    )

    with pytest.raises(ValueError, match=r"requires exactly.*audio_vae"):
        encoder_module.MiniMaxH3Encoder(vllm_config=config)


def test_encoder_does_not_infer_role_policies(monkeypatch, tmp_path) -> None:
    encoder_module, _ = _patch_encoder_constructors(monkeypatch, rank=0)
    root = _component_root(tmp_path)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=str(root / "text_encoder"),
            hf_config=SimpleNamespace(),
        )
    )

    with pytest.raises(ValueError, match="requires explicit minimax_h3_encoder_components"):
        encoder_module.MiniMaxH3Encoder(vllm_config=config)


def test_video_leader_policy_skips_nonleader(monkeypatch, tmp_path) -> None:
    encoder_module, _ = _patch_encoder_constructors(monkeypatch, rank=1)
    root = _component_root(tmp_path)
    model = encoder_module.MiniMaxH3Encoder(vllm_config=_encoder_config(root, video_mode="leader"))

    assert model.video_vae is None
    assert model.audio_vae is None
    assert (
        model._encode_media(
            MiniMaxH3EncoderMediaInput(
                task="ref2va",
                height=32,
                width=32,
                num_frames=48,
                latent_t=1,
                audio_t=80,
                images=(torch.zeros(32, 32, 3, dtype=torch.uint8),),
            )
        )
        is None
    )


def test_encoder_runs_video_and_audio_components_on_ar_model() -> None:
    from vllm_omni.model_executor.models.minimax_h3.encoder import MiniMaxH3Encoder

    class VideoVAE:
        def encode_image(self, _image):
            return torch.ones(1, 96)

        def encode_video(self, _frames):
            return torch.full((1, 96), 2.0), (1, 2, 2)

    class AudioVAE:
        def encode_waveform(self, waveform, _sample_rate):
            return torch.ones(160, 32), waveform.shape[-1] // 800

    model = MiniMaxH3Encoder.__new__(MiniMaxH3Encoder)
    torch.nn.Module.__init__(model)
    model._component_leader = True
    model.video_vae = VideoVAE()
    model.audio_vae = AudioVAE()
    media = MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=32,
        width=32,
        num_frames=48,
        latent_t=1,
        audio_t=80,
        images=(torch.zeros(32, 32, 3, dtype=torch.uint8),),
        videos=(torch.zeros(1, 32, 32, 3, dtype=torch.uint8),),
        video_audios=((torch.zeros(64_000), 32_000),),
    )

    conditioning = model._encode_media(media)

    assert conditioning is not None
    assert conditioning.visual_condition_shapes == ((1, 2, 2), (1, 2, 2))
    assert conditioning.audio_condition_lengths == (80,)
    assert conditioning.ref_blocks[1]["kind"] == "video_audio"


def test_prepare_encoder_inputs_keeps_reference_audio_budgets_separate(monkeypatch) -> None:
    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing

    frames = torch.zeros(1, 32, 32, 3, dtype=torch.uint8).numpy()
    waveform = torch.zeros(320_000)
    monkeypatch.setattr(processing, "resolve_minimax_h3_shape", lambda *_args: (32, 32, 240, 60, 400))
    monkeypatch.setattr(
        processing,
        "prepare_reference_videos",
        lambda *_args, **_kwargs: [
            {"prepared_path": "prepared.mp4", "original_path": "reference.mp4", "input_has_audio": True}
        ],
    )
    monkeypatch.setattr(processing, "load_video_frames", lambda _path: frames)
    monkeypatch.setattr(
        processing,
        "sample_reference_video_frames",
        lambda *_args, **_kwargs: {"frames": [frames[0]], "block_timestamps": [[0.0]]},
    )
    monkeypatch.setattr(processing, "load_video_audio", lambda *_args, **_kwargs: (waveform, 32_000))

    prepared = processing.prepare_encoder_inputs(
        {
            "prompt": "reference",
            "multi_modal_data": {"video": "reference.mp4", "audio": (waveform, 32_000)},
        },
        SimpleNamespace(extra_args={"task": "ref2va"}),
    )

    assert prepared.media.video_audios[0][0].shape[-1] == 320_000
    assert prepared.media.audios[0][0].shape[-1] == 320_000
    assert prepared.condition_labels == [("audio", 1), ("video", 1), ("audio", 2)]


@pytest.mark.parametrize(
    ("source_key", "mask_key", "mask_shape", "decoder"),
    [
        ("source_video", "video_noise_mask", (7, 4, 6), "prepare_edit_video"),
        ("source_audio", "audio_noise_mask", (37,), "load_audio_file"),
        ("source_video", "audio_noise_mask", (37,), "load_video_audio"),
    ],
)
def test_prepare_encoder_inputs_maps_corrupt_edit_media_to_client_error(
    monkeypatch,
    source_key,
    mask_key,
    mask_shape,
    decoder,
) -> None:
    from vllm_omni.errors import OmniClientError
    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing

    def fail_decode(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["ffmpeg"])

    monkeypatch.setattr(processing, "resolve_minimax_h3_shape", lambda *_args: (64, 96, 22, 7, 37))
    monkeypatch.setattr(processing, decoder, fail_decode)

    with pytest.raises(OmniClientError, match=f"could not decode {source_key}"):
        processing.prepare_encoder_inputs(
            {
                "prompt": "edit",
                "multi_modal_data": {
                    source_key: "corrupt.media",
                    mask_key: torch.zeros(mask_shape),
                },
            },
            SimpleNamespace(extra_args={"task": "t2va"}),
        )


@pytest.mark.parametrize(
    ("embedded_lengths", "standalone_lengths", "valid"),
    [((400,), (400,), True), ((320, 320), (400,), False), ((400,), (320, 320), False)],
)
def test_encode_media_keeps_audio_budgets_and_component_residency_separate(
    embedded_lengths, standalone_lengths, valid
) -> None:
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media

    active = None
    scopes = []

    class VideoVAE:
        def encode_image(self, _image):
            assert active is self
            return torch.ones(1, 96)

        def encode_video(self, _frames):
            assert active is self
            return torch.ones(1, 96), (1, 2, 2)

    class AudioVAE:
        def encode_waveform(self, waveform, _sample_rate):
            assert active is self
            length = waveform.shape[-1] // 800
            return torch.ones(2 * length, 32), length

    @contextmanager
    def component_scope(component):
        nonlocal active
        assert active is None
        active = component
        scopes.append(component)
        try:
            yield
        finally:
            active = None

    video_vae = VideoVAE()
    audio_vae = AudioVAE()
    media = MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=32,
        width=32,
        num_frames=360,
        latent_t=90,
        audio_t=600,
        images=(torch.zeros(32, 32, 3, dtype=torch.uint8),),
        videos=tuple(torch.zeros(1, 32, 32, 3, dtype=torch.uint8) for _ in embedded_lengths),
        video_audios=tuple((torch.zeros(length * 800), 32_000) for length in embedded_lengths),
        audios=tuple((torch.zeros(length * 800), 32_000) for length in standalone_lengths),
    )
    kwargs = dict(video_vae=video_vae, audio_vae=audio_vae, emit_conditioning=True, component_scope=component_scope)
    if valid:
        conditioning = encode_media(media, **kwargs)
        assert conditioning.audio_condition_lengths == embedded_lengths + standalone_lengths
        assert [block["kind"] for block in conditioning.ref_blocks] == ["image", "video_audio", "audio"]
        conditioning.to_omni_components()
    else:
        with pytest.raises(ValueError, match="at most 15 seconds"):
            encode_media(media, **kwargs)
    assert scopes == ([video_vae, audio_vae] if valid else [])


def test_encoder_output_reuses_encoder_handoff_and_round_trips(monkeypatch) -> None:
    expected = MiniMaxH3EncoderConditioning(
        hidden_states=torch.randn(3, 5120, dtype=torch.bfloat16),
        token_tags=torch.tensor([1, 0, 1], dtype=torch.int64),
        task="t2va",
        height=256,
        width=448,
        num_frames=17,
        latent_t=5,
        audio_t=10,
        video_edit_clean_rows=torch.zeros(5 * 8 * 14, 96),
        video_edit_mask=torch.zeros(5, 16, 28),
    )

    def fail_value_scan(*_args, **_kwargs):
        raise AssertionError("wire must trust encoder-boundary value validation")

    monkeypatch.setattr(torch, "isfinite", fail_value_scan)

    source = SimpleNamespace(
        finished=True,
        request_id="request-1",
        outputs=[SimpleNamespace(multimodal_output=flatten_payload(expected.to_omni_payload()))],
    )
    prompt = {
        "prompt": "hello",
        "additional_information": {"global_request_id": ["request-1"]},
    }

    result = encoder2diffusion([source], prompt)

    assert result is not None
    payload = result["additional_information"]["encoder_output"]
    actual = MiniMaxH3EncoderConditioning.from_omni_payload(payload)
    monkeypatch.undo()
    torch.testing.assert_close(actual.hidden_states, expected.hidden_states)
    torch.testing.assert_close(actual.token_tags, expected.token_tags)
    torch.testing.assert_close(actual.video_edit_mask, expected.video_edit_mask)

    layout = payload["kv_metadata"][MINIMAX_H3_ENCODER_LAYOUT_KEY]
    assert int(layout[1]) == STAGE_SCHEMA_VERSION
    legacy_layout = layout.clone()
    legacy_layout[1] = STAGE_SCHEMA_VERSION - 1
    payload["kv_metadata"][MINIMAX_H3_ENCODER_LAYOUT_KEY] = legacy_layout

    with pytest.raises(ValueError, match=rf"unsupported MiniMax H3 encoder wire schema 1:{STAGE_SCHEMA_VERSION - 1}"):
        MiniMaxH3EncoderConditioning.from_omni_payload(payload)


def test_encoder_releases_workspace_after_cpu_payload(monkeypatch) -> None:
    from vllm_omni.model_executor.models.minimax_h3 import encoder as encoder_module

    model = encoder_module.MiniMaxH3Encoder.__new__(encoder_module.MiniMaxH3Encoder)
    torch.nn.Module.__init__(model)
    model._component_leader = True
    model._token_tags = torch.tensor([1])
    info = {
        "meta": {"minimax_h3_encoder_request": {}},
        "_minimax_h3_encoder_media_conditioning": MiniMaxH3EncoderMediaConditioning(
            task="t2va",
            height=256,
            width=448,
            num_frames=17,
            latent_t=5,
            audio_t=10,
        ),
    }
    empty_cache_calls = 0

    def empty_cache() -> None:
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    monkeypatch.setattr(torch.accelerator, "empty_cache", empty_cache)
    output = model.make_omni_output(
        torch.zeros(1, 5120),
        model_intermediate_buffer=[info],
        request_sample_eligible=[True],
    )

    assert empty_cache_calls == 1
    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs["embed"]["embedding"][0].device.type == "cpu"


def test_driving_audio_is_not_a_short_reference_and_survives_stage_transport():
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing

    sampling = OmniDiffusionSamplingParams(
        width=64,
        height=64,
        fps=24,
        extra_args={
            "aspect_ratio": "1:1",
            "duration": 20,
            "long_video": True,
            "audio_mode": "lock_source",
        },
    )
    prepared = processing.prepare_encoder_inputs(
        {"prompt": "A singer", "multi_modal_data": {"audio": (torch.zeros(75 * 800), 800)}}, sampling
    )
    media = MiniMaxH3EncoderMediaInput.from_mm_tensors(prepared.media.to_mm_tensors(), prepared.media.to_metadata())
    assert media.task == "t2va"
    assert media.audio_mode == "lock_source"
    assert prepared.condition_labels == []

    class FakeAudioVAE:
        def encode_waveform(self, waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
            return torch.ones(6000, 32), 3000

    audio_vae = FakeAudioVAE()
    conditioning = processing.encode_media(media, video_vae=None, audio_vae=audio_vae, emit_conditioning=True)
    assert conditioning.audio_condition_lengths == (3000,)
    assert conditioning.ref_blocks == ()


def test_driving_audio_does_not_override_fl2va_task_inference():
    from PIL import Image

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as processing

    sampling = OmniDiffusionSamplingParams(
        width=64,
        height=64,
        fps=24,
        num_frames=124,
        extra_args={"audio_mode": "lock_source"},
    )
    prepared = processing.prepare_encoder_inputs(
        {
            "prompt": "A singer",
            "multi_modal_data": {
                "image": Image.new("RGB", (256, 256)),
                "audio": (torch.zeros(5 * 800), 800),
            },
        },
        sampling,
    )

    assert prepared.media.task == "fl2va"
    assert prepared.media.audio_mode == "lock_source"
    assert prepared.condition_labels == [("image", 1)]
