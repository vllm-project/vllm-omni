# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _media(**kwargs):
    return MiniMaxH3EncoderMediaInput(
        task="ref2va", height=768, width=1344, num_frames=124, latent_t=37, audio_t=207, **kwargs
    )


def test_h3_full_pipeline_profiler_includes_local_encoding():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    targets = MiniMaxH3Pipeline._PROFILER_TARGETS
    assert targets == [
        "encode_prompt",
        "_encode_local_media",
        "diffuse",
        "decode",
        "video_vae.decode_latent",
        "audio_vae.decode_latent",
        "prepare_encode",
        "denoise_step",
        "post_decode",
    ]
    assert "_encode_video_conditions" not in targets
    assert "_encode_video_audio_conditions" not in targets
    assert "_encode_audio_conditions" not in targets


def test_decoded_output_is_unchanged_without_active_model_offload():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._model_cpu_offload_modules = []
    output = Mock()

    assert pipeline._offload_model_cpu_stage_output(output) is output
    output.cpu.assert_not_called()


def test_reply_rank_copies_decoded_output_to_host(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._model_cpu_offload_modules = [Mock()]
    monkeypatch.setattr(pipeline, "_is_output_owner_rank", lambda: True)
    output = Mock()
    host_output = Mock()
    output.cpu.return_value = host_output

    assert pipeline._offload_model_cpu_stage_output(output) is host_output
    output.cpu.assert_called_once_with()


def test_non_reply_rank_drops_decoded_output_storage(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._model_cpu_offload_modules = [Mock()]
    monkeypatch.setattr(pipeline, "_is_output_owner_rank", lambda: False)
    output = torch.ones((2, 3), dtype=torch.float16)

    placeholder = pipeline._offload_model_cpu_stage_output(output)

    assert placeholder.device.type == "meta"
    assert placeholder.shape == output.shape
    assert placeholder.dtype == output.dtype


def test_decode_releases_only_audio_before_return(monkeypatch):
    """Video is released by the callers of ``decode`` -- after quantization."""
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    video = torch.ones((1, 1, 4, 4))
    audio = torch.ones((1, 8))
    pipeline = SimpleNamespace(
        device=torch.device("cpu"),
        video_vae=Mock(decode_latent=Mock(return_value=video)),
        audio_vae=Mock(decode_latent=Mock(return_value=audio)),
        _component_on_device=lambda component: nullcontext(),
        _offload_model_cpu_stage_output=Mock(side_effect=lambda value: value),
        _release_stage_cache=Mock(),
    )
    monkeypatch.setattr(
        module.current_omni_platform,
        "create_autocast_context",
        lambda **kwargs: nullcontext(),
    )

    actual_video, actual_audio = MiniMaxH3Pipeline.decode(
        pipeline,
        torch.zeros(1),
        torch.zeros(1),
        height=2,
        width=3,
    )

    assert actual_video.shape[-2:] == (2, 3)
    assert actual_audio is audio
    calls = pipeline._offload_model_cpu_stage_output.call_args_list
    assert len(calls) == 1
    assert calls[0].args[0] is audio


def _recording_release(events, releases):
    """Stand-in for ``_offload_model_cpu_stage_output`` that records what it got."""

    def release(value):
        events.append("release")
        releases.append(value)
        return value

    return release


def test_forward_releases_video_after_quantization_and_before_next_seed(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    # MiniMaxH3VideoVAE.decode_latent ends in frames.float(), so this is what
    # the release hook would see if it ran before the quantizer.
    decoded_dtype = torch.float32
    events: list[str] = []
    releases: list[torch.Tensor] = []

    def diffuse(**kwargs):
        events.append(f"diffuse:{kwargs['seed']}")
        return torch.zeros(1), torch.zeros(1)

    def decode(video_latent, audio_latent, *, height, width):
        events.append("decode")
        return (
            torch.zeros((1, 3, 2, height, width), dtype=decoded_dtype),
            torch.zeros((1, 8)),
        )

    # ``forward`` indexes ``preencode_mp4`` (#7018): this test drives the
    # tensor-decode branch, which is the one the release hook wraps.
    context = {"num_outputs": 2, "seed": 41, "height": 4, "width": 4, "preencode_mp4": False}
    pipeline = SimpleNamespace(
        load_vae_decoder=True,
        od_config=SimpleNamespace(),
        _extract_prompt=lambda raw: ("a prompt", {}),
        _extract_text_conditioning=lambda raw: None,
        _extract_prepared_reference_videos=lambda raw: None,
        _prepare_request_inputs=lambda *args, **kwargs: context,
        _denoise_kwargs=lambda ctx: {},
        diffuse=diffuse,
        decode=decode,
        _offload_model_cpu_stage_output=_recording_release(events, releases),
        _release_stage_cache=lambda: None,
    )
    request = SimpleNamespace(prompts=["a prompt"], sampling_params=SimpleNamespace())

    output = MiniMaxH3Pipeline.forward(pipeline, request)

    # The release must see the quantized frames, not the decoded floats.
    assert len(releases) == 2
    assert [tensor.dtype for tensor in releases] == [torch.uint8, torch.uint8]
    assert decoded_dtype is not torch.uint8
    # ... and it must still happen before the next seed reloads the DiT.
    assert events == ["diffuse:41", "decode", "release", "diffuse:42", "decode", "release"]
    assert output.output[0].dtype is torch.uint8


def test_forward_drops_the_decoded_video_before_the_next_seed_diffuses(monkeypatch):
    """Releasing the copy is not enough if a local still pins the original.

    ``_offload_model_cpu_stage_output`` hands back a *new* tensor, so the
    decoded frames stay alive for as long as anything still references them.
    The whole point of this branch is that they are gone before the next seed
    reloads the DiT, and ``diffuse`` is where that reload happens.
    """
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    decoded_refs: list[weakref.ref] = []
    alive_at_diffuse: list[int] = []

    def diffuse(**kwargs):
        alive_at_diffuse.append(sum(1 for ref in decoded_refs if ref() is not None))
        return torch.zeros(1), torch.zeros(1)

    def decode(video_latent, audio_latent, *, height, width):
        video = torch.zeros((1, 3, 2, height, width), dtype=torch.float32)
        decoded_refs.append(weakref.ref(video))
        return video, torch.zeros((1, 8))

    # ``forward`` indexes ``preencode_mp4`` (#7018): this test drives the
    # tensor-decode branch, which is the one the release hook wraps.
    context = {"num_outputs": 2, "seed": 41, "height": 4, "width": 4, "preencode_mp4": False}
    pipeline = SimpleNamespace(
        load_vae_decoder=True,
        od_config=SimpleNamespace(),
        _extract_prompt=lambda raw: ("a prompt", {}),
        _extract_text_conditioning=lambda raw: None,
        _extract_prepared_reference_videos=lambda raw: None,
        _prepare_request_inputs=lambda *args, **kwargs: context,
        _denoise_kwargs=lambda ctx: {},
        diffuse=diffuse,
        decode=decode,
        _offload_model_cpu_stage_output=lambda value: value.clone(),
        _release_stage_cache=lambda: None,
    )
    request = SimpleNamespace(prompts=["a prompt"], sampling_params=SimpleNamespace())

    MiniMaxH3Pipeline.forward(pipeline, request)

    assert len(decoded_refs) == 2
    # Seed 1's diffuse sees nothing decoded yet; seed 2's must not see seed 1's
    # decoded frames still resident.
    assert alive_at_diffuse == [0, 0]


def test_post_decode_releases_video_after_quantization(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    events: list[str] = []
    releases: list[torch.Tensor] = []

    def decode(video_latent, audio_latent, *, height, width):
        events.append("decode")
        return (
            torch.zeros((1, 3, 2, height, width), dtype=torch.float16),
            torch.zeros((1, 8)),
        )

    pipeline = SimpleNamespace(
        od_config=SimpleNamespace(),
        _unpack_denoised_rows=lambda *args, **kwargs: (torch.zeros(1), torch.zeros(1)),
        load_vae_decoder=True,
        decode=decode,
        _offload_model_cpu_stage_output=_recording_release(events, releases),
        _release_stage_cache=lambda: None,
    )
    state = SimpleNamespace(
        latents=torch.zeros(1),
        extra={
            module._STEP_BRANCH: object(),
            module._STEP_AUDIO_ROWS: torch.zeros(1),
            module._STEP_SHAPE: {
                "latent_t": 1,
                "latent_h": 1,
                "latent_w": 1,
                "audio_t": 1,
                "height": 4,
                "width": 4,
            },
        },
    )

    output = MiniMaxH3Pipeline.post_decode(pipeline, state)

    assert events == ["decode", "release"]
    assert len(releases) == 1
    assert releases[0].dtype is torch.uint8
    assert output.output[0] is releases[0]


@pytest.mark.parametrize(
    ("available", "initialized", "rank", "expected"),
    [(False, False, 7, True), (True, False, 7, True), (True, True, 0, True), (True, True, 1, False)],
)
def test_output_owner_rank_matches_executor_contract(monkeypatch, available, initialized, rank, expected):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    monkeypatch.setattr(module.dist, "is_available", lambda: available)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(module.dist, "get_rank", lambda: rank)

    assert MiniMaxH3Pipeline._is_output_owner_rank() is expected


@pytest.mark.parametrize("load_text_encoder", [True, False])
def test_h3_model_cpu_offload_registers_direct_vae_stages(monkeypatch, load_text_encoder):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = torch.nn.Linear(2, 2)
    pipeline.transformers_ref = torch.nn.Linear(2, 2)
    pipeline.video_vae = torch.nn.Linear(2, 2)
    pipeline.audio_vae = torch.nn.Linear(2, 2)
    pipeline._encoder_modules = ["text_encoder"] if load_text_encoder else []
    if load_text_encoder:
        pipeline.text_encoder = torch.nn.Linear(2, 2)
    apply_offload = Mock()
    remove_offload = Mock()
    monkeypatch.setattr(module, "apply_sequential_offload", apply_offload)
    monkeypatch.setattr(module, "remove_sequential_offload", remove_offload)

    pipeline.enable_omni_model_cpu_offload(
        device=torch.device("cpu"),
        pin_memory=False,
        use_hsdp=False,
    )

    dits = [pipeline.transformer, pipeline.transformers_ref]
    stages = [*([pipeline.text_encoder] if load_text_encoder else []), pipeline.video_vae, pipeline.audio_vae]
    apply_offload.assert_called_once_with(
        dit_modules=dits,
        encoder_modules=stages,
        device=torch.device("cpu"),
        pin_memory=False,
        use_hsdp=False,
        offload_initial_dits=True,
    )

    pipeline.disable_omni_model_cpu_offload()

    remove_offload.assert_called_once_with([*dits, *stages])


def test_h3_model_cpu_offload_keeps_unselected_vaes_resident(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = torch.nn.Linear(2, 2)
    pipeline.transformers_ref = torch.nn.Linear(2, 2)
    pipeline.text_encoder = torch.nn.Linear(2, 2)
    pipeline.video_vae = torch.nn.Linear(2, 2)
    pipeline.audio_vae = torch.nn.Linear(2, 2)
    apply_offload = Mock()
    monkeypatch.setattr(module, "apply_sequential_offload", apply_offload)

    pipeline.enable_omni_model_cpu_offload(
        device=torch.device("cpu"),
        pin_memory=False,
        use_hsdp=False,
        offload_components=frozenset({"dit", "text_encoder"}),
    )

    dits = [pipeline.transformer, pipeline.transformers_ref]
    stages = [pipeline.text_encoder, pipeline.video_vae, pipeline.audio_vae]
    apply_offload.assert_called_once_with(
        dit_modules=dits,
        encoder_modules=stages,
        device=torch.device("cpu"),
        pin_memory=False,
        use_hsdp=False,
        offload_initial_dits=True,
        offload_dit_modules=dits,
        offload_encoder_modules=[pipeline.text_encoder],
    )


def test_h3_model_cpu_offload_rejects_unloaded_selected_encoder(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = torch.nn.Linear(2, 2)
    pipeline.transformers_ref = torch.nn.Linear(2, 2)
    pipeline.text_encoder = None
    pipeline.video_vae = torch.nn.Linear(2, 2)
    pipeline.audio_vae = torch.nn.Linear(2, 2)
    apply_offload = Mock()
    monkeypatch.setattr(module, "apply_sequential_offload", apply_offload)

    with pytest.raises(ValueError, match="no loaded text encoder"):
        pipeline.enable_omni_model_cpu_offload(
            device=torch.device("cpu"),
            pin_memory=False,
            use_hsdp=False,
            offload_components=frozenset({"text_encoder"}),
        )

    apply_offload.assert_not_called()


@pytest.mark.parametrize("decode_fails", [False, True])
def test_h3_model_cpu_offload_scopes_direct_vae_call(monkeypatch, decode_fails):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    component = torch.nn.Linear(2, 2)
    events: list[tuple[Any, ...]] = []

    @contextmanager
    def record_component(value):
        events.append(("activate", value))
        try:
            yield
        finally:
            events.append(("offload", value))

    monkeypatch.setattr(module, "sequential_offload_component", record_component)
    pipeline._model_cpu_offload_modules = [component]

    def decode():
        with pipeline._component_on_device(component):
            events.append(("decode", component))
            if decode_fails:
                raise RuntimeError("decode failed")

    if decode_fails:
        with pytest.raises(RuntimeError, match="decode failed"):
            decode()
    else:
        decode()

    assert events == [
        ("activate", component),
        ("decode", component),
        ("offload", component),
    ]


def _encode_media(pipeline, **kwargs):
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput
    from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media

    media = MiniMaxH3EncoderMediaInput(
        task="ref2va", height=32, width=32, num_frames=124, latent_t=31, audio_t=208, **kwargs
    )
    return encode_media(
        media,
        video_vae=getattr(pipeline, "video_vae", None),
        audio_vae=getattr(pipeline, "audio_vae", None),
        emit_conditioning=True,
        component_scope=pipeline._component_on_device,
    )


@pytest.mark.parametrize("encode_fails", [False, True])
def test_h3_model_cpu_offload_batches_visual_reference_scope(monkeypatch, encode_fails):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.video_vae = Mock()
    pipeline._model_cpu_offload_modules = [pipeline.video_vae]
    events = []

    @contextmanager
    def record_component(value):
        events.append(("activate", value))
        try:
            yield
        finally:
            events.append(("offload", value))

    def encode_image(image):
        events.append(("encode", image.size))
        if encode_fails and image.width == 32:
            raise RuntimeError("encode failed")
        return torch.ones(image.width // 16, 4)

    pipeline.video_vae.encode_image.side_effect = encode_image
    monkeypatch.setattr(module, "sequential_offload_component", record_component)
    images = (torch.zeros(16, 16, 3, dtype=torch.uint8), torch.zeros(16, 32, 3, dtype=torch.uint8))

    if encode_fails:
        with pytest.raises(RuntimeError, match="encode failed"):
            _encode_media(pipeline, images=images)
    else:
        result = _encode_media(pipeline, images=images)
        assert result.visual_condition.shape == (3, 4)
        assert result.visual_condition_shapes == ((1, 1, 1), (1, 1, 2))

    assert events == [
        ("activate", pipeline.video_vae),
        ("encode", (16, 16)),
        ("encode", (32, 16)),
        ("offload", pipeline.video_vae),
    ]


def test_h3_model_cpu_offload_shares_image_and_video_scope(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.video_vae = Mock()
    pipeline._model_cpu_offload_modules = [pipeline.video_vae]
    events = []

    @contextmanager
    def record_component(value):
        events.append(("activate", value))
        try:
            yield
        finally:
            events.append(("offload", value))

    def encode_image(image):
        events.append(("image", image.size))
        return torch.ones(1, 4)

    def encode_video(frames):
        events.append(("video", frames.shape))
        return torch.ones(3, 4), (3, 2, 2)

    pipeline.video_vae.encode_image.side_effect = encode_image
    pipeline.video_vae.encode_video.side_effect = encode_video
    monkeypatch.setattr(module, "sequential_offload_component", record_component)
    result = _encode_media(
        pipeline,
        images=(torch.zeros(16, 16, 3, dtype=torch.uint8), torch.zeros(16, 32, 3, dtype=torch.uint8)),
        videos=(torch.zeros(12, 32, 32, 3, dtype=torch.uint8),),
        video_audios=(None,),
    )

    assert result.visual_condition.shape == (5, 4)
    assert result.visual_condition_shapes == ((1, 1, 1), (1, 1, 2), (3, 2, 2))
    assert events == [
        ("activate", pipeline.video_vae),
        ("image", (16, 16)),
        ("image", (32, 16)),
        ("video", (12, 32, 32, 3)),
        ("offload", pipeline.video_vae),
    ]


@pytest.mark.parametrize("encode_fails", [False, True])
def test_h3_model_cpu_offload_shares_embedded_and_standalone_audio_scope(monkeypatch, encode_fails):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.audio_vae = Mock()
    pipeline._model_cpu_offload_modules = [pipeline.audio_vae]
    events: list[tuple] = []

    @contextmanager
    def record_component(value):
        events.append(("activate", value))
        try:
            yield
        finally:
            events.append(("offload", value))

    def encode_waveform(waveform, sample_rate):
        events.append(("audio", int(waveform[0]), sample_rate))
        if encode_fails:
            raise RuntimeError("audio encode failed")
        return torch.ones(80, 32), 80

    pipeline.audio_vae.encode_waveform.side_effect = encode_waveform
    monkeypatch.setattr(module, "sequential_offload_component", record_component)

    def call():
        return _encode_media(
            pipeline,
            video_audios=(),
            audios=((torch.ones(32_000), 16_000),),
        )

    if encode_fails:
        with pytest.raises(RuntimeError, match="audio encode failed"):
            call()
    else:
        result = call()
        assert result.audio_condition.shape == (80, 32)
        assert result.audio_condition_lengths == (80,)
        assert events[1:-1] == [("audio", 1, 16_000)]

    assert events[0] == ("activate", pipeline.audio_vae)
    assert events[-1] == ("offload", pipeline.audio_vae)
    assert events.count(("activate", pipeline.audio_vae)) == 1
    assert events.count(("offload", pipeline.audio_vae)) == 1


@pytest.mark.parametrize("distributed", [False, True])
def test_manual_component_offload_wins_over_requested_model_offload(distributed):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=True,
        enable_layerwise_offload=not distributed,
        enable_distributed_layerwise_offload=distributed,
    )
    pipeline._model_cpu_offload_modules = []
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected
    component = Mock()

    actual = pipeline._encode_text_hidden(torch.tensor([1, 2]), {})
    with pipeline._component_on_device(component):
        component.load_to_device.assert_called_once_with()
        component.offload_to_cpu.assert_not_called()

    assert actual is expected
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.offload_to_cpu.assert_called_once_with()
    component.offload_to_cpu.assert_called_once_with()


def test_requested_model_offload_without_backend_uses_resident_components():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=True,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
    )
    pipeline._model_cpu_offload_modules = []
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected

    actual = pipeline._encode_text_hidden(torch.tensor([1, 2]), {})

    assert actual is expected
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.encode_ids.assert_called_once()


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("text_resident", [False, True])
@pytest.mark.parametrize("name", ["video_vae", "encoder_component", "decoder_component", "audio_vae", "text_encoder"])
@pytest.mark.parametrize("fails", [False, True])
def test_dlo_component_residency_context(mocker, resident, text_resident, name, fails):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3.vae import _VideoVAEPartProxy

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(
        enable_distributed_layerwise_offload=True,
        dlo_vae_resident=resident,
        dlo_text_encoder_resident=text_resident,
    )
    for module_name in ("video_vae", "audio_vae", "text_encoder"):
        module = torch.nn.Module()
        module.load_to_device = mocker.Mock()
        module.offload_to_cpu = mocker.Mock()
        setattr(pipeline, module_name, module)
    for part in ("encoder", "decoder"):
        proxy = _VideoVAEPartProxy(pipeline.video_vae, part)
        proxy.load_to_device = mocker.Mock()
        proxy.offload_to_cpu = mocker.Mock()
        setattr(pipeline.video_vae, f"{part}_component", proxy)
    component = getattr(pipeline.video_vae, name) if name.endswith("_component") else getattr(pipeline, name)
    should_stage = not text_resident if name == "text_encoder" else not resident

    for _ in range(2):
        with pytest.raises(RuntimeError, match="decode failed") if fails else nullcontext():
            with pipeline._component_on_device(component):
                if fails:
                    raise RuntimeError("decode failed")
    assert component.load_to_device.call_count == 2 * should_stage
    assert component.offload_to_cpu.call_count == 2 * should_stage


@pytest.mark.parametrize("option", ["dlo_vae_resident", "dlo_text_encoder_resident"])
def test_dlo_residency_requires_dlo(option):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    with pytest.raises(ValueError, match="requires distributed layerwise offload"):
        MiniMaxH3Pipeline(od_config=OmniDiffusionConfig(**{option: True}))


def test_dlo_text_residency_rejects_explicit_encoder_offload():
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    config = OmniDiffusionConfig(
        dlo_text_encoder_resident=True,
        diffusion_offload_config={
            "mode": "layer",
            "components": ["dit", "text_encoder"],
            "layer_options": {"dit": {"weight_transfer": "allgather"}},
        },
    )
    with pytest.raises(ValueError, match="conflicts with diffusion_offload_config selecting text_encoder"):
        MiniMaxH3Pipeline(od_config=config)


@pytest.mark.parametrize("text_resident", [False, True])
@pytest.mark.parametrize("vae_resident", [False, True])
@pytest.mark.parametrize("encoder_loaded", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_dlo_text_residency_plan_and_placement(
    monkeypatch, mocker, text_resident, vae_resident, encoder_loaded, compact
):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module
    from vllm_omni.diffusion.offloader.base import OffloadConfig
    from vllm_omni.diffusion.offloader.component_utils import prepare_pipeline_components
    from vllm_omni.diffusion.offloader.plan_resolver import resolve_offload_plan

    # Stop after the real constructor builds its per-instance component plan,
    # before it resolves or loads checkpoints.
    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        pipeline_module, "is_minimax_h3_modular", mocker.Mock(side_effect=RuntimeError("stop after component plan"))
    )
    config = OmniDiffusionConfig(
        model="unused",
        enable_distributed_layerwise_offload=not compact,
        diffusion_offload_config=(
            {"mode": "layer", "components": ["dit"], "layer_options": {"dit": {"weight_transfer": "allgather"}}}
            if compact
            else None
        ),
        dlo_text_encoder_resident=text_resident,
        dlo_vae_resident=vae_resident,
        model_loaded={"text_encoder": encoder_loaded},
    )
    pipeline = object.__new__(pipeline_module.MiniMaxH3Pipeline)
    original_plan = pipeline._offload_plan
    with pytest.raises(RuntimeError, match="stop after component plan"):
        pipeline.__init__(od_config=config)
    assert pipeline._encoder_modules == (["text_encoder"] if encoder_loaded else [])
    assert ("text_encoder" in pipeline._offload_plan.encoder_block_attrs) is (encoder_loaded and not text_resident)
    assert ("text_encoder" in pipeline._offload_plan.on_demand_component_paths) is (
        encoder_loaded and not text_resident
    )
    assert ("video_vae" in pipeline._offload_plan.on_demand_component_paths) is (not vae_resident)
    assert "text_encoder" in original_plan.encoder_block_attrs
    assert original_plan.on_demand_component_paths == {"text_encoder", "video_vae", "audio_vae"}

    class _DiT(torch.nn.Module):
        _layerwise_offload_blocks_attrs = ["blocks"]

    pipeline.transformer = _DiT()
    pipeline.transformer.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(2)])
    pipeline.transformers_ref = None
    for name in ("text_encoder", "video_vae", "audio_vae"):
        module = torch.nn.Module()
        module.load_to_device = mocker.Mock()
        module.offload_to_cpu = mocker.Mock()
        module.to = mocker.Mock(wraps=module.to)
        setattr(pipeline, name, module)
    pipeline.text_encoder.vision = torch.nn.Module()
    pipeline.text_encoder.vision.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(2)])
    pipeline.text_encoder.text_model = torch.nn.Module()
    pipeline.text_encoder.text_model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(2)])
    resolved = resolve_offload_plan(pipeline, OffloadConfig.from_od_config(config))
    encoder_streamed = encoder_loaded and not text_resident and not compact
    assert bool(resolved.encoders) is encoder_loaded
    if encoder_loaded:
        assert bool(resolved.encoders[0].stacks) is encoder_streamed
        assert resolved.encoders[0].on_demand is encoder_streamed
    assert bool(resolved.dits[0].stacks)
    assert resolved.dits[0].selected
    assert all(component.on_demand is (not vae_resident and not compact) for component in resolved.vaes)

    install_encoder_hooks = mocker.Mock(return_value=True)
    prepare_pipeline_components(
        resolved, device=torch.device("cpu"), staged_components=[], enable_encoder_blocks=install_encoder_hooks
    )
    assert install_encoder_hooks.call_count == int(encoder_streamed)
    assert pipeline.text_encoder.offload_to_cpu.call_count == int(encoder_streamed)
    assert pipeline.text_encoder.to.call_count == int(encoder_loaded and not encoder_streamed)
