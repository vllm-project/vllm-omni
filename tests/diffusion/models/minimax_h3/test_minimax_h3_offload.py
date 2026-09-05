# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_h3_profiler_targets_cover_aggregate_reference_encoders():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    targets = MiniMaxH3Pipeline._PROFILER_TARGETS
    assert "_encode_visual_conditions" in targets
    assert "_encode_reference_audio_conditions" in targets
    assert "_encode_video_conditions" not in targets
    assert "_encode_video_audio_conditions" not in targets
    assert "_encode_audio_conditions" not in targets


@pytest.mark.parametrize("distributed", [False, True])
def test_manual_component_offload_wins_over_requested_model_offload(distributed):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(
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
    with pipeline._component_on_device(component, family="vae"):
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
        _component_on_device=lambda component, *, family: nullcontext(),
        _offload_model_cpu_stage_output=Mock(side_effect=lambda value: value),
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

    context = {"num_outputs": 2, "seed": 41, "height": 4, "width": 4}
    pipeline = SimpleNamespace(
        od_config=SimpleNamespace(),
        _extract_prompt=lambda raw: ("a prompt", {}),
        _extract_text_conditioning=lambda raw: None,
        _extract_prepared_reference_videos=lambda raw: None,
        _prepare_request_inputs=lambda **kwargs: context,
        _denoise_kwargs=lambda ctx: {},
        diffuse=diffuse,
        decode=decode,
        _offload_model_cpu_stage_output=_recording_release(events, releases),
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

    context = {"num_outputs": 2, "seed": 41, "height": 4, "width": 4}
    pipeline = SimpleNamespace(
        od_config=SimpleNamespace(),
        _extract_prompt=lambda raw: ("a prompt", {}),
        _extract_text_conditioning=lambda raw: None,
        _extract_prepared_reference_videos=lambda raw: None,
        _prepare_request_inputs=lambda **kwargs: context,
        _denoise_kwargs=lambda ctx: {},
        diffuse=diffuse,
        decode=decode,
        _offload_model_cpu_stage_output=lambda value: value.clone(),
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
        decode=decode,
        _offload_model_cpu_stage_output=_recording_release(events, releases),
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


def test_h3_model_cpu_offload_registers_direct_vae_stages(monkeypatch):
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
    remove_offload = Mock()
    monkeypatch.setattr(module, "apply_sequential_offload", apply_offload)
    monkeypatch.setattr(module, "remove_sequential_offload", remove_offload)

    pipeline.enable_omni_model_cpu_offload(
        device=torch.device("cpu"),
        pin_memory=False,
        use_hsdp=False,
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
    )

    pipeline.disable_omni_model_cpu_offload()

    remove_offload.assert_called_once_with([*dits, *stages])


@pytest.mark.parametrize("decode_fails", [False, True])
def test_h3_model_cpu_offload_scopes_direct_vae_call(monkeypatch, decode_fails):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    component = torch.nn.Linear(2, 2)
    events = []

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
        with pipeline._component_on_device(component, family="vae"):
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


@pytest.mark.parametrize("encode_fails", [False, True])
def test_h3_model_cpu_offload_batches_visual_reference_scope(monkeypatch, encode_fails):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
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
    monkeypatch.setattr(module, "_dit_rank_world", lambda: (None, 0, 1))
    monkeypatch.setattr(module, "_broadcast_tensor", lambda value, **kwargs: value)
    images = [Image.new("RGB", (16, 16)), Image.new("RGB", (32, 16))]

    if encode_fails:
        with pytest.raises(RuntimeError, match="encode failed"):
            pipeline._encode_visual_conditions(images, None, video_count=0)
    else:
        rows, shapes = pipeline._encode_visual_conditions(images, None, video_count=0)
        assert rows.shape == (3, 4)
        assert shapes == [(1, 1, 1), (1, 1, 2)]

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
    pipeline.device = torch.device("cpu")
    pipeline.video_vae = Mock()
    pipeline.video_vae.is_distributed_enabled.return_value = False
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
        events.append(("video", frames))
        return torch.ones(3, 4), (3, 2, 2)

    pipeline.video_vae.encode_image.side_effect = encode_image
    pipeline.video_vae.encode_video.side_effect = encode_video
    monkeypatch.setattr(module, "sequential_offload_component", record_component)
    monkeypatch.setattr(module, "_dit_rank_world", lambda: (None, 0, 1))
    monkeypatch.setattr(module, "_broadcast_tensor", lambda value, **kwargs: value)
    monkeypatch.setattr(module, "load_video_frames", lambda path: f"frames:{path}")
    images = [Image.new("RGB", (16, 16)), Image.new("RGB", (32, 16))]
    prepared_videos = [{"prepared_path": "reference.mp4"}]

    rows, shapes = pipeline._encode_visual_conditions(
        images,
        prepared_videos,
        video_count=1,
    )

    assert rows.shape == (5, 4)
    assert shapes == [(1, 1, 1), (1, 1, 2), (3, 2, 2)]
    assert events == [
        ("activate", pipeline.video_vae),
        ("image", (16, 16)),
        ("image", (32, 16)),
        ("video", "frames:reference.mp4"),
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
    events = []

    @contextmanager
    def record_component(value):
        events.append(("activate", value))
        try:
            yield
        finally:
            events.append(("offload", value))

    def encode_embedded(*args, **kwargs):
        events.append(("embedded",))
        if encode_fails:
            raise RuntimeError("audio encode failed")
        return torch.ones(1, 2), [1]

    def encode_standalone(*args, **kwargs):
        events.append(("standalone",))
        return torch.ones(1, 2), [1]

    pipeline._encode_video_audio_conditions_resident = encode_embedded
    pipeline._encode_audio_conditions_resident = encode_standalone
    monkeypatch.setattr(module, "sequential_offload_component", record_component)

    def call():
        return pipeline._encode_reference_audio_conditions(
            [{"input_has_audio": True}],
            has_audio=[True],
            standalone_audios=[(torch.ones(1), 16_000)],
            max_duration_seconds=1.0,
        )

    if encode_fails:
        with pytest.raises(RuntimeError, match="audio encode failed"):
            call()
    else:
        result = call()
        assert result[0].shape == (1, 2)
        assert result[2].shape == (1, 2)

    assert events[0] == ("activate", pipeline.audio_vae)
    assert events[-1] == ("offload", pipeline.audio_vae)
    assert events.count(("activate", pipeline.audio_vae)) == 1
    assert events.count(("offload", pipeline.audio_vae)) == 1


class _RecordingVAE(torch.nn.Module):
    """Stand-in VAE that records every device relocation requested on it."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.to_calls: list[tuple] = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def test_layerwise_backend_does_not_touch_pipeline_staged_vaes(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.offloader import layerwise_backend as layerwise_backend_module
    from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend
    from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

    class _Stream:
        def wait_stream(self, _stream) -> None:
            return None

        def wait_event(self, _event) -> None:
            return None

    monkeypatch.setattr(layerwise_backend_module.current_omni_platform, "Stream", lambda: _Stream())

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = torch.nn.Linear(2, 2)
    pipeline.text_encoder = None
    pipeline.video_vae = _RecordingVAE()
    pipeline.audio_vae = _RecordingVAE()
    # Registry-side VAE patch parallelism aliases the video VAE; it must not
    # become a discovery attribute either.
    pipeline.vae = pipeline.video_vae
    pipeline._dit_modules = ["transformer"]
    pipeline._encoder_modules = []

    backend = LayerWiseOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.LAYER_WISE, pin_cpu_memory=False),
        device=torch.device("cpu"),
    )
    backend.enable(pipeline)

    assert pipeline.video_vae.to_calls == []
    assert pipeline.audio_vae.to_calls == []

    discovered = ModuleDiscovery.discover(pipeline)
    assert discovered.vaes == []
    assert discovered.vae_names == []


def test_h3_offload_plan_does_not_stage_vaes_on_demand():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    plan = MiniMaxH3Pipeline._offload_plan
    assert plan.on_demand_component_paths == frozenset({"text_encoder"})


# ---------------------------------------------------------------------------
# Layer-wise component selection: the pipeline side of the contract.
#
# ``--layerwise-offload-components`` names the families the layer-wise backend
# may manage; a family left out stays fully device-resident. This pipeline
# stages the encoders and VAEs itself, so it has to read the same selection --
# otherwise it drags an unselected family back to the host after every use and
# the placement the backend published never holds.
# ---------------------------------------------------------------------------


def _residency_pipeline(**od_kwargs):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(**od_kwargs)
    pipeline._model_cpu_offload_modules = []
    pipeline.device = torch.device("cpu")
    return pipeline


def test_unselected_vae_family_is_still_staged_by_the_pipeline():
    pipeline = _residency_pipeline(
        enable_layerwise_offload=True,
        layerwise_offload_components="dit,text_encoder",
    )
    component = Mock()

    with pipeline._component_on_device(component, family="vae"):
        component.load_to_device.assert_called_once_with()
        component.offload_to_cpu.assert_not_called()

    # The VAEs are not reported by component discovery, so no backend places
    # them: dropping "vae" from the selection would leave nobody owning their
    # residency rather than making them resident.
    assert pipeline._stages_component_family("vae") is True
    component.offload_to_cpu.assert_called_once_with()


def test_unselected_vae_family_is_still_built_on_the_host():
    pipeline = _residency_pipeline(
        enable_layerwise_offload=True,
        layerwise_offload_components="dit,text_encoder",
    )
    # A device that differs from the host, so the assertion can fail.
    pipeline.device = torch.device("cuda")

    # Staged by the pipeline either way, so the weights are materialized on the
    # host and moved in only for the encode and decode windows.
    assert pipeline._vae_load_device() == torch.device("cpu")


def test_decode_stages_both_vaes_when_the_selection_omits_vae():
    # --layerwise-offload-components dit,text_encoder: both pipeline-staged VAEs
    # must still be loaded for the decode window and released afterwards.
    pipeline = _residency_pipeline(
        enable_layerwise_offload=True,
        layerwise_offload_components="dit,text_encoder",
    )
    pipeline.video_vae = Mock()
    pipeline.video_vae.decode_latent.return_value = torch.ones(1, 3, 2, 8, 8)
    pipeline.audio_vae = Mock()
    pipeline.audio_vae.decode_latent.return_value = torch.ones(1, 4)

    video, audio = pipeline.decode(torch.ones(1, 4, 2, 2, 2), torch.ones(1, 2), height=8, width=8)

    assert video.shape == (1, 3, 2, 8, 8)
    assert audio.shape == (1, 4)
    for vae in (pipeline.video_vae, pipeline.audio_vae):
        vae.load_to_device.assert_called_once_with()
        vae.offload_to_cpu.assert_called_once_with()


def test_default_selection_still_stages_every_family():
    # The lock on "default behavior is byte-for-byte unchanged": omitting the
    # option selects every family, so both sites behave exactly as before.
    pipeline = _residency_pipeline(enable_layerwise_offload=True)
    component = Mock()

    with pipeline._component_on_device(component, family="vae"):
        component.load_to_device.assert_called_once_with()
        component.offload_to_cpu.assert_not_called()

    component.offload_to_cpu.assert_called_once_with()
    assert pipeline._stages_component_family("text_encoder") is True
    assert pipeline._vae_load_device() == torch.device("cpu")


def test_distributed_layerwise_stages_families_the_selection_omits():
    # Distributed layer-wise offload shards every component it manages; its
    # residency does not go through the plain-layerwise selection.
    pipeline = _residency_pipeline(
        enable_distributed_layerwise_offload=True,
        layerwise_offload_components="dit",
    )
    component = Mock()

    with pipeline._component_on_device(component, family="vae"):
        component.load_to_device.assert_called_once_with()

    component.offload_to_cpu.assert_called_once_with()
    assert pipeline._vae_load_device() == torch.device("cpu")


def test_unselected_text_encoder_family_skips_the_host_round_trip():
    pipeline = _residency_pipeline(
        enable_layerwise_offload=True,
        layerwise_offload_components="dit,vae",
    )
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected

    actual = pipeline._encode_text_hidden(torch.tensor([1, 2]), {})

    assert actual is expected
    # Resident: loaded once, never pushed back to the host.
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.offload_to_cpu.assert_not_called()


def test_selected_text_encoder_family_keeps_the_host_round_trip():
    pipeline = _residency_pipeline(
        enable_layerwise_offload=True,
        layerwise_offload_components="dit,text_encoder",
    )
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected

    actual = pipeline._encode_text_hidden(torch.tensor([1, 2]), {})

    assert actual is expected
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.offload_to_cpu.assert_called_once_with()
