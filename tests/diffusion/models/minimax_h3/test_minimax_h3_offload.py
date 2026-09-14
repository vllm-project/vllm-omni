# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager
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


@pytest.mark.parametrize("encode_fails", [False, True])
def test_h3_model_cpu_offload_batches_visual_reference_scope(monkeypatch, encode_fails):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.video_vae = Mock()
    pipeline.video_vae.is_distributed_enabled.return_value = False
    pipeline.audio_vae = Mock()
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
    media = _media(images=(torch.zeros(16, 16, 3, dtype=torch.uint8), torch.zeros(16, 32, 3, dtype=torch.uint8)))

    if encode_fails:
        with pytest.raises(RuntimeError, match="encode failed"):
            pipeline._encode_local_media(media)
    else:
        conditioning = pipeline._encode_local_media(media)
        assert conditioning.visual_condition.shape == (3, 4)
        assert conditioning.visual_condition_shapes == ((1, 1, 1), (1, 1, 2))

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
    pipeline.audio_vae = Mock()
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
    monkeypatch.setattr(module, "_dit_rank_world", lambda: (None, 0, 1))
    monkeypatch.setattr(module, "_broadcast_tensor", lambda value, **kwargs: value)
    media = _media(
        images=(torch.zeros(16, 16, 3, dtype=torch.uint8), torch.zeros(16, 32, 3, dtype=torch.uint8)),
        videos=(torch.zeros(3, 32, 32, 3, dtype=torch.uint8),),
        video_audios=(None,),
    )
    conditioning = pipeline._encode_local_media(media)

    assert conditioning.visual_condition.shape == (5, 4)
    assert conditioning.visual_condition_shapes == ((1, 1, 1), (1, 1, 2), (3, 2, 2))
    assert events == [
        ("activate", pipeline.video_vae),
        ("image", (16, 16)),
        ("image", (32, 16)),
        ("video", (3, 32, 32, 3)),
        ("offload", pipeline.video_vae),
    ]


@pytest.mark.parametrize("encode_fails", [False, True])
def test_h3_model_cpu_offload_shares_embedded_and_standalone_audio_scope(monkeypatch, encode_fails):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.video_vae = Mock()
    pipeline.video_vae.is_distributed_enabled.return_value = False
    pipeline.video_vae.encode_video.return_value = (torch.ones(1, 96), (1, 2, 2))
    pipeline.audio_vae = Mock()
    pipeline._model_cpu_offload_modules = [pipeline.audio_vae]
    events: list[tuple[Any, ...]] = []

    @contextmanager
    def record_component(value):
        events.append(("activate", value))
        try:
            yield
        finally:
            events.append(("offload", value))

    def encode_audio(waveform, sample_rate):
        embedded = bool(waveform[0].item())
        events.append(("embedded" if embedded else "standalone",))
        if encode_fails and embedded:
            raise RuntimeError("audio encode failed")
        return torch.ones(160, 32), 80

    pipeline.audio_vae.encode_waveform.side_effect = encode_audio
    monkeypatch.setattr(module, "sequential_offload_component", record_component)
    monkeypatch.setattr(module, "_dit_rank_world", lambda: (None, 0, 1))

    def call():
        return pipeline._encode_local_media(
            _media(
                videos=(torch.zeros(1, 32, 32, 3, dtype=torch.uint8),),
                video_audios=((torch.ones(32_000), 16_000),),
                audios=((torch.zeros(32_000), 16_000),),
            )
        )

    if encode_fails:
        with pytest.raises(RuntimeError, match="audio encode failed"):
            call()
    else:
        result = call()
        assert result.audio_condition.shape == (320, 32)
        assert result.audio_condition_lengths == (80, 80)

    assert ("activate", pipeline.audio_vae) in events
    assert events[-1] == ("offload", pipeline.audio_vae)
    assert events.count(("activate", pipeline.audio_vae)) == 1
    assert events.count(("offload", pipeline.audio_vae)) == 1
