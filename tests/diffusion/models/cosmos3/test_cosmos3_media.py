# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from diffusers.video_processor import VideoProcessor

from vllm_omni.diffusion import ipc
from vllm_omni.diffusion.data import VideoOutputTransportConfig
from vllm_omni.diffusion.media import FloatVideoConsumer, VideoTensorEncoding
from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3
from vllm_omni.diffusion.postprocess import device_reduction
from vllm_omni.diffusion.postprocess import media as media_postprocess
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# RetinaFace (via cosmos_guardrail) disables autograd at import time. Restore
# the caller's grad mode so collection does not affect unrelated tests.
with torch.set_grad_enabled(torch.is_grad_enabled()):
    from vllm_omni.diffusion.models.cosmos3 import guardrails

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def config(enabled=True, guardrails_enabled=False, **kwargs):
    return SimpleNamespace(
        video_output_transport=VideoOutputTransportConfig(enable_device_postprocess=enabled),
        model_config={"guardrails": guardrails_enabled},
        **kwargs,
    )


def decoded(dtype=torch.bfloat16, device="cpu"):
    values = torch.tensor([-1.4, -1, -0.50390625, 0, 0.50390625, 1, 1.4], dtype=dtype, device=device)
    return values.view(1, 1, 7, 1, 1).expand(1, 3, 7, 2, 2).clone()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("chunk_bytes", [1, 64 << 20])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_reducer_preserves_cosmos3_bytes(monkeypatch, dtype, chunk_bytes, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setattr(device_reduction, "_VIDEO_CHUNK_BYTES", chunk_bytes)
    video = decoded(dtype, device)
    original = video.clone()
    expected = VideoProcessor().postprocess_video(video, output_type="np")
    actual = device_reduction.reduce_video_to_uint8_frames(video, preserve_input_dtype=True)
    np.testing.assert_array_equal(actual.cpu().numpy(), np.round(expected * 255).astype(np.uint8))
    assert actual.is_contiguous()
    assert torch.equal(video, original)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("output_type", [None, "np", "pt", "pil", "latent"])
@pytest.mark.parametrize("checks", [False, True])
def test_cosmos3_shared_transport_preserves_presentation_and_guardrail_bytes(monkeypatch, enabled, output_type, checks):
    cfg = config(enabled, checks)
    params = OmniDiffusionSamplingParams(output_type=output_type)
    video = decoded()
    captured = []

    def check(frames):
        captured.append(frames.copy())
        return frames

    monkeypatch.setattr(guardrails, "_video_guardrail", check)
    reference_video = guardrails.check_video_safety(video) if checks else video
    expected = VideoProcessor().postprocess_video(reference_video, output_type="np")
    result = pipeline_cosmos3._cosmos3_media_output({"video": video}, cfg, params)
    typed = not checks and output_type in (None, "np")
    assert (result.media is not None) == typed
    if typed:
        assert not result.media.prepared_for_transport
        result.media = device_reduction.prepare_diffusion_media_for_transport(
            result.media, od_config=cfg, sampling_params=params
        )
        assert (result.media.video.spec.encoding is VideoTensorEncoding.UINT8_FRAMES) == enabled
    monkeypatch.setattr(ipc, "_SHM_TENSOR_THRESHOLD", 1)
    ipc.pack_diffusion_output_shm(result)
    ipc.unpack_diffusion_output_shm(result)
    if typed:
        actual = media_postprocess.finalize_diffusion_media(result.media, sampling_params=params)["payload"]["video"]
    else:
        actual = pipeline_cosmos3.get_cosmos3_post_process_func(cfg)(result.output, sampling_params=params)
    narrowed = typed and enabled
    assert isinstance(actual, np.ndarray)
    if narrowed:
        expected = np.round(expected * 255).astype(np.uint8)
    np.testing.assert_array_equal(actual, expected)
    if checks:
        np.testing.assert_array_equal(captured[0], captured[1])


@pytest.mark.parametrize("kind", ["audio", "actions", "transfer"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_auxiliary_outputs_retain_legacy_float_ipc(monkeypatch, kind, enabled, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cfg = config(enabled)
    params = OmniDiffusionSamplingParams()
    payload = {"video": decoded(device=device)}
    metadata: dict[str, Any] = {"video": {"fps": 12}}
    extra = torch.arange(24, dtype=torch.float32, device=device).reshape(1, 2, 12)
    if kind == "audio":
        payload.update(audio=extra, audio_sample_rate=48000)
    elif kind == "actions":
        payload["actions"] = extra
        metadata["actions"] = {"action_mode": "policy"}
    else:
        metadata["transfer"] = {"controls": {"edge": extra}, "hints": ["edge"]}
    result = pipeline_cosmos3._cosmos3_media_output({"payload": payload, "metadata": metadata}, cfg, params)
    assert result.media is None
    assert result.output["payload"]["video"] is payload["video"]
    monkeypatch.setattr(ipc, "_SHM_TENSOR_THRESHOLD", 1)
    ipc.pack_diffusion_output_shm(result)
    ipc.unpack_diffusion_output_shm(result)
    actual = pipeline_cosmos3.get_cosmos3_post_process_func(cfg)(result.output, sampling_params=params)
    expected = VideoProcessor().postprocess_video(payload["video"], output_type="np")
    np.testing.assert_array_equal(actual["payload"]["video"], expected)
    assert actual["metadata"]["video"]["fps"] == 12
    if kind == "audio":
        torch.testing.assert_close(actual["payload"]["audio"], extra.cpu())
        assert actual["metadata"]["audio"]["sample_rate"] == 48000
    elif kind == "actions":
        torch.testing.assert_close(actual["payload"]["actions"], extra.cpu())
    else:
        torch.testing.assert_close(actual["metadata"]["transfer"]["controls"]["edge"], extra.cpu())


@pytest.mark.parametrize(
    "flag", ["enable_cpu_offload", "enable_layerwise_offload", "enable_distributed_layerwise_offload"]
)
def test_offload_retains_legacy_float_output(flag):
    cfg = config(**{flag: True})
    params = OmniDiffusionSamplingParams()
    video = decoded()
    result = pipeline_cosmos3._cosmos3_media_output({"video": video}, cfg, params)
    assert result.media is None
    assert result.output["video"] is video


def test_interpolation_uses_shared_float_video(monkeypatch):
    cfg = config()
    params = OmniDiffusionSamplingParams(enable_frame_interpolation=True)
    video = decoded()
    result = pipeline_cosmos3._cosmos3_media_output({"video": video}, cfg, params)
    prepared = device_reduction.prepare_diffusion_media_for_transport(
        result.media, od_config=cfg, sampling_params=params
    )
    assert prepared.video.tensor is video
    assert FloatVideoConsumer.FRAME_INTERPOLATION in prepared.video.constraints.pending_float_consumers
    monkeypatch.setattr(media_postprocess, "interpolate_video_tensor", lambda tensor, **kwargs: (tensor, 2))
    actual = media_postprocess.finalize_diffusion_media(prepared, sampling_params=params)
    assert actual["payload"]["video"].dtype == np.float32
    assert actual["metadata"]["video"] == {"video_fps_multiplier": 2}


@pytest.mark.parametrize(
    "server,request_override,expected",
    [(True, False, False), (True, True, True), (True, None, True), (False, True, False)],
)
def test_guardrail_legacy_path_respects_request_override(server, request_override, expected):
    params = OmniDiffusionSamplingParams(extra_args={"guardrails": request_override})
    result = pipeline_cosmos3._cosmos3_media_output({"video": decoded()}, config(guardrails_enabled=server), params)
    assert (result.media is None) == expected


@pytest.mark.parametrize("failure", ["output", "chunk"])
def test_shared_oom_fallback_preserves_cosmos3_float_output(monkeypatch, failure):
    import weakref

    cfg = config()
    params = OmniDiffusionSamplingParams()
    video = decoded()
    expected = VideoProcessor().postprocess_video(video, output_type="np")
    result = pipeline_cosmos3._cosmos3_media_output({"video": video}, cfg, params)
    buffers = []
    real_empty, real_to = torch.empty, torch.Tensor.to

    def empty(*args, **kwargs):
        if failure == "output":
            raise torch.OutOfMemoryError("output OOM")
        tensor = real_empty(*args, **kwargs)
        buffers.append(weakref.ref(tensor))
        return tensor

    def to(tensor, *args, **kwargs):
        if args == (torch.uint8,):
            buffers.append(weakref.ref(tensor))
            raise torch.OutOfMemoryError("chunk OOM")
        return real_to(tensor, *args, **kwargs)

    def empty_cache():
        assert all(ref() is None for ref in buffers)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", empty)
        patch.setattr(torch.Tensor, "to", to)
        patch.setattr(device_reduction.current_omni_platform, "empty_cache", empty_cache)
        patch.setattr(device_reduction.current_omni_platform, "synchronize", lambda: None)
        prepared = device_reduction.prepare_diffusion_media_for_transport(
            result.media, od_config=cfg, sampling_params=params
        )
    assert prepared.video.tensor is video
    actual = media_postprocess.finalize_diffusion_media(prepared, sampling_params=params)
    np.testing.assert_array_equal(actual["payload"]["video"], expected)


@pytest.mark.parametrize("output_type", [None, "np", "pil", "pt", "latent"])
def test_cosmos3_video_runs_through_engine(monkeypatch, output_type):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.request import OmniDiffusionRequest

    cfg = OmniDiffusionConfig(
        model=None,
        model_config={"guardrails": False},
        video_output_transport=VideoOutputTransportConfig(enable_device_postprocess=True),
    )
    cfg.model_class_name = "Cosmos3OmniDiffusersPipeline"
    params = OmniDiffusionSamplingParams(output_type=output_type)
    result = pipeline_cosmos3._cosmos3_media_output({"video": decoded()}, cfg, params)
    if result.media is not None:
        result.media = device_reduction.prepare_diffusion_media_for_transport(
            result.media, od_config=cfg, sampling_params=params
        )
    monkeypatch.setattr(ipc, "_SHM_TENSOR_THRESHOLD", 1)
    ipc.pack_diffusion_output_shm(result)
    ipc.unpack_diffusion_output_shm(result)
    engine = object.__new__(DiffusionEngine)
    engine.od_config = cfg
    engine.post_process_func = pipeline_cosmos3.get_cosmos3_post_process_func(cfg)
    engine._post_process_accepts_sampling_params = True
    request = OmniDiffusionRequest(prompt="test", sampling_params=params, request_id="cosmos3-media")
    outputs = engine.postprocess_output(request, result)
    assert len(outputs) == 1
    if output_type in (None, "np"):
        expected = device_reduction.reduce_video_to_uint8_frames(decoded(), preserve_input_dtype=True).numpy()
    else:
        expected = VideoProcessor().postprocess_video(decoded(), output_type="np")
    np.testing.assert_array_equal(outputs[0].images[0], expected)
