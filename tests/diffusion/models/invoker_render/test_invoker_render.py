# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import io
import json
import tempfile

import av
import numpy as np
import pytest
import torch
from fastapi import HTTPException, UploadFile
from PIL import Image
from safetensors.torch import save_file
from starlette.datastructures import Headers

from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.models.invoker_render.assets import InvokerRenderAssets
from vllm_omni.diffusion.models.invoker_render.causal import euler_step, generator
from vllm_omni.diffusion.models.invoker_render.codec import CausalVideoDecoder
from vllm_omni.diffusion.models.invoker_render.pipeline import get_invoker_render_pre_process_func
from vllm_omni.diffusion.models.invoker_render.schedule import WindowPlanner
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.entrypoints.openai.video.generation.helpers import (
    VideoRequestResources,
    _persist_uploaded_media_references,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def bundle(tmp_path):
    model = tmp_path / "transformer"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"_class_name": "MiniMaxH3Transformer3DModel", "num_layers": 1}))
    index = model / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"proj_in.weight": "shard.safetensors"}}))
    save_file({"proj_in.weight": torch.ones(2, 2)}, model / "shard.safetensors")
    decoder = tmp_path / "decoder.json"
    decoder.write_text("{}")
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"encoders": {"recipe": {"system_prompt": "Render the road"}}}))
    manifest = {
        "schema": "h3_omni_native/v1",
        "recipe": {
            "protocol": "cwm_causal/v1",
            "conditioning_mode": "static_image_qwen_proxy_vae_v1",
            "position_policy": "cwm_frame_aligned_v1",
            "sink_latents": 1,
            "recent_latents": 12,
            "bootstrap_latents": 9,
            "current_latent_schedule": [5],
            "segment_boundaries": [45, 70],
            "rope_reference_latents": 47,
            "guidance": {"mode": "plain", "drop": "proxy", "scale": 1.0},
            "diffusion": {
                "sampling_timesteps": {"args": {"T": 1.0, "num_sampling_steps": 4, "shift": 12.0}},
                "sampler": {"class_name": "EulerSampler"},
            },
        },
        "canvas": {"width": 1344, "height": 768, "proxy_width": 448, "proxy_height": 256},
        "cwm": {"posterior": "mean", "proxy_encode_width": 1344, "proxy_encode_height": 768, "proxy_resize": "nearest"},
        "codec": {
            "encoder": "h3",
            "decoder": "causal",
            "window": 16,
            "decoder_config": str(decoder),
            "decoder_weights": str(model / "shard.safetensors"),
        },
        "recipe_selection": {"source_artifact": str(artifact)},
        "server_args": {
            "transformer_weights_path": str(index),
            "component_paths": {
                "processor": str(tmp_path),
                "text_encoder": str(tmp_path),
                "video_vae": str(tmp_path),
            },
        },
    }
    (tmp_path / "native-runtime.json").write_text(json.dumps(manifest))
    return tmp_path


def test_bundle_discovery_and_missing_shard(bundle):
    assert resolve_model_class_name(str(bundle)) == "InvokerRenderPipeline"
    config = OmniDiffusionConfig.from_kwargs(model=str(bundle))
    config.enrich_config()
    assert config.model_class_name == "InvokerRenderPipeline"
    assert InvokerRenderAssets.read(str(bundle)).proxy_size == (448, 256)
    (bundle / "transformer" / "shard.safetensors").unlink()
    with pytest.raises(ValueError, match="shard.safetensors"):
        InvokerRenderAssets.read(str(bundle))


def test_admission_keeps_request_identity(bundle):
    pre_process = get_invoker_render_pre_process_func(OmniDiffusionConfig.from_kwargs(model=str(bundle)))
    sampling = OmniDiffusionSamplingParams(seed=42, extra_args={"session_id": "other"})
    request = OmniDiffusionRequest(
        {
            "prompt": "Render the road",
            "multi_modal_data": {
                "image": Image.new("RGB", (1344, 768)),
                "video": np.zeros((39, 256, 448, 3), dtype=np.uint8),
            },
        },
        sampling,
        "file-1",
    )
    result = pre_process(request)
    assert result.sampling_params.extra_args == {"session_id": "file-1", "close_session": True, "reset": True}
    assert result.sampling_params.num_frames == 39
    assert sampling.num_frames == 1
    result.prompt["multi_modal_data"]["video"] = result.prompt["multi_modal_data"]["video"][:-1]
    with pytest.raises(OmniClientError, match="at least 39 frames"):
        pre_process(result)


def test_window_boundaries_and_eos():
    short = WindowPlanner()
    assert not short.append()
    assert [(w.start, w.stop, w.eos) for w in short.append()] == [(0, 9, False)]
    assert [(w.start, w.stop, w.eos) for w in short.append(eos=True)] == [(9, 12, True)]
    with pytest.raises(ValueError, match="after EOS"):
        short.append()
    long = WindowPlanner()
    windows = [w for _ in range(15) for w in long.append()] + list(long.append(eos=True))
    assert [w.stop for w in windows] == [9, 14, 19, 24, 29, 34, 39, 44, 45, 50, 55, 60, 65, 70, 75, 77]
    assert [(w.start, w.rope_shift_blocks) for w in windows if w.start in (44, 45, 70)] == [(44, 0), (45, 1), (70, 6)]
    assert windows[-1].recent_start == 63


def test_euler_uses_bfloat16_projection_and_seed_scope():
    values = torch.randn((2, 2, 3), generator=generator("cpu", 42, "proxy", 0))
    assert torch.equal(values, torch.randn(values.shape, generator=generator("cpu", 42, "proxy", 0)))
    assert not torch.equal(values, torch.randn(values.shape, generator=generator("cpu", 42, "proxy", 1)))
    velocity = torch.full_like(values, 0.5)
    result = euler_step(
        values, velocity, torch.tensor(0.75, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)
    )
    torch.testing.assert_close(result, values.bfloat16().float() + 0.75 * velocity, atol=0, rtol=0)


@pytest.mark.parametrize("shift", [0, 8])
def test_decoder_window_partition_keeps_spatial_geometry(shift):
    values = torch.arange(2 * 17 * 31 * 6).view(1, 2, 17, 31, 2, 3)
    windows, mask, shape = CausalVideoDecoder._windows(values, shift)
    decoded = CausalVideoDecoder._unwindow(windows, shape, shift)
    assert mask.sum().item() == 2 * 17 * 31
    assert torch.equal(decoded, values.flatten(-2))


def test_mixed_matroska_upload_preserves_full_proxy(bundle, monkeypatch):
    path = bundle / "original.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=24)
        stream.width, stream.height, stream.pix_fmt = 448, 256, "bgr0"
        for index in range(39):
            frame = av.VideoFrame.from_ndarray(np.full((256, 448, 3), index, dtype=np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    rgb = io.BytesIO()
    Image.new("RGB", (1344, 768)).save(rgb, format="PNG")

    def uploads():
        return [
            UploadFile(
                io.BytesIO(rgb.getvalue()), filename="first.png", headers=Headers({"content-type": "image/png"})
            ),
            UploadFile(
                io.BytesIO(path.read_bytes()),
                filename="proxy.mkv",
                headers=Headers({"content-type": "video/x-matroska"}),
            ),
        ]

    monkeypatch.setattr(tempfile, "tempdir", str(bundle))
    rejected_resources = VideoRequestResources()
    with pytest.raises(HTTPException, match="MP4 or MOV"):
        asyncio.run(_persist_uploaded_media_references(uploads(), rejected_resources))
    assert not rejected_resources.paths
    resources = VideoRequestResources()
    try:
        images, videos, audios = asyncio.run(_persist_uploaded_media_references(uploads(), resources, proxy_video=True))
        assert len(images) == len(videos) == 1 and not audios
        assert resources.paths == videos
        pre_process = get_invoker_render_pre_process_func(OmniDiffusionConfig.from_kwargs(model=str(bundle)))
        request = OmniDiffusionRequest(
            {"prompt": "Render the road", "multi_modal_data": {"image": images[0], "video": videos}},
            OmniDiffusionSamplingParams(seed=42),
            "upload",
        )
        proxy = pre_process(request).prompt["multi_modal_data"]["video"]
        np.testing.assert_array_equal(proxy[:, 0, 0, 0], np.arange(39))
    finally:
        resources.cleanup()
