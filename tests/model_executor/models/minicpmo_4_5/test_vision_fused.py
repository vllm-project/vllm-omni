# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused packed SigLIP layers match the eager packed encode; prefetched frames serve only their own append."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5 import vision_fused
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    SiglipVisionConfig,
    SiglipVisionTransformer,
    _vision_encode_paths,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_fused_layers_match_eager_packed_layers() -> None:
    torch.manual_seed(0)
    sizes = dict(hidden_size=32, intermediate_size=72, num_hidden_layers=3, num_attention_heads=4)
    config = SiglipVisionConfig(**sizes, image_size=28, patch_size=2, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    vpm = SiglipVisionTransformer(config).eval()
    for norm in [module for module in vpm.modules() if isinstance(module, torch.nn.LayerNorm)]:
        torch.nn.init.normal_(norm.weight, mean=1.0, std=0.2)
        torch.nn.init.normal_(norm.bias, std=0.2)
    seq_groups, hidden = [(0, 3, 16), (48, 2, 15)], torch.randn((78, 32), generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        expected = hidden.clone()
        for layer in vpm.encoder.layers:
            expected = layer.forward_packed(expected, seq_groups)
        fused = vision_fused.encode_packed_fused(vpm, hidden.clone(), seq_groups)
        torch.testing.assert_close(fused, vpm.post_layernorm(expected), rtol=1e-5, atol=1e-5)


def test_prefetched_frames_are_chunked_and_keyed_by_append() -> None:
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.sessions, runtime._prefetched_vision = {}, {}
    runtime.processor = SimpleNamespace(process_image=lambda *args, **kwargs: None)
    runtime.stage_model = runtime.thinker = SimpleNamespace(config=SimpleNamespace(vision_batch_size=2))
    runtime._decode_video_frames_payload = lambda payload: list(payload["video_frames"])
    runtime._preprocess_frames = lambda frames: [f"p{frame}" for frame in frames]
    # Each encoded frame is tagged with its tower call's size: vision_batch_size appends per call.
    runtime._encode_processed_vision_batch = lambda items: [[f"e{item}/{len(items)}"] for item in items]
    frames = ["a", "b", "c"]
    appends = [
        {"session_id": f"s{i}", "epoch": 0, "seq": 1, "payload": {"video_frames": frames[: i + 1]}} for i in range(3)
    ]
    runtime.prefetch_vision(appends)
    assert runtime.take_prefetched_vision({**appends[1], "payload": {"video_frames": ["x", "y"]}}) is None  # changed
    assert runtime.take_prefetched_vision(appends[1]) is None  # and the entry is gone
    assert runtime.take_prefetched_vision({**appends[2], "seq": "2"}) is None  # another append
    last, payload = appends[2], appends[2]["payload"]
    assert runtime.frame_kwargs(last, payload) == {"encoded_frames": [["epa/3"], ["epb/3"], ["epc/3"]]}
    assert runtime.frame_kwargs(last, payload) == {"video_frames": frames}  # taken once


def test_packed_vision_adapter_protocol() -> None:
    shape, out_shape = (1, 3, 16), torch.Size([1, 4, 8])
    adapter = vision_fused._PackedVisionAdapter(lambda px, runs: px[:, :4, :8], [(2, 2, 1)], shape, out_shape)
    assert adapter.supports_encoder_cudagraph is True
    cfg = adapter.get_encoder_cudagraph_config()
    assert cfg.modalities == ["image"]
    assert cfg.buffer_keys == ["pixels"]
    assert cfg.out_hidden_size == 8
    inputs = adapter.prepare_encoder_cudagraph_capture_inputs(4, 1, 0, torch.device("cpu"), torch.float32)
    assert "pixels" in inputs.values
    dest: list[torch.Tensor | None] = [None]
    adapter.postprocess_encoder_output({"default": torch.ones(1, 4, 8)}, [0], [4], dest, clone=True)
    assert dest[0] is not None and dest[0].shape == (1, 4, 8)


@pytest.mark.parametrize(
    ("overrides", "encoder_graphs", "expected"),
    [
        ({}, True, (False, False)),  # default: the padded batch keeps the vpm encoder graph
        ({"vision_cuda_graph": True}, True, (True, True)),
        ({"vision_cuda_graph": True}, False, (False, False)),  # encoder_cuda_graph=False / --enforce-eager
        ({"vision_packed_encode": True}, True, (True, False)),  # packed eager, for A/B
        ({"vision_packed_encode": False, "vision_cuda_graph": True}, True, (False, False)),
    ],
)
def test_vision_encode_paths(overrides, encoder_graphs, expected) -> None:
    assert _vision_encode_paths(SimpleNamespace(**overrides), encoder_graphs=encoder_graphs) == expected
