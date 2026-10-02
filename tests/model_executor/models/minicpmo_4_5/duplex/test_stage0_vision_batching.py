# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o 4.5 omni-duplex camera frames: one encoder call per append and across sessions."""

from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
    MiniCPMO45Stage0DuplexRuntime,
    _MiniCPMO45Stage0SessionState,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMVImageProcessor,
    Resampler,
    SiglipVisionConfig,
    SiglipVisionTransformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Processor:
    """``process_image`` of the checkpoint's MiniCPMOProcessor, over the in-tree image processor."""

    tokenizer = None

    def __init__(self) -> None:
        self.image_processor = MiniCPMVImageProcessor(max_slice_nums=9, scale_resolution=448, patch_size=14)
        self.calls: list[int] = []

    def process_image(self, images, max_slice_nums=1):
        self.calls.append(int(max_slice_nums))
        return self.image_processor.preprocess(images, max_slice_nums=max_slice_nums, return_tensors="pt")


class _Thinker:
    """Real tiny SigLIP + resampler behind the Thinker's ``get_vision_hidden_states``."""

    def __init__(self) -> None:
        torch.manual_seed(0)
        config = SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=980,
            patch_size=14,
        )
        config._attn_implementation = "sdpa"
        self.vpm = SiglipVisionTransformer(config).eval()
        self.resampler = Resampler(num_queries=64, embed_dim=64, num_heads=4, kv_dim=32, adaptive=True).eval()
        for param in self.resampler.parameters():
            torch.nn.init.normal_(param, std=0.2)
        self.config = SimpleNamespace(vision_batch_size=16)
        self.vision_packed_encode = True
        self.llm = SimpleNamespace(model=SimpleNamespace(embed_tokens=torch.nn.Embedding(8, 64)))
        self.encoder_calls: list[int] = []

    def get_vision_hidden_states(self, data):
        self.encoder_calls.append(len(data["pixel_values"]))
        assert data["tgt_sizes"].device.type == "cpu"  # grids stay on the host: no device sync
        return MiniCPMO45OmniLLMForConditionalGeneration.get_vision_hidden_states(self, data)

    def parameters(self):
        return self.vpm.parameters()


def _runtime() -> MiniCPMO45Stage0DuplexRuntime:
    stage_model = SimpleNamespace(thinker=_Thinker(), processor=_Processor())
    return MiniCPMO45Stage0DuplexRuntime(stage_model, device="cpu")


def _jpeg(width: int, height: int, seed: int) -> str:
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(0, 256, (height, width, 3), generator=generator, dtype=torch.uint8).numpy()
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _append(session_id: str, seq: int, frames: list[str], epoch: int = 0) -> dict[str, Any]:
    return {
        "data_plane": True,
        "session_id": session_id,
        "epoch": epoch,
        "seq": seq,
        "payload": {"audio": "", "video_frames": frames},
    }


def _assert_same_blocks(actual: list[list[torch.Tensor]], expected: list[list[torch.Tensor]]) -> None:
    assert [len(frame) for frame in actual] == [len(frame) for frame in expected]
    for actual_frame, expected_frame in zip(actual, expected):
        for actual_block, expected_block in zip(actual_frame, expected_frame):
            assert actual_block.shape == (64, 64)
            torch.testing.assert_close(actual_block, expected_block, rtol=1e-5, atol=1e-5)


def _per_frame_reference(runtime: MiniCPMO45Stage0DuplexRuntime, frames: list[Image.Image]) -> list[list[torch.Tensor]]:
    """What the per-frame loop computed: each frame sliced and encoded alone."""
    reference = []
    for frame, max_slices in zip(frames, runtime._official_max_slice_nums(len(frames))):
        processed = runtime.processor.process_image([frame], max_slice_nums=max_slices)
        encoded = runtime._encode_processed_vision_batch([processed])
        assert encoded is not None
        reference.append(encoded[0])
    return reference


def test_stacked_pair_is_encoded_in_one_call_and_matches_per_frame() -> None:
    runtime = _runtime()
    frames = runtime._decode_video_frames_payload({"video_frames": [_jpeg(960, 540, 1), _jpeg(640, 480, 2)]})
    expected = _per_frame_reference(runtime, frames)
    runtime.thinker.encoder_calls.clear()

    with torch.inference_mode():
        encoded = runtime._stage_vision_embeddings(frames)

    assert runtime.thinker.encoder_calls == [4]  # HD base frame: 1 source + 2 slices, composite: 1
    assert encoded is not None
    _assert_same_blocks(encoded, expected)


def test_prefetch_encodes_all_sessions_in_one_call() -> None:
    runtime = _runtime()
    appends = [
        _append("a", 3, [_jpeg(640, 480, 10)]),
        _append("b", 7, [_jpeg(960, 540, 11), _jpeg(640, 480, 12)]),
        _append("c", 1, [_jpeg(480, 640, 13)]),
    ]
    expected = {
        duplex["session_id"]: _per_frame_reference(runtime, runtime._decode_video_frames_payload(duplex["payload"]))
        for duplex in appends
    }
    runtime.thinker.encoder_calls.clear()

    with torch.inference_mode():
        runtime.prefetch_vision(appends)

    assert runtime.thinker.encoder_calls == [1 + 4 + 1]
    for duplex in appends:
        encoded = runtime.take_prefetched_vision(duplex)
        assert encoded is not None
        assert len(encoded) == len(duplex["payload"]["video_frames"])
        _assert_same_blocks(encoded, expected[duplex["session_id"]])
        assert runtime.take_prefetched_vision(duplex) is None  # consumed once


def test_prefetch_chunk_size_follows_vision_batch_size_config() -> None:
    runtime = _runtime()
    assert runtime._vision_prefetch_chunk_size() == 16  # from _Thinker.config.vision_batch_size

    runtime.thinker.config.vision_batch_size = 3
    assert runtime._vision_prefetch_chunk_size() == 3

    # Missing/invalid config falls back to 16.
    runtime.thinker.config.vision_batch_size = None
    assert runtime._vision_prefetch_chunk_size() == 16
    del runtime.thinker.config.vision_batch_size
    assert runtime._vision_prefetch_chunk_size() == 16


def test_prefetch_pipelines_chunks_without_changing_results() -> None:
    """Small ``vision_batch_size`` forces multiple pipelined chunks; results must match single-chunk encoding."""
    runtime = _runtime()
    runtime.thinker.config.vision_batch_size = 2  # 5 one-slice jobs -> chunks of [2, 2, 1]
    appends = [_append(chr(ord("a") + i), i, [_jpeg(640, 480, 100 + i)]) for i in range(5)]
    expected = {
        duplex["session_id"]: _per_frame_reference(runtime, runtime._decode_video_frames_payload(duplex["payload"]))
        for duplex in appends
    }
    runtime.thinker.encoder_calls.clear()

    with torch.inference_mode():
        runtime.prefetch_vision(appends)

    assert runtime.thinker.encoder_calls == [2, 2, 1]  # one encode call per pipelined chunk
    for duplex in appends:
        encoded = runtime.take_prefetched_vision(duplex)
        assert encoded is not None
        _assert_same_blocks(encoded, expected[duplex["session_id"]])


def test_prefetch_single_job_skips_pool_and_still_encodes() -> None:
    runtime = _runtime()
    appends = [_append("solo", 1, [_jpeg(640, 480, 200)])]
    expected = _per_frame_reference(runtime, runtime._decode_video_frames_payload(appends[0]["payload"]))
    runtime.thinker.encoder_calls.clear()

    with torch.inference_mode():
        runtime.prefetch_vision(appends)

    assert runtime.thinker.encoder_calls == [1]
    encoded = runtime.take_prefetched_vision(appends[0])
    assert encoded is not None
    _assert_same_blocks(encoded, expected)


def test_prefetch_skips_prepared_appends_and_rejects_changed_payloads() -> None:
    runtime = _runtime()
    prepared = _MiniCPMO45Stage0SessionState(session_id="a")
    prepared.prepared_append_identity = (0, 3)
    prepared.prepared_inputs_embeds = torch.zeros(1, 64)
    runtime.sessions["a"] = prepared
    appends = [_append("a", 3, [_jpeg(640, 480, 20)]), _append("b", 4, [_jpeg(640, 480, 21)])]

    with torch.inference_mode():
        runtime.prefetch_vision(appends)

    assert runtime.thinker.encoder_calls == [1]  # session a replays its cached append
    assert runtime.take_prefetched_vision(appends[0]) is None
    changed = _append("b", 4, [_jpeg(640, 480, 22)])
    assert runtime.take_prefetched_vision(changed) is None


def test_prefetch_leaves_bad_frames_to_the_per_request_path() -> None:
    runtime = _runtime()
    good = _append("a", 1, [_jpeg(640, 480, 30)])
    bad = _append("b", 1, ["bm90IGEganBlZw=="])  # valid base64, not an image

    with torch.inference_mode():
        runtime.prefetch_vision([good, bad])

    assert runtime.take_prefetched_vision(bad) is None
    assert runtime.take_prefetched_vision(good) is not None
    with pytest.raises(ValueError, match="invalid omni duplex video frame payload"):
        runtime._decode_video_frames_payload(bad["payload"])


def test_prefetch_drops_unconsumed_entries_next_step() -> None:
    runtime = _runtime()
    duplex = _append("a", 1, [_jpeg(640, 480, 40)])
    with torch.inference_mode():
        runtime.prefetch_vision([duplex])
        runtime.prefetch_vision([])
    assert runtime.take_prefetched_vision(duplex) is None


def test_omni_preprocess_batch_prefetches_only_for_multiple_frame_appends() -> None:
    calls: list[list[str]] = []
    helper = SimpleNamespace(
        prefetch_vision=lambda appends: calls.append([a["session_id"] for a in appends]),
        batches_audio_encoder=lambda: False,
    )
    model = SimpleNamespace(
        model_stage="llm",
        _minicpmo45_duplex_data_plane_helper=helper,
        _duplex_data_plane_helper=lambda: helper,
    )
    buffer = {
        "r0": {"duplex": _append("a", 1, ["x"])},
        "r1": {"duplex": _append("b", 1, [])},  # audio-only append
        "r2": {"duplex": _append("c", 1, ["y"])},
        "r3": {"prompt_token_ids": [1, 2]},  # ordinary request
    }

    MiniCPMO45OmniForConditionalGeneration.preprocess_batch(
        model, req_ids=["r0", "r1", "r2", "r3"], model_intermediate_buffer=buffer, device=torch.device("cpu")
    )
    MiniCPMO45OmniForConditionalGeneration.preprocess_batch(
        model, req_ids=["r0", "r3"], model_intermediate_buffer=buffer, device=torch.device("cpu")
    )
    model.model_stage = "tts"
    MiniCPMO45OmniForConditionalGeneration.preprocess_batch(
        model, req_ids=["r0", "r2"], model_intermediate_buffer=buffer, device=torch.device("cpu")
    )

    assert calls == [["a", "c"], []]
