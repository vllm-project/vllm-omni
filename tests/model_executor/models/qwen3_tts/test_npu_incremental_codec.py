# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise incremental codec tail fallback and request slot reuse on NPU."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"npu": "A2"}, num_cards=1)]


@torch.inference_mode()
@pytest.mark.parametrize("explicit_head_dim", [False, True])
def test_partial_chunks_do_not_capture_and_slots_reset(monkeypatch, explicit_head_dim):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.platforms.npu.layers.incremental_codec import NPUIncrementalCodec, _ShapeGraph
    from vllm_omni.platforms.npu.models.qwen3_tts import _prepare_npu_code2wav_runtime

    _prepare_npu_code2wav_runtime()
    decoder = _make_decoder().to(device="npu", dtype=torch.bfloat16).eval()
    if explicit_head_dim:
        decoder.config.head_dim = decoder.config.hidden_size // decoder.config.num_attention_heads
    for module in decoder.modules():
        if hasattr(module, "precompute_embedding"):
            module.embedding_sum.normal_()
            module.precompute_embedding()
    decoder.precompute_snake_caches()
    arena = NPUIncrementalCodec(decoder, slots=2, max_batch=2, chunk_frames=4)
    codes = torch.randint(0, 32, (2, 2, 7), device="npu")
    caches: list[dict[str, int]] = [{}, {}]

    def unexpected_capture(self):
        pytest.fail("a live partial chunk must not capture a new graph")

    monkeypatch.setattr(_ShapeGraph, "capture", unexpected_capture)
    try:
        # Both requests span a captured full chunk and an uncaptured tail.
        first = arena.decode_requests(codes, [7, 7], caches)
        retained = [wave.clone() for wave in first]
        assert set(arena.graphs) == {(1, 1), (1, 4), (2, 1), (2, 4)}
        assert [cache["npu_produced"] for cache in caches] == [7, 7]
        for row in range(2):
            assert first[row].numel() == 7 * decoder.total_upsample
        # Poison history with another chunk, release, then replay the same
        # input with fresh requests. Output and all streaming state must reset.
        arena.decode_requests(codes[:, :, :3], [3, 3], caches)
        for cache in caches:
            arena.release(cache["npu_slot"])
        repeated = arena.decode_requests(codes, [7, 7], [{}, {}])
        for actual, expected, original in zip(repeated, retained, first, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(original, expected, atol=0, rtol=0)
    finally:
        arena.close()


def test_weight_reload_releases_incremental_graphs_before_loading(monkeypatch):
    pytest.importorskip("torch_npu")
    from vllm_omni.platforms.npu.models import qwen3_tts

    incremental = Mock()
    decoder = SimpleNamespace(_npu_conv_graphs=None, _npu_incremental=incremental)
    model = SimpleNamespace(decoder=decoder)

    def stop_at_loader(self, weights):
        incremental.close.assert_called_once_with()
        assert self.decoder._npu_incremental is None
        raise RuntimeError("stop after verifying pre-load cleanup")

    monkeypatch.setattr(qwen3_tts, "_original_load_weights", stop_at_loader)
    with pytest.raises(RuntimeError, match="stop after verifying"):
        qwen3_tts._patched_load_weights(model, [])
