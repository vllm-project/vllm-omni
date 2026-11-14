# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise incremental codec tail fallback and request slot reuse on NPU."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    EuclideanCodebook,
)

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
        if isinstance(module, EuclideanCodebook):
            module.embedding_sum.normal_()
    decoder.precompute_snake_caches()
    arena = NPUIncrementalCodec(decoder, slots=2, max_batch=2, chunk_frames=4)
    codes = torch.randint(0, 32, (2, 2, 7), device="npu")
    caches: list[dict[str, int]] = [{}, {}]

    def unexpected_capture(self):
        pytest.fail("a live partial chunk must not capture a new graph")

    monkeypatch.setattr(_ShapeGraph, "capture", unexpected_capture)
    try:
        # Both requests span a full chunk and an eager tail.
        first = arena.decode_requests(codes, [7, 7], caches)
        retained = [wave.clone() for wave in first]
        assert set(arena.graphs) == {(batch, length) for batch in (1, 2) for length in (1, 4)}
        assert len({graph.graph.pool() for graph in arena.graphs.values()}) == len(arena.graphs)
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
    decoder = SimpleNamespace(_npu_incremental=incremental)
    model = SimpleNamespace(decoder=decoder)

    def stop_at_loader(self, weights):
        incremental.close.assert_called_once_with()
        assert self.decoder._npu_incremental is None
        raise RuntimeError("stop after verifying pre-load cleanup")

    monkeypatch.setattr(qwen3_tts, "_original_load_weights", stop_at_loader)
    with pytest.raises(RuntimeError, match="stop after verifying"):
        qwen3_tts._patched_load_weights(model, [])


@torch.inference_mode()
def test_incremental_attention_does_not_change_stateless_backend(monkeypatch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.platforms.npu.layers.incremental_codec import NPUIncrementalCodec
    from vllm_omni.platforms.npu.models.qwen3_tts import _prepare_npu_code2wav_runtime

    _prepare_npu_code2wav_runtime()
    decoder = _make_decoder().to(device="npu", dtype=torch.bfloat16).eval()
    decoder.pre_transformer.config._attn_implementation = "sdpa"
    for module in decoder.modules():
        if isinstance(module, EuclideanCodebook):
            module.embedding_sum.normal_()
    decoder.precompute_snake_caches()
    codes = torch.randint(0, 32, (1, 2, 7), device="npu")
    expected = decoder(codes)
    attention = decoder.pre_transformer.layers[0].self_attn
    forward = Mock(wraps=attention.forward)
    monkeypatch.setattr(attention, "forward", forward)
    arena = NPUIncrementalCodec(decoder, slots=1, max_batch=1, chunk_frames=4)
    try:
        assert decoder.pre_transformer.config._attn_implementation == "sdpa"
        caches: list[dict[str, int]] = [{}]
        # Compare both a captured full chunk and an eager tail against the
        # same stateless codes, with BF16 streaming-codec tolerances.
        chunks = [
            arena.decode_requests(codes[:, :, :4], [4], caches)[0],
            arena.decode_requests(codes[:, :, 4:], [3], caches)[0],
        ]
        torch.testing.assert_close(torch.cat(chunks), expected[0, 0], atol=0.02, rtol=0.02)
        assert forward.call_args_list
        assert all(call.kwargs.get("force_eager_attention") is True for call in forward.call_args_list)
        forward.reset_mock()
        torch.testing.assert_close(decoder(codes), expected, atol=0, rtol=0)
        assert all(not call.kwargs.get("force_eager_attention", False) for call in forward.call_args_list)
        assert decoder.pre_transformer.config._attn_implementation == "sdpa"
    finally:
        arena.close()


@pytest.mark.parametrize("source", ["implicit", "additional", "connector"])
def test_loader_defaults_and_explicit_opt_out(monkeypatch, source):
    pytest.importorskip("torch_npu")
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.platforms.npu.layers import incremental_codec
    from vllm_omni.platforms.npu.models import qwen3_tts

    decoder = _make_decoder()
    model = SimpleNamespace(
        decoder=decoder,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                stage_connector_config={"extra": {"npu_incremental_codec": False} if source == "connector" else {}}
            ),
            # Connector opt-outs override explicit stage opt-ins.
            additional_config={"npu_incremental_codec": source == "connector"} if source != "implicit" else {},
            device_config=SimpleNamespace(device="cpu"),
            scheduler_config=SimpleNamespace(max_num_seqs=4),
        ),
    )
    monkeypatch.setattr(qwen3_tts, "_original_load_weights", lambda *_: {"loaded"})
    monkeypatch.setattr(qwen3_tts, "_prepare_npu_decoder_weights", lambda _: None)
    incremental = Mock()
    monkeypatch.setattr(incremental_codec, "NPUIncrementalCodec", incremental)
    assert qwen3_tts._patched_load_weights(model, []) == {"loaded"}
    if source == "implicit":
        incremental.assert_called_once_with(decoder, slots=4, max_batch=1, chunk_frames=25)
        assert decoder._npu_incremental is incremental.return_value
    else:
        incremental.assert_not_called()
        assert decoder._npu_incremental is None


@torch.inference_mode()
@pytest.mark.parametrize("max_batch", [1, 2])
def test_all_tail_lengths_match_eager_state(monkeypatch, max_batch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.platforms.npu.layers.incremental_codec import NPUIncrementalCodec, _ShapeGraph
    from vllm_omni.platforms.npu.models.qwen3_tts import _prepare_npu_code2wav_runtime

    _prepare_npu_code2wav_runtime()
    decoder = _make_decoder().to(device="npu", dtype=torch.bfloat16).eval()
    for module in decoder.modules():
        if isinstance(module, EuclideanCodebook):
            module.embedding_sum.normal_()
    decoder.precompute_snake_caches()
    arena = NPUIncrementalCodec(decoder, slots=2, max_batch=max_batch, chunk_frames=25)
    with monkeypatch.context() as patch:
        patch.setattr(NPUIncrementalCodec, "prewarm", lambda self: None)
        reference = NPUIncrementalCodec(decoder, slots=2, max_batch=2, chunk_frames=25)
    caches: list[dict[str, int]] = [{}, {}]
    ref_slots = [reference.alloc(), reference.alloc()]
    indices = torch.tensor(ref_slots, device="npu")
    run = _ShapeGraph.run

    def checked_run(graph, indices, codes, *, replay):
        if graph.arena is arena:
            assert replay
            assert (graph.graph is not None) == (graph.length in (1, 25))
        return run(graph, indices, codes, replay=replay)

    monkeypatch.setattr(_ShapeGraph, "run", checked_run)
    try:
        for length in range(1, 26):
            codes = torch.randint(0, 32, (2, 2, length), device="npu")
            expected = _ShapeGraph(reference, 2, length).run(indices, codes, replay=False)
            actual = arena.decode_requests(codes, [length, length], caches)
            for row in range(2):
                torch.testing.assert_close(actual[row], expected[row, 0], atol=0.015, rtol=0.03)
            torch.testing.assert_close(arena.frames, reference.frames)
    finally:
        arena.close()
        reference.close()
