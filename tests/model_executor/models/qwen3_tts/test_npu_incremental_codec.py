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
        # Both requests span a full chunk and an eager tail.
        first = arena.decode_requests(codes, [7, 7], caches)
        retained = [wave.clone() for wave in first]
        assert set(arena.graphs) == {(batch, length, 0) for batch in (1, 2) for length in (1, 4)}
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


@torch.inference_mode()
def test_dual_stream_matches_serial_state_and_owned_audio(monkeypatch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.platforms.npu.layers.incremental_codec import NPUIncrementalCodec, _ShapeGraph
    from vllm_omni.platforms.npu.models.qwen3_tts import _prepare_npu_code2wav_runtime

    _prepare_npu_code2wav_runtime()
    torch.manual_seed(318)
    decoder = _make_decoder().to(device="npu", dtype=torch.bfloat16).eval()
    for module in decoder.modules():
        if hasattr(module, "precompute_embedding"):
            module.embedding_sum.normal_()
            module.precompute_embedding()
    decoder.precompute_snake_caches()
    serial = NPUIncrementalCodec(decoder, slots=6, max_batch=2, chunk_frames=4)
    parallel = NPUIncrementalCodec(decoder, slots=6, max_batch=2, chunk_frames=4, streams=2)
    caches: list[list[dict[str, int]]] = [[{} for _ in range(5)] for _ in range(2)]
    retained = []

    def no_capture(self):
        pytest.fail("live decode must not capture")

    monkeypatch.setattr(_ShapeGraph, "capture", no_capture)
    try:
        assert len(parallel.graphs) == 2 * len(serial.graphs)
        pools = [{graph.graph.pool() for key, graph in parallel.graphs.items() if key[2] == lane} for lane in (0, 1)]
        assert len(pools[0]) == len(pools[1]) == 1
        assert pools[0].isdisjoint(pools[1])
        # More groups than lanes, ragged tails, zero-length input, and
        # lane -> owner -> lane reuse of the same graph storage.
        for lengths in ([7, 7, 3, 4, 1], [1], [4, 3, 7, 0, 4]):
            codes = torch.randint(0, 32, (len(lengths), 2, max(lengths)), device="npu")
            expected = serial.decode_requests(codes, lengths, caches[0][: len(lengths)])
            actual = parallel.decode_requests(codes, lengths, caches[1][: len(lengths)])
            for out, ref in zip(actual, expected, strict=True):
                torch.testing.assert_close(out, ref, atol=0, rtol=0)
                retained.append((out, out.clone()))
            for out, ref in zip(parallel.buffers, serial.buffers, strict=True):
                torch.testing.assert_close(out, ref, atol=0, rtol=0)
            assert caches[0] == caches[1]
        # Exercise the outer-capture eligibility branch without nesting a
        # real graph: both inner replay and side streams must be bypassed.
        codes = torch.randint(0, 32, (5, 2, 4), device="npu")
        expected = serial.decode_requests(codes, [4] * 5, caches[0])
        with monkeypatch.context() as patch:
            patch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
            patch.setattr(parallel, "streams", [Mock(), Mock()])
            actual = parallel.decode_requests(codes, [4] * 5, caches[1])
            for stream in parallel.streams:
                stream.wait_stream.assert_not_called()
        for out, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
        for arena, requests in zip((serial, parallel), caches, strict=True):
            for cache in requests:
                arena.release(cache["npu_slot"])
        codes = torch.randint(0, 32, (5, 2, 4), device="npu")
        caches = [[{} for _ in range(5)] for _ in range(2)]
        expected = serial.decode_requests(codes, [4] * 5, caches[0])
        actual = parallel.decode_requests(codes, [4] * 5, caches[1])
        for out, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
        for out, snapshot in retained:
            torch.testing.assert_close(out, snapshot, atol=0, rtol=0)

        # Fail after both lanes have submitted a group. The owner must still
        # fence their writes before inspecting state or recycling requests.
        decode = parallel._decode_same_length
        calls = 0

        def fail_third(slots, inputs, *, lane=0):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("injected group failure")
            return decode(slots, inputs, lane=lane)

        monkeypatch.setattr(parallel, "_decode_same_length", fail_third)
        with pytest.raises(RuntimeError, match="injected group failure"):
            parallel.decode_requests(codes, [4] * 5, caches[1])
        serial.decode_requests(codes[:4], [4] * 4, caches[0][:4])
        for out, ref in zip(parallel.buffers, serial.buffers, strict=True):
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
        for arena, requests in zip((serial, parallel), caches, strict=True):
            for cache in requests:
                arena.release(cache["npu_slot"])
        for out, ref in zip(parallel.buffers, serial.buffers, strict=True):
            torch.testing.assert_close(out, ref, atol=0, rtol=0)
    finally:
        parallel.close()
        serial.close()


@torch.inference_mode()
def test_selected_row_stores_preserve_unrelated_and_strided_state():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from vllm_omni.platforms.npu.layers.codec_row_scatter import scatter_rows

    for shape, dtype in (((8,), torch.long), ((8, 3, 7), torch.bfloat16), ((8, 2, 7, 16), torch.bfloat16)):
        target = torch.randn(shape, device="npu").to(dtype)
        indices = torch.tensor([6, 1, 4], device="npu")
        source_shape = (3, *shape[1:])
        if len(shape) == 4:
            source = torch.randn(3, 2, 10, 16, device="npu", dtype=dtype)[:, :, -7:]
            assert not source.is_contiguous()
        else:
            source = torch.randn(source_shape, device="npu").to(dtype)
        expected = target.clone().index_copy_(0, indices, source)
        scatter_rows(target, indices, source)
        torch.testing.assert_close(target, expected, atol=0, rtol=0)


def test_eager_loader_skips_requested_conv_capture(monkeypatch):
    pytest.importorskip("torch_npu")
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.platforms.npu.layers import codec_graph
    from vllm_omni.platforms.npu.models import qwen3_tts

    decoder = _make_decoder()
    model = SimpleNamespace(
        decoder=decoder,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                enforce_eager=True,
                stage_connector_config={
                    "extra": {"npu_conv_graph_shapes": [[1, 1], [2, 37]], "npu_incremental_codec": False},
                },
            ),
            device_config=SimpleNamespace(device="cpu"),
        ),
    )
    monkeypatch.setattr(qwen3_tts, "_original_load_weights", lambda *_: {"loaded"})
    monkeypatch.setattr(qwen3_tts, "_prepare_npu_decoder_weights", lambda _: None)
    capture = Mock(side_effect=AssertionError("enforce_eager must not capture"))
    monkeypatch.setattr(codec_graph, "NPUCodecGraphs", capture)
    assert qwen3_tts._patched_load_weights(model, []) == {"loaded"}
    assert decoder._npu_conv_graphs is None
    capture.assert_not_called()


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
        if hasattr(module, "precompute_embedding"):
            module.embedding_sum.normal_()
            module.precompute_embedding()
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


@pytest.mark.parametrize("has_triton", [False, True])
@pytest.mark.parametrize("source", ["implicit", "additional", "connector", "opt_in"])
def test_loader_defaults_and_explicit_opt_out(monkeypatch, has_triton, source):
    pytest.importorskip("torch_npu")
    import vllm.triton_utils

    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.model_executor.models.common.snake_activation import SnakeBeta
    from vllm_omni.platforms.npu.layers import codec_graph, incremental_codec
    from vllm_omni.platforms.npu.models import qwen3_tts

    decoder = _make_decoder()
    disabled = source in ("additional", "connector")
    explicit = {"npu_fused_snake": False, "npu_incremental_codec": False, "npu_conv_graph_shapes": []}
    model = SimpleNamespace(
        decoder=decoder,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                enforce_eager=False, stage_connector_config={"extra": explicit if source == "connector" else {}}
            ),
            # Explicit connector opt-outs must override stage defaults too.
            additional_config=(
                {"npu_fused_snake": True, "npu_incremental_codec": True}
                if source == "connector"
                else {"npu_fused_snake": True}
                if source == "opt_in"
                else explicit
                if source == "additional"
                else {}
            ),
            device_config=SimpleNamespace(device="cpu"),
            scheduler_config=SimpleNamespace(max_num_seqs=4),
        ),
    )
    monkeypatch.setattr(vllm.triton_utils, "HAS_TRITON", has_triton)
    monkeypatch.setattr(qwen3_tts, "_original_load_weights", lambda *_: {"loaded"})
    monkeypatch.setattr(qwen3_tts, "_prepare_npu_decoder_weights", lambda _: None)
    incremental = Mock()
    conv = Mock(side_effect=AssertionError("incremental or explicit [] must skip stateless capture"))
    monkeypatch.setattr(incremental_codec, "NPUIncrementalCodec", incremental)
    monkeypatch.setattr(codec_graph, "NPUCodecGraphs", conv)
    qwen3_tts._patched_load_weights(model, [])
    if disabled:
        incremental.assert_not_called()
    else:
        assert incremental.call_args.kwargs["streams"] == (2 if has_triton else 1)
        assert decoder._npu_incremental is incremental.return_value
    conv.assert_not_called()
    assert all(
        m.npu_fused is (has_triton and source == "opt_in") for m in decoder.modules() if isinstance(m, SnakeBeta)
    )


@pytest.mark.parametrize("has_triton,incremental", [(False, True), (True, False), (False, False)])
@pytest.mark.parametrize("source", ["additional", "connector"])
def test_explicit_dual_stream_conflict_fails_before_capture(monkeypatch, has_triton, incremental, source):
    pytest.importorskip("torch_npu")
    import vllm.triton_utils

    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.platforms.npu.layers import codec_graph, incremental_codec
    from vllm_omni.platforms.npu.models import qwen3_tts

    explicit = {"npu_codec_streams": 2, "npu_incremental_codec": incremental}
    model = SimpleNamespace(
        decoder=_make_decoder(),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(stage_connector_config={"extra": explicit if source == "connector" else {}}),
            additional_config=explicit if source == "additional" else {},
            device_config=SimpleNamespace(device="cpu"),
            scheduler_config=SimpleNamespace(max_num_seqs=4),
        ),
    )
    monkeypatch.setattr(vllm.triton_utils, "HAS_TRITON", has_triton)
    monkeypatch.setattr(qwen3_tts, "_original_load_weights", lambda *_: {"loaded"})
    monkeypatch.setattr(qwen3_tts, "_prepare_npu_decoder_weights", lambda _: None)
    capture = Mock()
    monkeypatch.setattr(incremental_codec, "NPUIncrementalCodec", capture)
    monkeypatch.setattr(codec_graph, "NPUCodecGraphs", capture)
    with pytest.raises(ValueError, match="npu_codec_streams=2 requires"):
        qwen3_tts._patched_load_weights(model, [])
    capture.assert_not_called()


@torch.inference_mode()
def test_tail_row_scatter_does_not_compile_for_new_strides(monkeypatch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from vllm.triton_utils import triton

    from vllm_omni.platforms.npu.layers.codec_row_scatter import scatter_rows

    target = torch.zeros(8, 3, 7, device="npu", dtype=torch.bfloat16)
    indices = torch.tensor([6, 1, 4], device="npu")
    first = torch.randn(3, 3, 8, device="npu", dtype=torch.bfloat16)[:, :, -7:]
    scatter_rows(target, indices, first)
    torch.npu.synchronize()

    def no_compile(*args, **kwargs):
        pytest.fail("a new tail stride must reuse the warmed scatter kernel")

    monkeypatch.setattr(triton.runtime.jit.JITFunction, "_do_compile", no_compile)
    for length in range(2, 26):
        source = torch.randn(3, 3, 7 + length, device="npu", dtype=torch.bfloat16)[:, :, -7:]
        expected = target.clone().index_copy_(0, indices, source)
        scatter_rows(target, indices, source)
        torch.testing.assert_close(target, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_all_tail_lengths_match_eager_state(monkeypatch):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.platforms.npu.layers.incremental_codec import NPUIncrementalCodec, _ShapeGraph
    from vllm_omni.platforms.npu.models.qwen3_tts import _prepare_npu_code2wav_runtime

    _prepare_npu_code2wav_runtime()
    decoder = _make_decoder().to(device="npu", dtype=torch.bfloat16).eval()
    for module in decoder.modules():
        if hasattr(module, "precompute_embedding"):
            module.embedding_sum.normal_()
            module.precompute_embedding()
    decoder.precompute_snake_caches()
    arena = NPUIncrementalCodec(decoder, slots=2, max_batch=2, chunk_frames=25)
    with monkeypatch.context() as patch:
        patch.setattr(NPUIncrementalCodec, "prewarm", lambda self: None)
        reference = NPUIncrementalCodec(decoder, slots=2, max_batch=2, chunk_frames=25)
    caches = [{}, {}]
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
