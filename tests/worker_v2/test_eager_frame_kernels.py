# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Settled-frame and codec-input kernels equal the torch indexing they replace."""

import pytest
import torch

from tests.helpers.device import DEVICE_MODULE, DEVICE_TYPE
from tests.helpers.mark import hardware_test

pytestmark = pytest.mark.core_model


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("step_dtype", [torch.float32, torch.bfloat16])
def test_settled_frames_match_torch(step_dtype):
    from vllm_omni.worker_v2.model_states.eager_mtp_kernels import settled_frames

    device = torch.device(DEVICE_TYPE, 0)
    H = 2048
    eager = torch.randn(16, H, device=device).to(torch.bfloat16)
    step = torch.randn(1, H, device=device, dtype=step_dtype)
    embeds = torch.randn(40, H, device=device, dtype=torch.bfloat16)
    slots = torch.tensor([3, 0, 15, 7], device=device)
    rows = torch.tensor([5, 9, 0, 39], device=device)
    expected = embeds.clone()
    expected.index_copy_(0, rows, (eager.index_select(0, slots) + step.to(eager.dtype)).to(expected.dtype))
    settled_frames(torch.stack((slots, rows)).contiguous(), 4, eager, step, embeds)
    torch.testing.assert_close(embeds, expected, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("n,size,frames", [(3, 8, 1), (8, 8, 1), (2, 4, 25)])
def test_codec_graph_inputs_match_the_copies(n, size, frames):
    from vllm.triton_utils import triton

    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.streaming_decoder import _graph_inputs_kernel

    device = torch.device(DEVICE_TYPE, 0)
    nq, scratch = 16, 99
    codes = torch.randint(0, 2048, (n, frames, nq), device=device, dtype=torch.int64)
    meta = torch.randint(0, 64, (5 * n,), device=device, dtype=torch.int64)
    slots, pos = meta[n : 2 * n], meta[4 * n :]
    statics = [
        torch.full((size, frames, nq), -1, device=device, dtype=torch.int32),
        torch.full((size,), -1, device=device, dtype=torch.int32),
        torch.full((size,), -1, device=device, dtype=torch.int32),
    ]
    expected = [t.clone() for t in statics]
    expected[0][:n].copy_(codes)
    expected[1][:n].copy_(slots)
    expected[2][:n].copy_(pos)
    expected[1][n:].fill_(scratch)
    expected[2][n:].zero_()
    width = frames * nq
    _graph_inputs_kernel[(size,)](
        codes, slots, pos, *statics, n, scratch, W=width, WP=triton.next_power_of_2(width)
    )  # fmt: skip
    for got, want in zip(statics, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("n,size", [(3, 8), (8, 8)])
def test_eager_post_fills_codec_graph_inputs_and_scratch(n, size):
    from vllm_omni.worker_v2.model_states.eager_mtp_kernels import eager_post

    device = torch.device(DEVICE_TYPE, 0)
    hidden, nq, scratch = 128, 16, 15
    slots = torch.arange(n, device=device).flip(0)
    pos = torch.arange(n, device=device) + 7
    meta = torch.stack((slots, slots, slots * 0, slots * 0, pos)).long()
    last = torch.arange(n, device=device)
    layer0 = torch.arange(n, device=device)
    frames = torch.randn(n, hidden, device=device, dtype=torch.bfloat16)
    codes = torch.randint(0, 2048, (n, nq), device=device)
    eager = torch.empty(n, hidden, device=device, dtype=frames.dtype)
    codes_out = torch.empty_like(codes)
    inputs = torch.zeros(n, device=device, dtype=torch.long)
    valid_out = torch.empty(n, device=device, dtype=torch.bool)
    static_codes = torch.full((size, 1, nq), -1, device=device, dtype=torch.int32)
    static_slots = torch.full((size,), -1, device=device, dtype=torch.int32)
    static_pos = torch.full_like(static_slots, -1)
    valid = eager_post(
        meta,
        n,
        last,
        layer0,
        frames,
        eager,
        codes,
        codes_out,
        inputs,
        valid_out,
        None,
        None,
        2048,
        codes32=static_codes,
        codec_slots=static_slots,
        codec_pos=static_pos,
        scratch=scratch,
    )
    torch.testing.assert_close(static_codes[:n, 0].long(), codes)
    torch.testing.assert_close(static_slots[:n].long(), slots)
    torch.testing.assert_close(static_pos[:n].long(), pos)
    assert (static_slots[n:] == scratch).all() and (static_pos[n:] == 0).all()
    assert (static_codes[n:] == 0).all()
    torch.testing.assert_close(eager.index_select(0, slots), frames)
    torch.testing.assert_close(codes_out, codes)
    assert valid.all() and valid_out.all()


@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards=1)
@pytest.mark.skipif(
    DEVICE_TYPE == "xpu",
    reason="VLLMZ-3554: the packed metadata holds device pointers as signed int64, and XPU USM addresses do not fit",
)
def test_history_scatter_preserves_independent_storage_and_offsets():
    import numpy as np

    from vllm_omni.utils.device_copy import DeviceStager
    from vllm_omni.worker_v2.model_states.eager_mtp_kernels import record_history

    device = torch.device(DEVICE_TYPE, 0)
    inputs = torch.randn(5, 128, device=device, dtype=torch.bfloat16)
    histories = [torch.full((9, 128), -1, device=device, dtype=inputs.dtype) for _ in range(3)]
    starts, offsets, counts = [2, 0, 5], [4, 1, 0], [3, 2, 0]
    expected = [t.clone() for t in histories]
    for t, start, offset, count in zip(expected, starts, offsets, counts):
        t[offset : offset + count].copy_(inputs[start : start + count])
    meta = np.array(([t.data_ptr() for t in histories], starts, offsets, counts), dtype=np.int64)
    meta[3] = np.cumsum(counts) - counts
    packed = np.concatenate((meta.reshape(-1), np.repeat(np.arange(3), counts)))
    stager = DeviceStager()
    record_history(stager(packed, device), inputs, 3, sum(counts))
    for got, want in zip(histories, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards=1)
@pytest.mark.skipif(DEVICE_TYPE == "xpu", reason="VLLMZ-3554: no graph-pool handle on torch.xpu")
@torch.inference_mode()
def test_codec_direct_inputs_wait_for_slow_side_stream():
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.streaming_decoder import StreamingDecodeGraphs
    from vllm_omni.worker_v2.model_states.mtp_model_state import FrameBatch, MTPState

    device = torch.device(DEVICE_TYPE, 0)
    hidden, n = 128, 2

    class Decoder:
        nq, scratch_slot, spf = 2, 3, 4

        def __init__(self):
            self.device = device

        def __call__(self, codes, slots, pos, *, initial=False):
            return codes[:, 0, :1].float().repeat(1, 4)

    decoder = Decoder()
    graphs = StreamingDecodeGraphs(decoder, [2])
    model = SimpleNamespace(
        stream_decoder=decoder,
        stream_graphs=graphs,
        stream_chunk_frames=25,
        _codebook_vocab_size=2048,
        get_stream_ref_context=lambda _buf: None,
        prime_stream_decoder=lambda _primes: None,
    )
    owner = SimpleNamespace(model=model, intermediate_buffer=SimpleNamespace(buffers=[{}, {}]))
    owner._embed_dim = 0
    owner.device = torch.device("cpu")
    owner.model.has_talker_mtp = False
    state = MTPState(owner)
    for name in ("_mtp_input_ids", "_mtp_input_embeds", "_mtp_hidden", "_mtp_text_step"):
        shape = (n,) if name == "_mtp_input_ids" else (n, hidden)
        setattr(state, name, torch.zeros(shape, device=device, dtype=torch.long if len(shape) == 1 else torch.float32))
    state.embeds = torch.zeros(n, hidden, device=device)
    weight = torch.randn(32, hidden, device=device)
    state._embedding_weight = lambda: weight
    batch = SimpleNamespace(num_reqs=n, query_start_loc=torch.arange(n + 1, device=device))
    mapping = FrameBatch([(0, 1, "a", False), (1, 0, "b", False)])
    outputs = []
    for value in range(10, 14):
        codes = torch.full((n, 2), value, device=device, dtype=torch.long)
        frames = torch.full((n, hidden), value, device=device)
        state._mtp_forward = lambda *args, _codes=codes, _frames=frames, **kwargs: (_frames, _codes)
        with DEVICE_MODULE.stream(state._decode_stream):
            DEVICE_MODULE._sleep(5_000_000)
        out = torch.zeros(n, 4, device=device)
        state._run_mtp_fused(
            batch,
            frames,
            torch.ones(n, 1, device=device, dtype=torch.long),
            {"model_outputs": out},
            mapping,
            torch.zeros(n, device=device, dtype=torch.long),
            torch.empty_like(codes),
            torch.empty(n, device=device, dtype=torch.bool),
            None,
        )
        outputs.append(out)
    state._decode_stream.synchronize()
    for value, out in zip(range(10, 14), outputs):
        torch.testing.assert_close(out.cpu(), torch.full((n, 4), float(value)))


@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards=1)
@torch.inference_mode()
def test_batched_history_recording_preserves_ragged_reordered_requests():
    from types import SimpleNamespace

    import numpy as np

    from vllm_omni.worker_v2.model_states.mtp_model_state import MTPState

    n, hidden = 32, 128
    owner = SimpleNamespace(
        model=SimpleNamespace(stream_decoder=object()),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=16)),
    )
    owner._embed_dim = 0
    owner.device = torch.device("cpu")
    owner.model.has_talker_mtp = False
    state = MTPState(owner)
    counts = np.ones(n, dtype=np.int64)
    counts[:2] = [3, 0]
    starts = np.concatenate(([0], counts.cumsum()))
    expected: dict[str, list[torch.Tensor]] = {str(i): [] for i in range(n)}
    for reverse in (False, True):
        ids = [str(i) for i in (range(n - 1, -1, -1) if reverse else range(n))]
        offsets = np.array([sum(t.shape[0] for t in expected[rid]) for rid in ids])
        batch = SimpleNamespace(num_reqs=n, req_ids=ids, query_start_loc_np=starts, num_computed_tokens_np=offsets)
        inputs = torch.randn(int(counts.sum()), hidden, device=DEVICE_TYPE, dtype=torch.bfloat16)
        state.record_inputs(batch, inputs)
        for row, rid in enumerate(ids):
            expected[rid].append(inputs[starts[row] : starts[row + 1]].clone())
    for rid, pieces in expected.items():
        want = torch.cat(pieces)
        history = state._talker_inputs[rid]
        assert history.length == want.shape[0]
        torch.testing.assert_close(history.embeds[: history.length], want, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_eager_mtp_kernels_compile_once_for_every_batch_size(monkeypatch):
    # Batches of one or of a multiple of 16 rows, and a first_audio view that
    # starts n bytes into a shared buffer, must reuse the kernels compiled for
    # the first batch: a compile mid-serving stalls every stream.
    from triton import knobs

    from vllm_omni.worker_v2.model_states import eager_mtp_kernels as kernels
    from vllm_omni.worker_v2.model_states.eager_mtp_kernels import (
        eager_post,
        eager_pre,
        record_history,
        settled_frames,
    )

    # Earlier tests may have compiled kernels for other batch sizes already.
    for kernel in (kernels._pre_kernel, kernels._post_kernel, kernels._settled_kernel, kernels._history_kernel):
        kernel.device_caches.clear()
    device = torch.device(DEVICE_TYPE, 0)
    # Request slots cover the largest batch below (33 rows).
    hidden, nq, vocab, slots_total = 128, 16, 64, 64

    def step(n):
        meta = torch.stack([torch.arange(n, device=device)] * 2 + [torch.zeros(n, device=device)] * 3).long()
        qsl = torch.arange(n + 1, device=device, dtype=torch.int32)
        sampled = torch.randint(0, vocab, (n, 1), device=device)
        text_hidden = torch.randn(n, hidden, device=device, dtype=torch.bfloat16)
        emb_w = torch.randn(vocab, hidden, device=device, dtype=torch.bfloat16)
        buffers = [torch.empty(n, hidden, device=device, dtype=torch.bfloat16) for _ in range(3)]
        last, layer0 = eager_pre(meta, n, qsl, sampled, text_hidden, emb_w, sampled.new_empty(n), *buffers)
        eager = torch.empty(slots_total, hidden, device=device, dtype=torch.bfloat16)
        codes = torch.randint(0, vocab, (n, nq), device=device)
        shared = torch.zeros(2 * n, device=device, dtype=torch.int8)
        eager_post(
            meta, n, last, layer0, buffers[0], eager, codes, torch.empty_like(codes),
            torch.zeros(n, device=device, dtype=torch.long), shared[:n],
            shared[n:], torch.ones(slots_total, device=device, dtype=torch.bool), vocab,
        )  # fmt: skip
        settled_frames(meta[:2].contiguous(), n, eager, torch.randn(1, hidden, device=device), text_hidden)
        # One input row per request into its own history; the fused profile packs from 32 requests.
        history = torch.empty(n, 2, hidden, device=device, dtype=torch.bfloat16)
        rows = torch.arange(n, device=device)
        addresses = torch.full((n,), history.data_ptr(), device=device) + rows * history[0].nbytes
        record_history(torch.cat((addresses, rows, rows * 0, rows, rows)), text_hidden, n, n)

    step(3)
    compiles = []
    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", lambda **kw: compiles.append(kw["fn"].name))
    for n in (1, 16, 5, 32, 33):
        step(n)
    assert compiles == []
