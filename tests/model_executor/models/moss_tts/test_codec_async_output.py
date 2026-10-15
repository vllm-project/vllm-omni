# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Waveform ownership across graph replays and asynchronous output copies."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import MossTTSCodecDecoder
from vllm_omni.model_executor.output_snapshot import PackedOutputSnapshot

pytestmark = [pytest.mark.core_model]


class CausalCodec(nn.Module):
    downsample_rate = 1

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.channels = channels
        self.config = SimpleNamespace(codebook_size=1024)

    def initialize_decoder_state_pool(self, capacity, scratch):
        self.state = torch.zeros(capacity + scratch, device=self.weight.device)

    def reset_decoder_state_slots(self, slots):
        self.state[slots] = 0

    def decode_streaming_batch(self, codes, lengths, slots, valid_rows):
        audio = codes.sum(0).float().cumsum(-1) + self.state[slots, None]
        self.state[slots] = audio[:, -1]
        channels = torch.arange(1, self.channels + 1, device=audio.device)
        return SimpleNamespace(audio=audio[:, None] * channels[None, :, None], audio_lengths=lengths)


class ReplayGraph:
    """A real CUDA graph with deliberately reused output storage."""

    batch_sizes = [1]

    def __init__(self, codec):
        self.codes = torch.ones((2, 1, 3), device="cuda", dtype=torch.long)
        self.slots = torch.zeros(1, device="cuda", dtype=torch.long)
        self.lengths = torch.full((1,), 3, device="cuda", dtype=torch.long)
        self.valid = torch.ones(1, device="cuda", dtype=torch.bool)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                codec.decode_streaming_batch(self.codes, self.lengths, self.slots, self.valid)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = codec.decode_streaming_batch(self.codes, self.lengths, self.slots, self.valid)
        codec.state.zero_()

    def decode(self, codes, slots, *, allow_frame_padding):
        if codes.shape[1:] != (1, 3):
            return None
        self.codes.copy_(codes)
        self.slots.copy_(slots)
        self.graph.replay()
        return self.output.audio, self.output.audio_lengths, 1


def make_decoder(*, v2=True, channels=2, device="cpu", model_type="moss_tts_local", streaming=True):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(n_vq=2, model_type=model_type),
            async_chunk=streaming,
            use_v2_model_runner=v2,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    decoder = MossTTSCodecDecoder(vllm_config=config)
    decoder._codec = CausalCodec(channels).to(device)
    decoder._n_channels = channels
    decoder._stream_max_step_frames = 3
    return decoder


@pytest.mark.cpu
@pytest.mark.parametrize("ramp,frames", [(None, [1, 15]), ([1, 4, 8, 15], [1, 4, 8, 15]), ([1, 4, 4, 15], [1, 4, 15])])
def test_ramp_chunks_are_captured_at_exact_frame_counts(ramp, frames):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(n_vq=2, model_type="moss_tts_local"),
            async_chunk=True,
            use_v2_model_runner=True,
            stage_connector_config={
                "extra": {
                    "initial_codec_chunk_frames": 1,
                    "codec_chunk_frames": 15,
                    "codec_chunk_ramp": ramp,
                }
            },
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    decoder = MossTTSCodecDecoder(vllm_config=config)
    assert decoder._streaming_graph_frame_sizes == frames
    assert decoder._stream_max_step_frames == 15


@pytest.mark.cpu
@pytest.mark.parametrize(
    "v2,streaming,model_type",
    [
        (False, True, "moss_tts_local"),
        (True, False, "moss_tts_local"),
        (True, True, "moss_tts_realtime"),
        (True, True, "moss_tts_local"),
    ],
)
def test_gpu_output_requires_local_streaming_mrv2_and_cuda(v2, streaming, model_type):
    decoder = make_decoder(v2=v2, streaming=streaming, model_type=model_type)
    assert decoder._gpu_stream_output == (v2 and streaming and model_type == "moss_tts_local")
    session = decoder._ensure_stream_session()
    slot = session.acquire()
    assert session.step({slot: torch.ones(2, 3)})[slot].device.type == "cpu"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("graph", [False, True])
@torch.no_grad()
def test_replays_terminal_tails_and_slot_reuse_preserve_pending_waveforms(v2, channels, graph):
    from vllm_omni.outputs import OmniModelRunnerOutput
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationAsyncOutput,
        OmniGenerationModelRunner,
        _contains_cuda_tensor,
    )

    decoder = make_decoder(v2=v2, channels=channels, device="cuda")
    # Real Local-v1.5 weights replace the class's 24 kHz default at load time.
    decoder._sr_tensor = torch.tensor(48_000, dtype=torch.int32)
    session = decoder._ensure_stream_session()
    if graph:
        session._cudagraph_wrapper = ReplayGraph(decoder._codec)
    copy_stream = torch.cuda.Stream()
    pending = []
    # Do not materialize any output until later forwards have replayed the
    # graph, reset terminal/cancelled state, and reused those state slots.
    for round_index in range(4):
        value = round_index + 1
        codes = torch.cat(
            [
                torch.full((14,), value, device="cuda", dtype=torch.long),
                torch.full((6,), value + 10, device="cuda", dtype=torch.long),
            ]
        )
        info = [{"request_id": rid, "meta": {"finished": rid == "tail"}} for rid in ["tail", "empty", "live"]]
        output = decoder.forward(codes, runtime_additional_information=info, seq_token_counts=[14, 0, 6])
        # The runner may overwrite its input buffer before codec completion.
        codes.fill_(-100)
        payload = output.multimodal_outputs
        assert _contains_cuda_tensor(payload) == v2
        assert isinstance(payload, PackedOutputSnapshot) == v2
        if v2:
            # Exactly one waveform slab, regardless of request/chunk count.
            assert sum(slab.is_cuda for _, slab in payload._groups) == 1
            result = OmniModelRunnerOutput(
                req_ids=["tail", "empty", "live"],
                req_id_to_index={"tail": 0, "empty": 1, "live": 2},
                sampled_token_ids=[[], [], []],
            )
            pending.append(
                OmniGenerationAsyncOutput(
                    model_runner_output=result,
                    multimodal_outputs=payload,
                    num_reqs=3,
                    main_stream=torch.cuda.current_stream(),
                    copy_stream=copy_stream,
                )
            )
        else:
            pending.append(OmniGenerationModelRunner._build_pooler_output(output, 3))
        decoder.on_requests_finished(["live"])
        assert decoder._stream_req_slots == {}
        assert not session._leased_slots

    for round_index, item in enumerate(pending):
        rows = item.get_output().multimodal_outputs if v2 else item
        for index, (frames, value) in enumerate([(7, round_index + 1), (0, 0), (3, round_index + 11)]):
            expected = 2 * value * torch.arange(1, frames + 1, dtype=torch.float32)
            if channels > 1:
                expected = expected[None] * torch.arange(1, channels + 1)[:, None]
            torch.testing.assert_close(rows[index]["model_outputs"], expected, rtol=0, atol=0)
            assert rows[index]["sr"].item() == 48_000


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.no_grad()
def test_owned_input_copy_clamps_entire_ragged_batch():
    from vllm_omni.worker_v2.omni_ar_model_runner import _async_copy_mm

    decoder = make_decoder(device="cuda", channels=2)
    codes = torch.tensor([-1, 1024, 1, -10, 2, 1025, 5000, -2], device="cuda")
    output = decoder.forward(
        codes,
        runtime_additional_information=[{"request_id": name} for name in ("a", "empty", "b")],
        seq_token_counts=[6, 0, 2],
    )
    codes.fill_(-100)
    cpu = _async_copy_mm(output.multimodal_outputs, 0, pin_memory=True)
    torch.accelerator.synchronize()
    expected = torch.tensor([[0.0, 1025.0, 2049.0], [0.0, 2050.0, 4098.0]])
    torch.testing.assert_close(cpu["model_outputs"][0], expected, rtol=0, atol=0)
    assert cpu["model_outputs"][1].numel() == 0
    torch.testing.assert_close(cpu["model_outputs"][2], torch.tensor([[1023.0], [2046.0]]), rtol=0, atol=0)
    decoder.on_requests_finished(["a", "b"])


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.no_grad()
def test_pinned_metadata_reuse_waits_for_dma_and_preserves_mixed_terminals():
    decoder = make_decoder(device="cuda")
    session = decoder._ensure_stream_session()
    # Force reuse every call, with an outstanding transfer on the first call.
    # This delay tests host-buffer ownership, never performance or overlap.
    session._metadata_ring = session._metadata_ring[:1]
    slots = [session.acquire() for _ in range(3)]
    pending = []
    for step in range(8):
        order = slots[step % 3 :] + slots[: step % 3]
        terminal = {order[1]}
        torch.cuda._sleep(2_000_000)
        ids, tails = session._stage_slot_ids(order, terminal)
        pending.append((ids.clone(), tails.clone(), order, list(terminal)))
    session.close()
    for ids, tails, expected_ids, expected_tails in pending:
        assert ids.cpu().tolist() == expected_ids
        assert tails.cpu().tolist() == expected_tails


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.no_grad()
def test_mixed_terminal_reset_does_not_reset_live_slot():
    decoder = make_decoder(device="cuda", channels=1)
    session = decoder._ensure_stream_session()
    first, second = session.acquire(), session.acquire()
    codes = torch.ones(2, 3, device="cuda")
    initial = session.step({first: codes, second: 2 * codes}, terminal_slots={first})
    following = session.step({second: codes, first: codes})
    torch.testing.assert_close(initial[first].cpu(), torch.tensor([[2.0, 4.0, 6.0]]), rtol=0, atol=0)
    torch.testing.assert_close(following[first].cpu(), torch.tensor([[2.0, 4.0, 6.0]]), rtol=0, atol=0)
    torch.testing.assert_close(following[second].cpu(), torch.tensor([[14.0, 16.0, 18.0]]), rtol=0, atol=0)
    session.close()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("explicit_copy_stream", [False, True])
@torch.no_grad()
def test_output_copy_waits_for_private_codec_stream(explicit_copy_stream):
    from vllm_omni.worker_v2.omni_ar_model_runner import _async_copy_mm

    decoder = make_decoder(device="cuda", channels=1)
    decoder._codec_stream = torch.cuda.Stream()
    with torch.cuda.stream(decoder._codec_stream):
        torch.cuda._sleep(20_000_000)
    codes = torch.ones(6, device="cuda", dtype=torch.long)
    output = decoder.forward(
        codes,
        runtime_additional_information=[{"request_id": "live"}],
        seq_token_counts=[6],
    )
    codes.fill_(-100)
    payload = output.multimodal_outputs
    assert isinstance(payload, PackedOutputSnapshot)
    assert payload.producer_event is not None
    # No explicit consumer wait here: the shared copier must honor readiness.
    copy_stream = torch.cuda.Stream()
    with torch.cuda.stream(copy_stream):
        cpu = _async_copy_mm(payload, 0, copy_stream=copy_stream if explicit_copy_stream else None, pin_memory=True)
    decoder.on_requests_finished(["live"])
    copy_stream.synchronize()
    torch.testing.assert_close(cpu["model_outputs"][0], torch.tensor([2.0, 4.0, 6.0]), rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("channels", [1, 2])
@torch.no_grad()
def test_short_terminal_row_crops_each_channel_into_output_slab(channels):
    decoder = make_decoder(device="cuda", channels=channels)
    session = decoder._ensure_stream_session()
    short, live = session.acquire(), session.acquire()
    codes = {short: torch.ones(2, 2, device="cuda"), live: 3 * torch.ones(2, 5, device="cuda")}
    slab = torch.empty(channels * 7, device="cuda")
    buffers = {
        short: slab[: channels * 2].view(channels, 2),
        live: slab[channels * 2 :].view(channels, 5),
    }
    result = session.step(codes, terminal_slots={short}, output_buffers=buffers)
    for slot, frames, value in [(short, 2, 1), (live, 5, 3)]:
        expected = 2 * value * torch.arange(1, frames + 1, dtype=torch.float32)[None]
        expected = expected * torch.arange(1, channels + 1)[:, None]
        assert result[slot].data_ptr() == buffers[slot].data_ptr()
        torch.testing.assert_close(result[slot].cpu(), expected, rtol=0, atol=0)
    session.close()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("layout", ["adjacent", "strided", "gapped"])
@torch.no_grad()
def test_batched_output_copy_preserves_rows_and_slab_ownership(channels, layout):
    decoder = make_decoder(device="cuda", channels=channels)
    session = decoder._ensure_stream_session()
    slots = [session.acquire() for _ in range(4)]
    slab = torch.full((6 if layout == "gapped" else 4, channels, 6), -1.0, device="cuda")
    if layout == "gapped":
        buffers = {slot: slab[row] for row, slot in zip([0, 1, 3, 4], slots, strict=True)}
        frames = 6
    elif layout == "adjacent":
        buffers = {slot: slab.reshape(4, channels * 6)[row].view(channels, 6) for row, slot in enumerate(slots)}
        frames = 6
    else:
        # Interleaved with other requests' output: the short views must not
        # overwrite the untouched part of another row/channel.
        buffers = {slot: slab[row, :, :3] for row, slot in enumerate(slots)}
        frames = 3
    codes = {slot: torch.full((2, frames), row + 1, device="cuda") for row, slot in enumerate(slots)}
    result = session.step(codes, output_buffers=buffers)
    for row, slot in enumerate(slots):
        expected = 2 * (row + 1) * torch.arange(1, frames + 1, dtype=torch.float32)[None]
        expected = expected * torch.arange(1, channels + 1)[:, None]
        assert result[slot].data_ptr() == buffers[slot].data_ptr()
        torch.testing.assert_close(result[slot].cpu(), expected, rtol=0, atol=0)
    if layout == "strided":
        assert torch.all(slab[:, :, 3:] == -1)
    elif layout == "gapped":
        assert torch.all(slab[[2, 5]] == -1)
    session.close()
