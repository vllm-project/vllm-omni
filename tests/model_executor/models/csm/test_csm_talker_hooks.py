# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the CSM stage-0 engine hooks, driven through the real OmniModelState.

Each test replays engine steps (preprocess, forward, sampling, post-sample) the
way the Model Runner V2 does. The backbone and the depth decoder are replaced by
deterministic stand-ins so a test can decide which first codebook each step
samples and when a stop frame appears.
"""

from collections.abc import Iterator
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.sampling_params import SamplingParams

from tests.model_executor.models.csm.test_csm_talker import _build_talker, _config
from vllm_omni.model_executor.models.csm.csm_talker import REFERENCE_AUDIO_KEY, CsmTalkerForConditionalGeneration
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.model_states.mtp_model_state import MTPState
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

CAPACITY = 4
# First codebooks the tests plan; each gets its own hidden direction below.
PLANNED_CODES = (0, 7, 9, 11, 13)


class _Batch:
    """The attributes of one scheduled batch that the model state reads."""

    def __init__(self, slots: list[int], req_ids: list[str], tokens: list[list[int]]) -> None:
        counts = [len(row) for row in tokens]
        starts = np.cumsum([0, *counts])
        self.idx_mapping_np = np.asarray(slots)
        self.idx_mapping = torch.tensor(slots)
        self.num_reqs = len(slots)
        self.req_ids = req_ids
        self.input_ids = torch.tensor([token for row in tokens for token in row])
        self.num_tokens = int(starts[-1])
        self.num_scheduled_tokens = counts
        self.query_start_loc_np = starts[:-1].tolist()
        self.query_start_loc = torch.tensor(starts, dtype=torch.int32)
        self.logits_indices = torch.tensor(starts[1:] - 1)


class _Engine:
    """Replays Model Runner V2 steps around a talker with stand-in backbone and depth decoder."""

    def __init__(self, talker: CsmTalkerForConditionalGeneration, monkeypatch: pytest.MonkeyPatch) -> None:
        self.talker = talker
        self.requests: dict[str, SimpleNamespace] = {}
        self.frame_calls: list[dict] = []
        self.first_codes: list[torch.Tensor] = []
        self.decode_inputs: list[tuple[str, torch.Tensor]] = []
        self.prefill_inputs: dict[str, torch.Tensor] = {}
        hidden_size = talker.config.hidden_size

        # lm_head scores planned code ``c`` from hidden direction ``PLANNED_CODES.index(c)``.
        with torch.no_grad():
            talker.lm_head.weight.zero_()
            for direction, code in enumerate(PLANNED_CODES):
                talker.lm_head.weight[code, direction] = 1.0

        def generate_frame(backbone_hidden, first_codes, **sampling):
            self.frame_calls.append(sampling)
            self.first_codes.append(first_codes.clone())
            steps = torch.arange(talker.num_codebooks)
            codes = (first_codes[:, None] + steps) % talker.codebook_vocab_size
            # A planned first code of 0 completes a stop frame; its last codebook is not zero.
            stop = first_codes == 0
            codes = torch.where(stop[:, None], torch.zeros_like(codes), codes)
            codes[:, -1] = torch.where(stop, torch.full_like(first_codes, 5), codes[:, -1])
            return codes, talker.embed_audio_frames(codes)

        monkeypatch.setattr(talker, "generate_frame", generate_frame)
        monkeypatch.setattr("vllm.forward_context.set_forward_context", lambda *_, **__: nullcontext())

        state = object.__new__(OmniModelState)
        state.model = talker
        state.device = torch.device("cpu")
        state.dtype = torch.float32
        state.scheduler_config = SimpleNamespace(max_num_seqs=CAPACITY)
        state.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(async_chunk=False),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        )
        state.has_preprocess = True
        state.has_postprocess = False
        state.have_multimodal_outputs = True
        state.preprocess_keeps_mm_inputs = False
        state._decode_preprocess_is_identity = False
        state._static_inputs_embeds = None
        state._embed_dim = hidden_size
        state.intermediate_buffer = OmniIntermediateBuffer(CAPACITY)
        state._mtp_state = MTPState(state)
        # Call the model's MTP directly; the graph-wrapper lookup needs a GPU platform.
        state._mtp_state._mtp_runner = None
        state._has_mtp = True
        self.state = state

    def add(
        self,
        req_id: str,
        slot: int,
        plan: list[int],
        *,
        seed: int | None = None,
        prompt_len: int = 3,
        prompt: list[int] | None = None,
        **request_fields,
    ) -> None:
        extra_args = {"tts_local_seed": seed} if seed is not None else None
        self.state.intermediate_buffer.buffers[slot] = {
            "req_id": req_id,
            "sampling_params": SamplingParams(max_tokens=64, extra_args=extra_args),
            **request_fields,
        }
        prompt = list(range(100, 100 + prompt_len)) if prompt is None else prompt
        self.requests[req_id] = SimpleNamespace(slot=slot, plan=plan, prompt=prompt, computed=0, tokens=[], frames=[])

    def finish(self, req_id: str) -> None:
        """What the runner does for a finished or aborted request."""
        request = self.requests.pop(req_id)
        self.talker.on_requests_finished({req_id})
        self.state.intermediate_buffer.buffers[request.slot] = {}

    def step(self, order: list[str]) -> torch.Tensor:
        """Run one engine step for ``order`` and return the sampled tokens."""
        talker, state = self.talker, self.state
        requests = [self.requests[req_id] for req_id in order]
        tokens = [request.prompt if not request.tokens else [request.tokens[-1]] for request in requests]
        batch = _Batch([request.slot for request in requests], list(order), tokens)
        prompt_len = np.zeros(CAPACITY, dtype=np.int32)
        computed = np.zeros(CAPACITY, dtype=np.int32)
        for request in requests:
            prompt_len[request.slot] = len(request.prompt)
            computed[request.slot] = request.computed
        req_states = SimpleNamespace(prompt_len=prompt_len, num_computed_tokens=computed)

        inputs = {"input_ids": batch.input_ids, "inputs_embeds": None}
        state.run_preprocess(batch, inputs, req_states)
        for row, request in enumerate(requests):
            if request.tokens:
                self.decode_inputs.append((order[row], inputs["inputs_embeds"][batch.logits_indices[row]].clone()))
            else:
                span = slice(batch.query_start_loc_np[row], int(batch.logits_indices[row]) + 1)
                self.prefill_inputs[order[row]] = inputs["inputs_embeds"][span].clone()

        # Stand-in backbone: each request's last position points at its planned first codebook.
        hidden = torch.zeros((batch.num_tokens, talker.config.hidden_size))
        for row, request in enumerate(requests):
            code = request.plan[len(request.tokens)]
            hidden[batch.logits_indices[row], PLANNED_CODES.index(code)] = 10.0

        text_hidden, outputs = state.postprocess_model_output(hidden, batch, req_states)
        num_output_tokens = torch.tensor([len(request.tokens) for request in requests])
        with talker.mrv2_sampling_context(req_ids=batch.req_ids, num_output_tokens=num_output_tokens):
            logits = talker.compute_logits(text_hidden[batch.logits_indices])
        sampled = logits.argmax(dim=-1, keepdim=True)
        state.post_sample(batch, text_hidden, sampled, outputs)

        codes = outputs["codes"]["audio"][batch.logits_indices]
        valid = outputs["meta"]["codec_frame_valid"][batch.logits_indices]
        for row, request in enumerate(requests):
            request.computed += len(tokens[row])
            request.tokens.append(int(sampled[row]))
            if bool(valid[row]):
                request.frames.append(codes[row].clone())
        return sampled.reshape(-1)


@pytest.fixture
def talker() -> Iterator[CsmTalkerForConditionalGeneration]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        torch.manual_seed(0)
        yield _build_talker(_config(), monkeypatch)


@pytest.fixture
def engine(talker: CsmTalkerForConditionalGeneration, monkeypatch: pytest.MonkeyPatch) -> _Engine:
    return _Engine(talker, monkeypatch)


def test_declared_capabilities(talker: CsmTalkerForConditionalGeneration) -> None:
    assert talker.has_talker_mtp and talker.has_preprocess and talker.have_multimodal_outputs
    assert talker.mtp_accepts_req_infos and talker.mtp_accepts_per_row_generators
    assert talker.logitsprocs_need_output_token_ids
    assert talker.stop_token_id == talker.config.codebook_pad_token_id == 2050
    assert talker.logits_vocab_size == talker.config.vocab_size == 2051
    assert talker.mtp_output_key == ("codes", "audio")
    assert talker.mtp_sampling_params == {"do_sample": True, "temperature": 0.9, "top_k": 50, "top_p": 1.0}
    assert talker.get_mtp_seed(SamplingParams(extra_args={"tts_local_seed": 7})) == 7
    assert talker.mtp_frame_valid(torch.tensor([0, 2049, 2050, 2051])).tolist() == [True, True, False, False]


def test_stop_token_follows_the_configured_padding_id(monkeypatch: pytest.MonkeyPatch) -> None:
    talker = _build_talker(_config(codebook_pad_token_id=2049), monkeypatch)
    assert talker.stop_token_id == 2049
    assert talker.mtp_frame_valid(torch.tensor([2048, 2049, 2050])).tolist() == [True, False, True]
    logits = talker.compute_logits(torch.zeros((1, talker.config.hidden_size)))
    assert torch.isneginf(logits[0]).nonzero().flatten().tolist() == [2049]


def test_deploy_sampling_params_override_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    talker = _build_talker(config, monkeypatch, subtalker_sampling_params={"do_sample": False, "top_k": 10})
    assert talker.mtp_sampling_params == {"do_sample": False, "temperature": 0.9, "top_k": 10, "top_p": 1.0}


def test_length_end_has_one_frame_per_token(engine: _Engine) -> None:
    engine.add("a", slot=0, plan=[7, 9, 11])
    for _ in range(3):
        engine.step(["a"])

    request = engine.requests["a"]
    assert request.tokens == [7, 9, 11]
    assert [int(frame[0]) for frame in request.frames] == [7, 9, 11]
    assert all(frame.shape == (32,) for frame in request.frames)


def test_stop_frame_forces_the_stop_token_on_the_next_step(engine: _Engine) -> None:
    # The plan would continue with 11, but the frame of step 1 is a stop frame.
    engine.add("a", slot=0, plan=[7, 0, 11, 11])
    for _ in range(3):
        engine.step(["a"])

    request = engine.requests["a"]
    assert request.tokens == [7, 0, engine.talker.stop_token_id]
    # The stop frame is published; the forced step publishes nothing.
    assert len(request.frames) == 2
    assert request.frames[1][:-1].eq(0).all() and int(request.frames[1][-1]) == 5


def test_the_step_after_the_stop_token_publishes_nothing(engine: _Engine) -> None:
    """Async scheduling runs one more step on a request that already sampled the stop token."""
    talker = engine.talker
    engine.add("a", slot=0, plan=[7, 0, 11, 11])
    engine.add("b", slot=1, plan=[9, 9, 9, 9, 9])
    for _ in range(3):
        engine.step(["a", "b"])
    stopped, other = engine.requests["a"], engine.requests["b"]
    assert stopped.tokens == [7, 0, talker.stop_token_id]
    assert len(stopped.frames) == 2

    # The scheduler issued this step before it saw the stop token, so the stop token is the input.
    sampled = engine.step(["a", "b"])

    assert sampled.tolist() == [talker.stop_token_id, 9]
    assert len(stopped.frames) == 2
    assert [int(frame[0]) for frame in other.frames] == [9, 9, 9, 9]
    # The depth decoder only ever sees codes: the stop token is replaced before the lookup.
    assert len(engine.first_codes) == 4
    for first_codes in engine.first_codes:
        assert talker.stop_token_id not in first_codes.tolist()
    assert engine.first_codes[-1].tolist() == [0, 9]

    # The runner reports the request finished only after that step; nothing of it stays behind.
    engine.finish("a")
    assert set(talker._stop_flags) == {"b"}
    assert engine.step(["b"]).tolist() == [9]
    assert set(talker._stop_flags) == {"b"}
    engine.finish("b")
    assert talker._stop_flags == {}


def test_reference_audio_is_embedded_at_prefill_through_the_model_state(engine: _Engine) -> None:
    talker = engine.talker
    audio, audio_eos = talker.config.audio_token_id, talker.config.audio_eos_token_id
    prompt = [100, 101, audio, audio, audio_eos, 102, 103]
    waveform = 0.1 * torch.randn(1920 * 2, generator=torch.Generator().manual_seed(0))
    engine.add("with_reference", slot=0, plan=[7, 9], prompt=prompt, **{REFERENCE_AUDIO_KEY: waveform})
    engine.add("text_only", slot=1, plan=[9, 9])

    with torch.no_grad():
        engine.step(["with_reference", "text_only"])
        codes = talker.codec_model.encode(waveform[None, None]).audio_codes[0].T
        frames = talker.embed_audio_frames(codes)
        audio_eos_frame = talker.embed_audio_frames(torch.zeros((1, talker.num_codebooks), dtype=torch.long))
        text = talker.embed_input_ids(torch.tensor(prompt))

    embeds = engine.prefill_inputs["with_reference"]
    torch.testing.assert_close(embeds[2:4], frames)
    torch.testing.assert_close(embeds[4:5], audio_eos_frame)
    assert torch.equal(embeds[:2], text[:2]) and torch.equal(embeds[5:], text[5:])
    assert not torch.isclose(embeds[2:5], text[2:5]).all(dim=-1).any()
    # The other request of the batch is untouched, and nothing is kept after the prefill.
    assert torch.equal(engine.prefill_inputs["text_only"], talker.embed_input_ids(torch.tensor([100, 101, 102])))
    assert talker._references == {}

    engine.step(["with_reference", "text_only"])
    assert [int(frame[0]) for frame in engine.requests["with_reference"].frames] == [7, 9]


def test_decode_input_is_the_previous_frame_embedding(engine: _Engine) -> None:
    engine.add("a", slot=0, plan=[7, 9, 11])
    for _ in range(3):
        engine.step(["a"])

    frames = engine.requests["a"].frames
    assert len(engine.decode_inputs) == 2
    for (_, embeds), frame in zip(engine.decode_inputs, frames):
        torch.testing.assert_close(embeds, engine.talker.embed_audio_frames(frame))


def test_stop_flag_follows_the_request_when_rows_reorder(engine: _Engine) -> None:
    engine.add("a", slot=0, plan=[7, 0, 11])
    engine.add("b", slot=1, plan=[9, 9, 9])
    engine.step(["a", "b"])
    engine.step(["a", "b"])
    sampled = engine.step(["b", "a"])

    assert sampled.tolist() == [9, engine.talker.stop_token_id]
    assert len(engine.requests["b"].frames) == 3


def test_stop_flag_survives_a_step_without_the_request(engine: _Engine) -> None:
    engine.add("a", slot=0, plan=[7, 0, 11])
    engine.add("b", slot=1, plan=[9, 9, 9, 9])
    engine.step(["a", "b"])
    engine.step(["a", "b"])
    assert engine.step(["b"]).tolist() == [9]
    assert engine.step(["a", "b"]).tolist() == [engine.talker.stop_token_id, 9]


def test_finished_requests_drop_their_flags(engine: _Engine) -> None:
    engine.add("a", slot=0, plan=[0, 11])
    engine.add("b", slot=1, plan=[0, 11])
    engine.step(["a", "b"])
    assert set(engine.talker._stop_flags) == {"a", "b"}

    engine.step(["a", "b"])
    engine.finish("a")  # finished on the stop token
    engine.finish("b")  # the same call covers an abort
    assert engine.talker._stop_flags == {}

    # A new request in a reused slot starts clean.
    engine.add("c", slot=0, plan=[7, 9])
    assert engine.step(["c"]).tolist() == [7]


def test_stop_mask_applies_only_while_sampling(engine: _Engine) -> None:
    engine.add("a", slot=0, plan=[0, 11])
    engine.step(["a"])
    talker = engine.talker
    hidden = torch.zeros((1, talker.config.hidden_size))
    hidden[0, PLANNED_CODES.index(11)] = 10.0

    # Outside the sampling context (prompt logprobs) nothing is forced.
    logits = talker.compute_logits(hidden)
    assert logits.shape == (1, talker.logits_vocab_size)
    assert int(logits.argmax(dim=-1)) == 11
    assert torch.isneginf(logits[:, talker.stop_token_id]).all()

    with talker.mrv2_sampling_context(req_ids=["a"], num_output_tokens=torch.tensor([1])):
        assert int(talker.compute_logits(hidden).argmax(dim=-1)) == talker.stop_token_id
        # A call with a different number of rows is left alone.
        assert talker.compute_logits(hidden.expand(2, -1)).argmax(dim=-1).tolist() == [11, 11]
    assert talker._forced_stop_mask is None


def test_depth_sampling_uses_model_wide_values_and_request_seeds(engine: _Engine) -> None:
    engine.add("seeded", slot=0, plan=[7, 9], seed=5)
    engine.add("unseeded", slot=1, plan=[9, 9])
    engine.step(["seeded", "unseeded"])

    call = engine.frame_calls[-1]
    assert (call["do_sample"], call["temperature"], call["top_k"], call["top_p"]) == (True, 0.9, 50, 1.0)
    uniforms = call["uniforms"]
    assert uniforms.shape == (2, 31, 2051)

    expected = torch.empty((1, 31, 2051)).uniform_(1e-20, 1.0 - 1e-20, generator=torch.Generator().manual_seed(5))
    assert torch.equal(uniforms[:1], expected)
    assert not torch.equal(uniforms[1:], expected)


def test_make_omni_output_allocates_token_major_frames(talker: CsmTalkerForConditionalGeneration) -> None:
    hidden = torch.zeros((5, talker.config.hidden_size))
    output = talker.make_omni_output(hidden, request_token_spans=[(0, 3), (3, 5)])
    assert output.text_hidden_states is hidden
    assert output.multimodal_outputs["codes"]["audio"].shape == (5, 32)
    assert output.multimodal_outputs["meta"]["codec_frame_valid"].tolist() == [0] * 5
    assert talker.make_omni_output(output) is output
