# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
import io
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.personaplex.duplex.policy import (
    SILENCE_TOKENS,
    SINE_TOKENS,
    ZERO_TEXT_TOKEN,
)
from vllm_omni.model_executor.models.personaplex.duplex.stage0 import (
    PersonaPlexStage0CapacityError,
    PersonaPlexStage0DuplexRuntime,
    PersonaPlexStage0PreparedAppend,
    PersonaPlexStage0StaleEpochError,
    load_personaplex_voice_state,
)
from vllm_omni.model_executor.models.personaplex.personaplex_talker import (
    PersonaPlexTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeCodec:
    """Shared streaming encoder whose code for a row is that row's frame count."""

    def __init__(self) -> None:
        self.encode_calls = 0
        self.reset_slots: list[int] = []
        self.frames: list[int] = []

    def streaming_init(self, batch_size: int, *, decode: bool = True) -> None:
        assert not decode, "Stage 0 only encodes"
        self.frames = [0] * batch_size

    def encode_frame(self, pcm: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        assert pcm.shape == (len(self.frames), 1920)
        assert active.shape == (len(self.frames),)
        self.encode_calls += 1
        codes = torch.zeros((len(self.frames), 8), dtype=torch.long, device=pcm.device)
        for row, is_active in enumerate(active.tolist()):
            if is_active:
                self.frames[row] += 1
                codes[row] = self.frames[row]
        return codes

    def reset_slot(self, row: int) -> None:
        self.reset_slots.append(row)
        self.frames[row] = 0


class _FakeTalker:
    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.frame_calls: list[dict[str, torch.Tensor]] = []

    def _build_prefill_embed(
        self,
        tokens,
        offset,
        span,
        device,
        silence=None,
        user_sine=None,
    ):
        del offset, silence, user_sine
        values = tokens[:span].to(device=device, dtype=torch.float32)
        return values[:, None].expand(-1, 4).contiguous()

    def _build_frame_embeds(self, text_tokens, last_agent, *, user_d0, user_d1):
        self.frame_calls.append(
            {
                "text_tokens": text_tokens.clone(),
                "last_agent": last_agent.clone(),
                "user_d0": user_d0.clone(),
                "user_d1": user_d1.clone(),
            }
        )
        return user_d0[:, :1].to(torch.float32).expand(-1, 4).contiguous()


def _duplex_info(*, seq: int, session_id: str = "session", epoch: int = 0):
    pcm = np.zeros(1920, dtype="<f4")
    return {
        "data_plane": True,
        "session_id": session_id,
        "epoch": epoch,
        "seq": seq,
        "payload": {
            "format": "pcm_f32le",
            "sample_rate_hz": 24000,
            "audio": base64.b64encode(pcm.tobytes()).decode("ascii"),
        },
        "runtime_config": {
            "personaplex_model_path": "/unused",
            "personaplex_voice_prompt": "NATF2.pt",
            "personaplex_persona": "Be concise.",
        },
    }


def _runtime(codec: _FakeCodec, max_sessions: int = 1, device: str = "cpu") -> PersonaPlexStage0DuplexRuntime:
    dev = torch.device(device)
    voice_embeddings = torch.arange(8, dtype=torch.float32, device=dev).reshape(2, 1, 1, 4)
    return PersonaPlexStage0DuplexRuntime(
        _FakeTalker(dev),
        model_path="/unused",
        device=device,
        codec=codec,
        max_sessions=max_sessions,
        tokenizer=lambda _text: [7, 8, 9],
        voice_loader=lambda _voice: {
            "embeddings": voice_embeddings,
            "cache": torch.zeros((1, 17, 4), dtype=torch.long, device=dev),
        },
    )


def test_first_append_prepends_voice_and_persona_once() -> None:
    codec = _FakeCodec()
    runtime = _runtime(codec)

    first = runtime.prepare_append(_duplex_info(seq=1), prompt_len=18)
    second = runtime.prepare_append(_duplex_info(seq=2), prompt_len=18)

    assert first.prefill_applied is True
    assert first.prompt_offset == 0
    assert first.user_frame.tolist() == [[1] * 8]
    assert first.inputs_embeds.shape == (18, 4)
    assert first.info_update["meta"] == {"pplex_frame": 18, "pplex_prefill_len": 17}
    assert second.prefill_applied is False
    assert second.prompt_offset == 17
    assert second.user_frame.tolist() == [[2] * 8]
    assert second.inputs_embeds.shape == (1, 4)
    assert second.info_update["meta"] == {"pplex_frame": 19, "pplex_prefill_len": 17}
    assert codec.encode_calls == 2


@pytest.mark.parametrize(
    ("append_count", "expected_tokens", "expected_provided"),
    [
        (
            1,
            [*SILENCE_TOKENS, 1, *SINE_TOKENS[1:]],
            [False, *([True] * 15)],
        ),
        (
            2,
            [*SILENCE_TOKENS, 2, *([1] * 7)],
            [*([False] * 8), *([True] * 8)],
        ),
    ],
)
def test_prepare_append_preserves_native_depformer_teacher_forcing(
    append_count: int,
    expected_tokens: list[int],
    expected_provided: list[bool],
) -> None:
    runtime = _runtime(_FakeCodec())

    for seq in range(1, append_count + 1):
        runtime.prepare_append(_duplex_info(seq=seq), prompt_len=18, request_id="req")
    tokens, provided = runtime.depformer_teacher_forcing(["req"])

    assert tokens.tolist() == [expected_tokens]
    assert provided.tolist() == [expected_provided]


def test_frame_embed_uses_previous_effective_agent_frame() -> None:
    fake_talker = SimpleNamespace(
        config=SimpleNamespace(
            num_audio_codebooks=16,
            audio_vocab_size=2048,
        ),
        input_embeddings=lambda stack: stack,
    )
    last_agent = torch.arange(10, 18)

    embeds = PersonaPlexTalkerForConditionalGeneration._build_frame_embed(
        fake_talker,
        torch.tensor([3]),
        last_agent,
        torch.arange(8),
        torch.device("cpu"),
    )

    assert embeds.shape == (1, 17)
    assert torch.equal(embeds[0, 1:9], last_agent)


def test_batched_frame_embeds_match_the_per_request_frame_embed() -> None:
    from vllm_omni.model_executor.models.personaplex.personaplex_embeddings import (
        PersonaPlexInputEmbeddings,
    )

    torch.manual_seed(0)
    config = SimpleNamespace(
        temporal_config=SimpleNamespace(hidden_size=8),
        num_audio_codebooks=16,
        text_embedding_rows=33,
        audio_vocab_size=2048,
    )
    talker = SimpleNamespace(config=config, input_embeddings=PersonaPlexInputEmbeddings(config))
    generator = torch.Generator().manual_seed(0)
    text = torch.randint(0, 32, (5,), generator=generator)
    last_agent, user_d0, user_d1 = (torch.randint(0, 2048, (5, 8), generator=generator) for _ in range(3))

    batched = PersonaPlexTalkerForConditionalGeneration._build_frame_embeds(
        talker,
        text,
        last_agent,
        user_d0=user_d0,
        user_d1=user_d1,
    )
    single = torch.cat(
        [
            PersonaPlexTalkerForConditionalGeneration._build_frame_embed(
                talker,
                text[row : row + 1],
                last_agent[row],
                last_agent[row],
                torch.device("cpu"),
                user_d0=user_d0[row],
                user_d1=user_d1[row],
            )
            for row in range(5)
        ]
    )

    assert torch.equal(batched, single)


def test_repeated_append_identity_does_not_advance_codec() -> None:
    codec = _FakeCodec()
    runtime = _runtime(codec)

    info = _duplex_info(seq=1)
    first = runtime.prepare_append(info, prompt_len=18, request_id="req")
    retry = runtime.prepare_append(info, prompt_len=18, request_id="req")

    assert codec.encode_calls == 1
    assert torch.equal(first.inputs_embeds, retry.inputs_embeds)
    assert torch.equal(first.user_frame, retry.user_frame)


def test_next_append_uses_prior_sample_and_causally_delayed_user_frame() -> None:
    runtime = _runtime(_FakeCodec())

    runtime.prepare_append(_duplex_info(seq=1), prompt_len=18, request_id="req")
    first_agent = torch.arange(8, dtype=torch.long)
    runtime.record_samples(request_ids=["req"], text_tokens=torch.tensor(101), agent_codes=first_agent)
    runtime.prepare_append(_duplex_info(seq=2), prompt_len=19, request_id="req")
    second_agent = torch.arange(10, 18, dtype=torch.long)
    runtime.record_samples(request_ids=["req"], text_tokens=torch.tensor(102), agent_codes=second_agent)
    runtime.prepare_append(_duplex_info(seq=3), prompt_len=20, request_id="req")

    silence = torch.tensor([SILENCE_TOKENS], dtype=torch.long)
    sine = torch.tensor([SINE_TOKENS], dtype=torch.long)
    first_call, second_call, third_call = runtime.stage_model.frame_calls
    assert torch.equal(first_call["last_agent"], silence)
    # Before live user frames fill the delayed user rows they read sine.
    assert all(torch.equal(first_call[key], sine) for key in ("user_d0", "user_d1"))

    assert second_call["text_tokens"].tolist() == [101]
    expected_first_effective = torch.cat([first_agent[None, :1], silence[:, 1:]], dim=1)
    assert torch.equal(second_call["last_agent"], expected_first_effective)
    assert torch.equal(second_call["user_d0"], torch.full((1, 8), 1, dtype=torch.long))
    assert torch.equal(second_call["user_d1"], sine)

    assert third_call["text_tokens"].tolist() == [102]
    assert torch.equal(third_call["last_agent"], second_agent[None])
    assert torch.equal(third_call["user_d0"], torch.full((1, 8), 2, dtype=torch.long))
    assert torch.equal(third_call["user_d1"], torch.full((1, 8), 1, dtype=torch.long))


def _prepare_two_sessions(
    runtime: PersonaPlexStage0DuplexRuntime,
) -> tuple[
    PersonaPlexStage0PreparedAppend,
    PersonaPlexStage0PreparedAppend,
    PersonaPlexStage0PreparedAppend,
    PersonaPlexStage0PreparedAppend,
]:
    first_1 = runtime.prepare_append(_duplex_info(seq=1), prompt_len=18)
    second_1 = runtime.prepare_append(
        _duplex_info(seq=1, session_id="other"),
        prompt_len=18,
    )
    first_2 = runtime.prepare_append(_duplex_info(seq=2), prompt_len=19)
    second_2 = runtime.prepare_append(
        _duplex_info(seq=2, session_id="other"),
        prompt_len=19,
    )
    return first_1, second_1, first_2, second_2


def test_live_sessions_keep_independent_streaming_encoders() -> None:
    runtime = _runtime(_FakeCodec(), max_sessions=2)

    first_1, second_1, first_2, second_2 = _prepare_two_sessions(runtime)

    assert first_1.user_frame[:, 0].tolist() == [1]
    assert second_1.user_frame[:, 0].tolist() == [1]
    assert first_2.user_frame[:, 0].tolist() == [2]
    assert second_2.user_frame[:, 0].tolist() == [2]
    # Each second frame reads its own session's first frame as the delayed user cb0.
    assert [call["user_d0"][:, 0].tolist() for call in runtime.stage_model.frame_calls[2:]] == [[1], [1]]


def test_stage0_session_capacity_fails_before_codec_state_is_shared() -> None:
    runtime = _runtime(_FakeCodec(), max_sessions=2)
    _prepare_two_sessions(runtime)

    with pytest.raises(RuntimeError, match="capacity 2"):
        runtime.prepare_append(
            _duplex_info(seq=1, session_id="third"),
            prompt_len=18,
        )


def test_close_session_resets_only_its_row_and_reuses_it() -> None:
    codec = _FakeCodec()
    runtime = _runtime(codec, max_sessions=2)
    _prepare_two_sessions(runtime)
    closed_slot = runtime.sessions[("session", 0)].slot
    other_slot = runtime.sessions[("other", 0)].slot

    runtime.close_session("session", 0)

    assert codec.reset_slots == [closed_slot]
    replacement = runtime.prepare_append(
        _duplex_info(seq=1, session_id="replacement"),
        prompt_len=18,
    )
    assert runtime.sessions[("replacement", 0)].slot == closed_slot
    assert replacement.user_frame[:, 0].tolist() == [1]
    assert codec.frames[other_slot] == 2


def test_a_new_epoch_replays_the_prefill_and_recycles_the_codec() -> None:
    codec = _FakeCodec()
    runtime = _runtime(codec)
    runtime.prepare_append(_duplex_info(seq=1), prompt_len=18, request_id="req-e0")
    runtime.prepare_append(_duplex_info(seq=2), prompt_len=19, request_id="req-e0")

    # A cancel advanced the fence: the next append is seq 1 of epoch 1 on a
    # fresh Stage 0 request, so the voice/persona prefill is replayed and the
    # earlier epoch's lockstep state is released first.
    restarted = runtime.prepare_append(_duplex_info(seq=1, epoch=1), prompt_len=18, request_id="req-e1")

    assert restarted.prefill_applied is True
    assert restarted.prompt_offset == 0
    assert restarted.user_frame[:, 0].tolist() == [1]
    assert restarted.info_update["meta"]["pplex_frame"] == 18
    assert list(runtime.sessions) == [("session", 1)]
    assert runtime.request_sessions == {"req-e1": ("session", 1)}
    assert codec.reset_slots == [0]
    assert runtime.sessions[("session", 1)].slot == 0


def test_a_late_finish_of_the_old_epoch_request_does_not_close_the_new_state() -> None:
    runtime = _runtime(_FakeCodec())
    runtime.prepare_append(_duplex_info(seq=1), prompt_len=18, request_id="req-e0")
    runtime.prepare_append(_duplex_info(seq=1, epoch=1), prompt_len=18, request_id="req-e1")

    runtime.close_request("req-e0")

    assert list(runtime.sessions) == [("session", 1)]
    runtime.close_request("req-e1")
    assert runtime.sessions == {}


def test_stage0_capacity_counts_live_epochs_not_superseded_ones() -> None:
    runtime = _runtime(_FakeCodec(), max_sessions=2)
    runtime.prepare_append(_duplex_info(seq=1), prompt_len=18)
    runtime.prepare_append(_duplex_info(seq=1, session_id="other"), prompt_len=18)

    # Restarting one session must not need a third encoder row.
    runtime.prepare_append(_duplex_info(seq=1, epoch=1), prompt_len=18)

    assert sorted(runtime.sessions) == [("other", 0), ("session", 1)]


def test_encode_appends_batches_sessions_into_one_encoder_call() -> None:
    codec = _FakeCodec()
    runtime = _runtime(codec, max_sessions=4)

    runtime.encode_appends([_duplex_info(seq=1), _duplex_info(seq=1, session_id="other")])
    first = runtime.prepare_append(_duplex_info(seq=1), prompt_len=18)
    other = runtime.prepare_append(_duplex_info(seq=1, session_id="other"), prompt_len=18)

    assert codec.encode_calls == 1
    assert first.user_frame[:, 0].tolist() == [1]
    assert other.user_frame[:, 0].tolist() == [1]


def _restart_overlap(runtime: PersonaPlexStage0DuplexRuntime) -> tuple[dict, dict]:
    """Epoch 0 of a session is live, then a cancel restarts it as epoch 1 while
    epoch 0's aborted request still has an append in the same scheduler step."""
    runtime.prepare_append(_duplex_info(seq=1), prompt_len=18, request_id="req-e0")
    runtime.prepare_append(_duplex_info(seq=2), prompt_len=19, request_id="req-e0")
    return _duplex_info(seq=3), _duplex_info(seq=1, epoch=1)


@pytest.mark.parametrize("max_sessions", [1, 2])
@pytest.mark.parametrize("encode_old_first", [True, False])
@pytest.mark.parametrize("prepare_old_first", [True, False])
def test_cancel_overlap_full_processing_order_never_re_leases_the_aborted_epoch(
    max_sessions: int, encode_old_first: bool, prepare_old_first: bool
) -> None:
    codec = _FakeCodec()
    runtime = _runtime(codec, max_sessions=max_sessions)
    old, new = _restart_overlap(runtime)

    # One scheduler step: batched encode, then per-request prepare, then sampling.
    runtime.encode_appends([old, new] if encode_old_first else [new, old])
    order = [("req-e0", old, 19), ("req-e1", new, 18)]
    restarted = None
    for request_id, duplex, prompt_len in order if prepare_old_first else order[::-1]:
        if request_id == "req-e0":
            with pytest.raises(PersonaPlexStage0StaleEpochError):
                runtime.prepare_append(duplex, prompt_len=prompt_len, request_id=request_id)
        else:
            restarted = runtime.prepare_append(duplex, prompt_len=prompt_len, request_id=request_id)
    for request_id in ("req-e0", "req-e1"):
        runtime.record_samples(request_ids=[request_id], text_tokens=5, agent_codes=list(range(8)))
    runtime.close_request("req-e0")  # the engine's late finish of the aborted request

    assert list(runtime.sessions) == [("session", 1)]
    assert len(runtime._free_slots) == max_sessions - 1
    assert restarted is not None and restarted.user_frame[:, 0].tolist() == [1]
    assert codec.encode_calls == 3
    assert runtime.sessions[("session", 1)].sampled_identity == (1, 1)


def test_codec_init_failure_propagates_after_one_attempt() -> None:
    attempts = 0

    def failing_codec():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("CUDA out of memory while building the Mimi encoder")

    runtime = PersonaPlexStage0DuplexRuntime(
        _FakeTalker(),
        model_path="/unused",
        device="cpu",
        codec_factory=failing_codec,
        max_sessions=16,
        tokenizer=lambda _text: [7, 8, 9],
        voice_loader=lambda _voice: {},
    )
    appends = [_duplex_info(seq=1, session_id=f"s{i}") for i in range(16)]

    with pytest.raises(RuntimeError, match="out of memory"):
        runtime.encode_appends(appends)
    assert attempts == 1
    assert not runtime.sessions


def test_capacity_error_is_a_dedicated_exception() -> None:
    runtime = _runtime(_FakeCodec(), max_sessions=1)
    runtime.prepare_append(_duplex_info(seq=1), prompt_len=18)

    with pytest.raises(PersonaPlexStage0CapacityError, match="capacity 1"):
        runtime.prepare_append(_duplex_info(seq=1, session_id="other"), prompt_len=18)


def test_talker_gives_the_aborted_epoch_a_neutral_row_in_the_same_step() -> None:
    codec = _FakeCodec()
    runtime = _runtime(codec, max_sessions=1)
    old, new = _restart_overlap(runtime)
    talker = PersonaPlexTalkerForConditionalGeneration.__new__(PersonaPlexTalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker._personaplex_duplex_stage0_runtime = runtime
    talker._dtype = torch.float32
    talker.mtp_hidden_size = 4

    infos = {
        "req-e0": {"duplex": old, "request_id": "req-e0", "duplex_prompt_len": 19, "duplex_token_offset": 0},
        "req-e1": {"duplex": new, "request_id": "req-e1", "duplex_prompt_len": 18, "duplex_token_offset": 0},
    }
    talker.preprocess_batch(req_ids=list(infos), model_intermediate_buffer=infos, device=torch.device("cpu"))
    outs = {
        req: talker.preprocess(torch.zeros(1, dtype=torch.long), None, _omni_is_prefill=True, **info)
        for req, info in infos.items()
    }

    _, stale_embeds, stale_info = outs["req-e0"]
    assert stale_info["duplex"] == {"stage0_stale": True}
    assert torch.count_nonzero(stale_embeds) == 0
    assert outs["req-e1"][2]["duplex"]["epoch"] == 1
    assert list(runtime.sessions) == [("session", 1)]

    received: dict[str, torch.Tensor] = {}

    def depformer(text_token, hidden, *, audio_tokens, audio_provided, num_steps):
        received["audio_tokens"] = audio_tokens
        received["audio_provided"] = audio_provided
        return torch.arange(2 * num_steps, dtype=torch.long).reshape(2, num_steps)

    talker.depformer = depformer
    talker.num_active_codebooks = 8
    talker.post_sample_talker_mtp(
        input_ids=torch.tensor([7, 9]),
        hidden_states=torch.zeros((2, 4)),
        req_ids=list(infos),
        req_infos=[outs[req][2] for req in infos],
    )

    # The aborted epoch's row is neutral: silence and nothing forced.
    assert received["audio_tokens"][0].tolist() == [*SILENCE_TOKENS, *SILENCE_TOKENS]
    assert not received["audio_provided"][0].any()


def _record_two_frames(runtime: PersonaPlexStage0DuplexRuntime, *, batched: bool) -> None:
    sessions = ["a", "b", "c"]
    for seq in (1, 2):
        runtime.encode_appends([_duplex_info(seq=seq, session_id=session_id) for session_id in sessions])
        for session_id in sessions:
            runtime.prepare_append(
                _duplex_info(seq=seq, session_id=session_id), prompt_len=17 + seq, request_id=session_id
            )
        # A request that reappears in the batch commits its session's sample once.
        request_ids = [*sessions, "a"]
        text = torch.tensor([100 * seq + row for row in range(4)])
        codes = torch.arange(32).reshape(4, 8) + 1000 * seq
        if batched:
            runtime.record_samples(request_ids=request_ids, text_tokens=text, agent_codes=codes)
        else:
            for row, request_id in enumerate(request_ids):
                runtime.record_samples(request_ids=[request_id], text_tokens=text[row], agent_codes=codes[row])
    runtime.encode_appends([_duplex_info(seq=3, session_id=session_id) for session_id in sessions])


def test_batched_record_samples_match_one_row_at_a_time() -> None:
    batched = _runtime(_FakeCodec(), max_sessions=4)
    sequential = _runtime(_FakeCodec(), max_sessions=4)

    _record_two_frames(batched, batched=True)
    _record_two_frames(sequential, batched=False)

    for key in ("text_tokens", "last_agent", "user_d0", "user_d1"):
        assert torch.equal(batched.stage_model.frame_calls[-1][key], sequential.stage_model.frame_calls[-1][key])
    last = batched.stage_model.frame_calls[-1]
    # The first append forces agent cb1..7 to silence; later frames keep the samples.
    assert last["text_tokens"].tolist() == [200, 201, 202]
    assert torch.equal(last["last_agent"], torch.arange(24).reshape(3, 8) + 2000)
    assert torch.equal(
        batched.depformer_teacher_forcing(["a", "b", "c"])[0],
        sequential.depformer_teacher_forcing(["a", "b", "c"])[0],
    )


def test_a_leased_slot_starts_from_the_initial_frame_state() -> None:
    runtime = _runtime(_FakeCodec(), max_sessions=1)
    for seq in (1, 2):
        runtime.prepare_append(_duplex_info(seq=seq), prompt_len=17 + seq, request_id="old")
        runtime.record_samples(request_ids=["old"], text_tokens=55, agent_codes=list(range(8)))
    runtime.close_request("old")
    runtime.stage_model.frame_calls.clear()

    runtime.prepare_append(_duplex_info(seq=1, session_id="new"), prompt_len=18, request_id="new")

    (call,) = runtime.stage_model.frame_calls
    assert call["text_tokens"].tolist() == [ZERO_TEXT_TOKEN]
    assert call["last_agent"].tolist() == [list(SILENCE_TOKENS)]
    assert call["user_d0"].tolist() == [list(SINE_TOKENS)]
    assert call["user_d1"].tolist() == [list(SINE_TOKENS)]


def test_a_newer_frame_replaces_an_encoded_frame_that_was_never_prepared() -> None:
    runtime = _runtime(_FakeCodec())

    runtime.encode_appends([_duplex_info(seq=1)])
    first = runtime.prepare_append(_duplex_info(seq=2), prompt_len=18, request_id="req")
    runtime.prepare_append(_duplex_info(seq=3), prompt_len=19, request_id="req")

    # Frame 1 was encoded but never prepared, so it is not a user frame of the
    # session: frame 2 is its first one and frame 3 reads it as the delayed cb0.
    assert first.user_frame[:, 0].tolist() == [2]
    assert first.info_update["meta"]["pplex_frame"] == 18
    assert runtime.stage_model.frame_calls[-1]["user_d0"][:, 0].tolist() == [2]
    assert runtime.stage_model.frame_calls[-1]["user_d1"].tolist() == [list(SINE_TOKENS)]
    assert runtime.sessions[("session", 0)].user_frames == 2


def test_live_appends_match_the_per_request_prepare() -> None:
    sessions = [f"s{i}" for i in range(4)]
    batched = _runtime(_FakeCodec(), max_sessions=4)
    reference = _runtime(_FakeCodec(), max_sessions=4)
    for rt in (batched, reference):
        rt.encode_appends([_duplex_info(seq=1, session_id=s) for s in sessions])
        for s in sessions:
            rt.prepare_append(_duplex_info(seq=1, session_id=s), prompt_len=18, request_id=s)
        rt.record_samples(
            request_ids=sessions,
            text_tokens=torch.arange(4) + 5,
            agent_codes=torch.zeros((4, 8), dtype=torch.long),
        )

    appends = [(s, _duplex_info(seq=2, session_id=s), 19) for s in sessions]
    batched.encode_appends([d for _, d, _ in appends])
    handled, embeds = batched.prepare_live_appends(appends)

    reference.encode_appends([_duplex_info(seq=2, session_id=s) for s in sessions])
    prepared = [
        reference.prepare_append(_duplex_info(seq=2, session_id=s), prompt_len=19, request_id=s) for s in sessions
    ]

    assert handled == list(range(4))
    assert torch.equal(embeds, torch.cat([item.inputs_embeds for item in prepared]))
    assert batched.request_sessions == reference.request_sessions
    for key in ("_last_text", "_last_agent", "_user_history", "_teacher_tokens", "_teacher_provided"):
        assert torch.equal(getattr(batched, key), getattr(reference, key)), key


def test_talker_batch_preprocess_writes_the_rows() -> None:
    sessions = [f"s{i}" for i in range(4)]
    runtime = _runtime(_FakeCodec(), max_sessions=4)
    runtime.encode_appends([_duplex_info(seq=1, session_id=s) for s in sessions])
    for s in sessions:
        runtime.prepare_append(_duplex_info(seq=1, session_id=s), prompt_len=18, request_id=s)
    runtime.record_samples(
        request_ids=sessions,
        text_tokens=torch.arange(4) + 5,
        agent_codes=torch.zeros((4, 8), dtype=torch.long),
    )

    talker = PersonaPlexTalkerForConditionalGeneration.__new__(PersonaPlexTalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker._personaplex_duplex_stage0_runtime = runtime
    talker._dtype = torch.float32
    talker.mtp_hidden_size = 4

    req_ids = [*sessions, "unhandled"]
    buffer = {s: {"duplex": _duplex_info(seq=2, session_id=s)} for s in req_ids}
    inputs_embeds = torch.full((5, 4), -1.0)
    handled = talker.preprocess_batch(
        req_ids=req_ids,
        model_intermediate_buffer=buffer,
        device=torch.device("cpu"),
        input_ids=torch.full((5,), 9, dtype=torch.int32),
        inputs_embeds=inputs_embeds,
        token_offsets=list(range(5)),
        num_scheduled_tokens=[1] * 5,
        num_computed_tokens=[18, 18, 18, 18, 0],
        prompt_lens=[19] * 5,
    )
    assert handled == set(sessions)
    assert torch.equal(inputs_embeds[4], torch.full((4,), -1.0))


def test_prefill_cache_and_voice_archive(tmp_path) -> None:
    runtime = _runtime(_FakeCodec(), max_sessions=4)
    first = runtime.prepare_append(_duplex_info(seq=1, session_id="a"), prompt_len=64)
    second = runtime.prepare_append(_duplex_info(seq=1, session_id="b"), prompt_len=64)
    assert first.prefill_applied is True and second.prefill_applied is True
    assert torch.equal(first.inputs_embeds[:-1], second.inputs_embeds[:-1])

    diff = _duplex_info(seq=1, session_id="c")
    diff["runtime_config"]["personaplex_persona"] = "Different persona."
    third = runtime.prepare_append(diff, prompt_len=64)
    assert not torch.equal(first.inputs_embeds[:-1], third.inputs_embeds[:-1])

    buf = io.BytesIO()
    torch.save({"embeddings": torch.arange(4.0).reshape(1, 4)}, buf)
    data = buf.getvalue()
    ti = tarfile.TarInfo("voices/A.pt")
    ti.size = len(data)
    with tarfile.open(tmp_path / "voices.tgz", "w:gz") as tar:
        tar.addfile(ti, io.BytesIO(data))
    assert "embeddings" in load_personaplex_voice_state(str(tmp_path), "A.pt")

    fake = SimpleNamespace(
        _duplex_stage0_runtime=lambda: SimpleNamespace(warm_prefill=lambda: (_ for _ in ()).throw(FileNotFoundError))
    )
    PersonaPlexTalkerForConditionalGeneration._warm_duplex_prefill(fake)
