# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.gander import CONTROL_TOKENS, dialogue_constraint

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_gander_batches_live_audio_under_physical_request_owners(mocker):
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model.config = SimpleNamespace(gander_unit8=True)
    retired_state = object()
    helper = SimpleNamespace(
        sessions={"s1": retired_state},
        thinker=SimpleNamespace(),
        batches_audio_encoder=lambda: True,
        needs_prefill=lambda state, epoch, seq: state is None,
        frame_kwargs=lambda duplex, payload: {},
        _decode_audio_payload=lambda payload: payload["audio"],
        _configure_streaming_processor=lambda state: None,
        _prepare_session_context=lambda state, config, runtime_config: None,
        prefetch_vision=mocker.Mock(),
        stage_prefill_batch=mocker.Mock(),
    )
    model._minicpmo45_duplex_data_plane_helper = helper
    model._commit_minicpmo45_duplex_pending_samples = mocker.Mock()
    buffers = {
        request_id: {
            "duplex": {
                "data_plane": True,
                "session_id": session_id,
                "epoch": 2,
                "turn_id": 3,
                "seq": 1,
                "payload": {"audio": audio},
            }
        }
        for request_id, session_id, audio in [("r1", "s1", "audio1"), ("r2", "s2", "audio2")]
    }
    # Replay and controls must not get mixed into live encoder batching.
    for request_id, flag in [("replay", "gander_replay"), ("control", "gander_control")]:
        buffers[request_id] = {
            "duplex": {
                "data_plane": True,
                "session_id": request_id,
                "epoch": 2,
                "seq": 1,
                "payload": {flag: True, "audio": "historical"},
            }
        }
    model.preprocess_batch(req_ids=list(buffers), model_intermediate_buffer=buffers, device=torch.device("cpu"))
    helper.stage_prefill_batch.assert_called_once()
    staged = helper.stage_prefill_batch.call_args.args[0]
    assert [(state.session_id, audio) for state, audio, _ in staged] == [("s1", "audio1"), ("s2", "audio2")]
    assert staged[0][0] is helper.sessions["r1"] and staged[1][0] is helper.sessions["r2"]
    assert helper.sessions["s1"] is retired_state
    assert all(kwargs["turn_id"] == 3 for _, _, kwargs in staged)
    assert "replay" not in helper.sessions and "control" not in helper.sessions


@pytest.fixture
def ids():
    names = ["listen_token_id", "speak_token_id", "chunk_eos_token_id", "turn_eos_token_id", *CONTROL_TOKENS]
    return {key: index for index, key in enumerate(names)}


@pytest.fixture
def gander_host_sampler(ids, mocker):
    import torch
    from transformers import PretrainedConfig

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "llm"
    model.config = PretrainedConfig(gander_unit8=True)
    model._minicpmo45_native_duplex_token_ids_cache = {**ids, "unit_token_id": 10}
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.sessions = {"request": _MiniCPMO45Stage0SessionState(session_id="session")}
    model._minicpmo45_duplex_data_plane_helper = runtime
    mocker.patch.object(model, "_minicpmo45_tokenizer", return_value=mocker.Mock(eos_token_id=None))
    return model


def _gander_host_metadata(params, history):
    import torch
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata

    return SamplingMetadata(
        temperature=torch.tensor([p[0] for p in params]),
        all_greedy=False,
        all_random=False,
        top_k=torch.tensor([p[1] for p in params]),
        top_p=torch.tensor([p[2] for p in params]),
        generators={row: torch.Generator().manual_seed(7346 + row) for row in range(len(params))},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(len(params)),
        presence_penalties=torch.zeros(len(params)),
        repetition_penalties=torch.ones(len(params)),
        output_token_ids=history,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


@pytest.mark.parametrize("temperature", [0.0, 0.75])
def test_gander_host_snapshot_refreshes_without_metadata_scalar_reads(gander_host_sampler, ids, mocker, temperature):
    from dataclasses import replace

    import torch

    from vllm_omni.model_executor.duplex_sampling import DuplexSamplingRow

    model = gander_host_sampler
    metadata = _gander_host_metadata([(0.5, 1, 0.5), (temperature, 5, 0.875)], [[], [ids["speak_token_id"]]])
    row = DuplexSamplingRow(1, "request", "session", 1, {}, 11, temperature=temperature, top_k=5, top_p=0.875)
    logits = torch.full((2, 128), float("-inf"))
    logits[1, 100:105] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    model.prepare_duplex_sampling(logits, metadata, (row,))
    mocker.patch.object(
        model, "_sampling_metadata_value", side_effect=AssertionError("Gander read a device sampling scalar")
    )
    filtered = mocker.spy(model, "_top_k_top_p_filter")
    assert model._sample_gander_dialogue_row(logits[1:2], metadata, row_idx=1, token_ids=ids) in range(100, 105)
    if temperature == 0:
        filtered.assert_not_called()
    else:
        assert filtered.call_args.kwargs == {"top_k": 5, "top_p": 0.875}
    # The same request can append with new parameters and move to another batch row.
    refreshed = replace(row, row_idx=0, seq=2, temperature=0.5, top_k=1, top_p=1.0)
    metadata.output_token_ids[0] = [ids["speak_token_id"]]
    model.prepare_duplex_sampling(logits, metadata, (refreshed,))
    assert model._sample_gander_dialogue_row(logits[1:2], metadata, row_idx=0, token_ids=ids) == 104
    assert filtered.call_args.kwargs == {"top_k": 1, "top_p": 1.0}
    assert model._minicpmo45_duplex_row_sampling_host == {0: (0.5, 1, 1.0)}
    model.prepare_duplex_sampling(logits, metadata, ())
    assert model._minicpmo45_duplex_row_sampling_host == {}


@pytest.mark.parametrize("temperature", [0.0, 0.75])
@pytest.mark.parametrize("phase", ["action", "speech", "boundary"])
def test_gander_host_snapshot_preserves_seeded_samples_and_rng(gander_host_sampler, ids, temperature, phase):
    import torch

    from vllm_omni.model_executor.duplex_sampling import DuplexSamplingRow

    model = gander_host_sampler
    history = [] if phase == "action" else [ids["speak_token_id"], *([100] * 8 if phase == "boundary" else [])]
    metadata = _gander_host_metadata([(temperature, 5, 0.875)], [history])
    row = DuplexSamplingRow(0, "request", "session", 1, {}, 11, temperature=temperature, top_k=5, top_p=0.875)
    logits = torch.linspace(-2, 2, 128).reshape(1, -1)
    logits[0, ids["chunk_eos_token_id"]] = 2.0
    model.prepare_duplex_sampling(logits, metadata, (row,))
    state = model._minicpmo45_duplex_data_plane_helper.sessions["request"]
    generator = metadata.generators[0]
    for seed in range(20):
        model._minicpmo45_duplex_row_sampling_host = {}
        state.generated_tokens = [100]
        generator.manual_seed(seed)
        expected = model._sample_gander_dialogue_row(logits, metadata, row_idx=0, token_ids=ids)
        expected_rng, expected_history = generator.get_state(), list(state.generated_tokens)
        model.prepare_duplex_sampling(logits, metadata, (row,))
        state.generated_tokens = [100]
        generator.manual_seed(seed)
        assert model._sample_gander_dialogue_row(logits, metadata, row_idx=0, token_ids=ids) == expected
        assert torch.equal(generator.get_state(), expected_rng)
        assert state.generated_tokens == expected_history


def test_gander_host_missing_snapshot_keeps_release_defaults(gander_host_sampler, ids, mocker):
    import torch

    model = gander_host_sampler
    metadata = _gander_host_metadata([(0.7, 20, 0.8)], [[ids["speak_token_id"]]])
    metadata.temperature = metadata.top_k = metadata.top_p = torch.empty(0)
    filtered = mocker.spy(model, "_top_k_top_p_filter")
    logits = torch.full((1, 128), float("-inf"))
    logits[0, 100] = 1.0
    assert model._sample_gander_dialogue_row(logits, metadata, row_idx=0, token_ids=ids) == 100
    assert filtered.call_args.kwargs == {"top_k": 20, "top_p": 0.8}
    assert torch.isclose(filtered.call_args.args[0][0, 100], torch.tensor(1.0 / 0.7))


def test_gander_host_mixed_chat_batch_keeps_peer_sample_and_rng(gander_host_sampler, ids, mocker):
    import torch
    from vllm.v1.outputs import SamplerOutput

    from vllm_omni.model_executor.duplex_sampling import DuplexSamplingRow

    model = gander_host_sampler
    metadata = _gander_host_metadata([(0.75, 5, 0.875), (0.0, 5, 0.875)], [[], [ids["speak_token_id"]]])
    row = DuplexSamplingRow(1, "request", "session", 1, {}, 11, temperature=0.0, top_k=5, top_p=0.875)
    logits = torch.full((2, 128), float("-inf"))
    logits[:, 100] = 20.0
    model.prepare_duplex_sampling(logits, metadata, (row,))
    before = [generator.get_state().clone() for generator in metadata.generators.values()]

    def standard_sample(values, sampling):
        values.zero_()
        for generator in sampling.generators.values():
            torch.rand((), generator=generator)
        return SamplerOutput(sampled_token_ids=torch.tensor([[101], [102]], dtype=torch.int32), logprobs_tensors=None)

    mocker.patch.object(model, "sampler", create=True, side_effect=standard_sample)
    mocker.patch.object(
        model, "_sampling_metadata_value", side_effect=AssertionError("Gander read a device sampling scalar")
    )
    output = model.sample(logits, metadata)
    assert output.sampled_token_ids.tolist() == [[101], [100]]
    assert not torch.equal(metadata.generators[0].get_state(), before[0])
    assert torch.equal(metadata.generators[1].get_state(), before[1])
    assert logits[1, 100] == 20.0


def test_first_unit_allows_only_dialogue_actions(ids):
    allow, values = dialogue_constraint([], ids)
    assert allow
    assert values == {
        ids[key] for key in ("listen_token_id", "speak_token_id", "backchannel_token_id", "interrupt_token_id")
    }
    assert ids["tool_call_token_id"] not in values


@pytest.mark.parametrize("action", ["speak_token_id", "backchannel_token_id"])
def test_eight_lexical_tokens_then_boundary(ids, action):
    allow, values = dialogue_constraint([ids[action], *range(100, 107)], ids)
    assert not allow
    assert ids["interrupt_token_id"] in values
    assert ids["tool_call_token_id"] in values
    allow, values = dialogue_constraint([ids[action], *range(100, 108)], ids)
    assert allow
    assert values == {ids["chunk_eos_token_id"], ids["turn_eos_token_id"]}


def test_turn_eos_cannot_emit_stale_text(ids):
    assert dialogue_constraint([ids["speak_token_id"], 100, ids["turn_eos_token_id"]], ids) == (
        True,
        {ids["chunk_eos_token_id"]},
    )


def test_gander_codec_budget_does_not_stop_first_unit_early():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _native_duplex_chunk_budget

    assert _native_duplex_chunk_budget({"gander_speech_tokens": 50, "turn_start": True}) == (51, 50)
    assert _native_duplex_chunk_budget({"gander_speech_tokens": 50, "turn_end": True}) == (4096, 0)
    assert _native_duplex_chunk_budget({"turn_start": True}) == (26, 0)


def test_interrupt_bypasses_talker(ids):
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin

    completion = SimpleNamespace(stop_reason=ids["interrupt_token_id"], token_ids=[ids["interrupt_token_id"]])
    decision = MiniCPMO45DuplexPlugin(lambda *args: None).decide_output(
        stage_id=0,
        final_stage_id=2,
        segment_finished=True,
        segment_token_ids=tuple(completion.token_ids),
        segment_output_metadata={"special_token_ids": ids},
        output=SimpleNamespace(outputs=[completion]),
    )
    assert decision is not None
    assert decision.metadata["duplex_native_decision"] == "interrupt"
    assert decision.ends_model_turn is True


def test_compose_release_preserves_separate_weight_sets(tmp_path):
    import json

    import torch
    from safetensors.torch import save_file

    from vllm_omni.model_executor.models.minicpmo_4_5.gander import compose_release

    source = tmp_path / "release"
    thinker = source / "thinker"
    talker = source / "talker"
    thinker.mkdir(parents=True)
    (talker / "assets").mkdir(parents=True)
    (talker / "assets" / "ref_audio.wav").write_bytes(b"reference")
    (thinker / "config.json").write_text(json.dumps({"version": "4.5"}))
    (thinker / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"llm.weight": "model.safetensors"}})
    )
    save_file({"llm.weight": torch.ones(2)}, thinker / "model.safetensors")
    save_file({"tts.weight": torch.ones(2)}, talker / "model.safetensors")
    (talker / "talker_config.json").write_text(json.dumps({"weights_are_complete": True, "tts_config": {}}))
    (source / "release_manifest.json").write_text(
        json.dumps(
            {
                "unit_contract": {"text_tokens_per_speak_unit": 8, "speech_tokens_per_speak_unit": 50},
                "components": {"thinker": {"parameter_bytes": 8}, "talker": {"parameter_bytes": 8}},
            }
        )
    )
    destination = compose_release(source, tmp_path / "composed")
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {"llm.weight": "model.safetensors", "tts.weight": "gander-talker.safetensors"}
    assert json.loads((destination / "config.json").read_text())["gander_unit8"] is True
    assert "gander_unit8" not in json.loads((thinker / "config.json").read_text())
    assert (destination / "assets" / "HT_ref_audio.wav").read_bytes() == b"reference"
    with pytest.raises(FileExistsError):
        compose_release(source, destination)


def test_gander_sampler_keeps_controls_out_of_speech(ids):
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.config = SimpleNamespace(gander_unit8=True)
    metadata = SimpleNamespace(all_greedy=True, output_token_ids=[[]], temperature=torch.tensor([0.0]))
    batch_logits = torch.zeros(2, 128)
    logits = batch_logits[:1]
    logits[0, 100] = 100
    logits[0, ids["tool_call_token_id"]] = 99
    logits[0, ids["speak_token_id"]] = 90
    original_logits = batch_logits.clone()
    assert model._sample_gander_dialogue_row(logits, metadata, row_idx=0, token_ids=ids) == ids["speak_token_id"]
    assert torch.equal(batch_logits, original_logits), "grammar masks must not change shared batch logits"
    metadata.output_token_ids = [[ids["speak_token_id"]]]
    assert model._sample_gander_dialogue_row(logits, metadata, row_idx=0, token_ids=ids) == 100
    metadata.output_token_ids = [[ids["listen_token_id"], ids["speak_token_id"], *range(100, 108)]]
    logits[0, ids["turn_eos_token_id"]] = 50
    assert model._sample_gander_dialogue_row(logits, metadata, row_idx=0, token_ids=ids) == ids["turn_eos_token_id"]


@pytest.mark.parametrize(
    "listen_logit,speak_logit,expected", [(10.0, 9.8, "speak_token_id"), (-9.0, -8.8, "listen_token_id")]
)
def test_gander_actions_use_native_cross_unit_repetition_penalty(ids, listen_logit, speak_logit, expected):
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    state = SimpleNamespace(generated_tokens=[ids["listen_token_id"]], gander_tools_enabled=False)
    model._minicpmo45_duplex_state_for_row = lambda row: state
    model._minicpmo45_tokenizer = lambda: SimpleNamespace(eos_token_id=None)
    metadata = SimpleNamespace(all_greedy=True, output_token_ids=[[]])
    logits = torch.full((1, 128), -100.0)
    logits[0, ids["listen_token_id"]] = listen_logit
    logits[0, ids["speak_token_id"]] = speak_logit
    before = logits.clone()
    # Released StreamDecoder divides repeated logits of either sign by 1.05.
    assert model._sample_gander_dialogue_row(logits, metadata, row_idx=0, token_ids=ids) == ids[expected]
    assert torch.equal(logits, before)


def test_gander_sampling_records_action_history_across_units(ids):
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    state = SimpleNamespace(generated_tokens=[], gander_tools_enabled=False)
    model._minicpmo45_duplex_state_for_row = lambda row: state
    model._minicpmo45_tokenizer = lambda: SimpleNamespace(eos_token_id=None)
    metadata = SimpleNamespace(all_greedy=True, output_token_ids=[[]])
    logits = torch.zeros(1, 128)
    logits[0, ids["listen_token_id"]] = 20
    for _ in range(MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE + 1):
        assert model._sample_gander_dialogue_row(logits, metadata, row_idx=0, token_ids=ids) == ids["listen_token_id"]
    assert state.generated_tokens == [ids["listen_token_id"]] * MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE


def test_gander_forced_scheduler_history_does_not_penalize_first_sample(ids):
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    state = SimpleNamespace(generated_tokens=[], gander_tools_enabled=False)
    model._minicpmo45_duplex_state_for_row = lambda row: state
    model._minicpmo45_tokenizer = lambda: SimpleNamespace(eos_token_id=None)
    metadata = SimpleNamespace(all_greedy=True, output_token_ids=[[ids["listen_token_id"]]])
    logits = torch.full((1, 128), -100.0)
    logits[0, ids["listen_token_id"]] = 10
    logits[0, ids["speak_token_id"]] = 9.8
    assert model._sample_gander_dialogue_row(logits, metadata, row_idx=0, token_ids=ids) == ids["listen_token_id"]
    assert state.generated_tokens == [ids["listen_token_id"]]


def test_gander_chunk_boundaries_and_replay_do_not_add_repetition_history(ids):
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    state = SimpleNamespace(generated_tokens=[100], gander_tools_enabled=False)
    model._minicpmo45_duplex_state_for_row = lambda row: state
    model._minicpmo45_tokenizer = lambda: SimpleNamespace(eos_token_id=None)
    metadata = SimpleNamespace(all_greedy=True, output_token_ids=[[ids["speak_token_id"], *range(100, 108)]])
    logits = torch.zeros(1, 128)
    logits[0, ids["chunk_eos_token_id"]] = 20
    assert model._sample_gander_dialogue_row(logits, metadata, row_idx=0, token_ids=ids) == ids["chunk_eos_token_id"]
    model._minicpmo45_duplex_row_payloads = {
        0: {"gander_replay": True, "gander_replay_output_ids": [ids["listen_token_id"]]}
    }
    assert model._sample_gander_dialogue_row(logits, metadata, row_idx=0, token_ids=ids) == ids["listen_token_id"]
    assert state.generated_tokens == [100]


@pytest.mark.parametrize("boundary_wins", [False, True])
def test_gander_boundary_is_decided_before_repetition_penalty(ids, boundary_wins):
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    state = SimpleNamespace(generated_tokens=[100], gander_tools_enabled=False)
    model._minicpmo45_duplex_state_for_row = lambda row: state
    model._minicpmo45_tokenizer = lambda: SimpleNamespace(eos_token_id=None)
    metadata = SimpleNamespace(all_greedy=True, output_token_ids=[[ids["speak_token_id"]]])
    logits = torch.full((1, 128), -100.0)
    logits[0, 100] = 10
    logits[0, ids["chunk_eos_token_id"]] = 10.1 if boundary_wins else 9.8
    # In the released decoder, EOS wins the unpenalized first draw or is
    # removed from the content draw. Penalizing 10 / 1.05 must not create EOS.
    expected = ids["chunk_eos_token_id"] if boundary_wins else 100
    assert model._sample_gander_dialogue_row(logits, metadata, row_idx=0, token_ids=ids) == expected
    assert state.generated_tokens == ([100] if boundary_wins else [100, 100])


def test_gander_boundary_draw_uses_raw_probability_and_request_generator(ids, mocker):
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    state = SimpleNamespace(generated_tokens=[100], gander_tools_enabled=False)
    model._minicpmo45_duplex_state_for_row = lambda row: state
    model._minicpmo45_tokenizer = lambda: SimpleNamespace(eos_token_id=None)
    generator = torch.Generator().manual_seed(7346)
    metadata = SimpleNamespace(
        all_greedy=False,
        output_token_ids=[[], [ids["speak_token_id"]]],
        temperature=0.1,
        top_k=1,
        top_p=0.1,
        generators={1: generator},
    )
    logits = torch.full((1, 128), float("-inf"))
    logits[0, 100] = 10
    logits[0, ids["chunk_eos_token_id"]] = 9.8
    # Raw P(EOS) is about .45. Temperature/top-k would make it effectively
    # zero; the content repetition penalty would instead make it dominant.
    draw = mocker.patch("torch.rand", return_value=torch.tensor(0.4))
    assert model._sample_gander_dialogue_row(logits, metadata, row_idx=1, token_ids=ids) == ids["chunk_eos_token_id"]
    draw.assert_called_once_with((), generator=generator, device=logits.device)
    assert state.generated_tokens == [100]


def test_gander_final_unit_drains_remaining_context():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _native_duplex_chunk_budget

    assert _native_duplex_chunk_budget({"gander_speech_tokens": 50, "turn_end": True}, remaining_context=700) == (
        700,
        0,
    )


@pytest.mark.parametrize("choice", ["auto", "none"])
def test_tool_choice_reaches_stage0_sampler(ids, choice):
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    helper = object.__new__(MiniCPMO45Stage0DuplexRuntime)
    helper._stage_runtime_ready = lambda: True
    helper._require_special_token_ids = lambda: None
    helper._decode_ref_audio_from_session_config = lambda config: None
    helper._encode_text = lambda text: []
    state = _MiniCPMO45Stage0SessionState(session_id="s")
    helper._prepare_session_context(state, {}, runtime_config={"gander_tools": [{}], "gander_tool_choice": choice})
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.config = SimpleNamespace(gander_unit8=True)
    model._minicpmo45_duplex_state_for_row = lambda row: state
    model._minicpmo45_tokenizer = lambda: SimpleNamespace(eos_token_id=None)
    sampling = SimpleNamespace(all_greedy=True, output_token_ids=[[]])
    logits = torch.zeros(1, 128)
    logits[0, ids["tool_call_token_id"]] = 100
    logits[0, ids["speak_token_id"]] = 90
    sampled = model._sample_gander_dialogue_row(logits, sampling, row_idx=0, token_ids=ids)
    assert sampled == ids["tool_call_token_id" if choice == "auto" else "speak_token_id"]


def test_gander_closes_turn_and_unit_before_next_audio(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import _apply_default_scheduler_policy
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy

    names = [*MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS.values(), *CONTROL_TOKENS.values()]
    token_ids = {token: index + 1 for index, token in enumerate(names)}
    tokenizer = SimpleNamespace(unk_token_id=0, convert_tokens_to_ids=lambda token: token_ids.get(token, 0))
    config = SimpleNamespace(max_tokens=20, temperature=0.7)
    runtime: dict = {"gander_enabled": True}
    _apply_default_scheduler_policy(runtime, config=config, tokenizer=tokenizer)
    stop = runtime["duplex_stage_sampling_params"]["0"]["stop_token_ids"]
    assert token_ids["<|turn_eos|>"] not in stop
    assert token_ids["<|chunk_eos|>"] in stop
    assert token_ids["<|interrupt|>"] in stop
    base: dict = {}
    _apply_default_scheduler_policy(base, config=config, tokenizer=tokenizer)
    base_stop = base["duplex_stage_sampling_params"]["0"]["stop_token_ids"]
    assert token_ids["<|turn_eos|>"] not in base_stop
    assert token_ids["<|interrupt|>"] not in base_stop


def test_identical_text_in_distinct_units_is_not_dropped_as_replay():
    from types import SimpleNamespace

    from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
        _native_duplex_segment_output_ids,
    )

    context = SimpleNamespace(bridge_states={"duplex": {"model_turn_id": 0}})
    first, _, starts = _native_duplex_segment_output_ids([4, 5], "same", context, request_id="req", unit_seq=1)
    assert first == [4, 5] and starts
    context.bridge_states["minicpmo45_tts_handoff"]["condition_seq"] = 0
    second, _, starts = _native_duplex_segment_output_ids([4, 5], "same", context, request_id="req", unit_seq=2)
    assert second == [4, 5] and not starts
