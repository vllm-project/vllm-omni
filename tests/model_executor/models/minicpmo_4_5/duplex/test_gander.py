# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.gander import CONTROL_TOKENS, dialogue_constraint

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def ids():
    names = ["listen_token_id", "speak_token_id", "chunk_eos_token_id", "turn_eos_token_id", *CONTROL_TOKENS]
    return {key: index for index, key in enumerate(names)}


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

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.runtime import MiniCPMO45DuplexRuntimeExtension

    completion = SimpleNamespace(stop_reason=ids["interrupt_token_id"], token_ids=[ids["interrupt_token_id"]])
    decision = MiniCPMO45DuplexRuntimeExtension().decide_output(
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
    logits = torch.zeros(1, 128)
    logits[0, 100] = 100
    logits[0, ids["tool_call_token_id"]] = 99
    logits[0, ids["speak_token_id"]] = 90
    assert (
        model._sample_minicpmo45_native_duplex_row(logits, metadata, row_idx=0, token_ids=ids) == ids["speak_token_id"]
    )
    metadata.output_token_ids = [[ids["speak_token_id"]]]
    assert model._sample_minicpmo45_native_duplex_row(logits, metadata, row_idx=0, token_ids=ids) == 100
    metadata.output_token_ids = [[ids["listen_token_id"], ids["speak_token_id"], *range(100, 108)]]
    logits[0, ids["turn_eos_token_id"]] = 50
    assert (
        model._sample_minicpmo45_native_duplex_row(logits, metadata, row_idx=0, token_ids=ids)
        == ids["turn_eos_token_id"]
    )


def test_gander_final_unit_drains_remaining_context():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _native_duplex_chunk_budget

    assert _native_duplex_chunk_budget({"gander_speech_tokens": 50, "turn_end": True}, remaining_context=700) == (
        700,
        0,
    )


def test_gander_closes_turn_and_unit_before_next_audio(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.adapter import MiniCPMO45NativeDuplexServingAdapter
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import MiniCPMO45DuplexPolicy

    names = [*MiniCPMO45DuplexPolicy.SPECIAL_TOKEN_FIELDS.values(), *CONTROL_TOKENS.values()]
    token_ids = {token: index + 1 for index, token in enumerate(names)}
    tokenizer = SimpleNamespace(unk_token_id=0, convert_tokens_to_ids=lambda token: token_ids.get(token, 0))
    monkeypatch.setattr(MiniCPMO45NativeDuplexServingAdapter, "_load_native_tokenizer", lambda _: tokenizer)
    stop = MiniCPMO45NativeDuplexServingAdapter._native_stage0_stop_token_ids(
        SimpleNamespace(hf_config=SimpleNamespace(gander_unit8=True))
    )
    assert token_ids["<|turn_eos|>"] not in stop
    assert token_ids["<|chunk_eos|>"] in stop
    assert token_ids["<|interrupt|>"] in stop
    base_stop = MiniCPMO45NativeDuplexServingAdapter._native_stage0_stop_token_ids(
        SimpleNamespace(hf_config=SimpleNamespace())
    )
    assert token_ids["<|turn_eos|>"] not in base_stop
    assert token_ids["<|interrupt|>"] not in base_stop
