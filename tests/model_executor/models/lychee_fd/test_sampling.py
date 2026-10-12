# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.lychee_fd.sampling import (
    LycheeControlMode,
    sample_control_tokens,
    sample_speech_tokens,
    update_control_modes,
    update_speaking_steps,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        control_token_chunk_size=10,
        sleep_token_id=30,
        detect_token_id=31,
        start_speaking_token_id=32,
        start_listening_token_id=33,
        keep_listening_token_id=34,
        keep_speaking_token_id=35,
        start_bc_token_id=36,
        stoken_pad_token_id=10,
        stoken_delay_token_id=11,
        stoken_delay_num=2,
        tts_start_token_id=12,
        tts_end_token_id=13,
        stoken_audio_token_id_min=15,
        stoken_token_ids_max=20,
        stoken_do_sample=True,
        stoken_temperature=0.7,
        stoken_top_k=0,
        stoken_top_p=1.0,
        stoken_no_repeat_ngram_size=4,
        stoken_max_tokens=1000,
    )


def test_control_grammar_forces_sleep_detect_and_listening_decision() -> None:
    config = _config()
    logits = torch.zeros((3, 40))
    logits[:, 39] = 100
    logits[2, config.start_speaking_token_id] = 4
    logits[2, config.keep_listening_token_id] = 3
    sampled = sample_control_tokens(
        logits,
        modes=torch.full((3,), int(LycheeControlMode.LISTENING)),
        ticks=torch.tensor([0, 8, 9]),
        config=config,
        allowing_backchannel=False,
    )
    assert sampled.tolist() == [config.sleep_token_id, config.detect_token_id, config.start_speaking_token_id]


def test_control_decision_uses_mode_specific_legal_set() -> None:
    config = _config()
    logits = torch.zeros((2, 40))
    logits[:, config.start_speaking_token_id] = 100
    logits[:, config.start_listening_token_id] = 3
    logits[:, config.keep_speaking_token_id] = 2
    sampled = sample_control_tokens(
        logits,
        modes=torch.tensor([int(LycheeControlMode.SPEAKING), int(LycheeControlMode.BACKCHANNEL)]),
        ticks=torch.tensor([9, 9]),
        config=config,
        allowing_backchannel=False,
    )
    assert sampled.tolist() == [config.start_listening_token_id, config.start_speaking_token_id]


def test_control_transitions_are_row_local() -> None:
    config = _config()
    modes = torch.tensor(
        [
            int(LycheeControlMode.LISTENING),
            int(LycheeControlMode.SPEAKING),
            int(LycheeControlMode.BACKCHANNEL),
        ]
    )
    updated = update_control_modes(
        modes,
        torch.tensor(
            [
                config.start_speaking_token_id,
                config.start_listening_token_id,
                config.start_bc_token_id,
            ]
        ),
        config=config,
    )
    assert updated.tolist() == [
        int(LycheeControlMode.SPEAKING),
        int(LycheeControlMode.LISTENING),
        int(LycheeControlMode.BACKCHANNEL),
    ]


def test_speech_prefix_is_pad_then_delay_start_and_audio() -> None:
    config = _config()
    logits = torch.zeros((1, 40))
    logits[:, 17] = 100

    def sample(mode: LycheeControlMode, step: int) -> int:
        generator = torch.Generator().manual_seed(0)
        return int(
            sample_speech_tokens(
                logits,
                modes=torch.tensor([int(mode)]),
                speaking_steps=torch.tensor([step]),
                config=config,
                generator=generator,
                temperature=0,
            )[0]
        )

    assert sample(LycheeControlMode.LISTENING, -1) == config.stoken_pad_token_id
    assert sample(LycheeControlMode.SPEAKING, 0) == config.stoken_delay_token_id
    assert sample(LycheeControlMode.SPEAKING, 1) == config.stoken_delay_token_id
    assert sample(LycheeControlMode.SPEAKING, 2) == config.tts_start_token_id
    assert sample(LycheeControlMode.SPEAKING, 3) == 17


def test_speech_sampling_applies_top_k_and_request_local_ngram_block() -> None:
    config = _config()
    config.stoken_top_k = 1
    logits = torch.zeros((1, 40))
    logits[:, 18] = 10
    logits[:, 19] = 9
    history = torch.tensor([[15, 16, 17, 18, 15, 16, 17, -1]])

    sampled = sample_speech_tokens(
        logits,
        modes=torch.tensor([int(LycheeControlMode.SPEAKING)]),
        speaking_steps=torch.tensor([7]),
        config=config,
        generator=torch.Generator().manual_seed(0),
        speech_history=history,
        history_lengths=torch.tensor([7]),
    )

    assert sampled.tolist() == [19]


def test_speech_sampling_forces_end_at_released_maximum() -> None:
    config = _config()
    config.stoken_max_tokens = 3
    logits = torch.zeros((1, 40))
    logits[:, 17] = 100
    sampled = sample_speech_tokens(
        logits,
        modes=torch.tensor([int(LycheeControlMode.SPEAKING)]),
        speaking_steps=torch.tensor([config.stoken_delay_num + config.stoken_max_tokens]),
        config=config,
        generator=torch.Generator().manual_seed(0),
        temperature=0,
    )
    assert sampled.tolist() == [config.tts_end_token_id]


def test_speaking_step_cursor_starts_after_transition_and_resets_on_listen() -> None:
    old_modes = torch.tensor([int(LycheeControlMode.LISTENING), int(LycheeControlMode.SPEAKING)])
    new_modes = torch.tensor([int(LycheeControlMode.SPEAKING), int(LycheeControlMode.LISTENING)])
    updated = update_speaking_steps(
        torch.tensor([-1, 7]),
        old_modes=old_modes,
        new_modes=new_modes,
    )
    assert updated.tolist() == [0, -1]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_speech_logits_fail_fast_without_sanitizing_or_fallback(bad):
    config = _config()
    logits = torch.zeros((1, 40))
    logits[0, 17] = bad
    with pytest.raises(FloatingPointError, match="Non-finite"):
        sample_speech_tokens(
            logits,
            modes=torch.tensor([0]),
            speaking_steps=torch.tensor([-1]),
            config=config,
            generator=torch.Generator(),
        )


def test_control_logits_fail_fast():
    config = _config()
    logits = torch.zeros((1, 40))
    logits[0, config.sleep_token_id] = float("nan")
    with pytest.raises(FloatingPointError, match="control logits"):
        sample_control_tokens(logits, modes=torch.tensor([0]), ticks=torch.tensor([0]), config=config)


def test_speech_never_samples_reserved_tokens_outside_actual_codec_codebook():
    config = _config()
    config.stoken_codec_vocab_size = 2
    logits = torch.zeros((1, 40))
    logits[0, 19] = 100
    logits[0, 16] = 10
    assert sample_speech_tokens(
        logits,
        modes=torch.tensor([1]),
        speaking_steps=torch.tensor([5]),
        config=config,
        generator=torch.Generator(),
        temperature=0,
    ).tolist() == [16]


def test_backchannel_decision_only_applies_start_speak_factor():
    config = _config()
    logits = torch.zeros((1, 40))
    logits[0, config.start_speaking_token_id] = 1
    logits[0, config.start_listening_token_id] = 1.1
    logits[0, config.start_bc_token_id] = 1.15
    # BC ignores the listening BC bias and speaking SL multiplier.
    result = sample_control_tokens(
        logits,
        modes=torch.tensor([2]),
        ticks=torch.tensor([9]),
        config=config,
        start_speak_token_factor=1.2,
        start_listen_token_factor=10,
        backchannel_token_bias=10,
    )
    assert result.tolist() == [config.start_speaking_token_id]


def test_backchannel_to_speaking_restarts_delay_cursor():
    assert update_speaking_steps(
        torch.tensor([27]), old_modes=torch.tensor([2]), new_modes=torch.tensor([1])
    ).tolist() == [0]


@pytest.mark.parametrize("mode,step", [(0, -1), (1, 0), (1, 1), (1, 2), (1, 1002)])
def test_forced_speech_tokens_do_not_advance_request_rng(mode, step):
    config = _config()
    generator = torch.Generator().manual_seed(91)
    untouched = torch.Generator().manual_seed(91)
    sample_speech_tokens(
        torch.zeros(1, 40),
        modes=torch.tensor([mode]),
        speaking_steps=torch.tensor([step]),
        config=config,
        generator=generator,
    )
    assert torch.rand((), generator=generator).item() == torch.rand((), generator=untouched).item()


def test_stochastic_codec_sample_advances_request_rng():
    config = _config()
    generator = torch.Generator().manual_seed(91)
    untouched = torch.Generator().manual_seed(91)
    sample_speech_tokens(
        torch.zeros(1, 40),
        modes=torch.tensor([1]),
        speaking_steps=torch.tensor([3]),
        config=config,
        generator=generator,
    )
    assert torch.rand((), generator=generator).item() != torch.rand((), generator=untouched).item()


def test_session_ngram_scatter_matches_authoritative_hf_processor_with_unsorted_candidates():
    from transformers import NoRepeatNGramLogitsProcessor

    from vllm_omni.model_executor.models.lychee_fd.sampling import _apply_no_repeat_ngram

    history = torch.tensor([[3, 14, 15, 16, 17, 13, 3, 3, 11, 12, 14, 15, 16]])
    candidates = torch.tensor([18, 13, 17, 14, 20])
    all_scores = torch.arange(30, dtype=torch.float32)[None, :]
    expected = NoRepeatNGramLogitsProcessor(4)(history, all_scores.clone())[:, candidates]
    actual = _apply_no_repeat_ngram(
        all_scores[:, candidates],
        candidate_ids=candidates,
        history=history,
        history_lengths=torch.tensor([history.shape[1]]),
        ngram_size=4,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
