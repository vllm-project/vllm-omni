# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CompilationMode

from vllm_omni.model_executor.models.indextts2.indextts2_talker import (
    IndexTTS2TalkerForConditionalGeneration,
    _normalize_indextts_checkpoint_key,
    _require_exact_indextts_prefill_span,
    resolve_indextts_conditioning_policy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_v25_conditioning_policy():
    policy = resolve_indextts_conditioning_policy("indextts2_5")

    assert policy.prefix_tokens == 3
    assert policy.uses_conformer_perceiver is False
    assert policy.uses_campplus_projection is True
    assert policy.uses_language_embedding is True


def test_v2_conditioning_policy_is_unchanged():
    policy = resolve_indextts_conditioning_policy("indextts2")

    assert policy.prefix_tokens == 34
    assert policy.uses_conformer_perceiver is True
    assert policy.uses_campplus_projection is False
    assert policy.uses_language_embedding is False


def test_v25_text_tokens_wrap_language_prefixed_ids_with_start_and_stop(
    monkeypatch,
):
    import vllm_omni.model_executor.models.indextts2.text_processing_v2_5 as text_processing

    monkeypatch.setattr(
        text_processing,
        "prepare_indextts25_text",
        lambda *args, **kwargs: ([58838, 42], 7),
    )
    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    talker.conditioning_policy = resolve_indextts_conditioning_policy("indextts2_5")
    talker.model_path = "/model"
    talker.config = SimpleNamespace(tokenizer_file="tokenizer.tiktoken")

    token_ids, language_id = talker._tokenize_text(
        "hello",
        torch.device("cpu"),
        lang="ja",
    )

    assert token_ids.tolist() == [[0, 58838, 42, 1]]
    assert language_id.tolist() == [7]


def test_v25_text_tokens_filter_existing_start_and_stop_ids(monkeypatch):
    import vllm_omni.model_executor.models.indextts2.text_processing_v2_5 as text_processing

    monkeypatch.setattr(
        text_processing,
        "prepare_indextts25_text",
        lambda *args, **kwargs: ([0, 58838, 1, 42, 0], 7),
    )
    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    talker.conditioning_policy = resolve_indextts_conditioning_policy("indextts2_5")
    talker.model_path = "/model"
    talker.config = SimpleNamespace(tokenizer_file="tokenizer.tiktoken")

    token_ids, language_id = talker._tokenize_text(
        "hello",
        torch.device("cpu"),
        lang="en",
    )

    assert token_ids.tolist() == [[0, 58838, 42, 1]]
    assert language_id.tolist() == [7]


def test_native_fsdp_checkpoint_keys_are_normalized_at_every_nesting_level():
    assert _normalize_indextts_checkpoint_key("_fsdp_wrapped_module.spk_emb_proj.weight") == "spk_emb_proj.weight"
    assert (
        _normalize_indextts_checkpoint_key("_fsdp_wrapped_module.mel_embedding._fsdp_wrapped_module.weight")
        == "mel_embedding.weight"
    )


def test_prefill_span_mismatch_fails_instead_of_padding_or_truncating():
    for actual_span_len in (9, 11):
        with pytest.raises(
            ValueError,
            match="model_type=indextts2_5.*lang=es.*text_token_count=6",
        ):
            _require_exact_indextts_prefill_span(
                actual_span_len=actual_span_len,
                scheduled_span_len=10,
                model_type="indextts2_5",
                lang="es",
                text_token_count=6,
            )


def test_matching_prefill_span_is_accepted():
    _require_exact_indextts_prefill_span(
        actual_span_len=10,
        scheduled_span_len=10,
        model_type="indextts2_5",
        lang="en",
        text_token_count=6,
    )


def _make_minimal_talker(
    monkeypatch,
    *,
    model_type,
    use_gpt_latent,
):
    import vllm_omni.model_executor.models.indextts2.indextts2_talker as talker_module

    seen = {}

    def fake_parallel_lm_head(vocab_size, hidden_size, *, bias=False, **_kwargs):
        seen["bias"] = bias
        return torch.nn.Linear(hidden_size, vocab_size, bias=bias)

    monkeypatch.setattr(talker_module, "ParallelLMHead", fake_parallel_lm_head)
    monkeypatch.setattr(
        talker_module,
        "make_layers",
        lambda *_args, **_kwargs: (
            0,
            1,
            torch.nn.ModuleList([torch.nn.Linear(8, 8)]),
        ),
    )
    monkeypatch.setattr(
        talker_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=lambda: True),
    )
    monkeypatch.setattr(talker_module, "get_speaker_cache", object)
    monkeypatch.setattr(
        talker_module,
        "LogitsProcessor",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        talker_module,
        "ConformerEncoder",
        lambda **_kwargs: torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        talker_module,
        "PerceiverResampler",
        lambda *_args, **_kwargs: torch.nn.Linear(1, 1),
    )

    gpt = {
        "model_dim": 8,
        "layers": 0,
        "heads": 1,
        "max_mel_tokens": 4,
        "max_text_tokens": 4,
        "number_mel_codes": 16,
        "start_mel_token": 14,
        "stop_mel_token": 15,
        "number_text_tokens": 32,
        "condition_num_latent": 1,
        "condition_module": {},
        "emo_condition_module": {},
    }
    config = SimpleNamespace(
        gpt=gpt,
        model_type=model_type,
        use_gpt_latent=use_gpt_latent,
        gpt_checkpoint="gpt.pth",
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model="/model", hf_config=config),
        cache_config=None,
        quant_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )

    return (
        IndexTTS2TalkerForConditionalGeneration(vllm_config=vllm_config),
        seen,
    )


@pytest.mark.parametrize(
    ("model_type", "use_gpt_latent"),
    [
        ("indextts2", True),
        ("indextts2_5", False),
    ],
)
def test_mel_head_preserves_official_checkpoint_bias_and_hidden_payload_policy(
    monkeypatch,
    model_type,
    use_gpt_latent,
):
    talker, seen = _make_minimal_talker(
        monkeypatch,
        model_type=model_type,
        use_gpt_latent=use_gpt_latent,
    )

    assert seen["bias"] is True
    assert talker.mel_head.bias is not None
    assert talker.omni_pooler_payload_include_hidden is use_gpt_latent
    assert talker.omni_payload_at_request_end is (model_type == "indextts2_5")


def test_v2_load_weights_loads_exact_mel_head_bias_parameter(monkeypatch):
    import vllm_omni.model_executor.models.indextts2.indextts2_talker as talker_module

    talker, _ = _make_minimal_talker(
        monkeypatch,
        model_type="indextts2",
        use_gpt_latent=True,
    )
    checkpoint = {
        name: parameter.detach().clone() for name, parameter in talker.named_parameters(remove_duplicate=False)
    }
    expected_bias = torch.arange(
        talker.number_mel_codes,
        dtype=talker.mel_head.bias.dtype,
    )
    checkpoint["mel_head.bias"] = expected_bias
    with torch.no_grad():
        talker.mel_head.bias.zero_()

    monkeypatch.setattr(
        talker_module,
        "resolve_model_file",
        lambda *_args, **_kwargs: "/model/gpt.pth",
    )
    monkeypatch.setattr(
        talker_module.torch,
        "load",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        talker_module,
        "is_pp_missing_parameter",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        talker,
        "_warmup_external_models",
        lambda _device: None,
    )

    loaded = talker.load_weights([])

    assert "mel_head.bias" in loaded
    torch.testing.assert_close(talker.mel_head.bias, expected_bias)


def test_compute_logits_passes_mel_head_bias_to_logits_processor():
    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.mel_head = torch.nn.Linear(3, 4, bias=True)
    hidden_states = torch.zeros(1, 3)
    seen = {}

    def fake_logits_processor(head, hidden, **kwargs):
        seen["head"] = head
        seen["hidden"] = hidden
        seen.update(kwargs)
        return hidden

    talker.logits_processor = fake_logits_processor

    result = talker.compute_logits(hidden_states)

    assert result is hidden_states
    assert seen["head"] is talker.mel_head
    assert seen["hidden"] is hidden_states
    assert seen["embedding_bias"] is talker.mel_head.bias


def _make_cached_v25_prefill_talker(monkeypatch):
    talker, _ = _make_minimal_talker(
        monkeypatch,
        model_type="indextts2_5",
        use_gpt_latent=False,
    )
    talker._speaker_cache = SimpleNamespace(
        make_cache_key=lambda *_args: "cached-speaker",
        get=lambda _key: {
            "S_ref": torch.ones(1, 3, 8),
            "ref_mel": torch.ones(1, 80, 3),
            "style": torch.ones(1, 192),
            "spk_cond_emb": torch.ones(1, 3, 8),
        },
    )
    monkeypatch.setattr(
        talker,
        "_compute_emotion_vector",
        lambda **_kwargs: torch.zeros(1, talker.model_dim),
    )
    monkeypatch.setattr(
        talker,
        "_tokenize_text",
        lambda *_args, **_kwargs: (torch.tensor([[1]]), None),
    )
    return talker


def test_prefill_preserves_duration_factor_in_metadata(monkeypatch):
    talker = _make_cached_v25_prefill_talker(monkeypatch)

    _, _, info_update = talker._preprocess_prefill(
        torch.zeros(5, dtype=torch.long),
        None,
        {
            "text": ["hello"],
            "voice": ["/speaker.wav"],
            "voice_name": ["cached"],
            "duration_factor": [0.5],
        },
        span_len=5,
    )

    assert info_update["meta"]["duration_factor"] == 0.5


@pytest.mark.parametrize("duration_factor", [0.49, 2.01])
def test_prefill_rejects_duration_factor_outside_supported_range(monkeypatch, duration_factor):
    talker = _make_cached_v25_prefill_talker(monkeypatch)

    with pytest.raises(
        ValueError,
        match="^IndexTTS 2.5 duration_factor must be between 0.5 and 2.0$",
    ):
        talker._preprocess_prefill(
            torch.zeros(5, dtype=torch.long),
            None,
            {
                "text": ["hello"],
                "voice": ["/speaker.wav"],
                "voice_name": ["cached"],
                "duration_factor": [duration_factor],
            },
            span_len=5,
        )


def test_no_latent_sampled_token_payload_emits_only_first_step_metadata():
    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    talker.use_gpt_latent = False
    hidden = torch.zeros(1, 8)
    info = {
        "codes": {"mel": torch.tensor([17])},
        "meta": {
            "mel_code_count": 1,
            "S_ref": torch.ones(1, 4),
            "ref_mel": torch.ones(1, 80, 3),
            "style": torch.ones(1, 192),
            "duration_factor": 0.5,
        },
    }

    first_decode_output = talker.make_omni_output(
        hidden,
        model_intermediate_buffer=[info],
    )

    assert "codes" not in first_decode_output.multimodal_outputs
    assert "hidden_states" not in first_decode_output.multimodal_outputs
    assert first_decode_output.multimodal_outputs["meta"][0]["S_ref"] is info["meta"]["S_ref"]
    assert first_decode_output.multimodal_outputs["meta"][0]["duration_factor"] == 0.5

    later = talker.make_omni_output(
        hidden,
        model_intermediate_buffer=[
            {
                "codes": {"mel": torch.tensor([23])},
                "meta": {"mel_code_count": 2},
            }
        ],
    )
    assert later.multimodal_outputs == {"meta": [{"use_gpt_latent": False}]}


def test_first_decode_payload_defaults_missing_duration_factor_to_one():
    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    talker.use_gpt_latent = False
    hidden = torch.zeros(1, 8)
    info = {
        "codes": {"mel": torch.tensor([17])},
        "meta": {
            "mel_code_count": 1,
            "S_ref": torch.ones(1, 4),
            "ref_mel": torch.ones(1, 80, 3),
            "style": torch.ones(1, 192),
        },
    }

    output = talker.make_omni_output(
        hidden,
        model_intermediate_buffer=[info],
    )

    assert output.multimodal_outputs["meta"][0]["duration_factor"] == 1.0


def test_latent_sampled_token_payload_keeps_aligned_gpu_deltas():
    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    talker.use_gpt_latent = True
    hidden = torch.zeros(1, 8)
    info = {
        "codes": {"mel": torch.tensor([17])},
        "meta": {
            "mel_code_count": 1,
            "latent_acc": torch.arange(8, dtype=torch.float32).reshape(1, 8),
            "duration_factor": 1.0,
        },
    }

    output = talker.make_omni_output(
        hidden,
        model_intermediate_buffer=[info],
    )

    assert output.multimodal_outputs["codes"]["mel"][0].tolist() == [[17]]
    assert output.multimodal_outputs["hidden_states"]["latent"][0].tolist() == [list(range(8))]


def test_v25_default_emotion_reuses_speaker_w2v_features(monkeypatch):
    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    speaker_features = torch.zeros(1, 4, 1024)
    expected = torch.ones(1, 1280)
    seen = {}

    def fake_merge(spk_cond_emb, emo_cond_emb, alpha):
        seen["spk"] = spk_cond_emb
        seen["emo"] = emo_cond_emb
        seen["alpha"] = alpha
        return expected

    monkeypatch.setattr(talker, "_merge_emovec", fake_merge)

    result = talker._compute_emotion_vector(
        wav_16k=None,
        speaker_audio_path="/speaker.wav",
        emo_audio_path=None,
        emo_voice_name=None,
        main_text="hello",
        use_emo_text=False,
        emo_text=None,
        emo_vector=None,
        emo_alpha=0.25,
        use_random=False,
        style=torch.zeros(1, 192),
        spk_cond_emb=speaker_features,
        w2v_model=None,
        w2v_proc=None,
        device=torch.device("cpu"),
    )

    assert result is expected
    assert seen == {
        "spk": speaker_features,
        "emo": speaker_features,
        "alpha": 1.0,
    }
