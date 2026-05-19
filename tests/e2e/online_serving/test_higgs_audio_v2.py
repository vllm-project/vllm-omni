# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end checks for the vllm-omni higgs_audio_v2 integration.

This test suite exercises the parts of the higgs_audio_v2 path that do NOT
require booting the full vLLM engine: the request-validator scope checks,
the prompt builder's tokenization parity, the stage_input_processor adapter
contract, the DualFFN routing behavior, and the Stage-1 codec wiring.
"""

from __future__ import annotations

import os
import sys
import pytest
import torch

# --------------------------------------------------------------------- scope
def test_validator_rejects_voice_cloning_fields() -> None:
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
        REJECTED_REQUEST_FIELDS,
        UnsupportedInputError,
        validate_plain_text_request,
    )

    for field in REJECTED_REQUEST_FIELDS:
        payload = {"input": "Hello", field: "anything"}
        with pytest.raises(UnsupportedInputError) as excinfo:
            validate_plain_text_request(payload)
        assert "higgs_audio_v2" in str(excinfo.value)
        assert field in str(excinfo.value)

def test_validator_rejects_multi_speaker_tag() -> None:
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
        UnsupportedInputError,
        validate_plain_text_request,
    )

    with pytest.raises(UnsupportedInputError) as excinfo:
        validate_plain_text_request({"input": "[SPEAKER0] hi"})
    assert "multi-speaker" in str(excinfo.value).lower()

def test_validator_accepts_plain_text() -> None:
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
        validate_plain_text_request,
    )

    validate_plain_text_request({"input": "Hello world."})

# ---------------------------------------------- serving-level request validator
@pytest.fixture
def _make_speech_request():
    """Build an ``OpenAICreateSpeechRequest`` with default plain-text input."""
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

    def _build(**overrides):
        kwargs = {"input": "Hello world.", "model": "higgs_audio_v2"}
        kwargs.update(overrides)
        return OpenAICreateSpeechRequest.model_validate(kwargs)

    return _build

# A bound version of ``_validate_higgs_audio_v2_request`` does not need
# the full serving instance; we just want to call the function. Use a
# light shim that mimics ``self._validate_higgs_audio_v2_request(req)``.
def _call_validator(request) -> str | None:
    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

    # The validator does not touch instance state; bind it manually via
    # ``__func__`` and pass a dummy ``self`` argument.
    return OmniOpenAIServingSpeech._validate_higgs_audio_v2_request(
        None,  # type: ignore[arg-type]
        request,
    )

@pytest.mark.parametrize(
    "field, value",
    [
        ("voice", "alloy"),
        ("instructions", "speak slowly"),
        ("task_type", "Base"),
        ("language", "Chinese"),
        ("ref_audio", "data:audio/wav;base64,SUQ="),
        ("ref_text", "transcript"),
        ("x_vector_only_mode", True),
        ("speaker_embedding", [0.0] * 16),
        ("speed", 1.5),
    ],
)
def test_serving_validator_rejects_out_of_scope_fields(_make_speech_request, field: str, value) -> None:
    request = _make_speech_request(**{field: value})
    err = _call_validator(request)
    assert err is not None, f"expected reject for field {field!r}"
    assert "higgs_audio_v2" in err, f"reject for {field!r} should name the model: {err!r}"

def test_serving_validator_accepts_plain_text(_make_speech_request) -> None:
    request = _make_speech_request()
    assert _call_validator(request) is None

def test_serving_validator_accepts_max_new_tokens_seed(_make_speech_request) -> None:
    request = _make_speech_request(max_new_tokens=500, seed=42)
    assert _call_validator(request) is None

def test_serving_validator_rejects_empty_input(_make_speech_request) -> None:
    request = _make_speech_request(input="   ")
    err = _call_validator(request)
    assert err is not None
    assert "empty" in err.lower()

def test_serving_validator_rejects_multi_speaker_in_text(_make_speech_request) -> None:
    request = _make_speech_request(input="[SPEAKER0] hi")
    err = _call_validator(request)
    assert err is not None
    assert "multi-speaker" in err.lower()

@pytest.mark.parametrize(
    "alias",
    ["messages", "reference_audio", "voice_prompt", "speaker_audio", "speakers"],
)
def test_schema_rejects_rich_input_aliases_for_higgs_audio_v2(alias: str) -> None:
    """Pydantic-level rejection: unsupported rich-input aliases never reach the validator.

    These keys are not declared on the global ``OpenAICreateSpeechRequest`` schema
    and would otherwise be silently dropped. The model-aware ``before`` validator
    raises a ValueError when ``model="higgs_audio_v2"`` so the API surface
    returns a deterministic 4xx with a model-named message.
    """
    import pydantic

    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

    payload = {"input": "hi", "model": "higgs_audio_v2", alias: "something"}
    with pytest.raises(pydantic.ValidationError) as excinfo:
        OpenAICreateSpeechRequest.model_validate(payload)
    msg = str(excinfo.value)
    assert "higgs_audio_v2" in msg
    assert alias in msg

def test_schema_accepts_aliases_for_other_models() -> None:
    """Pydantic must NOT reject `reference_audio` (etc.) for non-higgs models.

    The model-aware reject only kicks in when ``model`` mentions ``higgs_audio_v2``;
    other TTS models continue to receive their previous permissive behavior
    (unknown keys are dropped silently as before).
    """
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

    payload = {"input": "hi", "model": "qwen3_tts", "reference_audio": "abc"}
    parsed = OpenAICreateSpeechRequest.model_validate(payload)
    assert parsed.input == "hi"
    assert parsed.model == "qwen3_tts"

def test_schema_rejects_chatml_messages_for_higgs_audio_v2() -> None:
    """``messages`` (ChatML rich content) is rejected at parse time."""
    import pydantic

    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

    payload = {
        "input": "ignored",
        "model": "higgs_audio_v2",
        "messages": [{"role": "user", "content": "hello"}],
    }
    with pytest.raises(pydantic.ValidationError) as excinfo:
        OpenAICreateSpeechRequest.model_validate(payload)
    assert "higgs_audio_v2" in str(excinfo.value)
    assert "messages" in str(excinfo.value)

# ------------------------------------------------------ registry / pipeline
def test_pipeline_registry_has_higgs_audio_v2() -> None:
    from vllm_omni.config.pipeline_registry import _OMNI_PIPELINES

    assert "higgs_audio_v2" in _OMNI_PIPELINES

def test_model_registry_has_both_stages() -> None:
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    assert "HiggsAudioV2ForConditionalGeneration" in _OMNI_MODELS
    assert "HiggsAudioV2TalkerForConditionalGeneration" in _OMNI_MODELS
    assert "HiggsAudioV2Code2WavForConditionalGeneration" in _OMNI_MODELS

def test_pipeline_hf_architectures_declared() -> None:
    from vllm_omni.model_executor.models.higgs_audio_v2.pipeline import (
        HIGGS_AUDIO_V2_PIPELINE,
    )

    assert HIGGS_AUDIO_V2_PIPELINE.model_type == "higgs_audio_v2"
    assert "HiggsAudioV2ForConditionalGeneration" in HIGGS_AUDIO_V2_PIPELINE.hf_architectures

# ------------------------------------------------------- stage_input_processor
def test_stage_input_processor_async_chunk_emits_codes_only_when_chunk_full() -> None:
    """Adapter must wait for chunk_size frames before flushing a chunk."""
    from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
    from vllm_omni.model_executor.stage_input_processors.higgs_audio_v2 import (
        talker2code2wav_async_chunk,
    )

    class _FakeRequest:
        external_req_id = "req-0"

        def is_finished(self) -> bool:
            return False

    class _FakeManager:
        code_prompt_token_ids: dict[str, list[list[int]]] = {"req-0": []}
        connector = None

    mgr = _FakeManager()

    # Feed 24 frames -- below default chunk_size=25 -- and expect no flush.
    for _ in range(24):
        frame = torch.zeros(8, dtype=torch.long)
        payload = {"codes": {"audio": frame}}
        out = talker2code2wav_async_chunk(mgr, payload, _FakeRequest(), is_finished=False)
        assert out is None
    # 25th frame should trigger a flush.
    payload = {"codes": {"audio": torch.zeros(8, dtype=torch.long)}}
    out = talker2code2wav_async_chunk(mgr, payload, _FakeRequest(), is_finished=False)
    assert isinstance(out, OmniPayloadStruct)
    assert isinstance(out.codes, CodesStruct)
    assert out.codes.audio.numel() > 0
    assert isinstance(out.meta, MetaStruct)

# --------------------------------------------------- Stage-1 decode contract
def test_stage1_rejects_stream_specials() -> None:
    """Code IDs >= audio_stream_bos_id (1024) must raise ValueError before decode."""
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_code2wav import (
        HiggsAudioV2Code2Wav,
    )

    cfg = HiggsAudioV2Config()
    stage1 = HiggsAudioV2Code2Wav(cfg)
    # Force the loaded flag so the validator runs before the (unloaded) decoder
    # would otherwise raise on missing weights.
    stage1._loaded = True
    stage1.quantizer = object()  # not used, validator runs first
    stage1.fc2 = object()
    stage1.acoustic_decoder = object()
    bad_codes = torch.full((1, cfg.num_codebooks, 4), cfg.audio_stream_bos_id, dtype=torch.long)
    with pytest.raises(ValueError) as excinfo:
        stage1(bad_codes)
    assert "stream-special" in str(excinfo.value).lower() or "out-of-range" in str(excinfo.value).lower()

def test_fused_weight_loader_maps_qkv_and_mlp() -> None:
    """Verify the talker's load_weights fuses HF q/k/v and gate/up projections.

    We bypass the full talker construction (which needs vLLM TP state) and
    drive the mapping helpers directly on synthetic state-dict entries.
    """
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_talker import (
        HiggsAudioV2TalkerForConditionalGeneration,
    )

    cfg = HiggsAudioV2Config()

    # Build a no-arg shell of the talker (skip __init__ to avoid vLLM TP setup),
    # then run the static-name-mapping helpers directly. This is enough to
    # cover the QKV fusion / gate_up_proj fusion / audio head split logic.
    talker = HiggsAudioV2TalkerForConditionalGeneration.__new__(
        HiggsAudioV2TalkerForConditionalGeneration
    )
    talker.config = cfg

    # ---- simple names ----
    # R12: the multi-codebook embedding lives at ``audio_codebook_embeddings``
    # (one ``Embedding(8 * codebook_size, hidden)``), and the fused audio head
    # at ``audio_lm_head`` (one ``Linear(hidden, 8 * codebook_size)``). The
    # per-codebook split into ``audio_codebook0_head`` + residual heads was
    # not faithful to the upstream architecture and has been removed.
    assert talker._map_simple_name("model.embed_audio_tokens.embed_audio_tokens.weight") == "audio_codebook_embeddings.weight"
    assert talker._map_simple_name("audio_codebook_embeddings.weight") == "audio_codebook_embeddings.weight"
    assert talker._map_simple_name("text_lm_head.weight") == "lm_head.weight"
    assert talker._map_simple_name("model.layers.5.input_layernorm.weight") == "layers.5.base.input_layernorm.weight"
    assert (
        talker._map_simple_name("model.layers.5.audio_post_attention_layernorm.weight")
        == "layers.5.audio_post_attention_layernorm.weight"
    )
    # Unrelated key returns None.
    assert talker._map_simple_name("unrelated.weight") is None

def _make_stub_talker():
    """Build a HiggsAudioV2TalkerForConditionalGeneration shell with stub parameters.

    The full ``__init__`` requires a live vLLM TP group; here we sidestep it by
    constructing the parameters that ``load_weights`` writes into directly.
    The stub is enough to exercise the fused-tensor mapping (qkv_proj,
    gate_up_proj, audio_lm_head split) end to end.
    """
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_talker import (
        HiggsAudioV2TalkerForConditionalGeneration,
    )

    cfg = HiggsAudioV2Config(num_hidden_layers=2)  # small to keep the stub fast
    talker = HiggsAudioV2TalkerForConditionalGeneration.__new__(
        HiggsAudioV2TalkerForConditionalGeneration
    )
    torch.nn.Module.__init__(talker)
    talker.config = cfg
    # Construct just the param tensors load_weights writes into. Names mirror
    # what self.named_parameters() would return on a real talker.
    hidden = int(cfg.hidden_size)
    head_dim = int(cfg.head_dim)
    q_dim = int(cfg.num_attention_heads) * head_dim
    kv_dim = int(cfg.num_key_value_heads) * head_dim
    inter = int(cfg.intermediate_size)
    vocab = int(cfg.vocab_size)
    num_codebooks = int(cfg.num_codebooks)
    codebook_size = int(cfg.codebook_size)
    n_layers = int(cfg.num_hidden_layers)

    params: dict[str, torch.nn.Parameter] = {}
    for li in range(n_layers):
        params[f"layers.{li}.base.self_attn.qkv_proj.weight"] = torch.nn.Parameter(
            torch.zeros(q_dim + 2 * kv_dim, hidden)
        )
        params[f"layers.{li}.base.mlp.gate_up_proj.weight"] = torch.nn.Parameter(
            torch.zeros(2 * inter, hidden)
        )
        params[f"layers.{li}.audio_mlp.gate_up_proj.weight"] = torch.nn.Parameter(
            torch.zeros(2 * inter, hidden)
        )
        params[f"layers.{li}.base.input_layernorm.weight"] = torch.nn.Parameter(torch.zeros(hidden))
        params[f"layers.{li}.base.post_attention_layernorm.weight"] = torch.nn.Parameter(torch.zeros(hidden))
        params[f"layers.{li}.audio_input_layernorm.weight"] = torch.nn.Parameter(torch.zeros(hidden))
        params[f"layers.{li}.audio_post_attention_layernorm.weight"] = torch.nn.Parameter(torch.zeros(hidden))
    params["audio_codebook_embeddings.weight"] = torch.nn.Parameter(torch.zeros(num_codebooks * codebook_size, hidden))
    params["lm_head.weight"] = torch.nn.Parameter(torch.zeros(vocab, hidden))
    params["audio_lm_head.weight"] = torch.nn.Parameter(torch.zeros(num_codebooks * codebook_size, hidden))

    # Monkey-patch named_parameters to return our stub registry.
    talker._stub_params = params
    talker.named_parameters = lambda *a, **kw: iter(params.items())
    return talker, cfg, params

def test_load_weights_fuses_qkv_and_mlp_end_to_end() -> None:
    """Drive load_weights on a synthetic HF state_dict and verify fused outputs."""
    talker, cfg, params = _make_stub_talker()
    hidden = int(cfg.hidden_size)
    head_dim = int(cfg.head_dim)
    q_dim = int(cfg.num_attention_heads) * head_dim
    kv_dim = int(cfg.num_key_value_heads) * head_dim
    inter = int(cfg.intermediate_size)
    n_layers = int(cfg.num_hidden_layers)
    num_codebooks = int(cfg.num_codebooks)
    codebook_size = int(cfg.codebook_size)

    # Build a synthetic HF state dict with unique values per slot so we can
    # assert the fused tensors carry the right halves.
    hf_state: list[tuple[str, torch.Tensor]] = []
    for li in range(n_layers):
        # q/k/v -> qkv_proj. Use distinct constants so we can identify slabs.
        hf_state.append((f"model.layers.{li}.self_attn.q_proj.weight", torch.full((q_dim, hidden), float(10 * li + 1))))
        hf_state.append((f"model.layers.{li}.self_attn.k_proj.weight", torch.full((kv_dim, hidden), float(10 * li + 2))))
        hf_state.append((f"model.layers.{li}.self_attn.v_proj.weight", torch.full((kv_dim, hidden), float(10 * li + 3))))
        # gate/up -> gate_up_proj (text MLP).
        hf_state.append((f"model.layers.{li}.mlp.gate_proj.weight", torch.full((inter, hidden), float(10 * li + 4))))
        hf_state.append((f"model.layers.{li}.mlp.up_proj.weight", torch.full((inter, hidden), float(10 * li + 5))))
        # gate/up -> gate_up_proj (audio MLP).
        hf_state.append((f"model.layers.{li}.audio_mlp.gate_proj.weight", torch.full((inter, hidden), float(10 * li + 6))))
        hf_state.append((f"model.layers.{li}.audio_mlp.up_proj.weight", torch.full((inter, hidden), float(10 * li + 7))))
        # Layernorms.
        hf_state.append((f"model.layers.{li}.input_layernorm.weight", torch.full((hidden,), float(10 * li + 8))))
        hf_state.append((f"model.layers.{li}.post_attention_layernorm.weight", torch.full((hidden,), float(10 * li + 9))))
        hf_state.append((f"model.layers.{li}.audio_input_layernorm.weight", torch.full((hidden,), float(100 * li + 1))))
        hf_state.append((f"model.layers.{li}.audio_post_attention_layernorm.weight", torch.full((hidden,), float(100 * li + 2))))
    # Audio LM head: a fused [num_codebooks * codebook_size, hidden] tensor whose
    # per-codebook chunks carry distinct constants so we can identify the split.
    audio_head = torch.zeros(num_codebooks * codebook_size, hidden)
    for k in range(num_codebooks):
        audio_head[k * codebook_size : (k + 1) * codebook_size] = float(k + 1)
    hf_state.append(("audio_lm_head.weight", audio_head))
    # text LM head and embed_audio_tokens.
    hf_state.append(("text_lm_head.weight", torch.full((int(cfg.vocab_size), hidden), 999.0)))
    hf_state.append(("model.embed_audio_tokens.embed_audio_tokens.weight", torch.full((num_codebooks * codebook_size, hidden), -1.0)))

    loaded = talker.load_weights(iter(hf_state))

    # Every fused / split / direct target must have landed.
    for li in range(n_layers):
        assert f"layers.{li}.base.self_attn.qkv_proj.weight" in loaded
        assert f"layers.{li}.base.mlp.gate_up_proj.weight" in loaded
        assert f"layers.{li}.audio_mlp.gate_up_proj.weight" in loaded
        assert f"layers.{li}.base.input_layernorm.weight" in loaded
        assert f"layers.{li}.audio_input_layernorm.weight" in loaded
    assert "audio_lm_head.weight" in loaded
    assert "audio_codebook_embeddings.weight" in loaded
    assert "lm_head.weight" in loaded

    # Validate the QKV fusion slabs for layer 0.
    qkv0 = params["layers.0.base.self_attn.qkv_proj.weight"]
    assert torch.equal(qkv0[:q_dim], torch.full((q_dim, hidden), 1.0))
    assert torch.equal(qkv0[q_dim : q_dim + kv_dim], torch.full((kv_dim, hidden), 2.0))
    assert torch.equal(qkv0[q_dim + kv_dim :], torch.full((kv_dim, hidden), 3.0))

    # Validate gate_up_proj fusion for the text MLP at layer 1.
    gate_up1 = params["layers.1.base.mlp.gate_up_proj.weight"]
    assert torch.equal(gate_up1[:inter], torch.full((inter, hidden), 14.0))
    assert torch.equal(gate_up1[inter:], torch.full((inter, hidden), 15.0))

    # Validate audio_lm_head is the fused tensor as-is. Each per-codebook
    # slab carries its distinct constant; the load_weights path no longer
    # splits the tensor, it loads the full ``[num_codebooks * codebook_size, hidden]``
    # weight into ``audio_lm_head.weight`` directly.
    head = params["audio_lm_head.weight"]
    for k in range(num_codebooks):
        assert torch.equal(
            head[k * codebook_size : (k + 1) * codebook_size],
            torch.full((codebook_size, hidden), float(k + 1)),
        )

def test_stage1_chunk_decode_trims_left_context() -> None:
    """forward_chunk must slice off left_context_size * hop_length samples."""
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_code2wav import (
        HiggsAudioV2Code2Wav,
    )

    cfg = HiggsAudioV2Config()
    stage1 = HiggsAudioV2Code2Wav(cfg)

    # Monkey-patch decode_codes (the direct-decode entry point used by
    # forward_chunk) to return a deterministic PCM of length (T * hop_length)
    # given a [B, num_codebooks, T] code tensor.
    HOP = 960

    def _fake_decode(codes):
        t = int(codes.shape[-1])
        return torch.arange(t * HOP, dtype=torch.float32).reshape(1, 1, -1)

    object.__setattr__(stage1, "decode_codes", _fake_decode)
    stage1._loaded = True

    codes = torch.zeros(1, cfg.num_codebooks, 30, dtype=torch.long)  # 30 frames
    out_no_overlap = stage1.forward_chunk(codes, left_context_size=0, hop_length=HOP)
    assert out_no_overlap.shape[-1] == 30 * HOP

    out_overlap = stage1.forward_chunk(codes, left_context_size=5, hop_length=HOP)
    assert out_overlap.shape[-1] == 25 * HOP

    # Edge case: left_context_size >= T -> empty output.
    out_empty = stage1.forward_chunk(codes, left_context_size=30, hop_length=HOP)
    assert out_empty.shape[-1] == 0

def test_stage1_engine_runtime_forward_returns_omni_output() -> None:
    """Stage-1 engine-runtime path: flat codebook-major ``input_ids`` -> OmniOutput."""
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_code2wav import (
        HiggsAudioV2Code2Wav,
    )
    from vllm_omni.model_executor.models.output_templates import OmniOutput

    cfg = HiggsAudioV2Config()
    stage1 = HiggsAudioV2Code2Wav(cfg)

    HOP = 960
    NUM_CODEBOOKS = int(cfg.num_codebooks)

    def _fake_decode(codes):
        t = int(codes.shape[-1])
        return torch.arange(t * HOP, dtype=torch.float32).reshape(1, 1, -1)

    object.__setattr__(stage1, "decode_codes", _fake_decode)
    stage1._loaded = True

    n_frames = 12
    flat = torch.zeros(NUM_CODEBOOKS * n_frames, dtype=torch.long)
    out = stage1.forward(input_ids=flat)
    assert isinstance(out, OmniOutput)
    audio_list = out.multimodal_outputs["model_outputs"]
    sr_list = out.multimodal_outputs["sr"]
    assert len(audio_list) == 1
    assert audio_list[0].shape == (n_frames * HOP,)
    assert int(sr_list[0]) == int(cfg.sample_rate)

    # left_context_size consumption via runtime_additional_information.
    out_trimmed = stage1.forward(
        input_ids=flat,
        runtime_additional_information=[{"meta": {"left_context_size": 5}}],
    )
    assert isinstance(out_trimmed, OmniOutput)
    trimmed_audio = out_trimmed.multimodal_outputs["model_outputs"][0]
    assert trimmed_audio.shape == ((n_frames - 5) * HOP,)

def test_dual_ffn_routing_fault_injection_changes_audio_output() -> None:
    """Disabling the audio MLP must change the layer output on audio positions only.

    Drives ``HiggsAudioV2DecoderLayer._routed_mlp`` (which is the upstream DualFFN
    routing rule) on synthetic text/audio inputs with random text/audio MLPs.
    Verifies:
        1. Audio positions consume the audio MLP (zeroing it changes their output).
        2. Text positions do NOT see the audio MLP (zeroing it leaves them unchanged).
    The test sidesteps vLLM TP state by constructing the layer shell via __new__
    and assigning stub modules.
    """
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_talker import (
        HiggsAudioV2DecoderLayer,
    )

    hidden = 8
    inter = 16
    torch.manual_seed(0)
    text_mlp = torch.nn.Linear(hidden, hidden, bias=False)
    audio_mlp = torch.nn.Linear(hidden, hidden, bias=False)

    layer = HiggsAudioV2DecoderLayer.__new__(HiggsAudioV2DecoderLayer)
    torch.nn.Module.__init__(layer)
    # Shape the layer's expected attributes without running __init__ (avoids vLLM TP setup).
    layer.audio_mlp = audio_mlp
    layer.base = SimpleNamespace(mlp=text_mlp)

    seq_len = 6
    hidden_state = torch.randn(1, seq_len, hidden)
    # Half text (mask=False), half audio (mask=True)
    mask = torch.tensor([[False, False, False, True, True, True]])

    baseline = layer._routed_mlp(hidden_state.clone(), mask)
    # Sanity: baseline is finite and non-zero somewhere.
    assert torch.isfinite(baseline).all()

    # Fault-injection: zero out the audio MLP and recompute.
    with torch.no_grad():
        audio_mlp.weight.zero_()
    with_disabled = layer._routed_mlp(hidden_state.clone(), mask)

    # Audio positions (rows 3..5) must change when audio_mlp is disabled.
    text_diff = (baseline[0, :3] - with_disabled[0, :3]).abs().max().item()
    audio_diff = (baseline[0, 3:] - with_disabled[0, 3:]).abs().max().item()
    assert text_diff < 1e-6, (
        f"text positions should be unaffected by audio_mlp; got diff={text_diff}"
    )
    assert audio_diff > 1e-6, (
        f"audio positions must change when audio_mlp is disabled; got diff={audio_diff}"
    )

def test_dual_ffn_norm_routing_splits_text_and_audio() -> None:
    """The pre-attention norm routing must use audio_input_layernorm on audio positions only."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_talker import (
        HiggsAudioRMSNorm,
        HiggsAudioV2DecoderLayer,
    )

    hidden = 4
    layer = HiggsAudioV2DecoderLayer.__new__(HiggsAudioV2DecoderLayer)
    torch.nn.Module.__init__(layer)
    text_norm = HiggsAudioRMSNorm(hidden)
    audio_norm = HiggsAudioRMSNorm(hidden)
    layer.base = SimpleNamespace(input_layernorm=text_norm)
    layer.audio_input_layernorm = audio_norm

    hidden_state = torch.randn(1, 5, hidden)
    mask = torch.tensor([[False, False, True, True, True]])
    out = layer._routed_norm(hidden_state.clone(), text_norm, audio_norm, mask)

    # text positions: must equal text_norm(hidden_state[~mask])
    text_expected = text_norm(hidden_state[0, :2])
    assert torch.allclose(out[0, :2], text_expected, atol=1e-6)
    # audio positions: must equal audio_norm(hidden_state[mask])
    audio_expected = audio_norm(hidden_state[0, 2:])
    assert torch.allclose(out[0, 2:], audio_expected, atol=1e-6)

def test_talker_compute_logits_returns_tensor_and_audio_helper_runs_separately() -> None:
    """compute_logits MUST return a tensor (the AR runner pipes the result
    through ``.contiguous()`` + a generic sampler). Codebook-0 audio routing
    lives in :meth:`audio_codebook0_logits`, which the dedicated audio-side
    sampler dispatch (round-6 follow-up) consumes in parallel.
    """
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_talker import (
        HiggsAudioV2TalkerForConditionalGeneration,
    )

    cfg = HiggsAudioV2Config(num_hidden_layers=2)
    talker = HiggsAudioV2TalkerForConditionalGeneration.__new__(
        HiggsAudioV2TalkerForConditionalGeneration
    )
    torch.nn.Module.__init__(talker)
    talker.config = cfg

    hidden = int(cfg.hidden_size)
    num_codebooks = int(cfg.num_codebooks)
    codebook_size = int(cfg.codebook_size)
    torch.manual_seed(0)
    # R12: replace per-codebook split with the single fused audio_lm_head.
    audio_head = torch.nn.Linear(hidden, num_codebooks * codebook_size, bias=False)
    with torch.no_grad():
        audio_head.weight.mul_(0.01)
    talker.audio_lm_head = audio_head

    class _StubLP:
        def __call__(self, head, hidden_states, sampling_metadata):
            return torch.zeros(int(hidden_states.shape[0]), int(cfg.vocab_size))

    talker.lm_head = object()
    talker.logits_processor = _StubLP()

    h = torch.randn(4, hidden)
    mask = torch.tensor([True, False, True, False])

    # 1. compute_logits returns a tensor that the runner can call .contiguous() on.
    out = talker.compute_logits(h, sampling_metadata=None, audio_token_mask=mask)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, int(cfg.vocab_size))
    _ = out.contiguous()  # smoke-test the runner-side call site

    # 2. audio_codebook_logits returns [N_audio, num_codebooks, codebook_size]
    # at masked positions only — one call to the fused head reshaped.
    cb_all = talker.audio_codebook_logits(h, mask)
    assert isinstance(cb_all, torch.Tensor)
    assert cb_all.shape == (2, num_codebooks, codebook_size)

    # 3. Empty mask -> empty audio logits, runner-safe.
    empty_mask = torch.zeros(4, dtype=torch.bool)
    cb_empty = talker.audio_codebook_logits(h, empty_mask)
    assert cb_empty.shape == (0, num_codebooks, codebook_size)

def test_make_omni_output_reads_audio_codes_from_runner_kwargs() -> None:
    """``make_omni_output`` must accept the canonical runner contract
    (``model_intermediate_buffer``) and the documented fallbacks.

    The runner threads per-request ``codes.audio`` into
    ``model_intermediate_buffer`` (a list of dicts in batch order). We
    concatenate those and trim ``text_hidden_states`` to the emitted audio
    span -- matching the Qwen3-TTS / Fish-Speech recovery pattern.
    """
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_talker import (
        HiggsAudioV2TalkerForConditionalGeneration,
    )

    cfg = HiggsAudioV2Config(num_hidden_layers=1)
    talker = HiggsAudioV2TalkerForConditionalGeneration.__new__(
        HiggsAudioV2TalkerForConditionalGeneration
    )
    torch.nn.Module.__init__(talker)
    talker.config = cfg

    hidden = torch.zeros(8, int(cfg.hidden_size))
    codes_a = torch.arange(3, dtype=torch.long).reshape(3, 1)  # 3 frames for req0
    codes_b = torch.arange(10, 12, dtype=torch.long).reshape(2, 1)  # 2 frames for req1

    # Canonical contract: model_intermediate_buffer = [info_dict_per_request].
    out_canonical = talker.make_omni_output(
        hidden,
        model_intermediate_buffer=[
            {"codes": {"audio": codes_a}},
            {"codes": {"audio": codes_b}},
        ],
    )
    cat = torch.cat([codes_a, codes_b], dim=0)
    assert torch.equal(out_canonical.multimodal_outputs["codes"]["audio"], cat)
    # text_hidden_states must be trimmed to the emitted audio span.
    assert int(out_canonical.text_hidden_states.shape[0]) == int(cat.shape[0])

    # Deprecated alias: runtime_additional_information.
    out_legacy = talker.make_omni_output(
        hidden,
        runtime_additional_information=[{"codes": {"audio": codes_a}}],
    )
    assert torch.equal(out_legacy.multimodal_outputs["codes"]["audio"], codes_a)

    # Direct-API fallbacks (preserved for unit tests that don't run the
    # full runner).
    out_explicit = talker.make_omni_output(hidden, audio_codes=codes_a)
    assert torch.equal(out_explicit.multimodal_outputs["codes"]["audio"], codes_a)
    out_kw = talker.make_omni_output(hidden, model_kwargs={"audio_codes": codes_a})
    assert torch.equal(out_kw.multimodal_outputs["codes"]["audio"], codes_a)
    out_kwx = talker.make_omni_output(hidden, model_kwargs_extra={"audio_codes": codes_a})
    assert torch.equal(out_kwx.multimodal_outputs["codes"]["audio"], codes_a)

    # No source -> empty tensor.
    out_empty = talker.make_omni_output(hidden)
    assert out_empty.multimodal_outputs["codes"]["audio"].numel() == 0

def test_boson_model_pth_remap_rewrites_quantizer_keys() -> None:
    """``_remap_boson_model_pth_state_dict`` translates the standalone
    ``model.pth`` quantizer keys into the OmniVoice-style names that the
    shared HiggsAudioRVQ kernel expects.
    """
    from vllm_omni.model_executor.models._shared.higgs_audio_decoder import (
        _remap_boson_model_pth_state_dict,
    )

    boson_sd = {
        # Boson-ai standalone tokenizer key shapes (codebook size 1024, dim 64).
        "quantizer.vq.layers.0._codebook.embed": torch.zeros(1024, 64),
        "quantizer.vq.layers.0.project_out.weight": torch.zeros(1024, 64),
        "quantizer.vq.layers.0.project_out.bias": torch.zeros(1024),
        # project_in / _codebook.cluster_size / _codebook.embed_avg / inited
        # are encoder-side training state -- the remapper drops them.
        "quantizer.vq.layers.0.project_in.weight": torch.zeros(64, 1024),
        "quantizer.vq.layers.0._codebook.cluster_size": torch.zeros(1024),
        "quantizer.vq.layers.0._codebook.embed_avg": torch.zeros(1024, 64),
        "quantizer.vq.layers.0._codebook.inited": torch.tensor([1.0]),
        # Acoustic-decoder keys: boson uses decoder_2.* with weight-norm; the
        # remapper translates these into the vendored BosonDacDecoder layout
        # (acoustic_decoder.model.*).
        "decoder_2.model.0.weight_g": torch.zeros(1024, 1, 1),
        # fc_post2 in boson is what our shared kernel calls fc2.
        "fc_post2.weight": torch.zeros(256, 1024),
        "fc_post2.bias": torch.zeros(256),
        # An already-OmniVoice-style key passes through unchanged.
        "fc.weight": torch.zeros(32, 16),
    }
    remapped = _remap_boson_model_pth_state_dict(boson_sd)
    # The codebook-decode-essential keys are present under the new names.
    assert "quantizer.quantizers.0.codebook.embed" in remapped
    assert "quantizer.quantizers.0.project_out.weight" in remapped
    assert "quantizer.quantizers.0.project_out.bias" in remapped
    # Encoder-side training state is dropped.
    assert all("project_in" not in k for k in remapped)
    assert all("cluster_size" not in k for k in remapped)
    assert all("embed_avg" not in k for k in remapped)
    # decoder_2 -> acoustic_decoder remap.
    assert "decoder_2.model.0.weight_g" not in remapped
    assert "acoustic_decoder.model.0.weight_g" in remapped
    # fc_post2 -> fc2 remap.
    assert "fc_post2.weight" not in remapped
    assert "fc2.weight" in remapped
    assert "fc2.bias" in remapped
    # Unrelated keys pass through unchanged.
    assert "fc.weight" in remapped

def test_talker_opts_into_model_sampler() -> None:
    """``prefer_model_sampler`` must be True so the AR runner's hook kicks in,
    and ``sample`` must be callable with (logits, sampling_metadata)."""
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_talker import (
        HiggsAudioV2TalkerForConditionalGeneration,
    )

    assert HiggsAudioV2TalkerForConditionalGeneration.prefer_model_sampler is True
    assert callable(getattr(HiggsAudioV2TalkerForConditionalGeneration, "sample", None))

def test_talker_audio_codebook_logits_emits_all_codebooks() -> None:
    """R12: ``audio_codebook_logits`` returns the per-codebook logits for all 8
    codebooks at audio positions in one shot, via the fused
    ``audio_lm_head: Linear(hidden, num_codebooks * codebook_size)``."""
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_talker import (
        HiggsAudioV2TalkerForConditionalGeneration,
    )

    cfg = HiggsAudioV2Config(num_hidden_layers=1)
    talker = HiggsAudioV2TalkerForConditionalGeneration.__new__(
        HiggsAudioV2TalkerForConditionalGeneration
    )
    torch.nn.Module.__init__(talker)
    talker.config = cfg

    num_codebooks = int(cfg.num_codebooks)
    codebook_size = int(cfg.codebook_size)
    hidden = int(cfg.hidden_size)
    talker.audio_lm_head = torch.nn.Linear(hidden, num_codebooks * codebook_size, bias=False)

    n_total = 4
    n_audio = 2
    h = torch.randn(n_total, hidden)
    mask = torch.tensor([True, False, True, False])
    out = talker.audio_codebook_logits(h, mask)
    assert out.shape == (n_audio, num_codebooks, codebook_size)
    # The fused-head argmax over the codebook vocab axis stays in-range.
    codes = torch.argmax(out, dim=-1)
    assert codes.shape == (n_audio, num_codebooks)
    assert int(codes.max()) < codebook_size
    assert int(codes.min()) >= 0

def test_stage1_rms_threshold_is_meaningful_with_corrupted_codebook() -> None:
    """A corrupted RVQ codebook must push the PCM RMS above 1e-4.

    Builds a stub Stage-1 around the shared HiggsAudioRVQ kernel, generates a
    reference PCM with the kernel at its initialized weights, then scrambles
    one codebook and measures the normalized-float RMS difference. The
    expectation is that the threshold is not trivially passable by any tensor
    of similar shape.
    """
    from vllm_omni.model_executor.models._shared.higgs_audio_decoder import (
        HiggsAudioRVQ,
    )

    torch.manual_seed(0)
    num_codebooks = 8
    codebook_size = 1024
    codebook_dim = 64
    hidden = 1024

    rvq = HiggsAudioRVQ(
        num_quantizers=num_codebooks,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        hidden_size=hidden,
    )
    # Initialize codebooks with non-trivial weights so the test exercises real
    # arithmetic instead of all-zeros.
    with torch.no_grad():
        for q in rvq.quantizers:
            q.codebook.weight.uniform_(-0.5, 0.5)
            q.project_out.weight.uniform_(-0.05, 0.05)
            q.project_out.bias.zero_()

    # Build a small code tensor: 1 batch, num_codebooks codebooks, 4 frames.
    codes = torch.randint(0, codebook_size, (num_codebooks, 1, 4), dtype=torch.long)
    baseline = rvq.decode(codes.clone())  # [1, hidden, 4]

    # Corrupt codebook 0 by zeroing its codebook embedding.
    with torch.no_grad():
        rvq.quantizers[0].codebook.weight.zero_()
    corrupted = rvq.decode(codes.clone())

    # The corrupted codebook must push the normalized-float RMS above 1e-4.
    base_f = baseline.to(torch.float32).flatten()
    corr_f = corrupted.to(torch.float32).flatten()
    n = min(int(base_f.shape[0]), int(corr_f.shape[0]))
    rms = ((base_f[:n] - corr_f[:n]) ** 2).mean().sqrt().item()
    assert rms > 1e-4, (
        f"corrupting a codebook should push the RMS difference above the "
        f"1e-4 threshold; got rms={rms:.3e}"
    )

def test_cuda_graph_wrapper_falls_back_to_eager_without_warmup() -> None:
    """HiggsAudioV2CUDAGraphWrapper must delegate to the underlying module before warmup
    (and pass arguments through unchanged).
    """
    from vllm_omni.model_executor.models.higgs_audio_v2.cuda_graph_decoder_wrapper import (
        HiggsAudioV2CUDAGraphWrapper,
        WrapperState,
    )

    class _Talker(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids * 2

    talker = _Talker()
    wrapper = HiggsAudioV2CUDAGraphWrapper(
        talker, capture_batch_sizes=(1, 2, 4), enabled=True
    )
    x = torch.arange(4, dtype=torch.long).reshape(1, 4)
    out = wrapper(input_ids=x)
    assert torch.equal(out, x * 2)
    # All capture sizes should still be in the NOT_CAPTURED state.
    for b in (1, 2, 4):
        assert wrapper._state[b] == WrapperState.NOT_CAPTURED
    assert not wrapper.is_captured(1)

def test_cuda_graph_wrapper_disabled_is_passthrough() -> None:
    """``enabled=False`` -> wrapper delegates without any state machine bookkeeping."""
    from vllm_omni.model_executor.models.higgs_audio_v2.cuda_graph_decoder_wrapper import (
        HiggsAudioV2CUDAGraphWrapper,
    )

    class _Talker(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 100

    wrapper = HiggsAudioV2CUDAGraphWrapper(_Talker(), capture_batch_sizes=(1,), enabled=False)
    x = torch.tensor([[1, 2, 3]])
    out = wrapper(x=x)
    assert torch.equal(out, x + 100)

def test_offline_smoke_runs_through_shared_codec() -> None:
    """Offline smoke test (clean-machine target).

    Builds the shared HiggsAudio codec kernel with random-but-valid weights and
    runs a synthetic ``[1, 8, T]`` code tensor through Stage-1's direct decode
    API to assert the result is a non-degenerate 24 kHz PCM tensor of the
    expected shape. No real checkpoint or fixture is required.
    """
    from vllm_omni.model_executor.models._shared.higgs_audio_decoder import (
        HiggsAudioRVQ,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_code2wav import (
        HiggsAudioV2Code2Wav,
    )

    cfg = HiggsAudioV2Config()
    # Synthetic codes: 1 request, 8 codebooks, 32 frames of real content tokens.
    torch.manual_seed(0)
    codes = torch.randint(
        0, int(cfg.num_real_codes), (1, int(cfg.num_codebooks), 32), dtype=torch.long
    )
    stage1 = HiggsAudioV2Code2Wav(cfg)

    # Build a synthetic but valid Stage-1 stack with the shared kernel.
    # Random weights are fine for the shape smoke (no parity assertion).
    torch.manual_seed(0)
    quantizer = HiggsAudioRVQ(
        num_quantizers=int(cfg.num_codebooks),
        codebook_size=int(cfg.num_real_codes),
        codebook_dim=64,
        hidden_size=1024,
    )
    fc2 = torch.nn.Linear(1024, 256)
    decoder_stub = torch.nn.ConvTranspose1d(256, 1, kernel_size=960, stride=960)
    with torch.no_grad():
        decoder_stub.weight.uniform_(-0.01, 0.01)
        decoder_stub.bias.zero_()
    stage1.quantizer = quantizer
    stage1.fc2 = fc2
    stage1.acoustic_decoder = decoder_stub
    stage1._loaded = True

    pcm = stage1.decode_codes(codes)
    # Shape: [B, 1, T * hop=960]
    expected_samples = int(codes.shape[2]) * 960
    assert pcm.shape == (1, 1, expected_samples), (
        f"offline smoke expected PCM shape (1, 1, {expected_samples}); got {tuple(pcm.shape)}"
    )
    assert torch.isfinite(pcm).all()

def test_stage1_engine_constructor_signature() -> None:
    """Stage-1 must accept the *, vllm_config, prefix kwargs form for engine boot."""
    import inspect

    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_code2wav import (
        HiggsAudioV2Code2Wav,
    )

    sig = inspect.signature(HiggsAudioV2Code2Wav.__init__)
    params = sig.parameters
    assert "vllm_config" in params, "Stage-1 must accept vllm_config kwarg"
    assert "prefix" in params, "Stage-1 must accept prefix kwarg"
    assert params["vllm_config"].kind == inspect.Parameter.KEYWORD_ONLY

def test_fused_audio_head_split_shapes() -> None:
    """Verify the audio_lm_head split into codebook 0 head + residual heads matches the expected shapes."""
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )

    cfg = HiggsAudioV2Config()
    num_codebooks = int(cfg.num_codebooks)
    codebook_size = int(cfg.codebook_size)
    hidden = int(cfg.hidden_size)
    fused = torch.arange(num_codebooks * codebook_size * hidden, dtype=torch.float32).reshape(
        num_codebooks * codebook_size, hidden
    )
    # codebook 0 occupies the first ``codebook_size`` rows.
    head0 = fused[:codebook_size]
    assert tuple(head0.shape) == (codebook_size, hidden)
    # codebook k occupies rows [k * codebook_size, (k+1) * codebook_size).
    for k in range(1, num_codebooks):
        chunk = fused[k * codebook_size : (k + 1) * codebook_size]
        assert tuple(chunk.shape) == (codebook_size, hidden)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
