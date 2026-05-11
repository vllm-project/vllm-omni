# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Raon model and config helpers."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from vllm_omni.model_executor.models.raon.raon import (
    AudioDecodeState,
    RaonModel,
    RaonMultiModalProcessor,
    _infer_audio_placeholder_token_ids,
    _StepContext,
    build_audio_input_placeholder_ids,
    flatten_audio_embeddings,
    scatter_audio_input_embeddings,
)
from vllm_omni.model_executor.models.raon.raon_adaptors import EmbeddingAdaptorOutput
from vllm_omni.model_executor.models.raon.raon_audio_encoder import (
    Qwen3OmniAuTWrapper,
    compute_num_audio_input_tokens,
)
from vllm_omni.tokenizers.raon_tokenizer import (
    AUDIO_INPUT_PAD_TOKEN,
    AUDIO_OUTPUT_OPEN_SEQ,
    AUDIO_OUTPUT_PLACEHOLDER_SEQ,
    AUDIO_PLACEHOLDER_SEQ,
    SPEAKER_EMBEDDING_PLACEHOLDER_ID,
    SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN,
    resolve_audio_input_token_id,
    resolve_speaker_token_id,
    resolve_speaker_token_text,
)
from vllm_omni.transformers_utils.configs.raon import (
    EmbeddingAdaptorConfig,
    RaonConfig,
    SpeakerEncoderConfig,
    _build_subconfig,
    _load_raon_env_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _capture_records(target_logger: logging.Logger, level: int) -> tuple[list[logging.LogRecord], logging.Handler, int]:
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=level)
    prev_level = target_logger.level
    target_logger.addHandler(handler)
    target_logger.setLevel(level)
    return records, handler, prev_level


def _restore_records(target_logger: logging.Logger, handler: logging.Handler, prev_level: int) -> None:
    target_logger.removeHandler(handler)
    target_logger.setLevel(prev_level)


# ===================================================================
# Configuration: EmbeddingAdaptorConfig
# ===================================================================


class TestEmbeddingAdaptorConfig:
    def test_defaults(self):
        cfg = EmbeddingAdaptorConfig()
        assert cfg.input_size == 512
        assert cfg.output_size == 4096
        assert cfg.output_time_scale == 1.0
        assert cfg.num_layers == 1
        assert cfg.hidden_size is None
        assert cfg.use_post_norm is False
        assert cfg.norm_eps == 1e-6
        assert cfg.decoder_config is None

    def test_custom_values(self):
        cfg = EmbeddingAdaptorConfig(
            input_size=256,
            output_size=2048,
            output_time_scale=2.0,
            num_layers=3,
            hidden_size=1024,
            use_post_norm=True,
            norm_eps=1e-5,
        )
        assert cfg.input_size == 256
        assert cfg.output_size == 2048
        assert cfg.output_time_scale == 2.0
        assert cfg.num_layers == 3
        assert cfg.hidden_size == 1024
        assert cfg.use_post_norm is True
        assert cfg.norm_eps == 1e-5

    def test_decoder_config_from_dict(self):
        decoder_dict = {
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
        }
        cfg = EmbeddingAdaptorConfig(decoder_config=decoder_dict)
        from transformers import Qwen3Config

        assert isinstance(cfg.decoder_config, Qwen3Config)
        assert cfg.decoder_config.vocab_size == 32000

    def test_decoder_config_already_pretrained_config(self):
        existing = PretrainedConfig()
        cfg = EmbeddingAdaptorConfig(decoder_config=existing)
        assert cfg is not None

    def test_model_type(self):
        assert EmbeddingAdaptorConfig.model_type == "embedding_adaptor"


# ===================================================================
# Configuration: SpeakerEncoderConfig
# ===================================================================


class TestSpeakerEncoderConfig:
    @pytest.mark.parametrize(
        "field,default",
        [
            ("input_size", 512),
            ("output_size", 4096),
            ("num_heads", 8),
            ("min_seconds", 2.0),
            ("max_seconds", 10.0),
            ("frame_rate", 12.5),
            ("encoder_type", "from_scratch"),
            ("pretrained_model_id", None),
            ("pretrained_dim", None),
        ],
    )
    def test_defaults(self, field, default):
        cfg = SpeakerEncoderConfig()
        assert getattr(cfg, field) == default

    def test_custom_values(self):
        cfg = SpeakerEncoderConfig(
            input_size=128,
            output_size=1024,
            num_heads=4,
            min_seconds=1.0,
            max_seconds=5.0,
            frame_rate=25.0,
            encoder_type="pretrained",
            pretrained_model_id="some/model",
            pretrained_dim=768,
        )
        assert cfg.input_size == 128
        assert cfg.output_size == 1024
        assert cfg.num_heads == 4
        assert cfg.min_seconds == 1.0
        assert cfg.max_seconds == 5.0
        assert cfg.frame_rate == 25.0
        assert cfg.encoder_type == "pretrained"
        assert cfg.pretrained_model_id == "some/model"
        assert cfg.pretrained_dim == 768

    def test_model_type(self):
        assert SpeakerEncoderConfig.model_type == "speaker_encoder"


# ===================================================================
# Configuration: _build_subconfig
# ===================================================================


class TestBuildSubconfig:
    @pytest.mark.parametrize(
        "input_val,expected_type,check",
        [
            (None, type(None), lambda r: r is None),
            ({"some_field": 42}, PretrainedConfig, None),
            (
                {"model_type": "embedding_adaptor", "input_size": 128},
                EmbeddingAdaptorConfig,
                lambda r: r.input_size == 128,
            ),
            (
                {"model_type": "speaker_encoder", "num_heads": 4},
                SpeakerEncoderConfig,
                lambda r: r.num_heads == 4,
            ),
            ({"model_type": "nonexistent_xyz_123"}, PretrainedConfig, None),
        ],
        ids=["none", "no-model-type", "embedding-adaptor", "speaker-encoder", "unknown-type"],
    )
    def test_various_inputs(self, input_val, expected_type, check):
        result = _build_subconfig(input_val)
        if expected_type is type(None):
            assert result is None
        else:
            assert isinstance(result, expected_type)
        if check:
            assert check(result)

    def test_pretrained_config_passthrough(self):
        cfg = PretrainedConfig()
        assert _build_subconfig(cfg) is cfg

    def test_dict_with_default_model_type(self):
        result = _build_subconfig(
            {"input_size": 256},
            default_model_type="embedding_adaptor",
        )
        assert isinstance(result, EmbeddingAdaptorConfig)
        assert result.input_size == 256

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _build_subconfig(12345)  # type: ignore[arg-type]


# ===================================================================
# Configuration: RaonConfig._resolve_audio_output_token_id
# ===================================================================


class TestResolveAudioOutputTokenId:
    def _make_config(self, **kwargs) -> RaonConfig:
        return RaonConfig(**kwargs)

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({"audio_output_token_id": 151675}, 151675),
            ({"audio_token_id": 151675}, 151675),
            ({"audio_output_token_id": 100, "audio_token_id": 200}, 100),
            ({}, 151675),
            (
                {
                    "text_model_config": {"vocab_size": 151680},
                    "audio_tokenizer_config": {"codebook_size": 4096},
                },
                151675,
            ),
        ],
        ids=["explicit", "fallback-audio-token", "precedence", "default", "with-vocab-codebook"],
    )
    def test_resolution(self, kwargs, expected):
        cfg = self._make_config(**kwargs)
        assert cfg.audio_output_token_id == expected

    def test_int_coercion(self):
        cfg = self._make_config(audio_output_token_id="151675")  # type: ignore[arg-type]
        assert cfg.audio_output_token_id == 151675
        assert isinstance(cfg.audio_output_token_id, int)

    def test_default_logs_warning(self):
        from vllm_omni.transformers_utils.configs import raon as raon_config_module

        records, handler, prev_level = _capture_records(raon_config_module.logger, logging.WARNING)
        try:
            cfg = self._make_config(audio_input_token_id=151676)
        finally:
            _restore_records(raon_config_module.logger, handler, prev_level)

        assert cfg.audio_output_token_id == 151675
        assert any("audio_output_token_id omitted" in record.getMessage() for record in records)


# ===================================================================
# Configuration: RaonConfig._resolve_audio_input_token_id
# ===================================================================


class TestResolveAudioInputTokenId:
    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({"audio_input_token_id": 151676}, 151676),
            ({"audio_output_token_id": 151675}, 151676),
            ({"audio_output_token_id": 151675, "audio_input_token_id": 99999}, 99999),
            ({}, 151676),
        ],
        ids=["explicit", "fallback-placeholder", "explicit-overrides", "default"],
    )
    def test_resolution(self, kwargs, expected):
        cfg = RaonConfig(**kwargs)
        assert cfg.audio_input_token_id == expected

    def test_int_coercion(self):
        cfg = RaonConfig(audio_input_token_id="151676")  # type: ignore[arg-type]
        assert cfg.audio_input_token_id == 151676
        assert isinstance(cfg.audio_input_token_id, int)

    def test_default_logs_warning(self):
        from vllm_omni.transformers_utils.configs import raon as raon_config_module

        records, handler, prev_level = _capture_records(raon_config_module.logger, logging.WARNING)
        try:
            cfg = RaonConfig(audio_output_token_id=151675)
        finally:
            _restore_records(raon_config_module.logger, handler, prev_level)

        assert cfg.audio_input_token_id == 151676
        assert any("audio_input_token_id omitted" in record.getMessage() for record in records)


# ===================================================================
# Configuration: RaonConfig construction
# ===================================================================


class TestRaonConfigConstruction:
    def test_default_construction(self):
        cfg = RaonConfig()
        assert cfg.model_type == "raon"
        assert cfg.audio_output_token_id is None
        assert cfg.audio_input_token_id is None
        assert cfg.speaker_token_id is None
        assert cfg.speaker_embedding_to_code_predictor is None

    def test_env_defaults_to_gate055_noretry_and_logs(self, monkeypatch):
        from vllm_omni.transformers_utils.configs import raon as raon_config_module

        for name in (
            "RAON_TTS_LONG_MODE",
            "RAON_TTS_LONG_MAX_SENTENCES_PER_CHUNK",
            "RAON_TTS_LONG_ENABLE_FINAL_BEST_OF_K",
            "RAON_TTS_LONG_ENABLE_FINAL_EOS_MIN_GATE",
            "RAON_TTS_LONG_FINAL_EOS_MIN_DURATION_RATIO",
            "RAON_TTS_LONG_ENABLE_FINAL_EOS_RETRY",
        ):
            monkeypatch.delenv(name, raising=False)

        records, handler, prev_level = _capture_records(raon_config_module.logger, logging.INFO)
        try:
            env = _load_raon_env_config()
        finally:
            _restore_records(raon_config_module.logger, handler, prev_level)

        assert env.tts_long_mode == "rolling_icl"
        assert env.tts_long_max_sentences_per_chunk == 1
        assert env.tts_long_enable_final_best_of_k is False
        assert env.tts_long_enable_final_eos_min_gate is True
        assert env.tts_long_final_eos_min_duration_ratio == 0.55
        assert env.tts_long_enable_final_eos_retry is False
        assert any(
            "tts_long_mode=rolling_icl" in record.getMessage()
            and "tts_long_max_sentences_per_chunk=1" in record.getMessage()
            and "tts_long_enable_final_best_of_k=False" in record.getMessage()
            and "tts_long_enable_final_eos_min_gate=True" in record.getMessage()
            and "tts_long_final_eos_min_duration_ratio=0.55" in record.getMessage()
            and "tts_long_enable_final_eos_retry=False" in record.getMessage()
            for record in records
        )

    @pytest.mark.parametrize(
        "kwarg,field,config_type,check_field,check_val",
        [
            ("text_model_config", "text_model_config", PretrainedConfig, "vocab_size", 1024),
            ("input_adaptor_config", "input_adaptor_config", EmbeddingAdaptorConfig, "input_size", 256),
            ("output_adaptor_config", "output_adaptor_config", EmbeddingAdaptorConfig, "input_size", 128),
            ("speaker_encoder_config", "speaker_encoder_config", SpeakerEncoderConfig, "num_heads", 16),
        ],
    )
    def test_subconfig_from_dict(self, kwarg, field, config_type, check_field, check_val):
        if kwarg == "text_model_config":
            init = {kwarg: {check_field: check_val}}
        elif kwarg == "input_adaptor_config":
            init = {kwarg: {"input_size": 256, "output_size": 1024}}
        elif kwarg == "output_adaptor_config":
            init = {kwarg: {"input_size": 128, "output_size": 512}}
        else:
            init = {kwarg: {"num_heads": 16, "min_seconds": 3.0}}
        cfg = RaonConfig(**init)
        sub = getattr(cfg, field)
        assert isinstance(sub, config_type)
        assert getattr(sub, check_field) == check_val

    def test_text_config_alias(self):
        cfg = RaonConfig(text_model_config={"vocab_size": 512})
        assert cfg.text_config is cfg.text_model_config

    def test_get_text_config_returns_config(self):
        cfg = RaonConfig(text_model_config={"vocab_size": 512})
        assert cfg.get_text_config() is cfg.text_model_config

    def test_get_text_config_raises_when_missing(self):
        cfg = RaonConfig()
        with pytest.raises(ValueError, match="missing"):
            cfg.get_text_config()


# ===================================================================
# Audio codec: helpers
# ===================================================================


def _minimal_mimi_config(**overrides):
    from transformers.models.mimi import MimiConfig

    defaults = dict(
        hidden_size=64,
        num_filters=32,
        upsampling_ratios=[2, 2],
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        num_residual_layers=1,
        dilation_growth_rate=2,
        audio_channels=1,
        sampling_rate=24000,
        frame_rate=12.5,
        encodec_frame_rate=25.0,
        upsample_groups=1,
        pad_mode="constant",
        causal=True,
        num_quantizers=8,
        codebook_size=2048,
        codebook_dim=64,
        use_cache=False,
        return_dict=False,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
    )
    defaults.update(overrides)
    return MimiConfig(**defaults)


# ===================================================================
# Audio codec: padding cache
# ===================================================================


class TestMimiConvTranspose1dPaddingCache:
    def _make_cache(self, num_layers=2):
        from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import (
            MimiConvTranspose1dPaddingCache,
        )

        per_layer_padding = [
            torch.tensor(2, dtype=torch.int64),
            torch.tensor(4, dtype=torch.int64),
        ]
        per_layer_in_channels = [8, 4]
        return MimiConvTranspose1dPaddingCache(
            num_layers=num_layers,
            per_layer_padding=per_layer_padding,
            per_layer_in_channels=per_layer_in_channels,
        )

    def test_construction_sets_attributes(self):
        cache = self._make_cache()
        assert len(cache.per_layer_padding) == 2
        assert len(cache.per_layer_in_channels) == 2
        assert cache.padding_cache == [None, None]

    def test_construction_mismatched_lengths_raises(self):
        from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import (
            MimiConvTranspose1dPaddingCache,
        )

        with pytest.raises(ValueError):
            MimiConvTranspose1dPaddingCache(
                num_layers=2,
                per_layer_padding=[torch.tensor(2)],
                per_layer_in_channels=[8, 4],
            )

    def test_update_returns_zero_tensor_on_first_call(self):
        cache = self._make_cache()
        hidden_states = torch.ones(1, 8, 6)
        result = cache.update(hidden_states, layer_idx=0)
        assert result.shape == (1, 8, 2)
        assert result.sum().item() == 0.0

    def test_update_stores_tail_and_returns_on_next(self):
        cache = self._make_cache()
        first = torch.arange(1 * 8 * 6, dtype=torch.float).reshape(1, 8, 6)
        cache.update(first, layer_idx=0)
        stored = cache.padding_cache[0]
        assert stored is not None
        assert stored.shape == (1, 8, 2)
        assert torch.equal(stored, first[:, :, -2:])

        second = torch.zeros(1, 8, 6)
        returned = cache.update(second, layer_idx=0)
        assert torch.equal(returned, first[:, :, -2:])

    def test_zero_padding_returns_empty_tensor(self):
        from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import (
            MimiConvTranspose1dPaddingCache,
        )

        cache = MimiConvTranspose1dPaddingCache(
            num_layers=1,
            per_layer_padding=[torch.tensor(0, dtype=torch.int64)],
            per_layer_in_channels=[4],
        )
        hidden = torch.ones(1, 4, 8)
        result = cache.update(hidden, layer_idx=0)
        assert result.shape[-1] == 0


# ===================================================================
# Audio codec: decoder output
# ===================================================================


class TestStreamingMimiDecoderOutputDataclasses:
    def test_defaults(self):
        from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import (
            StreamingMimiDecoderOutput,
        )

        out = StreamingMimiDecoderOutput()
        assert out.audio_values is None
        assert out.decoder_past_key_values is None
        assert out.conv1d_padding_cache is None
        assert out.convtranspose1d_padding_cache is None


# ===================================================================
# Audio codec: decoder construction
# ===================================================================


class TestStreamingMimiDecoderConstruction:
    def test_constructs_with_small_config(self):
        from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import (
            StreamingMimiDecoder,
        )

        cfg = _minimal_mimi_config()
        decoder = StreamingMimiDecoder(cfg)
        assert hasattr(decoder, "layers")
        assert len(decoder._mimiconv1d_layer_names) > 0

    def test_layer_idx_assigned_to_conv1d_sublayers(self):
        from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import (
            StreamingMimiDecoder,
        )

        cfg = _minimal_mimi_config()
        decoder = StreamingMimiDecoder(cfg)
        for layer_idx, name in enumerate(decoder._mimiconv1d_layer_names):
            submod = decoder.get_submodule(name)
            assert submod.layer_idx == layer_idx

    def test_layer_idx_assigned_to_convtranspose1d_sublayers(self):
        from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import (
            StreamingMimiConvTranspose1d,
            StreamingMimiDecoder,
        )

        cfg = _minimal_mimi_config()
        decoder = StreamingMimiDecoder(cfg)
        for layer_idx, name in enumerate(decoder._mimiconvtranspose1d_layer_names):
            submod = decoder.get_submodule(name)
            assert isinstance(submod, StreamingMimiConvTranspose1d)
            assert submod.layer_idx == layer_idx


# ===================================================================
# Audio codec: transpose convolution
# ===================================================================


class TestStreamingMimiConvTranspose1dConstruction:
    def test_constructs_and_registers_buffers(self):
        from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import (
            StreamingMimiConvTranspose1d,
        )

        cfg = _minimal_mimi_config()
        layer = StreamingMimiConvTranspose1d(
            cfg,
            in_channels=16,
            out_channels=8,
            kernel_size=4,
            stride=2,
            layer_idx=0,
        )
        assert layer.in_channels == 16
        assert layer.layer_idx == 0
        assert hasattr(layer, "stride")
        assert hasattr(layer, "kernel_size")
        assert hasattr(layer, "padding_total")

    def test_padding_total_value(self):
        from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import (
            StreamingMimiConvTranspose1d,
        )

        cfg = _minimal_mimi_config()
        layer = StreamingMimiConvTranspose1d(
            cfg,
            in_channels=16,
            out_channels=8,
            kernel_size=6,
            stride=2,
            layer_idx=0,
        )
        assert int(layer.padding_total.item()) == 4

    def test_forward_non_causal_with_cache_raises(self):
        from vllm_omni.model_executor.models.raon.raon_audio_tokenizer import (
            MimiConvTranspose1dPaddingCache,
            StreamingMimiConvTranspose1d,
        )

        cfg = _minimal_mimi_config(causal=False)
        layer = StreamingMimiConvTranspose1d(
            cfg,
            in_channels=4,
            out_channels=2,
            kernel_size=4,
            stride=2,
            layer_idx=0,
        )
        dummy_cache = MimiConvTranspose1dPaddingCache(
            num_layers=1,
            per_layer_padding=[torch.tensor(2)],
            per_layer_in_channels=[4],
        )
        hidden = torch.randn(1, 4, 8)
        with pytest.raises(ValueError, match="causal"):
            layer.forward(hidden, padding_cache=dummy_cache)


# ===================================================================
# Audio input pipeline: embed_multimodal
# ===================================================================


class _FakeAudioEncoder(nn.Module):
    def __init__(self, embeds: torch.Tensor, sampling_rate: int = 24000):
        super().__init__()
        self.config = SimpleNamespace(sampling_rate=sampling_rate)
        self._embeds = embeds
        self.last_audio: torch.Tensor | None = None
        self.audio_inputs: list[torch.Tensor] = []
        self.calls = 0

    def forward(self, audio: torch.Tensor, **_: object) -> SimpleNamespace:
        self.calls += 1
        self.last_audio = audio
        self.audio_inputs.append(audio.clone())
        return SimpleNamespace(embeds=self._embeds.to(device=audio.device))


class _FakeInputAdaptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)
        self.last_inputs: torch.Tensor | None = None
        self.last_mask: torch.Tensor | None = None
        self.mask_history: list[torch.Tensor | None] = []

    @property
    def proj_dtype(self) -> torch.dtype:
        return self.proj.weight.dtype

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EmbeddingAdaptorOutput:
        self.last_inputs = inputs
        self.last_mask = mask
        self.mask_history.append(None if mask is None else mask.clone())
        return EmbeddingAdaptorOutput(outputs_embeds=inputs + 10.0, mask=mask)


def test_embed_multimodal_trims_each_audio_to_true_length_before_encoding():
    encoder_embeds = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        dtype=torch.float32,
    )

    model = object.__new__(RaonModel)
    model.model_stage = "stage0"
    model.samples_per_frame = 1920
    model.sampling_rate = 24000
    model.audio_input_token_id = 151676
    model.audio_output_token_id = 151675
    object.__setattr__(model, "audio_encoder", _FakeAudioEncoder(encoder_embeds, sampling_rate=24000))
    object.__setattr__(model, "input_adaptor", _FakeInputAdaptor())

    audio_waveforms = torch.randn((2, 5760), dtype=torch.float32)
    audio_lengths = torch.tensor([5760, 1921], dtype=torch.long)

    out = model.embed_multimodal(
        audio_waveforms=audio_waveforms,
        audio_lengths=audio_lengths,
    )

    assert model.audio_encoder.calls == 2
    assert [tuple(audio.shape) for audio in model.audio_encoder.audio_inputs] == [
        (1, 1, 5760),
        (1, 1, 1921),
    ]
    assert torch.equal(model.input_adaptor.last_inputs, encoder_embeds)

    assert len(model.input_adaptor.mask_history) == 2
    assert torch.equal(
        model.input_adaptor.mask_history[0],
        torch.tensor([[True, True, True]]),
    )
    assert torch.equal(
        model.input_adaptor.mask_history[1],
        torch.tensor([[True, True, False]]),
    )

    assert isinstance(out, tuple) and len(out) == 2
    assert torch.equal(out[0], torch.tensor([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]))
    assert torch.equal(out[1], torch.tensor([[11.0, 12.0], [13.0, 14.0]]))


# ===================================================================
# Multimodal processor: prompt updates
# ===================================================================


def test_prompt_updates_accept_chat_template_audio_pad_variant(stub_info, stub_mm_items):
    processor = object.__new__(RaonMultiModalProcessor)
    processor.info = stub_info

    updates = processor._get_prompt_updates(
        mm_items=stub_mm_items,
        hf_processor_mm_kwargs={},
        out_mm_kwargs={},
    )

    assert len(updates) == 4
    targets = {tuple(update.target) if not isinstance(update.target, str) else update.target for update in updates}
    assert targets == {
        AUDIO_PLACEHOLDER_SEQ,
        AUDIO_OUTPUT_PLACEHOLDER_SEQ,
        AUDIO_OUTPUT_OPEN_SEQ,
        (151669, 151673, 151670),
    }

    replacement = updates[0].replacement(0)
    assert replacement.full == build_audio_input_placeholder_ids(
        num_audio_tokens=2,
        audio_start_token_id=151669,
        audio_input_token_id=151676,
        audio_end_token_id=151670,
    )


# ===================================================================
# Output mode masks
# ===================================================================


def test_compute_logits_masks_text_and_audio_vocab_ranges():
    model = object.__new__(RaonModel)
    model.model_stage = "stage0"
    model._audio_only_allowed_text_token_ids = (2,)
    model.lm_head = None
    model.logits_processor = lambda lm_head, hidden_states: hidden_states.clone()
    model.audio_lm_head = lambda x: x[:, model.vocab_size - model.codebook_size :]
    model.vocab_size = 6
    model.codebook_size = 2
    model._step_ctx = _StepContext(
        talker_hidden=torch.zeros((3, 4), dtype=torch.float32),
        runtime_info=[
            {"output_mode": ["text_only"]},
            {"output_mode": ["audio_only"]},
            {"output_mode": ["text_and_audio"]},
        ],
    )

    raw_logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 1.1, 1.2],
            [1.1, 1.2, 1.3, 1.4, 2.1, 2.2],
            [2.1, 2.2, 2.3, 2.4, 3.1, 3.2],
        ],
        dtype=torch.float32,
    )

    masked = model.compute_logits(raw_logits)
    assert torch.isneginf(masked[0, 4:]).all()
    assert torch.equal(masked[1], raw_logits[1])
    assert torch.equal(masked[2], raw_logits[2])


def test_normalize_speaker_ref_audio_accepts_common_container_types():
    assert RaonModel._normalize_speaker_ref_audio("/tmp/ref.wav") == "/tmp/ref.wav"
    assert RaonModel._normalize_speaker_ref_audio(["/tmp/ref.wav"]) == "/tmp/ref.wav"
    assert RaonModel._normalize_speaker_ref_audio(b"/tmp/ref.wav") == "/tmp/ref.wav"
    assert RaonModel._normalize_speaker_ref_audio({"speaker_ref_audio": [b"/tmp/ref.wav"]}) == "/tmp/ref.wav"
    assert RaonModel._normalize_speaker_ref_audio({"ref_audio": {"path": "/tmp/ref.wav"}}) == "/tmp/ref.wav"


def test_make_omni_output_prefers_incremental_chunk_over_full_codec_payload():
    model = object.__new__(RaonModel)
    model.model_stage = "stage0"

    hidden = torch.randn((1, 4), dtype=torch.float32)
    full = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    chunk = torch.tensor([[4, 5, 6]], dtype=torch.long)

    out = model.make_omni_output(
        hidden,
        runtime_additional_information=[{"codec_codes": full, "codec_codes_chunk": chunk}],
    )

    payload = out.multimodal_outputs["codec_codes_chunk"]  # type: ignore[index]
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert torch.equal(payload[0], chunk)


def test_make_omni_output_keeps_codec_payloads_batch_aligned():
    model = object.__new__(RaonModel)
    model.model_stage = "stage0"

    hidden = torch.randn((3, 4), dtype=torch.float32)
    chunk_a = torch.tensor([[1, 2, 3]], dtype=torch.long)
    chunk_c = torch.tensor([[7, 8, 9]], dtype=torch.long)

    out = model.make_omni_output(
        hidden,
        runtime_additional_information=[
            {"_omni_req_id": "req-a", "codec_codes_chunk": chunk_a},
            {"_omni_req_id": "req-b"},
            {"_omni_req_id": "req-c", "codec_codes_chunk": chunk_c},
        ],
    )

    payload = out.multimodal_outputs["codec_codes_chunk"]  # type: ignore[index]
    assert isinstance(payload, list) and len(payload) == 3
    assert torch.equal(payload[0], chunk_a)
    assert payload[1] is None
    assert torch.equal(payload[2], chunk_c)


def test_find_audio_output_positions_token_mode_uses_explicit_trigger_ids():
    model = object.__new__(RaonModel)
    model.audio_output_token_id = 10
    model.vocab_size = 20
    model._audio_trigger_token_ids = (10, 12)

    input_ids = torch.tensor([9, 10, 11, 12, 13], dtype=torch.long)
    positions = model._find_audio_output_positions(input_ids)
    assert torch.equal(positions, torch.tensor([1, 3], dtype=torch.long))


def test_find_audio_output_positions_ignores_legacy_range_env(monkeypatch):
    model = object.__new__(RaonModel)
    model.audio_output_token_id = 10
    model.vocab_size = 20
    model._audio_trigger_token_ids = (10, 12)
    monkeypatch.setenv("RAON_AUDIO_TRIGGER_MODE", "range")

    input_ids = torch.tensor([9, 10, 11, 12, 13], dtype=torch.long)
    positions = model._find_audio_output_positions(input_ids)
    assert torch.equal(positions, torch.tensor([1, 3], dtype=torch.long))


def test_apply_top_p_logits_filter_masks_tail_tokens():
    logits = torch.tensor([[5.0, 4.0, 1.0, -1.0]], dtype=torch.float32)
    filtered = RaonModel._apply_top_p_logits_filter(logits.clone(), top_p=0.8)
    assert not torch.isneginf(filtered[0, 0])
    assert not torch.isneginf(filtered[0, 1])
    assert torch.isneginf(filtered[0, 2])
    assert torch.isneginf(filtered[0, 3])


def test_sample_first_audio_codes_applies_top_p():
    model = object.__new__(RaonModel)
    model.audio_lm_head = lambda _: torch.tensor([[5.0, 4.0, 1.0, -1.0]], dtype=torch.float32)

    prev_hidden = torch.zeros((1, 1, 4), dtype=torch.float32)
    sampled = model._sample_first_audio_codes(
        prev_hidden,
        temperature=1.0,
        top_k=-1,
        top_p=0.5,
    )
    assert sampled.shape == (1,)
    assert int(sampled[0].item()) == 0


@pytest.mark.parametrize(
    "pending_codes,expected_deferred",
    [
        (torch.tensor([[1, 2, 3]], dtype=torch.long), 1),
        (None, 0),
    ],
    ids=["with-pending-codes", "without-pending-codes"],
)
def test_audio_preprocess_bootstrap_bypass(pending_codes, expected_deferred):
    model = object.__new__(RaonModel)
    model.speaker_token_id = None
    model.audio_output_token_id = 7
    model._deferred_preprocess = []

    req_state = RaonModel._get_audio_decode_state({})
    req_state.audio_step_index = 0
    req_state.pending_audio_codes = pending_codes

    input_ids = torch.tensor([7], dtype=torch.long)
    input_embeds = torch.zeros((1, 4), dtype=torch.float32)

    _, _, update_dict = model.audio_preprocess(
        input_ids=input_ids,
        input_embeds=input_embeds,
        global_request_id=["req-1"],
        _decode_state=req_state,
    )

    assert update_dict.get("_decode_state") is req_state
    assert len(model._deferred_preprocess) == expected_deferred
    if pending_codes is not None:
        assert req_state.pending_audio_codes is None


def test_audio_preprocess_projects_cached_speaker_embedding_to_hidden_size():
    model = object.__new__(RaonModel)
    model.speaker_token_id = 99
    model.audio_output_token_id = 7
    model._deferred_preprocess = []
    model.speaker_encoder = SimpleNamespace(
        pretrained_dim=3,
        projection=nn.Linear(3, 4, bias=False, dtype=torch.float32),
    )
    model.is_pretrained_speaker_encoder = True

    with torch.no_grad():
        model.speaker_encoder.projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )

    input_ids = torch.tensor([10, 99, 11], dtype=torch.long)
    input_embeds = torch.zeros((3, 4), dtype=torch.float32)
    cached_spk = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)]

    _, new_embeds, update_dict = model.audio_preprocess(
        input_ids=input_ids,
        input_embeds=input_embeds,
        cached_spk_embedding=cached_spk,
    )

    expected = torch.tensor([1.0, 2.0, 3.0, 6.0], dtype=torch.float32)
    assert torch.equal(update_dict["speaker_embeds"], expected.unsqueeze(0))
    assert torch.equal(new_embeds[1], expected)


def test_forward_stage1_single_accepts_codec_codes_tensor_when_input_ids_empty():
    from vllm_omni.model_executor.models.raon.raon_code2wav import RaonCode2WavModel

    model = object.__new__(RaonCode2WavModel)
    model.sampling_rate = 24000
    model._resolve_stage1_streaming_info = lambda req_info: ("rid", False, False)
    model._resolve_stage1_codec_codes = lambda input_ids, req_info: torch.zeros((1, 2, 8), dtype=torch.long)
    model._decode_stage1_audio = lambda codec_codes: torch.ones((1, 320), dtype=torch.float32)
    model._trim_right_padding_audio = lambda audio, total_frames: audio
    model._trim_left_context_audio = lambda audio, left_context_size, total_frames: audio
    model._clear_streaming_state = lambda req_id: None

    req_info = {
        "global_request_id": "rid",
        "codec_codes": torch.zeros((2, 8), dtype=torch.long),
    }

    result = model._forward_stage1_single(torch.empty(0, dtype=torch.long), req_info)

    assert result.multimodal_outputs is not None
    assert result.multimodal_outputs["model_outputs"].shape[-1] == 320


def test_logits_router_apply_row_mode_adjustments_tracks_per_request_talker_hidden_for_text_only():
    from vllm_omni.model_executor.models.raon.logits_routing import LogitsRouter

    model = object.__new__(RaonModel)
    model.audio_output_token_id = 7
    model._step_talker_hidden_rows = {}
    model._step_decode_states = {}
    model._mask_audio_logits_for_text_mode = lambda logits, row_idx: None
    model._logits_router = LogitsRouter(model)

    logits = torch.zeros((2, 8), dtype=torch.float32)
    audio_hidden_states = torch.tensor(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
        dtype=torch.float32,
    )

    model._logits_router.apply_row_mode_adjustments(
        logits=logits,
        row_runtime_info=[
            {"_omni_req_id": "req-a", "output_mode": ["text_only"]},
            {"_omni_req_id": "req-b", "output_mode": ["text_only"]},
        ],
        output_token_ids=[[], []],
        audio_hidden_states=audio_hidden_states,
    )

    assert torch.equal(
        model._step_talker_hidden_rows["req-a"],
        audio_hidden_states[0:1],
    )
    assert torch.equal(
        model._step_talker_hidden_rows["req-b"],
        audio_hidden_states[1:2],
    )


def test_logits_router_audio_end_suppress_until_uses_request_min_steps():
    from vllm_omni.model_executor.models.raon.logits_routing import audio_end_suppress_until

    req_state = AudioDecodeState(audio_step_index=12, continuation_silence_frames=2)

    assert audio_end_suppress_until(req_state, {"audio_min_steps": [40]}) == 40


def test_postprocess_uses_per_request_talker_hidden_for_text_only_batch():
    model = object.__new__(RaonModel)
    model._step_decode_states = {}
    model._step_talker_hidden_rows = {
        "req-a": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        "req-b": torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32),
    }
    model._step_ctx = _StepContext(
        talker_hidden=torch.tensor(
            [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
            dtype=torch.float32,
        ),
        runtime_info=[],
    )

    out_a = model.postprocess(
        torch.tensor([[9.0, 9.0, 9.0]], dtype=torch.float32),
        _omni_req_id="req-a",
    )
    out_b = model.postprocess(
        torch.tensor([[8.0, 8.0, 8.0]], dtype=torch.float32),
        _omni_req_id="req-b",
    )

    assert torch.equal(out_a["prev_hidden"], torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))
    assert torch.equal(out_b["prev_hidden"], torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32))
    assert model._step_talker_hidden_rows == {}


def test_postprocess_clears_talker_hidden_row_before_persisting_decode_state():
    model = object.__new__(RaonModel)
    model._step_talker_hidden_rows = {}
    model._step_ctx = None
    req_state = RaonModel._get_audio_decode_state({})
    req_state._talker_hidden_row = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    model._step_decode_states = {"req-a": req_state}

    out = model.postprocess(
        torch.tensor([[9.0, 9.0, 9.0]], dtype=torch.float32),
        _omni_req_id="req-a",
    )

    assert torch.equal(out["prev_hidden"], torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))
    assert out["_decode_state"] is req_state
    assert req_state._talker_hidden_row is None


# ===================================================================
# Placeholders: math helpers
# ===================================================================


def test_placeholder_count_math_matches_masked_audio_frame_blocks():
    # Placeholder expansion must match the embeddings that survive the
    # input-adaptor mask, i.e. one token per 12.5 Hz audio frame block.
    assert compute_num_audio_input_tokens(0, sampling_rate=24000, frame_rate=12.5) == 0
    assert compute_num_audio_input_tokens(1, sampling_rate=24000, frame_rate=12.5) == 1
    assert compute_num_audio_input_tokens(1919, sampling_rate=24000, frame_rate=12.5) == 1
    assert compute_num_audio_input_tokens(1920, sampling_rate=24000, frame_rate=12.5) == 1
    assert compute_num_audio_input_tokens(1921, sampling_rate=24000, frame_rate=12.5) == 2
    # 1 second = 24000 samples -> 13 tokens
    assert compute_num_audio_input_tokens(24000, sampling_rate=24000, frame_rate=12.5) == 13
    # 6.16 seconds = 147840 samples -> 77 tokens
    assert compute_num_audio_input_tokens(147840, sampling_rate=24000, frame_rate=12.5) == 77
    # 8 seconds = 192000 samples -> 100 tokens
    assert compute_num_audio_input_tokens(192000, sampling_rate=24000, frame_rate=12.5) == 100
    # 30 seconds = 720000 samples -> 375 tokens
    assert compute_num_audio_input_tokens(720000, sampling_rate=24000, frame_rate=12.5) == 375


def test_audio_encoder_expected_output_length_uses_frame_blocks_as_minimum():
    wrapper = object.__new__(Qwen3OmniAuTWrapper)
    wrapper.input_sample_rate = 24000
    wrapper.encoder_sample_rate = 16000
    wrapper.frame_rate = 12.5

    for num_samples, expected in [
        (0, 0),
        (1, 1),
        (1920, 1),
        (1921, 2),
        (147840, 80),
        (192000, 104),
        (720000, 390),
    ]:
        assert wrapper.compute_expected_output_length(num_samples) == expected


# ===================================================================
# Placeholders: tokenizer resolution
# ===================================================================


class _StubTokenizer:
    def __init__(self, token_id: int, speaker_text: str = SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN):
        self._token_id = token_id
        self._speaker_text = speaker_text

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if text == AUDIO_INPUT_PAD_TOKEN:
            return [self._token_id]
        if text == self._speaker_text:
            return [SPEAKER_EMBEDDING_PLACEHOLDER_ID]
        if text == "<tts_pad>":
            return [151691]
        return [0]

    def decode(self, token_ids: list[int], **kwargs) -> str:
        if token_ids == [SPEAKER_EMBEDDING_PLACEHOLDER_ID]:
            return self._speaker_text
        if token_ids == [151691]:
            return "<tts_pad>"
        return "<unk>"


def test_audio_input_token_id_matches_tokenizer_mapping():
    tokenizer = _StubTokenizer(token_id=151676)
    token_id = resolve_audio_input_token_id(tokenizer, expected_audio_input_token_id=151676)
    assert token_id == 151676


def test_audio_input_token_id_mismatch_raises():
    tokenizer = _StubTokenizer(token_id=999)
    with pytest.raises(ValueError, match="audio_input_token_id mismatch"):
        resolve_audio_input_token_id(tokenizer, expected_audio_input_token_id=151676)


def test_speaker_token_id_prefers_canonical_placeholder_id():
    tokenizer = _StubTokenizer(token_id=151676)
    assert resolve_speaker_token_id(tokenizer) == SPEAKER_EMBEDDING_PLACEHOLDER_ID


def test_speaker_token_text_uses_tokenizer_surface_form():
    tokenizer = _StubTokenizer(token_id=151676, speaker_text="<tts_pad>")
    assert resolve_speaker_token_text(tokenizer) == "<tts_pad>"


# ===================================================================
# Placeholders: embedding scatter
# ===================================================================


def test_embedding_alignment_scatter_success():
    audio_input_token_id = 151676
    input_ids = torch.tensor([10, audio_input_token_id, 20, audio_input_token_id], dtype=torch.long)
    inputs_embeds = torch.zeros((4, 3), dtype=torch.float32)

    raw_mm = [
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
    ]
    mm_flat = flatten_audio_embeddings(
        raw_mm,
        hidden_size=3,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )

    scattered = scatter_audio_input_embeddings(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        audio_input_embeddings=mm_flat,
        audio_input_token_id=audio_input_token_id,
        is_multimodal=torch.tensor([False, True, False, True]),
    )
    assert torch.allclose(scattered[1], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(scattered[3], torch.tensor([2.0, 2.0, 2.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to cover CPU/CUDA mask mismatch")
def test_embedding_alignment_moves_cpu_multimodal_mask_to_input_device():
    audio_input_token_id = 151676
    device = torch.device("cuda")
    input_ids = torch.tensor([10, audio_input_token_id, 20, audio_input_token_id], dtype=torch.long, device=device)
    inputs_embeds = torch.zeros((4, 3), dtype=torch.float32, device=device)
    audio_embeddings = torch.tensor(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        dtype=torch.float32,
        device=device,
    )
    is_multimodal = torch.tensor([False, True, False, True], dtype=torch.bool)

    scattered = scatter_audio_input_embeddings(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        audio_input_embeddings=audio_embeddings,
        audio_input_token_id=audio_input_token_id,
        is_multimodal=is_multimodal,
    )

    assert torch.allclose(scattered[1].cpu(), torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(scattered[3].cpu(), torch.tensor([2.0, 2.0, 2.0]))


def test_embedding_alignment_placeholder_count_mismatch_raises():
    audio_input_token_id = 151676
    input_ids = torch.tensor([10, audio_input_token_id, 20], dtype=torch.long)
    inputs_embeds = torch.zeros((3, 2), dtype=torch.float32)
    mm_flat = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="Audio embedding alignment error"):
        scatter_audio_input_embeddings(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            audio_input_embeddings=mm_flat,
            audio_input_token_id=audio_input_token_id,
            is_multimodal=None,
        )


def test_build_audio_input_placeholder_ids_uses_secondary_audio_pad():
    ids = build_audio_input_placeholder_ids(
        num_audio_tokens=3,
        audio_start_token_id=151669,
        audio_input_token_id=151676,
        audio_end_token_id=151670,
    )
    assert ids == [151669, 151676, 151676, 151676, 151670]


def test_infer_audio_placeholder_token_ids_treats_legacy_input_as_input():
    token_ids = _infer_audio_placeholder_token_ids(
        "<|audio_start|><|audio_pad|><|audio_end|><|audio_start|><|audio_output_placeholder|><|audio_end|>",
        num_audios=2,
        audio_input_token_id=151676,
        audio_output_token_id=151675,
    )
    assert torch.equal(token_ids, torch.tensor([151676, 151675], dtype=torch.long))
