# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the non-Realtime VibeVoice TTS HF-schema config shim."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, Qwen2Config
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.vibevoice_acoustic_tokenizer.configuration_vibevoice_acoustic_tokenizer import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceAcousticTokenizerEncoderConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

_existing_acoustic_config = CONFIG_MAPPING.get("vibevoice_acoustic_tokenizer")

from vllm_omni.config.omni_config import _stage_sampling_params  # noqa: E402
from vllm_omni.config.pipeline_registry import resolve_pipeline_config  # noqa: E402
from vllm_omni.config.stage_config import StageExecutionType, load_deploy_config  # noqa: E402
from vllm_omni.engine.arg_utils import _resolve_vibevoice_tokenizer_contract  # noqa: E402
from vllm_omni.model_executor.models.vibevoice.pipeline import (  # noqa: E402
    VIBEVOICE_PIPELINE,
    VIBEVOICE_VALID_TOKEN_IDS,
)
from vllm_omni.model_executor.models.vibevoice.runtime_config import (  # noqa: E402
    VIBEVOICE_DEFAULT_GUIDANCE_SCALE,
    VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS,
    VIBEVOICE_MAX_DIFFUSION_GRAPH_BATCH_SIZE,
    VIBEVOICE_MAX_GUIDANCE_SCALE,
    VIBEVOICE_MAX_NUM_DIFFUSION_STEPS,
    VibeVoiceRuntimeConfig,
)
from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceConfig  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _official_checkpoint_config() -> dict:
    """Return the structural fields from Microsoft's original 1.5B checkpoint."""
    return {
        "acoustic_vae_dim": 64,
        "acoustic_tokenizer_config": {
            "causal": True,
            "channels": 1,
            "conv_bias": True,
            "conv_norm": "none",
            "corpus_normalize": 0.0,
            "decoder_depths": None,
            "decoder_n_filters": 32,
            "decoder_ratios": [8, 5, 5, 4, 2, 2],
            "disable_last_norm": True,
            "encoder_depths": "3-3-3-3-3-3-8",
            "encoder_n_filters": 32,
            "encoder_ratios": [8, 5, 5, 4, 2, 2],
            "fix_std": 0.5,
            "layer_scale_init_value": 1e-6,
            "layernorm": "RMSNorm",
            "layernorm_elementwise_affine": True,
            "layernorm_eps": 1e-5,
            "mixer_layer": "depthwise_conv",
            "model_type": "vibevoice_acoustic_tokenizer",
            "pad_mode": "constant",
            "std_dist_type": "gaussian",
            "vae_dim": 64,
            "weight_init_value": 0.01,
        },
        "architectures": ["VibeVoiceForConditionalGeneration"],
        "decoder_config": {
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "hidden_size": 1536,
            "initializer_range": 0.02,
            "intermediate_size": 8960,
            "max_position_embeddings": 65536,
            "max_window_layers": 28,
            "model_type": "qwen2",
            "num_attention_heads": 12,
            "num_hidden_layers": 28,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_scaling": None,
            "rope_theta": 1_000_000.0,
            "sliding_window": None,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 151936,
        },
        "diffusion_head_config": {
            "ddpm_batch_mul": 4,
            "ddpm_beta_schedule": "cosine",
            "ddpm_num_inference_steps": 20,
            "ddpm_num_steps": 1000,
            "diffusion_type": "ddpm",
            "head_ffn_ratio": 3.0,
            "head_layers": 4,
            "hidden_size": 1536,
            "latent_size": 64,
            "model_type": "vibevoice_diffusion_head",
            "prediction_type": "v_prediction",
            "rms_norm_eps": 1e-5,
            "speech_vae_dim": 64,
        },
        "model_type": "vibevoice",
        "semantic_tokenizer_config": {
            "causal": True,
            "channels": 1,
            "conv_bias": True,
            "conv_norm": "none",
            "corpus_normalize": 0.0,
            "disable_last_norm": True,
            "encoder_depths": "3-3-3-3-3-3-8",
            "encoder_n_filters": 32,
            "encoder_ratios": [8, 5, 5, 4, 2, 2],
            "fix_std": 0,
            "layer_scale_init_value": 1e-6,
            "layernorm": "RMSNorm",
            "layernorm_elementwise_affine": True,
            "layernorm_eps": 1e-5,
            "mixer_layer": "depthwise_conv",
            "model_type": "vibevoice_semantic_tokenizer",
            "pad_mode": "constant",
            "std_dist_type": "none",
            "vae_dim": 128,
            "weight_init_value": 0.01,
        },
        "semantic_vae_dim": 128,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.3",
        "custom_checkpoint_field": "preserved",
    }


def _hf_checkpoint_config() -> dict:
    """Return the structural fields from the converted 1.5B HF checkpoint."""
    return {
        "architectures": ["VibeVoiceForConditionalGeneration"],
        "audio_bos_token_id": 151652,
        "audio_config": {
            "channels": 1,
            "depths": [3, 3, 3, 3, 3, 3, 8],
            "downsampling_ratios": [2, 2, 4, 5, 5, 8],
            "ffn_expansion": 4,
            "hidden_act": "gelu",
            "hidden_size": 64,
            "initializer_range": 0.01,
            "kernel_size": 7,
            "layer_scale_init_value": 1e-6,
            "model_type": "vibevoice_acoustic_tokenizer",
            "num_filters": 32,
            "rms_norm_eps": 1e-5,
            "vae_std": 0.625,
            "weight_init_value": 0.01,
        },
        "audio_eos_token_id": 151653,
        "audio_token_id": 151654,
        "ddpm_beta_schedule": "squaredcos_cap_v2",
        "ddpm_num_inference_steps": 20,
        "ddpm_num_steps": 1000,
        "diffusion_max_period": 10000,
        "dtype": "bfloat16",
        "eos_token_id": 151643,
        "frequency_embedding_size": 256,
        "hidden_act": "silu",
        "intermediate_size": 4608,
        "mlp_bias": False,
        "model_type": "vibevoice",
        "num_head_layers": 4,
        "pad_token_id": 151643,
        "prediction_type": "v_prediction",
        "rms_norm_eps": 1e-5,
        "semantic_model_config": {
            "channels": 1,
            "depths": [3, 3, 3, 3, 3, 3, 8],
            "downsampling_ratios": [2, 2, 4, 5, 5, 8],
            "ffn_expansion": 4,
            "hidden_act": "gelu",
            "hidden_size": 128,
            "initializer_range": 0.01,
            "kernel_size": 7,
            "layer_scale_init_value": 1e-6,
            "model_type": "vibevoice_acoustic_tokenizer_encoder",
            "num_filters": 32,
            "rms_norm_eps": 1e-5,
            "vae_std": 0.625,
            "weight_init_value": 0.01,
        },
        "text_config": {
            "attention_dropout": 0.0,
            "dtype": "bfloat16",
            "hidden_act": "silu",
            "hidden_size": 1536,
            "initializer_range": 0.02,
            "intermediate_size": 8960,
            "max_position_embeddings": 65536,
            "max_window_layers": 28,
            "model_type": "qwen2",
            "num_attention_heads": 12,
            "num_hidden_layers": 28,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 1_000_000.0, "rope_type": "default"},
            "sliding_window": None,
            "tie_word_embeddings": True,
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 151936,
        },
        "tie_word_embeddings": True,
        "transformers_version": "5.13.0.dev0",
        "vocab_size": 151936,
        "custom_checkpoint_field": "preserved",
    }


@pytest.fixture
def checkpoint_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_official_checkpoint_config()), encoding="utf-8")
    return tmp_path


@pytest.fixture
def hf_checkpoint_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("vibevoice_hf_config")
    (path / "config.json").write_text(json.dumps(_hf_checkpoint_config()), encoding="utf-8")
    return path


def test_vibevoice_tokenizer_fallback_uses_preprocessor_contract(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"language_model_pretrained_name": "example/Qwen-tokenizer"}),
        encoding="utf-8",
    )

    assert _resolve_vibevoice_tokenizer_contract(str(tmp_path)) == "example/Qwen-tokenizer"


def test_vibevoice_tokenizer_contract_rejects_missing_model_name(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="language_model_pretrained_name"):
        _resolve_vibevoice_tokenizer_contract(str(tmp_path))


def test_vibevoice_tokenizer_contract_uses_vllm_hf_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str | None]] = []

    def fake_get_hf_file_to_dict(
        file_name: str,
        model: str,
        revision: str | None,
    ) -> dict[str, str]:
        calls.append((file_name, model, revision))
        return {"language_model_pretrained_name": "example/remote-tokenizer"}

    monkeypatch.setattr(
        "vllm_omni.engine.arg_utils.get_hf_file_to_dict",
        fake_get_hf_file_to_dict,
    )

    tokenizer = _resolve_vibevoice_tokenizer_contract("example/vibevoice", revision="test-revision")

    assert tokenizer == "example/remote-tokenizer"
    assert calls == [("preprocessor_config.json", "example/vibevoice", "test-revision")]


def test_vibevoice_tokenizer_contract_returns_none_when_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm_omni.engine.arg_utils.get_hf_file_to_dict",
        lambda *_args, **_kwargs: None,
    )

    assert _resolve_vibevoice_tokenizer_contract("example/vibevoice") is None


def test_vibevoice_tokenizer_contract_returns_none_when_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_resolution(*_args: object, **_kwargs: object) -> None:
        raise OSError("offline")

    monkeypatch.setattr(
        "vllm_omni.engine.arg_utils.get_hf_file_to_dict",
        fail_resolution,
    )

    assert _resolve_vibevoice_tokenizer_contract("example/vibevoice") is None


def test_vibevoice_tokenizer_contract_rejects_non_object_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm_omni.engine.arg_utils.get_hf_file_to_dict",
        lambda *_args, **_kwargs: ["not", "an", "object"],
    )

    with pytest.raises(ValueError, match="must contain a JSON object"):
        _resolve_vibevoice_tokenizer_contract("example/vibevoice")


def test_vibevoice_tokenizer_contract_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_json_decode(*_args: object, **_kwargs: object) -> None:
        raise json.JSONDecodeError("invalid", "{", 1)

    monkeypatch.setattr(
        "vllm_omni.engine.arg_utils.get_hf_file_to_dict",
        fail_json_decode,
    )

    with pytest.raises(ValueError, match="Invalid VibeVoice preprocessor config"):
        _resolve_vibevoice_tokenizer_contract("example/vibevoice")


def test_pipeline_is_registered_as_single_stage_ar_audio_generation():
    pipeline = resolve_pipeline_config("vibevoice", VibeVoiceConfig.from_dict(_official_checkpoint_config()))

    assert pipeline is VIBEVOICE_PIPELINE
    assert pipeline.model_type == "vibevoice"
    assert pipeline.model_arch == "VibeVoiceForConditionalGeneration"
    assert pipeline.default_deploy_config_name == "vibevoice.yaml"
    assert pipeline.validate() == []
    assert len(pipeline.stages) == 1

    stage = pipeline.stages[0]
    assert stage.stage_id == 0
    assert stage.model_stage == "vibevoice"
    assert stage.execution_type is StageExecutionType.LLM_AR
    assert stage.input_sources == ()
    assert stage.final_output is True
    assert stage.final_output_type == "audio"
    assert stage.owns_tokenizer is True
    assert stage.requires_multimodal_data is True
    assert stage.engine_output_type == "audio"
    assert stage.sampling_constraints == {
        "detokenize": False,
        "allowed_token_ids": VIBEVOICE_VALID_TOKEN_IDS,
        "stop_token_ids": [151643],
    }


def test_pr_token_gate_uses_eos_as_the_only_pipeline_stop():
    config = VibeVoiceConfig.from_dict(_official_checkpoint_config())
    pr_valid_token_ids = [
        config.audio_bos_token_id,
        config.audio_eos_token_id,
        config.audio_token_id,
        config.eos_token_id,
    ]

    assert pr_valid_token_ids == VIBEVOICE_VALID_TOKEN_IDS
    assert VIBEVOICE_PIPELINE.stages[0].sampling_constraints["stop_token_ids"] == [config.eos_token_id]
    assert config.audio_eos_token_id not in VIBEVOICE_PIPELINE.stages[0].sampling_constraints["stop_token_ids"]


def test_vllm_allowed_token_ids_matches_pr_logits_processor_semantics():
    params = SamplingParams(
        temperature=0.0,
        allowed_token_ids=VIBEVOICE_VALID_TOKEN_IDS,
    )
    assert params.allowed_token_ids == VIBEVOICE_VALID_TOKEN_IDS

    vocab_size = 151936
    mask = torch.ones((1, vocab_size), dtype=torch.bool)
    mask[:, VIBEVOICE_VALID_TOKEN_IDS] = False
    metadata = SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0),
        presence_penalties=torch.empty(0),
        repetition_penalties=torch.empty(0),
        output_token_ids=[[]],
        allowed_token_ids_mask=mask,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    logits = torch.zeros((1, vocab_size), dtype=torch.float32)
    filtered = Sampler().apply_logits_processors(
        logits,
        metadata,
        predict_bonus_token=False,
    )

    assert torch.isfinite(filtered[0, VIBEVOICE_VALID_TOKEN_IDS]).all()
    disallowed = torch.ones(vocab_size, dtype=torch.bool)
    disallowed[VIBEVOICE_VALID_TOKEN_IDS] = False
    assert torch.isneginf(filtered[0, disallowed]).all()


def test_single_stage_deploy_defaults_match_vibevoice_generation_contract():
    deploy_path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "vibevoice.yaml"
    deploy = load_deploy_config(deploy_path)

    assert deploy.async_chunk is False
    assert deploy.trust_remote_code is False
    assert deploy.dtype == "bfloat16"
    assert deploy.enable_prefix_caching is False
    assert deploy.enable_chunked_prefill is True
    assert len(deploy.stages) == 1

    stage = deploy.stages[0]
    assert stage.stage_id == 0
    assert stage.devices == "0"
    assert stage.tensor_parallel_size == 1
    assert stage.enforce_eager is True
    assert stage.async_scheduling is True
    assert stage.max_model_len == 65536
    assert stage.max_num_seqs == 4
    assert stage.engine_extras["kv_cache_memory_bytes"] == 8 * 1024**3
    assert stage.engine_extras["additional_config"] == {
        "vibevoice_runtime_config": {
            "negative_kv_cache_memory_bytes": 8 * 1024**3,
            "negative_kv_activation_margin_bytes": 512 * 1024**2,
            "diffusion_cuda_graph": True,
            "decode_cuda_graph": True,
        }
    }
    assert stage.engine_extras["limit_mm_per_prompt"] == {"audio": 8}
    assert stage.skip_mm_profiling is True
    assert stage.default_sampling_params == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "max_tokens": 40500,
        "extra_args": {"guidance_scale": 1.3, "num_diffusion_steps": 10},
    }
    assert stage.default_sampling_params["extra_args"] == {
        "guidance_scale": VIBEVOICE_DEFAULT_GUIDANCE_SCALE,
        "num_diffusion_steps": VIBEVOICE_DEFAULT_NUM_DIFFUSION_STEPS,
    }
    assert stage.max_num_seqs == VIBEVOICE_MAX_DIFFUSION_GRAPH_BATCH_SIZE
    assert VIBEVOICE_MAX_GUIDANCE_SCALE == 20.0
    assert VIBEVOICE_MAX_NUM_DIFFUSION_STEPS == 50

    effective = _stage_sampling_params(stage, VIBEVOICE_PIPELINE.stages[0])
    sampling_params = SamplingParams(**effective)
    assert sampling_params.allowed_token_ids == VIBEVOICE_VALID_TOKEN_IDS
    assert sampling_params.stop_token_ids == [151643]
    assert sampling_params.detokenize is False

    # Verify the deploy-level runtime config is parsed correctly by the model.
    runtime_config = VibeVoiceRuntimeConfig.from_vllm_config(
        SimpleNamespace(additional_config=stage.engine_extras["additional_config"])
    )
    assert runtime_config.diffusion_cuda_graph is True
    assert runtime_config.decode_cuda_graph is True
    assert stage.enforce_eager is True


def _runtime_config(**values: object) -> VibeVoiceRuntimeConfig:
    return VibeVoiceRuntimeConfig.from_vllm_config(
        SimpleNamespace(additional_config={"vibevoice_runtime_config": values})
    )


def test_runtime_config_is_deployment_only_and_resolves_warmup_batches() -> None:
    default = VibeVoiceRuntimeConfig.from_vllm_config(
        SimpleNamespace(
            additional_config={},
            model_config=SimpleNamespace(hf_config=SimpleNamespace(vibevoice_runtime_config={"wrong": 1})),
        )
    )
    assert default.negative_kv_cache_memory_bytes == 4 * 1024**3
    assert default.resolve_diffusion_graph_warmup_batch_sizes(4) == (1, 2, 3, 4)

    explicit = _runtime_config(
        negative_kv_cache_memory_bytes=4096,
        negative_kv_activation_margin_bytes=128,
        diffusion_graph_warmup_batch_sizes=[4, 1, 3, 3],
        diffusion_cuda_graph=False,
        decode_cuda_graph=False,
        cuda_graph_capture_failure_fatal=True,
    )
    assert explicit.negative_kv_cache_memory_bytes == 4096
    assert explicit.negative_kv_activation_margin_bytes == 128
    assert explicit.resolve_diffusion_graph_warmup_batch_sizes(4) == (1, 3, 4)
    assert explicit.diffusion_cuda_graph is False
    assert explicit.decode_cuda_graph is False
    assert explicit.cuda_graph_capture_failure_fatal is True

    assert _runtime_config(diffusion_graph_warmup_batch_sizes=[]).resolve_diffusion_graph_warmup_batch_sizes(4) == ()


def test_runtime_config_rejects_warmup_above_concurrency() -> None:
    config = _runtime_config(diffusion_graph_warmup_batch_sizes=[1, 5])
    with pytest.raises(ValueError, match=r"\[5\] exceed max_num_seqs=4"):
        config.resolve_diffusion_graph_warmup_batch_sizes(4)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"future_key": True}, "Unknown VibeVoice runtime config keys"),
        ({"diffusion_graph_warmup_batch_sizes": 1}, "must be a list or tuple"),
        ({"diffusion_graph_warmup_batch_sizes": [0]}, "positive integers"),
        ({"diffusion_cuda_graph": 0}, "must be a bool"),
        ({"negative_kv_cache_memory_bytes": 0}, "must be positive"),
        ({"negative_kv_cache_memory_bytes": True}, "must be an integer, not bool"),
        ({"negative_kv_activation_margin_bytes": -1}, "must be non-negative"),
    ],
)
def test_runtime_config_rejects_invalid_values(values: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _runtime_config(**values)


def test_runtime_config_rejects_non_mapping_schema() -> None:
    config = SimpleNamespace(additional_config={"vibevoice_runtime_config": "not-a-mapping"})
    with pytest.raises(ValueError, match="must be a mapping"):
        VibeVoiceRuntimeConfig.from_vllm_config(config)


def test_auto_config_normalizes_official_checkpoint_without_remote_code(checkpoint_dir):
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=False)

    assert isinstance(config, VibeVoiceConfig)
    assert config.model_type == "vibevoice"
    assert config.architectures == ["VibeVoiceForConditionalGeneration"]
    assert config.custom_checkpoint_field == "preserved"
    assert not hasattr(config, "diffusion_head_config")


def test_converted_hf_schema_is_also_accepted(hf_checkpoint_dir):
    config = AutoConfig.from_pretrained(hf_checkpoint_dir, trust_remote_code=False)

    assert isinstance(config, VibeVoiceConfig)
    assert isinstance(config.audio_config, VibeVoiceAcousticTokenizerConfig)
    assert isinstance(config.semantic_model_config, VibeVoiceAcousticTokenizerEncoderConfig)
    assert isinstance(config.text_config, Qwen2Config)


def test_hf_sub_configs_are_objectified(checkpoint_dir):
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=False)

    assert isinstance(config.audio_config, VibeVoiceAcousticTokenizerConfig)
    assert isinstance(config.semantic_model_config, VibeVoiceAcousticTokenizerEncoderConfig)
    assert isinstance(config.text_config, Qwen2Config)
    assert config.audio_config.hidden_size == 64
    assert config.semantic_model_config.hidden_size == 128


def test_text_and_diffusion_parameters_match_pr_contract(checkpoint_dir):
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=False)
    text_config = config.get_text_config()
    serialized = config.to_dict()

    assert text_config is config.text_config
    assert str(config.dtype).removeprefix("torch.") == "bfloat16"
    assert "torch_dtype" not in serialized
    assert config.pad_token_id == 151643
    assert config.eos_token_id == 151643
    assert config.audio_bos_token_id == 151652
    assert config.audio_eos_token_id == 151653
    assert config.audio_token_id == 151654
    assert config.hidden_size == 1536
    assert text_config.num_hidden_layers == 28
    assert text_config.num_attention_heads == 12
    assert text_config.num_key_value_heads == 2
    assert config.num_head_layers == 4
    assert config.intermediate_size == 4608
    assert config.frequency_embedding_size == 256
    assert config.diffusion_max_period == 10000
    assert config.prediction_type == "v_prediction"
    assert config.ddpm_num_steps == 1000
    assert config.ddpm_num_inference_steps == 20
    assert config.ddpm_beta_schedule == "squaredcos_cap_v2"


def test_tensor_parallel_plan_uses_pr_module_names():
    assert VibeVoiceConfig.base_model_tp_plan["language_model.layers.*.self_attn.q_proj"] == "colwise"
    assert VibeVoiceConfig.base_model_tp_plan["language_model.layers.*.self_attn.o_proj"] == "rowwise"


def test_config_round_trip_is_lossless(checkpoint_dir):
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=False)
    serialized = config.to_dict()
    restored = VibeVoiceConfig.from_dict(serialized)

    assert restored.to_dict() == serialized
    assert isinstance(restored.audio_config, VibeVoiceAcousticTokenizerConfig)
    assert isinstance(restored.semantic_model_config, VibeVoiceAcousticTokenizerEncoderConfig)
    assert isinstance(restored.text_config, Qwen2Config)


@pytest.mark.parametrize(
    ("original_key", "runtime_key"),
    [
        ("acoustic_tokenizer_config", "audio_config"),
        ("semantic_tokenizer_config", "semantic_model_config"),
        ("decoder_config", "text_config"),
    ],
)
def test_mixed_original_and_runtime_child_schemas_are_rejected(original_key, runtime_key):
    source = _official_checkpoint_config()
    source[runtime_key] = {"model_type": source[original_key]["model_type"]}

    with pytest.raises(ValueError, match=f"both `{original_key}` and `{runtime_key}`"):
        VibeVoiceConfig.from_dict(source)


def test_constructor_does_not_mutate_input_dicts_or_lists():
    source = _official_checkpoint_config()
    original = copy.deepcopy(source)

    config = VibeVoiceConfig.from_dict(source)
    config.audio_config.depths.append(99)
    config.semantic_model_config.downsampling_ratios.append(99)
    config.text_config.rope_parameters["rope_theta"] = 1.0

    assert source == original


def test_upstream_child_registrations_are_reused(checkpoint_dir):
    assert CONFIG_MAPPING["vibevoice_acoustic_tokenizer"] is not None
    if _existing_acoustic_config is not None:
        assert CONFIG_MAPPING["vibevoice_acoustic_tokenizer"] is _existing_acoustic_config

    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=False)
    assert type(config.audio_config) is VibeVoiceAcousticTokenizerConfig
    assert type(config.semantic_model_config) is VibeVoiceAcousticTokenizerEncoderConfig


def test_vllm_config_resolver_loads_vibevoice(checkpoint_dir):
    from vllm.transformers_utils.config import get_config

    config = get_config(str(checkpoint_dir), trust_remote_code=False)

    assert isinstance(config, VibeVoiceConfig)
    assert isinstance(config.get_text_config(), Qwen2Config)
    assert config.get_text_config().hidden_size == 1536


def test_registration_works_in_fresh_process(checkpoint_dir):
    code = f"""
from transformers import AutoConfig, Qwen2Config
import vllm_omni  # noqa: F401
from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceConfig
config = AutoConfig.from_pretrained({str(checkpoint_dir)!r}, trust_remote_code=False)
assert isinstance(config, VibeVoiceConfig)
assert isinstance(config.audio_config, CONFIG_MAPPING["vibevoice_acoustic_tokenizer"])
assert isinstance(config.get_text_config(), Qwen2Config)
assert config.hidden_size == 1536
"""
    # CONFIG_MAPPING is deliberately imported inside the isolated process so
    # this test does not rely on registrations inherited from pytest.
    code = "from transformers.models.auto.configuration_auto import CONFIG_MAPPING\n" + code
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
