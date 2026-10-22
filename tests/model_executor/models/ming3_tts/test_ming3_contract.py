# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm import SamplingParams
from vllm.config import CUDAGraphMode

from vllm_omni.model_executor.models.common.ming.cfm_cudagraph import CFMGraphExecutor
from vllm_omni.model_executor.models.common.ming.continuous_tts import (
    STOP_REASON_CONTINUE,
    STOP_REASON_MAX_DECODE_STEPS,
    STOP_REASON_STOP_HEAD,
    resolve_stop_decision,
)
from vllm_omni.model_executor.models.ming3_tts.config_ming3_tts import (
    Ming3TTSConfig,
    Ming3TTSRuntimeConfig,
)
from vllm_omni.model_executor.models.ming3_tts.constants import (
    AUDIO_END_TOKEN_ID,
    AUDIO_PATCH_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_SPEAKER_POSITIONS,
    SPEAKER_END_TOKEN_ID,
    SPEAKER_START_TOKEN_ID,
)
from vllm_omni.model_executor.models.ming3_tts.ming3_tts import (
    Ming3TTSForConditionalGeneration,
)
from vllm_omni.model_executor.models.ming3_tts.ming3_tts_llm import (
    Ming3TTSLLMModel,
)
from vllm_omni.model_executor.models.ming3_tts.prompt_assembly import (
    ASSISTANT_PREFIX,
    TEXT_PREFIX,
    USER_PREFIX,
    build_ming3_prompt,
    build_ming3_prompt_token_ids,
    resolve_podcast_decode_window,
    validate_ming3_tokenizer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def _prefill_wrapper(model, config):
    wrapper = Ming3TTSForConditionalGeneration.__new__(Ming3TTSForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.ming_config = config
    wrapper._prefill_conditioning = {}
    return wrapper


class FakeTokenizer:
    eos_token_id = 156892
    unk_token_id = None
    special = {
        "<|endoftext|>": 156892,
        "<|role_end|>": 156895,
        "<audioPatch>": AUDIO_PATCH_TOKEN_ID,
        "<audio>": AUDIO_START_TOKEN_ID,
        "</audio>": AUDIO_END_TOKEN_ID,
        "<spk>": SPEAKER_START_TOKEN_ID,
        "</spk>": SPEAKER_END_TOKEN_ID,
    }

    def __init__(self):
        self.special = dict(self.special)

    def convert_tokens_to_ids(self, token):
        return self.special.get(token)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        import regex as re

        tokens = re.split("(" + "|".join(re.escape(token) for token in self.special) + ")", text)
        return [
            item
            for part in tokens
            for item in ([self.special[part]] if part in self.special else [1000 + ord(c) for c in part])
        ]


def _config() -> Ming3TTSConfig:
    backbone = {
        "model_type": "bailing_moe_v3",
        "vocab_size": 157184,
        "hidden_size": 1536,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "layer_group_size": 4,
        "head_dim": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "short_conv_kernel_size": 4,
        "kda_safe_gate": True,
        "kda_lower_bound": -5.0,
    }
    audio = {
        "sample_rate": 44100,
        "patch_size": 4,
        "semantic_module_kwargs": None,
        "enc_kwargs": {"hop_size": 882, "latent_dim": 64},
        "dec_kwargs": {"output_dim": 882, "latent_dim": 64},
    }
    return Ming3TTSConfig(
        llm_config=backbone,
        audio_config=audio,
        tts_config={
            "sample_rate": 44100,
            "latent_dim": 64,
            "speaker_embedding_dim": 192,
            "patch_size": 1,
            "history_size": 1,
            "audio_patch_token_id": AUDIO_PATCH_TOKEN_ID,
            "audio_start_token_id": AUDIO_START_TOKEN_ID,
            "audio_end_token_id": AUDIO_END_TOKEN_ID,
            "speaker_start_token_id": SPEAKER_START_TOKEN_ID,
            "speaker_end_token_id": SPEAKER_END_TOKEN_ID,
        },
        aggregator_config={"in_channels": 64, "hidden_size": 1024},
        flow_config={"z_channels": 64, "hidden_size": 1024},
    )


def test_config_normalizes_bailing_and_keeps_independent_geometry():
    config = _config()
    runtime = Ming3TTSRuntimeConfig.from_hf_config(config)
    runtime.validate()
    assert config.llm_config.model_type == "bailing_hybrid"
    assert config.llm_config.checkpoint_model_type == "bailing_moe_v3"
    assert config.llm_config.kda_safe_gate is False
    assert config.llm_config.kda_lower_bound is None
    assert config.llm_config.kv_lora_rank + config.llm_config.qk_rope_head_dim == 576
    assert runtime.patch_size == runtime.history_patch_size == 1
    assert runtime.vae_patch_size == 4
    assert runtime.cfg == 1.0
    assert runtime.temperature == 0.0
    assert runtime.max_decode_steps == 800


@pytest.mark.parametrize("text", ["target", "Genre: x Mood: x Instrument: x Theme: x Duration: 3s"])
def test_prompt_contract_speaker_only(text):
    tokenizer = FakeTokenizer()
    ids, speaker_position, prompt_position = build_ming3_prompt_token_ids(tokenizer, target_text=text)
    expected = (
        tokenizer.encode(USER_PREFIX)
        + [SPEAKER_START_TOKEN_ID, AUDIO_PATCH_TOKEN_ID, SPEAKER_END_TOKEN_ID]
        + tokenizer.encode(TEXT_PREFIX)
        + tokenizer.encode(text)
        + tokenizer.encode(ASSISTANT_PREFIX)
        + [AUDIO_START_TOKEN_ID]
    )
    assert ids == expected
    assert ids[speaker_position] == AUDIO_PATCH_TOKEN_ID
    assert prompt_position is None


def test_prompt_contract_concatenates_reference_and_target_without_separator():
    tokenizer = FakeTokenizer()
    ids, _, prompt_position = build_ming3_prompt_token_ids(
        tokenizer,
        target_text="target",
        prompt_text="reference",
        prompt_latent_steps=3,
    )
    joined = tokenizer.encode("reference") + tokenizer.encode("target")
    text_start = len(tokenizer.encode(USER_PREFIX)) + 3 + len(tokenizer.encode(TEXT_PREFIX))
    assert ids[text_start : text_start + len(joined)] == joined
    assert ids[prompt_position:] == [AUDIO_PATCH_TOKEN_ID] * 3


def test_prefill_uses_vendor_prompt_posterior_and_continues_request_rng(monkeypatch):
    generator = torch.Generator().manual_seed(17)
    calls: dict[str, Any] = {}

    def get_request_generator(info, device):
        calls["generator_request"] = (info, device)
        return generator

    def resolve_prompt_latents(wrapper, info, **kwargs):
        calls["prompt"] = (wrapper, info, kwargs)
        return None

    from vllm_omni.model_executor.models.ming3_tts import ming3_tts as model_module

    monkeypatch.setattr(model_module, "_resolve_prompt_latents", resolve_prompt_latents)
    monkeypatch.setattr(model_module, "resolve_speaker_embeddings", lambda wrapper, info: [torch.zeros(192)])
    monkeypatch.setattr(model_module, "_find_speaker_placeholder_positions", lambda input_ids, config: [0])

    model = SimpleNamespace(
        _get_request_generator=get_request_generator,
        fm_dtype=torch.float32,
        project_speaker_embedding=lambda embedding: torch.zeros((1, 4)),
    )
    wrapper = _prefill_wrapper(model, SimpleNamespace(history_patch_size=1, latent_dim=64))
    info = {"prompt_waveform": torch.ones((1, 8)), "request_id": "request-1"}

    Ming3TTSForConditionalGeneration._prefill_preprocess(
        wrapper,
        torch.tensor([AUDIO_PATCH_TOKEN_ID]),
        torch.zeros((1, 4)),
        info,
    )

    assert calls["generator_request"] == (info, torch.device("cpu"))
    assert calls["prompt"][2] == {"generator": generator}


@pytest.mark.parametrize(
    ("info", "expected_seed"),
    [
        pytest.param({"request_id": "request-1", "tts_local_seed": 1234}, 1234, id="explicit-seed"),
        pytest.param({"request_id": "request-1"}, None, id="unseeded-speaker"),
    ],
)
def test_request_generator_seed_policy(info, expected_seed):
    model = SimpleNamespace(_request_generators={})
    generator = Ming3TTSLLMModel._get_request_generator(
        model,
        info,
        torch.device("cpu"),
    )

    actual_seed = None if generator is None else generator.initial_seed()
    assert actual_seed == expected_seed


def test_tokenizer_mismatch_is_not_repaired():
    tokenizer = FakeTokenizer()
    tokenizer.special["<audio>"] = -1
    with pytest.raises(ValueError, match="expected 157165"):
        validate_ming3_tokenizer(tokenizer)


@pytest.mark.parametrize(
    ("emitted", "probability", "reason", "token"),
    [
        (4, 1.0, STOP_REASON_CONTINUE, AUDIO_PATCH_TOKEN_ID),
        (5, 0.51, STOP_REASON_STOP_HEAD, AUDIO_END_TOKEN_ID),
        (800, 0.0, STOP_REASON_MAX_DECODE_STEPS, AUDIO_END_TOKEN_ID),
    ],
)
def test_stop_is_evaluated_after_emitting_latent(emitted, probability, reason, token):
    decision = resolve_stop_decision(
        emitted_steps=emitted,
        stop_probability=probability,
        stop_threshold=0.5,
        minimum_steps=5,
        maximum_steps=800,
        continue_token_id=AUDIO_PATCH_TOKEN_ID,
        stop_token_id=AUDIO_END_TOKEN_ID,
    )
    assert decision.reason == reason
    assert decision.next_token_id == token


def test_runtime_rejects_flow_vae_patch_confusion():
    config = _config()
    config.audio_config.patch_size = 1
    runtime = Ming3TTSRuntimeConfig.from_hf_config(config)
    with pytest.raises(ValueError, match="AudioVAE patch_size must be 4"):
        runtime.validate()


def test_runtime_rejects_incompatible_mla_geometry():
    config = _config()
    config.llm_config.kv_lora_rank = 256
    runtime = Ming3TTSRuntimeConfig.from_hf_config(config)
    with pytest.raises(ValueError, match="expected MLA head size 576"):
        runtime.validate()


def test_pipeline_contract():
    from vllm_omni.model_executor.models.ming3_tts.pipeline import (
        MING3_TTS_PIPELINE,
    )

    stage0, stage1 = MING3_TTS_PIPELINE.stages
    assert stage0.hf_config_name == "llm_config"
    assert stage0.owns_tokenizer is True
    assert stage0.sampling_constraints["max_tokens"] == 801
    assert stage1.final_output is True
    assert stage1.final_output_type == "audio"


@pytest.mark.parametrize(
    "eager,graph_mode,env_var,env_value,enabled",
    [
        (False, CUDAGraphMode.PIECEWISE, None, None, True),
        (True, CUDAGraphMode.PIECEWISE, None, None, False),
        (False, CUDAGraphMode.NONE, None, None, False),
        (True, CUDAGraphMode.NONE, "MING_CFM_CUDAGRAPH", None, True),
        (True, CUDAGraphMode.NONE, "MING_CFM_CUDAGRAPH", "0", False),
        (True, CUDAGraphMode.NONE, "MING_CFM_CUDAGRAPH", "1", True),
    ],
)
def test_cfm_graph_enablement(monkeypatch, eager, graph_mode, env_var, env_value, enabled):
    model = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=eager),
            compilation_config=SimpleNamespace(cudagraph_mode=graph_mode),
        )
    )
    if env_var is not None:
        monkeypatch.delenv(env_var, raising=False)
        if env_value is not None:
            monkeypatch.setenv(env_var, env_value)
    Ming3TTSLLMModel._init_cfm_graph_state(model, env_var)
    assert model._cfm_graph_enabled is enabled
    if not enabled:
        # Disabled warmup must not even access the flow modules or build graphs.
        Ming3TTSLLMModel._warmup_cfm_graphs(model)
        assert Ming3TTSLLMModel._maybe_build_cfm_graph(model, torch.empty(1, 1, 4)) is None
        assert not model._cfm_graphs


def test_cfm_graph_bucket_zero_pads_without_touching_live_rows():
    executor = SimpleNamespace(batch_size=4)
    live = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    solver_noise = live.unsqueeze(0).expand(10, -1, -1)

    model = SimpleNamespace(_cfm_graph_batch_sizes=[1, 2, 4])
    assert Ming3TTSLLMModel._select_cfm_graph_batch_size(model, 3) == 4
    padded = CFMGraphExecutor._pad_batch(executor, live)
    padded_solver_noise = CFMGraphExecutor._pad_batch(executor, solver_noise, dim=1)

    assert padded.shape == (4, 2)
    assert padded_solver_noise.shape == (10, 4, 2)
    torch.testing.assert_close(padded[:3], live)
    torch.testing.assert_close(padded_solver_noise[:, :3], solver_noise)
    assert torch.count_nonzero(padded[3]) == 0
    assert torch.count_nonzero(padded_solver_noise[:, 3]) == 0


def test_ming3_vectorizes_only_compatible_flow_requests():
    config = SimpleNamespace(
        history_patch_size=2,
        latent_dim=3,
        patch_size=1,
        llm_hidden_size=4,
        cfg=2.0,
        sigma=0.25,
        temperature=0.0,
    )
    calls = []

    def get_request_generator(info, device):
        return torch.Generator(device=device).manual_seed(info["seed"])

    def decode_one_step(**kwargs):
        calls.append(kwargs)
        batch_size = kwargs["hidden_states"].shape[0]
        sampled = kwargs["initial_noise"].clone()
        next_embeds = torch.zeros((batch_size, 1, config.llm_hidden_size))
        history = torch.cat([kwargs["latent_history"][:, config.patch_size :], sampled], dim=1)
        stop_probs = torch.zeros(batch_size)
        return sampled, next_embeds, history, stop_probs

    model = SimpleNamespace(
        vectorize_cfm_decode=True,
        _cfm_graph_enabled=True,
        fm_dtype=torch.float32,
        ming_config=config,
        _get_request_generator=get_request_generator,
        _decode_one_step=decode_one_step,
    )
    model._draw_request_cfm_noises = lambda generators, device: (
        Ming3TTSLLMModel._draw_request_cfm_noises(model, generators, device)
    )
    history = torch.zeros((1, config.history_patch_size, config.latent_dim))
    request_infos = [
        {"ming_latent_history": history, "ming_cfg": 2.0, "seed": 11},
        {"ming_latent_history": history, "ming_cfg": 2.0, "seed": 22},
        {"ming_latent_history": history, "ming_cfg": 3.0, "seed": 33},
    ]

    results = Ming3TTSLLMModel._vectorized_decode_results(
        model,
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
        request_infos,
        [1, 1, 1],
        [True, True, True],
    )

    assert set(results) == {0, 1}
    assert len(calls) == 1
    assert calls[0]["hidden_states"].shape == (2, 4)
    assert calls[0]["latent_history"].shape == (2, 2, 3)
    assert calls[0]["initial_noise"].shape == (2, 1, 3)
    assert calls[0]["solver_noise"].shape == (10, 2, 1, 3)
    assert calls[0]["cfg_scale"] == 2.0
    assert not torch.equal(calls[0]["initial_noise"][0], calls[0]["initial_noise"][1])


def test_sft_config_preserves_gate_and_sampling_defaults():
    legacy = _config()
    backbone = legacy.llm_config.to_dict()
    backbone.update(kda_safe_gate=True, kda_lower_bound=-5.0)
    config = Ming3TTSConfig(
        llm_config=backbone,
        audio_config=legacy.audio_config,
        tts_config={
            **legacy.tts_config,
            "task_contract": "lingv3_sft_multitask_v1",
            "sampling_temperature": 1.5,
            "minimum_decode_steps": 13,
            "maximum_decode_steps": 250,
        },
    )
    runtime = Ming3TTSRuntimeConfig.from_hf_config(config)
    runtime.validate()
    assert config.llm_config.kda_safe_gate is True
    assert config.llm_config.kda_lower_bound == -5.0
    assert runtime.temperature == 1.5
    assert runtime.stop_head_min_steps == 13
    assert runtime.max_decode_steps == 250


def test_unknown_task_contract_is_rejected():
    with pytest.raises(ValueError, match="Unsupported Ming3 task contract"):
        Ming3TTSConfig(tts_config={"task_contract": "unknown"})


@pytest.mark.parametrize(
    "tts_config,ref_text,max_tokens,error",
    [
        ({}, "reference", 5, None),
        ({"conditioning_mode": "speaker_only"}, "reference", None, None),
        ({"conditioning_mode": "speaker_only"}, None, None, None),
        ({"minimum_decode_steps": 13}, None, 12, "at least 13"),
        ({"minimum_decode_steps": 13}, None, 13, None),
    ],
)
def test_adapter_validates_checkpoint_conditioning_and_minimum(tts_config, ref_text, max_tokens, error):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter

    server = SimpleNamespace(
        engine_client=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(tts_config=tts_config))),
        _apply_uploaded_speaker=lambda request: None,
        _validate_ref_audio_format=lambda source: None,
    )
    adapter = Ming3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(
        input="Hello", ref_audio="data:audio/wav;base64,AA==", ref_text=ref_text, max_new_tokens=max_tokens
    )
    result = adapter.validate(request)
    if error is None:
        assert result is None
    else:
        assert error in result


@pytest.mark.parametrize("patch_size,encoder_depth", [(4, 6), (4, 24), (-1, 6)])
def test_prompt_encoder_preserves_configured_depth_and_posterior_without_dtype_changes(patch_size, encoder_depth):
    from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE, AudioVAEConfig
    from vllm_omni.model_executor.models.ming_tts.prompt_encoder import _resolve_prompt_latents

    audio_vae = (
        AudioVAE(
            AudioVAEConfig(
                enc_kwargs={
                    "backbone": dict(
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=encoder_depth,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        vocab_size=1,
                        _attn_implementation="sdpa",
                        use_cache=False,
                    ),
                    "input_dim": 882,
                    "hop_size": 882,
                    "latent_dim": 2,
                },
                patch_size=patch_size,
            ),
            build_decoder=False,
        )
        .eval()
        .to(dtype=torch.bfloat16)
    )
    encoder = audio_vae.encoder
    assert encoder.encoder.config.num_hidden_layers == encoder_depth
    state = {name: value.clone() for name, value in audio_vae.state_dict().items()}
    buffers = {name: value.clone() for name, value in encoder.named_buffers()}
    wrapper = SimpleNamespace(
        _prompt_encoder=audio_vae,
        ming_config=SimpleNamespace(sample_rate=44100, patch_size=1, latent_dim=2),
    )
    visited, outputs = [], []
    hooks = [
        layer.register_forward_hook(lambda module, args, output, i=i: visited.append(i))
        for i, layer in enumerate(encoder.encoder.layers)
    ]
    hooks.append(encoder.register_forward_hook(lambda module, args, output: outputs.append(output[0])))
    try:
        with torch.inference_mode():
            actual = _resolve_prompt_latents(
                wrapper,
                {"prompt_waveform": torch.zeros(1, 3528)},
                generator=torch.Generator().manual_seed(17),
            )
        assert visited == list(range(encoder_depth))
        assert len(encoder.encoder.layers) == encoder_depth
        assert encoder.encoder.config.num_hidden_layers == encoder_depth
        if patch_size != -1:
            assert encoder.aggregator.config.num_hidden_layers == 4
        mean, scale = outputs[0].transpose(1, 2).chunk(2, dim=1)
        noise = torch.randn(mean.shape, dtype=mean.dtype, generator=torch.Generator().manual_seed(17))
        expected = (mean + (torch.nn.functional.softplus(scale) + 1e-4) * noise).transpose(1, 2)
        assert torch.equal(actual["frames"], expected.squeeze(0).float())
        assert actual["frames"].shape == (1 if patch_size == 4 else 4, 2)
        assert set(audio_vae.state_dict()) == set(state)
        for name, value in audio_vae.state_dict().items():
            assert value.dtype == state[name].dtype
            assert torch.equal(value, state[name])
        for name, value in encoder.named_buffers():
            assert value.dtype == buffers[name].dtype
            assert torch.equal(value, buffers[name])
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("eos_id", [156892, 156895, 123])
def test_prompt_eos_contract(eos_id):
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = eos_id
    tokenizer.special = {**tokenizer.special, "<|role_end|>": 156895}
    if eos_id == 123:
        with pytest.raises(ValueError, match="EOS"):
            build_ming3_prompt_token_ids(tokenizer, target_text="测试")
    else:
        ids, _, _ = build_ming3_prompt_token_ids(tokenizer, target_text="测试")
        assert ids[-1] == AUDIO_START_TOKEN_ID
        assert 156895 not in ids


@pytest.mark.parametrize(
    "controls", [{"语速": "快速", "基频": "中", "音量": "高"}, {"情感": "高兴"}, {"方言": "广粤话"}]
)
def test_control_caption_precedes_audio(controls):
    from vllm_omni.model_executor.models.ming_tts.prompt_assembly import create_instruction

    tokenizer = FakeTokenizer()
    tokenizer.special["<|endoftext|>"] = 156892
    caption = create_instruction(controls)
    actual, speaker_position, _ = build_ming3_prompt_token_ids(tokenizer, target_text="测试文本", instruction=caption)
    base, _, _ = build_ming3_prompt_token_ids(tokenizer, target_text="测试文本")
    expected = base[:-1] + tokenizer.encode(caption) + [156892, AUDIO_START_TOKEN_ID]
    assert actual == expected
    assert actual[speaker_position] == AUDIO_PATCH_TOKEN_ID
    assert actual[-2:] == [156892, AUDIO_START_TOKEN_ID]


@pytest.mark.parametrize("eos_id", [156892, 156895])
@pytest.mark.parametrize("prompt_steps", [0, 2])
@pytest.mark.parametrize("speaker_count", [0, 1])
def test_control_caption_uses_checkpoint_eos(eos_id, prompt_steps, speaker_count):
    from vllm_omni.model_executor.models.ming_tts.prompt_assembly import create_instruction

    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = eos_id
    tokenizer.special.update({"<|endoftext|>": 156892, "<|role_end|>": 156895})
    caption = create_instruction({"风格": "温柔<|endoftext|>舒缓"})
    kwargs = dict(
        target_text="测试文本",
        prompt_text="参考文本" if prompt_steps else None,
        prompt_latent_steps=prompt_steps,
        speaker_count=speaker_count,
    )
    base, _, _ = build_ming3_prompt_token_ids(tokenizer, **kwargs)
    actual, _, _ = build_ming3_prompt_token_ids(tokenizer, instruction=caption, **kwargs)
    audio_start = len(base) - prompt_steps - 1
    expected = base[:audio_start] + tokenizer.encode(caption) + [eos_id] + base[audio_start:]
    assert actual == expected
    assert actual[-prompt_steps - 2] == eos_id


def test_prompt_contract_requires_instruction_without_reference_audio():
    with pytest.raises(ValueError, match="requires an instruction"):
        build_ming3_prompt(
            FakeTokenizer(),
            target_text="target",
            reference_waveform=None,
            reference_sample_rate=24000,
        )


@pytest.mark.parametrize(
    "minimum,maximum,emitted,probability,reason",
    [
        (8, 16, 7, 1.0, STOP_REASON_CONTINUE),
        (8, 16, 8, 1.0, STOP_REASON_STOP_HEAD),
        (8, 16, 16, 0.0, STOP_REASON_MAX_DECODE_STEPS),
        (20, 32, 13, 1.0, STOP_REASON_CONTINUE),
        (None, 32, 12, 1.0, STOP_REASON_CONTINUE),
        (None, 32, 13, 1.0, STOP_REASON_STOP_HEAD),
    ],
)
def test_request_decode_window_overrides_checkpoint_minimum(minimum, maximum, emitted, probability, reason):
    from vllm_omni.model_executor.models.ming_tts import config_ming_tts as state_keys

    model = SimpleNamespace(ming_config=SimpleNamespace(stop_head_min_steps=13, max_decode_steps=250))
    info = {state_keys.KEY_MAX_DECODE_STEPS: maximum}
    if minimum is not None:
        info[state_keys.KEY_MIN_DECODE_STEPS] = minimum
    Ming3TTSLLMModel._validate_decode_window(model, [info])
    result = Ming3TTSLLMModel._resolve_stop_decision(
        model,
        step=emitted - 1,
        stop_prob=probability,
        stop_threshold=0.5,
        min_stop_step=13,
        min_decode_steps=info[state_keys.KEY_MIN_DECODE_STEPS],
        max_decode_steps=maximum,
        audio_dummy_token_id=AUDIO_PATCH_TOKEN_ID,
        text_eos_token_id=AUDIO_END_TOKEN_ID,
    )
    assert result[0] == reason


@pytest.mark.parametrize(
    "text,expected",
    [
        ("speaker_1:你好。\nspeaker_2:再见。", (8, 24)),
        ("speaker_1:Hello there.\nspeaker_2:How are you today?", (8, 48)),
        ("speaker_1:你好。\nspeaker_2:Hello there.", (8, 28)),
        ("\n speaker_1:a\n speaker_2:b\n", (8, 16)),
        ("speaker_1:" + "你" * 700 + "\nspeaker_2:好", (600, 600)),
    ],
)
def test_podcast_decode_window_matches_text_units(text, expected):
    assert resolve_podcast_decode_window(text) == expected


def test_podcast_explicit_limit_only_tightens_the_window():
    text = "speaker_1:你好。\nspeaker_2:再见。"
    assert resolve_podcast_decode_window(text, max_decode_steps=16) == (8, 16)
    assert resolve_podcast_decode_window(text, max_decode_steps=800) == (8, 24)
    with pytest.raises(ValueError, match="text-derived minimum 8"):
        resolve_podcast_decode_window(text, max_decode_steps=7)


def test_podcast_prompt_keeps_speaker_order_and_audio_boundaries():
    from vllm_omni.model_executor.models.ming3_tts.constants import (
        KEY_SPEAKER_SAMPLE_RATES,
        KEY_SPEAKER_WAVEFORM,
        KEY_SPEAKER_WAVEFORM_LENGTHS,
    )
    from vllm_omni.model_executor.models.ming_tts.audio_prep import _find_speaker_placeholder_positions

    tokenizer = FakeTokenizer()
    prompt = build_ming3_prompt(
        tokenizer,
        target_text=" speaker_1:你好。\n speaker_2:再见。\n",
        reference_waveform=[torch.ones(1, 3), torch.full((1, 5), 2.0)],
        reference_sample_rate=[16000, 24000],
        prompt_text=" speaker_1:参考一。\n speaker_2:参考二。\n",
    )
    ids = prompt["prompt_token_ids"]
    slots = _find_speaker_placeholder_positions(torch.tensor(ids), Ming3TTSRuntimeConfig.from_hf_config(_config()))
    assert len(slots) == 2
    assert ids[slots[0] + 2 : slots[1] - 1] == tokenizer.encode("\n  speaker_2:")
    assert ids[-1] == AUDIO_PATCH_TOKEN_ID
    info = prompt["additional_information"]
    assert (info[KEY_MIN_DECODE_STEPS], info[KEY_MAX_DECODE_STEPS]) == (8, 24)
    assert info[KEY_SPEAKER_WAVEFORM_LENGTHS].tolist() == [3, 5]
    assert info[KEY_SPEAKER_SAMPLE_RATES].tolist() == [16000, 24000]
    assert info[KEY_SPEAKER_WAVEFORM].tolist() == [[1.0] * 3 + [2.0] * 5]
    assert info["prompt_waveform"].shape == (1, 3528)


def test_podcast_extractor_splits_clips_before_resampling():
    from vllm_omni.model_executor.models.ming3_tts.speaker_extractor import resolve_speaker_embeddings

    calls = []

    def extract(waveform, sample_rate):
        calls.append((waveform.tolist(), sample_rate))
        return torch.full((192,), float(len(calls)))

    wrapper = SimpleNamespace(
        _speaker_extractor=SimpleNamespace(extract_from_waveform=extract),
        ming_config=SimpleNamespace(sample_rate=44100),
    )
    info = build_ming3_prompt(
        FakeTokenizer(),
        target_text="speaker_1:你好。\nspeaker_2:再见。",
        reference_waveform=[torch.ones(1, 3), torch.full((1, 5), 2.0)],
        reference_sample_rate=[16000, 24000],
        prompt_text=" speaker_1:参考一。\n speaker_2:参考二。\n",
    )["additional_information"]
    speakers = resolve_speaker_embeddings(wrapper, info)
    assert calls == [([[1.0] * 3], 16000), ([[2.0] * 5], 24000)]
    assert len(speakers) == 2
    assert torch.equal(speakers[0], torch.ones(192))
    assert torch.equal(speakers[1], torch.full((192,), 2.0))


@pytest.mark.parametrize(
    "text,max_tokens,error",
    [
        ("speaker_1:你好。\nspeaker_2:再见。", None, None),
        ("speaker_1:你好。\nspeaker_2:再见。", 8, None),
        ("speaker_1:你好。\nspeaker_2:再见。", 7, "text-derived minimum 8"),
        ("speaker_1:你好。\nspeaker_3:再见。", None, "each supplied"),
        ("普通文本", None, "one 'speaker_N: text'"),
    ],
)
def test_podcast_adapter_validates_turns(text, max_tokens, error):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter

    server = SimpleNamespace(
        engine_client=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    tts_config={
                        "conditioning_mode": "speaker_only",
                        "task_decode_alignment": {"podcast": "old_shift"},
                        "minimum_decode_steps": 13,
                    }
                )
            )
        ),
        _apply_uploaded_speaker=lambda request: None,
        _validate_ref_audio_format=lambda source: None,
    )
    adapter = Ming3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(
        input=text,
        ref_audio=["data:audio/wav;base64,AA=="] * 2,
        ref_text=" speaker_1:参考一。\n speaker_2:参考二。\n",
        max_new_tokens=max_tokens,
    )
    result = adapter.validate(request)
    assert result is None if error is None else error in result


@pytest.mark.parametrize("speaker_count", [1, 2])
@pytest.mark.parametrize(
    "language,expected",
    [
        ("ja", "Japanese"),
        (" Japanese ", "Japanese"),
        ("de", "German"),
        ("fr", "French"),
        ("ko", "Korean"),
        ("es", "Spanish"),
        ("ZH_CN", "Chinese"),
        ("en", "English"),
        ("unknown", None),
        (" ", None),
    ],
)
def test_adapter_language_and_reference_continuation(speaker_count, language, expected):
    import asyncio

    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter

    async def resolve(source):
        return [0.25] * 3528, 44100, source

    tokenizer = FakeTokenizer()
    server = SimpleNamespace(
        engine_client=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    tts_config={
                        "conditioning_mode": "speaker_only",
                        "task_decode_alignment": {"podcast": "old_shift"},
                    }
                )
            )
        ),
        _tts_tokenizer=tokenizer,
        _resolve_ref_audio=resolve,
        _get_resolved_ref_audio_artifact_key=lambda key: key,
        _apply_uploaded_speaker=lambda request: None,
        _validate_ref_audio_format=lambda source: None,
    )
    adapter = Ming3TTSAdapter(SpeechServingContext(server=server))
    text = "こんにちは。" if speaker_count == 1 else "speaker_1:こんにちは。\nspeaker_2:はい。"
    transcript = "参考。" if speaker_count == 1 else "speaker_1:参考一。\nspeaker_2:参考二。"
    request = OpenAICreateSpeechRequest(
        input=text,
        ref_audio="reference" if speaker_count == 1 else ["reference", "other"],
        ref_text=transcript,
        language=language,
    )
    if expected is None:
        assert "target language" in adapter.validate(request)
        return
    assert adapter.validate(request) is None
    prepared = asyncio.run(adapter.build(request, [], True))
    expected_prefix = "<role>HUMAN</role>Please generate speech based on the following description.\n"
    expected_prefix += f"Target language: {expected}.\n  speaker_1:"
    ids = prepared.prompt["prompt_token_ids"]
    prefix_ids = tokenizer.encode(expected_prefix)
    assert ids[: len(prefix_ids)] == prefix_ids
    reference_start = len(ids) - speaker_count - 1
    assert ids[reference_start:] == [AUDIO_START_TOKEN_ID] + [AUDIO_PATCH_TOKEN_ID] * speaker_count
    assert (
        tokenizer.encode(transcript + text + ASSISTANT_PREFIX)
        == ids[reference_start - len(tokenizer.encode(transcript + text + ASSISTANT_PREFIX)) : reference_start]
    )
    assert prepared.tts_params["prompt_waveform"].shape[-1] == 3528 * speaker_count
    assert "prompt_latents" not in prepared.tts_params


@pytest.mark.parametrize("speaker_count", [1, 2])
@pytest.mark.parametrize("enabled,infer", [(False, False), (True, False), (True, True)])
def test_adapter_missing_language_preserves_defaults(speaker_count, enabled, infer):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter

    server = SimpleNamespace(
        engine_client=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    tts_config={"enable_tts_target_language": enabled, "infer_tts_language_from_text": infer}
                )
            )
        )
    )
    adapter = Ming3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(input="你好。", ref_audio=["reference"] * speaker_count)
    if speaker_count > 1 and enabled and not infer:
        with pytest.raises(ValueError, match="explicit language"):
            adapter._target_language(request)
    else:
        assert adapter._target_language(request) == ("Chinese" if speaker_count > 1 and enabled else None)


@pytest.mark.parametrize("instruction", ["温柔、舒缓", "Speaking instruction: Speak in this style: 温柔、舒缓"])
def test_freeform_instruction_precedes_audio(instruction):
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter

    tokenizer = FakeTokenizer()
    tokenizer.special["<|endoftext|>"] = 156892
    caption = Ming3TTSAdapter._instruction(instruction)
    assert caption == "Speaking instruction: Speak in this style: 温柔、舒缓"
    ids, _, _ = build_ming3_prompt_token_ids(tokenizer, target_text="你好。", instruction=caption)
    expected = tokenizer.encode(ASSISTANT_PREFIX + caption) + [156892, AUDIO_START_TOKEN_ID]
    assert ids[-len(expected) :] == expected


def test_no_speaker_slot_prompt_and_prefill(monkeypatch):
    import vllm_omni.model_executor.models.ming3_tts.ming3_tts as model_module

    tokenizer = FakeTokenizer()
    prompt = build_ming3_prompt(
        tokenizer,
        target_text="清晨的风穿过树林。",
        reference_waveform=None,
        reference_sample_rate=None,
        speaker_slot=False,
    )
    ids = torch.tensor(prompt["prompt_token_ids"])
    expected = tokenizer.encode(
        "<role>HUMAN</role>Please generate speech based on the following description.\n"
        " Text input:\n清晨的风穿过树林。<role>ASSISTANT</role><audio>"
    )
    assert ids.tolist() == expected
    info = prompt["additional_information"]
    assert set(info) == {KEY_MAX_DECODE_STEPS, KEY_SPEAKER_POSITIONS, "no_speaker_slot"}

    def forbidden(*args, **kwargs):
        raise AssertionError("No speaker or prompt audio extraction expected")

    monkeypatch.setattr(model_module, "resolve_speaker_embeddings", forbidden)
    monkeypatch.setattr(model_module, "_resolve_prompt_latents", forbidden)
    wrapper = _prefill_wrapper(
        torch.nn.Identity(),
        SimpleNamespace(history_patch_size=1, latent_dim=64, speaker_placeholder_token_id=SPEAKER_START_TOKEN_ID),
    )
    embeds = torch.ones((len(ids), 4))
    _, result, _ = Ming3TTSForConditionalGeneration._prefill_preprocess(wrapper, ids, embeds, info)
    assert torch.all(result == 1)


def test_no_speaker_slot_keeps_text_prefix_after_user_prompt():
    tokenizer = FakeTokenizer()
    text = "Genre: jazz Mood: calm Instrument: piano Theme: dawn Duration: short"
    ids, _, _ = build_ming3_prompt_token_ids(tokenizer, target_text=text, speaker_count=0)
    expected = tokenizer.encode(
        "<role>HUMAN</role>Please generate speech based on the following description.\n"
        f" Text input:\n{text}<role>ASSISTANT</role><audio>"
    )
    assert ids == expected


@pytest.mark.parametrize("instruction", [" ", '{"语速":', "[]", "{}", '{"语速": 3}', '{"未知": "中"}'])
def test_invalid_structured_instructions_are_not_freeform(instruction):
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter

    with pytest.raises(ValueError):
        Ming3TTSAdapter._instruction(instruction)


def test_full_structured_instruction_preserves_archived_token_order():
    import json

    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter
    from vllm_omni.model_executor.models.ming_tts.prompt_assembly import create_instruction

    caption = {
        "audio_sequence": [
            {
                "序号": 1,
                "说话人": "speaker_1",
                "方言": "川渝话",
                "情感": None,
                "风格": None,
                "语速": None,
                "基频": None,
                "音量": None,
                "BGM": dict.fromkeys(["Genre", "Mood", "Instrument", "Theme", "ENV", "SNR"]),
                "IP": None,
            }
        ]
    }
    raw = json.dumps(caption, ensure_ascii=False)
    normalized = create_instruction({"方言": "川渝话"})
    assert json.loads(normalized) == caption and normalized != raw
    instruction = Ming3TTSAdapter._instruction(raw)
    assert instruction == raw
    tokenizer = FakeTokenizer()
    actual, _, _ = build_ming3_prompt_token_ids(tokenizer, target_text="你好。", instruction=instruction)
    caption_tokens = tokenizer.encode(raw)
    assert actual[-len(caption_tokens) - 2 :] == caption_tokens + [tokenizer.eos_token_id, AUDIO_START_TOKEN_ID]
    invalid_cases: tuple[dict[str, Any], ...] = ({"audio_sequence": []}, {"audio_sequence": [{"方言": "川渝话"}]})
    for invalid in invalid_cases:
        with pytest.raises(ValueError):
            Ming3TTSAdapter._instruction(json.dumps(invalid, ensure_ascii=False))


@pytest.mark.parametrize("controls", [{"情感": "高兴"}, {"风格": "沉稳的男性旁白"}, {"IP": "阿狸"}])
def test_no_reference_controls_mask_only_speaker_slot(monkeypatch, controls):
    import asyncio
    import json

    from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter
    from vllm_omni.model_executor.models.ming3_tts import ming3_tts as model_module

    def forbidden(*args, **kwargs):
        pytest.fail("Masked conditioning must not resolve audio, extract or project speakers, or encode reference VAE")

    tokenizer = FakeTokenizer()
    tokenizer.special["<|endoftext|>"] = 156892
    server = SimpleNamespace(
        engine_client=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(tts_config={}))),
        _tts_tokenizer=tokenizer,
        _resolve_ref_audio=forbidden,
        _apply_uploaded_speaker=lambda request: None,
        _validate_ref_audio_format=forbidden,
    )
    adapter = Ming3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(input="你好。", instructions=json.dumps(controls, ensure_ascii=False))
    assert adapter.validate(request) is None
    prepared = asyncio.run(adapter.build(request, [], False))
    info = deserialize_additional_information(serialize_additional_information(prepared.tts_params))
    assert set(info) == {KEY_MAX_DECODE_STEPS, KEY_SPEAKER_POSITIONS, "mute_speaker"}
    ids = torch.tensor(prepared.prompt["prompt_token_ids"])
    assert (ids == SPEAKER_START_TOKEN_ID).sum().item() == 1
    slot = (ids == SPEAKER_START_TOKEN_ID).nonzero().item() + 1
    assert ids[-1].item() == AUDIO_START_TOKEN_ID
    embeds = torch.ones((len(ids), 4))
    monkeypatch.setattr(model_module, "resolve_speaker_embeddings", forbidden)
    monkeypatch.setattr(model_module, "_resolve_prompt_latents", forbidden)
    wrapper = _prefill_wrapper(
        SimpleNamespace(project_speaker_embedding=forbidden),
        SimpleNamespace(history_patch_size=1, latent_dim=64, speaker_placeholder_token_id=SPEAKER_START_TOKEN_ID),
    )
    _, result, _ = Ming3TTSForConditionalGeneration._prefill_preprocess(wrapper, ids, embeds, info)
    assert torch.count_nonzero(result[slot]).item() == 0
    assert torch.all(result[:slot] == 1) and torch.all(result[slot + 1 :] == 1)


@pytest.mark.parametrize(
    "fields,error",
    [
        ({"ref_text": "参考"}, "requires reference audio"),
        ({"instructions": ""}, "reference audio or"),
        ({"task_type": "Base"}, "reference audio or"),
    ],
)
def test_no_reference_request_rejects_invalid_conditioning(fields, error):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter

    adapter = Ming3TTSAdapter(
        SpeechServingContext(server=SimpleNamespace(_apply_uploaded_speaker=lambda request: None))
    )
    assert error in adapter.validate(OpenAICreateSpeechRequest(input="你好。", **fields))


def test_online_emergent_omits_speaker_conditioning():
    import asyncio

    from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter

    def forbidden(*args, **kwargs):
        pytest.fail("Emergent must not resolve or validate reference audio")

    server = SimpleNamespace(
        engine_client=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(tts_config={}))),
        _tts_tokenizer=FakeTokenizer(),
        _apply_uploaded_speaker=lambda request: None,
        _resolve_ref_audio=forbidden,
        _validate_ref_audio_format=forbidden,
    )
    adapter = Ming3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(input="清晨的风穿过树林。")
    assert adapter.validate(request) is None
    prepared = asyncio.run(adapter.build(request, [], False))
    ids = prepared.prompt["prompt_token_ids"]
    assert SPEAKER_START_TOKEN_ID not in ids
    assert ids[-1] == AUDIO_START_TOKEN_ID
    info = deserialize_additional_information(serialize_additional_information(prepared.tts_params))
    assert set(info) == {KEY_MAX_DECODE_STEPS, KEY_SPEAKER_POSITIONS, "no_speaker_slot"}


@pytest.mark.parametrize("wrapped", [False, True])
def test_named_speaker_cache_reuses_only_matching_voice_version(wrapped):
    from vllm_omni.model_executor.models.ming3_tts.constants import KEY_SPEAKER_EMBEDDING, KEY_SPEAKER_WAVEFORM
    from vllm_omni.model_executor.models.ming3_tts.speaker_extractor import resolve_speaker_embeddings
    from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

    calls = []

    def extract(waveform, sample_rate):
        calls.append((waveform, sample_rate))
        return torch.full((192,), float(len(calls)))

    cache = SpeakerEmbeddingCache()
    wrapper = SimpleNamespace(
        _speaker_cache=cache,
        _speaker_extractor=SimpleNamespace(extract_from_waveform=extract),
        ming_config=SimpleNamespace(sample_rate=24000),
    )
    info = {KEY_SPEAKER_WAVEFORM: torch.zeros(1, 2400)}
    for name, version, expected_calls in [("alice", 1, 1), ("alice", 1, 1), ("bob", 1, 2), ("alice", 2, 3)]:
        info.update(
            voice_name=[name] if wrapped else name, voice_created_at=torch.tensor([version]) if wrapped else version
        )
        embedding = resolve_speaker_embeddings(wrapper, info)[0]
        assert len(calls) == expected_calls
        assert torch.equal(embedding, torch.full((192,), float(expected_calls)))
    assert cache.get(cache.make_cache_key("alice", "ming_tts", 1)) is None
    direct = torch.zeros(192)
    assert torch.equal(resolve_speaker_embeddings(wrapper, {**info, KEY_SPEAKER_EMBEDDING: direct})[0], direct)
    assert len(calls) == 3
    for _ in range(2):
        resolve_speaker_embeddings(wrapper, {KEY_SPEAKER_WAVEFORM: info[KEY_SPEAKER_WAVEFORM]})
    assert len(calls) == 5


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("sft", [False, True])
@pytest.mark.parametrize("max_tokens", [None, 100])
def test_named_speaker_cache_metadata_excludes_inline_audio(inline, sft, max_tokens):
    import asyncio

    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter

    async def resolve_audio(source):
        return [0.0] * 2400, 24000, "audio-key"

    server = SimpleNamespace(
        engine_client=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(tts_config={"maximum_decode_steps": 250} if sft else {})
            )
        ),
        _tts_tokenizer=FakeTokenizer(),
        _resolve_ref_audio=resolve_audio,
        _get_resolved_ref_audio_artifact_key=lambda key: "content-key",
        uploaded_speakers={"alice": {"created_at": 123}},
        _voice_created_at=lambda name: 123,
    )
    adapter = Ming3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(
        input="Hello", voice="Alice", ref_audio="data:audio/wav;base64,AA==", max_new_tokens=max_tokens
    )
    prepared = asyncio.run(adapter.build(request, [], inline))
    info = prepared.prompt["additional_information"]
    maximum = max_tokens or (250 if sft else 800)
    assert info[KEY_MAX_DECODE_STEPS] == maximum
    sampling = adapter.apply_sampling_overrides([SamplingParams()], request)
    assert sampling[0].max_tokens == maximum + 1
    if inline:
        assert "voice_name" not in info and "voice_created_at" not in info
    else:
        assert info["voice_name"] == "alice" and info["voice_created_at"] == 123


def test_podcast_speaker_cache_reuses_individual_references():
    from vllm_omni.model_executor.models.ming3_tts.constants import (
        KEY_SPEAKER_SAMPLE_RATES,
        KEY_SPEAKER_WAVEFORM,
        KEY_SPEAKER_WAVEFORM_LENGTHS,
    )
    from vllm_omni.model_executor.models.ming3_tts.speaker_extractor import resolve_speaker_embeddings
    from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

    calls = []

    def extract(waveform, sample_rate):
        calls.append((waveform.clone(), sample_rate))
        return torch.full((192,), waveform[0, 0].item())

    wrapper = SimpleNamespace(
        _speaker_cache=SpeakerEmbeddingCache(),
        _speaker_extractor=SimpleNamespace(extract_from_waveform=extract),
        ming_config=SimpleNamespace(sample_rate=44100),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(model="checkpoint-a")),
    )

    def resolve(keys, values):
        return resolve_speaker_embeddings(
            wrapper,
            {
                "speaker_audio_keys": keys,
                KEY_SPEAKER_WAVEFORM: torch.tensor([values], dtype=torch.float32),
                KEY_SPEAKER_WAVEFORM_LENGTHS: torch.tensor([1, 1]),
                KEY_SPEAKER_SAMPLE_RATES: torch.tensor([16000, 16000]),
            },
        )

    first = resolve(["audio-a", "audio-b"], [1, 2])
    cached = resolve(["audio-a", "audio-b"], [1, 2])
    assert len(calls) == 2
    assert all(torch.equal(a, b) for a, b in zip(first, cached, strict=True))
    reversed_speakers = resolve(["audio-b", "audio-a"], [2, 1])
    assert len(calls) == 2
    assert torch.equal(reversed_speakers[0], first[1])
    assert torch.equal(reversed_speakers[1], first[0])
    changed = resolve(["audio-a", "audio-c"], [1, 3])
    assert len(calls) == 3 and torch.equal(changed[1], torch.full((192,), 3.0))
    wrapper.vllm_config.model_config.model = "checkpoint-b"
    resolve(["audio-a", "audio-b"], [1, 2])
    assert len(calls) == 5
    resolve([None, None], [1, 2])
    resolve([None, None], [1, 2])
    assert len(calls) == 9


@pytest.mark.parametrize("references", ["1", ["1"], ["1", "2"]])
@pytest.mark.parametrize("turn_text", ["你好。", "字" * 100])
def test_adapter_transports_content_keys_without_caching_prompt_latents(references, turn_text):
    import asyncio

    from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter

    async def resolve(source):
        return [float(source)] * 8, 44100, "resolved-" + source

    server = SimpleNamespace(
        engine_client=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(tts_config={"maximum_decode_steps": 250}))
        ),
        _tts_tokenizer=FakeTokenizer(),
        _resolve_ref_audio=resolve,
        _get_resolved_ref_audio_artifact_key=lambda key: {"resolved-1": "content-a", "resolved-2": "content-b"}[key],
    )
    adapter = Ming3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(
        input=f"speaker_1:{turn_text}\nspeaker_2:{turn_text}",
        ref_audio=references,
        ref_text="speaker_1:参考一。\nspeaker_2:参考二。",
    )
    for _ in range(2):
        prepared = asyncio.run(adapter.build(request, [], True))
        info = deserialize_additional_information(serialize_additional_information(prepared.tts_params))
        assert info["speaker_audio_keys"] == (["content-a", "content-b"] if references == ["1", "2"] else ["content-a"])
        assert "voice_name" not in info
        assert "prompt_latents" not in info
        assert isinstance(info["prompt_waveform"], torch.Tensor)
        expected = [1.0] * 8 + ([2.0] * 8 if references == ["1", "2"] else [])
        assert info["prompt_waveform"][0, : len(expected)].tolist() == expected
        if references == ["1", "2"]:
            minimum, maximum = (8, 24) if len(turn_text) == 3 else (200, 600)
            assert int(info[KEY_MIN_DECODE_STEPS]) == minimum
        else:
            maximum = 250
            assert KEY_MIN_DECODE_STEPS not in info
        assert int(info[KEY_MAX_DECODE_STEPS]) == maximum
        defaults = [SamplingParams(temperature=0.0, max_tokens=251)]
        sampling = adapter.apply_sampling_overrides(defaults, request, prepared.prompt)
        assert sampling[0].max_tokens == maximum + 1


@pytest.mark.parametrize("cfg", [None, 1.0, 2.0])
@pytest.mark.parametrize("cfg_key", ["ming3_cfg", "lingv3_cfg"])
def test_request_cfg_and_temperature_reach_flow_runtime(cfg, cfg_key):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter
    from vllm_omni.model_executor.models.ming3_tts.constants import KEY_CFG, KEY_TEMPERATURE

    server = SimpleNamespace(
        engine_client=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(tts_config={}))),
        _apply_uploaded_speaker=lambda request: None,
    )
    adapter = Ming3TTSAdapter(SpeechServingContext(server=server))
    extra = {"temperature": 1.5}
    if cfg is not None:
        extra[cfg_key] = cfg
    request = OpenAICreateSpeechRequest(input="Bonjour.", instructions="calm", seed=1234, extra_params=extra)
    assert adapter.validate(request) is None
    prompt: dict[str, Any] = {"additional_information": {}}
    # serving_speech applies extra_params.temperature before adapter overrides.
    defaults = [SamplingParams(temperature=1.5, max_tokens=251)]
    adapter.apply_sampling_overrides(defaults, request, prompt)
    info = prompt["additional_information"]
    assert info[KEY_TEMPERATURE] == 1.5 and info["tts_local_seed"] == 1234
    update: dict[str, Any] = {}
    Ming3TTSForConditionalGeneration._copy_request_state(SimpleNamespace(), update, info)
    assert update["ming_temperature"] == 1.5
    if cfg is None:
        assert KEY_CFG not in info and "ming_cfg" not in update
    else:
        assert info[KEY_CFG] == update["ming_cfg"] == cfg


@pytest.mark.parametrize("key", ["ming3_cfg", "lingv3_cfg", "temperature"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), "2", True])
def test_request_sampling_controls_reject_invalid_runtime_values(key, value):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter

    server = SimpleNamespace(
        engine_client=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(tts_config={}))),
        _apply_uploaded_speaker=lambda request: None,
    )
    adapter = Ming3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(input="Bonjour.", instructions="calm", extra_params={key: value})
    assert "finite non-negative" in adapter.validate(request)


@pytest.mark.parametrize("temperature", [None, 0.0, 1.5])
def test_flow_temperature_preserves_checkpoint_default_without_request_override(temperature: float | None):
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter
    from vllm_omni.model_executor.models.ming3_tts.constants import KEY_TEMPERATURE
    from vllm_omni.model_executor.models.ming_tts.patch_emission import _resolve_runtime_float

    server = SimpleNamespace(
        engine_client=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(tts_config={"sampling_temperature": 1.5}))
        ),
    )
    adapter = Ming3TTSAdapter(SpeechServingContext(server=server))
    extra = {} if temperature is None else {"temperature": temperature}
    request = OpenAICreateSpeechRequest(input="你好。", extra_params=extra)
    defaults = [SamplingParams(temperature=0.0, max_tokens=801)]
    prompt: dict[str, Any] = {"additional_information": {}}
    adapter.apply_sampling_overrides(defaults, request, prompt)
    info = prompt["additional_information"]
    if temperature is None:
        assert KEY_TEMPERATURE not in info
    else:
        assert info[KEY_TEMPERATURE] == temperature
    update: dict[str, Any] = {}
    Ming3TTSForConditionalGeneration._copy_request_state(SimpleNamespace(), update, info)
    effective = _resolve_runtime_float(update, "ming_temperature", adapter.tts_config["sampling_temperature"])
    assert effective == (1.5 if temperature is None else temperature)
    assert defaults[0].temperature == 0.0
