# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAEConfig
from vllm_omni.model_executor.models.common.ming.fm import Solver
from vllm_omni.model_executor.models.ming_tts.constants import (
    AGGREGATOR_HIDDEN_SIZE,
    HISTORY_PATCH_SIZE,
    KEY_DECODE_STEP,
    KEY_LATENT_HISTORY,
    KEY_NEXT_EMBEDS,
    KEY_REQUEST_ID,
    LATENT_DIM,
    LLM_HIDDEN_SIZE,
    LLM_VOCAB_SIZE,
    PATCH_SIZE,
    SAMPLE_RATE,
    VAE_PATCH_SIZE,
)
from vllm_omni.model_executor.models.ming_tts.flowloss_head import FlowLoss
from vllm_omni.model_executor.models.ming_tts.ming_tts import MingTTSForConditionalGeneration
from vllm_omni.model_executor.models.ming_tts.ming_tts_llm import MingLLMModel
from vllm_omni.model_executor.models.ming_tts.patch_emission import (
    MING_STOP_REASON_CONTINUE,
    MING_STOP_REASON_KEY,
    MING_STOP_REASON_MAX_DECODE_STEPS,
)
from vllm_omni.model_executor.models.ming_tts.validation import validate_ming_tts_config
from vllm_omni.model_executor.models.output_templates import OmniOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def test_ming_tts_audio_vae_uses_common_config():
    """AudioVAEConfig is shared by Ming dense and Ming flash modules."""
    cfg = AudioVAEConfig(sample_rate=16000, patch_size=-1)

    assert cfg.sample_rate == 16000
    assert cfg.patch_size == -1


def test_ming_tts_cfm_solver_uses_common_implementation():
    """Ming dense imports the shared solver implementation directly."""
    assert Solver.__module__ == "vllm_omni.model_executor.models.common.ming.fm"


def test_ming_tts_flowloss_preserves_checkpoint_prefix():
    flowloss = FlowLoss(z_channels=4, llm_cond_dim=8, hidden_size=16, depth=1, num_heads=2)

    assert any(name.startswith("cfm.model.") for name in flowloss.state_dict())


def test_ming_dense_validation_rejects_semantic_audio_vae_config():
    """Dense 0.5B validation rejects semantic AudioVAE configs."""
    cfg = SimpleNamespace(
        audio_dummy_token_id=151705,
        audio_eos_token_id=151704,
        text_eos_token_id=151669,
        audio_tokenizer_config=AudioVAEConfig(
            sample_rate=SAMPLE_RATE,
            patch_size=VAE_PATCH_SIZE,
            semantic_module_kwargs={"whisper_encoder": {}},
            enc_kwargs={"latent_dim": LATENT_DIM, "input_dim": 882, "hop_size": 882},
            dec_kwargs={"latent_dim": LATENT_DIM, "output_dim": 882},
        ),
        latent_dim=LATENT_DIM,
        patch_size=PATCH_SIZE,
        history_patch_size=HISTORY_PATCH_SIZE,
        llm_hidden_size=LLM_HIDDEN_SIZE,
        llm_vocab_size=LLM_VOCAB_SIZE,
        sample_rate=SAMPLE_RATE,
        vae_patch_size=VAE_PATCH_SIZE,
        llm_config={"hidden_size": LLM_HIDDEN_SIZE},
        aggregator_config={"hidden_size": AGGREGATOR_HIDDEN_SIZE},
        ditar_config={"hidden_size": AGGREGATOR_HIDDEN_SIZE},
        latent_chunk_size=1,
        latent_left_context=0,
        max_decode_steps=1,
        stop_head_threshold=0.5,
        stop_head_min_steps=0,
    )

    with pytest.raises(ValueError, match="semantic_module_kwargs"):
        validate_ming_tts_config(cfg)


def test_ming_instruction_parser_preserves_dense_and_flash_defaults():
    """Ming dense and Ming flash keep distinct instruction defaults."""
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving.uploaded_speakers = {"uploaded": {}}

    dense_plain = serving._parse_ming_instruction(SimpleNamespace(instructions="calm", language=None, voice=None))
    assert dense_plain == "calm"

    dense_with_fields = serving._parse_ming_instruction(
        SimpleNamespace(instructions="calm", language="Auto", voice="灵小甄")
    )
    assert dense_with_fields == {"IP": "灵小甄", "风格": "calm"}

    flash_fields = serving._parse_ming_instruction_fields(
        SimpleNamespace(instructions="calm", language="粤语", voice="灵小甄")
    )
    assert flash_fields == {"风格": "calm"}


def _make_ming_logits_model(vocab_size=8):
    model = object.__new__(MingLLMModel)
    model.ming_config = SimpleNamespace(
        llm_vocab_size=vocab_size,
        max_decode_steps=1,
        stop_head_min_steps=0,
        text_eos_token_id=7,
    )
    model._last_text_mode = False
    model._last_ming_next_token_ids = None
    return model


def test_ming_compute_logits_uses_cached_forced_next_token_ids():
    model = _make_ming_logits_model()
    model._last_ming_next_token_ids = [2, 5]

    logits = MingLLMModel.compute_logits(model, torch.zeros((2, 4)), SimpleNamespace())

    assert logits.shape == (2, 8)
    assert logits[0, 2].item() == 0.0
    assert logits[1, 5].item() == 0.0
    assert torch.isneginf(logits[0, [0, 1, 3, 4, 5, 6, 7]]).all()
    assert torch.isneginf(logits[1, [0, 1, 2, 3, 4, 6, 7]]).all()
    assert model._last_ming_next_token_ids is None


def test_ming_compute_logits_falls_back_to_dummy_token_id():
    model = _make_ming_logits_model()

    logits = MingLLMModel.compute_logits(model, torch.zeros((1, 4)), SimpleNamespace())
    assert logits.shape == (1, 8)
    assert logits[0, 7].item() == 0.0
    assert torch.isneginf(logits[0, [0, 1, 2, 3, 4, 5, 6]]).all()


def test_ming_forward_non_decode_return_clears_cached_forced_next_token_ids():
    class FakeBackbone:
        # MingLLMModel.forward calls the backbone positionally
        # (input_ids, positions, intermediate_tensors, inputs_embeds) so the
        # same signature works for both Qwen2Model (``positions``) and
        # BailingMoeModel (``position_ids``).
        def __call__(self, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs):
            return inputs_embeds

    model = _make_ming_logits_model()
    model.model = FakeBackbone()
    model._last_ming_next_token_ids = [2]

    output = MingLLMModel.forward(
        model,
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        inputs_embeds=torch.zeros((1, 4)),
        model_intermediate_buffer=[],
    )

    assert isinstance(output, OmniOutput)
    assert output.multimodal_outputs is None
    assert model._last_ming_next_token_ids is None


def test_ming_compute_logits_rejects_forced_token_batch_mismatch():
    model = _make_ming_logits_model()
    model._last_ming_next_token_ids = [2]

    with pytest.raises(RuntimeError, match="batch mismatch"):
        MingLLMModel.compute_logits(model, torch.zeros((2, 4)), SimpleNamespace())


def test_ming_compute_logits_text_mode_delegates_to_backbone():
    class FakeBackbone:
        def __init__(self):
            self.hidden_states = None

        def compute_logits(self, hidden_states):
            self.hidden_states = hidden_states
            return torch.ones((hidden_states.shape[0], 3))

    model = _make_ming_logits_model(vocab_size=3)
    model._last_text_mode = True
    model.model = FakeBackbone()
    hidden_states = torch.zeros((2, 4))

    logits = MingLLMModel.compute_logits(model, hidden_states, SimpleNamespace())

    assert torch.equal(model.model.hidden_states, hidden_states)
    assert torch.equal(logits, torch.ones((2, 3)))


def _make_ming_stage0_model():
    model = object.__new__(MingTTSForConditionalGeneration)
    model.model_stage = "llm"
    model.ming_config = SimpleNamespace(
        history_patch_size=HISTORY_PATCH_SIZE,
        latent_dim=LATENT_DIM,
        llm_hidden_size=LLM_HIDDEN_SIZE,
    )
    model._decode_state_cache = {}
    return model


def test_ming_decode_preprocess_prefers_local_gpu_cache():
    model = _make_ming_stage0_model()
    request_id = "req-1"
    cached_history = torch.full((HISTORY_PATCH_SIZE, LATENT_DIM), 7.0)
    cached_next = torch.full((1, LLM_HIDDEN_SIZE), 3.0)
    model._decode_state_cache[request_id] = {
        KEY_LATENT_HISTORY: cached_history,
        KEY_NEXT_EMBEDS: cached_next,
    }

    _, input_embeds, update = MingTTSForConditionalGeneration._decode_preprocess(
        model,
        input_ids=torch.tensor([1]),
        input_embeds=torch.zeros((1, LLM_HIDDEN_SIZE)),
        **{
            KEY_REQUEST_ID: request_id,
            KEY_DECODE_STEP: 4,
            KEY_LATENT_HISTORY: torch.zeros((HISTORY_PATCH_SIZE, LATENT_DIM)),
            KEY_NEXT_EMBEDS: torch.zeros((1, LLM_HIDDEN_SIZE)),
        },
    )

    assert torch.equal(update[KEY_LATENT_HISTORY], cached_history)
    assert torch.equal(input_embeds[0], cached_next[0])


def test_ming_postprocess_updates_and_clears_local_cache():
    model = _make_ming_stage0_model()
    request_id = "req-2"
    next_history = torch.randn(HISTORY_PATCH_SIZE, LATENT_DIM)
    next_embeds = torch.randn(1, LLM_HIDDEN_SIZE)
    pending = {
        KEY_LATENT_HISTORY: next_history,
        KEY_NEXT_EMBEDS: next_embeds,
        MING_STOP_REASON_KEY: MING_STOP_REASON_CONTINUE,
    }
    model.model = SimpleNamespace(pop_postprocess_update=lambda req_id: pending if req_id == request_id else {})

    update = MingTTSForConditionalGeneration.postprocess(
        model,
        hidden_states=torch.ones((1, LLM_HIDDEN_SIZE)),
        **{KEY_REQUEST_ID: request_id, KEY_DECODE_STEP: 1},
    )

    assert torch.equal(update[KEY_LATENT_HISTORY], next_history)
    assert torch.equal(update[KEY_NEXT_EMBEDS], next_embeds)
    assert request_id in model._decode_state_cache
    assert torch.equal(model._decode_state_cache[request_id][KEY_LATENT_HISTORY], next_history)

    terminal_pending = {
        KEY_LATENT_HISTORY: next_history,
        KEY_NEXT_EMBEDS: next_embeds,
        MING_STOP_REASON_KEY: MING_STOP_REASON_MAX_DECODE_STEPS,
    }
    model.model = SimpleNamespace(
        pop_postprocess_update=lambda req_id: terminal_pending if req_id == request_id else {}
    )
    MingTTSForConditionalGeneration.postprocess(
        model,
        hidden_states=torch.ones((1, LLM_HIDDEN_SIZE)),
        **{KEY_REQUEST_ID: request_id, KEY_DECODE_STEP: 2},
    )

    assert request_id not in model._decode_state_cache
