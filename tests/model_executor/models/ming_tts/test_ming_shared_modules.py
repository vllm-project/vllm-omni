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
    LATENT_DIM,
    LLM_HIDDEN_SIZE,
    LLM_VOCAB_SIZE,
    PATCH_SIZE,
    SAMPLE_RATE,
    VAE_PATCH_SIZE,
)
from vllm_omni.model_executor.models.ming_tts.flowloss_head import FlowLoss
from vllm_omni.model_executor.models.ming_tts.ming_tts_llm import MingLLMModel
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
        def __call__(self, **kwargs):
            return kwargs["inputs_embeds"]

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
