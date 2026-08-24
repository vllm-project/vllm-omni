# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.model_executor.models.common.ming.aggregator import Aggregator
from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAEConfig
from vllm_omni.model_executor.models.common.ming.cfm_cudagraph import CFMGraphExecutor, CFMSampler
from vllm_omni.model_executor.models.common.ming.cfm_dit import DiT
from vllm_omni.model_executor.models.common.ming.cfm_head import FlowLoss
from vllm_omni.model_executor.models.common.ming.cfm_solver import Solver, get_epss_timesteps
from vllm_omni.model_executor.models.common.ming.speaker_extractor import SpeakerEmbeddingExtractor
from vllm_omni.model_executor.models.common.ming.spk_embedding import SpkembExtractor
from vllm_omni.model_executor.models.ming_tts.constants import (
    AGGREGATOR_HIDDEN_SIZE,
    HISTORY_PATCH_SIZE,
    KEY_SPEAKER_EMBEDDING,
    KEY_SPEAKER_SAMPLE_RATES,
    KEY_SPEAKER_WAVEFORM,
    KEY_SPEAKER_WAVEFORM_LENGTHS,
    LATENT_DIM,
    LLM_HIDDEN_SIZE,
    LLM_VOCAB_SIZE,
    PATCH_SIZE,
    SAMPLE_RATE,
    SPEAKER_EMBEDDING_DIM,
    VAE_PATCH_SIZE,
)
from vllm_omni.model_executor.models.ming_tts.ming_tts_llm import MingLLMModel
from vllm_omni.model_executor.models.ming_tts.speaker_extractor import _resolve_speaker_embeddings
from vllm_omni.model_executor.models.ming_tts.validation import validate_ming_tts_config
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def test_ming_tts_audio_vae_uses_common_config():
    """AudioVAEConfig is shared by Ming dense and Ming flash modules."""
    cfg = AudioVAEConfig(sample_rate=16000, patch_size=-1)

    assert cfg.sample_rate == 16000
    assert cfg.patch_size == -1


def test_ming_tts_cfm_solver_uses_common_implementation():
    """Ming dense imports the shared solver implementation directly."""
    assert Solver.__module__ == "vllm_omni.model_executor.models.common.ming.cfm_solver"


@pytest.mark.parametrize(
    ("symbol", "expected_module"),
    [
        (Aggregator, "vllm_omni.model_executor.models.common.ming.aggregator"),
        (DiT, "vllm_omni.model_executor.models.common.ming.cfm_dit"),
        (FlowLoss, "vllm_omni.model_executor.models.common.ming.cfm_head"),
        (CFMGraphExecutor, "vllm_omni.model_executor.models.common.ming.cfm_cudagraph"),
        (CFMSampler, "vllm_omni.model_executor.models.common.ming.cfm_cudagraph"),
        (SpeakerEmbeddingExtractor, "vllm_omni.model_executor.models.common.ming.speaker_extractor"),
        (SpkembExtractor, "vllm_omni.model_executor.models.common.ming.spk_embedding"),
    ],
)
def test_ming_tts_reusable_modules_live_in_common(symbol, expected_module):
    """The pieces a second Ming-family TTS model reuses live under common/ming.

    Guards the promotion: if an implementation drifts back into `ming_tts`
    (or into another model package), a second model would have to reach across
    packages to get it.
    """
    assert symbol.__module__ == expected_module


def test_ming_flash_talker_uses_the_shared_aggregator():
    """Ming flash no longer carries its own copy of the aggregator.

    Both Ming packages ran near-identical aggregators one directory apart,
    which is the duplication this module layout exists to remove.
    """
    from vllm_omni.model_executor.models.ming_flash_omni import talker_module

    assert talker_module.Aggregator is Aggregator
    assert not any(
        isinstance(obj, type) and obj.__module__ == talker_module.__name__ and name == "Aggregator"
        for name, obj in vars(talker_module).items()
    )


def test_shared_aggregator_matches_the_ming_flash_construction():
    """The shared aggregator reproduces Ming flash's former default geometry.

    `talker_module.Aggregator` defaulted to `in_channels=64` while the
    shared class defaults to 4, so the call site now passes `latent_dim`
    explicitly. This pins the resulting shapes rather than the default.
    """
    latent_dim, llm_hidden = 64, 32
    aggregator = Aggregator(
        in_channels=latent_dim,
        llm_input_dim=llm_hidden,
        hidden_size=16,
        depth=1,
        num_heads=2,
    )

    assert aggregator.x_embedder.in_features == latent_dim
    assert aggregator.final_layer.linear.out_features == llm_hidden

    out = aggregator(torch.randn(1, 3, latent_dim))
    # Keeps only the CLS projection: [B, T, D] -> [B, 1, llm_hidden].
    assert out.shape == (1, 1, llm_hidden)


def test_shared_aggregator_rejects_mismatched_latent_width():
    """Wrong-width input fails loudly instead of silently projecting incorrect."""
    aggregator = Aggregator(in_channels=64, llm_input_dim=32, hidden_size=16, depth=1, num_heads=2)

    with pytest.raises(ValueError, match="feature dim mismatch"):
        aggregator(torch.randn(1, 3, 4))


def test_cfm_solver_owns_the_timestep_schedule():
    """get_epss_timesteps sits with the solver, not with the DiT blocks."""
    from vllm_omni.model_executor.models.common.ming import dit_blocks

    assert not hasattr(dit_blocks, "get_epss_timesteps")

    t = get_epss_timesteps(10, device="cpu", dtype=torch.float32)
    assert t.shape == (11,)
    assert float(t[0]) == 0.0 and float(t[-1]) == 1.0


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


class _FakeMingSpeakerExtractor:
    def __init__(self):
        self.calls: list[tuple[torch.Tensor, int]] = []

    def extract_from_waveform(self, waveform, sample_rate):
        waveform = torch.as_tensor(waveform).detach().clone()
        self.calls.append((waveform, int(sample_rate)))
        return torch.full((SPEAKER_EMBEDDING_DIM,), float(len(self.calls)), dtype=torch.float32)


def _make_ming_speaker_wrapper():
    return SimpleNamespace(
        _speaker_cache=SpeakerEmbeddingCache(max_bytes=1024 * 1024),
        _speaker_extractor=_FakeMingSpeakerExtractor(),
        ming_config=SimpleNamespace(sample_rate=SAMPLE_RATE),
    )


def test_ming_stage0_speaker_cache_extracts_once_and_honors_created_at():
    wrapper = _make_ming_speaker_wrapper()
    waveform = torch.arange(12, dtype=torch.float32).reshape(1, -1)
    info = {
        KEY_SPEAKER_WAVEFORM: waveform,
        KEY_SPEAKER_WAVEFORM_LENGTHS: torch.tensor([12]),
        "voice_name": "UploadedVoice",
        "voice_created_at": 123,
    }

    first = _resolve_speaker_embeddings(wrapper, info)
    cached = _resolve_speaker_embeddings(
        wrapper,
        {
            "voice_name": "uploadedvoice",
            "voice_created_at": 123,
        },
    )
    refreshed = _resolve_speaker_embeddings(
        wrapper,
        {
            **info,
            "voice_created_at": 124,
        },
    )

    assert first is not None and cached is not None and refreshed is not None
    torch.testing.assert_close(first[0], cached[0])
    assert torch.all(first[0] == 1)
    assert torch.all(refreshed[0] == 2)
    assert len(wrapper._speaker_extractor.calls) == 2


def test_ming_stage0_speaker_extraction_splits_multiple_waveforms():
    wrapper = _make_ming_speaker_wrapper()

    embeddings = _resolve_speaker_embeddings(
        wrapper,
        {
            KEY_SPEAKER_WAVEFORM: torch.arange(10, dtype=torch.float32).reshape(1, -1),
            KEY_SPEAKER_WAVEFORM_LENGTHS: torch.tensor([4, 6]),
            KEY_SPEAKER_SAMPLE_RATES: torch.tensor([16000, 24000]),
        },
    )

    assert embeddings is not None and len(embeddings) == 2
    assert [call[0].shape[-1] for call in wrapper._speaker_extractor.calls] == [4, 6]
    assert [call[1] for call in wrapper._speaker_extractor.calls] == [16000, 24000]


def test_ming_stage0_speaker_extraction_rejects_mismatched_sample_rates():
    wrapper = _make_ming_speaker_wrapper()

    with pytest.raises(ValueError, match="Invalid Ming speaker sample rates"):
        _resolve_speaker_embeddings(
            wrapper,
            {
                KEY_SPEAKER_WAVEFORM: torch.arange(10, dtype=torch.float32).reshape(1, -1),
                KEY_SPEAKER_WAVEFORM_LENGTHS: torch.tensor([4, 6]),
                KEY_SPEAKER_SAMPLE_RATES: torch.tensor([16000]),
            },
        )


def test_ming_stage0_prompt_waveform_does_not_trigger_speaker_extraction():
    wrapper = _make_ming_speaker_wrapper()

    embeddings = _resolve_speaker_embeddings(
        wrapper,
        {
            "prompt_waveform": torch.ones((1, 10)),
            "prompt_text": "Reference transcript.",
        },
    )

    assert embeddings is None
    assert not wrapper._speaker_extractor.calls


def test_ming_stage0_direct_speaker_embedding_bypasses_extractor():
    wrapper = _make_ming_speaker_wrapper()
    direct = torch.full((SPEAKER_EMBEDDING_DIM,), 0.25)

    embeddings = _resolve_speaker_embeddings(
        wrapper,
        {
            KEY_SPEAKER_EMBEDDING: direct,
            KEY_SPEAKER_WAVEFORM: torch.ones((1, 10)),
            "voice_name": "direct",
            "voice_created_at": 1,
        },
    )

    assert embeddings is not None
    torch.testing.assert_close(embeddings[0], direct)
    assert not wrapper._speaker_extractor.calls


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
