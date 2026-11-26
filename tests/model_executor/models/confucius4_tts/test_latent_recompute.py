# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from vllm_omni.model_executor.models.confucius4_tts import (
    confucius4_tts_latent_recompute as recompute_module,
)
from vllm_omni.model_executor.models.confucius4_tts.configuration_confucius4_tts import (
    Confucius4TTSConfig,
)
from vllm_omni.model_executor.models.confucius4_tts.confucius4_tts_latent_recompute import (
    Confucius4TTST2SRecompute,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 32
HEADS = 4
LAYERS = 2
COND_DIM = 12
BOS, EOS = 60, 61


class _SpeakerEncoderStub(nn.Module):
    """Mean-pools the reference features; the real ECAPA encoder is irrelevant here."""

    def __init__(self, _config) -> None:
        super().__init__()
        self.proj = nn.Linear(COND_DIM, HIDDEN)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features.mean(dim=1))


def _config() -> Confucius4TTSConfig:
    return Confucius4TTSConfig(
        num_layers=LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        max_text_tokens=16,
        max_semantic_tokens=20,
        text_vocab_size=50,
        text_embedding_dim=8,
        speaker_embedding_dim=COND_DIM,
        vocab_size=62,
        bos_token_id=BOS,
        eos_token_id=EOS,
    )


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> Confucius4TTST2SRecompute:
    monkeypatch.setattr(recompute_module, "build_speaker_encoder", _SpeakerEncoderStub)
    torch.manual_seed(0)
    model = Confucius4TTST2SRecompute(_config()).float().eval()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.2)
    return model


def _inputs(num_codes: int = 6):
    generator = torch.Generator().manual_seed(1)
    text_ids = torch.randint(0, 50, (5,), generator=generator)
    condition = torch.randn(1, 7, COND_DIM, generator=generator)
    codes = torch.randint(0, 60, (num_codes,), generator=generator)
    return text_ids, condition, codes


def _reference_hidden(module: Confucius4TTST2SRecompute, text_ids, condition, codes) -> torch.Tensor:
    """Upstream layout: HF GPT2Model with zeroed positions over the same embeddings."""
    gpt = GPT2Model(
        GPT2Config(
            vocab_size=62,
            n_positions=64,
            n_embd=HIDDEN,
            n_layer=LAYERS,
            n_head=HEADS,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    ).eval()
    state = {
        name.removeprefix("transformer."): value
        for name, value in module.state_dict().items()
        if name.startswith("transformer.")
    }
    missing = gpt.load_state_dict(state, strict=False)
    assert not [key for key in missing.missing_keys if not key.startswith(("wte", "wpe"))]
    with torch.no_grad():
        gpt.wpe.weight.zero_()
        speaker = module.speaker_encoder(condition).unsqueeze(1)
        text = module.text_position_embedding(module.text_projector(text_ids.reshape(1, -1)))
        semantic_ids = torch.cat([torch.tensor([[BOS]]), codes.reshape(1, -1), torch.tensor([[EOS]])], dim=1)
        semantic = module.semantic_position_embedding(module.semantic_embedding(semantic_ids))
        embeds = torch.cat([speaker, text, semantic], dim=1)
        return gpt(inputs_embeds=embeds).last_hidden_state


def test_matches_transformers_gpt2_and_slices_semantic_span(module) -> None:
    text_ids, condition, codes = _inputs()

    latent = module(text_ids, condition, codes, BOS, EOS)

    reference = _reference_hidden(module, text_ids, condition, codes)
    start = 1 + text_ids.shape[0]
    assert latent.shape == (codes.shape[0], HIDDEN)
    # Upstream slices ``hidden[:, 1 + T_text : -2]``: BOS .. c_{T-1}.
    torch.testing.assert_close(latent, reference[0, start:-2], atol=1e-5, rtol=1e-5)


def test_row_i_depends_only_on_tokens_before_it(module) -> None:
    text_ids, condition, codes = _inputs(num_codes=6)
    full = module(text_ids, condition, codes, BOS, EOS)

    # Row i is the state that predicted codes[i]: it may see codes[:i] only.
    for keep in range(1, codes.shape[0]):
        prefix = module(text_ids, condition, codes[:keep], BOS, EOS)
        torch.testing.assert_close(prefix, full[:keep], atol=1e-6, rtol=1e-6)

    changed = codes.clone()
    changed[-1] = (changed[-1] + 1) % 60
    torch.testing.assert_close(module(text_ids, condition, changed, BOS, EOS)[:-1], full[:-1])
    assert torch.equal(module(text_ids, condition, changed, BOS, EOS), full)


def test_latent_does_not_depend_on_the_eos_id(module) -> None:
    text_ids, condition, codes = _inputs()
    first = module(text_ids, condition, codes, BOS, EOS)
    other = module(text_ids, condition, codes, BOS, 7)
    assert torch.equal(first, other)


def test_accepts_unbatched_condition_vector(module) -> None:
    text_ids, condition, codes = _inputs()
    batched = module(text_ids, condition, codes, BOS, EOS)
    unbatched = module(text_ids, condition[0], codes, BOS, EOS)
    assert torch.equal(batched, unbatched)


def test_sequence_budget_matches_position_table(module) -> None:
    text_ids, condition, _ = _inputs()
    # max_semantic_tokens=20 leaves room for 18 codes plus BOS and EOS.
    assert module(text_ids, condition, torch.zeros(18, dtype=torch.long), BOS, EOS).shape == (18, HIDDEN)
    with pytest.raises(ValueError, match="exceeds 18 tokens"):
        module(text_ids, condition, torch.zeros(19, dtype=torch.long), BOS, EOS)
    with pytest.raises(ValueError, match="exceeds 16 tokens"):
        module(torch.zeros(17, dtype=torch.long), condition, torch.zeros(2, dtype=torch.long), BOS, EOS)


def test_load_state_copies_weights_and_ignores_training_heads(module) -> None:
    state = {name: torch.full_like(param, 0.5) for name, param in module.state_dict().items()}
    state["final_norm.weight"] = torch.ones(HIDDEN)
    state["semantic_head.weight"] = torch.ones(62, HIDDEN)

    loaded = module.load_state(state)

    assert loaded == set(dict(module.named_parameters()))
    assert torch.equal(module.semantic_embedding.weight, torch.full_like(module.semantic_embedding.weight, 0.5))


def test_load_state_rejects_missing_and_misshaped_tensors(module) -> None:
    state = {name: param.clone() for name, param in module.state_dict().items()}

    missing = {k: v for k, v in state.items() if k != "transformer.ln_f.weight"}
    with pytest.raises(RuntimeError, match="missing parameters: transformer.ln_f.weight"):
        module.load_state(missing)

    bad = {**state, "semantic_embedding.weight": torch.zeros(3, 3)}
    with pytest.raises(RuntimeError, match="shape mismatch for semantic_embedding.weight"):
        module.load_state(bad)


def test_load_state_warns_for_unknown_keys(module, monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(recompute_module.logger, "warning", lambda msg, *args: warnings.append(msg % args))
    state = {name: param.clone() for name, param in module.state_dict().items()}
    state["transformer.wte.weight"] = torch.zeros(1)

    module.load_state(state)

    assert warnings and "transformer.wte.weight" in warnings[0]
