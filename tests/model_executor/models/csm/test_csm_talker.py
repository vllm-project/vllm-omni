# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the CSM stage-0 core: weight mapping, embeddings, heads and frame rules.

The backbone's attention needs the engine's KV cache, so its forward pass is
not run here; these tests cover everything around it.
"""

from collections.abc import Iterator
from fnmatch import fnmatch
from types import SimpleNamespace

import pytest
import torch
import vllm.distributed.parallel_state as parallel_state
from transformers import CsmConfig, CsmForConditionalGeneration
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_omni.model_executor.models.csm.csm_talker import (
    CsmTalkerForConditionalGeneration,
    empty_frame_mask,
    stop_frame_mask,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

# Weight files of sesame/csm-1b (revision c92a71e1).
# RoPE settings of sesame/csm-1b (both stacks use the llama3 variant).
BACKBONE_ROPE = {
    "rope_type": "llama3",
    "rope_theta": 500000,
    "factor": 32.0,
    "high_freq_factor": 0.5,
    "low_freq_factor": 0.125,
    "original_max_position_embeddings": 1024,
}
DEPTH_ROPE = {
    "rope_type": "llama3",
    "rope_theta": 500000,
    "factor": 32.0,
    "high_freq_factor": 0.0078125,
    "low_freq_factor": 0.001953125,
    "original_max_position_embeddings": 16,
}

HUB_WEIGHT_FILES = [
    "ckpt.pt",
    "model.safetensors",
    "transformers-00001-of-00002.safetensors",
    "transformers-00002-of-00002.safetensors",
]


class _ModelConfig(SimpleNamespace):
    """Stand-in for vLLM's ModelConfig: unset fields read as None.

    The attention layer's constructor reads a few optional ModelConfig fields;
    a full ModelConfig would need a model directory on disk.
    """

    def __getattr__(self, name: str) -> None:
        return None


def _config(**overrides) -> CsmConfig:
    return CsmConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        depth_decoder_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "backbone_hidden_size": 64,
            "rope_parameters": dict(DEPTH_ROPE),
        },
        rope_parameters=dict(BACKBONE_ROPE),
        **overrides,
    )


def _build_talker(config: CsmConfig, monkeypatch: pytest.MonkeyPatch) -> CsmTalkerForConditionalGeneration:
    single_rank = SimpleNamespace(rank_in_group=0, world_size=1, is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(parallel_state, "_PP", single_rank)
    monkeypatch.setattr(parallel_state, "_TP", single_rank)
    vllm_config = VllmConfig()
    object.__setattr__(
        vllm_config, "model_config", _ModelConfig(dtype=torch.float32, hf_config=config, model="csm-test")
    )
    with set_current_vllm_config(vllm_config):
        return CsmTalkerForConditionalGeneration(vllm_config=vllm_config).eval()


@pytest.fixture(scope="module")
def reference() -> CsmForConditionalGeneration:
    torch.manual_seed(0)
    return CsmForConditionalGeneration(_config()).eval()


@pytest.fixture(scope="module")
def loaded(reference: CsmForConditionalGeneration) -> Iterator[tuple[CsmTalkerForConditionalGeneration, set[str]]]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        talker = _build_talker(reference.config, monkeypatch)
        names = talker.load_weights(reference.state_dict().items())
        yield talker, names


def _expected_target(name: str) -> str:
    """Where a transformers weight name lands in the talker."""
    if not name.startswith("backbone_model.layers."):
        return name
    for source, target in (
        (".q_proj.", ".qkv_proj."),
        (".k_proj.", ".qkv_proj."),
        (".v_proj.", ".qkv_proj."),
        (".gate_proj.", ".gate_up_proj."),
        (".up_proj.", ".gate_up_proj."),
    ):
        name = name.replace(source, target)
    return name


def test_every_parameter_is_loaded_and_every_weight_is_used(
    reference: CsmForConditionalGeneration, loaded: tuple[CsmTalkerForConditionalGeneration, set[str]]
) -> None:
    talker, names = loaded
    own = {name for name, _ in talker.named_parameters()}
    assert own <= names

    stage_weights = {name for name in reference.state_dict() if not name.startswith("codec_model.")}
    assert {_expected_target(name) for name in stage_weights} == names
    assert not any(name.startswith("codec_model.") for name in names)


def test_loaded_tensors_equal_the_checkpoint(
    reference: CsmForConditionalGeneration, loaded: tuple[CsmTalkerForConditionalGeneration, set[str]]
) -> None:
    talker, _ = loaded
    state = reference.state_dict()
    own = dict(talker.named_parameters())
    own.update({f"depth_decoder.{name}": tensor for name, tensor in talker.depth_decoder.state_dict().items()})

    for name, tensor in state.items():
        if name.startswith("codec_model.") or _expected_target(name) != name:
            continue
        assert torch.equal(own[name], tensor), name

    for layer in range(reference.config.num_hidden_layers):
        prefix = f"backbone_model.layers.{layer}"
        qkv = torch.cat([state[f"{prefix}.self_attn.{part}_proj.weight"] for part in "qkv"])
        assert torch.equal(own[f"{prefix}.self_attn.qkv_proj.weight"], qkv)
        gate_up = torch.cat([state[f"{prefix}.mlp.{part}_proj.weight"] for part in ("gate", "up")])
        assert torch.equal(own[f"{prefix}.mlp.gate_up_proj.weight"], gate_up)


def test_depth_decoder_shares_the_backbone_audio_table(
    loaded: tuple[CsmTalkerForConditionalGeneration, set[str]],
) -> None:
    talker, _ = loaded
    assert (
        talker.depth_decoder.model.embed_tokens.weight is talker.backbone_model.embed_tokens.embed_audio_tokens.weight
    )


def test_embeddings_and_first_codebook_head_match_transformers(
    reference: CsmForConditionalGeneration, loaded: tuple[CsmTalkerForConditionalGeneration, set[str]]
) -> None:
    talker, _ = loaded
    generator = torch.Generator().manual_seed(1)
    text_ids = torch.randint(0, reference.config.text_vocab_size, (7,), generator=generator)
    codes = torch.randint(0, reference.config.vocab_size, (3, reference.config.num_codebooks), generator=generator)
    hidden = torch.randn(5, reference.config.hidden_size, generator=generator)
    with torch.no_grad():
        torch.testing.assert_close(talker.embed_input_ids(text_ids), reference.embed_text_tokens(text_ids))
        torch.testing.assert_close(
            talker.embed_audio_frames(codes), reference.backbone_model.embed_tokens(codes[:, None, :])[:, 0]
        )
        torch.testing.assert_close(talker.compute_logits(hidden), reference.lm_head(hidden))


def test_generate_frame_matches_transformers_greedy(
    reference: CsmForConditionalGeneration, loaded: tuple[CsmTalkerForConditionalGeneration, set[str]]
) -> None:
    talker, _ = loaded
    generator = torch.Generator().manual_seed(2)
    hidden = torch.randn(2, reference.config.hidden_size, generator=generator)
    first_codes = torch.randint(0, reference.config.vocab_size, (2,), generator=generator)
    steps = reference.config.num_codebooks - 1
    with torch.no_grad():
        codes, frame_embeds = talker.generate_frame(hidden, first_codes)
        expected = reference.depth_decoder.generate(
            input_ids=torch.nn.functional.pad(first_codes[:, None], (1, 0), value=0),
            backbone_last_hidden_state=hidden.clone(),
            max_new_tokens=steps,
            min_new_tokens=steps,
            do_sample=False,
        )
        expected = (expected if isinstance(expected, torch.Tensor) else expected.sequences)[:, 1:]
        assert torch.equal(codes, expected)
        torch.testing.assert_close(frame_embeds, talker.embed_audio_frames(codes))


def test_only_the_transformers_shards_are_read() -> None:
    patterns = CsmTalkerForConditionalGeneration.allow_patterns_overrides
    selected = [name for name in HUB_WEIGHT_FILES if any(fnmatch(name, pattern) for pattern in patterns)]
    assert selected == ["transformers-00001-of-00002.safetensors", "transformers-00002-of-00002.safetensors"]
    assert CsmTalkerForConditionalGeneration.fall_back_to_pt_during_load is False


def test_untied_codebook_embeddings_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="untied codebook embeddings"):
        _build_talker(_config(tie_codebooks_embeddings=False), monkeypatch)


def test_stop_and_empty_frame_rules() -> None:
    num_codebooks = 32
    all_zero = torch.zeros(num_codebooks, dtype=torch.long)
    last_differs = all_zero.clone()
    last_differs[-1] = 7
    first_is_zero = torch.full((num_codebooks,), 5)
    first_is_zero[0] = 0
    middle_differs = all_zero.clone()
    middle_differs[10] = 3
    frames = torch.stack((all_zero, last_differs, first_is_zero, middle_differs))

    # Generation stops on the first 31 codebooks; audio is cut only when all 32 are zero.
    assert stop_frame_mask(frames).tolist() == [True, True, False, False]
    assert empty_frame_mask(frames).tolist() == [True, False, False, False]
    assert stop_frame_mask(frames[None]).shape == (1, 4)
