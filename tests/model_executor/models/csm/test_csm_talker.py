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

from vllm_omni.model_executor.models.csm.csm_mimi import CODEC_PREFIX, codec_weight_name
from vllm_omni.model_executor.models.csm.csm_talker import (
    REFERENCE_AUDIO_KEY,
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


def _build_talker(
    config: CsmConfig, monkeypatch: pytest.MonkeyPatch, **model_config
) -> CsmTalkerForConditionalGeneration:
    single_rank = SimpleNamespace(rank_in_group=0, world_size=1, is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(parallel_state, "_PP", single_rank)
    monkeypatch.setattr(parallel_state, "_TP", single_rank)
    vllm_config = VllmConfig()
    object.__setattr__(
        vllm_config,
        "model_config",
        _ModelConfig(dtype=torch.float32, hf_config=config, model="csm-test", **model_config),
    )
    with set_current_vllm_config(vllm_config):
        return CsmTalkerForConditionalGeneration(vllm_config=vllm_config).eval()


@pytest.fixture(scope="module")
def reference() -> CsmForConditionalGeneration:
    torch.manual_seed(0)
    model = CsmForConditionalGeneration(_config()).eval()
    with torch.no_grad():
        # A fresh Mimi codebook encodes every frame to code 0; give it content.
        for name, buffer in model.codec_model.named_buffers():
            if name.endswith("embed_sum"):
                buffer.normal_()
    return model


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

    assert {_expected_target(name) for name in reference.state_dict()} == names
    # The codec is loaded for reference audio.
    assert sum(name.startswith(CODEC_PREFIX) for name in names) == len(reference.codec_model.state_dict())


def test_loaded_tensors_equal_the_checkpoint(
    reference: CsmForConditionalGeneration, loaded: tuple[CsmTalkerForConditionalGeneration, set[str]]
) -> None:
    talker, _ = loaded
    state = reference.state_dict()
    own = dict(talker.named_parameters())
    own.update({f"depth_decoder.{name}": tensor for name, tensor in talker.depth_decoder.state_dict().items()})

    own.update({CODEC_PREFIX + name: tensor for name, tensor in talker.codec_model.state_dict().items()})

    for name, tensor in state.items():
        if _expected_target(name) != name:
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
        expected = reference.lm_head(hidden)
        # The padding id is the stop token: closed unless forced.
        expected[:, talker.stop_token_id] = float("-inf")
        torch.testing.assert_close(talker.compute_logits(hidden), expected)


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


def test_talker_mtp_computes_a_stop_token_row_on_a_valid_code(
    loaded: tuple[CsmTalkerForConditionalGeneration, set[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async scheduling hands the stop token back as a first codebook; it is never embedded."""
    talker, _ = loaded
    hidden_size = talker.config.hidden_size
    hidden = torch.randn((2, hidden_size), generator=torch.Generator().manual_seed(3))
    zeros = torch.zeros((2, hidden_size))
    seen: list[torch.Tensor] = []
    generate_frame = talker.generate_frame

    def recording(backbone_hidden, first_codes, **sampling):
        seen.append(first_codes.clone())
        return generate_frame(backbone_hidden, first_codes, **sampling)

    monkeypatch.setattr(talker, "generate_frame", recording)
    first = torch.tensor([[talker.stop_token_id], [7]])
    try:
        with torch.no_grad():
            embeds, codes = talker.talker_mtp(first, zeros, hidden, zeros, do_sample=False, req_ids=["stopped", "live"])
            flags = {req_id: bool(flag) for req_id, flag in talker._stop_flags.items()}
            _, alone = talker.talker_mtp(first[1:], zeros[1:], hidden[1:], zeros[1:], do_sample=False)
            discarded, _ = generate_frame(hidden[:1], torch.tensor([0]))
    finally:
        talker.on_requests_finished(["stopped", "live"])

    assert seen[0].tolist() == [0, 7]
    assert codes.shape == (2, talker.num_codebooks) and not codes[0].any()
    assert torch.isfinite(embeds).all()
    # The row that received the stop token stays stopped although its discarded frame was no stop frame.
    assert not stop_frame_mask(discarded).any()
    assert flags == {"stopped": True, "live": False}
    # The discarded row does not change the live one.
    assert int(codes[1, 0]) == 7 and codes[1, 1:].any()
    assert torch.equal(codes[1], alone[0])


def test_codec_loads_from_the_flat_names_of_the_checkpoint(
    reference: CsmForConditionalGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sesame/csm-1b stores the Mimi weights under the names used before transformers 5.21."""
    flat = {
        (CODEC_PREFIX + codec_weight_name(name[len(CODEC_PREFIX) :], nested=False))
        if name.startswith(CODEC_PREFIX)
        else name: tensor
        for name, tensor in reference.state_dict().items()
    }
    talker = _build_talker(reference.config, monkeypatch)

    talker.load_weights(flat.items())

    assert talker.codec_model.encoder.downsample.conv.weight.dtype == torch.float32
    for name, tensor in reference.codec_model.state_dict().items():
        assert torch.equal(talker.codec_model.state_dict()[name], tensor), name


def _reference_prompt(config: CsmConfig, frames: int) -> torch.Tensor:
    """Token ids of a reference turn, its audio placeholders and audio-EOS, then the turn to speak."""
    return torch.tensor([11, 12, 13, *[config.audio_token_id] * frames, config.audio_eos_token_id, 21, 22, 23, 24])


def _waveform(samples: int, seed: int = 4) -> torch.Tensor:
    return 0.1 * torch.randn(samples, generator=torch.Generator().manual_seed(seed))


def _transformers_prompt_embeds(
    reference: CsmForConditionalGeneration, input_ids: torch.Tensor, waveform: torch.Tensor
) -> torch.Tensor:
    return reference._merge_input_ids_with_input_values(
        input_ids=input_ids[None],
        input_values=waveform[None, None],
        input_values_cutoffs=torch.tensor([[waveform.shape[0]]]),
    )["inputs_embeds"][0]


@pytest.mark.parametrize("samples", [1920 * 6, 1920 * 6 + 1, 5000])
def test_reference_audio_prompt_embeddings_match_transformers(
    reference: CsmForConditionalGeneration, loaded: tuple[CsmTalkerForConditionalGeneration, set[str]], samples: int
) -> None:
    talker, _ = loaded
    waveform = _waveform(samples)
    frames = -(-samples // 1920)
    input_ids = _reference_prompt(reference.config, frames)
    with torch.no_grad():
        expected = _transformers_prompt_embeds(reference, input_ids, waveform)
        # The reference frames differ from each other and from the audio-EOS frame.
        codes = reference.codec_model.encode(waveform[None, None]).audio_codes
        assert codes.shape == (1, 32, frames) and codes.unique().numel() > 1
        _, embeds, update = talker.preprocess(
            input_ids,
            None,
            req_id="a",
            _omni_is_prefill=True,
            _omni_num_computed_tokens=0,
            _omni_prompt_len=input_ids.shape[0],
            **{REFERENCE_AUDIO_KEY: waveform},
        )

    torch.testing.assert_close(embeds, expected)
    assert update == {}
    # Text positions keep their text embeddings; the placeholders do not.
    text_embeds = talker.embed_input_ids(input_ids)
    assert torch.equal(embeds[:3], text_embeds[:3]) and torch.equal(embeds[-4:], text_embeds[-4:])
    assert not torch.isclose(embeds[3:-4], text_embeds[3:-4]).all(dim=-1).any()
    # Nothing is kept once the whole prompt is embedded.
    assert talker._references == {}


@pytest.mark.parametrize("cuts", [(2,), (5,), (3, 9), (4, 10, 12), (1, 2, 3, 9, 10)])
def test_reference_audio_survives_a_prefill_split_across_steps(
    reference: CsmForConditionalGeneration,
    loaded: tuple[CsmTalkerForConditionalGeneration, set[str]],
    cuts: tuple[int, ...],
) -> None:
    talker, _ = loaded
    waveform = _waveform(1920 * 6)
    input_ids = _reference_prompt(reference.config, 6)
    bounds = [0, *cuts, input_ids.shape[0]]
    spans = []
    with torch.no_grad():
        expected = _transformers_prompt_embeds(reference, input_ids, waveform)
        for begin, end in zip(bounds[:-1], bounds[1:]):
            _, embeds, _ = talker.preprocess(
                input_ids[begin:end],
                None,
                req_id="a",
                _omni_is_prefill=True,
                _omni_num_computed_tokens=begin,
                _omni_prompt_len=input_ids.shape[0],
                **{REFERENCE_AUDIO_KEY: waveform},
            )
            spans.append(embeds)
            assert ("a" in talker._references) == (end < input_ids.shape[0])

    torch.testing.assert_close(torch.cat(spans), expected)


def test_audio_tokens_without_a_recording_keep_their_text_embeddings(
    loaded: tuple[CsmTalkerForConditionalGeneration, set[str]],
) -> None:
    talker, _ = loaded
    input_ids = _reference_prompt(talker.config, 2)
    with torch.no_grad():
        _, embeds, _ = talker.preprocess(input_ids, None, req_id="a", _omni_is_prefill=True)
        assert torch.equal(embeds, talker.embed_input_ids(input_ids))


@pytest.mark.parametrize(
    "edit",
    [
        lambda ids, config: torch.cat((ids[:3], ids[4:])),  # one placeholder short
        lambda ids, config: torch.cat((ids[:3], ids[3:4], ids[3:])),  # one placeholder too many
        lambda ids, config: ids[ids != config.audio_eos_token_id],  # no audio-EOS token
        lambda ids, config: torch.cat((ids, ids[3:4])),  # a stray placeholder after the turn
        lambda ids, config: ids[(ids != config.audio_token_id) & (ids != config.audio_eos_token_id)],  # text only
        lambda ids, config: ids[:6],  # the prompt ends inside the recording
    ],
)
def test_prompt_that_does_not_match_the_recording_is_rejected(
    loaded: tuple[CsmTalkerForConditionalGeneration, set[str]], edit
) -> None:
    talker, _ = loaded
    input_ids = edit(_reference_prompt(talker.config, 6), talker.config)
    with torch.no_grad(), pytest.raises(ValueError, match="does not match its reference audio"):
        talker.preprocess(
            input_ids,
            None,
            req_id="a",
            _omni_is_prefill=True,
            _omni_num_computed_tokens=0,
            _omni_prompt_len=input_ids.shape[0],
            **{REFERENCE_AUDIO_KEY: _waveform(1920 * 6)},
        )
    assert talker._references == {}


def test_finished_requests_drop_their_reference(loaded: tuple[CsmTalkerForConditionalGeneration, set[str]]) -> None:
    talker, _ = loaded
    input_ids = _reference_prompt(talker.config, 6)
    with torch.no_grad():
        # Aborted after the first prefill span.
        talker.preprocess(
            input_ids[:5],
            None,
            req_id="a",
            _omni_is_prefill=True,
            _omni_num_computed_tokens=0,
            _omni_prompt_len=input_ids.shape[0],
            **{REFERENCE_AUDIO_KEY: _waveform(1920 * 6)},
        )
    assert set(talker._references) == {"a"}

    talker.on_requests_finished(["a"])

    assert talker._references == {}
