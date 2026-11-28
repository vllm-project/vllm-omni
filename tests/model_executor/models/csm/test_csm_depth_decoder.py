# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the CSM depth decoder against transformers' reference implementation."""

import copy

import pytest
import torch
from transformers import CsmConfig, CsmForConditionalGeneration
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from vllm_omni.model_executor.models.csm.csm_depth_decoder import (
    CsmDepthDecoder,
    apply_sampling_filters,
    draw_frame_uniforms,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

BACKBONE_HIDDEN = 64

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


def _config(depth_rope: dict | None = None) -> CsmConfig:
    return CsmConfig(
        hidden_size=BACKBONE_HIDDEN,
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
            "backbone_hidden_size": BACKBONE_HIDDEN,
            "rope_parameters": dict(DEPTH_ROPE if depth_rope is None else depth_rope),
        },
        rope_parameters=dict(BACKBONE_ROPE),
    )


@pytest.fixture(scope="module")
def reference() -> CsmForConditionalGeneration:
    torch.manual_seed(0)
    return CsmForConditionalGeneration(_config()).eval()


def _new_decoder(reference: CsmForConditionalGeneration) -> CsmDepthDecoder:
    decoder = CsmDepthDecoder(reference.config.depth_decoder_config).eval()
    decoder.load_state_dict(reference.depth_decoder.state_dict(), strict=True)
    return decoder


@pytest.fixture(scope="module")
def decoder(reference: CsmForConditionalGeneration) -> CsmDepthDecoder:
    return _new_decoder(reference)


def _inputs(batch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, BACKBONE_HIDDEN, generator=generator)
    first_codes = torch.randint(0, 2051, (batch,), generator=generator)
    return hidden, first_codes


def _reference_codes(
    reference: CsmForConditionalGeneration, hidden: torch.Tensor, first_codes: torch.Tensor
) -> torch.Tensor:
    """Greedy frame from transformers; position 0 of its output is a placeholder."""
    steps = reference.config.num_codebooks - 1
    with torch.no_grad():
        output = reference.depth_decoder.generate(
            input_ids=torch.nn.functional.pad(first_codes[:, None], (1, 0), value=0),
            backbone_last_hidden_state=hidden.clone(),
            max_new_tokens=steps,
            min_new_tokens=steps,
            do_sample=False,
        )
    sequences = output if isinstance(output, torch.Tensor) else output.sequences
    return sequences[:, 1:]


def test_parameter_names_match_transformers(reference: CsmForConditionalGeneration) -> None:
    ours = CsmDepthDecoder(reference.config.depth_decoder_config).state_dict()
    theirs = reference.depth_decoder.state_dict()
    assert set(ours) == set(theirs)
    assert {name: tuple(tensor.shape) for name, tensor in ours.items()} == {
        name: tuple(tensor.shape) for name, tensor in theirs.items()
    }


def test_greedy_frame_matches_transformers(reference: CsmForConditionalGeneration, decoder: CsmDepthDecoder) -> None:
    hidden, first_codes = _inputs(batch=4, seed=1)
    with torch.no_grad():
        codes, frame_embeds = decoder.generate_frame(hidden, first_codes)

    expected = _reference_codes(reference, hidden, first_codes)
    assert codes.shape == (4, reference.config.num_codebooks)
    assert torch.equal(codes, expected)
    with torch.no_grad():
        expected_embeds = reference.backbone_model.embed_tokens(expected[:, None, :])[:, 0]
    torch.testing.assert_close(frame_embeds, expected_embeds, rtol=0, atol=1e-6)


def test_batch_matches_one_row_at_a_time(decoder: CsmDepthDecoder) -> None:
    hidden, first_codes = _inputs(batch=3, seed=2)
    with torch.no_grad():
        batched, batched_embeds = decoder.generate_frame(hidden, first_codes)
        for row in range(3):
            single, single_embeds = decoder.generate_frame(hidden[row : row + 1], first_codes[row : row + 1])
            assert torch.equal(single[0], batched[row])
            torch.testing.assert_close(single_embeds[0], batched_embeds[row], rtol=0, atol=1e-5)


def test_scratch_from_an_earlier_frame_is_not_read(
    reference: CsmForConditionalGeneration, decoder: CsmDepthDecoder
) -> None:
    earlier_hidden, earlier_codes = _inputs(batch=4, seed=3)
    hidden, first_codes = _inputs(batch=2, seed=4)
    with torch.no_grad():
        decoder.generate_frame(earlier_hidden, earlier_codes)
        reused, _ = decoder.generate_frame(hidden, first_codes)
        fresh, _ = _new_decoder(reference).generate_frame(hidden, first_codes)
    assert torch.equal(reused, fresh)


def test_reserved_scratch_is_reused(reference: CsmForConditionalGeneration) -> None:
    decoder = _new_decoder(reference)
    decoder.reserve(4)
    scratch = decoder._kv_scratch
    assert scratch is not None and scratch.shape[2] == 4
    with torch.no_grad():
        decoder.generate_frame(*_inputs(batch=3, seed=5))
    assert decoder._kv_scratch is scratch


@pytest.mark.parametrize(
    ("temperature", "top_k", "top_p"),
    [(1.0, 0, 1.0), (0.9, 50, 1.0), (0.7, 5, 1.0), (1.0, 0, 0.8), (0.9, 5, 0.6), (1.3, 3, 0.3)],
)
def test_sampling_filters_match_transformers_warpers(temperature: float, top_k: int, top_p: float) -> None:
    # Few distinct values, so the k-th score is tied in most rows.
    scores = torch.randint(0, 6, (8, 40), generator=torch.Generator().manual_seed(6)).float()
    warpers = LogitsProcessorList()
    if temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        warpers.append(TopKLogitsWarper(top_k=top_k))
    if top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=top_p))

    expected = warpers(torch.zeros((8, 1), dtype=torch.long), scores.clone())
    actual = apply_sampling_filters(scores.clone(), temperature=temperature, top_k=top_k, top_p=top_p)
    assert torch.equal(actual, expected)


def test_top_k_keeps_scores_tied_with_the_kth() -> None:
    scores = torch.tensor([[3.0, 3.0, 3.0, 1.0]])
    kept = torch.isfinite(apply_sampling_filters(scores, top_k=2))
    assert kept.tolist() == [[True, True, True, False]]


def test_seeded_rows_do_not_depend_on_the_batch(decoder: CsmDepthDecoder) -> None:
    hidden, first_codes = _inputs(batch=3, seed=7)
    seeds = [11, 22, 33]
    sampling = {"do_sample": True, "temperature": 0.9, "top_k": 50}

    def frame(rows: list[int]) -> torch.Tensor:
        uniforms = draw_frame_uniforms(
            len(rows),
            decoder.num_steps,
            decoder.vocab_size,
            device="cpu",
            generators=[torch.Generator().manual_seed(seeds[row]) for row in rows],
        )
        with torch.no_grad():
            codes, _ = decoder.generate_frame(hidden[rows], first_codes[rows], uniforms=uniforms, **sampling)
        return codes

    together = frame([0, 1, 2])
    assert torch.equal(frame([2, 0, 1]), together[[2, 0, 1]])
    for row in range(3):
        assert torch.equal(frame([row])[0], together[row])


def test_different_seeds_give_different_frames(decoder: CsmDepthDecoder) -> None:
    hidden, first_codes = _inputs(batch=1, seed=7)
    uniforms = draw_frame_uniforms(
        2,
        decoder.num_steps,
        decoder.vocab_size,
        device="cpu",
        generators=[torch.Generator().manual_seed(seed) for seed in (11, 12)],
    )
    with torch.no_grad():
        codes, _ = decoder.generate_frame(
            hidden.expand(2, -1), first_codes.expand(2), do_sample=True, temperature=0.9, top_k=50, uniforms=uniforms
        )
    assert not torch.equal(codes[0], codes[1])


def test_greedy_consumes_no_random_numbers(decoder: CsmDepthDecoder) -> None:
    hidden, first_codes = _inputs(batch=2, seed=8)
    torch.manual_seed(1234)
    state = torch.get_rng_state()
    with torch.no_grad():
        decoder.generate_frame(hidden, first_codes, do_sample=False, temperature=0.9, top_k=50)
    assert torch.equal(torch.get_rng_state(), state)


def test_top_k_one_sampling_equals_greedy(decoder: CsmDepthDecoder) -> None:
    hidden, first_codes = _inputs(batch=2, seed=9)
    with torch.no_grad():
        greedy, _ = decoder.generate_frame(hidden, first_codes)
        sampled, _ = decoder.generate_frame(hidden, first_codes, do_sample=True, top_k=1)
    assert torch.equal(sampled, greedy)


def test_draw_frame_uniforms_checks_generator_count() -> None:
    with pytest.raises(ValueError, match="expected 2 generators"):
        draw_frame_uniforms(2, 3, 5, device="cpu", generators=[torch.Generator().manual_seed(0)])


def test_seeded_and_unseeded_rows_share_a_batch() -> None:
    def draw() -> torch.Tensor:
        return draw_frame_uniforms(2, 3, 5, device="cpu", generators=[torch.Generator().manual_seed(5), None])

    first, second = draw(), draw()
    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[1], second[1])
    assert bool(((first > 0) & (first < 1)).all())


@pytest.mark.parametrize("depth_rope", [DEPTH_ROPE, {"rope_type": "default", "rope_theta": 500000.0}])
def test_rotary_tables_match_transformers(depth_rope: dict) -> None:
    config = _config(depth_rope).depth_decoder_config
    reference = CsmForConditionalGeneration(_config(depth_rope)).depth_decoder.model.rotary_emb
    decoder = CsmDepthDecoder(config)
    positions = torch.arange(decoder.max_positions)[None]
    cos, sin = reference(torch.zeros(1, dtype=torch.float32), positions)
    torch.testing.assert_close(decoder.rope_cos, cos[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(decoder.rope_sin, sin[0], rtol=0, atol=1e-6)


def test_default_rope_frame_matches_transformers() -> None:
    torch.manual_seed(0)
    reference = CsmForConditionalGeneration(_config({"rope_type": "default", "rope_theta": 500000.0})).eval()
    hidden, first_codes = _inputs(batch=2, seed=10)
    with torch.no_grad():
        codes, _ = _new_decoder(reference).generate_frame(hidden, first_codes)
    assert torch.equal(codes, _reference_codes(reference, hidden, first_codes))


def test_dynamic_rope_is_rejected(reference: CsmForConditionalGeneration) -> None:
    config = copy.deepcopy(reference.config.depth_decoder_config)
    config.rope_parameters = {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": 2.0}
    with pytest.raises(NotImplementedError, match="dynamic RoPE"):
        CsmDepthDecoder(config)
