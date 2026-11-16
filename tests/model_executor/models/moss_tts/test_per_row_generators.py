# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regression tests for MOSS-TTS per-row talker generators."""

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts import modeling_moss_tts_local

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

BATCH = 4
VOCAB = 1024


def _logits() -> torch.Tensor:
    return torch.randn(BATCH, VOCAB, generator=torch.Generator().manual_seed(7))


def _row_generators() -> list[torch.Generator]:
    return [torch.Generator().manual_seed(100 + row) for row in range(BATCH)]


def _sample(
    logits: torch.Tensor,
    generators: list[torch.Generator | None] | None,
) -> torch.Tensor:
    return modeling_moss_tts_local._sample_token(
        logits,
        temperature=1.7,
        top_k=25,
        top_p=0.8,
        do_sample=True,
        generators=generators,
    )


def test_normalize_generators_rejects_length_mismatch() -> None:
    assert modeling_moss_tts_local._normalize_generators(None, BATCH) is None

    generators = _row_generators()
    normalized = modeling_moss_tts_local._normalize_generators(generators, BATCH)
    assert normalized is not None
    assert all(row is expected for row, expected in zip(normalized, generators))

    with pytest.raises(ValueError, match=f"Expected {BATCH} per-row generators, but got {BATCH - 1}"):
        modeling_moss_tts_local._normalize_generators(_row_generators()[:-1], BATCH)
    with pytest.raises(ValueError, match=f"Expected {BATCH} per-row generators, but got {BATCH + 1}"):
        modeling_moss_tts_local._normalize_generators(_row_generators() + [torch.Generator()], BATCH)


def test_sample_token_validates_generator_count() -> None:
    """A short generators list must fail loudly instead of sampling the tail rows
    from the global RNG."""
    logits = _logits()

    with pytest.raises(ValueError, match=f"Expected {BATCH} per-row generators, but got {BATCH - 1}"):
        _sample(logits, _row_generators()[:-1])


def test_sample_token_per_row_generators_are_reproducible() -> None:
    logits = _logits()

    torch.testing.assert_close(_sample(logits, _row_generators()), _sample(logits, _row_generators()))


def test_sample_token_rows_do_not_depend_on_batch_composition() -> None:
    """Each row must be sampled by its own generator alone, so batching a row
    with other requests cannot change its output."""
    logits = _logits()

    batched = _sample(logits, _row_generators())
    for row in range(BATCH):
        single = modeling_moss_tts_local._sample_token(
            logits[row : row + 1],
            temperature=1.7,
            top_k=25,
            top_p=0.8,
            do_sample=True,
            generators=[torch.Generator().manual_seed(100 + row)],
        )
        torch.testing.assert_close(batched[row], single[0])


def test_sample_token_all_none_generators_keep_batched_path() -> None:
    """The runner only passes ``generators`` when a row is explicitly seeded, so
    an all-``None`` list must behave exactly like the scalar batched path."""
    logits = _logits()

    with_generators = modeling_moss_tts_local._sample_token(
        logits,
        temperature=1.7,
        top_k=25,
        top_p=0.8,
        do_sample=True,
        generator=torch.Generator().manual_seed(42),
        generators=[None] * BATCH,
    )
    without_generators = modeling_moss_tts_local._sample_token(
        logits,
        temperature=1.7,
        top_k=25,
        top_p=0.8,
        do_sample=True,
        generator=torch.Generator().manual_seed(42),
    )
    torch.testing.assert_close(with_generators, without_generators)


def test_sample_token_single_seeded_row_is_reproducible() -> None:
    """One seeded row routes the batch through the per-row loop; that row must
    stay reproducible while unseeded rows keep using the global RNG."""
    logits = _logits()

    def run() -> torch.Tensor:
        generators: list[torch.Generator | None] = [None] * BATCH
        generators[1] = torch.Generator().manual_seed(101)
        return _sample(logits, generators)

    torch.testing.assert_close(run()[1], run()[1])


def test_generate_frame_forwards_request_generators() -> None:
    """The frame loop must not drop request-owned RNGs before RVQ sampling."""
    rvq, width, vocab = 4, 3, 16

    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.codec_embedding = torch.nn.ModuleList([torch.nn.Embedding(vocab, width) for _ in range(rvq - 1)])

        def forward(self, embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            return embeds

    @dataclass(frozen=True)
    class LocalConfig:
        rvq: int
        hidden_size: int

    class LocalHarness:
        def __init__(self) -> None:
            self.config = LocalConfig(rvq=rvq, hidden_size=width)
            self.model = Decoder()

    local = LocalHarness()
    heads = torch.nn.ModuleList([torch.nn.Linear(width, vocab, bias=False) for _ in range(rvq)])
    for head in heads:
        head.weight.data.zero_()
    hidden = torch.zeros(2, width)

    def generate(rows: torch.Tensor, generators: list[torch.Generator]) -> torch.Tensor:
        return modeling_moss_tts_local.MossTTSRealtimeLocalTransformer.generate_frame(
            local,
            rows,
            heads,
            top_k=0,
            top_p=1.0,
            generators=generators,
        )

    batched_generators = [torch.Generator().manual_seed(seed) for seed in (17, 29)]
    independent_generators = [torch.Generator().manual_seed(seed) for seed in (17, 29)]
    actual = generate(hidden, batched_generators)
    expected = torch.cat(
        [generate(hidden[row : row + 1], [independent_generators[row]]) for row in range(hidden.shape[0])]
    )

    torch.testing.assert_close(actual, expected)
    for batched, independent in zip(batched_generators, independent_generators, strict=True):
        torch.testing.assert_close(batched.get_state(), independent.get_state())
