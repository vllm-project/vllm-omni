# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Drama 3 codebook tower key/value history.

The tower runs teacher-forced over the prompt at prefill and one row per
request at decode. The tests use only ``CodebookTower.run`` and the outputs
of the checkpoint-named layers, so they do not depend on how the key/value
history is stored.
"""

import pytest
import torch

from vllm_omni.model_executor.models.fish_omni.codebook_tower import CodebookTower, TowerBatch
from vllm_omni.model_executor.models.fish_omni.configuration_fish_omni import FishOmniAudioDecoderConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SEMANTIC_BEGIN = 100
SAMPLING = (1.0, 8, 0.9)


def _tiny_tower() -> CodebookTower:
    torch.manual_seed(0)
    config = FishOmniAudioDecoderConfig(
        dim=32,
        n_head=4,
        n_local_heads=2,
        head_dim=8,
        intermediate_size=48,
        num_codebooks=4,
        vocab_size=16,
        shared_enrichment_layers=2,
        rotary_percent=0.5,
    )
    tower = CodebookTower(config, semantic_begin_id=SEMANTIC_BEGIN, cache_slots=4, max_len=32).eval()
    with torch.no_grad():
        for name, param in tower.named_parameters():
            if name.endswith("norm.weight"):
                param.normal_(std=0.1)
    return tower


def _sequence(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Backbone hidden states [rows, dim] and frames [rows + 1, 1 + Q] (text rows, then semantic rows)."""
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn((rows, 32), generator=generator)
    codes = torch.randint(0, 16, (rows + 1, 4), generator=generator)
    text_ids = torch.randint(0, SEMANTIC_BEGIN, (rows + 1,), generator=generator)
    is_text = torch.arange(rows + 1) < 2
    codes[is_text] = 0
    main = torch.where(is_text, text_ids, SEMANTIC_BEGIN + codes[:, 0])
    return hidden, torch.cat((main.unsqueeze(-1), codes), dim=-1)


def _run(tower: CodebookTower, hidden, frames, first: int, end: int, slot: int, sampled: bool = False):
    """Run rows first..end-1 of one request; frame p is the input and p + 1 the target."""
    return tower.run(
        hidden=hidden[first:end],
        input_codes=frames[first:end, 1:],
        input_is_semantic=tower.is_semantic(frames[first:end, 0]),
        next_main_tokens=frames[first + 1 : end + 1, 0],
        next_codes=None if sampled else frames[first + 1 : end + 1, 1:],
        batch=TowerBatch.extend(slot, first, end, torch.device("cpu")),
        sampling=SAMPLING,
    )


class LayerTap:
    """Records each tower layer's output and the residual heads' logits."""

    def __init__(self, tower: CodebookTower) -> None:
        self.outputs: list[torch.Tensor] = []
        self.logits: list[torch.Tensor] = []
        layers = [*tower.audio_decoder_mtp_shared.layers, *tower.audio_decoder_mtp.layers]
        heads = dict(zip(tower.audio_decoder_mtp.layers, tower.audio_decoder_mtp.heads))
        for layer in layers:
            layer.register_forward_hook(self._hook(heads.get(layer)))

    def _hook(self, head):
        def hook(module, args, output):
            self.outputs.append(output.detach().clone())
            if head is not None:
                self.logits.append(head(output).detach().clone())

        return hook

    def take(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        outputs, logits = self.outputs, self.logits
        self.outputs, self.logits = [], []
        return outputs, logits


@torch.no_grad()
@pytest.mark.parametrize("n", [2, 5])
def test_prefill_then_decode_matches_longer_prefill(n):
    tower = _tiny_tower()
    tap = LayerTap(tower)
    hidden, frames = _sequence(n + 1, seed=n)

    full_codes = _run(tower, hidden, frames, 0, n + 1, slot=0)
    full_outputs, full_logits = tap.take()

    _run(tower, hidden, frames, 0, n, slot=1)
    tap.take()
    step_codes = _run(tower, hidden, frames, n, n + 1, slot=1)
    step_outputs, step_logits = tap.take()

    assert torch.equal(step_codes, full_codes[-1:])
    assert len(step_outputs) == len(full_outputs) == 2 + 3
    for full, step in zip(full_outputs + full_logits, step_outputs + step_logits):
        torch.testing.assert_close(step, full[-1:], rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_batched_decode_matches_per_request_decode():
    lengths = (5, 3)
    requests = [_sequence(n + 1, seed=10 + n) for n in lengths]

    # One tower per request decodes alone; a third tower decodes both rows at once.
    single_outputs, single_logits = [], []
    for slot, ((hidden, frames), n) in enumerate(zip(requests, lengths)):
        tower = _tiny_tower()
        tap = LayerTap(tower)
        _run(tower, hidden, frames, 0, n, slot=slot)
        tap.take()
        _run(tower, hidden, frames, n, n + 1, slot=slot)
        outputs, logits = tap.take()
        single_outputs.append(outputs)
        single_logits.append(logits)

    tower = _tiny_tower()
    tap = LayerTap(tower)
    for slot, ((hidden, frames), n) in enumerate(zip(requests, lengths)):
        _run(tower, hidden, frames, 0, n, slot=slot)
    tap.take()
    tower.run(
        hidden=torch.cat([hidden[n : n + 1] for (hidden, _), n in zip(requests, lengths)]),
        input_codes=torch.cat([frames[n : n + 1, 1:] for (_, frames), n in zip(requests, lengths)]),
        input_is_semantic=torch.cat(
            [tower.is_semantic(frames[n : n + 1, 0]) for (_, frames), n in zip(requests, lengths)]
        ),
        next_main_tokens=torch.cat([frames[n + 1 : n + 2, 0] for (_, frames), n in zip(requests, lengths)]),
        next_codes=torch.cat([frames[n + 1 : n + 2, 1:] for (_, frames), n in zip(requests, lengths)]),
        batch=TowerBatch.decode(torch.tensor([0, 1]), torch.tensor(lengths)),
        sampling=SAMPLING,
    )
    batch_outputs, batch_logits = tap.take()

    for row in range(2):
        for batched, single in zip(batch_outputs + batch_logits, single_outputs[row] + single_logits[row]):
            torch.testing.assert_close(batched[row : row + 1], single, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_sampled_decode_returns_code0_from_main_token():
    tower = _tiny_tower()
    hidden, frames = _sequence(4, seed=3)
    _run(tower, hidden, frames, 0, 3, slot=0)
    codes = _run(tower, hidden, frames, 3, 4, slot=0, sampled=True)
    assert codes.shape == (1, 4)
    assert int(codes[0, 0]) == int(frames[4, 0]) - SEMANTIC_BEGIN
    assert bool(((codes >= 0) & (codes < 16)).all())
