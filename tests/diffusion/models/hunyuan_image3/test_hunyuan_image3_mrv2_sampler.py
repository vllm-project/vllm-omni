# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for HunyuanImage3 MRv2 custom sampler (forced transitions,
ratio restriction, comprehension blocking, slot reuse, and zero host sync)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.hunyuan_image3.mrv2_sampling import HunyuanImage3MRv2Sampler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

END_OF_THINK = 100
RECAPTION = 101
END_OF_RECAPTION = 102
ANSWER = 103
BOI = 104
SIZE_TOKEN = 105
EOS = 106
RATIO_START = 200
RATIO_END = 210
RATIO_OTHER_START = 220
RATIO_OTHER_END = 223
TIMESTEP = 224
CFG = 110
IMG_TOK = 111
JOINT_SEP = 112
EOI = 113
VOCAB = 300


class _StubModel:
    """Minimal HunyuanImage3 model stub carrying only sampler attributes."""

    def __init__(self, *, is_comprehension: bool = False):
        self._is_comprehension = is_comprehension
        self._eos_token_id = EOS
        self._end_of_think_id = END_OF_THINK
        self._recaption_id = RECAPTION
        self._end_of_recaption_id = END_OF_RECAPTION
        self._answer_id = ANSWER
        self._mrope_boi_token_id = BOI
        self._size_token_id = SIZE_TOKEN
        self._start_ratio_id = RATIO_START
        self._end_ratio_id = RATIO_END
        self._ratio_other_slices = [(RATIO_OTHER_START, RATIO_OTHER_END + 1)]
        self._all_ratio_ids = set(range(RATIO_START, RATIO_END + 1))
        self._all_ratio_ids.update(range(RATIO_OTHER_START, RATIO_OTHER_END + 1))

        self._stage_transitions: dict[int, list[int]] = {}
        if not is_comprehension:
            self._stage_transitions[END_OF_THINK] = [RECAPTION]
            self._stage_transitions[END_OF_RECAPTION] = [ANSWER, BOI, SIZE_TOKEN]

        self._cfg_token_id = CFG
        self._blocked_token_ids: set[int] = {BOI, EOI, SIZE_TOKEN, CFG, IMG_TOK, TIMESTEP, JOINT_SEP}
        self._blocked_token_ids.update(self._all_ratio_ids)
        self._blocked_token_ids_tensor = None
        self._sampler = None


class _FakeBaseSampler:
    """Minimal MRv2 Sampler stub exposing req_states and recording mutated logits."""

    def __init__(self, max_num_reqs: int = 8, max_seq_len: int = 32, device: str = "cpu"):
        self.device = torch.device(device)
        self.req_states = SimpleNamespace(
            max_num_reqs=max_num_reqs,
            device=self.device,
            all_token_ids=SimpleNamespace(
                gpu=torch.zeros((max_num_reqs, max_seq_len), dtype=torch.int32, device=device)
            ),
            prompt_len=SimpleNamespace(gpu=torch.zeros(max_num_reqs, dtype=torch.int32, device=device)),
            total_len=SimpleNamespace(gpu=torch.zeros(max_num_reqs, dtype=torch.int32, device=device)),
        )
        self.added: list[tuple[int, object]] = []
        self.staged_flushes = 0
        self.last_logits: torch.Tensor | None = None

    def add_request(self, req_idx: int, sampling_params: object) -> None:
        self.added.append((req_idx, sampling_params))

    def apply_staged_writes(self) -> None:
        self.staged_flushes += 1

    def __call__(self, logits: torch.Tensor, input_batch: object) -> torch.Tensor:
        self.last_logits = logits.clone()
        return self.last_logits


def _set_slot_tokens(
    base: _FakeBaseSampler,
    slot: int,
    prompt_tokens: list[int],
    output_tokens: list[int],
) -> int:
    all_toks = prompt_tokens + output_tokens
    base.req_states.all_token_ids.gpu[slot].zero_()
    if all_toks:
        base.req_states.all_token_ids.gpu[slot, : len(all_toks)] = torch.tensor(
            all_toks, dtype=torch.int32, device=base.device
        )
    base.req_states.prompt_len.gpu[slot] = len(prompt_tokens)
    base.req_states.total_len.gpu[slot] = len(all_toks)
    return len(all_toks)


def _make_batch(
    slots: list[int],
    seq_lens: list[int] | None,
    *,
    device: str = "cpu",
    num_draft_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_reqs=len(slots),
        idx_mapping=torch.tensor(slots, dtype=torch.int32, device=device),
        idx_mapping_np=np.array(slots, dtype=np.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device) if seq_lens is not None else None,
        num_draft_tokens=num_draft_tokens,
    )


def _sampling_params(ar_task_mode: str | None = None) -> SimpleNamespace:
    extra_args = {"ar_task_mode": ar_task_mode} if ar_task_mode is not None else None
    return SimpleNamespace(extra_args=extra_args)


def _assert_forced_token(row_logits: torch.Tensor, expected_token: int) -> None:
    min_score = torch.finfo(row_logits.dtype).min
    assert row_logits[expected_token].item() == 0.0
    mask = torch.ones(row_logits.shape[0], dtype=torch.bool, device=row_logits.device)
    mask[expected_token] = False
    assert bool((row_logits[mask] == min_score).all().item())


class TestHunyuanImage3MRv2Sampler:
    def test_mrv2_custom_sampler_hook_registers_wrapper(self):
        from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
            HunyuanImage3ForConditionalGeneration,
        )

        model = _StubModel(is_comprehension=False)
        base = _FakeBaseSampler()
        wrapped, spec_verifier = HunyuanImage3ForConditionalGeneration.mrv2_custom_sampler(model, base)

        assert isinstance(wrapped, HunyuanImage3MRv2Sampler)
        assert spec_verifier is None
        assert wrapped.omni_static_staged_writes is True

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_full_generation_transition_chain(self, dtype: torch.dtype):
        model = _StubModel(is_comprehension=False)
        base = _FakeBaseSampler()
        sampler = HunyuanImage3MRv2Sampler(base, model)
        sampler.add_request(0, _sampling_params())
        sampler.apply_staged_writes()

        prompt = [10, 20, END_OF_THINK]  # trigger inside prompt must be ignored
        steps: list[tuple[list[int], int | None]] = [
            ([], None),  # prefill end: 0 output tokens, prompt </think> ignored
            ([1, 2], None),  # free think text
            ([1, 2, END_OF_THINK], RECAPTION),  # </think> -> <recaption>
            ([1, 2, END_OF_THINK, RECAPTION], None),  # free recaption text
            ([1, 2, END_OF_THINK, RECAPTION, 5, END_OF_RECAPTION], ANSWER),  # </recaption> -> <answer>
            ([1, 2, END_OF_THINK, RECAPTION, 5, END_OF_RECAPTION, ANSWER], BOI),  # -> <boi>
            ([1, 2, END_OF_THINK, RECAPTION, 5, END_OF_RECAPTION, ANSWER, BOI], SIZE_TOKEN),  # -> <img_size_*>
            ([1, 2, END_OF_THINK, RECAPTION, 5, END_OF_RECAPTION, ANSWER, BOI, SIZE_TOKEN], RATIO_OTHER_START + 1),
            (
                [
                    1,
                    2,
                    END_OF_THINK,
                    RECAPTION,
                    5,
                    END_OF_RECAPTION,
                    ANSWER,
                    BOI,
                    SIZE_TOKEN,
                    RATIO_OTHER_START + 1,
                ],
                EOS,
            ),
        ]

        for outputs, expected_forced in steps:
            seq_len = _set_slot_tokens(base, 0, prompt, outputs)
            logits = torch.zeros((1, VOCAB), dtype=dtype)
            logits[0, 50] = 20.0  # non-ratio token with highest raw logit
            logits[0, RATIO_START + 2] = 6.0
            logits[0, RATIO_OTHER_START + 1] = 12.0  # highest ratio token (in extra slice)
            out = sampler(logits, _make_batch([0], [seq_len]))

            if expected_forced is None:
                assert out[0, 50].item() == 20.0
                assert out[0, RECAPTION].item() == 0.0
            else:
                _assert_forced_token(out[0], expected_forced)

    def test_diverged_and_nested_triggers_match_v1_semantics(self):
        model = _StubModel(is_comprehension=False)
        base = _FakeBaseSampler()
        sampler = HunyuanImage3MRv2Sampler(base, model)
        for slot in range(3):
            sampler.add_request(slot, _sampling_params())
        sampler.apply_staged_writes()

        # Slot 0: diverged after </recaption> (999 != ANSWER) -> stop forcing
        l0 = _set_slot_tokens(base, 0, [1], [END_OF_RECAPTION, 999])
        # Slot 1: nested </think> after </recaption> -> most recent trigger wins
        l1 = _set_slot_tokens(base, 1, [1], [END_OF_RECAPTION, ANSWER, END_OF_THINK])
        # Slot 2: chunked prefill (seq_len < prompt_len) -> never force
        _set_slot_tokens(base, 2, [1, 2, END_OF_THINK, 4], [])

        logits = torch.zeros((3, VOCAB))
        logits[:, 50] = 7.0
        out = sampler(logits, _make_batch([0, 1, 2], [l0, l1, 3]))

        assert out[0, 50].item() == 7.0
        _assert_forced_token(out[1], RECAPTION)
        assert out[2, 50].item() == 7.0

    def test_mixed_batch_reordering_and_slot_reuse(self):
        model = _StubModel(is_comprehension=False)
        base = _FakeBaseSampler(max_num_reqs=6)
        sampler = HunyuanImage3MRv2Sampler(base, model)
        min_score = torch.finfo(torch.float32).min

        # Slot 1: comprehension, Slot 4: generation at </think>, Slot 2: generation at </recaption>+ANSWER
        sampler.add_request(1, _sampling_params("comprehension"))
        sampler.add_request(4, _sampling_params(None))
        sampler.add_request(2, _sampling_params("generation"))
        sampler.apply_staged_writes()

        l1 = _set_slot_tokens(base, 1, [7], [END_OF_THINK])
        l4 = _set_slot_tokens(base, 4, [7], [END_OF_THINK])
        l2 = _set_slot_tokens(base, 2, [7], [END_OF_RECAPTION, ANSWER])

        # Reordered batch: [slot 4, slot 1, slot 2] + 1 padded row beyond num_reqs
        logits = torch.zeros((4, VOCAB))
        logits[:, 50] = 7.0
        logits[3, 50] = 9.0
        out = sampler(logits, _make_batch([4, 1, 2], [l4, l1, l2]))

        _assert_forced_token(out[0], RECAPTION)
        assert out[1, 50].item() == 7.0
        for tid in (BOI, EOI, SIZE_TOKEN, CFG, IMG_TOK, TIMESTEP, JOINT_SEP, RATIO_START, RATIO_OTHER_START):
            assert out[1, tid].item() == min_score
        _assert_forced_token(out[2], BOI)
        # Padded row beyond num_reqs stays untouched
        assert out[3, 50].item() == 9.0

        # Slot reuse without explicit apply_staged_writes: lazy sync still picks up new roles
        sampler.add_request(1, _sampling_params(None))  # now generation
        sampler.add_request(4, _sampling_params("comprehension"))  # now comprehension

        l1_new = _set_slot_tokens(base, 1, [9], [END_OF_THINK])
        l4_new = _set_slot_tokens(base, 4, [9], [END_OF_THINK])
        logits2 = torch.zeros((2, VOCAB))
        logits2[:, 50] = 7.0
        out2 = sampler(logits2, _make_batch([1, 4], [l1_new, l4_new]))

        _assert_forced_token(out2[0], RECAPTION)
        assert out2[1, 50].item() == 7.0
        assert out2[1, CFG].item() == min_score
        assert out2[1, BOI].item() == min_score

    def test_total_len_fallback_when_seq_lens_absent(self):
        model = _StubModel(is_comprehension=False)
        base = _FakeBaseSampler()
        sampler = HunyuanImage3MRv2Sampler(base, model)
        sampler.add_request(0, _sampling_params())
        sampler.apply_staged_writes()

        _set_slot_tokens(base, 0, [1], [END_OF_THINK])
        logits = torch.zeros((1, VOCAB))
        logits[0, 50] = 7.0
        out = sampler(logits, _make_batch([0], None))
        _assert_forced_token(out[0], RECAPTION)

    def test_static_comprehension_engine_blocks_scaffold_tokens(self):
        model = _StubModel(is_comprehension=True)
        base = _FakeBaseSampler()
        sampler = HunyuanImage3MRv2Sampler(base, model)
        sampler.add_request(0, _sampling_params(None))
        sampler.apply_staged_writes()

        seq_len = _set_slot_tokens(base, 0, [1], [END_OF_THINK])
        logits = torch.zeros((1, VOCAB))
        logits[0, 50] = 7.0
        min_score = torch.finfo(torch.float32).min
        out = sampler(logits, _make_batch([0], [seq_len]))

        assert out[0, 50].item() == 7.0
        for tid in (BOI, EOI, SIZE_TOKEN, CFG, IMG_TOK, TIMESTEP, JOINT_SEP, RATIO_START, RATIO_OTHER_START):
            assert out[0, tid].item() == min_score

    @pytest.mark.parametrize("is_comprehension", [False, True])
    def test_no_per_step_host_synchronization(self, monkeypatch, is_comprehension: bool):
        model = _StubModel(is_comprehension=is_comprehension)
        base = _FakeBaseSampler()
        sampler = HunyuanImage3MRv2Sampler(base, model)
        sampler.add_request(0, _sampling_params())
        sampler.add_request(1, _sampling_params("comprehension"))
        sampler.apply_staged_writes()

        l0 = _set_slot_tokens(base, 0, [1, 2], [END_OF_RECAPTION, ANSWER, BOI, SIZE_TOKEN])
        l1 = _set_slot_tokens(base, 1, [1, 2], [END_OF_THINK])
        logits = torch.randn((2, VOCAB))
        batch = _make_batch([0, 1], [l0, l1])

        # Warm lazy device-tensor cache once, then forbid any host sync during decode.
        sampler(logits.clone(), batch)

        def fail_sync(*_args, **_kwargs):
            raise AssertionError("HunyuanImage3MRv2Sampler must not perform per-step host sync")

        monkeypatch.setattr(torch.Tensor, "item", fail_sync)
        monkeypatch.setattr(torch.Tensor, "tolist", fail_sync)

        out = sampler(logits.clone(), batch)
        assert out.shape == (2, VOCAB)

    def test_speculative_decoding_fails_closed(self):
        model = _StubModel(is_comprehension=False)
        base = _FakeBaseSampler()
        sampler = HunyuanImage3MRv2Sampler(base, model)
        logits = torch.zeros((2, VOCAB))
        batch = _make_batch([0], [4], num_draft_tokens=1)
        with pytest.raises(NotImplementedError, match="speculative decoding"):
            sampler(logits, batch)
