# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for HunyuanImage3 AR sampler logic (stage transitions,
ratio restriction, comprehension blocking)."""

import pytest
import torch
from vllm.config import VllmConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Fake token IDs for testing (avoid importing the real model).
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


class FakeSamplerModel:
    """Minimal stub that replicates the sampler-relevant attributes of
    HunyuanImage3ForConditionalGeneration without loading real weights."""

    vllm_config: VllmConfig

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

        # Mirrors the model: the blocklist is always built so generation-mode
        # engines can apply it to per-request comprehension rows (#6088).
        self._cfg_token_id = CFG
        self._blocked_token_ids: set[int] = {BOI, EOI, SIZE_TOKEN, CFG, IMG_TOK, TIMESTEP, JOINT_SEP}
        self._blocked_token_ids.update(self._all_ratio_ids)

    # Bind the real methods from the model class.
    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
        HunyuanImage3ForConditionalGeneration as _Real,
    )

    mrv2_custom_sampler = _Real.mrv2_custom_sampler


class FakeTokenEmbedding:
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                input_ids.to(dtype=torch.float32),
                input_ids.to(dtype=torch.float32) + 0.5,
            ],
            dim=-1,
        )


class FakeEmbeddingModel:
    def __init__(self):
        self.model = FakeTokenEmbedding()
        self._timestep_token_id = TIMESTEP

    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
        HunyuanImage3ForConditionalGeneration as _Real,
    )

    embed_input_ids = _Real.embed_input_ids

    def _timestep_encode(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (1, 2),
            -7.0,
            device=timestep.device,
            dtype=timestep.dtype,
        )


class TestTimestepEmbedding:
    def test_timestep_embedding_avoids_scalar_item_sync(self, monkeypatch):
        model = FakeEmbeddingModel()
        input_ids = torch.tensor([1, TIMESTEP, 2, TIMESTEP], dtype=torch.long)

        def fail_item(_self):
            raise AssertionError("embed_input_ids should not synchronize timestep count")

        monkeypatch.setattr(torch.Tensor, "item", fail_item)

        actual = model.embed_input_ids(input_ids)
        expected = model.model.embed_input_ids(input_ids)
        expected[input_ids == TIMESTEP] = -7.0

        torch.testing.assert_close(actual, expected)

    def test_timestep_embedding_preserves_no_timestep_inputs(self, monkeypatch):
        model = FakeEmbeddingModel()
        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)

        def fail_item(_self):
            raise AssertionError("embed_input_ids should not synchronize timestep count")

        monkeypatch.setattr(torch.Tensor, "item", fail_item)

        actual = model.embed_input_ids(input_ids)

        torch.testing.assert_close(actual, model.model.embed_input_ids(input_ids))


class TestMRv2Sampler:
    @staticmethod
    def _make_sampler(mocker, model, histories, extra_args=None):
        from vllm.sampling_params import SamplingParams

        from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

        num_reqs = len(histories)
        # A prompt ending in a trigger must not activate generation rules.
        prompt = [1, END_OF_RECAPTION]
        all_tokens = torch.zeros(num_reqs, max(map(len, histories)) + len(prompt) + 8, dtype=torch.int32)
        for slot, history in enumerate(histories):
            tokens = prompt + history
            all_tokens[slot, : len(tokens)] = torch.tensor(tokens)
        base_sampler = mocker.Mock(side_effect=lambda logits, input_batch: logits)
        base_sampler.req_states.prompt_len.gpu = torch.full((num_reqs,), len(prompt), dtype=torch.int32)
        base_sampler.req_states.all_token_ids.gpu = all_tokens
        model.vllm_config = mocker.Mock(speculative_config=None)
        state = OmniModelState.__new__(OmniModelState)
        state.model = model
        sampler, rejection = state.custom_sampler(base_sampler)
        assert rejection is None
        for slot in range(num_reqs):
            sampler.add_request(slot, SamplingParams(extra_args=extra_args[slot] if extra_args else None))
        batch = mocker.Mock()
        batch.idx_mapping = torch.arange(num_reqs)
        batch.seq_lens = torch.tensor([len(prompt) + len(history) for history in histories], dtype=torch.int32)
        batch.num_reqs = num_reqs
        return sampler, batch

    @pytest.mark.parametrize("is_comprehension", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("extra_ratios", [False, True])
    def test_sampling_constraints(self, mocker, is_comprehension, dtype, extra_ratios):
        model = FakeSamplerModel(is_comprehension=is_comprehension)
        if not extra_ratios:
            model._ratio_other_slices = []
            model._all_ratio_ids = set(range(RATIO_START, RATIO_END + 1))
        histories = [
            [],
            [END_OF_THINK],
            [END_OF_THINK, RECAPTION],
            [END_OF_RECAPTION],
            [END_OF_RECAPTION, ANSWER],
            [END_OF_RECAPTION, ANSWER, BOI],
            [END_OF_RECAPTION, ANSWER, BOI, SIZE_TOKEN],
            [END_OF_RECAPTION, 50, BOI],
            [RATIO_START],
            [RATIO_OTHER_END],
            [END_OF_RECAPTION, ANSWER, END_OF_THINK],
            [50, 51],
        ]
        extra_args: list[dict[str, str] | None] = [None] * len(histories)
        histories += histories
        extra_args += [{"ar_task_mode": "comprehension"}] * len(extra_args)
        sampler, batch = self._make_sampler(mocker, model, histories, extra_args)
        logits = torch.randn(len(histories), 300, dtype=dtype)
        # A non-ratio token can have the highest score; size still selects
        # the best allowed ratio, including the optional extra ratio range.
        logits[:, 50] = 20
        logits[:, RATIO_START + 3] = 10
        logits[:, RATIO_OTHER_START] = 15
        best_ratio = RATIO_OTHER_START if extra_ratios else RATIO_START + 3
        forced_tokens = [
            None,
            RECAPTION,
            None,
            ANSWER,
            BOI,
            SIZE_TOKEN,
            best_ratio,
            None,
            EOS,
            EOS if extra_ratios else None,
            RECAPTION,
            None,
        ]
        expected = logits.clone()
        min_score = torch.finfo(dtype).min
        for row, forced in enumerate(forced_tokens * 2):
            if is_comprehension or extra_args[row]:
                expected[row, sorted(model._blocked_token_ids)] = min_score
            elif forced is not None:
                expected[row].fill_(min_score)
                expected[row, forced] = 0
        actual = sampler(logits, batch)
        torch.testing.assert_close(actual, expected)
        sampler.base_sampler.assert_called_once()

    def test_reordered_slots_and_reused_slot(self, mocker):
        from vllm.sampling_params import SamplingParams

        model = FakeSamplerModel()
        sampler, batch = self._make_sampler(
            mocker, model, [[END_OF_THINK], [END_OF_RECAPTION]], [{"ar_task_mode": "comprehension"}, None]
        )
        batch.idx_mapping = torch.tensor([1, 0])
        logits = sampler(torch.ones(2, 300), batch)
        assert logits[0].argmax().item() == ANSWER
        assert logits[1, 50].item() == 1

        sampler.add_request(0, SamplingParams())
        logits = sampler(torch.ones(2, 300), batch)
        assert logits[0, ANSWER].item() == 0
        assert logits[1, RECAPTION].item() == 0
        assert logits[:, 50].tolist() == [torch.finfo(logits.dtype).min] * 2

    def test_chunked_prefill_ignores_prompt_triggers(self, mocker):
        model = FakeSamplerModel()
        sampler, batch = self._make_sampler(mocker, model, [[], []])
        batch.seq_lens = torch.tensor([1, 2])
        logits = torch.randn(2, 300)
        torch.testing.assert_close(sampler(logits.clone(), batch), logits)

    def test_device_history_after_resume(self, mocker):
        from vllm.sampling_params import SamplingParams

        model = FakeSamplerModel()
        sampler, batch = self._make_sampler(mocker, model, [[END_OF_RECAPTION, ANSWER, BOI]])
        # Re-registration after preemption reconstructs from MRv2 history.
        sampler.add_request(0, SamplingParams())
        logits = sampler(torch.ones(1, 300), batch)
        assert logits.argmax().item() == SIZE_TOKEN

    def test_sampling_does_not_read_device_scalars(self, mocker, monkeypatch):
        model = FakeSamplerModel()
        sampler, batch = self._make_sampler(mocker, model, [[SIZE_TOKEN], [END_OF_THINK]])

        def fail_read(*args, **kwargs):
            raise AssertionError("MRv2 constraints must keep token history on device")

        monkeypatch.setattr(torch.Tensor, "item", fail_read)
        monkeypatch.setattr(torch.Tensor, "tolist", fail_read)
        monkeypatch.setattr(torch.Tensor, "cpu", fail_read)
        sampler(torch.ones(2, 300), batch)

    def test_model_registers_sampler(self, mocker):
        model = FakeSamplerModel()
        sampler, batch = self._make_sampler(mocker, model, [[END_OF_THINK]])
        # #10668: the raw model can prefer another </think>; the registered
        # MRv2 path must force <recaption> before the AR-to-DiT bridge.
        logits = torch.zeros(1, 300)
        logits[0, END_OF_THINK] = 10
        assert sampler(logits, batch).argmax().item() == RECAPTION

    def test_speculative_decoding_rejected(self, mocker):
        model = FakeSamplerModel()
        model.vllm_config = mocker.Mock(speculative_config=object())
        with pytest.raises(NotImplementedError, match="speculative decoding"):
            model.mrv2_custom_sampler(mocker.Mock())
