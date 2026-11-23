# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.model_executor.models.phonon import model_state as state_module
from vllm_omni.model_executor.models.phonon.phonon_talker import PhononTalkerForConditionalGeneration
from vllm_omni.transformers_utils.configs.phonon import PhononConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def sampling(monkeypatch):
    state = state_module.PhononModelState.__new__(state_module.PhononModelState)
    model = PhononTalkerForConditionalGeneration.__new__(PhononTalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = PhononConfig(eos_token_id=7, audio_continue_token_id=4)
    model.continue_token_id, model.eos_token_id = 4, 7
    model._continue = torch.tensor([True, False])
    state.model = model
    batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([2, 0]),
        idx_mapping=torch.tensor([2, 0]),
        seq_lens=torch.tensor([4, 3]),
        cu_num_logits=torch.tensor([0, 1, 2]),
    )
    sampler = Sampler.__new__(Sampler)
    sampler.compute_nans = False
    sampler.return_sampling_mask = False
    sampler.trace_replay_state = None
    sampler.get_logprobs_dims = lambda rows: None
    sampler.thinking_budget_state = SimpleNamespace(enabled=False, use_thinking_budget=np.zeros(3, dtype=bool))
    sampler.logits_processors = []
    sampler.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=torch.tensor([3, 9, 6])))

    def counts(num_sampled, seq_lens, cu_num_logits, slots, prefill):
        assert slots is batch.idx_mapping
        assert seq_lens is batch.seq_lens
        assert cu_num_logits is batch.cu_num_logits
        ready = seq_lens >= prefill[slots]
        return num_sampled * ready, torch.where(ready, cu_num_logits.diff() - num_sampled, 0)

    monkeypatch.setattr(state_module, "get_num_sampled_and_rejected", counts)
    return state, batch, sampler


def test_configured_tokens_prefill_counts_and_output_ownership(sampling):
    state, batch, sampler = sampling
    output = state.sample_determined_tokens(batch, sampler)
    assert output.sampled_token_ids.tolist() == [[4], [7]]
    assert output.num_sampled.tolist() == [0, 1]
    assert output.num_rejected.tolist() == [0, 0]
    assert output.logprobs_tensors is None
    assert output.sampling_mask_tensors is None
    logits = state.model.compute_logits(torch.zeros(2, 8))
    torch.testing.assert_close(output.sampled_token_ids.flatten(), logits.argmax(-1))
    state.model._continue.zero_()
    assert output.sampled_token_ids.tolist() == [[4], [7]]
    batch.seq_lens[0] = 6
    assert state.sample_determined_tokens(batch, sampler).num_sampled.tolist() == [1, 1]


@pytest.mark.parametrize(
    "feature",
    [
        "compute_nans",
        "return_sampling_mask",
        "trace_replay_state",
        "logprobs",
        "thinking",
        "custom",
        "missing",
        "shape",
    ],
)
def test_incompatible_sampling_falls_back(sampling, feature):
    state, batch, sampler = sampling
    if feature in ("compute_nans", "return_sampling_mask"):
        setattr(sampler, feature, True)
    elif feature == "trace_replay_state":
        sampler.trace_replay_state = object()
    elif feature == "logprobs":
        sampler.get_logprobs_dims = lambda rows: (1, 0)
    elif feature == "thinking":
        sampler.thinking_budget_state.enabled = True
        sampler.thinking_budget_state.use_thinking_budget[2] = True
    elif feature == "custom":
        sampler.logits_processors = [object()]
    elif feature == "missing":
        state.model._continue = None
    else:
        state.model._continue = torch.ones(1, dtype=torch.bool)
    assert state.sample_determined_tokens(batch, sampler) is None


@pytest.mark.parametrize(
    "processor_type,attribute",
    [(LogitBiasState, "use_logit_bias"), (PenaltiesState, "use_penalty"), (BadWordsState, "num_bad_words")],
)
def test_builtin_processors_fall_back_only_for_active_requests(sampling, processor_type, attribute):
    state, batch, sampler = sampling
    processor = processor_type.__new__(processor_type)
    flags = np.zeros(3, dtype=bool)
    setattr(processor, attribute, SimpleNamespace(np=flags) if processor_type is BadWordsState else flags)
    sampler.logits_processors = [processor]
    flags[1] = True
    assert state.sample_determined_tokens(batch, sampler) is not None
    flags[2] = True
    assert state.sample_determined_tokens(batch, sampler) is None


def test_eos_protocol_config_defaults_and_validation():
    assert PhononConfig(eos_token_id=None).eos_token_id == 1
    assert PhononConfig(eos_token_id=[7]).eos_token_id == 7
    with pytest.raises(ValueError, match="distinct"):
        PhononConfig(eos_token_id=0)


def test_tail_countdown_and_reused_slot_return_to_continue(sampling):
    state, batch, sampler = sampling
    state.remaining_tail = torch.full((3,), -1, dtype=torch.long)
    state.tail_frames = torch.tensor([3, 3, 1])
    state.active = torch.ones(3, dtype=torch.bool)
    state.next_embeddings = torch.zeros(3, 8)
    state.condition = torch.zeros(8)
    state.model.phonon_config = {"eos_threshold": 0.5}
    state.model.flow_lm = SimpleNamespace(
        sample=lambda hidden, noise: (noise, torch.ones(2, 1)),
        input_linear=torch.nn.Linear(4, 8, bias=False),
        emb_std=torch.ones(4),
        emb_mean=torch.zeros(4),
    )
    with torch.inference_mode():
        for expected in ([4, 4], [7, 4], [7, 4], [7, 7]):
            _, _, _, keep = state._flow_postprocess(torch.zeros(2, 8), torch.zeros(2, 4), batch.idx_mapping)
            state.model._continue = keep
            output = state.sample_determined_tokens(batch, sampler)
            assert output.sampled_token_ids.flatten().tolist() == expected
        state.remaining_tail[2].fill_(-1)
        state.active[2].fill_(True)
        _, _, _, state.model._continue = state._flow_postprocess(
            torch.zeros(2, 8),
            torch.zeros(2, 4),
            batch.idx_mapping,
        )
        assert state.sample_determined_tokens(batch, sampler).sampled_token_ids[0, 0] == 4


def test_sampler_subclass_uses_standard_path(sampling):
    state, batch, sampler = sampling

    class OtherSampler(Sampler):
        pass

    sampler.__class__ = OtherSampler
    assert state.sample_determined_tokens(batch, sampler) is None


@pytest.mark.parametrize("eos", [None, 7])
def test_pipeline_enforces_configured_stop_without_tokenizer(eos):
    from vllm_omni.config.stage_config import merge_sampling_constraints
    from vllm_omni.model_executor.models.phonon.pipeline import resolve_phonon_pipeline

    config = PhononConfig(eos_token_id=eos)
    stage = resolve_phonon_pipeline(config).stages[0]
    merged = merge_sampling_constraints({"detokenize": False}, stage.sampling_constraints)
    assert config.eos_token_id in merged["stop_token_ids"]
