# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig
from vllm.config import CUDAGraphMode, SpeculativeConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionBackend
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import AttentionGroup

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.v1_compat import gpu_model_runner as v1_runner
from vllm_omni.worker.v1_compat.extract_hidden_states import ExtractHiddenStatesProposer
from vllm_omni.worker.v1_compat.gemma4 import Gemma4Proposer
from vllm_omni.worker.v1_compat.input_batch import InputBatch
from vllm_omni.worker.v1_compat.llm_base_proposer import SpecDecodeBaseProposer
from vllm_omni.worker.v1_compat.ngram_proposer import NgramProposer
from vllm_omni.worker.v1_compat.sampler import Sampler
from vllm_omni.worker.v1_compat.spec_decode_utils import compute_new_slot_mapping
from vllm_omni.worker.v1_compat.step3p5 import Step3p5MTPProposer
from vllm_omni.worker.v1_compat.suffix_decoding import SuffixDecodingProposer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_legacy_constructor_dispatches_real_cpu_ngram_proposals(monkeypatch):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    spec = SpeculativeConfig(method="ngram", num_speculative_tokens=2, prompt_lookup_min=1, prompt_lookup_max=2)
    runner = object.__new__(OmniGPUModelRunner)
    runner.vllm_config = SimpleNamespace(
        speculative_config=spec,
        model_config=SimpleNamespace(max_model_len=32),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    runner.speculative_config = spec
    runner.device = torch.device("cpu")
    runner.sampler = Sampler(logprobs_mode="raw_logprobs")
    runner.max_num_reqs = 2
    runner.max_model_len = 32
    monkeypatch.setattr(v1_runner, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))

    v1_runner.V1GPUModelRunnerMixin._init_speculative_decoding(runner)

    assert isinstance(runner.drafter, NgramProposer)
    history = np.zeros((2, 32), dtype=np.int32)
    history[0, :7] = [1, 2, 3, 1, 2, 3, 1]
    history[1, :4] = [8, 9, 8, 9]
    runner.input_batch = SimpleNamespace(num_tokens_no_spec=np.array([7, 4]), token_ids_cpu=history)
    output = SchedulerOutput.make_empty()
    output.total_num_scheduled_tokens = 7
    output.num_spec_tokens_to_schedule = 2
    empty = torch.empty(0)
    assert runner.propose_draft_token_ids(output, [[1], []], None, empty, empty, None, None, None, None) == [
        [2, 3],
        [],
    ]


def test_step3p5_draft_groups_keep_distinct_padded_slot_mappings():
    proposer = object.__new__(Step3p5MTPProposer)
    proposer.kv_cache_gid = 0
    proposer.max_positions = 8
    proposer.device = torch.device("cpu")
    proposer._slot_mapping_buffer = torch.zeros(8, dtype=torch.int64)
    proposer._per_group_block_tables = {}
    proposer._per_group_slot_mappings = {}
    proposer._per_group_slot_mapping_buffers = {}
    proposer.draft_attn_groups = [
        SimpleNamespace(kv_cache_group_id=0, layer_names=["draft.attn.0"]),
        SimpleNamespace(kv_cache_group_id=1, layer_names=["draft.attn.1"]),
    ]
    proposer.set_per_group_attn_metadata(0, torch.tensor([[1]]), torch.tensor([11, 12, 13]))
    proposer.set_per_group_attn_metadata(1, torch.tensor([[2]]), torch.tensor([21, 22, 23]))

    mappings = proposer._get_slot_mapping(5)

    assert mappings["draft.attn.0"].tolist() == [11, 12, 13, -1, -1]
    assert mappings["draft.attn.1"].tolist() == [21, 22, 23, -1, -1]
    assert mappings["draft.attn.0"].data_ptr() != mappings["draft.attn.1"].data_ptr()
    proposer.set_per_group_attn_metadata(1, torch.tensor([[3]]), torch.tensor([31, 32]))
    updated = proposer._get_slot_mapping(4)
    assert updated["draft.attn.1"].tolist() == [31, 32, -1, -1]
    assert updated["draft.attn.0"].tolist() == [11, 12, 13, -1]


def test_real_suffix_backend_accepts_legacy_batch_and_retires_finished_requests():
    pytest.importorskip(
        "arctic_inference.suffix_decoding",
        reason="Real suffix decoding requires the baseline optional ArcticInference dependency",
    )
    config = SimpleNamespace(
        speculative_config=SpeculativeConfig(method="suffix", num_speculative_tokens=2),
        model_config=SimpleNamespace(max_model_len=32),
    )
    proposer = SuffixDecodingProposer(config)
    batch = object.__new__(InputBatch)
    batch._req_ids = ["request"]
    batch.req_id_to_index = {"request": 0}
    batch.num_prompt_tokens = np.array([7], dtype=np.int32)
    batch.num_tokens_no_spec = np.array([7], dtype=np.int32)
    batch.token_ids_cpu = np.array([[1, 2, 3, 1, 2, 3, 1]], dtype=np.int32)

    drafts = proposer.propose(2, batch, [[2]])

    assert len(drafts) == 1 and len(drafts[0]) <= 2
    assert "request" in proposer.suffix_cache.active_requests
    batch._req_ids = []
    batch.req_id_to_index = {}
    assert proposer.propose(2, batch, []) == []
    assert "request" not in proposer.suffix_cache.active_requests


@pytest.mark.parametrize(
    ("query_starts", "host_query_starts", "new_tokens", "positions", "rejected", "expected"),
    [
        ([0, 2, 3], [0, 1, 3], 0, [0, 3, 4], [False] * 3, [40, 43, 84]),
        (
            [0, 2, 3],
            [0, 1, 3],
            2,
            [0, 3, 4, 5, 3, 4, 8],
            [False, False, False, True, False, False, False],
            [40, 43, 44, -1, 83, 84, -1],
        ),
        ([0, 2, 2, 3], [0, 2, 2, 3], 0, [0, 3, 4], [False] * 3, [40, 43, 124]),
    ],
    ids=["device-boundaries", "expanded-rejected-and-limit", "zero-query-padding"],
)
def test_parallel_draft_slots_follow_real_attention_request_boundaries(
    query_starts, host_query_starts, new_tokens, positions, rejected, expected
):
    num_reqs = len(query_starts) - 1
    metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor(query_starts, dtype=torch.int32),
        # Adaptive verification can retain different per-request CPU bounds.
        query_start_loc_cpu=torch.tensor(host_query_starts, dtype=torch.int32),
        seq_lens=torch.full((num_reqs,), 8, dtype=torch.int32),
        num_reqs=num_reqs,
        num_actual_tokens=query_starts[-1],
        max_query_len=max(end - start for start, end in zip(query_starts, query_starts[1:])),
        max_seq_len=8,
        block_table_tensor=torch.tensor([[10 * (row + 1), 10 * (row + 1) + 1] for row in range(num_reqs)]),
        slot_mapping=torch.empty(query_starts[-1], dtype=torch.int64),
    )

    slots = compute_new_slot_mapping(
        metadata,
        torch.tensor(positions),
        torch.tensor(rejected),
        block_size=4,
        num_new_tokens=new_tokens,
        max_model_len=8,
    )

    assert slots.tolist() == expected


def _draft_attention_metadata():
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.int32),
        seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=3,
        max_query_len=2,
        max_seq_len=5,
        block_table_tensor=torch.tensor([[1, 3], [2, 4]], dtype=torch.int32),
        slot_mapping=torch.tensor([11, 12, 13]),
    )


@pytest.mark.parametrize("proposer_type", [SpecDecodeBaseProposer, Gemma4Proposer, Step3p5MTPProposer])
@pytest.mark.parametrize("packed", [False, True])
def test_draft_attention_groups_keep_native_mapping_fast_build_and_layer_ownership(proposer_type, packed, monkeypatch):
    proposer = object.__new__(proposer_type)
    proposer.draft_attn_groups = []
    proposer._per_group_block_tables = {1: torch.tensor([[7, 8], [9, 10], [99, 99]], dtype=torch.int32)}
    proposer._per_group_slot_mappings = {1: torch.tensor([21, 22, 23, 999])}
    metadata = _draft_attention_metadata()
    original_table = metadata.block_table_tensor.clone()
    seen = {}
    for gid in (0, 1):
        group = AttentionGroup(
            CacheOnlyAttentionBackend,
            [f"group{gid}.a", f"group{gid}.b"],
            FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=16, dtype=torch.float32),
            gid,
        )
        group.create_metadata_builders(VllmConfig(), torch.device("cpu"))
        builder = group.get_metadata_builder()

        def capture_build(common_prefix_len, common_attn_metadata, fast_build=False, _native=builder.build, _gid=gid):
            seen[_gid] = (common_prefix_len, common_attn_metadata, fast_build)
            return _native(common_prefix_len, common_attn_metadata, fast_build)

        monkeypatch.setattr(builder, "build", capture_build)
        if packed:
            group.kernel_block_size = 2
            group.kernel_block_stride = 4
            group.kernel_block_offsets = torch.arange(2, dtype=torch.int32)
            group.kernel_block_table = torch.empty((1, 2, 4), dtype=torch.int32)
        proposer.draft_attn_groups.append(group)

    per_group, per_layer = proposer.build_per_group_and_layer_attn_metadata(metadata, draft_index=1)

    for gid in (0, 1):
        prefix, common, fast = seen[gid]
        assert prefix == 0 and fast is True
        expected_table = (
            proposer._per_group_block_tables[gid][:2]
            if gid == 1 and proposer_type is not SpecDecodeBaseProposer
            else original_table
        )
        if packed:
            expected_table = (expected_table.unsqueeze(-1) * 4 + torch.arange(2, dtype=expected_table.dtype)).flatten(1)
        torch.testing.assert_close(common.block_table_tensor, expected_table)
        expected_slots = (
            proposer._per_group_slot_mappings[gid][:3]
            if gid == 1 and proposer_type is Step3p5MTPProposer
            else metadata.slot_mapping
        )
        assert per_group[gid].slot_mapping.data_ptr() == expected_slots.data_ptr()
        assert per_layer[f"group{gid}.a"] is per_layer[f"group{gid}.b"] is per_group[gid]
    torch.testing.assert_close(metadata.block_table_tensor, original_table)


def test_aiter_draft_metadata_keeps_capture_safe_decode_and_native_scale_identity(monkeypatch):
    from vllm.v1.attention.backends.rocm_aiter_fa import (
        AiterFlashAttentionBackend,
        AiterFlashAttentionMetadataBuilder,
    )

    from vllm_omni.worker.v1_compat import spec_decode_utils

    # Exercise the native data classes and preserved algorithm on CPU;
    # this does not claim an actual ROCm kernel/device run.
    monkeypatch.setattr(spec_decode_utils, "current_platform", SimpleNamespace(is_rocm=lambda: True))
    builder = object.__new__(AiterFlashAttentionMetadataBuilder)
    builder.k_scale = {"layer": torch.ones(1)}
    builder.v_scale = {"layer": torch.ones(1)}
    metadata = _draft_attention_metadata()
    group = AttentionGroup(
        AiterFlashAttentionBackend,
        ["layer"],
        FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=16, dtype=torch.float32),
        0,
        metadata_builders=[builder],
    )
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.draft_attn_groups = [group]

    per_group, per_layer = proposer.build_per_group_and_layer_attn_metadata(metadata)

    result = per_group[0]
    assert per_layer["layer"] is result
    assert result.num_decodes == 2 and result.num_decode_tokens == 3
    assert result.num_prefills == result.num_extends == result.num_extend_tokens == 0
    assert result.prefill_metadata is result.extend_metadata is None
    assert result.decode_metadata.max_query_len == 2
    assert result.query_start_loc is metadata.query_start_loc
    assert result.block_table is metadata.block_table_tensor
    assert result.slot_mapping is metadata.slot_mapping
    assert result.k_scale is builder.k_scale and result.v_scale is builder.v_scale


def test_cache_only_extraction_uses_real_native_builder_and_keeps_sampled_column(monkeypatch):
    observed = {}

    class CacheSink(torch.nn.Module):
        def forward(self, hidden_states):
            observed["hidden"] = hidden_states.clone()
            observed["metadata"] = get_forward_context().attn_metadata["layer"]

    proposer = object.__new__(ExtractHiddenStatesProposer)
    proposer.num_speculative_tokens = 1
    proposer.model = CacheSink()
    proposer.hidden_states = torch.empty((3, 2, 4))
    proposer.attn_layer_names = ["layer"]
    proposer.vllm_config = VllmConfig()
    proposer.eplb_state = None
    proposer._slot_mapping_buffer = torch.empty(3, dtype=torch.int64)
    metadata = _draft_attention_metadata()
    group = AttentionGroup(
        CacheOnlyAttentionBackend,
        ["layer"],
        FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=16, dtype=torch.float32),
        0,
    )
    group.create_metadata_builders(proposer.vllm_config, torch.device("cpu"))
    proposer.attn_metadata_builder = group.get_metadata_builder()
    monkeypatch.setattr(
        proposer, "_determine_batch_execution_and_padding", lambda count: (CUDAGraphMode.NONE, count, None)
    )
    hidden = [torch.arange(12).reshape(3, 4).float(), torch.ones((3, 4))]
    sampled = torch.tensor([[7, 8], [9, 10]])

    result = proposer.propose(1, sampled, hidden, metadata)

    assert result.tolist() == [[7], [9]]
    torch.testing.assert_close(observed["hidden"], torch.stack(hidden, dim=1))
    assert observed["metadata"].slot_mapping is metadata.slot_mapping


@pytest.mark.parametrize("pass_hidden_states", [False, True])
@pytest.mark.parametrize(
    ("config_fields", "expected"),
    [
        ({"pard_token": 41}, 41),
        ({"ptd_token_id": 42}, 42),
        ({"pard_token": 0, "ptd_token_id": 42}, 0),
        ({"dspark_noise_token_id": 7, "pard_token": 41, "ptd_token_id": 42}, 7),
        ({"mask_token_id": 0, "dspark_noise_token_id": 7, "pard_token": 41, "ptd_token_id": 42}, 0),
        (
            {
                "dflash_config": {"mask_token_id": 0},
                "mask_token_id": 5,
                "dspark_noise_token_id": 7,
                "pard_token": 41,
                "ptd_token_id": 42,
            },
            0,
        ),
    ],
    ids=[
        "parallel-token",
        "parallel-token-id",
        "parallel-priority",
        "noise-priority",
        "mask-priority",
        "dflash-priority",
    ],
)
def test_parallel_drafting_config_token_fields(config_fields, expected, pass_hidden_states):
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.draft_model_config = SimpleNamespace(hf_config=PretrainedConfig(**config_fields))
    proposer.pass_hidden_states_to_model = pass_hidden_states
    proposer.parallel_drafting_hidden_state_tensor = None
    proposer.hidden_size = 4
    proposer.dtype = torch.float32
    proposer.device = torch.device("cpu")

    proposer._init_parallel_drafting_params()

    assert proposer.parallel_drafting_token_id == expected
    if pass_hidden_states:
        hidden = proposer.parallel_drafting_hidden_state_tensor
        assert isinstance(hidden, torch.Tensor)
        assert hidden.shape == (4,)
        assert hidden.dtype == torch.float32
        assert hidden.device == torch.device("cpu")
    else:
        assert proposer.parallel_drafting_hidden_state_tensor is None


def test_parallel_drafting_config_error_names_supported_fields():
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.draft_model_config = SimpleNamespace(hf_config=PretrainedConfig())
    proposer.pass_hidden_states_to_model = False

    with pytest.raises(ValueError, match="pard_token.*ptd_token_id"):
        proposer._init_parallel_drafting_params()
