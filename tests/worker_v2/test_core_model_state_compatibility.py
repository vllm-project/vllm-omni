# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Core ModelState compatibility through the production MRv2 AR runner."""

from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.config import LoadConfig, VllmConfig
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.worker.gpu import model_runner as core_module
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.core.sched.omni_scheduling_coordinator import uses_native_mrv2_data_plane
from vllm_omni.core.sched.output import OmniCachedRequestData
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2 import omni_ar_model_runner as ar_module
from vllm_omni.worker_v2 import omni_model_runner as base_module
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.model_states.omni_mamba_hybrid import OmniMambaHybridModelState
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner
from vllm_omni.worker_v2.staged_writes import StagedWrites

Base = base_module.OmniGPUModelRunner
AR = ar_module.OmniARModelRunner
CoreRunner = core_module.GPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FixtureOmniConfig(OmniModelConfig):
    is_hybrid = False
    is_attention_free = False
    uses_mrope = False

    def get_inputs_embeds_size(self):
        return 2


def make_config(*, native=False, kv=None, eager=True):
    mc = object.__new__(FixtureOmniConfig)
    mc.__dict__.update(
        dtype=torch.float32,
        max_model_len=32,
        enable_prompt_embeds=False,
        async_chunk=False,
        engine_output_type="text",
        logits_processors=None,
        logprobs_mode="raw_logprobs",
        use_fp64_gumbel=False,
        enable_trace_replay=False,
        return_sampling_mask=False,
        supports_native_mrv2_data_plane=native,
        enforce_eager=eager,
        omni_kv_config=kv,
        architectures=["DiagnosticModel"],
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size=1,
        enable_batch_sharded_sampling=False,
    )
    return SimpleNamespace(
        model_config=mc,
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
        cache_config=SimpleNamespace(mamba_cache_mode="none", mamba_block_size=16, use_kda_recoverssm=False),
        parallel_config=parallel,
        load_config=LoadConfig(load_format="dummy"),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE),
        additional_config={},
        speculative_config=None,
        watermark_config=None,
        kv_transfer_config=None,
        use_flashinfer_replayssm=False,
    )


class CoreState(ModelState):
    """Legal core state: no Omni-only hooks or buffer."""

    def __init__(self, config, model, encoder_cache, device):
        super().__init__(config, model, encoder_cache, device)
        self.events = model.events
        self.live = {}
        self.events.append(("state_created", type(self).__name__))
        model.created_states.append(self)

    def add_request(self, slot, data):
        self.live[data.req_id] = slot
        self.events.append(("state_add", data.req_id, slot))

    def remove_request(self, req_id: str):
        if not isinstance(req_id, str):
            raise TypeError(f"expected string request ID; got {type(req_id).__name__}")
        self.events.append(("state_remove", req_id))
        self.live.pop(req_id, None)

    def prepare_inputs_embeds(self, *args):
        return None

    def prepare_inputs(self, *args):
        self.events.append(("prepare_inputs",))
        return {}

    def prepare_dummy_inputs(self, *args):
        return {}

    def prepare_attn(self, *args, **kwargs):
        self.events.append(("prepare_attn",))
        return {}

    def preprocess_state(self, *args):
        self.events.append(("preprocess_state",))

    def postprocess_state(self, *args):
        self.events.append(("postprocess_state",))

    def apply_staged_writes(self):
        self.events.append(("state_apply",))


class CapabilityControlState(CoreState):
    """Explicit positive capability fixture, not an existing Omni model state."""

    def __init__(self, *args):
        super().__init__(*args)
        self.intermediate_buffer = OmniIntermediateBuffer(self.max_num_reqs)

    def _get_mtp_capture_sizes(self):
        return [1]


class TinyModel(torch.nn.Module):
    def __init__(self, state_cls=CoreState, *, capture=False):
        super().__init__()
        self.events: list[tuple[Any, ...]] = []
        self.created_states = []
        self.state_cls = state_cls
        self.weight = torch.nn.Parameter(torch.zeros(1))
        if capture:
            self.capture_stream_decode_graphs = lambda sizes: self.events.append(("capture", sizes))

    def get_model_state_cls(self):
        self.events.append(("get_model_state_cls",))
        return self.state_cls

    def forward(
        self,
        input_ids,
        positions,
        inputs_embeds=None,
        intermediate_tensors=None,
        sampling_metadata=None,
        logits_index=None,
        sampler=None,
    ):
        # Accept the existing three legacy inputs explicitly: R01 stays out of scope.
        self.events.append(
            (
                "forward",
                input_ids.tolist(),
                sampling_metadata is not None,
                logits_index.tolist() if isinstance(logits_index, torch.Tensor) else logits_index,
            )
        )
        return torch.stack([input_ids.float(), input_ids.float() + 0.5], dim=-1)

    def compute_logits(self, hidden):
        return hidden

    def on_requests_finished(self, req_ids):
        self.events.append(("model_finished", sorted(req_ids)))


class CPUField:
    def __init__(self, shape):
        self.np = np.zeros(shape, dtype=np.int32)
        self.gpu = torch.from_numpy(self.np)

    def stage_write_elem(self, row, value):
        self.np[row] = value

    def stage_write(self, row, start, values):
        self.np[row, start : start + len(values)] = values

    def copy_to_uva(self):
        pass

    def apply_write(self):
        pass


def request_state():
    obj = object.__new__(RequestState)
    obj.req_id_to_index, obj.index_to_req_id, obj.free_indices = {}, {}, [0, 1]
    obj.max_num_reqs, obj.max_model_len, obj.max_num_batched_tokens = 2, 32, 16
    obj.num_speculative_steps = 0
    obj.max_seq_len = np.zeros(2, dtype=np.int32)
    for name in ("prompt_len", "prefill_len", "total_len", "num_computed_tokens"):
        setattr(obj, name, CPUField(2))
    obj.all_token_ids = CPUField((2, 32))
    obj.num_computed_prefill_tokens = np.zeros(2, dtype=np.int32)
    obj.num_computed_tokens_np = np.zeros(2, dtype=np.int32)
    obj.draft_tokens = torch.zeros(2, 0, dtype=torch.int64)
    obj.last_sampled_tokens = torch.zeros(2, 1, dtype=torch.int64)
    return obj


class CPUCopy:
    """Only replaces asynchronous output copies, not AR sampling control flow."""

    def __init__(self, *, model_runner_output, sampler_output, **kwargs):
        self.copy_event = None
        self.output = model_runner_output
        self.output.sampled_token_ids = sampler_output.sampled_token_ids.tolist()

    def get_output(self):
        return self.output


class PayloadPlane:
    """Observable connector I/O boundary for capability tests."""

    def __init__(self):
        self.payload = {"A": {"sentinel": "must-not-be-consumed"}}
        self.popped = []

    def pop_local_stage_payload(self, req_id):
        self.popped.append(req_id)
        return self.payload.pop(req_id, None)


def loaded_runner(monkeypatch, *, config=None, model=None, runner_cls=AR):
    """Run the actual Omni load_model and actual core load_model + both factories.

    GPU sampler allocation/logit-worker construction are CPU boundaries. The
    parent load method, model-state creation, config routing and assignment
    remain real; no ready-made state is manually injected.
    """
    cfg = config or make_config()
    model = model or TinyModel()
    runner = object.__new__(runner_cls)
    runner.vllm_config, runner.model_config, runner.scheduler_config = cfg, cfg.model_config, cfg.scheduler_config
    runner.parallel_config, runner.compilation_config = cfg.parallel_config, cfg.compilation_config
    runner.cache_config, runner.load_config, runner.speculative_config = cfg.cache_config, cfg.load_config, None
    runner.device = torch.device("cpu")
    runner.max_num_reqs, runner.max_num_tokens, runner.vocab_size = 2, 16, 32
    runner.req_states = request_state()
    runner.encoder_cache = runner.lora_config = runner.speculator = runner.pp_handler = None
    runner.pooling_runner = runner.prompt_logprobs_worker = runner.rejection_sampler = None
    runner.is_first_pp_rank = runner.is_last_pp_rank = True
    runner.is_pooling_model = runner.is_encoder_decoder = False
    runner.use_aux_hidden_state_outputs = False
    runner.num_speculative_steps = 0
    runner.batch_sharder = None
    runner._staged_writes = StagedWrites()
    runner.jit_warmup_registry = JitWarmupRegistry(cfg)
    runner._omni_data_plane = (
        PayloadPlane() if uses_native_mrv2_data_plane(cfg.model_config, use_v2_model_runner=True) else None
    )
    runner._native_output_materializer = None
    runner._kv_extracted_req_ids = None
    runner.kv_transfer_manager = None
    runner.eplb = SimpleNamespace(
        prepare_load=lambda: model.events.append(("prepare_load",)),
        maybe_register_model=lambda *a: False,
        maybe_start_async_loop=lambda *a: None,
        prepare_forward=lambda *a: model.events.append(("prepare_forward",)),
        step=lambda **kw: None,
    )
    runner.lora_state = SimpleNamespace(remove_request=lambda req_id: model.events.append(("lora_remove", req_id)))
    runner.draft_tokens_handler = SimpleNamespace(discard=lambda req_id: model.events.append(("draft_discard", req_id)))
    runner.aux_output_connector = None
    runner.supports_mm_inputs = False
    runner.pcp_manager = None
    runner.omni_prefix_cache = runner._omni_prefix_cache_cfg = None
    runner._prefix_cache_group_view = runner._prefix_cache_adapter = None
    runner.execute_model_state = None
    runner.check_ep_fault = False
    runner._last_multimodal_snapshot_slot = None

    class CPUSampler:
        def __init__(self, **kwargs):
            model.events.append(("sampler_constructed",))
            self.penalties_state = SimpleNamespace(output_bin_counts=None)

    class CPULogprobs:
        def __init__(self, *args, **kwargs):
            pass

        def compute_prompt_logprobs(self, *args):
            return {}

        def compute_prompt_token_id_logprobs(self, *args):
            return {}

        def remove_request(self, req_id):
            pass

    monkeypatch.setattr(core_module, "DeviceMemoryProfiler", lambda: nullcontext(SimpleNamespace(consumed_memory=0)))
    monkeypatch.setattr(core_module, "Sampler", CPUSampler)
    monkeypatch.setattr(core_module, "PromptLogprobsWorker", CPULogprobs)
    monkeypatch.setattr(core_module, "StructuredOutputsWorker", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(core_module, "build_custom_logits_processors", lambda *a: [])
    model.loading_runner = runner
    runner.load_model(model=model)
    assert runner.model_state.model is model
    if model.created_states:
        assert runner.model_state is model.created_states[-1]
    return runner, model


def configure_step(monkeypatch, runner, model, mode=CUDAGraphMode.NONE):
    for req_id in ("B", "A"):
        runner.req_states.add_request(req_id, 2, [2, 5], 0, 3)
        runner.model_state.add_request(
            runner.req_states.req_id_to_index[req_id],
            SimpleNamespace(req_id=req_id, mm_features=[], num_computed_tokens=0),
        )
    runner.req_states.apply_staged_writes = lambda: model.events.append(("request_apply",))
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[], num_blocks=2)
    runner.attn_groups = []
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None, append_block_ids=lambda *a, **kw: None)
    runner.kv_connector = SimpleNamespace(pre_forward=lambda *a: None, post_forward=lambda ids: None)
    runner.ec_connector = SimpleNamespace(no_forward=lambda output: EMPTY_MODEL_RUNNER_OUTPUT)
    batch = SimpleNamespace(
        req_ids=["A"],
        num_reqs=1,
        num_reqs_after_padding=1,
        input_ids=torch.tensor([2, 5]),
        positions=torch.tensor([0, 1]),
        num_tokens=2,
        num_tokens_after_padding=2,
        is_padding=False,
        idx_mapping_np=np.array([0]),
        idx_mapping=torch.tensor([0]),
        num_scheduled_tokens=np.array([2]),
        prefill_len_np=np.array([2]),
        num_computed_prefill_tokens_np=np.array([0]),
        query_start_loc_np=np.array([0, 2]),
        query_start_loc=torch.tensor([0, 2]),
        seq_lens=torch.tensor([2]),
        num_draft_tokens=0,
        logits_indices=torch.tensor([1]),
        max_query_len=2,
        seq_lens_cpu_upper_bound=torch.tensor([2]),
        pcp=None,
        dcp_local_seq_lens=None,
        is_prefilling_np=np.array([True]),
        prompt_lens=torch.tensor([2]),
        fast_prefill=False,
        prefill_runs_as_decode_np=None,
    )
    desc = SimpleNamespace(cg_mode=mode, num_reqs=1, num_tokens=2, num_active_loras=0, max_query_len=2)
    monkeypatch.setattr(
        runner,
        "gather_batch_req_state",
        lambda *a: (SimpleNamespace(req_ids=["A"], num_tokens=2, has_prefill=False), None),
    )
    monkeypatch.setattr(runner, "_dispatch_batch_descriptor", lambda **kw: (desc, None))
    monkeypatch.setattr(runner, "prepare_inputs", lambda *a: batch)
    monkeypatch.setattr(runner, "prepare_attn", lambda *a: ((), torch.empty(0)))
    monkeypatch.setattr(runner, "_forward_context", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(base_module, "build_slot_mappings_by_layer", lambda *a: {})
    runner.cudagraph_manager = None

    def cpu_sample(self, hidden, batch, grammar):
        model.events.append(("sample", hidden.tolist()))
        model.sampled_hidden = hidden
        tokens = model.compute_logits(hidden)[batch.logits_indices].argmax(-1).reshape(-1, 1)
        count = torch.ones(batch.num_reqs, dtype=torch.int32)
        rejected = torch.zeros_like(count)
        return SamplerOutput(tokens, None, None, count, rejected), count, rejected

    def cpu_post_update(
        indices, computed, last_tokens, bins, tokens, counts, rejected, query, all_ids, lengths, drafts, state_drafts
    ):
        computed[indices] += query[1:] - query[:-1]
        last_tokens[indices] = tokens
        model.events.append(("post_update_kernel_boundary",))

    monkeypatch.setattr(CoreRunner, "sample", cpu_sample)
    monkeypatch.setattr(core_module, "post_update", cpu_post_update)
    monkeypatch.setattr(ar_module, "OmniAsyncOutput", CPUCopy)
    runner.main_stream = SimpleNamespace(wait_event=lambda event: None)
    runner.output_copy_stream = SimpleNamespace()
    runner._release_multimodal_snapshot = lambda *a: None
    return batch


def scheduled(*, finished=()):
    result = SchedulerOutput.make_empty()
    result.num_scheduled_tokens = {"A": 2}
    result.total_num_scheduled_tokens = 2
    result.finished_req_ids = set(finished)
    return result


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
def test_core_state_runs_ar_step_sampling_and_parent_teardown(monkeypatch, mode):
    runner, model = loaded_runner(monkeypatch)
    assert runner._omni_data_plane is None
    assert type(runner.model_state) is CoreState
    batch = configure_step(monkeypatch, runner, model, mode)
    runner.cudagraph_manager = SimpleNamespace(
        run_pw_graph=lambda model, inputs: model(**inputs),
        run_fullgraph=lambda descriptor: model(input_ids=batch.input_ids, positions=batch.positions),
    )

    assert runner.execute_model(scheduled()) is None
    hidden = runner.execute_model_state.hidden_states
    torch.testing.assert_close(hidden, torch.tensor([[2.0, 2.5], [5.0, 5.5]]))
    assert runner.sample_tokens(None).get_output().sampled_token_ids == [[1]]
    assert model.sampled_hidden is hidden
    assert runner.model_state is model.created_states[0]

    finish = scheduled(finished=("A",))
    runner.finish_requests(finish)
    assert runner.req_states.req_id_to_index == {"B": 1}
    assert runner.model_state.live == {"B": 1}
    assert runner.req_states.free_indices == [0]
    runner.req_states.add_request("C", 2, [2, 5], 0, 3)
    runner.model_state.add_request(0, SimpleNamespace(req_id="C"))
    assert runner.req_states.req_id_to_index == {"B": 1, "C": 0}
    assert runner.model_state.live == {"B": 1, "C": 0}

    tags = [event[0] for event in model.events]
    assert tags.index("get_model_state_cls") < tags.index("state_created") < tags.index("sampler_constructed")
    step_hooks = {"preprocess_state", "prepare_attn", "prepare_inputs", "forward", "sample", "postprocess_state"}
    assert [tag for tag in tags if tag in step_hooks] == [
        "preprocess_state",
        "prepare_attn",
        "prepare_inputs",
        "forward",
        "sample",
        "postprocess_state",
    ]
    assert tags.index("model_finished") < tags.index("state_remove")
    assert [event for event in model.events if event[0] == "state_remove"] == [("state_remove", "A")]


@pytest.mark.parametrize("hook", ["run_preprocess", "run_postprocess"])
@pytest.mark.parametrize("behavior", ["not-callable", "raises"])
def test_optional_execution_hook_errors_are_reported(monkeypatch, hook, behavior):
    runner, model = loaded_runner(monkeypatch)
    configure_step(monkeypatch, runner, model)

    def fail(*args):
        raise AttributeError("error inside the hook")

    if behavior == "not-callable":
        setattr(runner.model_state, hook, 33)
        with pytest.raises(TypeError, match=rf"CoreState\.{hook} must be callable"):
            runner.execute_model(scheduled())
    else:
        setattr(runner.model_state, hook, fail)
        with pytest.raises(AttributeError, match="error inside the hook"):
            runner.execute_model(scheduled())
    assert any(event[0] == "forward" for event in model.events) == (hook == "run_postprocess")
    assert runner.execute_model_state is None


@pytest.mark.parametrize("dummy_run", [False, True], ids=["empty-step", "dummy-step"])
def test_inactive_execution_does_not_call_optional_hooks(monkeypatch, dummy_run):
    runner, model = loaded_runner(monkeypatch)
    batch = configure_step(monkeypatch, runner, model)
    runner.model_state.run_preprocess = runner.model_state.run_postprocess = 33
    monkeypatch.setattr(runner, "_make_dummy_input_batch", lambda *args, **kwargs: batch)
    runner.kv_connector.no_forward = lambda output: EMPTY_MODEL_RUNNER_OUTPUT

    if dummy_run:
        runner.execute_model(scheduled(), dummy_run=True, skip_attn_for_dummy_run=True)
        assert any(event[0] == "forward" for event in model.events)
    else:
        assert runner.execute_model(SchedulerOutput.make_empty()) is EMPTY_MODEL_RUNNER_OUTPUT
        assert not any(event[0] == "forward" for event in model.events)


@pytest.mark.parametrize("state_cls", [CoreState, OmniModelState, OmniMambaHybridModelState])
@pytest.mark.parametrize(
    "finished,preempted",
    [(set(), set()), ({"A"}, set()), (set(), {"A"}), ({"A"}, {"A"}), ({"unknown"}, set())],
    ids=["no-requests", "finished", "preempted", "overlap", "unknown"],
)
def test_finish_requests_cleans_each_id_once(monkeypatch, mocker, state_cls, finished, preempted):
    runner, model = loaded_runner(monkeypatch, model=TinyModel(state_cls))
    configure_step(monkeypatch, runner, model)
    if isinstance(runner.model_state, OmniModelState):
        runner.model_state.intermediate_buffer.buffers[0]["old_chunk"] = torch.ones(1)
    state_remove = mocker.spy(runner.model_state, "remove_request")
    request_remove = mocker.spy(runner.req_states, "remove_request")
    output = scheduled(finished=finished)
    output.preempted_req_ids = preempted

    runner.finish_requests(output)

    ended = sorted(finished | preempted)
    assert [call.args for call in state_remove.call_args_list] == [(req_id,) for req_id in ended]
    assert [call.args for call in request_remove.call_args_list] == [(req_id,) for req_id in ended]
    released = "A" in ended
    expected = {"B": 1} if released else {"B": 1, "A": 0}
    assert runner.req_states.req_id_to_index == expected
    assert runner.req_states.free_indices == ([0] if released else [])
    if isinstance(runner.model_state, OmniModelState):
        assert runner.model_state.intermediate_buffer.req_id_to_index == expected
        assert bool(runner.model_state.intermediate_buffer.buffers[0]) == (not released)
    else:
        assert runner.model_state.live == expected
    assert model.events.count(("lora_remove", "A")) == int(released)
    if released:
        runner.req_states.add_request("C", 1, [7], 0, 1)
        slot = runner.req_states.req_id_to_index["C"]
        assert slot == 0
        runner.model_state.add_request(slot, SimpleNamespace(req_id="C", mm_features=[], num_computed_tokens=0))
        if isinstance(runner.model_state, OmniModelState):
            assert runner.model_state.intermediate_buffer.buffers[slot] == {"req_id": "C"}
        else:
            assert runner.model_state.live == {"B": 1, "C": 0}


def cached_chunk_output(prompt_token_ids):
    output = scheduled()
    req_ids = list(prompt_token_ids)
    count = len(req_ids)
    output.scheduled_cached_reqs = OmniCachedRequestData(
        req_ids=req_ids,
        new_token_ids=[[] for _ in req_ids],
        new_block_ids=[()] * count,
        num_computed_tokens=[0] * count,
        num_output_tokens=[0] * count,
        prompt_token_ids=prompt_token_ids,
        additional_information={},
    )
    return output


@pytest.mark.parametrize("mode", ["none", "align"])
def test_cached_chunk_reset_preserves_peer_and_hybrid_slot(monkeypatch, mocker, mode):
    config = make_config()
    config.cache_config.mamba_cache_mode = mode
    runner, model = loaded_runner(
        monkeypatch, config=config, model=TinyModel(OmniMambaHybridModelState), runner_cls=OmniGenerationModelRunner
    )
    configure_step(monkeypatch, runner, model)
    state = runner.model_state
    state.num_accepted_tokens_gpu.copy_(torch.tensor([9, 7], dtype=torch.int32))
    if mode == "align":
        state._mamba_state_idx_gpu.copy_(torch.tensor([5, 6], dtype=torch.int32))
    state.intermediate_buffer.buffers[0]["old"] = True
    state.intermediate_buffer.buffers[1]["peer"] = True
    state_remove = mocker.spy(state, "remove_request")
    state_add = mocker.spy(state, "add_request")
    state_flush = mocker.spy(state, "apply_staged_writes")
    request_flush = mocker.spy(runner.req_states, "apply_staged_writes")
    order = mocker.Mock()
    order.attach_mock(state_remove, "remove")
    order.attach_mock(state_add, "add")
    order.attach_mock(request_flush, "request_flush")
    order.attach_mock(state_flush, "state_flush")

    runner._handle_async_chunk_updates(cached_chunk_output({"A": None, "unknown": None}))
    assert order.mock_calls == []
    runner._handle_async_chunk_updates(cached_chunk_output({"A": []}))

    state_remove.assert_called_once_with("A")
    state_add.assert_called_once()
    assert state_add.call_args.args[0] == 0
    state_flush.assert_called_once_with()
    assert [call[0] for call in order.mock_calls] == ["remove", "add", "request_flush", "state_flush"]
    assert runner.req_states.req_id_to_index == {"B": 1, "A": 0}
    assert runner.req_states.free_indices == []
    assert runner.req_states.prompt_len.np.tolist() == [0, 2]
    assert state.intermediate_buffer.buffers == [{"req_id": "A"}, {"req_id": "B", "peer": True}]
    assert state.num_accepted_tokens_gpu.tolist() == [1, 7]
    if mode == "align":
        assert state._mamba_state_idx_gpu.tolist() == [-1, 6]


def test_cached_chunk_readds_released_request_through_core(monkeypatch):
    runner, model = loaded_runner(monkeypatch, runner_cls=OmniGenerationModelRunner)
    batch = configure_step(monkeypatch, runner, model)
    runner.model_config.async_chunk = True
    runner._release_generation_slots(batch)
    assert runner.req_states.req_id_to_index == {"B": 1}
    assert runner.req_states.free_indices == [0]
    assert not any(event[0] == "model_finished" for event in model.events)
    runner.adaptive_verification = None
    runner.lora_state.add_request = lambda *args: None
    runner.sampler.apply_staged_writes = lambda: None
    model.events.clear()

    runner._handle_async_chunk_updates(cached_chunk_output({"A": [7, 8]}))

    lifecycle = {"state_remove", "state_add", "request_apply", "state_apply"}
    assert [event for event in model.events if event[0] in lifecycle] == [
        ("state_remove", "A"),
        ("state_add", "A", 0),
        ("request_apply",),
        ("state_apply",),
    ]
    assert runner.req_states.req_id_to_index == runner.model_state.live == {"B": 1, "A": 0}
    assert runner.req_states.free_indices == []
    assert runner.req_states.all_token_ids.np[0, :2].tolist() == [7, 8]


@pytest.mark.parametrize("capability", ["native", "kv-send", "kv-receive", "stream-capture", "ar-structured"])
def test_required_capability_fails_during_real_load(monkeypatch, capability):
    config = make_config(
        native=capability == "native",
        kv={"need_send_cache": True}
        if capability == "kv-send"
        else {"need_recv_cache": True}
        if capability == "kv-receive"
        else None,
        eager=capability != "stream-capture",
    )
    model = TinyModel(capture=capability == "stream-capture")
    model._returns_tuple = capability == "ar-structured"
    name = {
        "native": "Native MRv2 data plane",
        "kv-send": "Omni KV transfer",
        "kv-receive": "Omni KV transfer",
        "stream-capture": "stream capture",
        "ar-structured": "AR structured output",
    }[capability]
    original_factory = core_module.init_model_state

    with pytest.raises(NotImplementedError, match=f"{name}.*CoreState"):
        loaded_runner(monkeypatch, config=config, model=model)

    runner = model.loading_runner
    assert len(model.created_states) == 1
    assert not hasattr(runner, "model_state")
    assert ("sampler_constructed",) not in model.events
    assert runner.kv_transfer_manager is None
    assert core_module.init_model_state is original_factory
    if capability == "native":
        assert runner._omni_data_plane.popped == []


@pytest.mark.parametrize("primary", [{}, None], ids=["empty-primary", "fallback"])
def test_optional_configurations_admit_core_state(monkeypatch, primary):
    config = make_config(kv=primary, eager=True)
    connector = {"type": "fallback"}
    config.kv_transfer_config = SimpleNamespace(omni_connector_config=connector)
    runner, model = loaded_runner(monkeypatch, config=config, model=TinyModel(capture=True))
    assert runner.model_state is model.created_states[0]
    assert runner.kv_transfer_manager is None
    manager = runner._ensure_kv_transfer_manager()
    assert not manager.config.need_send_cache and not manager.config.need_recv_cache
    if primary is None:
        assert manager.config.connector_config == connector
    else:
        assert not manager.config.connector_config


def test_capable_state_passes_native_kv_and_capture_validation(monkeypatch):
    config = make_config(native=True, kv={"need_send_cache": True}, eager=False)
    runner, model = loaded_runner(monkeypatch, config=config, model=TinyModel(CapabilityControlState, capture=True))
    assert runner.model_state is model.created_states[0]
    assert runner._omni_data_plane.popped == []
    assert runner.kv_transfer_manager is None


@pytest.mark.parametrize("missing", ["update", "buffers"])
def test_native_plane_rejects_partial_buffer_before_receiving(monkeypatch, missing):
    class PartialBufferState(CoreState):
        def __init__(self, *args):
            super().__init__(*args)
            buffer = {"update": lambda *args: None, "buffers": [{}]}
            del buffer[missing]
            self.intermediate_buffer = SimpleNamespace(**buffer)

    model = TinyModel(PartialBufferState)
    with pytest.raises(NotImplementedError, match=rf"Native MRv2 data plane.*\.{missing} on PartialBufferState"):
        loaded_runner(monkeypatch, model=model, config=make_config(native=True))
    assert model.loading_runner._omni_data_plane.popped == []
    assert ("sampler_constructed",) not in model.events


@pytest.mark.parametrize("value", ["external-A", b"external-A", ["external-A"]])
def test_kv_resolver_preserves_global_id(monkeypatch, value):
    runner, model = loaded_runner(monkeypatch, model=TinyModel(CapabilityControlState))
    configure_step(monkeypatch, runner, model)
    runner.model_state.intermediate_buffer.buffers[0]["global_request_id"] = value
    assert runner._resolve_global_request_id("A") == "external-A"
    assert runner._resolve_global_request_id("unknown") == "unknown"


def test_kv_resolver_rejects_mapped_request_without_buffer(monkeypatch):
    runner, model = loaded_runner(monkeypatch)
    configure_step(monkeypatch, runner, model)
    with pytest.raises(NotImplementedError, match=r"Omni KV transfer.*\.buffers on CoreState"):
        runner._resolve_global_request_id("A")


def test_kv_send_pre_hook_uses_orchestrator_request_id(monkeypatch):
    config = make_config(kv={"need_send_cache": True})
    runner, model = loaded_runner(monkeypatch, config=config, model=TinyModel(CapabilityControlState))
    configure_step(monkeypatch, runner, model)
    runner.model_state.intermediate_buffer.buffers[0]["global_request_id"] = "external-A"
    runner.kv_caches = [torch.empty(1)]
    runner.cache_config.block_size = 16
    runner.cache_config.cache_dtype = "auto"
    output = scheduled()
    output.finished_requests_needing_kv_transfer = {"A": {}}
    sent: list[str] = []

    def send(*, finished_reqs, kv_caches, block_size, cache_dtype, request_id_resolver):
        assert kv_caches is runner.kv_caches
        assert block_size == 16 and cache_dtype == "auto"
        sent.extend(request_id_resolver(req_id) for req_id in finished_reqs)
        return list(finished_reqs)

    runner.kv_transfer_manager = SimpleNamespace(handle_finished_requests_kv_transfer=send)
    runner._handle_kv_transfer_pre(output)
    assert sent == ["external-A"]
    assert runner._kv_extracted_req_ids == ["A"]


@pytest.mark.parametrize("profile_only", [False, True])
def test_model_owned_capture_uses_state_sizes_and_original_buckets(monkeypatch, profile_only):
    class CaptureState(CapabilityControlState):
        def _get_mtp_capture_sizes(self):
            return [4, 2, 2]

    config = make_config(eager=False)
    config.scheduler_config.max_num_seqs = 4
    runner, model = loaded_runner(monkeypatch, config=config, model=TinyModel(CaptureState, capture=True))
    parent_calls = []

    def parent_capture(self, *, profile_only):
        parent_calls.append(profile_only)
        return 7

    monkeypatch.setattr(CoreRunner, "capture_model", parent_capture)
    assert runner.capture_model(profile_only=profile_only) == 7
    assert parent_calls == [profile_only]
    assert [event for event in model.events if event[0] == "capture"] == (
        [] if profile_only else [("capture", [1, 2, 4])]
    )
    descriptor = runner._dispatch_mtp_batch_descriptor(3)
    assert descriptor.cg_mode == CUDAGraphMode.FULL
    assert descriptor.num_tokens == descriptor.num_reqs == 4
    fallback = runner._dispatch_mtp_batch_descriptor(5)
    assert fallback.cg_mode == CUDAGraphMode.NONE and fallback.num_tokens == 5


def test_profile_capture_with_no_sizing_does_not_call_model_hook(monkeypatch):
    runner, model = loaded_runner(monkeypatch, model=TinyModel(capture=True), config=make_config(eager=True))
    monkeypatch.setattr(CoreRunner, "capture_model", lambda self, *, profile_only: 7)
    assert runner.capture_model(profile_only=True) == 7
    assert not any(event[0] == "capture" for event in model.events)


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("kind", ["empty-omni-output", "omni-output", "empty-aux-tuple", "aux-tuple", "dict"])
def test_rejects_unconverted_outputs_before_unpack(monkeypatch, mode, kind):
    runner, model = loaded_runner(monkeypatch)
    configure_step(monkeypatch, runner, model, mode)
    hidden = torch.zeros(2, 2)
    raw = {
        "empty-omni-output": OmniOutput(text_hidden_states=hidden, multimodal_outputs={}),
        "omni-output": OmniOutput(text_hidden_states=hidden, multimodal_outputs={"audio": hidden}),
        "empty-aux-tuple": (hidden, None),
        "aux-tuple": (hidden, {"audio": hidden}),
        "dict": {"hidden": hidden},
    }[kind]
    monkeypatch.setattr(model, "forward", lambda **kwargs: raw)
    runner.cudagraph_manager = SimpleNamespace(
        run_pw_graph=lambda model, inputs: model(**inputs),
        run_fullgraph=lambda descriptor: raw,
    )
    with pytest.raises(NotImplementedError, match="structured output.*CoreState"):
        runner.execute_model(scheduled())
    assert runner.execute_model_state is None


@pytest.mark.parametrize("source", ["platform-full", "kstep-converter"])
def test_optional_output_producers_are_checked_before_unpack(monkeypatch, mocker, source):
    runner, model = loaded_runner(monkeypatch)
    mode = CUDAGraphMode.FULL if source == "platform-full" else CUDAGraphMode.NONE
    configure_step(monkeypatch, runner, model, mode)
    structured = OmniOutput(text_hidden_states=torch.ones(2, 2), multimodal_outputs={"audio": torch.ones(2)})
    producer = mocker.Mock(return_value=structured)
    if source == "platform-full":
        monkeypatch.setattr(runner, "_full_graph_step_output", producer)
        runner.cudagraph_manager = SimpleNamespace(run_fullgraph=mocker.Mock())
    else:
        runner._kstep_frames = 2
        monkeypatch.setattr(model, "flush_cuda_kstep_feedback", lambda: None, raising=False)
        monkeypatch.setattr(runner, "_build_kstep_model_kwargs_extra", lambda *args: {"step": 1})
        monkeypatch.setattr(model, "make_omni_output", producer, raising=False)

    with pytest.raises(NotImplementedError, match=r"structured output \(OmniOutput\).*CoreState"):
        runner.execute_model(scheduled())

    producer.assert_called_once()
    assert runner.execute_model_state is None
    assert runner._last_multimodal_outputs is None
    if source == "platform-full":
        runner.cudagraph_manager.run_fullgraph.assert_not_called()
    else:
        assert producer.call_args.kwargs["request_sample_eligible"] == [True]


def test_cached_payload_fails_before_core_request_update(monkeypatch):
    runner, model = loaded_runner(monkeypatch)
    configure_step(monkeypatch, runner, model)
    output = scheduled()
    output.scheduled_cached_reqs.additional_information = {"A": {"sentinel": True}}
    calls = []
    monkeypatch.setattr(CoreRunner, "update_requests", lambda *args: calls.append("core_update"))
    with pytest.raises(NotImplementedError, match="Cached additional_information.*CoreState"):
        runner.update_requests(output)
    assert calls == []
    output.scheduled_cached_reqs.additional_information = {"A": {}, "unknown": {"sentinel": True}}
    runner.update_requests(output)
    assert calls == ["core_update"]


def test_cached_payload_merges_serialized_info_once_after_core_update(monkeypatch, mocker):
    runner, model = loaded_runner(monkeypatch, model=TinyModel(CapabilityControlState))
    configure_step(monkeypatch, runner, model)
    buffer = runner.model_state.intermediate_buffer
    buffer.update(0, {"embed": {"keep": 7}})
    buffer.update(1, {"peer": True})
    encoded = serialize_additional_information({"embed": {"decode": torch.tensor([[3.0, 4.0]])}})
    output = scheduled()
    output.scheduled_cached_reqs = OmniCachedRequestData(
        req_ids=["A"],
        new_token_ids=[[]],
        new_block_ids=[None],
        num_computed_tokens=[1],
        num_output_tokens=[0],
        prompt_token_ids={},
        additional_information={"A": encoded, "B": None, "unknown": "unused"},
    )
    resolver = mocker.spy(base_module, "_resolve_additional_information")
    original_update = CoreRunner.update_requests

    def core_update(self, scheduler_output):
        assert "decode" not in buffer.buffers[0]["embed"]
        original_update(self, scheduler_output)
        model.events.append(("core_update", int(self.req_states.num_computed_tokens_np[0])))

    monkeypatch.setattr(CoreRunner, "update_requests", core_update)
    runner.update_requests(output)
    resolver.assert_called_once_with(encoded)
    assert model.events.count(("core_update", 1)) == 1
    assert buffer.buffers[0]["embed"]["keep"] == 7
    torch.testing.assert_close(buffer.buffers[0]["embed"]["decode"], torch.tensor([[3.0, 4.0]]))
    assert buffer.buffers[1] == {"peer": True}

    output.scheduled_cached_reqs.additional_information = {"A": "malformed"}
    with pytest.raises(ValueError, match="Failed to decode additional_information"):
        runner.update_requests(output)
    assert model.events.count(("core_update", 1)) == 1
    torch.testing.assert_close(buffer.buffers[0]["embed"]["decode"], torch.tensor([[3.0, 4.0]]))


def test_existing_multimodal_hook_runs_after_output_conversion(monkeypatch):
    class State(CoreState):
        def postprocess_model_output(self, output, batch, states):
            self.events.append(("converted",))
            return output, {"audio": torch.ones(1)}

        def run_postprocess(self, hidden, batch, multimodal_outputs):
            self.events.append(("feedback", multimodal_outputs["audio"].tolist()))

    model = TinyModel(State)
    model.postprocess_requires_multimodal_outputs = True
    runner, model = loaded_runner(monkeypatch, model=model)
    configure_step(monkeypatch, runner, model)
    runner.execute_model(scheduled())
    assert ("feedback", [1.0]) not in model.events
    runner.sample_tokens(None)
    assert model.events.count(("feedback", [1.0])) == 1
    assert model.events.index(("converted",)) < model.events.index(("feedback", [1.0]))


def test_missing_multimodal_feedback_hook_fails_during_load(monkeypatch):
    class ConverterOnlyState(CoreState):
        def postprocess_model_output(self, output, batch, states):
            return output, {}

    model = TinyModel(ConverterOnlyState)
    model.postprocess_requires_multimodal_outputs = True

    with pytest.raises(NotImplementedError, match="multimodal feedback requires run_postprocess on ConverterOnlyState"):
        loaded_runner(monkeypatch, model=model)

    assert len(model.created_states) == 1
    assert not hasattr(model.loading_runner, "model_state")
    assert ("sampler_constructed",) not in model.events


@pytest.mark.parametrize("kind", ["omni-output", "drafter-state"])
def test_ar_rejects_pending_aux_without_consuming_the_step(monkeypatch, kind):
    runner, model = loaded_runner(monkeypatch)
    configure_step(monkeypatch, runner, model)
    runner.execute_model(scheduled())
    step = runner.execute_model_state
    aux = {"existing": torch.ones(2)}
    if kind == "omni-output":
        runner._last_aux_output = aux
        error = "structured output"
    else:
        step = step._replace(aux_hidden_states=[aux["existing"]])
        runner.execute_model_state = step
        error = "AR auxiliary output"

    with pytest.raises(NotImplementedError, match=f"{error}.*CoreState"):
        runner.sample_tokens(None)

    assert runner.execute_model_state is step
    if kind == "omni-output":
        assert runner._last_aux_output is aux
    else:
        assert step.aux_hidden_states[0] is aux["existing"]
    assert not any(event[0] == "sample" for event in model.events)


@pytest.mark.parametrize("kind", ["direct", "no-gather", "no-converter", "legacy"])
def test_generation_output_uses_existing_stage_contract(monkeypatch, kind):
    state_cls = CapabilityControlState if kind in ("no-converter", "legacy") else CoreState
    model = TinyModel(state_cls)
    converted = OmniOutput(text_hidden_states=torch.zeros(2, 2), multimodal_outputs={})
    seen = []

    def convert(output, *, model_intermediate_buffer, runtime_additional_information):
        assert model_intermediate_buffer is runtime_additional_information
        seen.append(model_intermediate_buffer)
        return converted

    if kind == "legacy":
        model.mrv2_forward_returns_tensor = True
        model.make_omni_output = convert

    runner, model = loaded_runner(monkeypatch, model=model, runner_cls=OmniGenerationModelRunner)
    configure_step(monkeypatch, runner, model)
    runner._prev_step_idle = False
    runner._dummy_hidden = torch.empty(1)
    if kind == "direct":
        monkeypatch.setattr(model, "forward", lambda **kwargs: converted)
    if kind == "legacy":
        runner.model_state.intermediate_buffer.buffers[0]["sentinel"] = "A"

    if kind in ("direct", "legacy"):
        runner.execute_model(scheduled())
        assert runner._gen_model_output is converted
        assert runner.execute_model_state.attn_metadata is None
        assert ("prepare_inputs",) in model.events
        assert ("prepare_attn",) not in model.events
        assert seen == ([[{"sentinel": "A"}]] if kind == "legacy" else [])
    else:
        detail = "intermediate_buffer.gather" if kind == "no-gather" else "model.make_omni_output"
        with pytest.raises(NotImplementedError, match=detail):
            runner.execute_model(scheduled())
        assert runner.execute_model_state is None


@pytest.mark.parametrize("has_buffer", [False, True])
def test_declared_generation_converter_capability_fails_during_load(monkeypatch, has_buffer):
    model = TinyModel(CapabilityControlState if has_buffer else CoreState)
    model.mrv2_forward_returns_tensor = True
    detail = r"model\.make_omni_output.*CapabilityControlState" if has_buffer else "gather.*CoreState"
    with pytest.raises(NotImplementedError, match=f"generation output.*{detail}"):
        loaded_runner(monkeypatch, model=model, runner_cls=OmniGenerationModelRunner)
    assert ("sampler_constructed",) not in model.events


@pytest.mark.parametrize("dummy_run", [False, True], ids=["real", "dummy"])
def test_generation_converter_gets_buffers_in_batch_order(monkeypatch, dummy_run):
    model = TinyModel(CapabilityControlState)
    runner, model = loaded_runner(monkeypatch, model=model, runner_cls=OmniGenerationModelRunner)
    batch = configure_step(monkeypatch, runner, model)
    batch.req_ids = ["B", "A"]
    batch.num_reqs = batch.num_reqs_after_padding = 2
    batch.idx_mapping = torch.tensor([1, 0])
    batch.idx_mapping_np = batch.idx_mapping.numpy()
    batch.num_scheduled_tokens = np.array([1, 1])
    batch.query_start_loc = torch.tensor([0, 1, 2])
    batch.query_start_loc_np = batch.query_start_loc.numpy()
    runner._prev_step_idle = False
    runner._dummy_hidden = torch.empty(1)
    runner._retains_state_across_chunks = False
    state = runner.model_state
    state.intermediate_buffer.buffers[:] = [{"sentinel": "A"}, {"sentinel": "B"}]
    seen = []

    def convert(output, *, model_intermediate_buffer, runtime_additional_information):
        assert model_intermediate_buffer is runtime_additional_information
        seen.extend(model_intermediate_buffer)
        return OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": [row[:1] for row in output]})

    monkeypatch.setattr(model, "make_omni_output", convert, raising=False)
    monkeypatch.setattr(runner, "_make_dummy_input_batch", lambda *args, **kwargs: batch)
    output = scheduled()
    output.num_scheduled_tokens = {"B": 1, "A": 1}
    runner.execute_model(output, dummy_run=dummy_run)

    assert seen == [{"sentinel": "B"}, {"sentinel": "A"}]
    assert isinstance(runner._gen_model_output, OmniOutput)
    if not dummy_run:
        result = runner.sample_tokens()
        assert result.req_ids == ["B", "A"]
        assert result.sampled_token_ids == [[], []]
        assert [item["model_outputs"].tolist() for item in result.multimodal_outputs] == [[2.0], [5.0]]
        assert runner.execute_model_state is None
        assert runner.req_states.num_computed_tokens.np.tolist() == [2, 2]
    assert not {"prepare_attn", "preprocess_state", "postprocess_state"}.intersection(e[0] for e in model.events)


@pytest.mark.parametrize("specialized", [False, True])
def test_real_factory_wraps_only_exact_default_for_omni_stage(specialized):
    created = []

    class SpecializedState(DefaultModelState):
        def __init__(self, *args):
            super().__init__(*args)
            created.append(self)

    model = TinyModel(SpecializedState if specialized else DefaultModelState)
    # These classes do not record construction in TinyModel.created_states.
    state = base_module.init_omni_model_state(make_config(), model, None, torch.device("cpu"))
    assert type(state) is (SpecializedState if specialized else OmniModelState)
    assert created == ([state] if specialized else [])


def test_factory_preserves_core_state_sampler_override(monkeypatch):
    class CustomSamplerState(CoreState):
        def custom_sampler(self, sampler):
            self.events.append(("state_sampler", self, sampler))
            return sampler, None

    config = make_config()
    config._check_supports_watermarking = MethodType(VllmConfig._check_supports_watermarking, config)
    runner, model = loaded_runner(monkeypatch, config=config, model=TinyModel(CustomSamplerState))
    assert runner.model_state is model.created_states[0]
    assert type(runner.model_state) is CustomSamplerState
    assert model.events.index(("sampler_constructed",)) < model.events.index(
        ("state_sampler", runner.model_state, runner.sampler)
    )
