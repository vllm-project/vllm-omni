# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Core ModelState compatibility through the production MRv2 AR runner."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.config import LoadConfig
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu import model_runner as core_module
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.core.sched.omni_scheduling_coordinator import uses_native_mrv2_data_plane
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2 import omni_ar_model_runner as ar_module
from vllm_omni.worker_v2 import omni_model_runner as base_module
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner
from vllm_omni.worker_v2.staged_writes import StagedWrites

Base = base_module.OmniGPUModelRunner
AR = ar_module.OmniARModelRunner
CoreRunner = core_module.GPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FixtureOmniConfig(OmniModelConfig):
    is_hybrid = False
    is_attention_free = False

    def get_inputs_embeds_size(self):
        return 2


def make_config(*, native=False, kv=None, eager=True):
    mc = object.__new__(FixtureOmniConfig)
    mc.__dict__.update(
        dtype=torch.float32,
        max_model_len=32,
        enable_prompt_embeds=False,
        uses_mrope=False,
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
        cache_config=SimpleNamespace(mamba_cache_mode="none", use_kda_recoverssm=False),
        parallel_config=parallel,
        load_config=LoadConfig(load_format="dummy"),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE),
        additional_config={},
        speculative_config=None,
        watermark_config=None,
        kv_transfer_config=None,
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
    """Observable I/O boundary used only in unsupported capability probes."""

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
    assert runner.model_state is model.created_states[-1]
    return runner, model


def configure_step(monkeypatch, runner, model, mode=CUDAGraphMode.NONE):
    for req_id in ("B", "A"):
        runner.req_states.add_request(req_id, 2, [2, 5], 0, 3)
        runner.model_state.add_request(runner.req_states.req_id_to_index[req_id], SimpleNamespace(req_id=req_id))
    runner.req_states.apply_staged_writes = lambda: model.events.append(("request_apply",))
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[], num_blocks=2)
    runner.attn_groups = []
    runner.block_tables = SimpleNamespace(apply_staged_writes=lambda: None, append_block_ids=lambda *a, **kw: None)
    runner.kv_connector = SimpleNamespace(pre_forward=lambda *a: None, post_forward=lambda ids: None)
    runner.ec_connector = SimpleNamespace()
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


def test_core_state_runs_ar_step_sampling_and_parent_teardown(monkeypatch):
    runner, model = loaded_runner(monkeypatch)
    assert runner._omni_data_plane is None
    assert type(runner.model_state) is CoreState
    configure_step(monkeypatch, runner, model)

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
    assert (
        tags.index("preprocess_state") < tags.index("forward") < tags.index("sample") < tags.index("postprocess_state")
    )
    assert tags.index("model_finished") < tags.index("state_remove")
    assert [event for event in model.events if event[0] == "state_remove"] == [("state_remove", "A")]


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


def test_optional_configurations_admit_core_state(monkeypatch):
    config = make_config(kv={}, eager=True)
    config.kv_transfer_config = SimpleNamespace(omni_connector_config={"type": "fallback"})
    runner, model = loaded_runner(monkeypatch, config=config, model=TinyModel(capture=True))
    assert runner.model_state is model.created_states[0]
    assert runner.kv_transfer_manager is None


def test_capable_state_passes_native_kv_and_capture_validation(monkeypatch):
    config = make_config(native=True, kv={"need_send_cache": True}, eager=False)
    runner, model = loaded_runner(monkeypatch, config=config, model=TinyModel(CapabilityControlState, capture=True))
    assert runner.model_state is model.created_states[0]
    assert runner._omni_data_plane.popped == []
    assert runner.kv_transfer_manager is None


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("kind", ["empty-omni-output", "empty-aux-tuple"])
def test_rejects_unconverted_outputs_before_unpack(monkeypatch, mode, kind):
    runner, model = loaded_runner(monkeypatch)
    configure_step(monkeypatch, runner, model, mode)
    hidden = torch.zeros(2, 2)
    raw = (
        OmniOutput(text_hidden_states=hidden, multimodal_outputs={}) if kind == "empty-omni-output" else (hidden, None)
    )
    monkeypatch.setattr(model, "forward", lambda **kwargs: raw)
    runner.cudagraph_manager = SimpleNamespace(
        run_pw_graph=lambda model, inputs: model(**inputs),
        run_fullgraph=lambda descriptor: raw,
    )
    with pytest.raises(NotImplementedError, match="structured output.*CoreState"):
        runner.execute_model(scheduled())
    assert runner.execute_model_state is None


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


def test_ar_rejects_pending_aux_without_consuming_the_step(monkeypatch):
    runner, model = loaded_runner(monkeypatch)
    configure_step(monkeypatch, runner, model)
    runner.execute_model(scheduled())
    step = runner.execute_model_state
    aux = {"existing": torch.ones(2)}
    runner._last_aux_output = aux

    with pytest.raises(NotImplementedError, match="structured output.*CoreState"):
        runner.sample_tokens(None)

    assert runner.execute_model_state is step
    assert runner._last_aux_output is aux
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


def test_declared_generation_converter_capability_fails_during_load(monkeypatch):
    model = TinyModel()
    model.mrv2_forward_returns_tensor = True
    with pytest.raises(NotImplementedError, match="generation output.*gather.*CoreState"):
        loaded_runner(monkeypatch, model=model, runner_cls=OmniGenerationModelRunner)
    assert ("sampler_constructed",) not in model.events
