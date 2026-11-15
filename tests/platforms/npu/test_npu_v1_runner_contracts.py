# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run NPU call boundaries on CPU without importing the Ascend runtime.

Only the hardware-dependent enclosing execution is omitted: calls and K8
proposal methods are compiled directly from production source, and use the
actual retained V1 signature, slot mapper and frame-drafting algorithm.
"""

from __future__ import annotations

import ast
import copy
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_omni.worker.v1_compat.gpu_model_runner import V1GPUModelRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
_ROOT = Path(__file__).resolve().parents[3]
_NPU = _ROOT / "vllm_omni/platforms/npu/worker"


def _class(filename, name):
    tree = ast.parse((_NPU / filename).read_text())
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def _method(filename, cls, name):
    return next(n for n in _class(filename, cls).body if isinstance(n, ast.FunctionDef) and n.name == name)


def _call(method, name):
    return next(
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == name
    )


def _eval_call(call, values):
    return eval(compile(ast.Expression(call), "<NPU production call>", "eval"), values)


class _Contracts:
    _get_slot_mappings = V1GPUModelRunnerMixin._get_slot_mappings

    def __init__(self):
        self.device = torch.device("cpu")
        self.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=None, layer_names=["layer0"])]
        )
        self.slots = torch.tensor([11, 12, 99, 99])
        self.input_batch = SimpleNamespace(block_table=[SimpleNamespace(slot_mapping=SimpleNamespace(gpu=self.slots))])
        self.seen = None

    def _build_attention_metadata(self, *args, **kwargs):
        self.seen = (
            inspect.signature(V1GPUModelRunnerMixin._build_attention_metadata).bind(self, *args, **kwargs).arguments
        )
        assert self.seen["slot_mappings"][0].data_ptr() == self.slots.data_ptr()
        return {"layer0": self.seen["slot_mappings"][0]}, None


@pytest.mark.parametrize(
    "filename,cls,method_name",
    [
        ("npu_ar_model_runner.py", "NPUARModelRunner", "execute_model"),
        ("npu_generation_model_runner.py", "NPUGenerationModelRunner", "execute_model"),
        ("npu_generation_model_runner.py", "NPUGenerationModelRunner", "_dummy_run"),
        ("npu_model_runner.py", "OmniNPUModelRunner", "_dummy_run"),
    ],
)
def test_metadata_call_preserves_group_mapping_and_padded_dimensions(filename, cls, method_name):
    method = _method(filename, cls, method_name)
    call = _call(method, "_build_attention_metadata")
    runner = _Contracts()
    values = dict(
        self=runner,
        num_tokens_unpadded=2,
        num_tokens_padded=4,
        num_reqs=1,
        num_reqs_padded=2,
        max_num_scheduled_tokens=2,
        max_query_len=2,
        ubatch_slices_attn=None,
        ubatch_slices_padded=None,
        ubatch_slices=None,
        pad_attn=True,
        logits_indices=None,
        use_spec_decode=True,
        scheduler_output=SimpleNamespace(num_scheduled_tokens={"r": 2}),
        num_scheduled_tokens_np=[2],
        num_scheduled_tokens=[2],
        cascade_attn_prefix_lens=None,
        is_graph_capturing=True,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_desc=object(),
        skip_gdn_state_update=True,
    )
    # Execute the real slot-mapping preparation, including the real CPU mapper.
    preparation = [
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "_get_slot_mappings"
        and n.lineno < call.lineno
    ]
    if preparation:
        stmt = max(preparation, key=lambda n: n.lineno)
        exec(compile(ast.Module(body=[stmt], type_ignores=[]), "<NPU mapping>", "exec"), values)
    metadata, _ = _eval_call(call, values)
    assert runner.seen is not None
    assert runner.seen["num_tokens"] == 2
    assert runner.seen["num_tokens_padded"] == 4
    assert runner.seen["num_reqs_padded"] == 2
    assert metadata["layer0"].tolist() == [11, 12, -1, -1]
    if method_name == "execute_model":
        assert runner.seen["num_scheduled_tokens"] is values["scheduler_output"].num_scheduled_tokens
    else:
        assert runner.seen["for_cudagraph_capture"] is True


def test_ar_draft_call_uses_shared_order_and_mapping_identity():
    method = _method("npu_ar_model_runner.py", "NPUARModelRunner", "sample_tokens")
    call = _call(method, "propose_draft_token_ids")
    names = [
        "scheduler_output",
        "sampled_token_ids",
        "hidden_states",
        "sample_hidden_states",
        "aux_hidden_states",
        "spec_decode_metadata",
        "spec_decode_common_attn_metadata",
        "slot_mappings",
    ]
    values = {name: object() for name in names}
    sampling_metadata = object()

    def capture(*args, **kwargs):
        bound = inspect.signature(V1GPUModelRunnerMixin.propose_draft_token_ids).bind(runner, *args, **kwargs)
        return bound.arguments

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(sampling_metadata=sampling_metadata), propose_draft_token_ids=capture
    )
    values.update(self=runner, positions=object(), batch_desc=object())
    # The historical signature tried reading total_num_scheduled_tokens too.
    values["scheduler_output"] = SimpleNamespace(total_num_scheduled_tokens=5)
    actual = _eval_call(call, values)
    for name in names:
        key = "common_attn_metadata" if name == "spec_decode_common_attn_metadata" else name
        assert actual[key] is values[name]
    assert actual["sampling_metadata"] is sampling_metadata


def test_ar_bookkeeping_preserves_exact_shared_inputs():
    method = _method("npu_ar_model_runner.py", "NPUARModelRunner", "sample_tokens")
    call = _call(method, "_bookkeeping_sync")
    values = {name: object() for name in ["sampler_output", "logits", "hidden_states", "spec_decode_metadata"]}
    values["scheduler_output"] = SimpleNamespace(total_num_scheduled_tokens=7)

    def capture(*args, **kwargs):
        return inspect.signature(V1GPUModelRunnerMixin._bookkeeping_sync).bind(runner, *args, **kwargs).arguments

    runner = SimpleNamespace(_bookkeeping_sync=capture)
    values["self"] = runner
    actual = _eval_call(call, values)
    assert actual["num_scheduled_tokens"] == 7
    for name in ["scheduler_output", "sampler_output", "logits", "hidden_states"]:
        assert actual[name] is values[name]


def _proposal_class():
    method = copy.deepcopy(_method("npu_ar_model_runner.py", "NPUARModelRunner", "propose_draft_token_ids"))

    class Base:
        def propose_draft_token_ids(self, *args, **kwargs):
            return (
                inspect.signature(V1GPUModelRunnerMixin.propose_draft_token_ids).bind(self, *args, **kwargs).arguments
            )

    cls = ast.ClassDef(
        name="NPUProposal", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
            type_ignores=[],
        )
    )
    scope = dict(Base=Base, logging=logging, logger=logging.getLogger(__name__), torch=torch)
    exec(compile(module, "<NPU proposal method>", "exec"), scope)
    return scope["NPUProposal"]


@pytest.mark.parametrize("frames", [0, 8])
def test_npu_proposal_keeps_generic_mapping_or_real_k8_drafts(frames):
    runner = _proposal_class()()
    runner.model = SimpleNamespace(
        supports_multi_frame_decode=True,
        batch_stop_logits=None,
        merge_frame_outputs=lambda *args: None,
        set_batch_stop_logits=lambda *args: None,
        take_batch_stop_logits=lambda *args: None,
    )
    runner.num_spec_tokens = max(frames - 1, 0)
    runner.input_batch = SimpleNamespace(req_ids=["r"], num_reqs=1)
    runner.speculative_config = SimpleNamespace(use_ngram_gpu=lambda: False)
    scheduler = SimpleNamespace(scheduled_spec_decode_tokens={"r": []})
    mapping = {"layer0": torch.tensor([11])}
    sampled = [[0]]
    result = runner.propose_draft_token_ids(
        scheduler, sampled, object(), object(), object(), None, None, object(), mapping
    )
    if frames:
        assert result == [[0] * 7]
        scheduler.scheduled_spec_decode_tokens = {"r": [0] * 7}
        # A live request stopped inside its K8 span: preserve the existing fold.
        assert runner.propose_draft_token_ids(
            scheduler, sampled, object(), object(), object(), None, None, object(), mapping
        ) == [[]]
    else:
        assert result["scheduler_output"] is scheduler
        assert result["sampled_token_ids"] is sampled
        assert result["slot_mappings"] is mapping


def test_generation_profile_forwards_randomize_inputs():
    method = copy.deepcopy(_method("npu_generation_model_runner.py", "NPUGenerationModelRunner", "profile_run"))
    method.decorator_list = []
    namespace: dict = dict(gc=SimpleNamespace(collect=lambda: None), logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<NPU profile>", "exec"), namespace)
    calls = []

    def dummy_run(*args, **kwargs):
        calls.append((args, kwargs))
        return None, None

    runner = SimpleNamespace(
        supports_mm_inputs=False,
        max_num_tokens=13,
        _dummy_run=dummy_run,
        _sync_device=lambda: None,
        encoder_cache={},
    )
    namespace["profile_run"](runner, randomize_inputs=True)
    assert calls == [((13,), {"is_profile": True, "randomize_inputs": True})]


@pytest.mark.parametrize(
    "filename,cls",
    [("npu_ar_model_runner.py", "NPUARModelRunner"), ("npu_generation_model_runner.py", "NPUGenerationModelRunner")],
)
def test_execute_sample_state_keeps_same_per_layer_mapping(filename, cls):
    state = copy.deepcopy(_class("npu_ar_model_runner.py", "ExecuteModelState"))
    scope = dict(NamedTuple=NamedTuple)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), state],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<NPU ephemeral state>", "exec"), scope)
    create = next(
        n
        for n in ast.walk(_method(filename, cls, "execute_model"))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "ExecuteModelState"
    )
    mapping = {"layer0": torch.tensor([11])}
    values = {n.id: object() for n in ast.walk(create) if isinstance(n, ast.Name)}
    values.update(ExecuteModelState=scope["ExecuteModelState"], slot_mappings=mapping)
    state_value = _eval_call(create, values)
    assert state_value.slot_mappings is mapping
    sid = values["prefix_cache_step_id"] if cls == "NPUARModelRunner" else None
    assert state_value.prefix_cache_step_id is sid
    assert state_value[-1] is sid
    values.pop("slot_mappings")
    consume = next(
        n
        for n in ast.walk(_method(filename, cls, "sample_tokens"))
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Attribute) and n.value.attr == "execute_model_state"
    )
    values["self"] = SimpleNamespace(execute_model_state=state_value)
    exec(compile(ast.Module(body=[consume], type_ignores=[]), "<NPU consume state>", "exec"), values)
    assert values["slot_mappings" if cls == "NPUARModelRunner" else "_slot_mappings"] is mapping
    assert values["prefix_cache_step_id" if cls == "NPUARModelRunner" else "_prefix_cache_step_id"] is sid


@pytest.mark.parametrize(
    "filename", ["npu_ar_model_runner.py", "npu_generation_model_runner.py", "npu_model_runner.py"]
)
def test_npu_v1_boundary_imports_resolve_on_current_pin(filename):
    import importlib

    imports = [
        n
        for n in ast.walk(ast.parse((_NPU / filename).read_text()))
        if isinstance(n, ast.ImportFrom)
        and n.module
        and (
            n.module.endswith(
                (
                    "gpu_model_runner",
                    "ubatch_utils",
                    "spec_decode.metadata",
                    "structured_output.utils",
                    "compilation.monitor",
                )
            )
            or n.module.startswith("vllm_omni.worker.v1_compat")
        )
    ]
    assert imports
    for node in imports:
        assert node.module is not None
        module = importlib.import_module(node.module)
        for imported in node.names:
            assert hasattr(module, imported.name)


@pytest.mark.parametrize(
    "filename,cls",
    [("npu_generation_model_runner.py", "NPUGenerationModelRunner"), ("npu_model_runner.py", "OmniNPUModelRunner")],
)
@pytest.mark.parametrize("skip", [False, True])
def test_dummy_gdn_state_skip_preserves_existing_buffer_update(filename, cls, skip):
    import numpy as np

    method = _method(filename, cls, "_dummy_run")
    branch = next(n for n in ast.walk(method) if isinstance(n, ast.If) and ast.unparse(n.test) == "self._has_gdn")
    copied = []
    buffer = SimpleNamespace(np=np.full(4, 99), copy_to_gpu=lambda: copied.append(True))
    runner = SimpleNamespace(_has_gdn=True, gdn_query_start_loc=buffer)
    values = dict(self=runner, skip_gdn_state_update=skip, num_reqs_padded=2, cum_num_tokens=np.array([1, 2]))
    exec(compile(ast.Module(body=[branch], type_ignores=[]), "<NPU GDN buffer update>", "exec"), values)
    assert buffer.np.tolist() == ([0, 0, 0, 0] if skip else [99, 1, 2, 99])
    assert copied == [True]


def test_npu_kv_initializer_keeps_default_backend_call_and_rejects_unsupported_modes():
    calls = []

    class BackendBoundary:
        def initialize_kv_cache(self, config, kv_cache_allocation_context=None):
            calls.append((config, kv_cache_allocation_context))

    method = copy.deepcopy(_method("npu_model_runner.py", "OmniNPUModelRunner", "initialize_kv_cache"))
    scope: dict = dict(inspect=inspect, NPUModelRunner=BackendBoundary)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<NPU KV initialization>", "exec"), scope)
    runner = SimpleNamespace(_omni_prefix_cache_cfg=object(), _init_omni_aux_output=lambda config: None)
    config, context = object(), object()
    init = scope["initialize_kv_cache"]
    init(runner, config, kv_cache_allocation_context=context)
    assert calls == [(config, context)]
    assert runner.extensible_kv_cache is None
    calls.clear()
    with pytest.raises(ValueError, match="fixed KV"):
        init(runner, config, extensible=True)
    with pytest.raises(NotImplementedError, match="profiling KV"):
        init(runner, config, is_profiling=True)
    assert not calls
