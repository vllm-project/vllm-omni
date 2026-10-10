# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU ownership/event oracle; no NPU import or device-performance assertion."""

import __future__

import ast
import importlib.util
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(scope="module")
def leased():
    path = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/layers/lingbot_leased_post_attention.py"
    spec = importlib.util.spec_from_file_location("sp8_leased_post_attention_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Runtime:
    def __init__(self):
        self.stream = "producer"
        self.events = []
        self.log = []
        self.supported = True
        self.fail_capture = False

    def eligible(self, inputs):
        return self.supported

    def current_stream(self, device):
        return self.stream

    def empty(self, value):
        return torch.empty(value.shape, dtype=value.dtype, device=value.device)

    def record_owner(self, value, stream):
        self.log.append(("owner", value.data_ptr(), stream))

    def record_event(self, stream):
        event = (len(self.events), stream)
        self.events.append(event)
        self.log.append(("record", stream, event))
        return event

    def wait_event(self, stream, event):
        self.log.append(("wait", stream, event))

    def capture(self, operation, inputs, stream):
        self.log.append(("capture", stream))
        if self.fail_capture:
            raise RuntimeError("injected capture failure")
        with torch.inference_mode():
            outputs = tuple(torch.empty_like(value) for value in operation(*inputs))

        class Graph:
            def replay(self):
                with torch.inference_mode():
                    for target, result in zip(outputs, operation(*inputs), strict=True):
                        target.copy_(result)

        return Graph(), outputs, ("unique pool", object())


def operands(tokens=3, width=8):
    torch.manual_seed(17)
    hidden = torch.randn(1, tokens, width).bfloat16()
    projection = torch.randn_like(hidden)
    gate = torch.randn(1, 1, 1, width)
    scale, shift = torch.randn_like(hidden), torch.randn_like(hidden)
    return projection, hidden, gate, scale, shift


def controller(leased, *, tokens=3, width=8, max_signatures=4):
    block = SimpleNamespace(
        self_attn=SimpleNamespace(o=torch.nn.Linear(width, width).bfloat16()),
        norm3=torch.nn.LayerNorm(width).bfloat16(),
    )
    for module in (block.self_attn.o, block.norm3):
        module.requires_grad_(False)
    runtime = Runtime()
    failures = []
    instance = leased.LeasedPostAttention(
        block, runtime=runtime, max_signatures=max_signatures, failure_reporter=lambda *args: failures.append(args)
    )
    return block, instance, runtime, failures


def produce(lease, projection):
    lease.projection_input.copy_(projection)
    lease.mark_projection_written()
    return lease.finish()


def test_direct_producer_outputs_match_native_rounding_without_output_clone(leased):
    block, runner, runtime, _ = controller(leased)
    projection, *inputs = operands()
    before = [value.clone() for value in (projection, *inputs)]
    expected = leased.compute(block, projection, *inputs)
    with runner.acquire(*inputs) as lease:
        outputs = produce(lease, projection)
        assert all(
            torch.equal(actual.view(torch.uint8), wanted.view(torch.uint8))
            for actual, wanted in zip(outputs, expected, strict=True)
        )
        assert lease.consume() is outputs
        assert runner.stats["projection_copy_bytes"] == runner.stats["output_clone_bytes"] == 0
    assert all(torch.equal(a, b) for a, b in zip(before, (projection, *inputs), strict=True))


def test_two_slots_keep_both_active_outputs_and_fence_before_reuse(leased):
    _, runner, runtime, _ = controller(leased)
    projection, *inputs = operands()
    with ExitStack() as stack:
        left = stack.enter_context(runner.acquire(*inputs))
        first = produce(left, projection)
        retained = [value.clone() for value in first]
        right = stack.enter_context(runner.acquire(*inputs))
        second = produce(right, projection + 1)
        assert left.projection_written and right.projection_written
        assert {value.data_ptr() for value in first}.isdisjoint(value.data_ptr() for value in second)
        assert all(torch.equal(value, old) for value, old in zip(first, retained, strict=True))
        with pytest.raises(RuntimeError, match="Both leased graph slots"):
            with runner.acquire(*inputs):
                pass
        left.consume("consumer-A")
    runtime.log.clear()
    with runner.acquire(*inputs) as again:
        produce(again, projection + 2)
    waits = [item for item in runtime.log if item[0] == "wait"]
    assert waits and any(item[2][1] == "consumer-A" for item in waits)
    assert runtime.log[0][0] == "wait"
    assert runner.stats["captures"] == 2


def test_closed_lease_rejects_outputs_producer_and_consumer(leased):
    _, runner, _, _ = controller(leased)
    projection, *inputs = operands()
    with runner.acquire(*inputs) as lease:
        produce(lease, projection)
    for action in (lambda: lease.outputs, lambda: lease.projection_input, lambda: lease.consume()):
        with pytest.raises(RuntimeError, match="outside its consumer lifetime"):
            action()


def test_each_consumer_stream_waits_ready_once_then_has_release_fence(leased):
    _, runner, runtime, _ = controller(leased)
    projection, *inputs = operands()
    with runner.acquire(*inputs) as lease:
        produce(lease, projection)
        lease.consume("cross-attention")
        lease.consume("cross-attention")
        lease.consume("residual")
        slot = lease.slot
    assert [event[1] for event in slot.completion_events] == ["producer", "cross-attention", "residual"]
    assert len([item for item in runtime.log if item[0] == "wait"]) == 2


def test_projection_staging_is_measured_and_does_not_alias_source(leased):
    _, runner, _, _ = controller(leased)
    projection, *inputs = operands()
    with runner.acquire(*inputs) as lease:
        assert lease.projection_input.data_ptr() != projection.data_ptr()
        lease.stage_projection(projection)
        lease.finish()
    assert runner.stats["projection_copy_bytes"] == projection.numel() * projection.element_size()
    assert runner.stats["conditioning_copy_bytes"] == sum(value.numel() * value.element_size() for value in inputs)


def test_missing_producer_poison_retains_started_owners(leased):
    _, runner, _, failures = controller(leased)
    _, *inputs = operands()
    with pytest.raises(RuntimeError, match="without graph submission"):
        with runner.acquire(*inputs):
            pass
    assert runner.failed and runner.quarantine and len(failures) == 1
    with pytest.raises(RuntimeError, match="poisoned"):
        with runner.acquire(*inputs):
            pass


def test_capture_failure_never_falls_back_and_retains_active_slot(leased):
    _, runner, runtime, failures = controller(leased)
    projection, *inputs = operands()
    runtime.fail_capture = True
    with pytest.raises(RuntimeError, match="injected capture failure"):
        with runner.acquire(*inputs) as lease:
            produce(lease, projection)
    assert runner.failed and lease.slot.active and failures


@pytest.mark.parametrize("bad_field", ["gate_dtype", "camera_dtype", "shape", "runtime"])
def test_unsupported_inputs_fall_back_before_slot_allocation(leased, bad_field):
    _, runner, runtime, _ = controller(leased)
    projection, hidden, gate, scale, shift = operands()
    if bad_field == "gate_dtype":
        gate = gate.bfloat16()
    if bad_field == "camera_dtype":
        scale = scale.float()
    if bad_field == "shape":
        shift = shift[:, :1]
    if bad_field == "runtime":
        runtime.supported = False
    with runner.acquire(hidden, gate, scale, shift) as lease:
        assert lease is None
    assert not runner.arenas and not runtime.log and not runner.failed


def test_weight_mutation_is_rejected_before_slot_write(leased):
    block, runner, runtime, _ = controller(leased)
    _, *inputs = operands()
    with torch.no_grad():
        block.self_attn.o.weight.add_(1)
    with pytest.raises(RuntimeError, match="parameter owner changed"):
        with runner.acquire(*inputs):
            pass
    assert not runner.arenas and not runtime.log


def test_new_signature_limit_falls_back_without_evicting_live_arena(leased):
    _, runner, _, _ = controller(leased, max_signatures=1)
    projection, *inputs = operands()
    with runner.acquire(*inputs) as lease:
        produce(lease, projection)
    _, *different = operands(tokens=4)
    original = next(iter(runner.arenas.values()))
    with runner.acquire(*different) as lease:
        assert lease is None
    assert next(iter(runner.arenas.values())) is original


def test_stream_change_before_producer_submission_is_fatal(leased):
    _, runner, runtime, failures = controller(leased)
    projection, *inputs = operands()
    with pytest.raises(RuntimeError, match="acquired stream"):
        with runner.acquire(*inputs) as lease:
            lease.projection_input.copy_(projection)
            runtime.stream = "other-producer"
            lease.mark_projection_written()
    assert runner.failed and failures


def test_default_install_and_missing_controller_do_not_change_module(leased):
    block = SimpleNamespace()
    leased.install(SimpleNamespace(blocks=[block]))
    assert not hasattr(block, "_lingbot_leased_post_attention")
    _, hidden, gate, scale, shift = operands()
    with leased.acquire(block, hidden, gate, scale, shift) as lease:
        assert lease is None


def reverse_exchange_oracle(destination_rank):
    """Execute the real permutation against an eight-peer CPU collective stub."""
    tokens, heads, width = 3, 5, 4
    sources = [
        torch.arange(8 * tokens * heads * width, dtype=torch.int32).reshape(1, 8 * tokens, heads, width) + rank * 10000
        for rank in range(8)
    ]
    sends = [
        source.reshape(1, 8, tokens, heads, width)
        .transpose(0, 3)
        .transpose(0, 1)
        .contiguous()
        .reshape(8, heads, tokens, 1, width)
        for source in sources
    ]

    class Dist:
        def __init__(self):
            self.calls = 0

        def get_world_size(self, group):
            return 8

        def all_to_all_single(self, target, send, group):
            self.calls += 1
            assert torch.equal(send, sends[destination_rank])
            target.copy_(torch.stack([value[destination_rank] for value in sends]))

    dist = Dist()
    source = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/distributed/comm.py"
    functions: list[ast.stmt] = [
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef)
    ]
    namespace = dict(
        torch=torch,
        dist=dist,
        current_omni_platform=SimpleNamespace(synchronize=lambda: None),
        __name__="cpu_owned_reverse_exchange",
    )
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]), str(source), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    expected = torch.cat(
        [value[:, destination_rank * tokens : (destination_rank + 1) * tokens] for value in sources], dim=2
    )
    return namespace["all_to_all_4D"], sources, expected, dist


@pytest.mark.parametrize("rank", range(8))
def test_owned_reverse_exchange_preserves_head_token_order_and_canaries(rank):
    exchange, sources, expected, dist = reverse_exchange_oracle(rank)
    old_sources = [value.clone() for value in sources]
    owner = torch.full((expected.numel() + 64,), -777, dtype=torch.int32)
    target = owner[32:-32].reshape(expected.shape)
    result = exchange(sources[rank], 1, 2, group="ulysses", _output_destination=target)
    assert result is target and torch.equal(result, expected) and dist.calls == 1
    assert torch.all(owner[:32] == -777) and torch.all(owner[-32:] == -777)
    assert all(torch.equal(value, old) for value, old in zip(sources, old_sources, strict=True))
    native = exchange(sources[rank], 1, 2, group="ulysses")
    assert torch.equal(native, expected) and dist.calls == 2


@pytest.mark.parametrize("invalid", ["shape", "dtype", "stride", "source_alias", "forward_direction"])
def test_owned_reverse_exchange_rejects_invalid_destination_before_collective(invalid):
    exchange, sources, expected, dist = reverse_exchange_oracle(0)
    target = torch.empty_like(expected)
    scatter, gather = 1, 2
    if invalid == "shape":
        target = target[:, :-1]
    if invalid == "dtype":
        target = target.float()
    if invalid == "stride":
        target = torch.empty((*target.shape[:-1], 8), dtype=target.dtype)[..., ::2]
    if invalid == "source_alias":
        target = sources[0].reshape(expected.shape)
    if invalid == "forward_direction":
        scatter, gather = 2, 1
    with pytest.raises(ValueError, match="invalid before communication"):
        exchange(sources[0], scatter, gather, group="ulysses", _output_destination=target)
    assert dist.calls == 0


def test_owned_reverse_exchange_retains_packing_receive_and_target_after_collective_failure(monkeypatch):
    exchange, sources, expected, dist = reverse_exchange_oracle(0)
    target = torch.empty_like(expected)
    submitted: list[torch.Tensor] = []
    failures = []

    def failed_collective(output, input_, group):
        dist.calls += 1
        submitted.extend((input_, output))
        raise RuntimeError("injected failure after collective submission")

    dist.all_to_all_single = failed_collective
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.layers.lingbot_sp8_fatal",
        SimpleNamespace(report_failure=lambda *args: failures.append(args)),
    )
    with pytest.raises(RuntimeError, match="after collective submission"):
        exchange(sources[0], 1, 2, group="ulysses", _output_destination=target)
    assert dist.calls == 1 and len(failures) == 1
    owners = failures[0][2]
    assert all(any(owner is tensor for owner in owners) for tensor in (sources[0], *submitted, target))
