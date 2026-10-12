# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.lychee_fd.token2wav import LycheeToken2WavCore
from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.flashcosyvoice.modules import hifigan
from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.flashcosyvoice.modules.hifigan_components import (
    f0_math as math,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class OriginalF0(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_class = 1
        layers: list[nn.Module] = []
        for i in range(5):
            layers.extend((hifigan.weight_norm(hifigan.Conv1d(80 if i == 0 else 512, 512, 3, padding=1)), nn.ELU()))
        self.condnet = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        return self.classifier(self.condnet(x).transpose(1, 2)).squeeze(-1).abs()


def test_real_constructor_state_weight_norm_future_rng_and_cpu_math():
    torch.manual_seed(37)
    expected = OriginalF0().eval()
    future_expected = torch.rand(8)
    torch.manual_seed(37)
    actual = hifigan.ConvRNNF0Predictor().eval()
    future_actual = torch.rand(8)
    torch.testing.assert_close(future_actual, future_expected, rtol=0, atol=0)
    assert expected.state_dict().keys() == actual.state_dict().keys() and len(actual.state_dict()) == 17
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
    for i in range(0, 10, 2):
        assert hasattr(actual.condnet[i], "parametrizations")
        assert "Conv" in actual.condnet[i].__class__.__name__
        torch.testing.assert_close(actual.condnet[i].weight, expected.condnet[i].weight, rtol=0, atol=0)
    duplicate = deepcopy(actual)
    assert all(not child._f0_plans for child in duplicate.modules() if isinstance(child, math.F0Conv1d))
    inputs = torch.randn(1, 80, 14)
    with torch.inference_mode():
        torch.testing.assert_close(actual(inputs), expected(inputs), rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_gradient_training_default_padding_group_dtype_semantics():
    torch.manual_seed(41)
    expected = hifigan.Conv1d(4, 6, 3, padding=1, groups=2).double()
    actual = math.F0Conv1d(4, 6, 3, padding=1, groups=2).double()
    actual.load_state_dict(expected.state_dict())
    a = torch.randn(2, 4, 9, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    torch.testing.assert_close(actual(b), expected(a), rtol=0, atol=0)
    actual(b).sum().backward()
    expected(a).sum().backward()
    torch.testing.assert_close(b.grad, a.grad, rtol=0, atol=0)
    torch.testing.assert_close(actual.weight.grad, expected.weight.grad, rtol=0, atol=0)
    elu = math.F0ELU(alpha=2).eval()
    torch.testing.assert_close(elu(a), nn.ELU(alpha=2)(a), rtol=0, atol=0)


def test_eval_gradient_requirements_and_qualified_shape_predicates(monkeypatch):
    monkeypatch.setattr(math, "_qualified_runtime", lambda _x: True)
    model = math.F0Conv1d(80, 512, 3, padding=1).eval()
    x = torch.ones(1, 80, 14)
    assert not model._eligible(x, model.weight, model.bias)
    with torch.no_grad():
        assert model._eligible(x, model.weight, model.bias)
        assert not model._eligible(torch.ones(2, 80, 14), model.weight, model.bias)
        assert not model._eligible(torch.ones(1, 80, 13), model.weight, model.bias)
        model.train()
        assert not model._eligible(x, model.weight, model.bias)
    model.eval().requires_grad_(False)
    assert model._eligible(x, model.weight, model.bias)
    assert not model._eligible(x.requires_grad_(), model.weight, model.bias)


def test_cpu_and_profile_getters_do_not_initialize_cuda(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("CUDA initialized in CPU path")

    for name in ("init", "_lazy_init", "get_device_capability"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    assert not math._qualified_runtime(torch.ones(1, 512, 14))
    math.F0Conv1d(2, 2, 3).to("cpu")
    assert not torch.cuda.is_initialized()


class FakePlan:
    instances: list["FakePlan"] = []
    bytes_each = 1024**2
    fail_execute = False
    fail_allocate = False

    def __init__(self, shape, weight_shape, device):
        self.shape, self.weight_shape, self.device = shape, weight_shape, device
        self.workspace_bytes, self.closed, self.calls = self.bytes_each, 0, 0
        self.instances.append(self)

    def allocate_workspace(self):
        if self.fail_allocate:
            raise RuntimeError("workspace allocation failure")

    def execute(self, x, w, bias):
        self.calls += 1
        if self.fail_execute:
            raise RuntimeError("execution failure")
        return torch.nn.functional.conv1d(x, w, bias, padding=1)

    def close(self):
        self.closed += 1


@pytest.fixture
def fake_cache(monkeypatch):
    FakePlan.instances = []
    FakePlan.bytes_each = 1024**2
    FakePlan.fail_execute = False
    FakePlan.fail_allocate = False
    monkeypatch.setattr(math, "_F0Plan", FakePlan)
    monkeypatch.setattr(torch.accelerator, "device_index", lambda _device: nullcontext())
    monkeypatch.setattr(math.F0Conv1d, "_eligible", lambda *_args: True)
    return math.F0Conv1d(80, 512, 3, padding=1).eval()


def exercise(model, length):
    with torch.no_grad():
        return model(torch.ones(1, 80, length))


def test_lru_four_reuses_normal_response_three_shapes_and_evicts(fake_cache):
    model = fake_cache
    for length in (50, 58, 32, 50, 58, 32):
        exercise(model, length)
    assert len(FakePlan.instances) == 3 and all(p.calls == 2 for p in FakePlan.instances)
    exercise(model, 14)
    exercise(model, 20)
    assert len(model._f0_plans) == 4 and FakePlan.instances[0].closed == 1
    model.clear_released_plans()
    model.clear_released_plans()
    assert not model._f0_plans and all(p.closed == 1 for p in FakePlan.instances)


def test_total_cached_workspace_is_bounded_before_allocation(fake_cache):
    FakePlan.bytes_each = 6 * 1024**2
    for length in (50, 58, 32, 14):
        exercise(fake_cache, length)
    assert len(fake_cache._f0_plans) == 2
    assert sum(p.workspace_bytes for p in fake_cache._f0_plans.values()) <= math.MAX_WORKSPACE_BYTES
    assert all(p.closed == 1 for p in FakePlan.instances[:-2])


@pytest.mark.parametrize("failure", ["fail_allocate", "fail_execute"])
def test_cached_failure_cleans_plan_and_raises_without_delegate(fake_cache, failure):
    setattr(FakePlan, failure, True)
    with pytest.raises(RuntimeError):
        exercise(fake_cache, 50)
    assert not fake_cache._f0_plans and FakePlan.instances[0].closed == 1


def test_module_apply_and_shutdown_cleanup_and_copy_excludes_cache(fake_cache):
    exercise(fake_cache, 50)
    copy = deepcopy(fake_cache)
    assert not copy._f0_plans and not any(key.startswith("_f0") for key in copy.state_dict())
    fake_cache.double()
    assert FakePlan.instances[0].closed == 1 and not fake_cache._f0_plans
    f0 = hifigan.ConvRNNF0Predictor()
    f0.condnet[0] = fake_cache
    exercise(fake_cache.float(), 50)
    hift = hifigan.HiFTGenerator.__new__(hifigan.HiFTGenerator)
    nn.Module.__init__(hift)
    hift.f0_predictor = f0
    core = LycheeToken2WavCore.__new__(LycheeToken2WavCore)
    nn.Module.__init__(core)
    core.hift = hift
    core.shutdown()
    core.shutdown()
    assert not fake_cache._f0_plans and FakePlan.instances[-1].closed == 1


class FakeAPI:
    def __init__(self, workspace=0, fail_attribute=None):
        self.workspace, self.fail_attribute = workspace, fail_attribute
        self.created, self.destroyed, self.attributes, self.events = [], [], [], []
        self.functions = {}

    def __getattr__(self, name):
        if name in self.functions:
            return self.functions[name]

        def call(*args):
            if name == "cudnnGetVersion":
                return 92000
            if name == "cudnnGetErrorString":
                return b"injected error"
            if name == "cudnnCreate":
                args[0]._obj.value = 900
            elif name == "cudnnDestroy":
                self.events.append("destroy_handle")
            elif name == "cudnnBackendCreateDescriptor":
                value = 100 + len(self.created)
                args[1]._obj.value = value
                self.created.append(value)
            elif name == "cudnnBackendDestroyDescriptor":
                self.destroyed.append(args[0].value)
                self.events.append("destroy_desc")
            elif name == "cudnnBackendSetAttribute":
                self.attributes.append((args[1], list(args[4])))
                if args[1] == self.fail_attribute:
                    return 1
            elif name == "cudnnBackendGetAttribute":
                args[4]._obj.value = 1
                args[5]._obj.value = self.workspace
            elif name == "cudnnBackendExecute":
                self.events.append("execute")
            return 0

        self.functions[name] = call
        return call


def fake_library(monkeypatch, library):
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *a, **kw: (
            "0 /torch/libcudnn.so.9\n" if str(self) == "/proc/self/maps" else original(self, *a, **kw)
        ),
    )
    monkeypatch.setattr(math.C, "CDLL", lambda _path: library)


@pytest.mark.parametrize("channels", [80, 512])
@pytest.mark.parametrize("length", sorted(math.QUALIFIED_LENGTHS))
def test_real_qualified_descriptor_dimensions_and_cleanup(monkeypatch, channels, length):
    lib = FakeAPI()
    fake_library(monkeypatch, lib)
    plan = math._F0Plan((1, channels, length), (512, channels, 3), "cuda:0")
    assert [v for a, v in lib.attributes if a == 902] == [
        [1, channels, 1, length],
        [512, channels, 1, 3],
        [1, 512, 1, length],
    ]
    assert [v for a, v in lib.attributes if a == 901] == [[0], [0], [0]]
    assert [v for a, v in lib.attributes if a == 1301] == [[38]]
    plan.close()
    plan.close()
    assert sorted(lib.created) == sorted(lib.destroyed)
    assert lib.events.count("destroy_handle") == 1


@pytest.mark.parametrize("kind", ["workspace", "attribute"])
def test_partial_build_failure_cleans_owned_api_resources(monkeypatch, kind):
    lib = FakeAPI(
        workspace=math.MAX_WORKSPACE_BYTES + 1 if kind == "workspace" else 0,
        fail_attribute=1301 if kind == "attribute" else None,
    )
    fake_library(monkeypatch, lib)
    with pytest.raises(RuntimeError):
        math._F0Plan((1, 80, 50), (512, 80, 3), "cuda:0")
    assert sorted(lib.created) == sorted(lib.destroyed) and lib.events.count("destroy_handle") == 1


@pytest.mark.parametrize("event_failure", [None, "create", "record"])
def test_cross_stream_wait_record_and_owned_event_cleanup(monkeypatch, event_failure):
    lib = FakeAPI()
    fake_library(monkeypatch, lib)
    plan = math._F0Plan((1, 80, 50), (512, 80, 3), "cuda:0")

    class Stream:
        cuda_stream = 7

        def wait_event(self, event):
            lib.events.append("wait_previous")

        def synchronize(self):
            lib.events.append("synchronize_stream")

    stream = Stream()

    class Event:
        def __init__(self):
            if event_failure == "create":
                raise RuntimeError("event creation failed")

        def record(self, owner):
            assert owner is stream
            lib.events.append("record_current")
            if event_failure == "record":
                raise RuntimeError("event record failed")

        def synchronize(self):
            lib.events.append("synchronize_owned")

    class Tensor:
        dtype = torch.float32
        is_cuda = True
        device = torch.device("cuda:0")

        def __init__(self, shape):
            self.shape = shape

        def contiguous(self):
            return self

        def record_stream(self, owner):
            assert owner is stream
            lib.events.append("record_tensor")

        def data_ptr(self):
            return 1024

        def reshape(self, *shape):
            return Tensor(shape)

        def __add__(self, _other):
            return self

    plan.workspace = Tensor((0,))
    plan.last_event = SimpleNamespace(synchronize=lambda: lib.events.append("synchronize_owned"))
    monkeypatch.setattr(torch.accelerator, "device_index", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch, "empty", lambda shape, **_kw: Tensor(shape))
    if event_failure is None:
        plan.execute(Tensor((1, 80, 50)), Tensor((512, 80, 3)), Tensor((512,)))
        assert lib.events.index("wait_previous") < lib.events.index("execute") < lib.events.index("record_current")
    else:
        with pytest.raises(RuntimeError, match="event"):
            plan.execute(Tensor((1, 80, 50)), Tensor((512, 80, 3)), Tensor((512,)))
    assert lib.events.count("record_tensor") == 5
    plan.close()
    fence = "synchronize_stream" if event_failure else "synchronize_owned"
    assert lib.events.index(fence) < lib.events.index("destroy_handle")
    assert sorted(lib.created) == sorted(lib.destroyed)


def test_f0_shutdown_attempts_all_owners_after_first_cleanup_failure(monkeypatch):
    model = hifigan.ConvRNNF0Predictor()
    calls = []

    def first():
        calls.append(0)
        raise RuntimeError("first owner cleanup failed")

    monkeypatch.setattr(model.condnet[0], "clear_released_plans", first)
    for i in range(2, 10, 2):
        monkeypatch.setattr(model.condnet[i], "clear_released_plans", lambda i=i: calls.append(i))
    with pytest.raises(RuntimeError, match="first owner"):
        model.shutdown()
    assert calls == [0, 2, 4, 6, 8]


def test_f0_libdevice_is_fixed_and_missing_file_is_explicit(monkeypatch):
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-13.2")
    monkeypatch.setenv("LYCHEE_RELEASED_LIBDEVICE_PATH", "/ar/override.bc")
    monkeypatch.setattr(Path, "is_file", lambda path: path == math.F0_LIBDEVICE)
    assert math._released_f0_libdevice_path() == "/usr/local/cuda-12.9/nvvm/libdevice/libdevice.10.bc"
    monkeypatch.setattr(Path, "is_file", lambda _path: False)
    with pytest.raises(RuntimeError, match="fixed CUDA12.9"):
        math._released_f0_libdevice_path()


def test_cuda_autocast_configuration_delegates_before_runtime_device_calls(monkeypatch):
    x = SimpleNamespace(is_cuda=True, dtype=torch.float32, device=torch.device("cuda:0"))
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda _device: True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Unqualified autocast path inspected CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    assert not math._qualified_runtime(x)


def test_f0_elu_missing_default_libdevice_uses_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(math, "_qualified_runtime", lambda inputs: True)
    monkeypatch.setattr(math, "F0_LIBDEVICE", tmp_path / "missing-libdevice.bc")
    module = math.F0ELU().eval()
    inputs = torch.randn(1, 512, 14)
    with torch.no_grad():
        actual = module(inputs)
    torch.testing.assert_close(actual, nn.ELU()(inputs))


def test_f0_elu_available_matching_toolchain_keeps_released_kernel(monkeypatch, tmp_path):
    monkeypatch.setattr(math, "_qualified_runtime", lambda inputs: True)
    path = tmp_path / "libdevice.10.bc"
    path.write_text("CPU dispatch fixture; never compiled")
    monkeypatch.setattr(math, "F0_LIBDEVICE", path)
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(inputs, output, count, **kwargs):
                calls.append((grid, count, kwargs))
                output.copy_(nn.ELU()(inputs))

            return launch

    monkeypatch.setattr(math, "_elu_kernel", Kernel())
    module = math.F0ELU().eval()
    inputs = torch.randn(1, 512, 14)
    with torch.no_grad():
        actual = module(inputs)
    torch.testing.assert_close(actual, nn.ELU()(inputs))
    assert len(calls) == 1
    assert calls[0][2]["extern_libs"] == {"libdevice": str(path)}
    assert calls[0][2]["enable_fp_fusion"] is False
