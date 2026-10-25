# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import builtins
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeMusa:
    def is_available(self):
        return True

    def device_count(self):
        return 2

    def get_device_name(self, index):
        return ["MTT S5000", "MTT S5000"][index]


class FakeTorchVersion:
    debug = False
    cuda = None
    hip = None
    musa = "5.2.0"


class FakeCuda:
    def is_available(self):
        return False


def load_collect_env(monkeypatch, musa=FakeMusa()):
    fake_vllm = ModuleType("vllm")
    fake_vllm_envs = ModuleType("vllm.envs")
    fake_vllm_envs.environment_variables = {}
    fake_vllm.envs = fake_vllm_envs

    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "2.7.0"
    fake_torch.version = FakeTorchVersion()
    fake_torch.cuda = FakeCuda()
    if musa is not None:
        fake_torch.musa = musa

    monkeypatch.setitem(sys.modules, "regex", re)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", fake_vllm_envs)
    monkeypatch.setitem(sys.modules, "vllm_omni", ModuleType("vllm_omni"))
    monkeypatch.delitem(sys.modules, "torch_musa", raising=False)

    module_name = "collect_env_under_test"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    path = Path(__file__).parents[1] / "collect_env.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_native_musa_availability_and_device_names(monkeypatch):
    collect_env = load_collect_env(monkeypatch)

    assert collect_env.get_musa_available() == "True"
    assert collect_env.get_musa_gpu_info() == "MTT S5000\nMTT S5000"


def test_cuda_is_not_used_as_a_musa_fallback(monkeypatch):
    collect_env = load_collect_env(monkeypatch, musa=None)

    assert collect_env._get_musa_module() is None
    assert collect_env.get_musa_available() == "N/A"
    assert collect_env.get_musa_gpu_info() is None


def test_musa_visible_devices_prefers_primary_variable(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0,2")
    monkeypatch.setenv("MTHREADS_VISIBLE_DEVICES", "1,3")

    assert collect_env.get_musa_visible_devices() == "MUSA_VISIBLE_DEVICES=0,2"


def test_musa_visible_devices_preserves_explicit_empty_value(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("MTHREADS_VISIBLE_DEVICES", "1,3")

    assert collect_env.get_musa_visible_devices() == "MUSA_VISIBLE_DEVICES="


def test_musa_visible_devices_falls_back_to_mthreads(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    monkeypatch.delenv("MUSA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("MTHREADS_VISIBLE_DEVICES", "1,2,3")

    assert collect_env.get_musa_visible_devices() == "MTHREADS_VISIBLE_DEVICES=1,2,3"


@pytest.mark.parametrize("label", ["package version", "driver version"])
def test_musa_driver_version_accepts_supported_labels(monkeypatch, label):
    collect_env = load_collect_env(monkeypatch)

    assert collect_env.get_musa_driver_version(lambda _: (0, f"{label}: 3.3.6-server", "")) == "3.3.6-server"


def test_musa_runtime_errors_do_not_abort_collection(monkeypatch):
    class BrokenMusa:
        def is_available(self):
            raise RuntimeError("runtime unavailable")

        def device_count(self):
            raise OSError("driver unavailable")

    collect_env = load_collect_env(monkeypatch, BrokenMusa())

    assert collect_env.get_musa_available() == "N/A"
    assert collect_env.get_musa_gpu_info() is None


def test_musa_import_attribute_error_does_not_abort_runtime_probes(monkeypatch):
    collect_env = load_collect_env(monkeypatch, musa=None)
    original_import = builtins.__import__

    def broken_torch_musa_import(name, *args, **kwargs):
        if name == "torch_musa":
            raise AttributeError("incompatible torch API")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_torch_musa_import)

    assert collect_env.get_musa_available() == "N/A"
    assert collect_env.get_musa_gpu_info() is None


def test_musa_package_version_uses_distribution_metadata(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    monkeypatch.setattr(collect_env.importlib_metadata, "version", lambda name: "2.11.0.post1+musa5.2.0")

    assert collect_env.get_musa_package_version() == "2.11.0.post1+musa5.2.0"


def test_pip_inventory_excludes_musa_specific_packages_by_default(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    package_output = "\n".join(
        (
            "torchada==1.0.0",
            "mthreads-ml-py==2.0.0",
            "unrelated-package==3.0.0",
        )
    )

    _, output = collect_env.get_pip_packages(lambda _: (0, package_output, ""))

    assert output == ""


def test_conda_inventory_excludes_musa_specific_packages_by_default(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    package_output = "\n".join(
        (
            "# packages in environment",
            "torchada                   1.0.0",
            "mthreads-ml-py             2.0.0",
            "unrelated-package          3.0.0",
        )
    )

    output = collect_env.get_conda_packages(lambda _: (0, package_output, ""))

    assert output == ""


def test_musa_package_version_falls_back_to_module_version(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    torch_musa = ModuleType("torch_musa")
    torch_musa.__version__ = "2.11.0.post1+musa5.2.0"
    monkeypatch.setitem(sys.modules, "torch_musa", torch_musa)

    def missing_distribution(_):
        raise collect_env.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(collect_env.importlib_metadata, "version", missing_distribution)

    assert collect_env.get_musa_package_version() == "2.11.0.post1+musa5.2.0"


def test_musa_import_attribute_error_does_not_abort_package_probe(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    original_import = builtins.__import__

    def missing_distribution(_):
        raise collect_env.importlib_metadata.PackageNotFoundError

    def broken_torch_musa_import(name, *args, **kwargs):
        if name == "torch_musa":
            raise AttributeError("incompatible torch API")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(collect_env.importlib_metadata, "version", missing_distribution)
    monkeypatch.setattr(builtins, "__import__", broken_torch_musa_import)

    assert collect_env.get_musa_package_version() is None


def test_additional_musa_packages_are_only_probed_after_musa_detection(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    requested = []

    def package_version(name):
        requested.append(name)
        return {"torchada": "1.0.0", "mthreads-ml-py": "2.0.0"}[name]

    monkeypatch.setattr(collect_env.importlib_metadata, "version", package_version)

    assert collect_env.get_musa_additional_package_versions(False) == (None, None)
    assert requested == []
    assert collect_env.get_musa_additional_package_versions(True) == ("1.0.0", "2.0.0")
    assert requested == ["torchada", "mthreads-ml-py"]


def test_system_env_appends_musa_fields(monkeypatch):
    collect_env = load_collect_env(monkeypatch)

    assert collect_env.SystemEnv._fields[-8:] == (
        "musa_available",
        "musa_runtime_version",
        "musa_package_version",
        "musa_driver_version",
        "musa_gpu_models",
        "musa_visible_devices",
        "torchada_package_version",
        "mthreads_ml_py_package_version",
    )


def test_system_env_accepts_legacy_positional_values(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    legacy_field_count = len(collect_env.SystemEnv._fields) - 8

    env = collect_env.SystemEnv(*(["legacy"] * legacy_field_count))

    assert env.musa_available == "N/A"
    assert env.musa_runtime_version is None
    assert env.musa_package_version is None
    assert env.musa_driver_version is None
    assert env.musa_gpu_models is None
    assert env.musa_visible_devices is None
    assert env.torchada_package_version is None
    assert env.mthreads_ml_py_package_version is None


def test_get_env_info_collects_musa_fields_once(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    values = {
        "get_musa_available": "True",
        "get_musa_package_version": "2.11.0.post1+musa5.2.0",
        "get_musa_driver_version": "3.3.6-server",
        "get_musa_gpu_info": "MTT S5000",
        "get_musa_visible_devices": "MTHREADS_VISIBLE_DEVICES=1,2,3",
    }
    calls = {name: 0 for name in values}
    collected_patterns = {}
    additional_package_calls = []

    for name, value in values.items():

        def probe(*_args, _name=name, _value=value):
            calls[_name] += 1
            return _value

        monkeypatch.setattr(collect_env, name, probe)

    def get_pip_packages(_run_lambda, patterns=None):
        collected_patterns["pip"] = patterns
        return "pip", ""

    def get_conda_packages(_run_lambda, patterns=None):
        collected_patterns["conda"] = patterns
        return ""

    monkeypatch.setattr(collect_env, "get_pip_packages", get_pip_packages)
    monkeypatch.setattr(collect_env, "get_conda_packages", get_conda_packages)

    def get_musa_additional_package_versions(musa_detected):
        additional_package_calls.append(musa_detected)
        return "1.0.0", "2.0.0"

    monkeypatch.setattr(collect_env, "get_musa_additional_package_versions", get_musa_additional_package_versions)

    simple_probes = {
        "get_rocm_version": None,
        "get_vllm_version": "dev",
        "get_vllm_omni_version": "dev",
        "summarize_vllm_build_flags": "flags",
        "get_gpu_topo": None,
        "get_python_platform": "platform",
        "get_running_cuda_version": None,
        "get_cuda_module_loading_config": "N/A",
        "get_gpu_info": None,
        "get_nvidia_driver_version": None,
        "get_cudnn_version": None,
        "get_os": "os",
        "get_libc_version": "libc",
        "get_gcc_version": None,
        "get_clang_version": None,
        "get_cmake_version": None,
        "get_cachingallocator_config": "",
        "is_xnnpack_available": "N/A",
        "get_cpu_info": "cpu",
        "get_env_vars": "",
    }
    for name, value in simple_probes.items():
        monkeypatch.setattr(collect_env, name, lambda *_args, _value=value: _value)

    env = collect_env.get_env_info()

    assert env.musa_available == "True"
    assert env.musa_runtime_version == "5.2.0"
    assert env.musa_package_version == "2.11.0.post1+musa5.2.0"
    assert env.musa_driver_version == "3.3.6-server"
    assert env.musa_gpu_models == "MTT S5000"
    assert env.musa_visible_devices == "MTHREADS_VISIBLE_DEVICES=1,2,3"
    assert env.torchada_package_version == "1.0.0"
    assert env.mthreads_ml_py_package_version == "2.0.0"
    assert calls == {name: 1 for name in values}
    assert additional_package_calls == [True]
    assert {"torchada", "mthreads-ml-py"}.isdisjoint(collected_patterns["pip"])
    assert {"torchada", "mthreads-ml-py"}.isdisjoint(collected_patterns["conda"])


def test_pretty_str_includes_labeled_musa_block(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    env = collect_env.SystemEnv(*(["value"] * len(collect_env.SystemEnv._fields)))._replace(
        musa_available="True",
        musa_runtime_version=None,
        musa_package_version="2.11.0.post1+musa5.2.0",
        musa_driver_version="3.3.6-server",
        musa_gpu_models="MTT S5000",
        musa_visible_devices="MTHREADS_VISIBLE_DEVICES=1,2,3",
        torchada_package_version="1.0.0",
        mthreads_ml_py_package_version="2.0.0",
    )

    output = collect_env.pretty_str(env)

    assert "MUSA available               : True" in output
    assert "MUSA runtime version         : None" in output
    assert "torch-musa package version   : 2.11.0.post1+musa5.2.0" in output
    assert "torchada package version     : 1.0.0" in output
    assert "mthreads-ml-py package version: 2.0.0" in output
    assert "MUSA driver version          : 3.3.6-server" in output
    assert "MUSA GPU models              : MTT S5000" in output
    assert "MUSA visible devices         : MTHREADS_VISIBLE_DEVICES=1,2,3" in output


def test_pretty_str_omits_musa_block_without_musa_signals(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    env = collect_env.SystemEnv(*(["value"] * len(collect_env.SystemEnv._fields)))._replace(
        musa_available="N/A",
        musa_runtime_version=None,
        musa_package_version=None,
        musa_driver_version=None,
        musa_gpu_models=None,
        musa_visible_devices=None,
        torchada_package_version=None,
        mthreads_ml_py_package_version=None,
    )

    output = collect_env.pretty_str(env)

    assert "MUSA Info" not in output
    assert "MUSA available" not in output
    assert ("Is XNNPACK available         : value\n\n==============================\n          CPU Info") in output


def test_pretty_str_includes_musa_block_for_partial_installation(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    env = collect_env.SystemEnv(*(["value"] * len(collect_env.SystemEnv._fields)))._replace(
        musa_available="N/A",
        musa_runtime_version=None,
        musa_package_version="2.11.0.post1+musa5.2.0",
        musa_driver_version=None,
        musa_gpu_models=None,
        musa_visible_devices=None,
        torchada_package_version=None,
        mthreads_ml_py_package_version=None,
    )

    output = collect_env.pretty_str(env)

    assert "MUSA Info" in output
    assert "MUSA available               : N/A" in output
    assert "torch-musa package version   : 2.11.0.post1+musa5.2.0" in output
    assert ("Is XNNPACK available         : value\n\n==============================\n         MUSA Info") in output


def test_environment_inventory_includes_musa_prefixes(monkeypatch):
    collect_env = load_collect_env(monkeypatch)
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0,2")
    monkeypatch.setenv("MTHREADS_VISIBLE_DEVICES", "1,3")
    monkeypatch.setenv("TORCHADA_HOME", "/opt/torchada")

    output = collect_env.get_env_vars()

    assert "MUSA_VISIBLE_DEVICES=0,2" in output
    assert "MTHREADS_VISIBLE_DEVICES=1,3" in output
    assert "TORCHADA_HOME=/opt/torchada" in output
