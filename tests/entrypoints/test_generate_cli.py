# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the `vllm generate --omni` CLI subcommand.

These tests load the production module directly from source while stubbing
its heavy dependencies, so parser and command assertions exercise the real
`OmniGenerateCommand` implementation.
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import PIL.Image
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATE_PATH = _REPO_ROOT / "vllm_omni" / "entrypoints" / "cli" / "generate.py"


def _stub_package(monkeypatch: pytest.MonkeyPatch, name: str) -> types.ModuleType:
    pkg = types.ModuleType(name)
    pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, pkg)
    return pkg


def _load_generate_module(monkeypatch: pytest.MonkeyPatch):
    for pkg_name in [
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.cli",
        "vllm.utils",
        "vllm_omni",
        "vllm_omni.entrypoints",
        "vllm_omni.entrypoints.cli",
    ]:
        _stub_package(monkeypatch, pkg_name)

    cli_types_mod = types.ModuleType("vllm.entrypoints.cli.types")

    class CLISubcommand:
        pass

    cli_types_mod.CLISubcommand = CLISubcommand
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.cli.types", cli_types_mod)

    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda _name: object()
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_mod)

    argparse_utils_mod = types.ModuleType("vllm.utils.argparse_utils")

    class FlexibleArgumentParser(argparse.ArgumentParser):
        pass

    argparse_utils_mod.FlexibleArgumentParser = FlexibleArgumentParser
    monkeypatch.setitem(sys.modules, "vllm.utils.argparse_utils", argparse_utils_mod)

    spec = importlib.util.spec_from_file_location(
        "vllm_omni.entrypoints.cli.generate",
        _GENERATE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _install_cmd_stubs(
    monkeypatch: pytest.MonkeyPatch,
    outputs: list[SimpleNamespace],
):
    for pkg_name in [
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.utils",
        "vllm_omni.entrypoints",
        "vllm_omni.inputs",
    ]:
        _stub_package(monkeypatch, pkg_name)

    torch_mod = types.ModuleType("torch")

    class Generator:
        def __init__(self, device: str):
            self.device = device
            self.seed = None

        def manual_seed(self, seed: int):
            self.seed = seed
            return self

    torch_mod.Generator = Generator
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    diffusion_data_mod = types.ModuleType("vllm_omni.diffusion.data")

    class DiffusionParallelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    diffusion_data_mod.DiffusionParallelConfig = DiffusionParallelConfig
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.data", diffusion_data_mod)

    omni_mod = types.ModuleType("vllm_omni.entrypoints.omni")

    class Omni:
        last_init_kwargs = None
        last_generate_args = None

        def __init__(self, **kwargs):
            type(self).last_init_kwargs = kwargs

        def generate(self, prompt_data, sampling_params):
            type(self).last_generate_args = (prompt_data, sampling_params)
            return outputs

    omni_mod.Omni = Omni
    monkeypatch.setitem(sys.modules, "vllm_omni.entrypoints.omni", omni_mod)

    inputs_data_mod = types.ModuleType("vllm_omni.inputs.data")

    class OmniDiffusionSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    inputs_data_mod.OmniDiffusionSamplingParams = OmniDiffusionSamplingParams
    monkeypatch.setitem(sys.modules, "vllm_omni.inputs.data", inputs_data_mod)

    platforms_mod = types.ModuleType("vllm_omni.platforms")
    platforms_mod.current_omni_platform = SimpleNamespace(device_type="cuda")
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms", platforms_mod)

    return Omni


def _make_args(output: str, num_images: int = 1) -> argparse.Namespace:
    return argparse.Namespace(
        model="Qwen/Qwen-Image",
        prompt="hello",
        negative_prompt=None,
        output=output,
        height=512,
        width=512,
        num_inference_steps=30,
        guidance_scale=4.0,
        cfg_scale=4.0,
        seed=123,
        num_images=num_images,
        tensor_parallel_size=2,
        stage_configs_path=None,
        enforce_eager=False,
        enable_cpu_offload=False,
    )


def _make_outputs(images: list[PIL.Image.Image]) -> list[SimpleNamespace]:
    return [SimpleNamespace(request_output=SimpleNamespace(images=images))]


@pytest.fixture
def generate_module(monkeypatch: pytest.MonkeyPatch):
    return _load_generate_module(monkeypatch)


def test_parser_registers_generate(generate_module):
    parser = generate_module.FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")
    generate_module.OmniGenerateCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "generate",
            "--model",
            "test-model",
            "--prompt",
            "hello",
            "--omni",
        ]
    )

    assert args.model == "test-model"
    assert args.prompt == "hello"
    assert args.omni is True


def test_default_args(generate_module):
    parser = generate_module.FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")
    generate_module.OmniGenerateCommand().subparser_init(subparsers)

    args = parser.parse_args(["generate", "--model", "m", "--prompt", "p"])

    assert args.height == 1024
    assert args.width == 1024
    assert args.num_inference_steps == 50
    assert args.guidance_scale == 4.0
    assert args.cfg_scale == 4.0
    assert args.seed == 42
    assert args.num_images == 1
    assert args.output == "output.png"


def test_single_image_output_without_suffix(generate_module, monkeypatch, tmp_path):
    img = PIL.Image.new("RGB", (64, 64), "blue")
    omni_cls = _install_cmd_stubs(monkeypatch, _make_outputs([img]))

    args = _make_args(str(tmp_path / "result"))
    generate_module.OmniGenerateCommand.cmd(args)

    assert (tmp_path / "result.png").exists()
    assert omni_cls.last_init_kwargs["mode"] == "text-to-image"
    assert omni_cls.last_init_kwargs["parallel_config"].tensor_parallel_size == 2
    assert omni_cls.last_generate_args[0]["prompt"] == "hello"
    assert omni_cls.last_generate_args[1].num_outputs_per_prompt == 1


def test_multi_image_output_without_suffix(generate_module, monkeypatch, tmp_path):
    imgs = [PIL.Image.new("RGB", (64, 64), color) for color in ["red", "green", "blue"]]
    _install_cmd_stubs(monkeypatch, _make_outputs(imgs))

    args = _make_args(str(tmp_path / "result"), num_images=3)
    generate_module.OmniGenerateCommand.cmd(args)

    assert (tmp_path / "result_0.png").exists()
    assert (tmp_path / "result_1.png").exists()
    assert (tmp_path / "result_2.png").exists()


def test_no_output_raises(generate_module, monkeypatch, tmp_path):
    _install_cmd_stubs(monkeypatch, [])

    with pytest.raises(RuntimeError, match="No output generated"):
        generate_module.OmniGenerateCommand.cmd(_make_args(str(tmp_path / "out.png")))


def test_no_images_raises(generate_module, monkeypatch, tmp_path):
    _install_cmd_stubs(monkeypatch, [SimpleNamespace(request_output=SimpleNamespace(images=[]))])

    with pytest.raises(RuntimeError, match="No images in output"):
        generate_module.OmniGenerateCommand.cmd(_make_args(str(tmp_path / "out.png")))
