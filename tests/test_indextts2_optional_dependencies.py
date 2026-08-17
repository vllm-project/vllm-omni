# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import importlib.util
import platform
import sys
import types
from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).parents[1]
TEXT_PROCESSING_PATH = ROOT / "vllm_omni" / "model_executor" / "models" / "indextts2" / "text_processing_v2_5.py"
INDEXTTS2_FRONTEND_PACKAGES = {
    "fugashi",
    "unidic-lite",
    "wetext",
    "wetextprocessing",
}
INSTALL_COMMAND = "pip install 'vllm-omni[indextts2]'"


def _load_text_processing_module(monkeypatch: pytest.MonkeyPatch):
    package_name = "_indextts2_optional_dependencies_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(TEXT_PROCESSING_PATH.parent)]
    tokenizer = types.ModuleType(f"{package_name}.tokenizer_v2_5")
    tokenizer.INDEXTTS25_TOKENIZER_FILE = "tokenizer.tiktoken"
    tokenizer.encode_indextts25_text = lambda *args, **kwargs: []
    tokenizer.lang_to_token = lambda lang: 0
    tokenizer.normalize_language_code = lambda lang: lang
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, tokenizer.__name__, tokenizer)

    module_name = f"{package_name}.text_processing_v2_5"
    spec = importlib.util.spec_from_file_location(module_name, TEXT_PROCESSING_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _requirements_from_file(path: Path) -> list[Requirement]:
    requirements = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-r ", "--requirement ")):
            included = line.split(maxsplit=1)[1]
            requirements.extend(_requirements_from_file(path.parent / included))
            continue
        requirements.append(Requirement(line))
    return requirements


def _block_import(monkeypatch: pytest.MonkeyPatch, package: str) -> None:
    original_import = builtins.__import__

    def controlled_import(name, *args, **kwargs):
        if name == package or name.startswith(f"{package}."):
            raise ModuleNotFoundError(f"No module named {package!r}", name=package)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", controlled_import)


def test_indextts2_extra_declares_platform_specific_frontend_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = [Requirement(value) for value in project["project"]["optional-dependencies"]["indextts2"]]

    assert {
        (requirement.name.lower(), str(requirement.specifier), str(requirement.marker) if requirement.marker else None)
        for requirement in requirements
    } == {
        ("fugashi", ">=1.2.0", None),
        ("unidic-lite", ">=1.0.0", None),
        ("wetext", ">=0.0.9", 'sys_platform != "linux"'),
        ("wetextprocessing", "", 'sys_platform == "linux"'),
    }


def test_common_requirements_exclude_indextts2_frontend_dependencies():
    requirements = _requirements_from_file(ROOT / "requirements" / "common.txt")

    assert not ({requirement.name.lower() for requirement in requirements} & INDEXTTS2_FRONTEND_PACKAGES)


def test_wetext_normalizer_constructor_error_propagates(monkeypatch):
    text_processing_v2_5 = _load_text_processing_module(monkeypatch)
    sentinel = RuntimeError("wetext constructor sentinel")

    class FailingNormalizer:
        def __init__(self, *args, **kwargs):
            raise sentinel

    wetext = types.ModuleType("wetext")
    wetext.Normalizer = FailingNormalizer
    monkeypatch.setitem(sys.modules, "wetext", wetext)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    with pytest.raises(RuntimeError) as exc_info:
        text_processing_v2_5._load_wetext_normalizer("zh")

    assert exc_info.value is sentinel


def test_japanese_tagger_constructor_error_propagates(monkeypatch):
    text_processing_v2_5 = _load_text_processing_module(monkeypatch)
    sentinel = RuntimeError("fugashi constructor sentinel")

    fugashi = types.ModuleType("fugashi")

    def failing_tagger():
        raise sentinel

    fugashi.Tagger = failing_tagger
    monkeypatch.setitem(sys.modules, "fugashi", fugashi)
    monkeypatch.setitem(sys.modules, "unidic_lite", types.ModuleType("unidic_lite"))

    with pytest.raises(RuntimeError) as exc_info:
        text_processing_v2_5._load_japanese_tagger()

    assert exc_info.value is sentinel


def test_missing_wetext_backend_reports_indextts2_extra(monkeypatch):
    text_processing_v2_5 = _load_text_processing_module(monkeypatch)
    text_processing_v2_5._load_wetext_normalizer.cache_clear()
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    _block_import(monkeypatch, "wetext")

    with pytest.raises(RuntimeError, match="IndexTTS 2.5 text normalization") as exc_info:
        text_processing_v2_5._load_wetext_normalizer("zh")

    assert INSTALL_COMMAND in str(exc_info.value)
    text_processing_v2_5._load_wetext_normalizer.cache_clear()


def test_missing_japanese_tagger_reports_indextts2_extra(monkeypatch):
    text_processing_v2_5 = _load_text_processing_module(monkeypatch)
    text_processing_v2_5._load_japanese_tagger.cache_clear()
    _block_import(monkeypatch, "fugashi")

    with pytest.raises(RuntimeError, match="IndexTTS 2.5 Japanese preprocessing") as exc_info:
        text_processing_v2_5._load_japanese_tagger()

    assert INSTALL_COMMAND in str(exc_info.value)
    text_processing_v2_5._load_japanese_tagger.cache_clear()
