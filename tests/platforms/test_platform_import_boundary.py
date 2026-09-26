# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Import-boundary tests for the in-tree platform ownership rule (RFC #6691).

Two rules, checked statically over the ``vllm_omni`` source tree:

* **Neutral code stays neutral.** A module outside ``vllm_omni/platforms/<backend>/``
  never imports a concrete ``vllm_omni.platforms.<backend>`` module, at any nesting
  depth. Neutral code selects backend behavior through ``current_omni_platform``,
  a semantic capability, or a registry / qualified-name hook owned by the platform.
* **Platform classes import lazily.** Importing ``vllm_omni.platforms.<backend>`` or
  its ``platform`` module must not eagerly pull in model, layer, operator,
  attention, quantization, or connector implementation bodies, whether they live
  under the platform tree or with a consumer subsystem. A platform hook may still
  import such code inside a function when it registers or selects it.

Existing violations are recorded in exact, named allowlists below. An allowlist
can only shrink: a new violation fails the test, and so does an entry whose debt
has already been paid. Each entry names the RFC #6691 migration item that
retires it.

The subprocess test at the end covers the runtime side of the same contract:
the platform detector and neutral connector code import with every optional
hardware SDK blocked.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "vllm_omni"
PLATFORMS_ROOT = SRC_ROOT / "platforms"
PLATFORMS_PACKAGE = "vllm_omni.platforms"

# Backends are the directories under ``vllm_omni/platforms/`` that ship a ``platform.py``.
BACKENDS: tuple[str, ...] = tuple(
    sorted(entry.name for entry in PLATFORMS_ROOT.iterdir() if entry.is_dir() and (entry / "platform.py").is_file())
)

# Implementation trees under ``platforms/<backend>/`` that RFC #6691 R4 names as
# migration debt: model patches, layers, quantization, and connector bodies.
PLATFORM_IMPLEMENTATION_SUBTREES: tuple[str, ...] = ("models", "layers", "quant", "omni_connectors")

# Consumer-owned implementation subsystems a platform class must not import eagerly.
CONSUMER_IMPLEMENTATION_PREFIXES: tuple[str, ...] = (
    "vllm_omni.model_executor.models",
    "vllm_omni.model_executor.layers",
    "vllm_omni.diffusion.models",
    "vllm_omni.diffusion.layers",
    "vllm_omni.diffusion.attention.backends",
    "vllm_omni.quantization",
    "vllm_omni.diffusion.quantization",
    "vllm_omni.distributed.omni_connectors",
)

# Registries a platform class may import eagerly: they carry names, not implementations.
ACCEPTED_REGISTRY_MODULES: frozenset[str] = frozenset({"vllm_omni.diffusion.attention.backends.registry"})

# Rule A debt: (neutral module, concrete platform module it imports) -> RFC #6691 item.
NEUTRAL_IMPORT_ALLOWLIST: dict[tuple[str, str], str] = {
    (
        "vllm_omni.diffusion.attention.backends.flash_attn",
        "vllm_omni.platforms.npu.quant.kv_quant_npu",
    ): "NPU group 2: move kv_quant_npu (fused attention, not checkpoint quantization) to the NPU attention backend owner",
    (
        "vllm_omni.distributed.omni_connectors",
        "vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector",
    ): "NPU group 3 / rollout step 9: make distributed/omni_connectors the canonical Yuanrong connector home",
    (
        "vllm_omni.distributed.omni_connectors.connectors.yuanrong_transfer_engine_connector",
        "vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector",
    ): "NPU group 3 / rollout step 9: this re-export points the wrong way (consumer path -> platform path)",
    (
        "vllm_omni.distributed.omni_connectors.factory",
        "vllm_omni.platforms.npu.omni_connectors",
    ): "NPU group 3 / rollout step 9: the connector factory should resolve the consumer-owned module",
    (
        "vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav",
        "vllm_omni.platforms.npu.models.step_audio2_token2wav",
    ): "NPU group 1: move the StepAudio2/CosyVoice NPU adaptation next to the StepAudio2 consumer package",
    (
        "vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec",
        "vllm_omni.platforms.npu.models.moss_tts_streaming_decode_wrapper",
    ): "NPU group 1 (added after the RFC inventory): move the MOSS-TTS NPU graph wrapper to the moss_tts package",
    (
        "vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav",
        "vllm_omni.platforms.npu.models.step_audio2_token2wav",
    ): "NPU group 1: move the StepAudio2 token2wav NPU patch body to the StepAudio2 consumer package",
    (
        "vllm_omni.platforms",
        "vllm_omni.platforms.xpu",
    ): "XPU housekeeping: the detector sets XPUOmniPlatform.dist_backend during detection instead of in the platform's own setup",
}

# Rule B debt: (backend, importing module, eagerly imported implementation module) -> RFC #6691 item.
PLATFORM_EAGER_IMPORT_ALLOWLIST: dict[tuple[str, str, str], str] = {}


# ---------------------------------------------------------------------------
# Static import collection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportSite:
    module: str
    target: str
    lineno: int
    eager: bool  # executed when the module is imported (module/class level, not inside a function)


def _module_name(path: Path) -> tuple[str, bool]:
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _module_path(module: str) -> Path | None:
    base = REPO_ROOT.joinpath(*module.split("."))
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _resolve_from_import(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = module.split(".") if is_package else module.split(".")[:-1]
    if node.level > 1:
        package_parts = package_parts[: len(package_parts) - (node.level - 1)]
    base = ".".join(package_parts)
    if node.module:
        return f"{base}.{node.module}" if base else node.module
    return base


def _walk_imports(body: list[ast.stmt], module: str, is_package: bool, eager: bool) -> Iterator[ImportSite]:
    for node in body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield ImportSite(module, alias.name, node.lineno, eager)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_import(node, module, is_package)
            for alias in node.names:
                # ``from pkg import name`` addresses the submodule ``pkg.name`` when
                # one exists on disk, otherwise an attribute of ``pkg``.
                submodule = f"{base}.{alias.name}" if base else alias.name
                target = submodule if alias.name != "*" and _module_path(submodule) is not None else base
                yield ImportSite(module, target, node.lineno, eager)
        elif isinstance(node, ast.If):
            if not _is_type_checking_guard(node):
                yield from _walk_imports(node.body, module, is_package, eager)
            yield from _walk_imports(node.orelse, module, is_package, eager)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
            yield from _walk_imports(node.body, module, is_package, eager)
            yield from _walk_imports(getattr(node, "orelse", []), module, is_package, eager)
        elif isinstance(node, ast.Try):
            yield from _walk_imports(node.body, module, is_package, eager)
            for handler in node.handlers:
                yield from _walk_imports(handler.body, module, is_package, eager)
            yield from _walk_imports(node.orelse, module, is_package, eager)
            yield from _walk_imports(node.finalbody, module, is_package, eager)
        elif isinstance(node, ast.Match):
            for case in node.cases:
                yield from _walk_imports(case.body, module, is_package, eager)
        elif isinstance(node, ast.ClassDef):
            yield from _walk_imports(node.body, module, is_package, eager)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from _walk_imports(node.body, module, is_package, eager=False)


@cache
def _imports_of(path: Path) -> tuple[ImportSite, ...]:
    module, is_package = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(_walk_imports(tree.body, module, is_package, eager=True))


def _source_files() -> list[Path]:
    return sorted(path for path in SRC_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _is_under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _concrete_platform_module(target: str) -> str | None:
    """Return *target* when it addresses a concrete ``vllm_omni.platforms.<backend>`` module."""
    for backend in BACKENDS:
        if _is_under(target, f"{PLATFORMS_PACKAGE}.{backend}"):
            return target
    return None


def _is_neutral(path: Path) -> bool:
    """Everything outside ``vllm_omni/platforms/<backend>/`` is neutral, including the detector."""
    try:
        relative = path.relative_to(PLATFORMS_ROOT)
    except ValueError:
        return True
    return relative.parts[0] not in BACKENDS


def _is_platform_implementation(module: str) -> bool:
    for backend in BACKENDS:
        for subtree in PLATFORM_IMPLEMENTATION_SUBTREES:
            if _is_under(module, f"{PLATFORMS_PACKAGE}.{backend}.{subtree}"):
                return True
    return False


def _is_consumer_implementation(module: str) -> bool:
    if module in ACCEPTED_REGISTRY_MODULES:
        return False
    return any(_is_under(module, prefix) for prefix in CONSUMER_IMPLEMENTATION_PREFIXES)


def _format_sites(sites: list[tuple[ImportSite, str]]) -> str:
    return "\n".join(f"  {site.module} -> {target} (line {site.lineno})" for site, target in sites)


# ---------------------------------------------------------------------------
# Rule A: neutral code never imports a concrete platform module
# ---------------------------------------------------------------------------


def _neutral_violations() -> dict[tuple[str, str], list[ImportSite]]:
    found: dict[tuple[str, str], list[ImportSite]] = {}
    for path in _source_files():
        if not _is_neutral(path):
            continue
        for site in _imports_of(path):
            target = _concrete_platform_module(site.target)
            if target is None:
                continue
            # ``from a.b import c`` yields ``a.b`` and ``a.b.c``; keep the longest
            # candidate that is a real module so the report names the module.
            if _module_path(target) is None:
                continue
            found.setdefault((site.module, target), []).append(site)
    return found


def test_backends_are_discovered() -> None:
    assert "cuda" in BACKENDS, BACKENDS
    assert all((PLATFORMS_ROOT / backend / "__init__.py").is_file() for backend in BACKENDS)


def test_neutral_modules_do_not_import_concrete_platforms() -> None:
    found = _neutral_violations()
    unexpected = sorted(key for key in found if key not in NEUTRAL_IMPORT_ALLOWLIST)
    stale = sorted(key for key in NEUTRAL_IMPORT_ALLOWLIST if key not in found)

    problems: list[str] = []
    if unexpected:
        sites = [(site, target) for module, target in unexpected for site in found[(module, target)]]
        problems.append(
            "Neutral modules import concrete platform modules (RFC #6691 R2). Route the call "
            "through current_omni_platform, a semantic capability, or a platform-owned hook that "
            "returns a qualified name. Only tracked migration debt may be added to "
            "NEUTRAL_IMPORT_ALLOWLIST, with the RFC item that retires it:\n" + _format_sites(sites)
        )
    if stale:
        problems.append(
            "NEUTRAL_IMPORT_ALLOWLIST entries no longer match an import; remove them so the list keeps shrinking:\n"
            + "\n".join(f"  {module} -> {target}" for module, target in stale)
        )
    assert not problems, "\n\n".join(problems)


def test_neutral_import_allowlist_points_at_existing_modules() -> None:
    for module, target in NEUTRAL_IMPORT_ALLOWLIST:
        assert _module_path(module) is not None, module
        assert _module_path(target) is not None, target
        assert _concrete_platform_module(target) is not None, target


# ---------------------------------------------------------------------------
# Rule B: importing a platform class does not eagerly load implementation bodies
# ---------------------------------------------------------------------------


def _eager_platform_closure(backend: str) -> dict[tuple[str, str], list[ImportSite]]:
    """Implementation modules reached by eager imports from the backend package and its ``platform`` module.

    Eager edges into other ``vllm_omni.platforms`` modules are followed (thin
    hooks are platform-owned); edges into neutral modules are only inspected.
    """
    roots = [f"{PLATFORMS_PACKAGE}.{backend}", f"{PLATFORMS_PACKAGE}.{backend}.platform"]
    queue = [module for module in roots if _module_path(module) is not None]
    assert queue == roots, f"{backend} is missing __init__.py or platform.py"
    seen: set[str] = set()
    found: dict[tuple[str, str], list[ImportSite]] = {}
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        path = _module_path(module)
        if path is None:
            continue
        for site in _imports_of(path):
            if not site.eager:
                continue
            target = site.target
            if _is_platform_implementation(target) or _is_consumer_implementation(target):
                if _module_path(target) is None:
                    continue  # attribute candidate, the module candidate is reported separately
                found.setdefault((module, target), []).append(site)
            elif _is_under(target, PLATFORMS_PACKAGE) and _module_path(target) is not None:
                queue.append(target)
    return found


@pytest.mark.parametrize("backend", BACKENDS)
def test_platform_class_does_not_eagerly_import_implementation_bodies(backend: str) -> None:
    found = _eager_platform_closure(backend)
    unexpected = sorted(
        (backend, module, target)
        for module, target in found
        if (backend, module, target) not in PLATFORM_EAGER_IMPORT_ALLOWLIST
    )
    stale = sorted(
        key for key in PLATFORM_EAGER_IMPORT_ALLOWLIST if key[0] == backend and (key[1], key[2]) not in found
    )

    problems: list[str] = []
    if unexpected:
        sites = [(site, target) for _, module, target in unexpected for site in found[(module, target)]]
        problems.append(
            f"Importing the {backend} platform class eagerly loads implementation bodies (RFC #6691 R4). "
            "Import them inside the hook that registers or selects them, or move the body to its "
            "consumer owner. Only tracked migration debt may be added to PLATFORM_EAGER_IMPORT_ALLOWLIST:\n"
            + _format_sites(sites)
        )
    if stale:
        problems.append(
            "PLATFORM_EAGER_IMPORT_ALLOWLIST entries no longer match an import; remove them:\n"
            + "\n".join(f"  {module} -> {target}" for _, module, target in stale)
        )
    assert not problems, "\n\n".join(problems)


# ---------------------------------------------------------------------------
# Runtime side: detector and neutral code import with every optional SDK blocked
# ---------------------------------------------------------------------------

OPTIONAL_HARDWARE_SDKS: tuple[str, ...] = (
    "torch_npu",
    "vllm_ascend",
    "aiter",
    "yr",
    "torchada",
    "vllm_musa",
    "amdsmi",
)

NEUTRAL_IMPORT_PROBES: tuple[str, ...] = (
    "vllm_omni.platforms",
    "vllm_omni.platforms.interface",
    "vllm_omni.diffusion.attention.backends.registry",
    "vllm_omni.distributed.omni_connectors",
    "vllm_omni.distributed.omni_connectors.factory",
)

_SDK_ISOLATION_SCRIPT = """
import importlib
import sys

BLOCKED = {blocked!r}
PROBES = {probes!r}
BACKENDS = {backends!r}

for name in BLOCKED:
    sys.modules[name] = None  # ``import name`` now raises ImportError

for name in PROBES:
    importlib.import_module(name)

import vllm_omni.platforms as platforms

qualname = platforms.resolve_current_omni_platform_cls_qualname()
if not isinstance(qualname, str) or not qualname:
    raise SystemExit(f"detector returned {{qualname!r}}")

leaked = [name for name in BLOCKED if sys.modules.get(name) is not None]
if leaked:
    raise SystemExit(f"blocked SDKs were populated in sys.modules: {{leaked}}")

detected = next((b for b in BACKENDS if qualname.startswith(f"vllm_omni.platforms.{{b}}.")), None)
loaded = [
    f"vllm_omni.platforms.{{b}}.platform"
    for b in BACKENDS
    if b != detected and f"vllm_omni.platforms.{{b}}.platform" in sys.modules
]
if loaded:
    raise SystemExit(f"neutral imports loaded platform classes: {{loaded}}")
# vllm-omni logs its import-time patches to stdout, so mark the result line.
print("DETECTED", qualname)
"""


def test_detector_and_neutral_code_import_without_optional_sdks() -> None:
    script = _SDK_ISOLATION_SCRIPT.format(
        blocked=OPTIONAL_HARDWARE_SDKS,
        probes=NEUTRAL_IMPORT_PROBES,
        backends=BACKENDS,
    )
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-4000:]}"
    detected = [line for line in result.stdout.splitlines() if line.startswith("DETECTED ")]
    assert detected and detected[-1].split(" ", 1)[1].startswith("vllm_omni.platforms."), result.stdout
