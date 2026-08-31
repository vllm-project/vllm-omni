# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drift checks for the reviewed environment-variable inventory."""

import ast
import builtins
from collections import Counter
from pathlib import Path

import pytest

from vllm_omni.config.environment_variable_inventory import (
    ENVIRONMENT_VARIABLE_INVENTORY,
    EnvironmentVariableCategory,
    ModelEnvironmentVariableDisposition,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "vllm_omni"
_INVENTORY_MODULE = _PACKAGE_ROOT / "config" / "environment_variable_inventory.py"
_REFERENCE_PAGE = _REPO_ROOT / "docs" / "configuration" / "environment_variables.md"


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for statement in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value.value
    return constants


def _os_import_aliases(tree: ast.Module) -> tuple[set[str], set[str], set[str]]:
    os_names: set[str] = set()
    getenv_names: set[str] = set()
    environ_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "os":
                    os_names.add(imported.asname or "os")
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for imported in node.names:
                if imported.name == "getenv":
                    getenv_names.add(imported.asname or imported.name)
                elif imported.name == "environ":
                    environ_names.add(imported.asname or imported.name)
    return os_names, getenv_names, environ_names


def _is_environ(node: ast.expr, os_names: set[str], environ_names: set[str]) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in os_names
        and node.attr == "environ"
    ) or (isinstance(node, ast.Name) and node.id in environ_names)


def _environment_key_expressions(
    node: ast.AST,
    os_names: set[str],
    getenv_names: set[str],
    environ_names: set[str],
) -> list[ast.expr]:
    """Return key expressions used by one direct environment access."""
    if isinstance(node, ast.Call) and node.args:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in os_names
            and func.attr == "getenv"
        ) or (isinstance(func, ast.Name) and func.id in getenv_names):
            return [node.args[0]]
        if (
            isinstance(func, ast.Attribute)
            and _is_environ(func.value, os_names, environ_names)
            and func.attr in {"get", "pop", "setdefault"}
        ):
            return [node.args[0]]
    elif isinstance(node, ast.Subscript) and _is_environ(node.value, os_names, environ_names):
        return [node.slice]
    elif isinstance(node, ast.Compare):
        expressions: list[ast.expr] = []
        left_values = [node.left, *node.comparators[:-1]]
        for left, operator, right in zip(left_values, node.ops, node.comparators):
            if isinstance(operator, (ast.In, ast.NotIn)) and _is_environ(right, os_names, environ_names):
                expressions.append(left)
        return expressions
    return []


def _resolve_environment_name(expression: ast.expr, constants: dict[str, str]) -> str | None:
    value: str | None = None
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        value = expression.value
    elif isinstance(expression, ast.Name):
        value = constants.get(expression.id)
    return value or None


def _environment_wrapper_parameters(
    tree: ast.Module,
    os_names: set[str],
    getenv_names: set[str],
    environ_names: set[str],
) -> dict[str, set[tuple[str, int | None]]]:
    """Find local helpers that forward one parameter as an environment key."""
    wrappers: dict[str, set[tuple[str, int | None]]] = {}
    for function in (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        positional = [*function.args.posonlyargs, *function.args.args]
        parameter_indexes: dict[str, int | None] = {parameter.arg: index for index, parameter in enumerate(positional)}
        parameter_indexes.update({parameter.arg: None for parameter in function.args.kwonlyargs})
        forwarded: set[tuple[str, int | None]] = set()
        for node in ast.walk(function):
            for expression in _environment_key_expressions(node, os_names, getenv_names, environ_names):
                if isinstance(expression, ast.Name) and expression.id in parameter_indexes:
                    forwarded.add((expression.id, parameter_indexes[expression.id]))
        if forwarded:
            wrappers[function.name] = forwarded
    return wrappers


def _environment_accesses(path: Path) -> set[str]:
    """Return statically resolvable names accessed through ``os``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants = _module_string_constants(tree)
    os_names, getenv_names, environ_names = _os_import_aliases(tree)
    wrappers = _environment_wrapper_parameters(tree, os_names, getenv_names, environ_names)
    names: set[str] = set()

    for node in ast.walk(tree):
        for expression in _environment_key_expressions(node, os_names, getenv_names, environ_names):
            if name := _resolve_environment_name(expression, constants):
                names.add(name)

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            for parameter_name, position in wrappers.get(node.func.id, set()):
                argument = node.args[position] if position is not None and position < len(node.args) else None
                if argument is None:
                    argument = next(
                        (keyword.value for keyword in node.keywords if keyword.arg == parameter_name),
                        None,
                    )
                if argument is not None and (name := _resolve_environment_name(argument, constants)):
                    names.add(name)

    return names


def test_inventory_matches_reviewed_snapshot_counts():
    """Make an inventory expansion an explicit review decision."""
    category_counts = Counter(item.category for item in ENVIRONMENT_VARIABLE_INVENTORY.values())
    assert category_counts == {
        EnvironmentVariableCategory.PUBLIC_OMNI: 24,
        EnvironmentVariableCategory.INHERITED_VLLM: 20,
        EnvironmentVariableCategory.PLATFORM_EXTERNAL: 27,
        EnvironmentVariableCategory.MODEL_SPECIFIC: 56,
        EnvironmentVariableCategory.BENCHMARK_TRANSITIONAL: 20,
        EnvironmentVariableCategory.INTERNAL: 2,
    }

    disposition_counts = Counter(
        item.model_disposition
        for item in ENVIRONMENT_VARIABLE_INVENTORY.values()
        if item.category is EnvironmentVariableCategory.MODEL_SPECIFIC
    )
    assert {disposition: disposition_counts[disposition] for disposition in ModelEnvironmentVariableDisposition} == {
        ModelEnvironmentVariableDisposition.PROMOTE: 33,
        ModelEnvironmentVariableDisposition.REQUEST_SCOPE: 6,
        ModelEnvironmentVariableDisposition.EXTERNAL: 0,
        ModelEnvironmentVariableDisposition.INTERNALIZE: 11,
        ModelEnvironmentVariableDisposition.DEPRECATE_REMOVE: 6,
    }


def test_new_public_omni_names_use_project_prefix():
    """Grandfather legacy names without allowing more prefix exceptions."""
    legacy_public_names = {
        "DIFFUSION_ATTENTION_BACKEND",
        "DIFFUSION_CACHE_ADAPTER",
        "DIFFUSION_CACHE_BACKEND",
        "OMNI_DIFFUSION_PROMPT_EMBED_CACHE",
        "OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE",
        "OMNI_DIFFUSION_SESSION_STATE_MANAGER",
        "OMNI_DIFFUSION_SESSION_STATE_MANAGER_MAX_SESSIONS",
        "SPEAKER_MAX_UPLOADED",
        "SPEAKER_SAMPLES_DIR",
        "VLLM_VIDEO_ASYNC_CHUNK",
        "VLLM_VIDEO_AUDIO_DELTA_MODE",
    }
    public_names = {
        name
        for name, item in ENVIRONMENT_VARIABLE_INVENTORY.items()
        if item.category is EnvironmentVariableCategory.PUBLIC_OMNI
    }

    assert {name for name in public_names if not name.startswith("VLLM_OMNI_")} == legacy_public_names


def test_statically_resolvable_environment_accesses_are_classified():
    discovered: set[str] = set()
    for path in _PACKAGE_ROOT.rglob("*.py"):
        discovered.update(_environment_accesses(path))

    assert discovered - ENVIRONMENT_VARIABLE_INVENTORY.keys() == set()


def test_model_and_benchmark_inventory_entries_are_still_referenced():
    """Reject transitional rows after their implementation has disappeared."""
    package_source = "\n".join(
        path.read_text(encoding="utf-8") for path in _PACKAGE_ROOT.rglob("*.py") if path != _INVENTORY_MODULE
    )
    transitional_categories = {
        EnvironmentVariableCategory.MODEL_SPECIFIC,
        EnvironmentVariableCategory.BENCHMARK_TRANSITIONAL,
    }

    stale_names = {
        name
        for name, item in ENVIRONMENT_VARIABLE_INVENTORY.items()
        if item.category in transitional_categories and name not in package_source
    }

    assert stale_names == set()


def test_environment_scanner_covers_indirection_aliases_membership_and_casing():
    expected_by_path = {
        "metrics/definitions.py": {
            "VLLM_OMNI_BENCH_AUDIO_CHANNELS",
            "VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE",
        },
        "entrypoints/openai/video_stream_envs.py": {
            "VLLM_VIDEO_ASYNC_CHUNK",
            "VLLM_VIDEO_AUDIO_DELTA_MODE",
        },
        "model_executor/models/mimo_audio/mimo_audio.py": {"model_stage"},
        "model_executor/models/moss_tts/modeling_moss_tts_local_depth.py": {"MOSS_TTS_DEBUG_STOP"},
        "distributed/ray_utils/utils.py": {"RAY_RAYLET_PID"},
    }
    for relative_path, expected in expected_by_path.items():
        assert expected <= _environment_accesses(_PACKAGE_ROOT / relative_path)


def test_generated_server_storage_environment_names_are_classified():
    from vllm_omni.config.server_settings import FileBackend

    generated_names = {
        f"VLLM_OMNI_SERVER_STORAGE__{field_name.upper()}"
        for field_name in FileBackend.model_fields
        if field_name != "type"
    }
    assert generated_names <= ENVIRONMENT_VARIABLE_INVENTORY.keys()


def test_public_omni_variables_are_in_the_reference_page():
    reference = _REFERENCE_PAGE.read_text(encoding="utf-8")
    missing = {
        name
        for name, item in ENVIRONMENT_VARIABLE_INVENTORY.items()
        if item.is_public_omni and f"`{name}`" not in reference
    }
    assert missing == set()


def test_secret_values_are_marked_for_redaction():
    assert {name for name, item in ENVIRONMENT_VARIABLE_INVENTORY.items() if item.redact_value} == {
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "OPENAI_API_KEY",
    }


def test_collect_env_reports_only_safe_public_omni_values(monkeypatch):
    from collect_env import get_env_vars

    monkeypatch.setenv("DIFFUSION_CACHE_BACKEND", "tea_cache")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("VLLM_OMNI_REPLICA_ID", "internal-replica")

    report = get_env_vars()

    assert "DIFFUSION_CACHE_BACKEND=tea_cache" in report.splitlines()
    assert "hf-secret" not in report
    assert "openai-secret" not in report
    assert "VLLM_OMNI_REPLICA_ID" not in report


@pytest.mark.parametrize("error_type", [ImportError, OSError, AssertionError, RuntimeError])
def test_collect_env_survives_inventory_import_error(monkeypatch, error_type):
    from collect_env import get_env_vars

    real_import = builtins.__import__

    def fail_inventory_import(name, *args, **kwargs):
        if name == "vllm_omni.config.environment_variable_inventory":
            raise error_type("broken vllm-omni installation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_inventory_import)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert "CUDA_VISIBLE_DEVICES=0" in get_env_vars().splitlines()
