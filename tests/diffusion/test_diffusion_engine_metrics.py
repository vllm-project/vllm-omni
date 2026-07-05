# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-level regression tests for diffusion output/engine helpers.

These tests verify naming conventions and patterns by inspecting source code
at the function level using AST. They are intentionally coupled to the source
layout and should be updated whenever the inspected helper code is refactored.
"""

from __future__ import annotations

import ast
import os

_ENGINE_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "vllm_omni",
        "diffusion",
        "diffusion_engine.py",
    )
)
_FORMATTER_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "vllm_omni",
        "diffusion",
        "output_formatter.py",
    )
)


def _read_source(path: str) -> str:
    with open(path) as f:
        return f.read()


def _get_function_source(source: str, class_name: str | None, func_name: str) -> str:
    """Extract the source of a specific function/method using AST.

    Args:
        source: Full file source code.
        class_name: Enclosing class name, or None for module-level functions.
        func_name: Function/method name.

    Returns:
        Source code of the function body.
    """
    tree = ast.parse(source)
    if class_name is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == func_name:
                        result = ast.get_source_segment(source, item)
                        assert result is not None, f"{class_name}.{func_name} source not found"
                        return result
    else:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                result = ast.get_source_segment(source, node)
                assert result is not None, f"{func_name} source not found"
                return result
    raise AssertionError(f"Function {class_name + '.' if class_name else ''}{func_name} not found in source")


class TestMetricKeys:
    """Verify metric naming conventions in diffusion output formatting."""

    def test_no_duplicate_preprocess_key(self) -> None:
        """format_diffusion_outputs() should not duplicate 'preprocess_time_ms'."""
        source = _read_source(_FORMATTER_PATH)
        formatter_source = _get_function_source(source, None, "format_diffusion_outputs")
        assert "preprocessing_time_ms" not in formatter_source, (
            "Found duplicate key 'preprocessing_time_ms' in "
            "format_diffusion_outputs() — should only use 'preprocess_time_ms'"
        )

    def test_metric_key_naming_consistency(self) -> None:
        """Metric keys should map to the explicit timing fields."""
        source = _read_source(_FORMATTER_PATH)
        formatter_source = _get_function_source(source, None, "format_diffusion_outputs")
        lines = formatter_source.split("\n")

        found_exec = False
        found_total = False
        for line in lines:
            if '"diffusion_engine_exec_time_ms"' in line:
                found_exec = True
                assert "timings.exec_time_s" in line, (
                    "diffusion_engine_exec_time_ms should measure executor time only (timings.exec_time_s)"
                )
            if '"diffusion_engine_total_time_ms"' in line:
                found_total = True
                assert "timings.total_time_ms" in line, (
                    "diffusion_engine_total_time_ms should measure full step time (timings.total_time_ms)"
                )
        assert found_exec, "diffusion_engine_exec_time_ms key not found in format_diffusion_outputs()"
        assert found_total, "diffusion_engine_total_time_ms key not found in format_diffusion_outputs()"


class TestDummyRunAllocation:
    """Verify _dummy_run generates exact-sized audio arrays."""

    def test_no_oversized_allocation(self) -> None:
        """_dummy_run should not allocate more audio than needed."""
        source = _read_source(_ENGINE_PATH)
        dummy_source = _get_function_source(source, "DiffusionEngine", "_dummy_run")
        assert "audio_sr * audio_duration_sec" not in dummy_source, (
            "_dummy_run should generate exact-sized audio, not allocate and slice"
        )
