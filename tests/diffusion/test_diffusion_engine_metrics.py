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

    def test_timing_metric_key_naming_consistency(self) -> None:
        """Timing metrics should be attached by the engine orchestration path."""
        source = _read_source(_FORMATTER_PATH)
        formatter_source = _get_function_source(source, None, "format_diffusion_outputs")
        engine_source = _read_source(_ENGINE_PATH)
        step_streaming_source = _get_function_source(engine_source, "DiffusionEngine", "step_streaming")
        lines = formatter_source.split("\n")

        for line in lines:
            if '"diffusion_engine_exec_time_ms"' in line:
                raise AssertionError("diffusion_engine_exec_time_ms should be attached in step_streaming()")
            if '"diffusion_engine_total_time_ms"' in line:
                raise AssertionError("diffusion_engine_total_time_ms should be attached in step_streaming()")

        assert '"diffusion_engine_exec_time_ms": exec_total_time * 1000' in step_streaming_source
        assert '"diffusion_engine_total_time_ms": step_total_ms' in step_streaming_source
        assert '"postprocess_time_ms": postprocess_time * 1000' in step_streaming_source


class TestDummyRunAllocation:
    """Verify _dummy_run generates exact-sized audio arrays."""

    def test_no_oversized_allocation(self) -> None:
        """_dummy_run should not allocate more audio than needed."""
        source = _read_source(_ENGINE_PATH)
        dummy_source = _get_function_source(source, "DiffusionEngine", "_dummy_run")
        assert "audio_sr * audio_duration_sec" not in dummy_source, (
            "_dummy_run should generate exact-sized audio, not allocate and slice"
        )
