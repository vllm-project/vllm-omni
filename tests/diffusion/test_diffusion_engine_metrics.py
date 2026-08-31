# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Source-level regression tests for diffusion output/engine helpers.

These tests verify naming conventions and patterns by inspecting source code
at the function level using AST. They are intentionally coupled to the source
layout and should be updated whenever the inspected helper code is refactored.
"""

from __future__ import annotations

import ast
import os

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


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
_METRICS_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "vllm_omni",
        "metrics",
        "utils.py",
    )
)
_INLINE_CLIENT_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "vllm_omni",
        "diffusion",
        "inline_stage_diffusion_client.py",
    )
)
_STAGE_PROC_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "vllm_omni",
        "diffusion",
        "stage_diffusion_proc.py",
    )
)
_ORCHESTRATOR_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "vllm_omni",
        "engine",
        "orchestrator.py",
    )
)


def _read_source(path: str) -> str:
    with open(path, encoding="utf-8") as f:
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
        # step_total_ms is no longer emitted as a metric (no family consumes
        # it); it lives only in the debug log breakdown below.
        assert '"diffusion_engine_total_time_ms"' not in step_streaming_source
        assert '"postprocess_time_ms": postprocess_time * 1000' in step_streaming_source

    def test_step_streaming_piggybacks_scheduler_waiting_snapshot(self) -> None:
        source = _read_source(_ENGINE_PATH)
        step_streaming_source = _get_function_source(source, "DiffusionEngine", "step_streaming")
        assert "diffusion_scheduler_waiting_metrics(" in step_streaming_source
        assert "_scheduler_num_waiting_reqs" in step_streaming_source

    def test_step_streaming_preserves_snapshot_on_postprocess_error(self) -> None:
        source = _read_source(_ENGINE_PATH)
        step_streaming_source = _get_function_source(source, "DiffusionEngine", "step_streaming")
        assert "scheduler_metrics = diffusion_scheduler_waiting_metrics(" in step_streaming_source
        assert 'setattr(exc, "diffusion_metrics", scheduler_metrics)' in step_streaming_source
        assert step_streaming_source.index("scheduler_metrics =") < step_streaming_source.index(
            "self.postprocess_output("
        )

    def test_abort_keeps_output_consumers_alive_for_terminal_snapshot(self) -> None:
        inline_source = _read_source(_INLINE_CLIENT_PATH)
        abort_source = _get_function_source(inline_source, "InlineStageDiffusionClient", "abort_requests_async")
        assert ".cancel(" not in abort_source

        proc_source = _read_source(_STAGE_PROC_PATH)
        run_loop_source = _get_function_source(proc_source, "StageDiffusionProc", "run_loop")
        abort_branch = run_loop_source[run_loop_source.index('elif msg_type == "abort":') :]
        abort_branch = abort_branch[: abort_branch.index('elif msg_type == "collective_rpc":')]
        assert ".cancel(" not in abort_branch

    def test_orchestrator_consumes_metrics_only_output_without_routing(self) -> None:
        """A metrics-only sentinel contributes its queue depth and is not routed.

        Absorption lives in ``_absorb_diffusion_metrics`` rather than inline in a
        loop, so both orchestration loops share one implementation. The ordering
        is asserted where it now lives: the snapshot inside the helper, and the
        absorb-before-route guard inside every loop that polls diffusion output.
        """
        source = _read_source(_ORCHESTRATOR_PATH)

        # Snapshot before the verdict, so a sentinel still reports its waiting
        # depth on the way out instead of being dropped unaccounted.
        absorb_source = _get_function_source(source, "Orchestrator", "_absorb_diffusion_metrics")
        snapshot_pos = absorb_source.index("_update_stage_replica_waiting(")
        sentinel_pos = absorb_source.index("diffusion_output.request_id == DIFFUSION_METRICS_ONLY_REQUEST_ID")
        assert snapshot_pos < sentinel_pos

        # Absorb before routing, or a sentinel reaches the downstream consumer
        # as if it were a real request output.
        for loop_name in ("_orchestration_loop", "_orchestration_loop_event_driven"):
            loop_source = _get_function_source(source, "Orchestrator", loop_name)
            absorb_pos = loop_source.index("self._absorb_diffusion_metrics(")
            route_pos = loop_source.index("record_output_timestamps(")
            assert absorb_pos < route_pos, loop_name


class TestVaeDecodeEmit:
    """VAE decode timing is sourced from ``DiffusionOutput.stage_durations``
    (populated by the diffusion pipeline profiler) and emitted as
    ``vae_decode_time_ms`` so the ``_MS_TO_S`` accumulator picks it up.
    """

    def test_extract_vae_decode_ms_helper_exists(self) -> None:
        source = _read_source(_METRICS_PATH)
        helper_src = _get_function_source(source, None, "extract_diffusion_vae_decode_ms")
        assert ".vae.decode" in helper_src, "helper must key on the '.vae.decode' suffix"

    def test_step_streaming_emits_vae_decode_time_ms(self) -> None:
        source = _read_source(_ENGINE_PATH)
        step_streaming_source = _get_function_source(source, "DiffusionEngine", "step_streaming")
        assert "extract_diffusion_vae_decode_ms(" in step_streaming_source, (
            "step_streaming must call extract_diffusion_vae_decode_ms to source VAE decode timing"
        )
        assert '"vae_decode_time_ms"' in step_streaming_source, (
            "step_streaming must emit vae_decode_time_ms key for the _MS_TO_S accumulator"
        )


class TestDiffuseForwardEmit:
    """Forward-only timing → forward_time_ms, mirroring TestVaeDecodeEmit."""

    def test_extract_diffuse_ms_helper_exists(self) -> None:
        source = _read_source(_METRICS_PATH)
        helper_src = _get_function_source(source, None, "extract_diffusion_denoise_ms")
        assert ".diffuse" in helper_src, "helper must key on the '.diffuse' suffix"

    def test_step_streaming_emits_forward_time_ms(self) -> None:
        source = _read_source(_ENGINE_PATH)
        step_streaming_source = _get_function_source(source, "DiffusionEngine", "step_streaming")
        assert "extract_diffusion_denoise_ms(" in step_streaming_source, (
            "step_streaming must call extract_diffusion_denoise_ms to source forward-only timing"
        )
        assert '"forward_time_ms"' in step_streaming_source, (
            "step_streaming must emit forward_time_ms key for the _MS_TO_S accumulator"
        )


class TestKvRecvEmit:
    """KV-recv timing → kv_recv_time_ms, sourced from DiffusionOutput.kv_recv_ms
    (set by the runner's _prepare_request_for_forward, not the profiler)."""

    def test_step_streaming_emits_kv_recv_time_ms(self) -> None:
        source = _read_source(_ENGINE_PATH)
        step_streaming_source = _get_function_source(source, "DiffusionEngine", "step_streaming")
        assert "kv_recv_ms" in step_streaming_source, "step_streaming must read output.kv_recv_ms"
        assert '"kv_recv_time_ms"' in step_streaming_source, (
            "step_streaming must emit kv_recv_time_ms key for the _MS_TO_S accumulator"
        )


class TestDummyRunAllocation:
    """Verify _dummy_run generates exact-sized audio arrays."""

    def test_no_oversized_allocation(self) -> None:
        """_dummy_run should not allocate more audio than needed."""
        source = _read_source(_ENGINE_PATH)
        dummy_source = _get_function_source(source, "DiffusionEngine", "_dummy_run")
        assert "audio_sr * audio_duration_sec" not in dummy_source, (
            "_dummy_run should generate exact-sized audio, not allocate and slice"
        )
