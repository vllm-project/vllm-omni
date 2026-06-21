# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static regression checks for VoxCPM2 runner-assisted unified graph hooks.

These tests intentionally avoid importing torch/vLLM so they can run in a
lightweight local checkout. Runtime audio quality still requires a CUDA test.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "vllm_omni" / "worker" / "gpu_ar_model_runner.py"
TALKER = REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "voxcpm2" / "voxcpm2_talker.py"


def test_ar_runner_exposes_runner_assisted_full_metadata_hook():
    source = RUNNER.read_text()

    assert "get_runner_assisted_full_attention_metadata_request" in source
    assert "set_runner_assisted_full_attention_metadata_context" in source
    assert "runner_assisted_full_attn" in source
    assert "runner_assisted_full_attn_capture" in source
    assert "pad_attn = cudagraph_mode == CUDAGraphMode.FULL or runner_assisted_full_attn" in source
    assert "for_cudagraph_capture=runner_assisted_full_attn_capture" in source
    assert "cudagraph_runtime_mode=(" in source
    assert "CUDAGraphMode.FULL if runner_assisted_full_attn else cudagraph_mode" in source
    refresh_source = source[source.index("def _refresh_runner_assisted_full_attention_metadata_buffers") :]
    refresh_source = refresh_source[: refresh_source.index("def _set_runner_assisted_full_attention_metadata_context")]
    assert "num_computed_tokens_cpu" in refresh_source
    assert "num_scheduled_tokens_np" in refresh_source
    assert "optimistic_seq_lens_cpu[:num_reqs]=1" not in "".join(refresh_source.split())
    padding_source = source[
        source.index("runner_assisted_full_attn_request = self._get_runner_assisted_full_attention_metadata_request") :
    ]
    padding_source = padding_source[: padding_source.index("ubatch_slices, ubatch_slices_padded")]
    assert "num_reqs_padded, runner_assisted_full_attn_capture =" in padding_source
    assert "num_tokens_padded = max(num_tokens_padded, num_reqs_padded)" in padding_source


def test_voxcpm2_batch_unified_graph_requires_runner_metadata_marker():
    source = TALKER.read_text()
    compact_source = "".join(source.split())

    assert "enable_runner_assisted_unified_decode_graph" in source
    assert "get_runner_assisted_full_attention_metadata_request" in source
    assert "set_runner_assisted_full_attention_metadata_context" in source
    assert "_runner_assisted_unified_decode_graph_active" in source
    assert "runner_full_metadata_missing" in source
    assert "_build_unified_graph_bucket_sizes" in source
    assert "_select_unified_graph_bucket_size" in source

    capture_source = source[source.index("def _capture_unified_decode_graph") :]
    capture_source = capture_source[: capture_source.index("def _unified_decode_graph_skip_reason")]
    assert "override_forward_context" in capture_source
    assert "_nullify_volatile_metadata" in capture_source
    assert "capture_context = override_forward_context(self._nullify_volatile_metadata(ctx))" in capture_source
    assert "size>1andself._runtime_config.enable_runner_assisted_unified_decode_graph" in compact_source
    forward_source = source[source.index("def _forward_unified_decode") :]
    forward_source = forward_source[: forward_source.index("# -------------------- vllm hooks")]
    assert "graph_size = self._select_unified_graph_bucket_size(num_reqs)" in forward_source
    assert "self._unified_graphs[graph_size]" in forward_source
    assert "g.input_embeds[num_reqs:graph_size].zero_()" in forward_source
    assert "andnotcfg.enable_runner_assisted_unified_decode_graph" not in compact_source
    needs_source = source[source.index("def get_runner_assisted_full_attention_metadata_request") :]
    needs_source = needs_source[: needs_source.index("def set_runner_assisted_full_attention_metadata_context")]
    assert "_select_unified_graph_bucket_size(num_reqs)" in needs_source
    assert "_should_use_decode_graph(num_reqs)" not in needs_source
