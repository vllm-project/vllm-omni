# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static regression checks for VoxCPM2 runner-assisted unified graph hooks.

These tests intentionally avoid importing torch/vLLM so they can run in a
lightweight local checkout. Runtime audio quality still requires a CUDA test.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "vllm_omni" / "worker" / "gpu_ar_model_runner.py"
RUNNER_ASSISTED_METADATA = REPO_ROOT / "vllm_omni" / "worker" / "runner_assisted_metadata.py"
TALKER = REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "voxcpm2" / "voxcpm2_talker.py"
VOXCPM2_RUNTIME_CONFIG = REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "voxcpm2" / "runtime_config.py"
VOXCPM2_PIPELINE = REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "voxcpm2" / "pipeline.py"
VOXCPM2_SCHEDULER = REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "voxcpm2" / "scheduler.py"
OMNI_AR_SCHEDULER = REPO_ROOT / "vllm_omni" / "core" / "sched" / "omni_ar_scheduler.py"
DEPLOY = REPO_ROOT / "vllm_omni" / "deploy" / "voxcpm2.yaml"


def test_ar_runner_exposes_runner_assisted_full_metadata_hook():
    source = RUNNER.read_text()

    assert "Models without this hook keep the normal runner path" in source
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
    assert "runner_assisted_full_attn_request.num_reqs_padded" in padding_source
    assert "runner_assisted_full_attn_request.for_cudagraph_capture" in padding_source
    assert "num_reqs_padded * max_num_scheduled_tokens" in padding_source


def test_ar_runner_without_model_hook_stays_on_normal_path():
    source = RUNNER.read_text()
    contract_source = RUNNER_ASSISTED_METADATA.read_text()
    request_source = source[source.index("def _get_runner_assisted_full_attention_metadata_request") :]
    request_source = request_source[
        : request_source.index("def _refresh_runner_assisted_full_attention_metadata_buffers")
    ]
    compact_request_source = "".join(request_source.split())

    assert "Models without this hook keep the normal runner path" in request_source
    assert "class RunnerAssistedAttentionMetadataProvider(Protocol)" in contract_source
    assert "class RunnerAssistedFullAttentionMetadataRequest(NamedTuple)" in contract_source
    assert "tuple[int, bool]" not in contract_source
    assert 'hook=getattr(self.model,"get_runner_assisted_full_attention_metadata_request",None)' in (
        compact_request_source
    )
    assert "ifnotcallable(hook):returnNone" in compact_request_source
    assert ".tolist()" not in compact_request_source
    assert "isinstance(request,RunnerAssistedFullAttentionMetadataRequest)" in compact_request_source
    assert "request.num_reqs_padded" in request_source
    assert "request.for_cudagraph_capture" in request_source
    assert "exceptException" not in compact_request_source

    set_context_source = source[source.index("def _set_runner_assisted_full_attention_metadata_context") :]
    set_context_source = set_context_source[: set_context_source.index("def _deferred_prefix_cache_mm_keys")]
    assert "except Exception" not in set_context_source


def test_voxcpm2_graph_paths_fail_closed_and_preserve_deterministic_noise():
    source = TALKER.read_text()
    runtime_config_source = VOXCPM2_RUNTIME_CONFIG.read_text()
    compact_source = "".join(source.split())

    assert "_voxcpm2_compile_without_inductor_cudagraphs" in source
    assert "def _compile_without_inductor_cudagraphs" not in source
    assert "_voxcpm2_unwrap_torch_compile" in source
    assert "_orig_mod" in source
    compile_source = source[source.index("def _voxcpm2_compile_without_inductor_cudagraphs") :]
    compile_source = compile_source[: compile_source.index("def _setup_torch_compile")]
    assert "torch._inductor.config" not in compile_source
    assert '"triton.cudagraphs": False' in compile_source
    assert '"triton.cudagraph_trees": False' in compile_source

    assert "self._enable_unified_decode_graph=self._runtime_config.unified_decode_graph_available(" in compact_source
    assert "andnotself.deterministic_cfm_noise" in "".join(runtime_config_source.split())
    assert "decode_tail_graph" not in source

    unified_source = source[source.index("def _forward_unified_decode") :]
    unified_source = unified_source[: unified_source.index("# -------------------- vllm hooks")]
    assert "except Exception:" not in unified_source
    assert "_failed_unified_graph_sizes" not in source
    assert "self._enable_unified_decode_graph = False" not in unified_source
    assert "_forward_unified_decode_fallback" not in source
    assert "capture_failed" not in compact_source
    assert "record_fallback" not in source
    assert "fallbacks" not in source
    assert "**info_dict: Unpack[VoxCPM2PreprocessInput]" in source
    assert "**info: Unpack[VoxCPM2PostprocessInput]" in source
    assert "reference_audio: Any" not in source
    assert "prompt_audio: Any" not in source
    assert "def _nullify_volatile_metadata(ctx: _ForwardContextLike) -> _ForwardContextLike" in source
    assert "class _PrefillInputs(NamedTuple)" in source
    assert "-> _PrefillInputs" in source
    assert 'inputs["audio_feat"]' not in source
    assert 'inputs["text_token"]' not in source

    capture_source = source[source.index("def _capture_unified_decode_graph") :]
    capture_source = capture_source[: capture_source.index("def _unified_decode_graph_skip_reason")]
    assert "_voxcpm2_compile_unified_capture_estimator" in capture_source
    assert "_voxcpm2_compile_unified_capture_feat_encoder" in capture_source
    graph_body = capture_source[capture_source.index("with torch.cuda.graph") :]
    assert "g.cfm_noise.normal_()" not in graph_body
    assert "g.cfm_noise.normal_()" in unified_source

    assert "_pre_capture_unified_graphs" not in source
    assert "_unified_graph_pre_capture_sizes" not in source
    assert "unified_decode_graph_pre_capture_sizes" not in source


def test_voxcpm2_batch_unified_graph_requires_runner_metadata_marker():
    source = TALKER.read_text()
    compact_source = "".join(source.split())

    assert "enable_runner_assisted_unified_decode_graph" not in source
    assert "allow_unified_decode_graph_batch_attention" not in source
    assert "get_runner_assisted_full_attention_metadata_request" in source
    assert "set_runner_assisted_full_attention_metadata_context" in source
    assert "RunnerAssistedFullAttentionMetadataRequest" in source
    assert "_runner_assisted_unified_decode_graph_active" in source
    assert "runner_full_metadata_missing" in source
    assert "_build_unified_graph_bucket_sizes" in source
    assert "_select_unified_graph_bucket_size" in source

    capture_source = source[source.index("def _capture_unified_decode_graph") :]
    capture_source = capture_source[: capture_source.index("def _unified_decode_graph_skip_reason")]
    assert "override_forward_context" in capture_source
    assert "_nullify_volatile_metadata" in capture_source
    assert "capture_context = override_forward_context(self._nullify_volatile_metadata(ctx))" in capture_source
    assert "ifsize>1:continue" not in compact_source
    forward_source = source[source.index("def _forward_unified_decode") :]
    forward_source = forward_source[: forward_source.index("# -------------------- vllm hooks")]
    assert "graph_size = self._select_unified_graph_bucket_size(num_reqs)" in forward_source
    assert "self._unified_graphs[graph_size]" in forward_source
    assert "g.input_embeds[num_reqs:graph_size].zero_()" not in forward_source
    assert "padded rows as valid duplicates" in forward_source
    assert "g.input_embeds[last : last + 1].expand" in forward_source
    assert "ifnum_reqs>1andnotself._runner_assisted_unified_decode_graph_active" in compact_source
    assert "andnotcfg.enable_runner_assisted_unified_decode_graph" not in compact_source
    needs_source = source[source.index("def get_runner_assisted_full_attention_metadata_request") :]
    needs_source = needs_source[: needs_source.index("def set_runner_assisted_full_attention_metadata_context")]
    assert "_select_unified_graph_bucket_size(num_reqs)" in needs_source
    assert "_should_use_decode_graph(num_reqs)" not in needs_source
    assert "RunnerAssistedFullAttentionMetadataRequest(" in needs_source


def test_voxcpm2_unified_skip_preserves_segmented_decode_graphs():
    source = TALKER.read_text()
    compact_source = "".join(source.split())
    forward_source = source[source.index("def forward(") :]
    forward_source = forward_source[: forward_source.index("# -------------------- prefill / decode helpers")]

    assert "use_segmented_decode_graph" in forward_source
    assert "unified_skip_reasonisnotNone" in compact_source
    assert "notself._enable_unified_decode_graphorunified_skip_reasonisnotNone" in compact_source
    assert "ifuse_segmented_decode_graph:" in compact_source
    assert "anduse_segmented_decode_graph" in compact_source


def test_voxcpm2_scheduler_policy_stays_model_local():
    common_source = OMNI_AR_SCHEDULER.read_text()
    voxcpm2_scheduler_source = VOXCPM2_SCHEDULER.read_text()
    pipeline_source = VOXCPM2_PIPELINE.read_text()

    assert "VoxCPM2TalkerForConditionalGeneration" not in common_source
    assert "voxcpm2_runtime_config" not in common_source
    assert "pure_decode_graph" not in common_source
    assert "_schedule_with_optional_waiting_deferral" not in common_source
    assert "def _should_defer_waiting_admission(self) -> bool:" in common_source

    assert "class VoxCPM2OmniARAsyncScheduler(OmniARAsyncScheduler)" in voxcpm2_scheduler_source
    assert "from .runtime_config import _VoxCPM2RuntimeConfig" in voxcpm2_scheduler_source
    assert "_VoxCPM2RuntimeConfig.from_vllm_config(self.vllm_config)" in voxcpm2_scheduler_source
    assert "unified_decode_graph_available(use_cuda_graph=current_omni_platform.is_cuda())" in voxcpm2_scheduler_source
    assert "_should_defer_waiting_for_unified_decode_graph" in voxcpm2_scheduler_source
    assert "def schedule(" not in voxcpm2_scheduler_source
    assert "create_request_queue" not in voxcpm2_scheduler_source
    assert "original_waiting.prepend_requests(deferred_waiting)" not in voxcpm2_scheduler_source
    assert "return self._should_defer_waiting_for_unified_decode_graph()" in voxcpm2_scheduler_source
    assert "vllm_omni.model_executor.models.voxcpm2.scheduler.VoxCPM2OmniARAsyncScheduler" in pipeline_source


def test_voxcpm2_deploy_defaults_to_full_unified_graph_only():
    source = DEPLOY.read_text()

    assert "max_num_seqs: 8" in source
    assert "enable_unified_decode_graph: true" in source
    assert "unified_decode_graph_max_batch_size: 8" in source
    assert "unified_decode_graph_pre_capture_sizes" not in source
    assert "enable_runner_assisted_unified_decode_graph" not in source
    assert "allow_unified_decode_graph_batch_attention" not in source
    assert "decode_graph_capture_policy" not in source
