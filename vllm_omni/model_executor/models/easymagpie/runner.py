# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import gc
import os
import sys
from collections import Counter
from typing import Any

import torch

from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.model_executor.models.easymagpie.batched_feedback import feedback_merge_status, try_merge_feedback
from vllm_omni.model_executor.models.easymagpie.feedback_snapshot import is_owned_feedback
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_ar_worker import GPUARWorker
from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker

CPU_QUERY_VECTOR_ENABLED = True


def merge_streaming_additional_information(
    cached: dict[str, Any],
    incoming: dict[str, Any],
    accumulated_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Merge one streaming chunk without dropping persistent model state."""
    accumulated_keys = accumulated_keys or set()
    merged = dict(cached)

    for key, value in incoming.items():
        if not isinstance(value, dict):
            merged[key] = value
            continue

        old_value = merged.get(key)
        merged_sub = dict(old_value) if isinstance(old_value, dict) else {}
        for subkey, subvalue in value.items():
            if (key, subkey) in accumulated_keys and isinstance(subvalue, torch.Tensor):
                new_tensor = subvalue.detach().to("cpu").contiguous()
                old_tensor = merged_sub.get(subkey)
                merged_sub[subkey] = new_tensor if old_tensor is None else torch.cat((old_tensor, new_tensor), dim=0)
            else:
                merged_sub[subkey] = subvalue
        merged[key] = merged_sub

    meta = dict(merged.get("meta", {}))
    meta["num_processed_tokens"] = 0
    meta["resumable"] = True
    merged["meta"] = meta
    return merged


class EasyMagpieGPUARModelRunner(GPUARModelRunner):
    """GPU AR runner that restores streaming chunk metadata propagation."""

    # The output-builder threads sleep while D2H finishes, leaving CPU time
    # for the scheduler and codec instead of spinning in cudaEventSynchronize.
    omni_output_copy_blocking = True

    def _query_starts_cpu(self, count: int, site: str) -> list[int]:
        """Read the existing CPU int32 buffer once; return a fresh Python list."""
        host = self.query_start_loc.cpu
        if CPU_QUERY_VECTOR_ENABLED:
            counters = getattr(self, "_cpu_query_vector_counts", None)
            if counters is None:
                counters = self._cpu_query_vector_counts = Counter()
            # Metadata-only guards preserve the legacy behavior for any
            # unexpected frontend/buffer type, dtype, rank or short capacity.
            if (
                isinstance(host, torch.Tensor)
                and host.device.type == "cpu"
                and host.dtype == torch.int32
                and host.ndim == 1
                and 0 <= count <= host.numel()
            ):
                result = host[:count].tolist()
                counters[site + "_calls"] += 1
                counters[site + "_rows"] += count
                return result
            counters["fallback_calls"] += 1
        return [int(host[index]) for index in range(count)]

    def _maybe_run_batch_preprocess(self, req_ids, device, input_ids, inputs_embeds, num_scheduled_tokens_np):
        self._omni_num_scheduled_tokens_np = num_scheduled_tokens_np
        handled = super()._maybe_run_batch_preprocess(
            req_ids, device, input_ids, inputs_embeds, num_scheduled_tokens_np
        )
        prepare = getattr(self.model, "prepare_decode_batch", None)
        if callable(prepare):
            starts = self._query_starts_cpu(len(req_ids), "batch_preprocess")
            counts = [int(self._omni_num_scheduled_tokens_np[index]) for index in range(len(req_ids))]
            use_device_state = getattr(self.model, "_device_decode_state", None) is not None
            if use_device_state or getattr(self.model, "_prefill_graphs_enabled", False):
                self.model.prepare_query_dispatch(starts, counts, device)
            rows, request_keys = [], []
            for index, (req_id, start, count) in enumerate(zip(req_ids, starts, counts, strict=True)):
                request = self.requests[req_id]
                computed = int(self.input_batch.num_computed_tokens_cpu[index])
                if count == 1 and computed >= len(request.prompt_token_ids or []):
                    rows.append((start, self.model_intermediate_buffer.get(req_id, {})))
                    if use_device_state:
                        request_keys.append((req_id, id(self.requests[req_id])))
            if use_device_state:
                if getattr(self.model, "_retain_live_decode_leases", False):
                    # Keep the preceding snapshot alive until after pruning: a
                    # completed object cannot have its Python id recycled for
                    # a same-ID admission while its old lease still exists.
                    live_requests = dict(self.requests)
                    live_keys = [(key, id(request)) for key, request in live_requests.items()]
                    prepare(rows, device, request_keys=request_keys, live_request_keys=live_keys)
                    self._dds_live_request_objects = live_requests
                else:
                    prepare(rows, device, request_keys=request_keys)
            else:
                prepare(rows, device)

        if self._try_complete_q1_preprocess(input_ids, inputs_embeds, num_scheduled_tokens_np):
            return frozenset(req_ids)
        return handled

    def _try_complete_q1_preprocess(self, input_ids, inputs_embeds, counts) -> bool:
        """Consume an eligible DDS batch without per-row tensor copy launches.

        Called by the batch-preprocess hook after the base runner has
        prepared connector metadata, positions and graph-compatible buffers.
        Every guard runs before any update is consumed. The old loop handles
        the entire batch if one row cannot use this path.
        """
        from vllm_omni.model_executor.models.easymagpie.easymagpie import EasyMagpieTTSForConditionalGeneration

        model = self.get_model()
        if type(model) is not EasyMagpieTTSForConditionalGeneration or not getattr(
            model, "_batched_decode_inputs_enabled", False
        ):
            return False
        self._batched_decode_input_attempts = getattr(self, "_batched_decode_input_attempts", 0) + 1
        if (
            getattr(model, "_device_decode_state", None) is None
            or getattr(self, "has_talker_mtp", False)
            or getattr(self, "enable_prompt_embeds", False)
            or getattr(self, "speculative_config", None) is not None
            or getattr(self.model_config, "is_encoder_decoder", False)
            or getattr(self.parallel_config, "pipeline_parallel_size", 1) != 1
            or inputs_embeds is None
            or input_ids is None
            or input_ids.ndim != 1
            or not input_ids.is_contiguous()
            or input_ids.storage_offset() != 0
        ):
            return False
        req_ids = self.input_batch.req_ids
        if not req_ids or len(counts) != len(req_ids) or any(int(count) != 1 for count in counts):
            return False
        starts = self._query_starts_cpu(len(req_ids), "decode_fastpath")
        if starts != list(range(len(req_ids))) or inputs_embeds.shape[0] < len(req_ids):
            return False
        prepared = model._prepared_decode_updates
        if set(prepared) != set(starts):
            return False
        metadata = []
        for index, (req_id, start) in enumerate(zip(req_ids, starts, strict=True)):
            request = self.requests.get(req_id)
            prompt = getattr(request, "prompt_token_ids", None)
            if request is None or prompt is None:
                return False
            computed = int(self.input_batch.num_computed_tokens_cpu[index])
            if computed < len(prompt):
                return False
            update = prepared[start]
            if set(update) - {"decode_offset", "text_tokens"} or any(
                isinstance(value, torch.Tensor) for value in update.values()
            ):
                return False
            metadata.append((req_id, start, computed, len(prompt)))

        # The normal EasyMagpie single-token hook returns input_ids unchanged
        # and one zero embedding. Preserve padded rows exactly as the old loop.
        inputs_embeds[: len(req_ids)].zero_()
        for req_id, start, computed, prompt_len in metadata:
            info = self.model_intermediate_buffer.get(req_id, {})
            info.update(
                request_id=req_id,
                duplex_token_offset=computed,
                duplex_prompt_len=prompt_len,
                _omni_prompt_len=prompt_len,
                _omni_num_computed_tokens=computed,
                _omni_is_prefill=False,
            )
            self._update_intermediate_buffer(req_id, prepared.pop(start))
        self._batched_decode_input_hits = getattr(self, "_batched_decode_input_hits", 0) + 1
        self._batched_decode_input_requests = getattr(self, "_batched_decode_input_requests", 0) + len(req_ids)
        return True

    def get_prepost_status(self) -> dict[str, Any]:
        """Read-only on-demand counters; no device reads or synchronization."""
        model = self.get_model()
        return {
            "cpu_query_vector": {
                "enabled": CPU_QUERY_VECTOR_ENABLED,
                **dict(getattr(self, "_cpu_query_vector_counts", {})),
            },
            "device_decode_state": getattr(model, "_device_decode_state", None) is not None,
            "retain_live_decode_leases": getattr(model, "_retain_live_decode_leases", False),
            "decode_lease_status": (
                model._device_decode_state.lease_status()
                if getattr(model, "_device_decode_state", None) is not None
                else None
            ),
            "feedback_merge": feedback_merge_status(self),
            "batched_owned_feedback": getattr(model, "_batched_owned_feedback_enabled", False),
            "batched_decode_inputs": getattr(model, "_batched_decode_inputs_enabled", False),
            "batch_input_attempts": getattr(self, "_batched_decode_input_attempts", 0),
            "batch_input_hits": getattr(self, "_batched_decode_input_hits", 0),
            "batch_input_requests": getattr(self, "_batched_decode_input_requests", 0),
            "owned_feedback_view_stores": getattr(self, "_owned_feedback_view_stores", 0),
        }

    def _preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors=None):
        try:
            return super()._preprocess(scheduler_output, num_input_tokens, intermediate_tensors)
        finally:
            clear = getattr(self.model, "clear_decode_batch", None)
            if callable(clear):
                clear()

    def _process_additional_information_updates(
        self,
        hidden_states,
        multimodal_outputs,
        num_scheduled_tokens_np,
        scheduler_output,
        combined_hidden_states=None,
        combined_multimodal_outputs=None,
        req_ids_filter=None,
        req_ids=None,
        query_start_loc_cpu=None,
    ):
        if try_merge_feedback(
            self,
            hidden_states,
            multimodal_outputs,
            num_scheduled_tokens_np,
            combined_hidden_states=combined_hidden_states,
            combined_multimodal_outputs=combined_multimodal_outputs,
            req_ids_filter=req_ids_filter,
            req_ids=req_ids,
            query_start_loc_cpu=query_start_loc_cpu,
        ):
            return None
        return super()._process_additional_information_updates(
            hidden_states,
            multimodal_outputs,
            num_scheduled_tokens_np,
            scheduler_output,
            combined_hidden_states,
            combined_multimodal_outputs,
            req_ids_filter=req_ids_filter,
            req_ids=req_ids,
            query_start_loc_cpu=query_start_loc_cpu,
        )

    def _store_value(self, dest: dict, key: str, value: Any, gpu_keys: set) -> None:
        # Only postprocess views certified against immutable output snapshots
        # bypass cloning. Ordinary tensors keep upstream ownership semantics.
        if key in gpu_keys and is_owned_feedback(value, key):
            self._owned_feedback_view_stores = getattr(self, "_owned_feedback_view_stores", 0) + 1
            dest[key] = value.detach()
            return
        return super()._store_value(dest, key, value, gpu_keys)

    def _update_intermediate_buffer(self, req_id: str, upd: dict) -> None:
        if not isinstance(upd, dict) or not upd:
            return
        request = self.requests.get(req_id)
        if request is None:
            return

        gpu_keys = getattr(getattr(self, "model", None), "gpu_resident_buffer_keys", set())
        top_level_gpu_keys = {key for key in gpu_keys if isinstance(key, str)}
        nested_gpu_keys = {key for key in gpu_keys if isinstance(key, tuple) and len(key) == 2}
        existing = self.model_intermediate_buffer.setdefault(req_id, {})
        for key, value in upd.items():
            if isinstance(value, dict):
                existing_sub = existing.setdefault(key, {})
                resident_qualifiers = {qualifier for type_key, qualifier in nested_gpu_keys if type_key == key}
                for qualifier, subvalue in value.items():
                    self._store_value(existing_sub, qualifier, subvalue, resident_qualifiers)
            else:
                self._store_value(existing, key, value, top_level_gpu_keys)
        request.additional_information_cpu = existing

    def _update_streaming_request(self, req_id, new_req_data):
        payload = getattr(new_req_data, "additional_information", None)
        incoming = deserialize_additional_information(payload)
        if isinstance(incoming, dict) and incoming:
            model = getattr(self, "model", None)
            accumulated_keys = getattr(model, "streaming_accumulated_keys", set())
            cached = self.model_intermediate_buffer.get(req_id, {})
            merged = merge_streaming_additional_information(cached, incoming, accumulated_keys)
            self.model_intermediate_buffer[req_id] = merged
            setattr(self.requests[req_id], "additional_information_cpu", merged)

        return super()._update_streaming_request(req_id, new_req_data)


class EasyMagpieGPUARWorker(GPUARWorker):
    model_runner_cls = EasyMagpieGPUARModelRunner

    def init_device(self):
        configure_inference(self.vllm_config)
        if getattr(self.vllm_config.model_config.hf_config, "easymagpie_optimized", False):
            sys.setswitchinterval(0.001)
        return super().init_device()


class EasyMagpieCodecWorker(GPUGenerationWorker):
    def init_device(self):
        configure_inference(self.vllm_config, codec=True)
        return super().init_device()

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        if getattr(self.vllm_config.model_config.hf_config, "easymagpie_optimized", False):
            capacity = min(128, self.vllm_config.scheduler_config.max_num_seqs)
            with torch.inference_mode():
                self.model_runner.model.prewarm_codec_graphs(range(1, capacity + 1), descending=True)
            # Graphs live for the worker lifetime; keep them out of request-time
            # cyclic GC scans, matching the reference codec startup policy.
            gc.freeze()
        return result


def configure_inference(vllm_config, *, codec: bool = False) -> None:
    """Select the C256 numerical and buffer policy before constructing a worker."""
    if not getattr(vllm_config.model_config.hf_config, "easymagpie_optimized", False):
        return
    if os.environ.get("NVIDIA_TF32_OVERRIDE") is not None:
        raise ValueError("Unset NVIDIA_TF32_OVERRIDE for the EasyMagpie optimized precision policy")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # The reference talker retains FP16 reductions; only the FP16 codec
    # requires FP32 reductions. Applying the codec setting to both changes
    # the talker's GEMM policy as well as its performance.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = not codec
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    for option in (
        "DEVICE_DECODE_STATE",
        "PREFILL_GRAPHS",
        "BATCHED_OWNED_FEEDBACK",
        "BATCHED_DECODE_INPUTS",
        "DDS_RETAIN_LIVE_LEASES",
        "BATCHED_FEEDBACK_MERGE",
        "DDS_ROUTING_POOL",
        "QUERY_DISPATCH_POOL",
        "LOCAL_KV_PREFIX_LIST",
        "CODEC_LEASED_BUFFERS",
        "VLLM_CODEC_CUDA_GRAPHS",
        "VLLM_CODEC_UNIFORM_GROUPS",
        "VLLM_CODEC_CUDNN_TRANSPOSE",
        "VLLM_CODEC_SGLANG_ACTIVATION",
        "VLLM_CODEC_SGLANG_ACTIVATION_FUSED",
        "VLLM_CODEC_FP16_ACTIVATION_FUSED",
        "VLLM_CODEC_SHARED_METADATA_GUARD",
        "VLLM_CODEC_BATCHED_PAYLOAD_VALIDITY",
    ):
        os.environ.setdefault(f"EASYMAGPIE_{option}", "1")
    os.environ.setdefault("EASYMAGPIE_VLLM_CODEC_GRAPH_MAX_ENTRIES", "384")
