# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/modeling_bailingmm.py
from __future__ import annotations

import os
import time
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.bailing_moe import BailingMoeModel
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.common.ming.aggregator import Aggregator
from vllm_omni.model_executor.models.common.ming.cfm_cudagraph import (
    CFMGraphExecutor,
    CFMSampler,
)
from vllm_omni.model_executor.models.common.ming.cfm_head import FlowLoss
from vllm_omni.model_executor.models.common.ming.cfm_solver import DEFAULT_CFM_STEPS
from vllm_omni.model_executor.models.common.ming.continuous_tts import STOP_REASON_CODES
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

from .config_ming_tts import (
    KEY_CFG,
    KEY_DECODE_STEP,
    KEY_LATENT_HISTORY,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_NEXT_EMBEDS,
    KEY_REQUEST_ID,
    KEY_SIGMA,
    KEY_STOP_REASON,
    KEY_TEMPERATURE,
    KEY_TEXT_MODE,
    MingTTSConfig,
)
from .constants import SPEAKER_EMBEDDING_DIM
from .patch_emission import (
    _coerce_latent_history,
    _get_request_token_counts,
    _normalize_request_infos,
    _resolve_ming_stop_decision,
    _resolve_optional_runtime_int,
    _resolve_runtime_float,
    _resolve_runtime_int,
    _validate_ming_decode_window,
)

logger = init_logger(__name__)


class MingLLMModel(nn.Module):
    latent_patch_key = "ming_latent_patch"
    next_embeds_key = "ming_next_embeds"
    new_history_key = "ming_new_history"
    decode_step_key = "ming_decode_step"
    has_patch_key = "ming_has_patch"
    stop_reason_key = KEY_STOP_REASON
    stop_probability_key = "ming_stop_prob"
    stop_reason_codes = STOP_REASON_CODES
    vectorize_cfm_decode = False
    # Drop the tensor on purpose, otherwise upstream vLLM would reject.
    # The vendor never reaches them either (audio_gate/image_gate never fire)
    # dropping them keeps routing identical to the reference implementation.
    # https://github.com/inclusionAI/Ming-omni-tts/blob/main/modeling_bailingmm.py#L427-L428
    # https://github.com/inclusionAI/Ming-omni-tts/blob/main/modeling_bailing_moe.py#L510-L520
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".mlp.audio_gate.": None,
            ".mlp.image_gate.": None,
        },
        orig_to_new_prefix={
            "model.lm_head.": None,
            "model.model.": "model.",
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.ming_config = MingTTSConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.fm_dtype = _resolve_ming_runtime_dtype(vllm_config)
        if self.ming_config.model_variant == "moe":
            # BailingMoeModel reads ``vllm_config.model_config.hf_config`` directly
            # (no get_text_config()), so re-wrap with the nested bailing_moe config.
            llm_config = vllm_config.model_config.hf_config.get_text_config()
            llm_vllm_config = vllm_config.with_hf_config(llm_config)
            self.model = BailingMoeModel(vllm_config=llm_vllm_config, prefix=maybe_prefix(prefix, "model"))
        else:
            self.model = Qwen2Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.linear_proj_audio = Aggregator(
            in_channels=self.ming_config.latent_dim,
            llm_input_dim=self.ming_config.llm_hidden_size,
            **self.ming_config.aggregator_config,
        )
        self.flowloss = FlowLoss(
            z_channels=self.ming_config.latent_dim,
            llm_cond_dim=self.ming_config.llm_hidden_size,
            **self.ming_config.ditar_config,
        )
        self.stop_head = nn.Linear(self.ming_config.llm_hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(SPEAKER_EMBEDDING_DIM, self.ming_config.llm_hidden_size, bias=True)
        self.flowloss.to(dtype=self.fm_dtype)
        self.linear_proj_audio.to(dtype=self.fm_dtype)
        self.stop_head.to(dtype=self.fm_dtype)
        self.spk_head.to(dtype=self.fm_dtype)
        self._pending_postprocess_updates: dict[str, dict[str, Any]] = {}
        self._last_ming_next_token_ids: list[int] | None = None
        self._last_text_mode = False
        # CUDAGraph for the flow-matching (CFM) diffusion head — the TTFP/latency
        # hot spot. The legacy MING_CFM_CUDAGRAPH switch controls this
        # independently of the backbone's eager/graph configuration.
        self._init_cfm_graph_state("MING_CFM_CUDAGRAPH")

    def embed_input_ids(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor | None = None, **_: Any
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            return inputs_embeds
        return self.model.embed_input_ids(input_ids)

    def project_speaker_embedding(self, spk_emb: torch.Tensor) -> torch.Tensor:
        return self.spk_head(spk_emb)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        latent_history: torch.Tensor | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        seq_token_counts: list[int] | None = None,
        request_token_spans: list[tuple[int, int]] | None = None,
        request_sample_eligible: list[bool] | None = None,
        **kwargs: object,
    ) -> OmniOutput | IntermediateTensors | torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        runtime_information = (
            model_intermediate_buffer
            if model_intermediate_buffer is not None
            else kwargs.get("runtime_additional_information")
        )
        request_infos = _normalize_request_infos(runtime_information)
        self._validate_decode_window(request_infos)
        # Positional call: Qwen2Model names the 2nd arg ``positions`` while
        # BailingMoeModel names it ``position_ids`` — both share the signature
        # (input_ids, positions, intermediate_tensors, inputs_embeds).
        backbone_out = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )
        if isinstance(backbone_out, IntermediateTensors):
            self._last_ming_next_token_ids = None
            return backbone_out
        hidden_states = _extract_hidden_states(backbone_out)
        token_counts = _get_request_token_counts(
            hidden_states,
            request_infos,
            seq_token_counts,
            request_token_spans,
        )
        text_mode = bool(request_infos) and all(bool(info.get(KEY_TEXT_MODE, False)) for info in request_infos)
        if request_infos and any(bool(info.get(KEY_TEXT_MODE, False)) for info in request_infos) and not text_mode:
            raise RuntimeError("Mixed Ming text/audio modes in one Stage-0 batch are unsupported.")
        if text_mode:
            self._last_ming_next_token_ids = None
            self._last_text_mode = True
            return OmniOutput(
                text_hidden_states=hidden_states,
                multimodal_outputs={KEY_TEXT_MODE: True},
                intermediate_tensors=intermediate_tensors,
            )
        self._last_text_mode = False
        if latent_history is None and not token_counts:
            self._last_ming_next_token_ids = None
            return OmniOutput(
                text_hidden_states=hidden_states, multimodal_outputs=None, intermediate_tensors=intermediate_tensors
            )
        if latent_history is not None and not token_counts:
            token_counts, request_infos = [hidden_states.shape[0]], [{KEY_LATENT_HISTORY: latent_history}]

        if request_sample_eligible is None:
            sample_eligible = [True] * len(token_counts)
        elif len(request_sample_eligible) != len(token_counts):
            raise RuntimeError(
                "Ming request_sample_eligible/request span mismatch: "
                f"got {len(request_sample_eligible)} flags for {len(token_counts)} requests."
            )
        else:
            sample_eligible = [bool(flag) for flag in request_sample_eligible]

        total_tokens = hidden_states.shape[0]
        latent_patch_tokens = next_embed_tokens = new_history_tokens = None
        decode_step_tokens = has_patch = stop_reason_code_tokens = None
        next_token_ids = [int(self.ming_config.text_eos_token_id)] * len(token_counts)
        pending_updates: dict[str, dict[str, Any]] = {}
        cursor = 0
        any_decode = False
        vectorized_results = self._vectorized_decode_results(
            hidden_states,
            request_infos,
            token_counts,
            sample_eligible,
        )
        for req_idx, token_count in enumerate(token_counts):
            end = min(cursor + token_count, total_tokens)
            if end <= cursor:
                continue
            req_info = request_infos[req_idx] if req_idx < len(request_infos) else {}
            if not sample_eligible[req_idx]:
                cursor = end
                continue
            req_id = req_info.get(KEY_REQUEST_ID)
            req_history = _coerce_latent_history(
                req_info.get(KEY_LATENT_HISTORY), device=hidden_states.device, dtype=self.fm_dtype, cfg=self.ming_config
            )
            if req_history is None:
                cursor = end
                continue
            decode_step = int(req_info.get(KEY_DECODE_STEP, req_info.get("generated_len", 0)))
            decode_hidden, output_index = (
                (hidden_states[cursor:end], cursor) if token_count == 1 else (hidden_states[end - 1 : end], end - 1)
            )
            decode_result = vectorized_results.get(req_idx)
            if decode_result is None:
                decode_result = self._decode_one_step(
                    hidden_states=decode_hidden,
                    latent_history=req_history,
                    cfg_scale=_resolve_runtime_float(req_info, KEY_CFG, self.ming_config.cfg),
                    sigma=_resolve_runtime_float(req_info, KEY_SIGMA, self.ming_config.sigma),
                    temperature=_resolve_runtime_float(req_info, KEY_TEMPERATURE, self.ming_config.temperature),
                    generator=self._get_request_generator(req_info, hidden_states.device),
                )
            sampled_token_latent, next_embeds, new_history, stop_probs = decode_result
            req_max_decode_steps = _resolve_runtime_int(
                req_info, KEY_MAX_DECODE_STEPS, self.ming_config.max_decode_steps
            )
            req_min_decode_steps = _resolve_optional_runtime_int(req_info, KEY_MIN_DECODE_STEPS, 0)
            if latent_patch_tokens is None:
                latent_patch_tokens = sampled_token_latent.new_zeros(
                    (total_tokens, self.ming_config.patch_size, self.ming_config.latent_dim)
                )
                next_embed_tokens = next_embeds.new_zeros((total_tokens, 1, self.ming_config.llm_hidden_size))
                new_history_tokens = new_history.new_zeros(
                    (total_tokens, self.ming_config.history_patch_size, self.ming_config.latent_dim)
                )
                decode_step_tokens = torch.zeros((total_tokens,), dtype=torch.int32, device=hidden_states.device)
                has_patch = torch.zeros((total_tokens,), dtype=torch.bool, device=hidden_states.device)
                stop_reason_code_tokens = torch.zeros((total_tokens,), dtype=torch.int32, device=hidden_states.device)
            latent_patch_tokens[output_index : output_index + 1] = sampled_token_latent
            next_embed_tokens[output_index : output_index + 1] = next_embeds
            new_history_tokens[output_index : output_index + 1] = new_history
            decode_step_tokens[output_index : output_index + 1] = decode_step
            has_patch[output_index : output_index + 1] = True
            stop_reason, _, _, _, next_token_id = self._resolve_stop_decision(
                step=decode_step,
                stop_prob=float(stop_probs.reshape(-1)[0].item()),
                stop_threshold=float(self.ming_config.stop_head_threshold),
                min_stop_step=int(self.ming_config.stop_head_min_steps),
                min_decode_steps=req_min_decode_steps,
                max_decode_steps=req_max_decode_steps,
                audio_dummy_token_id=int(self.ming_config.audio_dummy_token_id),
                text_eos_token_id=int(self.ming_config.text_eos_token_id),
            )
            stop_reason_code_tokens[output_index : output_index + 1] = self.stop_reason_codes[stop_reason]
            self._on_stop_decision(req_info, stop_reason)
            next_token_ids[req_idx] = int(next_token_id)
            if isinstance(req_id, str):
                pending_updates[req_id] = {
                    KEY_LATENT_HISTORY: new_history,
                    KEY_NEXT_EMBEDS: next_embeds,
                    self.latent_patch_key: sampled_token_latent,
                    self.stop_probability_key: stop_probs,
                    self.stop_reason_key: stop_reason,
                }
            any_decode = True
            cursor = end

        self._pending_postprocess_updates = pending_updates
        if not any_decode:
            self._last_ming_next_token_ids = None
            return OmniOutput(
                text_hidden_states=hidden_states, multimodal_outputs=None, intermediate_tensors=intermediate_tensors
            )
        self._last_ming_next_token_ids = next_token_ids
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs={
                self.latent_patch_key: latent_patch_tokens,
                self.next_embeds_key: next_embed_tokens,
                self.new_history_key: new_history_tokens,
                self.decode_step_key: decode_step_tokens,
                self.has_patch_key: has_patch,
                self.stop_reason_key: stop_reason_code_tokens,
            },
            intermediate_tensors=intermediate_tensors,
        )

    def pop_postprocess_update(self, req_id: str) -> dict[str, Any]:
        return self._pending_postprocess_updates.pop(req_id, {}) if isinstance(req_id, str) else {}

    def _validate_decode_window(self, request_infos: list[dict[str, Any]]) -> None:
        _validate_ming_decode_window(
            request_infos,
            min_stop_step=int(self.ming_config.stop_head_min_steps),
            default_max_decode_steps=self.ming_config.max_decode_steps,
        )

    def _resolve_stop_decision(self, **kwargs: Any) -> tuple[str, bool, bool, int, int]:
        return _resolve_ming_stop_decision(**kwargs)

    def _get_request_generator(self, req_info: dict[str, Any], device: torch.device) -> torch.Generator | None:
        """Return a request-scoped RNG; No-op for this base MingLLMModel - use the default RNG."""
        del req_info, device
        return None

    def _on_stop_decision(self, req_info: dict[str, Any], stop_reason: str) -> None:
        """Handle stop-related cleanup; No-op for this base MingLLMModel."""
        del req_info, stop_reason

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor | None:
        del sampling_metadata
        if self._last_text_mode:
            return (
                None
                if hidden_states is None or hidden_states.numel() == 0
                else self.model.compute_logits(hidden_states)
            )
        if hidden_states is None or hidden_states.numel() == 0:
            return None
        if hidden_states.dim() != 2:
            raise RuntimeError(
                f"Expected hidden_states rank-2 [B,H] in compute_logits, got {tuple(hidden_states.shape)}"
            )
        batch_size = hidden_states.shape[0]
        next_token_ids = self._last_ming_next_token_ids
        self._last_ming_next_token_ids = None
        if next_token_ids is None:
            logger.debug(
                "Missing Ming forced next-token ids before compute_logits. "
                "Using dummy next-token IDs (likely during a profiling or dummy run)."
            )
            next_token_ids = [int(self.ming_config.text_eos_token_id)] * batch_size
        if len(next_token_ids) != batch_size:
            raise RuntimeError(
                "Ming forced next-token batch mismatch: "
                f"got {len(next_token_ids)} ids for hidden_states batch {batch_size}."
            )
        logits = torch.full(
            (batch_size, self.ming_config.llm_vocab_size),
            float("-inf"),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        for i, next_token_id in enumerate(next_token_ids):
            logits[i, int(next_token_id)] = 0.0
        return logits

    def sample(self, logits, sampling_metadata):
        if logits is None:
            return None
        if self._last_text_mode:
            return self.model.sample(logits, sampling_metadata)
        del sampling_metadata
        return SamplerOutput(
            sampled_token_ids=logits.argmax(dim=-1, keepdim=True).to(dtype=torch.int32),
            logprobs_tensors=None,
        )

    def _init_cfm_graph_state(self, env_var: str | None = None) -> None:
        compilation_config = self.vllm_config.compilation_config
        if env_var is not None:
            self._cfm_graph_enabled = os.environ.get(env_var, "1") == "1"
        else:
            self._cfm_graph_enabled = (
                not self.vllm_config.model_config.enforce_eager
                and compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            )
        self._cfm_graphs: dict[int, CFMGraphExecutor] = {}
        self._cfm_graph_failed_batch_sizes: set[int] = set()
        self._cfm_graph_warmed_up = False

        raw_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None) or [1]
        scheduler_config = getattr(self.vllm_config, "scheduler_config", None)
        max_num_seqs = max(1, int(getattr(scheduler_config, "max_num_seqs", 1)))
        self._cfm_graph_batch_sizes = sorted({int(size) for size in raw_sizes if 0 < int(size) <= max_num_seqs})
        self._cfm_graph_num_warmups = max(
            1,
            int(getattr(compilation_config, "cudagraph_num_of_warmups", 1)),
        )

    def _select_cfm_graph_batch_size(self, actual_batch_size: int) -> int | None:
        return next((size for size in self._cfm_graph_batch_sizes if size >= actual_batch_size), None)

    @torch.no_grad()
    def _warmup_cfm_graphs(self) -> None:
        if self._cfm_graph_warmed_up or not self._cfm_graph_enabled:
            return
        device = next(self.flowloss.parameters()).device
        if device.type != "cuda" or not self._cfm_graph_batch_sizes:
            return
        self._cfm_graph_warmed_up = True
        self.flowloss.eval()
        self.linear_proj_audio.eval()
        self.stop_head.eval()

        logger.info(
            "Ming CFM CUDA Graph warmup: batch_sizes=%s warmups=%d",
            self._cfm_graph_batch_sizes,
            self._cfm_graph_num_warmups,
        )
        start_s = time.perf_counter()
        with current_omni_platform.device(device):
            for batch_size in reversed(self._cfm_graph_batch_sizes):
                if batch_size in self._cfm_graph_failed_batch_sizes:
                    continue
                try:
                    hidden_states = torch.zeros(
                        (batch_size, self.ming_config.llm_hidden_size),
                        device=device,
                        dtype=self.fm_dtype,
                    )
                    latent_history = torch.zeros(
                        (
                            batch_size,
                            self.ming_config.history_patch_size,
                            self.ming_config.latent_dim,
                        ),
                        device=device,
                        dtype=self.fm_dtype,
                    )
                    initial_noise = torch.zeros(
                        (
                            batch_size,
                            self.ming_config.patch_size,
                            self.ming_config.latent_dim,
                        ),
                        device=device,
                        dtype=torch.float32,
                    )
                    solver_noise = torch.zeros(
                        (
                            DEFAULT_CFM_STEPS,
                            batch_size,
                            self.ming_config.patch_size,
                            self.ming_config.latent_dim,
                        ),
                        device=device,
                        dtype=torch.float32,
                    )
                    self._decode_one_step(
                        hidden_states=hidden_states,
                        latent_history=latent_history,
                        cfg_scale=float(self.ming_config.cfg),
                        sigma=float(self.ming_config.sigma),
                        temperature=float(self.ming_config.temperature),
                        initial_noise=initial_noise,
                        solver_noise=solver_noise,
                    )
                except Exception:
                    self._cfm_graphs.pop(batch_size, None)
                    self._cfm_graph_failed_batch_sizes.add(batch_size)
                    logger.warning(
                        "Ming CFM CUDA Graph warmup failed for batch=%d; using eager for this shape.",
                        batch_size,
                        exc_info=True,
                    )
        torch.accelerator.synchronize(device)
        captured = sorted(batch_size for batch_size, graph_exec in self._cfm_graphs.items() if graph_exec.initialized)
        logger.info(
            "Ming CFM CUDA Graph warmup complete: captured=%s failed=%s in %.1f ms",
            captured,
            sorted(self._cfm_graph_failed_batch_sizes),
            (time.perf_counter() - start_s) * 1000.0,
        )

    def _vectorized_decode_results(
        self,
        hidden_states: torch.Tensor,
        request_infos: list[dict[str, Any]],
        token_counts: list[int],
        sample_eligible: list[bool],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Batch compatible flow-head calls while preserving request RNGs."""
        if not self.vectorize_cfm_decode or len(token_counts) < 2:
            return {}

        records = []
        cursor = 0
        total_tokens = hidden_states.shape[0]
        for req_idx, token_count in enumerate(token_counts):
            end = min(cursor + token_count, total_tokens)
            if end <= cursor:
                continue
            if not sample_eligible[req_idx]:
                cursor = end
                continue
            req_info = request_infos[req_idx] if req_idx < len(request_infos) else {}
            req_history = _coerce_latent_history(
                req_info.get(KEY_LATENT_HISTORY),
                device=hidden_states.device,
                dtype=self.fm_dtype,
                cfg=self.ming_config,
            )
            if req_history is not None:
                decode_hidden = hidden_states[cursor:end] if token_count == 1 else hidden_states[end - 1 : end]
                records.append(
                    (
                        req_idx,
                        req_info,
                        decode_hidden,
                        req_history,
                        _resolve_runtime_float(req_info, KEY_CFG, self.ming_config.cfg),
                        _resolve_runtime_float(req_info, KEY_SIGMA, self.ming_config.sigma),
                        _resolve_runtime_float(req_info, KEY_TEMPERATURE, self.ming_config.temperature),
                    )
                )
            cursor = end

        groups: dict[tuple[Any, ...], list[int]] = {}
        for record_idx, record in enumerate(records):
            _, _, decode_hidden, req_history, cfg_scale, sigma, temperature = record
            key = (
                cfg_scale,
                sigma,
                temperature,
                tuple(decode_hidden.shape[1:]),
                tuple(req_history.shape[1:]),
                decode_hidden.dtype,
                req_history.dtype,
                decode_hidden.device,
                req_history.device,
            )
            groups.setdefault(key, []).append(record_idx)

        results = {}
        for record_indices in groups.values():
            if len(record_indices) < 2:
                continue
            selected = [records[index] for index in record_indices]
            generators = [self._get_request_generator(record[1], hidden_states.device) for record in selected]
            initial_noise, solver_noise = self._draw_request_cfm_noises(generators, hidden_states.device)
            batch_result = self._decode_one_step(
                hidden_states=torch.cat([record[2] for record in selected], dim=0),
                latent_history=torch.cat([record[3] for record in selected], dim=0),
                cfg_scale=selected[0][4],
                sigma=selected[0][5],
                temperature=selected[0][6],
                initial_noise=initial_noise,
                solver_noise=solver_noise,
            )
            for batch_idx, record in enumerate(selected):
                results[record[0]] = tuple(tensor[batch_idx : batch_idx + 1] for tensor in batch_result)
        return results

    def _draw_request_cfm_noises(
        self,
        generators: list[torch.Generator | None],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial_noise = []
        solver_noise = []
        shape = (1, self.ming_config.patch_size, self.ming_config.latent_dim)
        for generator in generators:
            initial_noise.append(
                torch.randn(
                    (1, self.ming_config.latent_dim, self.ming_config.patch_size),
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                ).transpose(1, 2)
            )
            # Match the active batch-1 path's RNG call boundaries so a fixed
            # request seed is stable when scheduler concurrency changes.
            if self._cfm_graph_enabled:
                request_solver_noise = torch.randn(
                    (DEFAULT_CFM_STEPS, *shape),
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
            else:
                request_solver_noise = torch.stack(
                    [
                        torch.randn(
                            shape,
                            device=device,
                            dtype=torch.float32,
                            generator=generator,
                        )
                        for _ in range(DEFAULT_CFM_STEPS)
                    ]
                )
            solver_noise.append(request_solver_noise)
        return torch.cat(initial_noise, dim=0), torch.cat(solver_noise, dim=1)

    def _maybe_build_cfm_graph(self, z_diff_cond: torch.Tensor):
        """Return the smallest configured graph bucket for this CUDA batch."""
        if not self._cfm_graph_enabled or not z_diff_cond.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        actual_batch_size = int(z_diff_cond.shape[0])
        batch_size = self._select_cfm_graph_batch_size(actual_batch_size)
        if batch_size is None or batch_size in self._cfm_graph_failed_batch_sizes:
            return None
        graph_exec = self._cfm_graphs.get(batch_size)
        if graph_exec is not None:
            return graph_exec
        try:
            # TODO(perf): move the compile into a vLLM support_torch_compile
            # adapter once the shared Ming CFM modules are migrated.
            dit_model = self.flowloss.cfm.model
            if not getattr(dit_model, "_ming_compiled", False):
                try:
                    dit_model.forward = torch.compile(
                        dit_model.forward,
                        fullgraph=False,
                        dynamic=False,
                        options={
                            "triton.cudagraphs": False,
                            "triton.cudagraph_trees": False,
                        },
                    )
                    dit_model._ming_compiled = True
                    logger.info("Ming CFM DiT torch.compile enabled (inductor cudagraphs off).")
                except Exception as exc:
                    logger.warning("Ming CFM torch.compile failed (%s); using uncompiled DiT.", exc)

            graph_exec = CFMGraphExecutor(
                CFMSampler(self.flowloss.cfm.model, steps=DEFAULT_CFM_STEPS),
                self.linear_proj_audio,
                self.stop_head,
                patch_size=self.ming_config.patch_size,
                latent_dim=self.ming_config.latent_dim,
                batch_size=batch_size,
                steps=DEFAULT_CFM_STEPS,
                num_warmups=self._cfm_graph_num_warmups,
            )
            self._cfm_graphs[batch_size] = graph_exec
        except Exception:
            self._cfm_graphs.pop(batch_size, None)
            self._cfm_graph_failed_batch_sizes.add(batch_size)
            logger.warning(
                "Ming CFM CUDA Graph setup failed for batch=%d; using eager for this shape.",
                batch_size,
                exc_info=True,
            )
            return None
        return graph_exec

    def _decode_one_step(
        self,
        *,
        hidden_states: torch.Tensor,
        latent_history: torch.Tensor,
        cfg_scale: float,
        sigma: float,
        temperature: float,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
        solver_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.dim() != 2:
            raise RuntimeError(f"Expected decode hidden_states rank-2 [B,H], got {tuple(hidden_states.shape)}")
        if latent_history.dim() != 3:
            raise RuntimeError(f"Expected latent_history rank-3 [B,T,D], got {tuple(latent_history.shape)}")
        if hidden_states.shape[0] != latent_history.shape[0]:
            raise RuntimeError(
                "Batch mismatch: "
                f"hidden_states B={hidden_states.shape[0]} "
                f"vs latent_history B={latent_history.shape[0]}"
            )
        # [Batch, Hidden] -> [Batch, Time, Hidden] = [B, 1, H] for FlowLoss conditioning.
        z_diff_cond = hidden_states.to(dtype=self.fm_dtype).unsqueeze(1)
        if not torch.isfinite(z_diff_cond).all():
            raise RuntimeError("Non-finite z_diff_cond before FlowLoss.sample().")

        # Fast path: CUDAGraph-captured flow head (CFM sampling + Aggregator +
        # stop head). A failed batch shape falls back to eager independently.
        sampled_token_latent = next_embeds = stop_probs = None
        # cfg < 1e-5 disables CFG; the eager flow head handles this with a
        # dedicated unconditional branch (c=zeros) that the captured graph does
        # not replicate, so fall back to eager there to match FlowLoss.sample.
        graph_exec = self._maybe_build_cfm_graph(z_diff_cond) if cfg_scale >= 1e-5 else None
        if graph_exec is not None:
            try:
                graph_result = graph_exec.execute(
                    z_diff_cond,
                    latent_history,
                    cfg_scale,
                    sigma,
                    temperature,
                    generator=generator,
                    initial_noise=initial_noise,
                    solver_noise=solver_noise,
                )
                if graph_result is not None:
                    sampled_token_latent, next_embeds, stop_full = graph_result
                    stop_probs = stop_full[:, 1]
            except Exception:
                batch_size = graph_exec.batch_size
                self._cfm_graphs.pop(batch_size, None)
                self._cfm_graph_failed_batch_sizes.add(batch_size)
                logger.warning(
                    "Ming CFM CUDA Graph failed for batch=%d; using eager for this shape.",
                    batch_size,
                    exc_info=True,
                )
                sampled_token_latent = next_embeds = stop_probs = None

        if sampled_token_latent is None:
            sampled_token_latent = self.flowloss.sample(
                z=z_diff_cond,
                latent_history=latent_history,
                cfg=cfg_scale,
                patch_size=self.ming_config.patch_size,
                sigma=sigma,
                temperature=temperature,
                generator=generator,
                initial_noise=initial_noise,
                solver_noise=solver_noise,
            )
            # Aggregator expects [Batch, Time, Dimension] = [B, 4, 64] and returns [B, 1, H].
            next_embeds = self.linear_proj_audio(sampled_token_latent)
            stop_probs = self.stop_head(hidden_states.to(dtype=self.fm_dtype)).softmax(dim=-1)[:, 1]

        expected_shape = (hidden_states.shape[0], self.ming_config.patch_size, self.ming_config.latent_dim)
        if tuple(sampled_token_latent.shape) != expected_shape:
            raise RuntimeError(
                f"FlowLoss output shape mismatch: got {tuple(sampled_token_latent.shape)}, expected {expected_shape}"
            )
        new_history = torch.cat([latent_history[:, self.ming_config.patch_size :, :], sampled_token_latent], dim=1)
        if not torch.isfinite(sampled_token_latent).all():
            raise RuntimeError("Non-finite sampled_token_latent in Ming decode step.")
        if not torch.isfinite(next_embeds).all():
            raise RuntimeError("Non-finite next_embeds in Ming decode step.")
        if not torch.isfinite(stop_probs).all():
            raise RuntimeError("Non-finite stop_probs in Ming decode step.")
        return sampled_token_latent, next_embeds, new_history, stop_probs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params = AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
        _warn_missing_prefix("flowloss", params_dict, loaded_params, prefix="flowloss.")
        _warn_missing_prefix("linear_proj_audio", params_dict, loaded_params, prefix="linear_proj_audio.")
        _warn_missing_prefix("stop_head", params_dict, loaded_params, prefix="stop_head.")
        _warn_missing_prefix("spk_head", params_dict, loaded_params, prefix="spk_head.")
        return loaded_params


def _extract_hidden_states(backbone_out: object) -> torch.Tensor:
    if isinstance(backbone_out, torch.Tensor):
        return backbone_out
    if hasattr(backbone_out, "last_hidden_state"):
        return backbone_out.last_hidden_state
    if isinstance(backbone_out, (tuple, list)) and backbone_out and isinstance(backbone_out[0], torch.Tensor):
        return backbone_out[0]
    raise TypeError(f"Unsupported backbone forward output type: {type(backbone_out)}")


def _resolve_ming_runtime_dtype(vllm_config: VllmConfig) -> torch.dtype:
    dtype = getattr(vllm_config.model_config, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        normalized = dtype.strip().lower()
        if normalized in ("float16", "half", "torch.float16"):
            return torch.float16
        if normalized in ("bfloat16", "bf16", "torch.bfloat16"):
            return torch.bfloat16
        if normalized in ("float32", "fp32", "torch.float32"):
            return torch.float32
    return torch.float32


def _warn_missing_prefix(
    module_name: str,
    params_dict: dict[str, nn.Parameter],
    loaded_params: set[str],
    prefix: str,
) -> None:
    missing = {key for key in params_dict if key.startswith(prefix)} - loaded_params
    if not missing:
        return
    msg = (
        f"MingLLMModel: {len(missing)} {module_name} params not loaded "
        f"(prefix={prefix}). First few: {sorted(missing)[:5]}"
    )
    raise RuntimeError(msg)


__all__ = [
    "Aggregator",
    "FlowLoss",
    "MingLLMModel",
    "_coerce_latent_history",
    "_extract_hidden_states",
    "_resolve_ming_runtime_dtype",
    "_resolve_ming_stop_decision",
    "_warn_missing_prefix",
]
