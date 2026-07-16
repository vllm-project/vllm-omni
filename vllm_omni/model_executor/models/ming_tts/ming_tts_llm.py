# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/modeling_bailingmm.py
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .aggregator import Aggregator
from .config_ming_tts import (
    KEY_CFG,
    KEY_DECODE_STEP,
    KEY_LATENT_HISTORY,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_NEXT_EMBEDS,
    KEY_REQUEST_ID,
    KEY_SIGMA,
    KEY_TEMPERATURE,
    KEY_TEXT_MODE,
    MingTTSConfig,
)
from .flowloss_head import FlowLoss
from .patch_emission import (
    MING_STOP_REASON_CODES,
    MING_STOP_REASON_CONTINUE,
    MING_STOP_REASON_KEY,
    MING_STOP_REASON_MAX_DECODE_STEPS,
    MING_STOP_REASON_STOP_HEAD,
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

# CFM ODE integration steps; must match CFM.sample(steps=...) default (10).
_CFM_STEPS = 10


class MingLLMModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.model.": "model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.ming_config = MingTTSConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.fm_dtype = _resolve_ming_runtime_dtype(vllm_config)
        if self.ming_config.model_variant == "moe":
            from vllm.model_executor.models.bailing_moe import BailingMoeModel

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
        self.spk_head = nn.Linear(192, self.ming_config.llm_hidden_size, bias=True)
        self.flowloss.to(dtype=self.fm_dtype)
        self.linear_proj_audio.to(dtype=self.fm_dtype)
        self.stop_head.to(dtype=self.fm_dtype)
        self.spk_head.to(dtype=self.fm_dtype)
        self._pending_postprocess_updates: dict[str, dict[str, Any]] = {}
        self._last_ming_next_token_ids = None
        self._last_text_mode = False
        # CUDAGraph for the flow-matching (CFM) diffusion head — the TTFP/latency
        # hot spot. Built lazily on the first batch=1 CUDA decode; set
        # MING_CFM_CUDAGRAPH=0 to force the eager flow head.
        self._cfm_graph = None
        self._cfm_graph_enabled = os.environ.get("MING_CFM_CUDAGRAPH", "1") == "1"

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
        **kwargs: object,
    ) -> OmniOutput | IntermediateTensors | torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        if model_intermediate_buffer is None:
            model_intermediate_buffer = kwargs.get("runtime_additional_information")
        request_infos = _normalize_request_infos(model_intermediate_buffer)
        _validate_ming_decode_window(
            request_infos,
            min_stop_step=int(self.ming_config.stop_head_min_steps),
            default_max_decode_steps=self.ming_config.max_decode_steps,
        )
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
        token_counts = _get_request_token_counts(hidden_states, request_infos, seq_token_counts)
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

        total_tokens = hidden_states.shape[0]
        latent_patch_tokens = next_embed_tokens = new_history_tokens = None
        decode_step_tokens = has_patch = stop_reason_code_tokens = None
        next_token_ids = []
        pending_updates: dict[str, dict[str, Any]] = {}
        cursor = 0
        any_decode = False
        for req_idx, token_count in enumerate(token_counts):
            end = min(cursor + token_count, total_tokens)
            if end <= cursor:
                continue
            req_info = request_infos[req_idx] if req_idx < len(request_infos) else {}
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
            sampled_token_latent, next_embeds, new_history, stop_probs = self._decode_one_step(
                hidden_states=decode_hidden,
                latent_history=req_history,
                cfg_scale=_resolve_runtime_float(req_info, KEY_CFG, self.ming_config.cfg),
                sigma=_resolve_runtime_float(req_info, KEY_SIGMA, self.ming_config.sigma),
                temperature=_resolve_runtime_float(req_info, KEY_TEMPERATURE, self.ming_config.temperature),
            )
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
            stop_reason, _, _, _, next_token_id = _resolve_ming_stop_decision(
                step=decode_step,
                stop_prob=float(stop_probs.reshape(-1)[0].item()),
                stop_threshold=float(self.ming_config.stop_head_threshold),
                min_stop_step=int(self.ming_config.stop_head_min_steps),
                min_decode_steps=req_min_decode_steps,
                max_decode_steps=req_max_decode_steps,
                audio_dummy_token_id=int(self.ming_config.audio_dummy_token_id),
                text_eos_token_id=int(self.ming_config.text_eos_token_id),
            )
            stop_reason_code_tokens[output_index : output_index + 1] = MING_STOP_REASON_CODES[stop_reason]
            next_token_ids.append(int(next_token_id))
            if isinstance(req_id, str):
                pending_updates[req_id] = {
                    KEY_LATENT_HISTORY: new_history,
                    KEY_NEXT_EMBEDS: next_embeds,
                    "ming_latent_patch": sampled_token_latent,
                    "ming_stop_prob": stop_probs,
                    MING_STOP_REASON_KEY: stop_reason,
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
                "ming_latent_patch": latent_patch_tokens,
                "ming_next_embeds": next_embed_tokens,
                "ming_new_history": new_history_tokens,
                "ming_decode_step": decode_step_tokens,
                "ming_has_patch": has_patch,
                MING_STOP_REASON_KEY: stop_reason_code_tokens,
            },
            intermediate_tensors=intermediate_tensors,
        )

    def pop_postprocess_update(self, req_id: str) -> dict[str, Any]:
        return self._pending_postprocess_updates.pop(req_id, {}) if isinstance(req_id, str) else {}

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

    def _maybe_build_cfm_graph(self, z_diff_cond: torch.Tensor):
        """Lazily build the CFM CUDAGraph executor (batch=1, CUDA only)."""
        if not self._cfm_graph_enabled:
            return None
        if z_diff_cond.shape[0] != 1 or not z_diff_cond.is_cuda:
            return None
        if self._cfm_graph is not None:
            return self._cfm_graph
        try:
            from .fm.cfm_cudagraph import CFMGraphExecutor, CFMSampler

            # TODO(perf):
            #   (1) compile per-block (dit_model.blocks) instead of the
            #   whole forward for tighter fusion / fewer graph breaks;
            #   (2) move the compile + a synthetic warmup forward to a post-weight-load hook
            dit_model = self.flowloss.cfm.model
            if not getattr(dit_model, "_ming_compiled", False):
                try:
                    # Scoped options (avoid mutating global inductor config).
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

            sampler = CFMSampler(self.flowloss.cfm.model, steps=_CFM_STEPS)
            self._cfm_graph = CFMGraphExecutor(
                sampler,
                self.linear_proj_audio,
                self.stop_head,
                patch_size=self.ming_config.patch_size,
                latent_dim=self.ming_config.latent_dim,
                steps=_CFM_STEPS,
            )
            logger.info("Ming CFM CUDAGraph enabled (steps=%d, patch=%d).", _CFM_STEPS, self.ming_config.patch_size)
        except Exception as exc:
            logger.warning("Ming CFM CUDAGraph unavailable (%s); using eager flow head.", exc)
            self._cfm_graph_enabled = False
            return None
        return self._cfm_graph

    def _decode_one_step(
        self,
        *,
        hidden_states: torch.Tensor,
        latent_history: torch.Tensor,
        cfg_scale: float,
        sigma: float,
        temperature: float,
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
        # stop head). Falls back permanently to the eager path on any failure.
        sampled_token_latent = next_embeds = stop_probs = None
        # cfg < 1e-5 disables CFG; the eager flow head handles this with a
        # dedicated unconditional branch (c=zeros) that the captured graph does
        # not replicate, so fall back to eager there to match FlowLoss.sample.
        graph_exec = self._maybe_build_cfm_graph(z_diff_cond) if cfg_scale >= 1e-5 else None
        if graph_exec is not None:
            try:
                sampled_token_latent, next_embeds, stop_full = graph_exec.execute(
                    z_diff_cond, latent_history, cfg_scale, sigma, temperature
                )
                stop_probs = stop_full[:, 1]
            except Exception as exc:
                logger.warning("Ming CFM CUDAGraph failed (%s); falling back to eager flow head.", exc)
                self._cfm_graph = None
                self._cfm_graph_enabled = False
                sampled_token_latent = next_embeds = stop_probs = None

        if sampled_token_latent is None:
            sampled_token_latent = self.flowloss.sample(
                z=z_diff_cond,
                latent_history=latent_history,
                cfg=cfg_scale,
                patch_size=self.ming_config.patch_size,
                sigma=sigma,
                temperature=temperature,
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
    "MING_STOP_REASON_CODES",
    "MING_STOP_REASON_CONTINUE",
    "MING_STOP_REASON_KEY",
    "MING_STOP_REASON_MAX_DECODE_STEPS",
    "MING_STOP_REASON_STOP_HEAD",
    "MingLLMModel",
    "_coerce_latent_history",
    "_extract_hidden_states",
    "_resolve_ming_runtime_dtype",
    "_resolve_ming_stop_decision",
    "_warn_missing_prefix",
]
