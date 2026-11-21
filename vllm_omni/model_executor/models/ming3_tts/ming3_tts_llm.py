# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming3 continuous-latent AR stage."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsPP
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.common.ming.aggregator import Aggregator
from vllm_omni.model_executor.models.common.ming.cfm_head import FlowLoss
from vllm_omni.model_executor.models.common.ming.continuous_tts import (
    STOP_REASON_CODES,
    STOP_REASON_CONTINUE,
    resolve_stop_decision,
)
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_CFG,
    KEY_DECODE_STEP,
    KEY_LATENT_HISTORY,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_REQUEST_ID,
    KEY_SIGMA,
    KEY_TEMPERATURE,
    KEY_TEXT_MODE,
)
from vllm_omni.model_executor.models.ming_tts.ming_tts_llm import (
    MingLLMModel,
    _extract_hidden_states,
    _resolve_ming_runtime_dtype,
    _warn_missing_prefix,
)
from vllm_omni.model_executor.models.ming_tts.patch_emission import (
    _coerce_latent_history,
    _get_request_token_counts,
    _normalize_request_infos,
    _resolve_optional_runtime_int,
    _resolve_runtime_float,
    _resolve_runtime_int,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .config_ming3_tts import (
    Ming3TTSRuntimeConfig,
    get_ming3_backbone_config,
)
from .constants import (
    KEY_STOP_REASON,
    SPEAKER_EMBEDDING_DIM,
)
from .kda_prefill import install_sync_free_kda_prefill

logger = init_logger(__name__)


def _gather_rows(hidden_states: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Select hidden rows without building a device index tensor."""
    first = indices[0]
    if indices == list(range(first, first + len(indices))):
        return hidden_states[first : first + len(indices)]
    return torch.cat([hidden_states[index : index + 1] for index in indices])


def _assert_all_parameter_tensors_loaded(module: nn.Module, loaded_names: set[str], *, label: str) -> None:
    aliases_by_tensor: dict[int, set[str]] = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        aliases_by_tensor.setdefault(id(parameter), set()).add(name)
    missing = [sorted(aliases) for aliases in aliases_by_tensor.values() if aliases.isdisjoint(loaded_names)]
    if missing:
        raise RuntimeError(
            f"Ming3 {label}: {len(missing)} distinct parameter tensors were not loaded. "
            f"First few aliases: {missing[:5]}"
        )


class Ming3TTSLLMModel(MingLLMModel, HasInnerState, IsHybrid, SupportsPP):
    packed_modules_mapping = BailingMoeV3ForCausalLM.packed_modules_mapping
    latent_patch_key = "ming3_latent_patch"
    next_embeds_key = "ming3_next_embeds"
    new_history_key = "ming3_new_history"
    decode_step_key = "ming3_decode_step"
    has_patch_key = "ming3_has_patch"
    stop_reason_key = KEY_STOP_REASON
    stop_probability_key = "ming3_stop_prob"
    stop_reason_codes = STOP_REASON_CODES
    vectorize_cfm_decode = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.ming_config = Ming3TTSRuntimeConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.fm_dtype = _resolve_ming_runtime_dtype(vllm_config)
        self.model = BailingMoeV3ForCausalLM(
            vllm_config=get_ming3_backbone_config(vllm_config),
            prefix=maybe_prefix(prefix, "model"),
        )
        if not vllm_config.model_config.enforce_eager and (
            vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            install_sync_free_kda_prefill(self.model)
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
        # Keep Tiny's fast paths enabled without changing shared-module defaults.
        for block in (*self.flowloss.cfm.model.blocks, *self.linear_proj_audio.blocks):
            block.attn.short_attention = True
            block.mlp.fused_gelu = True
        self.stop_head = nn.Linear(self.ming_config.llm_hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(SPEAKER_EMBEDDING_DIM, self.ming_config.llm_hidden_size, bias=True)
        for module in (
            self.flowloss,
            self.linear_proj_audio,
            self.stop_head,
            self.spk_head,
        ):
            module.to(dtype=self.fm_dtype)
        self._pending_postprocess_updates: dict[str, dict[str, Any]] = {}
        self._last_ming_next_token_ids = None
        # Model Runner V2: the step's continue/stop tokens, decided on the device.
        self._mrv2_next_token_ids: torch.Tensor | None = None
        self._last_text_mode = False
        self._init_cfm_graph_state()
        self._request_generators: dict[str, torch.Generator] = {}
        # Per-request flow-head feedback (latent history, next input embedding)
        # kept on the device between AR steps instead of round-tripping
        # through the host-side runtime buffer.
        self._decode_state: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}

    def make_empty_intermediate_tensors(self, *args: Any, **kwargs: Any):
        return self.model.make_empty_intermediate_tensors(*args, **kwargs)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return BailingMoeV3ForCausalLM.get_mamba_state_shape_from_config(get_ming3_backbone_config(vllm_config))

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return BailingMoeV3ForCausalLM.get_mamba_state_dtype_from_config(get_ming3_backbone_config(vllm_config))

    @classmethod
    def get_mamba_state_copy_func(cls):
        return BailingMoeV3ForCausalLM.get_mamba_state_copy_func()

    def set_initial_history(self, request_id: str, history: torch.Tensor) -> None:
        """Seed a request's device-resident flow history at prefill."""
        history = _coerce_latent_history(history, device=history.device, dtype=self.fm_dtype, cfg=self.ming_config)
        self._decode_state[request_id] = (history, None)

    def next_embeds_for(self, request_id: str) -> torch.Tensor | None:
        state = self._decode_state.get(request_id)
        return None if state is None else state[1]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        latent_history: torch.Tensor | None = None,
        **kwargs: object,
    ) -> OmniOutput | IntermediateTensors | torch.Tensor:
        if latent_history is not None:
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                latent_history=latent_history,
                **kwargs,
            )
        # Only the backbone runs here, so a decode batch can replay as one
        # FULL CUDA graph. The flow head, stop decision and host transfer run
        # afterwards in make_omni_output, outside graph capture.
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput | IntermediateTensors,
        *,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        seq_token_counts: list[int] | None = None,
        request_token_spans: list[tuple[int, int]] | None = None,
        request_sample_eligible: list[bool] | None = None,
        **kwargs: object,
    ) -> OmniOutput | IntermediateTensors:
        if isinstance(model_outputs, (OmniOutput, IntermediateTensors)):
            if isinstance(model_outputs, IntermediateTensors):
                self._last_ming_next_token_ids = None
            return model_outputs
        runtime_information = (
            model_intermediate_buffer
            if model_intermediate_buffer is not None
            else kwargs.get("runtime_additional_information")
        )
        request_infos = _normalize_request_infos(runtime_information)
        if any(info.get(KEY_TEXT_MODE) for info in request_infos):
            raise RuntimeError("Ming3 Stage 0 does not support Ming text mode")
        self._validate_decode_window(request_infos)
        self._last_text_mode = False
        intermediate_tensors = None
        hidden_states = _extract_hidden_states(model_outputs)
        token_counts = _get_request_token_counts(hidden_states, request_infos, seq_token_counts, request_token_spans)
        if request_sample_eligible is not None and len(request_sample_eligible) != len(token_counts):
            raise RuntimeError(
                "Ming request_sample_eligible/request span mismatch: "
                f"got {len(request_sample_eligible)} flags for {len(token_counts)} requests."
            )

        # One row per request that emits a latent this step: its last scheduled
        # token conditions the flow head.
        rows: list[tuple[int, dict[str, Any], Any, int, torch.Tensor]] = []
        total_tokens = hidden_states.shape[0]
        cursor = 0
        for req_idx, token_count in enumerate(token_counts):
            end = min(cursor + token_count, total_tokens)
            if end <= cursor:
                continue
            cursor = end
            if request_sample_eligible is not None and not request_sample_eligible[req_idx]:
                continue
            req_info = request_infos[req_idx] if req_idx < len(request_infos) else {}
            req_id = req_info.get(KEY_REQUEST_ID)
            state = self._decode_state.get(req_id) if isinstance(req_id, str) else None
            history = (
                state[0]
                if state is not None
                else _coerce_latent_history(
                    req_info.get(KEY_LATENT_HISTORY),
                    device=hidden_states.device,
                    dtype=self.fm_dtype,
                    cfg=self.ming_config,
                )
            )
            if history is not None:
                rows.append((req_idx, req_info, req_id, end - 1, history))

        self._pending_postprocess_updates = {}
        if not rows:
            self._last_ming_next_token_ids = None
            return OmniOutput(
                text_hidden_states=hidden_states, multimodal_outputs=None, intermediate_tensors=intermediate_tensors
            )

        groups: dict[tuple[float, float, float], list[int]] = {}
        for row_idx, (_, req_info, _, _, _) in enumerate(rows):
            key = (
                _resolve_runtime_float(req_info, KEY_CFG, self.ming_config.cfg),
                _resolve_runtime_float(req_info, KEY_SIGMA, self.ming_config.sigma),
                _resolve_runtime_float(req_info, KEY_TEMPERATURE, self.ming_config.temperature),
            )
            groups.setdefault(key, []).append(row_idx)

        group_outputs = []
        host_values = []
        finite_checks = []
        for (cfg_scale, sigma, temperature), row_indices in groups.items():
            output_indices = [rows[i][3] for i in row_indices]
            decode_hidden = _gather_rows(hidden_states, output_indices)
            history = rows[row_indices[0]][4] if len(row_indices) == 1 else torch.cat([rows[i][4] for i in row_indices])
            latent, next_embeds, stop_probs, finite = self._sample_flow_group(
                decode_hidden,
                history,
                [rows[i][1] for i in row_indices],
                cfg_scale=cfg_scale,
                sigma=sigma,
                temperature=temperature,
            )
            new_history = torch.cat([history[:, self.ming_config.patch_size :, :], latent], dim=1)
            group_outputs.append((row_indices, output_indices, latent, next_embeds, new_history))
            host_values.append(stop_probs.reshape(-1).float())
            if finite is not None:
                finite_checks.append(finite.reshape(1).float())

        # The only host sync of the step: stop probabilities for the forced
        # next-token decision, plus the deferred non-finite guard.
        host_list = torch.cat([*host_values, *finite_checks]).tolist()
        stop_by_row: dict[int, float] = {}
        offset = 0
        for row_indices, *_ in group_outputs:
            for row_idx in row_indices:
                stop_by_row[row_idx] = host_list[offset]
                offset += 1
        if any(value != 1.0 for value in host_list[offset:]):
            raise RuntimeError("Non-finite Ming3 flow-head output in decode step.")

        # CUDA graphs pad the batch, so compare against the scheduled tokens.
        if len(group_outputs) == 1 and group_outputs[0][1] == list(range(cursor)):
            latent = group_outputs[0][2]
            if cursor == total_tokens:
                latent_patch_tokens = latent
            else:
                latent_patch_tokens = latent.new_zeros((total_tokens, *latent.shape[1:]))
                latent_patch_tokens[:cursor] = latent
            has_patch = torch.zeros((total_tokens,), dtype=torch.bool, device=hidden_states.device)
            has_patch[:cursor] = True
        else:
            latent_patch_tokens = hidden_states.new_zeros(
                (total_tokens, self.ming_config.patch_size, self.ming_config.latent_dim), dtype=self.fm_dtype
            )
            has_patch = torch.zeros((total_tokens,), dtype=torch.bool, device=hidden_states.device)
            for _, output_indices, latent, _, _ in group_outputs:
                for j, output_index in enumerate(output_indices):
                    latent_patch_tokens[output_index : output_index + 1] = latent[j : j + 1]
                    has_patch[output_index] = True

        next_token_ids = [int(self.ming_config.text_eos_token_id)] * len(token_counts)
        for row_indices, _, latent, next_embeds, new_history in group_outputs:
            for j, row_idx in enumerate(row_indices):
                req_idx, req_info, req_id, _, _ = rows[row_idx]
                decode_step = int(req_info.get(KEY_DECODE_STEP, req_info.get("generated_len", 0)))
                stop_reason, _, _, _, next_token_id = self._resolve_stop_decision(
                    step=decode_step,
                    stop_prob=stop_by_row[row_idx],
                    stop_threshold=float(self.ming_config.stop_head_threshold),
                    min_stop_step=int(self.ming_config.stop_head_min_steps),
                    min_decode_steps=_resolve_optional_runtime_int(req_info, KEY_MIN_DECODE_STEPS, 0),
                    max_decode_steps=_resolve_runtime_int(
                        req_info, KEY_MAX_DECODE_STEPS, self.ming_config.max_decode_steps
                    ),
                    audio_dummy_token_id=int(self.ming_config.audio_dummy_token_id),
                    text_eos_token_id=int(self.ming_config.text_eos_token_id),
                )
                self._on_stop_decision(req_info, stop_reason)
                next_token_ids[req_idx] = int(next_token_id)
                if isinstance(req_id, str):
                    self._decode_state[req_id] = (new_history[j : j + 1], next_embeds[j : j + 1])
                    self._pending_postprocess_updates[req_id] = {self.stop_reason_key: stop_reason}

        self._last_ming_next_token_ids = next_token_ids
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs={
                self.latent_patch_key: latent_patch_tokens,
                self.has_patch_key: has_patch,
            },
            intermediate_tensors=intermediate_tensors,
        )

    def _sample_flow_group(
        self,
        hidden_states: torch.Tensor,
        latent_history: torch.Tensor,
        req_infos: list[dict[str, Any]],
        *,
        cfg_scale: float,
        sigma: float,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run the flow head for requests sharing sampling controls.

        Returns the device-side finite flag instead of checking it, so the
        caller can fold it into the step's single host transfer.
        """
        device = hidden_states.device
        # Per-request generators keep a seeded request reproducible regardless
        # of which requests share its batch.
        generators = [self._get_request_generator(info, device) for info in req_infos]
        return self._sample_flow_rows(
            hidden_states, latent_history, generators, cfg_scale=cfg_scale, sigma=sigma, temperature=temperature
        )

    def _sample_flow_rows(
        self,
        hidden_states: torch.Tensor,
        latent_history: torch.Tensor,
        generators: list[torch.Generator | None],
        *,
        cfg_scale: float,
        sigma: float,
        temperature: float,
        check_finite: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run the flow head for rows sharing sampling controls, one generator (or None) per row.

        With ``check_finite`` the graph path also returns a device flag that
        every output is finite; callers that check rows themselves skip it.
        """
        device = hidden_states.device
        z_diff_cond = hidden_states.to(dtype=self.fm_dtype).unsqueeze(1)
        graph_exec = self._maybe_build_cfm_graph(z_diff_cond) if cfg_scale >= 1e-5 else None
        # Keep RNG call boundaries stable even after a bucket is disabled,
        # or when CFG selects the eager-only unconditional path.
        initial_noise, solver_noise = self._draw_request_cfm_noises(generators, device)
        if graph_exec is not None:
            graph_result = None
            try:
                graph_result = graph_exec.execute(
                    z_diff_cond,
                    latent_history,
                    cfg_scale,
                    sigma,
                    temperature,
                    initial_noise=initial_noise,
                    solver_noise=solver_noise,
                )
            except Exception:
                self._cfm_graphs.pop(graph_exec.batch_size, None)
                self._cfm_graph_failed_batch_sizes.add(graph_exec.batch_size)
                logger.warning(
                    "Ming3 CFM CUDA Graph failed for batch=%d; using eager for this shape.",
                    graph_exec.batch_size,
                    exc_info=True,
                )
            if graph_result is not None:
                latent, next_embeds, stop_full = graph_result
                stop_probs = stop_full[:, 1]
                finite = None
                if check_finite:
                    finite = (
                        torch.isfinite(z_diff_cond).all()
                        & torch.isfinite(latent).all()
                        & torch.isfinite(next_embeds).all()
                        & torch.isfinite(stop_probs).all()
                    )
                return latent, next_embeds, stop_probs, finite

        # Eager and graph execution consume the same explicitly drawn noise.
        outputs = [
            self._decode_one_step(
                hidden_states=hidden_states[i : i + 1],
                latent_history=latent_history[i : i + 1],
                cfg_scale=cfg_scale,
                sigma=sigma,
                temperature=temperature,
                generator=generator,
                initial_noise=initial_noise[i : i + 1],
                solver_noise=solver_noise[:, i : i + 1],
                check_finite=check_finite,
            )
            for i, generator in enumerate(generators)
        ]
        return (
            torch.cat([output[0] for output in outputs]),
            torch.cat([output[1] for output in outputs]),
            torch.cat([output[3].reshape(-1) for output in outputs]),
            None,
        )

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor | None:
        if self._last_text_mode or hidden_states is None or hidden_states.numel() == 0:
            return super().compute_logits(hidden_states, sampling_metadata)
        device_token_ids = self._mrv2_next_token_ids
        if device_token_ids is not None and device_token_ids.numel() == hidden_states.shape[0]:
            # Model Runner V2 decides every row's token on the device.
            self._mrv2_next_token_ids = None
            logits = torch.full(
                (hidden_states.shape[0], self.ming_config.llm_vocab_size),
                float("-inf"),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            return logits.scatter_(1, device_token_ids.reshape(-1, 1), 0.0)
        next_token_ids = self._last_ming_next_token_ids
        if next_token_ids is None or len(next_token_ids) != hidden_states.shape[0]:
            return super().compute_logits(hidden_states, sampling_metadata)
        self._last_ming_next_token_ids = None
        # Every row forces one token; write the common continue token with one
        # kernel and patch only the rows that stop.
        continue_token_id = int(self.ming_config.audio_dummy_token_id)
        logits = torch.full(
            (hidden_states.shape[0], self.ming_config.llm_vocab_size),
            float("-inf"),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        logits[:, continue_token_id] = 0.0
        for i, next_token_id in enumerate(next_token_ids):
            if next_token_id != continue_token_id:
                logits[i, continue_token_id] = float("-inf")
                logits[i, int(next_token_id)] = 0.0
        return logits

    def _validate_decode_window(self, request_infos: list[dict[str, Any]]) -> None:
        for index, info in enumerate(request_infos):
            maximum = _resolve_runtime_int(info, KEY_MAX_DECODE_STEPS, self.ming_config.max_decode_steps)
            # The shared Ming loop reads this field with a zero default; resolve Ming3's fallback here.
            info.setdefault(KEY_MIN_DECODE_STEPS, self.ming_config.stop_head_min_steps)
            minimum = max(1, _resolve_optional_runtime_int(info, KEY_MIN_DECODE_STEPS, 0))
            if maximum < minimum:
                raise ValueError(
                    f"Ming3 request idx={index}: maximum decode steps {maximum} is smaller than minimum {minimum}"
                )

    def _resolve_stop_decision(
        self,
        *,
        step: int,
        stop_prob: float,
        stop_threshold: float,
        min_stop_step: int,
        min_decode_steps: int,
        max_decode_steps: int,
        audio_dummy_token_id: int,
        text_eos_token_id: int,
    ) -> tuple[str, bool, bool, int, int]:
        decision = resolve_stop_decision(
            emitted_steps=step + 1,
            stop_probability=stop_prob,
            stop_threshold=stop_threshold,
            minimum_steps=max(1, min_decode_steps),
            maximum_steps=max_decode_steps,
            continue_token_id=audio_dummy_token_id,
            stop_token_id=text_eos_token_id,
        )
        return (
            decision.reason,
            decision.should_stop,
            decision.forced,
            decision.minimum_steps,
            decision.next_token_id,
        )

    def _get_request_generator(self, req_info: dict[str, Any], device: torch.device) -> torch.Generator | None:
        request_id = req_info.get(KEY_REQUEST_ID, req_info.get("request_id"))
        if not isinstance(request_id, str) or not request_id:
            return None
        generator = self._request_generators.get(request_id)
        if generator is not None:
            return generator
        raw_seed = req_info.get("tts_local_seed")
        if raw_seed is None:
            # An unseeded request cannot be replayed, so a randomly seeded
            # generator would only force per-request RNG calls each step.
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed(int(raw_seed))
        self._request_generators[request_id] = generator
        return generator

    def _on_stop_decision(self, req_info: dict[str, Any], stop_reason: str) -> None:
        if stop_reason != STOP_REASON_CONTINUE:
            request_id = req_info.get(KEY_REQUEST_ID)
            if isinstance(request_id, str):
                self._request_generators.pop(request_id, None)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            self._request_generators.pop(request_id, None)
            self._pending_postprocess_updates.pop(request_id, None)
            self._decode_state.pop(request_id, None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = list(weights)
        unknown = [
            name
            for name, _ in weights
            if not name.startswith(("model.", "linear_proj_audio.", "flowloss.", "stop_head.", "spk_head."))
        ]
        if unknown:
            raise RuntimeError(f"Ming3 Stage-0 received {len(unknown)} unclassified tensors: {unknown[:8]}")

        backbone_weights = [(name[len("model.") :], value) for name, value in weights if name.startswith("model.")]
        if not backbone_weights:
            raise RuntimeError("Ming3 Stage-0 received no model.* backbone tensors")
        loaded_backbone = self.model.load_weights(backbone_weights)
        _assert_all_parameter_tensors_loaded(self.model, loaded_backbone, label="Bailing-v3 backbone")

        tts_weights = [(name, value) for name, value in weights if not name.startswith("model.")]
        loaded_tts = AutoWeightsLoader(self).load_weights(tts_weights)
        params = dict(self.named_parameters(remove_duplicate=False))
        for module_name in ("flowloss", "linear_proj_audio", "stop_head", "spk_head"):
            _warn_missing_prefix(
                module_name,
                params,
                loaded_tts,
                prefix=f"{module_name}.",
            )
        self._warmup_cfm_graphs()
        return {f"model.{name}" for name in loaded_backbone} | loaded_tts


__all__ = ["Ming3TTSLLMModel"]
