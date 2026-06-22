# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.minimind_o.minimind_omni_config import (
    MiniMindOmniConfig,
    MiniMindOmniTalkerConfig,
)
from vllm_omni.model_executor.models.minimind_o.minimind_omni_thinker import (
    MiniMindBlock,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput


class MiniMindOmniTalkerHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int = 8,
        rank: int = 256,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.base = ReplicatedLinear(
            input_size=in_features,
            output_size=out_features,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=maybe_prefix(prefix, "base"),
        )
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    ReplicatedLinear(
                        input_size=in_features,
                        output_size=rank,
                        bias=False,
                        quant_config=quant_config,
                        return_bias=False,
                        prefix=maybe_prefix(prefix, f"adapters.{i}.0"),
                    ),
                    nn.GELU(),
                    ReplicatedLinear(
                        input_size=rank,
                        output_size=out_features,
                        bias=False,
                        quant_config=quant_config,
                        return_bias=False,
                        prefix=maybe_prefix(prefix, f"adapters.{i}.2"),
                    ),
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        base_out = self.base(x)
        return [base_out + adapter(x) for adapter in self.adapters]


class MiniMindOmniTalkerEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_layers: int = 8,
        rank: int = 256,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.base = VocabParallelEmbedding(
            num_embeddings,
            embedding_dim,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "base"),
        )
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    VocabParallelEmbedding(
                        num_embeddings,
                        rank,
                        quant_config=quant_config,
                        prefix=maybe_prefix(prefix, f"adapters.{i}.0"),
                    ),
                    nn.GELU(),
                    ReplicatedLinear(
                        input_size=rank,
                        output_size=embedding_dim,
                        bias=False,
                        quant_config=quant_config,
                        return_bias=False,
                        prefix=maybe_prefix(prefix, f"adapters.{i}.2"),
                    ),
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        adapted = [base_out[:, i, :, :] + adapter(x[:, i, :]) for i, adapter in enumerate(self.adapters)]
        return torch.stack(adapted, dim=0).mean(dim=0)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class MiniMindOmniTalkerForConditionalGeneration(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"talker.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: MiniMindOmniConfig = vllm_config.model_config.hf_config
        talker_config: MiniMindOmniTalkerConfig = config.talker_config
        self.config = config
        self.talker_config = talker_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.audio_pad_token = int(talker_config.audio_pad_token)
        self.audio_stop_token = int(talker_config.audio_stop_token)
        self.internal_stop_token_id = int(talker_config.internal_stop_token_id)
        self.audio_spk_token = int(talker_config.audio_spk_token)
        self.audio_vocab_size = int(talker_config.audio_vocab_size)
        self.num_code_layers = int(talker_config.num_code_layers)
        self.max_steps_after_last_thinker_token = int(config.talker_max_steps_after_last_thinker_token)
        self.has_preprocess = True
        self.has_postprocess = True
        self.have_multimodal_outputs = True
        self.mtp_hidden_size = int(talker_config.hidden_size)
        self.talker_mtp_output_key = ("codes", "audio")
        self.talker_mtp_active_mask_size = self.num_code_layers
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("hidden_states", "bridge"),
            ("hidden_states", "last"),
        }

        # init mask layers
        layer_ids = torch.arange(self.num_code_layers)
        step_ids = torch.arange(self.num_code_layers + 1) - 1  # [-1, 0, ..., num_code_layers-1]
        self.register_buffer(
            "_code_layer_masks",
            step_ids.unsqueeze(-1) >= layer_ids.unsqueeze(0),  # [num_code_layers+1, num_code_layers]
            persistent=False,
        )

        self.layers = nn.ModuleList(
            [
                MiniMindBlock(
                    talker_config,
                    layer_idx=i,
                    cache_config=vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=prefix,
                )
                for i in range(talker_config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(talker_config.hidden_size, eps=talker_config.rms_norm_eps)
        self.lm_head = MiniMindOmniTalkerHead(
            talker_config.hidden_size,
            self.audio_vocab_size,
            num_layers=self.num_code_layers,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.embed_tokens = MiniMindOmniTalkerEmbedding(
            self.audio_vocab_size,
            talker_config.hidden_size,
            num_layers=self.num_code_layers,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.codec_proj = nn.Sequential(
            ReplicatedLinear(
                input_size=talker_config.hidden_size,
                output_size=talker_config.hidden_size,
                quant_config=self.quant_config,
                return_bias=False,
                prefix=maybe_prefix(prefix, "codec_proj.0"),
            ),
            nn.GELU(),
            ReplicatedLinear(
                input_size=talker_config.hidden_size,
                output_size=talker_config.hidden_size,
                quant_config=self.quant_config,
                return_bias=False,
                prefix=maybe_prefix(prefix, "codec_proj.2"),
            ),
            RMSNorm(talker_config.hidden_size, eps=talker_config.rms_norm_eps),
        )
        self.embed_proj = nn.Sequential(
            ReplicatedLinear(
                input_size=talker_config.text_hidden_size,
                output_size=talker_config.text_hidden_size,
                quant_config=self.quant_config,
                return_bias=False,
                prefix=maybe_prefix(prefix, "embed_proj.0"),
            ),
            nn.GELU(),
            ReplicatedLinear(
                input_size=talker_config.text_hidden_size,
                output_size=talker_config.hidden_size,
                quant_config=self.quant_config,
                return_bias=False,
                prefix=maybe_prefix(prefix, "embed_proj.2"),
            ),
            RMSNorm(talker_config.hidden_size, eps=talker_config.rms_norm_eps),
        )
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))
        self.spk_proj = ReplicatedLinear(
            input_size=talker_config.spk_emb_size,
            output_size=talker_config.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            return_bias=False,
            prefix=maybe_prefix(prefix, "spk_proj"),
        )
        self.prefer_model_sampler = True
        self.sampler = Sampler()
        self._stop_pending_by_req: dict[str, bool] = {}
        self.make_empty_intermediate_tensors = lambda: None

    def get_language_model(self) -> nn.Module:
        return self

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        num_items = 0
        for value in kwargs.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                num_items = int(value.shape[0])
                break
            if isinstance(value, (list, tuple)):
                num_items = len(value)
                break
        if num_items <= 0:
            return []

        ref_param = next(self.parameters())
        return [
            torch.zeros(
                (1, self.talker_config.hidden_size),
                device=ref_param.device,
                dtype=ref_param.dtype,
            )
            for _ in range(num_items)
        ]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        audio_ids = self._audio_ids_from_layer0(input_ids)
        return self.codec_proj(self.embed_tokens(audio_ids)).reshape(-1, self.talker_config.hidden_size)

    def _audio_ids_from_layer0(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim == 3:
            return input_ids.to(dtype=torch.long)
        batch_size, seq_len = input_ids.shape
        audio_ids = torch.full(
            (batch_size, self.num_code_layers, seq_len),
            self.audio_pad_token,
            dtype=torch.long,
            device=input_ids.device,
        )
        audio_ids[:, 0, :] = input_ids.to(dtype=torch.long).clamp(min=0, max=self.audio_vocab_size - 1)
        return audio_ids

    def _select_bridge_states(
        self,
        info_dict: dict[str, Any],
        span_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, bool, int, int, int]:
        hidden_info = info_dict.get("hidden_states", {}) if isinstance(info_dict, dict) else {}
        bridge = hidden_info.get("bridge") if isinstance(hidden_info, dict) else None
        if not isinstance(bridge, torch.Tensor):
            raise ValueError("MiniMind talker requires hidden_states.bridge tensor in additional_information.")

        target_dtype = self.text_scale.dtype
        if bridge.device != device or bridge.dtype != target_dtype:
            bridge = bridge.to(device=device, dtype=target_dtype, non_blocking=True)
            hidden_info["bridge"] = bridge
        if bridge.ndim == 3:
            bridge = bridge.reshape(-1, bridge.shape[-1])
        bridge_len = int(bridge.shape[0])
        if bridge.shape[0] < span_len:
            raise ValueError(f"bridge hidden states length {bridge.shape[0]} is shorter than scheduled span {span_len}")

        prompt_len = int(info_dict.get("_omni_prompt_len", 0) or 0)
        ids = info_dict.get("ids") if isinstance(info_dict, dict) else None
        if isinstance(ids, dict):
            text_prompt_len = len(ids.get("prompt") or [])
            if text_prompt_len > 0:
                prompt_len = text_prompt_len

        raw_num_computed = info_dict.get("_omni_num_computed_tokens")
        num_computed = prompt_len if raw_num_computed is None else int(raw_num_computed)
        raw_is_prefill = info_dict.get("_omni_is_prefill")
        is_prefill = num_computed < prompt_len if raw_is_prefill is None else bool(raw_is_prefill)

        # Decode input at position `num_computed` should be conditioned on
        # the matching thinker bridge row, not on bridge[-1] for every step.
        start = max(0, min(num_computed, max(0, bridge.shape[0] - span_len)))
        end = start + span_len
        return bridge[start:end], is_prefill, prompt_len, num_computed, bridge_len

    def _make_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        bridge_states: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        audio_ids = self._audio_ids_from_layer0(input_ids)
        talker_emb = self.embed_tokens(audio_ids)
        if spk_emb is not None:
            if spk_emb.ndim == 1:
                spk_emb = spk_emb.unsqueeze(0)
            spk_emb = spk_emb.to(device=talker_emb.device, dtype=talker_emb.dtype)
            spk_mask = (audio_ids[:, 0, :] == self.audio_spk_token).unsqueeze(-1)
            talker_emb = torch.where(spk_mask, self.spk_proj(spk_emb).unsqueeze(1), talker_emb)
        text_part = self.embed_proj(bridge_states.to(device=talker_emb.device, dtype=talker_emb.dtype))
        codec_part = self.codec_proj(talker_emb.reshape(-1, talker_emb.shape[-1]))
        return text_part * self.text_scale + codec_part * self.audio_scale

    def _sample_codebook_logits_batch(
        self,
        logits_by_layer: list[torch.Tensor],
        *,
        do_sample: bool = True,
        temperature: float = 0.2,
        top_k: int = 50,
    ) -> torch.Tensor:
        if not logits_by_layer:
            return torch.empty((0, 0), dtype=torch.long, device=self.text_scale.device)

        logits = torch.stack(logits_by_layer, dim=0)
        num_layers, batch, vocab = logits.shape
        flat = logits.reshape(num_layers * batch, vocab).float()
        if not do_sample:
            sampled = flat.argmax(dim=-1)
        else:
            temperature = max(float(temperature), 1e-5)
            flat = flat / temperature
            if top_k > 0 and top_k < vocab:
                top_val, top_idx = flat.topk(top_k, dim=-1)
                sample = torch.multinomial(torch.softmax(top_val, dim=-1), 1)
                sampled = top_idx.gather(-1, sample).squeeze(-1)
            else:
                sampled = torch.multinomial(torch.softmax(flat, dim=-1), 1).squeeze(-1)
        return sampled.reshape(num_layers, batch).transpose(0, 1).contiguous()

    def preprocess(
        self,
        input_ids: torch.Tensor,
        _input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        span_len = int(input_ids.shape[0])
        bridge_states, is_prefill, prompt_len, num_computed, bridge_len = self._select_bridge_states(
            info_dict, span_len, input_ids.device
        )
        spk_emb = info_dict.get("spk_emb")
        embeds = self._make_inputs_embeds(
            input_ids.view(1, -1),
            bridge_states,
            spk_emb=spk_emb,
        )
        update: dict[str, Any] = {}
        if span_len == 1:
            hidden = info_dict.get("hidden_states", {})
            last_hidden = hidden.get("last") if isinstance(hidden, dict) else None
            if not isinstance(last_hidden, torch.Tensor):
                last_hidden = torch.zeros(
                    (1, self.talker_config.hidden_size),
                    device=input_ids.device,
                    dtype=embeds.dtype,
                )
            text_step = self.embed_proj(bridge_states.to(device=embeds.device, dtype=embeds.dtype))
            audio_step = max(0, num_computed - prompt_len) - 1
            mask_idx = max(0, min(audio_step + 1, self.num_code_layers))
            active_mask = self._code_layer_masks[mask_idx : mask_idx + 1].to(device=embeds.device)
            update["mtp_inputs"] = (
                last_hidden.reshape(1, -1).to(device=embeds.device, dtype=embeds.dtype),
                text_step.reshape(1, -1),
                active_mask,
            )
            if not is_prefill:
                request_id = info_dict.get("request_id")
                # Watchdog for cases where the thinker stops at max_tokens
                # without emitting EOS; cap only the talker tail after the
                # final thinker bridge state, not the total audio length.
                steps_after_last_thinker = max(0, num_computed - bridge_len + 1)
                if (
                    isinstance(request_id, str)
                    and self.max_steps_after_last_thinker_token >= 0
                    and num_computed >= bridge_len
                    and steps_after_last_thinker >= self.max_steps_after_last_thinker_token
                ):
                    self._stop_pending_by_req[request_id] = True
        else:
            update.setdefault("codes", {})["audio"] = torch.full(
                (span_len, self.num_code_layers),
                self.audio_pad_token,
                dtype=torch.long,
                device=input_ids.device,
            )
        return input_ids, embeds, update

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            raise NotImplementedError("MiniMind talker does not support pipeline-parallel intermediate tensors.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided.")
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        _sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        # vLLM samples one token stream. Return layer-0 codec logits here; the
        # remaining adapter heads are kept for code2wav/full-code integration.
        return self.lm_head(hidden_states)[0]

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        logits_for_sampler = logits
        if 0 <= self.internal_stop_token_id < logits.shape[-1]:
            logits_for_sampler = logits.clone()
            logits_for_sampler[:, self.internal_stop_token_id] = -torch.inf

        sampler_output = self.sampler(logits_for_sampler, sampling_metadata)
        request_ids = list(getattr(sampling_metadata, "request_ids", None) or [])
        if not request_ids:
            return sampler_output

        sampled = sampler_output.sampled_token_ids
        for row in range(sampled.shape[0]):
            request_id = request_ids[row] if row < len(request_ids) else None
            if request_id is None or not self._stop_pending_by_req.get(request_id):
                continue
            sampled[row, 0] = self.internal_stop_token_id
            self._stop_pending_by_req.pop(request_id, None)
        return sampler_output

    @torch.inference_mode()
    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(input_ids.shape[0])
        hidden = last_talker_hidden.reshape(batch, -1).to(device=input_embeds.device, dtype=input_embeds.dtype)
        logits_by_layer = self.lm_head(hidden)

        layer0 = input_ids.reshape(batch).to(device=input_embeds.device, dtype=torch.long)
        residual_codes = self._sample_codebook_logits_batch(
            logits_by_layer[1:],
            do_sample=True if do_sample is None else bool(do_sample),
            temperature=0.2 if temperature is None else float(temperature),
            top_k=50 if top_k is None else int(top_k),
        )
        audio_codes = torch.cat((layer0.unsqueeze(-1), residual_codes), dim=-1)

        if active_mask is not None:
            active = active_mask.to(device=audio_codes.device, dtype=torch.bool)
            if active.ndim == 1:
                active = active.unsqueeze(0)
            audio_codes = audio_codes.masked_fill(~active[:, : audio_codes.shape[-1]], self.audio_pad_token)

        code_embeds = self.codec_proj(self.embed_tokens(audio_codes.unsqueeze(-1)).reshape(batch, -1))
        next_embeds = (
            code_embeds * self.audio_scale
            + text_step.reshape(batch, -1).to(
                device=code_embeds.device,
                dtype=code_embeds.dtype,
            )
            * self.text_scale
        )
        return next_embeds, audio_codes

    def _normalise_audio_code_rows(
        self,
        audio: Any,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
            return None
        rows = audio.to(device=device if device is not None else audio.device, dtype=torch.long)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[-1] != self.num_code_layers:
            return None
        return rows

    def _ready_diagonal_audio_frames(
        self,
        history: torch.Tensor | None,
        current: torch.Tensor,
        emitted_frames: int,
    ) -> torch.Tensor | None:
        if history is not None:
            history = self._normalise_audio_code_rows(history, device=current.device)
        rows = current if history is None else torch.cat((history, current), dim=0)
        total_rows = int(rows.shape[0])
        ready_frames = max(0, total_rows - self.num_code_layers + 1)
        if ready_frames <= emitted_frames:
            return None

        frames: list[torch.Tensor] = []
        delay = self.num_code_layers - 1
        for frame_idx in range(emitted_frames, ready_frames):
            end = frame_idx + delay
            frame = torch.stack([rows[end - delay + layer, layer] for layer in range(self.num_code_layers)])
            # Upstream emits only fully active frames; stop/pad rows terminate audio.
            if (frame >= 2048).any():
                continue
            frames.append(frame)
        if not frames:
            return None
        return torch.stack(frames, dim=0).to(dtype=torch.long)

    def postprocess(self, hidden_states: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}

        request_id = kwargs.get("request_id")
        is_prefill = bool(kwargs.get("_omni_is_prefill", False))
        update: dict[str, Any] = {"hidden_states": {"last": hidden_states[-1:].detach()}}
        if is_prefill:
            return update

        codes = kwargs.get("codes", {}) if isinstance(kwargs, dict) else {}
        audio_raw = codes.get("audio") if isinstance(codes, dict) else None
        current = self._normalise_audio_code_rows(
            audio_raw,
        )
        if current is None:
            return update

        history_raw = codes.get("history") if isinstance(codes, dict) else None
        history = self._normalise_audio_code_rows(
            history_raw,
            device=current.device,
        )
        rows = current if history is None else torch.cat((history, current), dim=0)
        if isinstance(request_id, str) and current[-1, -1].item() == self.audio_stop_token:
            self._stop_pending_by_req[request_id] = True
        ready_frames = max(0, int(rows.shape[0]) - self.num_code_layers + 1)
        update.setdefault("codes", {})["history"] = rows.detach()
        update.setdefault("meta", {})["emitted_audio_frames"] = ready_frames
        return update

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for req_id in finished_req_ids:
            self._stop_pending_by_req.pop(req_id, None)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **_kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        info_dicts = _kwargs.get("model_intermediate_buffer") or _kwargs.get("runtime_additional_information")
        audio_codes_list: list[torch.Tensor] = []
        if isinstance(info_dicts, list):
            for info in info_dicts:
                if not isinstance(info, dict):
                    continue
                codes = info.get("codes", {})
                if not isinstance(codes, dict):
                    continue
                audio_raw = codes.get("audio")
                history_raw = codes.get("history")
                current = self._normalise_audio_code_rows(audio_raw)
                if current is None:
                    continue
                meta = info.get("meta", {})
                emitted = int(meta.get("emitted_audio_frames", 0)) if isinstance(meta, dict) else 0
                frames = self._ready_diagonal_audio_frames(history_raw, current, emitted)
                if frames is not None and frames.numel() > 0:
                    audio_codes_list.append(frames)
        if not audio_codes_list:
            return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={})
        audio_codes = torch.cat(audio_codes_list, dim=0)
        return OmniOutput(
            text_hidden_states=model_outputs[: audio_codes.shape[0]],
            multimodal_outputs={"codes": {"audio": audio_codes}},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_weights: set[str] = set()
        use_moe = bool(self.talker_config.use_moe)

        expert_params_mapping = []
        if use_moe:
            for layer in self.layers:
                mlp = getattr(layer, "mlp", None)
                experts = getattr(mlp, "experts", None)
                if experts is not None and hasattr(experts, "expert_mapping"):
                    expert_params_mapping = experts.expert_mapping
                    break
            if not expert_params_mapping:
                raise RuntimeError("MiniMind talker MoE is enabled but no FusedMoE expert_mapping was found.")

        for name, loaded_weight in self.hf_to_vllm_mapper.apply(weights):
            if name.startswith(("thinker.", "model.", "audio_proj.", "vision_proj.")):
                continue
            if "rotary_emb.inv_freq" in name:
                continue

            if use_moe and ".mlp.experts." in name:
                for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    param = params_dict.get(mapped_name)
                    if param is None:
                        break
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        mapped_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_weights.add(mapped_name)
                    break
                continue

            name_parts = name.split(".")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name_parts:
                    continue
                mapped_parts = [param_name if part == weight_name else part for part in name_parts]
                mapped_name = ".".join(mapped_parts)
                if mapped_name not in params_dict:
                    break
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_weights.add(mapped_name)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_weights.add(name)

        return loaded_weights
