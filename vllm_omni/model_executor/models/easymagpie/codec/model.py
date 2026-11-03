# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""vLLM/vLLM-Omni model adapter for the stateful packed codec."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.model_executor.models.easymagpie.codec.config import EasyMagpieCodecConfig
from vllm_omni.model_executor.models.easymagpie.codec.packed import PackedEasyMagpieCodec
from vllm_omni.model_executor.models.easymagpie.watermark import (
    AudioChunk,
    DisabledAudioWatermarker,
    create_audio_watermarker,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput


def _chunk_request_id(info: dict[str, Any] | None, index: int) -> str:
    if isinstance(info, dict):
        for key in ("request_id", "req_id", "external_req_id"):
            value = info.get(key)
            if value is not None:
                return str(value)
    return f"codec-{index}"


class EasyMagpieCodecForConditionalGeneration(nn.Module):
    """One-placeholder-per-frame vLLM model producing packed waveform chunks.

    Real code matrices arrive through ``runtime_additional_information``. The
    scheduled ``input_ids`` are only placeholders, but their per-request counts
    must equal the number of EasyMagpie model frames so vLLM metadata is at the
    base time resolution used by every cache layer.
    """

    input_modalities = "audio"
    has_inner_state = True
    is_attention_free = True
    # Eager decoding allocates fresh audio; internal graph replay returns a
    # clone. The runner can retain these outputs during its asynchronous copy.
    owns_generation_output_storage = True

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int]]:
        from vllm_omni.model_executor.models.easymagpie.codec.packed import CODEC_STATE_ELEMENTS

        return ((CODEC_STATE_ELEMENTS,),)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype]:
        return (vllm_config.model_config.dtype,)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.linear_attention_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not isinstance(config, EasyMagpieCodecConfig):
            # AutoConfig registration is process-global and some inspection
            # paths may deserialize a generic PretrainedConfig first.
            config = EasyMagpieCodecConfig(**config.to_dict())
        self.config = config
        from vllm_omni.model_executor.models.easymagpie.codec.payload_validity import flag

        self._batched_payload_validity = flag() == "1"
        self._payload_counts = Counter()
        self._payload_fallbacks = Counter()
        self.vllm_config = vllm_config
        self.codec = PackedEasyMagpieCodec(
            config,
            dtype=vllm_config.model_config.dtype,
            prefix=prefix,
        )
        from vllm_omni.model_executor.models.easymagpie.codec.cuda_graph import CodecCudaGraphs, codec_graphs_enabled

        self._codec_graphs = CodecCudaGraphs(self.codec, vllm_config) if codec_graphs_enabled() else None
        from vllm_omni.model_executor.models.easymagpie.codec.uniform_groups import (
            CodecUniformGroups,
            uniform_groups_enabled,
        )

        self._codec_groups = (
            CodecUniformGroups(self.codec, vllm_config, self._codec_graphs) if uniform_groups_enabled() else None
        )
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.requires_raw_input_tokens = True
        self.watermarker = create_audio_watermarker(
            vllm_config.device_config.device,
            sample_rate=self.config.output_sample_rate,
            enabled=self.config.watermark_enabled,
            models_dir=self._watermark_models_dir(self.config, vllm_config),
        )

    @staticmethod
    def _watermark_models_dir(config: Any, vllm_config: VllmConfig) -> Path | None:
        relative = getattr(config, "watermark_checkpoint", None)
        if not relative:
            return None
        path = Path(str(relative))
        if path.is_absolute():
            return path
        model = getattr(getattr(vllm_config, "model_config", None), "model", None)
        if model:
            return Path(str(model)) / path
        return path

    def prewarm_codec_graphs(self, batch_sizes, frame_sizes=(2, 6, 8), metadata_template=None, *, descending=False):
        """Warm codec graph shapes after state-cache binding and before service."""
        if self._codec_graphs is None:
            return self.codec_graph_status()
        return self._codec_graphs.prewarm(batch_sizes, frame_sizes, metadata_template, descending=descending)

    def codec_graph_status(self):
        status = (
            {"enabled": False, "entries": 0, "captures": 0, "replays": 0, "fallbacks_by_reason": {}}
            if self._codec_graphs is None
            else self._codec_graphs.status()
        )
        status["uniform_groups"] = {"enabled": False} if self._codec_groups is None else self._codec_groups.status()
        return status

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], 1), dtype=torch.float32, device=input_ids.device)

    def codec_payload_status(self):
        return {
            "enabled": self._batched_payload_validity,
            "counts": dict(self._payload_counts),
            "fallbacks": dict(self._payload_fallbacks),
            "retained_payloads": 0,
            "retained_requests": 0,
            "retained_tensors": 0,
        }

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def _payload_codes(
        self,
        runtime_infos: list[dict[str, Any]],
        device: torch.device,
        request_token_spans: list[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, list[int], list[int], list[str]]:
        batched = False
        if getattr(self, "_batched_payload_validity", False):
            from vllm_omni.model_executor.models.easymagpie.codec.payload_validity import payload_batch_reason

            # An override may define other padding/control rules; retain it.
            native = EasyMagpieCodecForConditionalGeneration._valid_sample_count
            reason = (
                "validity_override"
                if type(self) is not EasyMagpieCodecForConditionalGeneration
                or getattr(self._valid_sample_count, "__func__", None) is not native
                else payload_batch_reason(self, runtime_infos, device, request_token_spans)
            )
            batched = reason is None
            if not batched:
                self._payload_counts["native_calls"] += 1
                self._payload_fallbacks[reason] += 1
        q = self.config.num_stacked_codebooks
        packed: list[torch.Tensor] = []
        frame_counts: list[int] = []
        valid_sample_counts: list[int] = []
        request_ids: list[str] = []
        if request_token_spans is not None and len(request_token_spans) != len(runtime_infos):
            raise ValueError(f"got {len(request_token_spans)} request spans for {len(runtime_infos)} codec payloads")
        for index, info in enumerate(runtime_infos):
            scheduled_frames = None
            if request_token_spans is not None:
                start, end = request_token_spans[index]
                scheduled_frames = end - start
                if scheduled_frames == 0:
                    continue
            codes = info.get("codes", {}) if isinstance(info, dict) else {}
            audio = codes.get("audio") if isinstance(codes, dict) else None
            tensor = torch.as_tensor(audio, dtype=torch.long)
            if tensor.ndim == 2:
                if tensor.shape[1] != q:
                    raise ValueError(f"expected codec payload [frames, {q}], got {tuple(tensor.shape)}")
                rows = tensor.contiguous()
            elif tensor.ndim == 1:
                flat = tensor.reshape(-1)
                if flat.numel() % q:
                    raise ValueError(f"codec payload length {flat.numel()} is not divisible by {q}")
                frames = flat.numel() // q
                rows = flat.view(q, frames).transpose(0, 1).contiguous()
            else:
                raise ValueError(f"expected a 1-D or 2-D codec payload, got {tuple(tensor.shape)}")
            if scheduled_frames is not None and rows.shape[0] != scheduled_frames:
                raise ValueError(
                    f"scheduled {scheduled_frames} placeholders for request {index}, "
                    f"but its codec payload has {rows.shape[0]} frames"
                )
            packed.append(rows)
            frame_counts.append(int(rows.shape[0]))
            valid_sample_counts.append(0 if batched else self._valid_sample_count(rows))
            request_ids.append(_chunk_request_id(info, index))
        if not packed:
            empty = torch.empty((0, q), dtype=torch.long, device=device)
            return empty, [], [], []
        if all(rows.device.type == "cpu" for rows in packed):
            codes = torch.cat(packed, dim=0)
            if batched:
                from vllm_omni.model_executor.models.easymagpie.codec.payload_validity import (
                    batched_valid_sample_counts,
                )

                valid_sample_counts = batched_valid_sample_counts(self, codes, frame_counts)
                self._payload_counts["batched_calls"] += 1
                self._payload_counts["batched_payloads"] += len(frame_counts)
                self._payload_counts["batched_frames"] += sum(frame_counts)
            codes = codes.to(device=device, non_blocking=True)
        else:
            codes = torch.cat([rows.to(device=device, non_blocking=True) for rows in packed], dim=0)
        return codes, frame_counts, valid_sample_counts, request_ids

    def _valid_sample_count(self, rows: torch.Tensor) -> int:
        frames = int(rows.shape[0])
        if frames == 0:
            return 0

        values = rows.detach()
        if values.device.type != "cpu":
            values = values.to(device="cpu")
        padding = values.lt(0).all(dim=1).nonzero(as_tuple=False)
        valid_frames = int(padding[0].item()) if padding.numel() else frames
        if valid_frames == 0:
            return 0

        control = (
            values[valid_frames - 1]
            .view(-1, self.config.frame_stacking_factor)
            .ge(self.config.codebook_size)
            .any(dim=0)
        )
        indices = control.nonzero(as_tuple=False)
        if indices.numel():
            valid_subframes = int(indices[0].item())
            return (valid_frames - 1) * self.config.samples_per_frame + (
                valid_subframes * self.config.samples_per_codec_frame
            )
        return valid_frames * self.config.samples_per_frame

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        codec_codes: torch.Tensor | None = None,
        request_token_spans: list[tuple[int, int]] | None = None,
        **_: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        if input_ids is None:
            input_ids = torch.empty((0,), dtype=torch.long, device=self.vllm_config.device_config.device)

        if codec_codes is not None:
            source_codes = codec_codes.to(dtype=torch.long)
            valid_sample_counts = [self._valid_sample_count(source_codes)]
            codes = source_codes.to(device=input_ids.device, non_blocking=True)
            frame_counts = [int(codes.shape[0])]
            request_ids = ["codec-codes"]
        elif runtime_additional_information:
            codes, frame_counts, valid_sample_counts, request_ids = self._payload_codes(
                runtime_additional_information,
                input_ids.device,
                request_token_spans,
            )
        else:
            # Profile runs do not carry connector payloads. One scheduled
            # placeholder still represents one model frame.
            frames = int(input_ids.numel())
            codes = torch.zeros(
                (frames, self.config.num_stacked_codebooks),
                dtype=torch.long,
                device=input_ids.device,
            )
            frame_counts = [frames]
            valid_sample_counts = [frames * self.config.samples_per_frame]
            request_ids = ["profile"]

        if codes.shape[0] != input_ids.numel():
            raise ValueError(
                "EasyMagpie native codec requires one scheduled placeholder per model frame: "
                f"got {input_ids.numel()} placeholders and {codes.shape[0]} code frames"
            )
        groups = getattr(self, "_codec_groups", None)
        graphs = getattr(self, "_codec_graphs", None)
        if groups is not None:
            packed_audio = groups.run(codes, frame_counts)
        else:
            packed_audio = self.codec(codes) if graphs is None else graphs.run(codes)
        outputs: list[torch.Tensor] = []
        offset = 0
        for frames, valid_samples in zip(frame_counts, valid_sample_counts, strict=True):
            samples = frames * self.config.samples_per_frame
            outputs.append(packed_audio[offset : offset + valid_samples].float())
            offset += samples
        watermarker = getattr(self, "watermarker", None)
        if (
            watermarker is not None
            and not isinstance(watermarker, DisabledAudioWatermarker)
            and (codec_codes is not None or runtime_additional_information)
        ):
            final_flags = (
                self._chunk_terminal_flags(codes, frame_counts, valid_sample_counts)
                if getattr(watermarker, "needs_terminal_flags", True)
                else [False] * len(outputs)
            )
            chunks = [
                AudioChunk(request_id=request_id, is_final=is_final, audio=audio)
                for request_id, is_final, audio in zip(request_ids, final_flags, outputs, strict=True)
            ]
            outputs = watermarker.apply(chunks)
        sample_rate = torch.tensor(self.config.output_sample_rate, dtype=torch.int32)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": outputs,
                "sr": [sample_rate for _ in outputs],
            },
        )

    def _chunk_terminal_flags(self, codes, frame_counts, valid_sample_counts):
        """Inspect actual control codes with one batched host read, only for watermarking."""
        rows, owners = [], []
        offset = 0
        for index, (frames, valid_samples) in enumerate(zip(frame_counts, valid_sample_counts, strict=True)):
            if frames:
                # A zero-subframe terminal is the next row after complete audio;
                # padding-only trimming selects a negative padding row instead.
                rows.append(offset + min(valid_samples // self.config.samples_per_frame, frames - 1))
                owners.append(index)
            offset += frames
        flags = [False] * len(frame_counts)
        if rows:
            indices = torch.tensor(rows, dtype=torch.long, device=codes.device)
            terminal = codes.index_select(0, indices).ge(self.config.codebook_size).any(dim=1).tolist()
            for index, is_final in zip(owners, terminal, strict=True):
                flags[index] = is_final
        return flags

    def make_omni_output(self, model_outputs: OmniOutput | tuple, **_: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(f"unexpected EasyMagpie codec output: {type(model_outputs)}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapped = ((f"codec.{name}" if name.startswith("audio_decoder.") else name, tensor) for name, tensor in weights)
        return AutoWeightsLoader(self).load_weights(
            (name, tensor) for name, tensor in mapped if not name.startswith("codec.dequantizer.")
        )
