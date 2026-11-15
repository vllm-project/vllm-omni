# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streaming Ming3 AudioVAE stage using the shared Ming implementation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import KEY_REQUEST_ID
from vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae import (
    MingAudioVAEModel,
    _coerce_finished,
    _coerce_latent_chunk,
    _first_present,
    _has_ming_context,
    _resolve_request_id,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .batched_vae_decoder import BatchedStreamingVAEDecoder
from .config_ming3_tts import Ming3TTSRuntimeConfig


class Ming3TTSAudioVAEModel(MingAudioVAEModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Keep the mature streaming decode/cache implementation but construct it
        # from Ming3's audio_config rather than Ming's audio_tokenizer_config.
        super(MingAudioVAEModel, self).__init__()
        del prefix
        self.vllm_config = vllm_config
        self.ming_config = Ming3TTSRuntimeConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()
        self.audio = AudioVAE(self.ming_config.audio_tokenizer_config, build_encoder=False)
        # Decode every active stream's chunk in one batched decoder pass; the
        # frame-holdback upsampler keeps the per-latent emission timing.
        self.audio.decoder.upsampling.frame_holdback = True
        self._batched: BatchedStreamingVAEDecoder | None = None
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self._past_key_values: dict[str, Any] = {}
        self._stream_state: dict[str, tuple[Any, Any, Any]] = {}
        self._state_access_times: OrderedDict[str, float] = OrderedDict()

    def _clear_request_state(self, request_id: str) -> None:
        super()._clear_request_state(request_id)
        if self._batched is not None:
            self._batched.release(request_id)

    def _batched_decoder(self) -> BatchedStreamingVAEDecoder:
        if self._batched is None:
            max_seqs = int(getattr(self.vllm_config.scheduler_config, "max_num_seqs", 64))
            self._batched = BatchedStreamingVAEDecoder(self.audio.decoder, max_streams=max(256, 4 * max_seqs))
        return self._batched

    @torch.inference_mode()
    def forward(
        self, *args: Any, model_intermediate_buffer: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> OmniOutput:
        info_list = model_intermediate_buffer if isinstance(model_intermediate_buffer, list) else None
        if info_list is None:
            runtime_infos = kwargs.get("runtime_additional_information")
            info_list = runtime_infos if isinstance(runtime_infos, list) else None
        info_list = info_list or [{}]
        sr_tensor = torch.tensor(int(self.ming_config.sample_rate), dtype=torch.int32)
        device = self.vllm_config.device_config.device
        dtype = next(self.audio.parameters()).dtype
        outputs: list[torch.Tensor | None] = []
        pending: list[tuple[int, str, torch.Tensor, bool]] = []
        for idx in range(max(len(info_list), 1)):
            info = info_list[idx] if idx < len(info_list) and isinstance(info_list[idx], dict) else {}
            meta = info.get("meta") or {}
            kv_meta = info.get("kv_metadata") or {}
            has_ming_context = _has_ming_context(info)
            if has_ming_context and KEY_REQUEST_ID not in info and KEY_REQUEST_ID not in kv_meta:
                raise RuntimeError(f"Ming3 Stage-1 received a payload without {KEY_REQUEST_ID}. keys={sorted(info)}")
            request_id = _resolve_request_id(info, idx)
            self._touch_request_state(request_id)
            finished = _coerce_finished(
                _first_present(
                    meta.get("stream_finished"),
                    meta.get("finished"),
                    info.get("stream_finished"),
                    kv_meta.get("stream_finished"),
                    torch.tensor(True),
                )
            )
            if _first_present(meta.get("_ming_payload_stripped"), info.get("_ming_payload_stripped"), False):
                raise RuntimeError(f"Ming3 Stage-1 payload was stripped before model entry. request_id={request_id}")
            latent = _first_present(
                info.get("latent"), info.get("ming_latent_patches"), kv_meta.get("ming_latent_patches")
            )
            # Validate on the host; all chunks of the step move to the device together below.
            latent_tensor = (
                None
                if latent is None
                else _coerce_latent_chunk(
                    latent,
                    device=latent.device if isinstance(latent, torch.Tensor) else torch.device("cpu"),
                    dtype=latent.dtype if isinstance(latent, torch.Tensor) else torch.float32,
                    latent_dim=self.ming_config.latent_dim,
                    patch_size=self.ming_config.patch_size,
                )
            )
            if latent_tensor is None or latent_tensor.numel() == 0:
                if has_ming_context and not finished:
                    raise RuntimeError(f"Ming3 Stage-1 received no latent chunk for unfinished request {request_id}")
                if finished:
                    self._clear_request_state(request_id)
                outputs.append(torch.zeros((0,), dtype=torch.float32, device=device))
                continue
            outputs.append(None)
            pending.append((idx, request_id, latent_tensor, finished))

        if pending:
            # One host-to-device copy for every chunk instead of a blocking copy each.
            chunks = [item[2] for item in pending]
            if all(chunk.device.type == "cpu" for chunk in chunks):
                host = torch.cat([chunk.to(torch.float32) for chunk in chunks])
                if device.type == "cuda":
                    host = host.pin_memory()
                moved = host.to(device, non_blocking=True).to(dtype).split([chunk.shape[0] for chunk in chunks])
            else:
                moved = [chunk.to(device=device, dtype=dtype) for chunk in chunks]
            pending = [
                (idx, request_id, chunk, finished) for (idx, request_id, _, finished), chunk in zip(pending, moved)
            ]

        decoder = self._batched_decoder()
        while pending:
            # A request with several queued chunks decodes them in order.
            seen: set[str] = set()
            batch: list[tuple[int, str, torch.Tensor, bool]] = []
            rest: list[tuple[int, str, torch.Tensor, bool]] = []
            for item in pending:
                (rest if item[1] in seen else batch).append(item)
                seen.add(item[1])
            waveforms = decoder.decode([(request_id, latent, finished) for _, request_id, latent, finished in batch])
            for (idx, request_id, _, finished), waveform in zip(batch, waveforms, strict=True):
                outputs[idx] = waveform
                if finished:
                    self._clear_request_state(request_id)
            pending = rest
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": outputs, "sr": [sr_tensor] * len(outputs)},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = list(weights)
        params = dict(self.named_parameters(remove_duplicate=False))
        buffers = dict(self.named_buffers(remove_duplicate=False))
        targets = set(params) | set(buffers)
        required = set(self.state_dict())
        unknown = [name for name, _ in weights if name not in targets]
        if unknown:
            raise RuntimeError(f"Ming3 AudioVAE received {len(unknown)} unknown checkpoint tensors: {unknown[:8]}")
        loaded: set[str] = set()
        for name, value in weights:
            if name in params:
                parameter = params[name]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, value)
            else:
                buffer = buffers[name]
                buffer.copy_(value.to(device=buffer.device, dtype=buffer.dtype))
            loaded.add(name)
        missing = {name for name in required if name.startswith("audio.")} - loaded
        if missing:
            raise RuntimeError(
                f"Ming3 AudioVAE left {len(missing)} checkpoint-backed tensors unloaded: {sorted(missing)[:8]}"
            )
        # The stage runs eagerly; its decoder captures its own graphs once the
        # weights are in place, before memory profiling.
        if next(self.audio.parameters()).is_cuda:
            self._batched_decoder().capture_graphs()
        return loaded

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            self._clear_request_state(request_id)


__all__ = ["Ming3TTSAudioVAEModel"]
