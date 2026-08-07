# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-1 Code2Wav model for PersonaPlex (Moshi finetune).

This is the ``LLM_GENERATION`` codec stage of the 2-stage PersonaPlex pipeline.
It consumes the per-frame audio codebooks produced by the talker (stage 0) and
the depformer (built by the lead), and turns them into 24 kHz PCM by calling the
external Mimi neural codec from the ``moshi`` package.

Layout contract (mirrors ``Qwen3TTSCode2Wav``):

* Per request, ``input_ids`` holds a flat codebook-major codec sequence
  ``[k * F]`` where ``k`` is the number of *active* codebooks Mimi decodes
  (``num_codebooks``, i.e. the ``cb 0..7`` slice) and ``F`` is the number of
  codec frames. The talker emits a 17-row token stack per frame; the input
  processor (built by the lead) keeps only rows ``1:9`` (the 8 PCM-bearing audio
  codebooks) and flattens them codebook-major before this stage.
* ``forward(...)`` returns an :class:`OmniOutput` whose ``multimodal_outputs``
  carries ``{"model_outputs": [wav_per_request], "sr": [sr_per_request]}`` — the
  exact shape ``Qwen3TTSCode2Wav`` returns, so the downstream audio packer is
  unchanged.

The Mimi decoder is the transformers ``MimiModel`` (kyutai/mimi weights), not
in vLLM's safetensors loader, so they are loaded eagerly in ``load_weights``
part of the vLLM weights iterator (the codec owns its own checkpoint; see the
experimental fullduplex package for the same pattern).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

_MIMI_DECODE_BATCH_FRAMES = 5


def _codec_ids_from_payload_or_input(
    input_ids: torch.Tensor,
    runtime_info: Mapping[str, Any] | None,
) -> torch.Tensor:
    """Prefer connector-delivered codec ids over scheduler placeholders."""
    if isinstance(runtime_info, Mapping):
        codes = runtime_info.get("codes")
        if isinstance(codes, Mapping):
            audio = codes.get("audio")
            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                return audio.reshape(-1).to(device=input_ids.device, dtype=torch.long)
            if isinstance(audio, (list, tuple)) and audio:
                return torch.as_tensor(audio, device=input_ids.device, dtype=torch.long).reshape(-1)
    return input_ids.reshape(-1).to(dtype=torch.long)


class PersonaPlexCode2Wav(nn.Module):
    """Stage-1 Code2Wav model for PersonaPlex (GenerationModelRunner).

    Wraps the external Mimi decoder. The vLLM generation runner only calls a
    handful of methods on this module: ``embed_input_ids`` (dummy embeddings),
    ``compute_logits`` (none -- this stage never samples), ``forward`` (the
    actual codec->PCM decode), ``make_omni_output`` (output normalization), and
    ``load_weights`` (eager Mimi construction).
    """

    input_modalities = "audio"
    requires_request_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.config = vllm_config.model_config.hf_config

        # Runner-facing capability flags, matching Qwen3TTSCode2Wav so the
        # GenerationModelRunner drives this stage the same way.
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        mimi_cfg = getattr(self.config, "mimi_config", None)
        # Number of *active* codebooks Mimi decodes to PCM (cb 0..7).
        self._num_codebooks = int(getattr(mimi_cfg, "num_codebooks", 8))
        self._output_sample_rate = int(getattr(mimi_cfg, "sample_rate", 24000))
        self._samples_per_frame = int(getattr(mimi_cfg, "samples_per_frame", 1920))
        self._mimi_name = getattr(mimi_cfg, "mimi_name", None) or getattr(self.config, "mimi_name", None)
        self._max_codec_sessions = int(getattr(vllm_config.model_config, "duplex_max_sessions", 1))

        # The Mimi module is constructed in load_weights() (it owns its own
        # weight format) and assigned here so vLLM's memory profiler can see it.
        self.mimi: nn.Module | None = None
        self._additional_mimi = nn.ModuleList()
        self._mimi_device: torch.device | None = None
        self._request_codes: dict[str, torch.Tensor] = {}
        self._request_codec_slots: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Runner-facing no-op / placeholder hooks (mirror Qwen3TTSCode2Wav).
    # ------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # This stage ignores token embeddings; keep a stable dummy embedding so
        # the vLLM runner has a valid hidden-state placeholder.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        # Code2Wav never samples; the runner short-circuits when logits are None.
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, object]]:
        return [{"meta": {"personaplex_dummy_profile": True}} for _ in range(num_reqs)]

    def _split_request_ids(
        self,
        ids: torch.Tensor,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """Split concatenated ``input_ids`` into per-request segments.

        Uses ``seq_token_counts`` (injected by the runner via model_kwargs) when
        available, falling back to forward-context ``ubatch_slices`` when
        micro-batching is active. Returns ``[ids]`` for single-request batches.
        Mirrors ``Qwen3TTSCode2Wav._split_request_ids``.
        """
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    # ------------------------------------------------------------------
    # Decode.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode flat codebook-major codec ids into PCM via Mimi.

        ``input_ids`` per request is ``[k * F]`` (codebook-major), where ``k``
        is ``self._num_codebooks``. The connector may send either a new delta
        chunk or the cumulative prefix of a resumable request. Request-local
        code history identifies the new suffix, and the streaming Mimi decoder
        consumes each new frame exactly once.
        """
        sr_val = int(self._output_sample_rate)
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))
        num_req = len(request_ids_list)
        state_ids = self._resolve_request_ids(
            num_req,
            kwargs.get("request_ids"),
            runtime_additional_information,
        )

        if self.mimi is None:
            raise RuntimeError("PersonaPlexCode2Wav.forward called before Mimi was loaded in load_weights().")

        k = int(self._num_codebooks)
        device = self._mimi_device or ids.device
        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        for i, req_ids in enumerate(request_ids_list):
            runtime_info = (
                runtime_additional_information[i]
                if runtime_additional_information is not None and i < len(runtime_additional_information)
                else None
            )
            meta = runtime_info.get("meta") if isinstance(runtime_info, Mapping) else None
            if isinstance(meta, Mapping) and meta.get("personaplex_dummy_profile") is True:
                continue
            flat = _codec_ids_from_payload_or_input(req_ids, runtime_info)
            n = int(flat.numel())
            if n == 0 or n % k != 0:
                if n > 0:
                    logger.warning(
                        "PersonaPlex Code2Wav input_ids length %d not divisible by "
                        "num_codebooks %d; skipping malformed request.",
                        n,
                        k,
                    )
                continue
            frames = n // k
            codes_kf = flat.reshape(k, frames)
            state_id = state_ids[i]
            delta_kf = self._new_code_suffix(state_id, codes_kf)
            if delta_kf.shape[1] == 0:
                continue
            wav = self._decode_streaming_frames(state_id, delta_kf.to(device=device))
            if wav.numel() > 0:
                audios[i] = wav.to(dtype=torch.float32).reshape(-1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    @staticmethod
    def _resolve_request_ids(
        count: int,
        request_ids: object,
        runtime_additional_information: list[dict[str, Any]] | None,
    ) -> list[str | None]:
        if isinstance(request_ids, list | tuple):
            if len(request_ids) != count:
                raise ValueError(
                    "PersonaPlex Code2Wav request id count does not match inputs: "
                    f"request_ids={len(request_ids)}, inputs={count}"
                )
            return [str(request_id) for request_id in request_ids]

        infos = runtime_additional_information or []
        resolved: list[str | None] = []
        for index in range(count):
            info = infos[index] if index < len(infos) else None
            request_id = None
            if isinstance(info, Mapping):
                request_id = info.get("request_id")
                if request_id is None:
                    meta = info.get("meta")
                    if isinstance(meta, Mapping):
                        request_id = meta.get("request_id")
            resolved.append(str(request_id) if request_id is not None else None)
        return resolved

    def _new_code_suffix(self, request_id: str | None, codes_kf: torch.Tensor) -> torch.Tensor:
        if request_id is None:
            return codes_kf

        incoming = codes_kf.detach().to(device="cpu", dtype=torch.long)
        previous = self._request_codes.get(request_id)
        if (
            previous is not None
            and incoming.shape[1] >= previous.shape[1]
            and torch.equal(incoming[:, : previous.shape[1]], previous)
        ):
            delta = incoming[:, previous.shape[1] :]
            self._request_codes[request_id] = incoming
            return delta

        self._request_codes[request_id] = incoming if previous is None else torch.cat([previous, incoming], dim=1)
        return incoming

    def _decode_streaming_frames(self, request_id: str | None, codes_kf: torch.Tensor) -> torch.Tensor:
        codec, ephemeral = self._codec_for_request(request_id)
        if not hasattr(codec, "decode_frame"):
            raise RuntimeError("PersonaPlex Code2Wav requires a streaming Mimi decoder")

        decode_frames = getattr(codec, "decode_frames", None)
        chunks: list[torch.Tensor] = []
        for start in range(0, codes_kf.shape[1], _MIMI_DECODE_BATCH_FRAMES):
            frame_batch = codes_kf[:, start : start + _MIMI_DECODE_BATCH_FRAMES]
            if frame_batch.shape[1] > 1 and callable(decode_frames):
                chunks.append(self._flatten_wav(decode_frames(frame_batch.unsqueeze(0))))
            else:
                chunks.extend(
                    self._flatten_wav(codec.decode_frame(frame_batch[:, frame].unsqueeze(0)))
                    for frame in range(frame_batch.shape[1])
                )
        wav = torch.cat(chunks, dim=0) if chunks else codes_kf.new_empty(0, dtype=torch.float32)
        if ephemeral:
            codec.reset_streaming()
        return wav

    def _set_mimi_codecs(self, codecs: list[nn.Module]) -> None:
        if not codecs:
            raise ValueError("PersonaPlex Code2Wav requires at least one Mimi decoder")
        self.mimi = codecs[0]
        self._additional_mimi = nn.ModuleList(codecs[1:])
        self._request_codec_slots.clear()

    def _mimi_codecs(self) -> list[nn.Module]:
        if self.mimi is None:
            return []
        return [self.mimi, *self._additional_mimi]

    def _codec_for_request(self, request_id: str | None) -> tuple[nn.Module, bool]:
        codecs = self._mimi_codecs()
        if not codecs:
            raise RuntimeError("PersonaPlexCode2Wav.forward called before Mimi was loaded in load_weights().")
        occupied = set(self._request_codec_slots.values())
        if request_id is not None:
            slot = self._request_codec_slots.get(request_id)
            if slot is None:
                slot = next((index for index in range(len(codecs)) if index not in occupied), None)
                if slot is None:
                    raise RuntimeError(f"PersonaPlex Code2Wav decoder capacity {len(codecs)} is exhausted")
                codecs[slot].streaming_init(1)
                self._request_codec_slots[request_id] = slot
            return codecs[slot], False

        slot = next((index for index in range(len(codecs)) if index not in occupied), None)
        if slot is None:
            raise RuntimeError(f"PersonaPlex Code2Wav decoder capacity {len(codecs)} is exhausted")
        codecs[slot].streaming_init(1)
        return codecs[slot], True

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            state_id = str(request_id)
            self._request_codes.pop(state_id, None)
            slot = self._request_codec_slots.pop(state_id, None)
            codecs = self._mimi_codecs()
            if slot is not None and slot < len(codecs):
                codecs[slot].reset_streaming()

    @staticmethod
    def _flatten_wav(wav: torch.Tensor) -> torch.Tensor:
        """Normalize Mimi's decode output to a 1D PCM tensor.

        Mimi may return ``[B, samples]`` or ``[B, C, samples]`` depending on the
        moshi version; PersonaPlex is mono (``C == 1``), so collapse to 1D.
        """
        if wav.dim() == 3:
            # [B, C, T] -> take batch 0, channel 0.
            return wav[0, 0]
        if wav.dim() == 2:
            # [B, T] -> batch 0.
            return wav[0]
        return wav.reshape(-1)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)

        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(
                "PersonaPlexCode2Wav expected OmniOutput, OmniOutput tuple, "
                f"or (audio_tensor, sr) outputs, got {type(model_outputs)}"
            )

        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_tensor,
                "sr": sr,
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Construct and load the external Mimi decoder.

        The primary vLLM weights iterator carries no Code2Wav parameters (Mimi
        owns its own checkpoint format), so it is drained and the Mimi module is
        built from the moshi package's loader.
        """
        # Drain the primary iterator so callers don't hang on an unconsumed
        # generator; none of these weights belong to this stage.
        for _ in weights:
            pass

        device = self.vllm_config.device_config.device
        from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
            PersonaPlexMimiCodec,
        )

        checkpoint = Path(self.model_path) / (self._mimi_name or "tokenizer-e351c8d8-checkpoint125.safetensors")
        codecs = [
            PersonaPlexMimiCodec(
                checkpoint=str(checkpoint) if checkpoint.is_file() else None,
                device=str(device),
            ).eval()
            for _ in range(self._max_codec_sessions)
        ]
        self._set_mimi_codecs(codecs)
        self._mimi_device = torch.device(str(device))
        reported_sr = getattr(codecs[0].model.config, "sampling_rate", None)
        if reported_sr is not None:
            self._output_sample_rate = int(reported_sr)
        bytes_per_mib = 1024**2
        per_slot_weight_mib = (
            sum(parameter.numel() * parameter.element_size() for parameter in codecs[0].parameters()) / bytes_per_mib
        )
        total_weight_mib = (
            sum(parameter.numel() * parameter.element_size() for codec in codecs for parameter in codec.parameters())
            / bytes_per_mib
        )
        logger.info(
            "PersonaPlex Code2Wav loaded %d Mimi stream(s): "
            "per_slot_weight_mib=%.2f total_weight_mib=%.2f "
            "num_codebooks=%d sample_rate=%d samples_per_frame=%d",
            len(codecs),
            per_slot_weight_mib,
            total_weight_mib,
            self._num_codebooks,
            self._output_sample_rate,
            self._samples_per_frame,
        )
        return {name for name, _ in self.named_parameters() if name.startswith(("mimi.", "_additional_mimi."))}
