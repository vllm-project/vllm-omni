# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
``duplex`` subpackage of this model folder for the same pattern).
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

    One streaming decoder serves every session: each request id leases a row of
    its streaming state, and a step decodes all requests' new frames together,
    one ``decode_frame`` call per frame index across all rows. With
    ``mimi_cuda_graphs`` each such call replays a CUDA graph captured at load.
    """

    input_modalities = "audio"
    requires_request_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self._async_chunk = bool(getattr(vllm_config.model_config, "async_chunk", False))
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
        # Rows 0..max_sessions-1 are leased per request id; the last row is
        # scratch for a request without an id and is reset after every use.
        self._scratch_row = self._max_codec_sessions
        self._num_codec_rows = self._max_codec_sessions + 1

        # The Mimi module is constructed in load_weights() (it owns its own
        # weight format) and assigned here so vLLM's memory profiler can see it.
        self.mimi: nn.Module | None = None
        self._mimi_device: torch.device | None = None
        self._request_rows: dict[str, int] = {}
        self._consumed_full_payload_requests: set[str] = set()

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
        is ``self._num_codebooks``. Async inputs contain newly generated delta
        frames. Sync connector payloads contain the full sequence and may
        persist across forwards, so they are consumed once per request. Every
        request of the step is decoded in the same shared-decoder pass (see
        ``_decode_pending``).
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
        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req
        # (output index, request id, new codes [k, F]) of every request with frames to decode.
        pending: list[tuple[int, str | None, torch.Tensor]] = []

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
            if not self._async_chunk and state_id is not None:
                if state_id in self._consumed_full_payload_requests:
                    continue
                self._consumed_full_payload_requests.add(state_id)
            pending.append((i, state_id, codes_kf))

        for i, wav in self._decode_pending(pending):
            audios[i] = wav

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

    def _decode_pending(
        self,
        pending: list[tuple[int, str | None, torch.Tensor]],
    ) -> list[tuple[int, torch.Tensor]]:
        """Decode the step's requests: one pass for all leased rows, one more per extra id-less request.

        An id-less request decodes on the scratch row, which is reset after each pass.
        """
        leased = [(i, self._lease_row(request_id), codes) for i, request_id, codes in pending if request_id is not None]
        anonymous = [(i, self._scratch_row, codes) for i, request_id, codes in pending if request_id is None]
        passes = [leased + anonymous[:1], *([item] for item in anonymous[1:])]
        decoded: list[tuple[int, torch.Tensor]] = []
        for items in passes:
            if items:
                decoded.extend(self._decode_rows(items))
        return decoded

    def _decode_rows(self, items: list[tuple[int, int, torch.Tensor]]) -> list[tuple[int, torch.Tensor]]:
        """Decode frame ``f`` of every ``(output index, row, codes [k, F])`` item in one call, for each ``f``.

        A row without a frame ``f`` is inactive for that call, so its streaming state does not advance.
        """
        codec = self.mimi
        k = int(self._num_codebooks)
        num_frames = max(int(codes_kf.shape[1]) for _, _, codes_kf in items)
        codes = torch.zeros((num_frames, self._num_codec_rows, k), dtype=torch.long)
        active = torch.zeros((num_frames, self._num_codec_rows), dtype=torch.bool)
        for _, row, codes_kf in items:
            frames = int(codes_kf.shape[1])
            codes[:frames, row] = codes_kf.to(device="cpu", dtype=torch.long).T
            active[:frames, row] = True
        codes = codes.to(device=self._mimi_device)
        active = active.to(device=self._mimi_device)
        pcm = torch.stack([codec.decode_frame(codes[f], active[f]) for f in range(num_frames)])
        pcm = pcm.to(device="cpu", dtype=torch.float32)  # [F, rows, samples]
        if any(row == self._scratch_row for _, row, _ in items):
            codec.reset_slot(self._scratch_row)
        # Clone so each request's PCM owns its storage instead of viewing the whole pass.
        return [(i, pcm[: codes_kf.shape[1], row].clone().reshape(-1)) for i, row, codes_kf in items]

    def _lease_row(self, request_id: str) -> int:
        row = self._request_rows.get(request_id)
        if row is not None:
            return row
        occupied = set(self._request_rows.values())
        row = next((index for index in range(self._max_codec_sessions) if index not in occupied), None)
        if row is None:
            raise RuntimeError(f"PersonaPlex Code2Wav decoder capacity {self._max_codec_sessions} is exhausted")
        self._request_rows[request_id] = row
        return row

    def _install_mimi(self, codec: nn.Module, device: torch.device) -> None:
        """Share ``codec`` across sessions: one streaming row per session plus the scratch row."""
        codec.streaming_init(self._num_codec_rows)
        self.mimi = codec
        self._mimi_device = device
        self._request_rows.clear()

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            state_id = str(request_id)
            self._consumed_full_payload_requests.discard(state_id)
            row = self._request_rows.pop(state_id, None)
            if row is not None and self.mimi is not None:
                self.mimi.reset_slot(row)

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
        codec = PersonaPlexMimiCodec(
            checkpoint=str(checkpoint) if checkpoint.is_file() else None,
            device=str(device),
        ).eval()
        # Allocate the streaming state and the decode graph's pool here, not on
        # the first request, so vLLM's memory profiling sees them.
        self._install_mimi(codec, torch.device(str(device)))
        # Stage 1 runs with enforce_eager (the runner's own capture records
        # nothing for this model); the flag alone decides the codec's graph.
        if getattr(self.config, "mimi_cuda_graphs", False):
            codec.capture_decode_graph()
        reported_sr = getattr(codec.model.config, "sampling_rate", None)
        if reported_sr is not None:
            self._output_sample_rate = int(reported_sr)
        weight_mib = sum(parameter.numel() * parameter.element_size() for parameter in codec.parameters()) / 1024**2
        logger.info(
            "PersonaPlex Code2Wav loaded one shared Mimi decoder: rows=%d (%d sessions + 1 scratch) "
            "weight_mib=%.2f num_codebooks=%d sample_rate=%d samples_per_frame=%d",
            self._num_codec_rows,
            self._max_codec_sessions,
            weight_mib,
            self._num_codebooks,
            self._output_sample_rate,
            self._samples_per_frame,
        )
        return {name for name, _ in self.named_parameters() if name.startswith("mimi.")}
