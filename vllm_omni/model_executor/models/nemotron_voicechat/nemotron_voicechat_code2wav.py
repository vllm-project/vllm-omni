# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-2 Code2Wav for NemotronVoiceChat: RVQ-VAE codec decode to 22.05 kHz PCM.

``LLM_GENERATION`` stage consuming the talker's per-frame 31-quantizer code
stacks and decoding them with the vendored NeMo RVQ-VAE codec. The decode runs
in fp32 (the NeMo reference wraps codec decode in an fp32 context; bf16 collapses
the reconstruction).

Input contract (mirrors ``PersonaPlexCode2Wav``): the connector payload carries
``{"codes": {"audio": Tensor[frames, num_quantizers]}}`` per request; the
scheduler-side ``input_ids`` are placeholders. Codes must lie in
``[0, codebook_size)`` — out-of-range codes are a hard error (an early bf16-drift
signal upstream), never clamped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.nemotron_voicechat.runtime_info import scalar_bool
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def validate_code_stack(codes: torch.Tensor, num_quantizers: int, codebook_size: int) -> torch.Tensor:
    """Validate and reshape a code payload to ``[frames, num_quantizers]``.

    Out-of-range codes are a hard error (an early numeric-drift signal
    upstream), never clamped.
    """
    q = num_quantizers
    if codes.dim() == 1:
        if codes.numel() % q != 0:
            raise ValueError(
                f"NemotronVoiceChat code2wav received a flat code sequence of length "
                f"{codes.numel()} which is not divisible by num_quantizers={q}."
            )
        codes = codes.reshape(-1, q)
    if codes.dim() != 2 or codes.shape[-1] != q:
        raise ValueError(f"NemotronVoiceChat code2wav expects codes shaped [frames, {q}], got {tuple(codes.shape)}.")
    codes = codes.to(dtype=torch.long)
    bad = (codes < 0) | (codes >= codebook_size)
    if bool(bad.any()):
        bad_vals = codes[bad]
        raise ValueError(
            "NemotronVoiceChat code2wav received out-of-range codec codes "
            f"(min={int(bad_vals.min())}, max={int(bad_vals.max())}, count={int(bad.sum())}); "
            f"valid range is [0, {codebook_size}). Refusing to clamp — this usually "
            "signals numeric drift or a corrupted stage payload upstream."
        )
    return codes


class NemotronVoiceChatCode2Wav(nn.Module):
    """RVQ-VAE codec decoder stage (GenerationModelRunner)."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        codec_cfg = getattr(self.config, "code2wav_config", None)
        self._num_quantizers = int(getattr(codec_cfg, "num_quantizers", 31))
        self._codebook_size = int(getattr(codec_cfg, "codebook_size", 1024))
        self._sample_rate = int(getattr(codec_cfg, "sample_rate", 22050))
        # Samples per 12.5 Hz frame (streaming chunks slice new audio by frame).
        self._wav_per_frame = int(getattr(codec_cfg, "wav_to_token_ratio", 1764))

        # Runner-facing capability flags (same contract as PersonaPlexCode2Wav).
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        # Constructed in load_weights (owns the tts_model.audio_codec.* subtree).
        self.audio_codec: nn.Module | None = None
        self._duplex_codec_caches: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Runner-facing placeholder hooks.
    # ------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, object]]:
        return [{"meta": {"nemotron_voicechat_dummy_profile": True}} for _ in range(num_reqs)]

    # ------------------------------------------------------------------
    # Decode.
    # ------------------------------------------------------------------
    def _codes_from_runtime_info(self, runtime_info: Mapping[str, Any] | None) -> torch.Tensor | None:
        """Extract the codes payload; None means "no codes info at all".

        An EMPTY tensor is returned as-is: it is an explicit empty payload
        (e.g. zero acoustic frames after the prompt trim) and must yield empty
        audio, NOT a fallback decode of the placeholder input_ids.
        """
        if not isinstance(runtime_info, Mapping):
            return None
        if "codes" in runtime_info:
            codes = runtime_info["codes"]
            if isinstance(codes, Mapping):
                codes = codes.get("audio")
        else:
            codes = runtime_info.get("codes.audio")
        if isinstance(codes, list | tuple):
            codes = torch.as_tensor(codes, dtype=torch.long)
        if isinstance(codes, torch.Tensor):
            return codes
        return None

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
        sr_tensor = torch.tensor(self._sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        infos = runtime_additional_information or []
        num_req = max(len(infos), 1)
        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        if self.audio_codec is None:
            raise RuntimeError("NemotronVoiceChatCode2Wav.forward called before load_weights().")
        device = next(self.audio_codec.parameters()).device

        for i in range(num_req):
            runtime_info = infos[i] if i < len(infos) else None
            meta = runtime_info.get("meta") if isinstance(runtime_info, Mapping) else None
            if isinstance(meta, Mapping) and meta.get("nemotron_voicechat_dummy_profile") is True:
                continue
            codec_streaming = scalar_bool(meta.get("codec_streaming")) if isinstance(meta, Mapping) else False
            if isinstance(runtime_info, Mapping):
                codec_streaming = codec_streaming or scalar_bool(runtime_info.get("meta.codec_streaming"))
            cache_request_id = meta.get("request_id") if isinstance(meta, Mapping) else None
            if cache_request_id is None and isinstance(runtime_info, Mapping):
                cache_request_id = runtime_info.get("meta.request_id")
            codes = self._codes_from_runtime_info(runtime_info)
            if codes is None:
                # No input_ids fallback (unlike PersonaPlexCode2Wav): this
                # pipeline's scheduler-side prompt is all-zero placeholders, so
                # decoding it would produce plausible-sounding garbage whenever
                # n_frames % num_quantizers == 0. A missing payload is a bug.
                raise ValueError(
                    "NemotronVoiceChat code2wav request carries no 'codes' payload in its "
                    "runtime info; the talker stage must ship the [frames, num_quantizers] "
                    "code stack (placeholder input_ids are never decoded)."
                )
            codes = validate_code_stack(codes, self._num_quantizers, self._codebook_size).to(device=device)
            logger.debug(
                "Nemotron VoiceChat code2wav input: request=%s shape=%s codec_streaming=%s",
                cache_request_id,
                tuple(codes.shape),
                codec_streaming,
            )
            frames = codes.shape[0]
            if frames == 0:
                continue
            # Async-chunk streaming: the payload carries the CUMULATIVE trimmed
            # code history plus meta.left_context_size = frames whose audio was
            # already emitted by earlier chunks. Mirroring NeMo's
            # decode_one_audio_step (unbounded window), we re-decode the whole
            # prefix and slice off only the new tail — seam-free by
            # construction. Sync full-payload chunks carry no left context.
            left_context_frames = 0
            if isinstance(meta, Mapping) and meta.get("left_context_size") is not None:
                left_context_frames = max(int(meta["left_context_size"]), 0)
            elif isinstance(runtime_info, Mapping) and runtime_info.get("meta.left_context_size") is not None:
                left_context_frames = max(int(runtime_info["meta.left_context_size"]), 0)
            if left_context_frames >= frames:
                continue  # terminal empty chunk: all frames already emitted
            lens = torch.tensor([frames], device=device, dtype=torch.long)
            codec_cache = None
            if codec_streaming:
                if not isinstance(cache_request_id, str) or not cache_request_id:
                    raise ValueError("Nemotron VoiceChat incremental codec input lacks meta.request_id")
                from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.ear_tts_vae_codec import (
                    CausalConv1dCache,
                )

                codec_cache = self._duplex_codec_caches.setdefault(cache_request_id, CausalConv1dCache())
            # NeMo decodes the codec strictly in fp32.
            with torch.autocast(device_type=device.type, enabled=False):
                wav, wav_len = self.audio_codec.decode(
                    codes.unsqueeze(0),
                    lens,
                    cache=codec_cache,
                )
            wav = wav.reshape(-1)[: int(wav_len.reshape(-1)[0])]
            if left_context_frames > 0 and not codec_streaming:
                wav = wav[left_context_frames * self._wav_per_frame :]
            audios[i] = wav.detach().to(dtype=torch.float32).cpu()
            logger.debug(
                "Nemotron VoiceChat code2wav output: request=%s frames=%s samples=%s",
                cache_request_id,
                frames,
                int(audios[i].numel()),
            )

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            self._duplex_codec_caches.pop(str(request_id), None)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        raise TypeError(f"NemotronVoiceChatCode2Wav expected OmniOutput, got {type(model_outputs)}")

    # ------------------------------------------------------------------
    # Weights.
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Build the vendored RVQ-VAE codec and load ``tts_model.audio_codec.*``.

        The vLLM safetensors iterator streams the full unified checkpoint; only
        the codec subtree is materialized here (everything else is skipped
        without copying). The codec stays fp32.
        """
        from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.ear_tts_vae_codec import (
            RVQVAEModel,
        )

        codec_cfg = dict(getattr(self.config, "tts_cfg", {}).get("codec_config") or {})
        if not codec_cfg:
            raise ValueError(
                "NemotronVoiceChat checkpoint config lacks "
                "'model.speech_generation.model.codec_config'; cannot build the RVQ-VAE codec."
            )
        from omegaconf import DictConfig

        codec = RVQVAEModel(DictConfig(codec_cfg))

        prefix = "tts_model.audio_codec."
        state: dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            if name.startswith(prefix):
                state[name.removeprefix(prefix)] = tensor.to(dtype=torch.float32)
        if not state:
            raise ValueError(
                f"No '{prefix}*' tensors found in the checkpoint; the NemotronVoiceChat "
                "unified model.safetensors is required for the code2wav stage."
            )
        missing, unexpected = codec.load_state_dict(state, strict=False)
        if missing:
            raise ValueError(f"NemotronVoiceChat code2wav is missing codec weights: {sorted(missing)[:10]} ...")
        if unexpected:
            logger.warning(
                "NemotronVoiceChat code2wav ignoring %d unexpected codec tensors (e.g. %s)",
                len(unexpected),
                sorted(unexpected)[:5],
            )
        device = self.vllm_config.device_config.device
        codec = codec.to(device=device, dtype=torch.float32).eval()
        self.audio_codec = codec
        logger.info(
            "NemotronVoiceChat code2wav loaded RVQ-VAE codec: %d tensors, num_quantizers=%d, sr=%d",
            len(state),
            self._num_quantizers,
            self._sample_rate,
        )
        return {name for name, _ in self.named_parameters()}
