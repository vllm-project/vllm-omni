# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kokoro-82M single-stage TTS for vLLM-Omni.

Kokoro is non-autoregressive: one network pass turns a <=510-phoneme chunk into
its whole waveform. It runs in a single AR worker stage using the generator
pattern from MOSS-TTS-Nano, with one AR step per chunk:

- The first ``forward()`` for a request converts its text to phoneme chunks and
  stores a per-request generator in ``self._stream_gens``.
- Each ``forward()`` synthesizes the request's next chunk and returns its audio
  (delta semantics: only the new samples).
- ``compute_logits()`` emits the stop token on the step that yields the last
  chunk, so the scheduler finishes the request.

Per-request state (generator, RNG stream) is keyed by the engine request id and
released in ``on_requests_finished``.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.kokoro.configuration_kokoro import (
    KOKORO_CONTINUE_TOKEN_ID,
    KOKORO_STOP_TOKEN_ID,
    KokoroConfig,
)
from vllm_omni.model_executor.models.kokoro.kokoro_modules import KokoroNetwork, fold_weight_norm
from vllm_omni.model_executor.models.kokoro.text_frontend import (
    DEFAULT_VOICE,
    KokoroTextFrontend,
    list_voices,
    phonemes_to_ids,
    resolve_lang_code,
    resolve_voices_dir,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# Checkpoint components, in the order the reference loads them.
_COMPONENTS = ("bert", "bert_encoder", "predictor", "decoder", "text_encoder")


def _pick(info: dict[str, Any], key: str, default: Any) -> Any:
    """Read a scalar from ``additional_information`` (list-wrapped or plain)."""
    val = info.get(key, default)
    if isinstance(val, (list, tuple)):
        val = val[0] if val else default
    return default if val is None else val


def _flatten_checkpoint(weights: Iterable[tuple[str, Any]]) -> dict[str, dict[str, torch.Tensor]]:
    """Group the ``.pth`` items into per-component state dicts.

    vLLM's ``.pt`` iterator yields the checkpoint's top-level items, which for
    Kokoro are whole component state dicts saved from ``DataParallel``
    (``module.`` prefix).
    """
    components: dict[str, dict[str, torch.Tensor]] = {}
    for name, value in weights:
        if isinstance(value, dict):
            components[name] = {k.removeprefix("module."): v for k, v in value.items()}
        else:
            component, _, param = name.partition(".")
            components.setdefault(component, {})[param.removeprefix("module.")] = value
    return components


class KokoroForConditionalGeneration(nn.Module):
    """Kokoro-82M with chunk-per-step streaming audio output."""

    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True
    # The checkpoint is a single ``kokoro-v1_0.pth``; voicepacks are separate
    # ``voices/*.pt`` files, all loaded in ``load_weights()``.
    allow_patterns_overrides = ["*.pth"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config: KokoroConfig = vllm_config.model_config.hf_config
        self.model_path: str = vllm_config.model_config.model
        self.revision: str | None = vllm_config.model_config.revision
        self._device = current_omni_platform.get_torch_device()

        self.network = KokoroNetwork(self.config)
        self.network.eval()
        self.frontend = KokoroTextFrontend()

        # Every voice pack, loaded on the device in load_weights().
        self._voicepacks: dict[str, torch.Tensor] = {}
        self._stream_gens: dict[str, Iterator[tuple[torch.Tensor | None, bool]]] = {}
        # Requests that already yielded their last chunk. With async scheduling
        # the engine schedules one more step before it sees the stop token;
        # that step must not start a fresh generator and re-synthesize.
        self._finished_keys: set[str] = set()
        # Per-row "last chunk" flags of the most recent forward() batch.
        self._ar_last_chunk_flags: list[bool] = []

    # ------------------------------------------------------------------
    # Weights and voices
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        components = _flatten_checkpoint(weights)
        missing = [c for c in _COMPONENTS if c not in components]
        if missing:
            raise ValueError(f"Kokoro checkpoint is missing components: {missing}")
        for name in _COMPONENTS:
            result = getattr(self.network, name).load_state_dict(
                fold_weight_norm(components[name], self._device), strict=False
            )
            bad = [k for k in result.missing_keys if not KokoroNetwork.is_adain_affine_param(k)]
            if bad or result.unexpected_keys:
                raise ValueError(
                    f"Kokoro {name} weights do not match the network: "
                    f"missing={bad[:5]} unexpected={result.unexpected_keys[:5]}"
                )
        self.network.reset_adain_affine()
        # Build the English G2P now: it loads spaCy (downloading its small
        # English model on first use), which would otherwise stall the first
        # request.
        self.frontend.warmup("a")
        # Load every voice pack now (54 x 510 x 256 fp32, ~28 MB): a lazy load
        # would put hub and disk I/O inside forward(), stalling the whole batch.
        voices_dir = resolve_voices_dir(self.model_path, self.revision)
        self._voicepacks = {
            voice: torch.load(os.path.join(voices_dir, f"{voice}.pt"), map_location="cpu", weights_only=True).to(
                self._device
            )
            for voice in list_voices(voices_dir)
        }
        if not self._voicepacks:
            raise ValueError(f"Kokoro checkpoint has no voices/*.pt packs in {voices_dir}")
        return {f"network.{n}" for n, _ in self.network.named_parameters()}

    def _voicepack(self, voice: str) -> torch.Tensor:
        # A lookup, never a path: offline voice names skip the adapter's check.
        pack = self._voicepacks.get(voice)
        if pack is None:
            raise ValueError(f"Unknown Kokoro voice {voice!r}")
        return pack

    # ------------------------------------------------------------------
    # Per-request chunk generator
    # ------------------------------------------------------------------

    def _create_stream_gen(self, info: dict[str, Any]) -> Iterator[tuple[torch.Tensor | None, bool]]:
        """Yield ``(waveform, is_last)`` once per phoneme chunk of the request.

        A failure yields ``(None, True)``, which ``forward()`` reports as
        ``meta["error"]``. The catches are broad so that one request's G2P or
        synthesis error fails only that request, not the whole batch.
        """
        text = str(_pick(info, "text", "") or "")
        voice = str(_pick(info, "voice", DEFAULT_VOICE))
        speed = float(_pick(info, "speed", 1.0))
        language = _pick(info, "language", None)
        seed = _pick(info, "seed", None)
        try:
            lang_code = resolve_lang_code(voice, str(language) if language else None)
            chunks = self.frontend.phoneme_chunks(text, lang_code)
            pack = self._voicepack(voice)
        except Exception:
            logger.exception("Kokoro failed to prepare request (voice=%r, text=%r)", voice, text[:80])
            yield None, True
            return
        if not chunks:
            yield torch.zeros((0,), dtype=torch.float32), True
            return

        generator = torch.Generator(device=self._device)
        generator.manual_seed(int(seed) if seed is not None else random.randrange(2**63))
        for i, ps in enumerate(chunks):
            input_ids = torch.tensor([phonemes_to_ids(ps, self.config.vocab)], dtype=torch.long, device=self._device)
            try:
                # The style vector is picked by phoneme count; the frontend caps
                # chunks at MAX_PHONEMES, which must not outgrow the pack.
                if not 0 < len(ps) <= pack.shape[0]:
                    raise ValueError(f"chunk has {len(ps)} phonemes; voice packs cover 1-{pack.shape[0]}")
                with torch.inference_mode():
                    audio, _ = self.network(input_ids, pack[len(ps) - 1], speed, generator)
            except Exception:
                logger.exception("Kokoro synthesis failed on chunk %d/%d", i + 1, len(chunks))
                yield None, True
                return
            yield audio.float().reshape(-1).cpu(), i == len(chunks) - 1

    # ------------------------------------------------------------------
    # AR runner interface
    # ------------------------------------------------------------------

    def _dummy_hidden(self, input_ids: torch.Tensor | None) -> torch.Tensor:
        n = 1 if input_ids is None else max(1, input_ids.shape[0])
        return torch.zeros((n, self.config.hidden_size), device=self._device, dtype=torch.float32)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        sr = torch.tensor(self.config.sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)
        hidden = self._dummy_hidden(input_ids)
        infos = runtime_additional_information or [{}]

        outputs: list[torch.Tensor] = []
        last_flags: list[bool] = []
        failed_rows: set[int] = set()
        for info in infos:
            # The engine list-wraps additional_information values; unwrap so
            # the key matches the ids passed to on_requests_finished().
            request_id = _pick(info, "global_request_id", None)
            # Rows without an id (profiling runs, a row whose buffer is already
            # gone) and requests that already yielded their last chunk finish
            # without synthesis.
            if request_id is None or str(request_id) in self._finished_keys:
                outputs.append(empty)
                last_flags.append(True)
                continue
            request_key = str(request_id)
            gen = self._stream_gens.get(request_key)
            if gen is None:
                gen = self._stream_gens[request_key] = self._create_stream_gen(info)
            try:
                chunk, is_last = next(gen)
            except StopIteration:
                chunk, is_last = empty, True
            if chunk is None:
                failed_rows.add(len(outputs))
                chunk = empty
            if is_last:
                self._stream_gens.pop(request_key, None)
                self._finished_keys.add(request_key)
            outputs.append(chunk)
            last_flags.append(is_last)

        self._ar_last_chunk_flags = last_flags
        multimodal_outputs: dict[str, Any] = {"model_outputs": outputs, "sr": [sr] * len(outputs)}
        if failed_rows:
            # Only on steps where a request failed; 1 marks the failed rows.
            multimodal_outputs["meta"] = {
                "error": [torch.tensor([int(row in failed_rows)], dtype=torch.int32) for row in range(len(outputs))]
            }
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=multimodal_outputs)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Release per-request state; also drops generators of aborted requests."""
        for req_id in finished_req_ids:
            self._finished_keys.discard(str(req_id))
            gen = self._stream_gens.pop(str(req_id), None)
            if gen is not None:
                gen.close()

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        """Emit the stop token for rows that yielded their last chunk, continue otherwise."""
        num_rows = hidden_states.shape[0]
        logits = torch.full(
            (num_rows, self.config.vocab_size), -1.0e9, dtype=torch.float32, device=hidden_states.device
        )
        flags = self._ar_last_chunk_flags
        for row in range(num_rows):
            # A row missing from the last forward() batch is finished rather
            # than stranded.
            is_last = flags[row] if row < len(flags) else True
            logits[row, KOKORO_STOP_TOKEN_ID if is_last else KOKORO_CONTINUE_TOKEN_ID] = 1.0e6
        return logits

    def embed_input_ids(self, input_ids: torch.Tensor, multimodal_embeddings=None, is_multimodal=None) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], self.config.hidden_size), device=input_ids.device, dtype=torch.float32)
