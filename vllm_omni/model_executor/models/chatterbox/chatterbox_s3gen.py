"""Chatterbox S3Gen vocoder for vLLM-Omni.

S3Gen converts discrete speech tokens from T3 into 24 kHz waveforms via:
1. S3Tokenizer embedding lookup
2. UpsampleConformerEncoder (tokens → hidden)
3. CausalConditionalCFM flow matching (hidden → mel, 2 ODE steps for Turbo)
4. HiFTGenerator vocoder (mel → waveform)

This is the Stage-1 generation model, following the same pattern as
``Qwen3TTSCode2Wav`` — it consumes raw input token IDs, runs inference in a
single forward pass, and returns waveform audio via ``OmniOutput``.

All S3Gen sub-components (tokenizer, conformer, CFM, HiFi-GAN, CAM++ speaker
encoder) are loaded directly from safetensors checkpoints in ``_ensure_model_loaded``.

Fixes over PR #1517:
- Uses ``meanflow=True`` for Turbo variant with ``s3gen_meanflow.safetensors``
- Filters EOS/SOS/OOV tokens (>= 6561) before vocoder
- Passes ``ref_dict`` (speaker embedding) from T3 stage via runtime info
- Appends silence tokens (S3GEN_SIL=4299) to reduce truncation artifacts
- Uses correct ``inference()`` API with ``ref_dict`` and ``n_cfm_timesteps``
- **Batched S3Gen inference** (Phase 3): pads concurrent requests to a
  single ``(B, T_max)`` tensor and issues one S3Gen forward per step.
  Breaks the serial bottleneck seen at concurrency >= 16 in Phase 1.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

S3GEN_SR = 24000
S3_SR = 16000
SPEECH_VOCAB_SIZE = 6561  # Valid speech tokens are [0, 6561); SOS/EOS and padding are >= this
S3GEN_SIL = 4299  # Silence token

# Upper bounds so the per-instance caches can't grow without limit on a
# long-running server (distinct voices in _ref_dict_cache; in-flight requests
# in _chunk_tails, whose entries would otherwise leak when a request aborts
# mid-stream and never hits the is_last cleanup path).
_REF_DICT_CACHE_MAX = 16
_CHUNK_TAILS_MAX = 512


class ChatterboxS3Gen(nn.Module):
    """Stage-1 vocoder for Chatterbox TTS (GenerationModelRunner).

    Consumes speech tokens (from T3) and optional speaker/reference conditioning,
    then decodes waveform audio at 24 kHz.
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.requires_raw_input_tokens = True

        # Determine variant from config: Turbo uses meanflow, Original does not.
        hf_config = vllm_config.model_config.hf_config
        self._meanflow = getattr(hf_config, "s3gen_meanflow", True)
        self._n_cfm_timesteps = 2 if self._meanflow else 10

        self._s3gen_model = None
        # Speaker conditioning cached by reference-audio path (plus the
        # "__default__" conds.pt entry). Keying by path keeps each distinct
        # voice isolated so one request can't leak its speaker state into
        # another's output.
        self._ref_dict_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._logged_stats = False

        # Per-request tail buffer for overlap-add crossfade between chunks.
        # Each chunk holds back its last _crossfade_samples samples; when the
        # next chunk for the same request arrives we mix the buffered tail
        # (fade-out) with the new chunk's head (fade-in) to smooth the CFM
        # re-noise discontinuity at chunk boundaries.
        self._crossfade_ms = 20
        self._crossfade_samples = (S3GEN_SR * self._crossfade_ms) // 1000  # 480
        self._chunk_tails: OrderedDict[str, np.ndarray] = OrderedDict()

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def _ensure_model_loaded(self):
        """Lazy-load the full S3Gen pipeline from checkpoint."""
        if self._s3gen_model is not None:
            return self._s3gen_model

        device = self.vllm_config.device_config.device

        # Use the vendored s3gen_core package (Phase 3) which has the
        # batch-safety fixes applied. The upstream ``chatterbox`` package is
        # still required for the ``s3tokenizer`` cold path (embed_ref()).
        try:
            from .s3gen_core import S3Token2Wav
        except ImportError as e:
            raise ImportError(
                "s3gen_core vendored package failed to import. "
                "Also ensure chatterbox-tts is installed for s3tokenizer: "
                "pip install chatterbox-tts"
            ) from e

        model = S3Token2Wav(meanflow=self._meanflow)

        # Load weights from the correct safetensors file.
        weight_filename = "s3gen_meanflow.safetensors" if self._meanflow else "s3gen.safetensors"
        weight_path = cached_file(self.model_path, weight_filename)
        if weight_path is None:
            raise FileNotFoundError(f"{weight_filename} not found in model checkpoint at {self.model_path}")
        weights = load_file(weight_path)
        model.load_state_dict(weights, strict=False)
        model.to(device).eval()

        self._s3gen_model = model
        logger.info(
            "S3Gen loaded: meanflow=%s, weights=%s, device=%s",
            self._meanflow,
            weight_filename,
            device,
        )
        return self._s3gen_model

    def _build_ref_dict_from_audio(self, ref_audio_path: str, device: torch.device) -> dict[str, Any]:
        """Build ref_dict from reference audio file using S3Gen's embed_ref."""
        import soundfile as sf

        model = self._ensure_model_loaded()

        data, sr = sf.read(ref_audio_path, dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim == 2:
            wav = wav.mean(dim=1)  # stereo → mono
        # embed_ref expects (B, T) at any sample rate — it resamples internally.
        ref_dict = model.embed_ref(wav, int(sr), device=device)
        logger.info("Built ref_dict from audio: %s (sr=%d, keys=%s)", ref_audio_path, sr, list(ref_dict.keys()))
        return ref_dict

    def _load_default_ref_dict(self, device: torch.device) -> dict[str, Any]:
        """Load default speaker conditioning from conds.pt shipped with the model.

        Chatterbox always requires reference audio conditioning.  When the user
        does not supply ``--ref-audio``, we fall back to the pre-computed
        conditioning stored in ``conds.pt`` (key ``"gen"``).
        """
        conds_path = cached_file(self.model_path, "conds.pt")
        if conds_path is None:
            raise FileNotFoundError(
                "conds.pt not found in model checkpoint.  Chatterbox always requires "
                "reference audio conditioning.  Either supply --ref-audio or ensure "
                "conds.pt is present in the model directory."
            )
        # Prefer the safe loader; conds.pt is shipped inside the (trusted) model
        # checkpoint, but some builds store non-tensor objects under "gen" that
        # require a full unpickle, so fall back rather than hard-fail.
        try:
            conds = torch.load(conds_path, map_location=device, weights_only=True)
        except Exception as exc:
            logger.warning(
                "conds.pt could not be loaded with weights_only=True (%s); "
                "falling back to a full unpickle of the trusted model file.",
                exc,
            )
            conds = torch.load(conds_path, map_location=device, weights_only=False)
        ref_dict = conds.get("gen", {})
        # Move all tensors to the correct device.
        ref_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in ref_dict.items()}
        logger.info("Loaded default speaker conditioning from conds.pt")
        return ref_dict

    @staticmethod
    def _truthy_flag(value: Any) -> bool:
        """Normalize a finished-style flag (bool / tensor / list) to bool."""
        if isinstance(value, torch.Tensor):
            return bool(value.reshape(-1)[0].item()) if value.numel() > 0 else False
        if isinstance(value, list):
            return bool(value[0]) if value else False
        return bool(value)

    def _get_ref_dict(self, kwargs: dict[str, Any], device: torch.device) -> dict[str, Any] | None:
        """Resolve speaker conditioning (ref_dict) for the current batch.

        ref_dicts are cached by reference-audio path so ``embed_ref()`` runs
        once per voice (not per chunk). A request that supplies no reference
        audio resolves to the default ``conds.pt`` conditioning — never to
        another request's cached voice, which previously contaminated
        default-voice output once any cloning request had run.

        NOTE: a single batched forward still applies one ref_dict to every
        row. When a batch is not single-voice — more than one distinct path,
        OR a mix of cloning and default-voice rows — the other rows get the
        wrong speaker; a warning is logged. True per-row multi-voice batching
        is tracked as a follow-up.
        """
        runtime_info_list = kwargs.get("runtime_additional_information") or []
        n_rows = 0
        paths: list[str] = []
        for info in runtime_info_list:
            if not isinstance(info, dict):
                continue
            n_rows += 1
            # The stage processor ships the reference-audio path on the
            # payload's ``speaker`` field (OmniPayloadStruct forbids ad-hoc
            # keys); accept the legacy flat key too.
            ref_audio_path = info.get("ref_audio_path") or info.get("speaker")
            if ref_audio_path:
                paths.append(str(ref_audio_path))

        distinct = list(dict.fromkeys(paths))
        # The batch is single-voice only if every row resolves to the same
        # voice: all rows carry the same one path, or none carry a path (all
        # default). Any other mix — >1 distinct path, or some-cloned/some-
        # default — would broadcast one voice onto rows that wanted another.
        non_uniform = len(distinct) > 1 or (distinct and len(paths) < n_rows)
        if non_uniform:
            logger.warning(
                "S3Gen batch is not single-voice (%d rows, %d with ref_audio, %d "
                "distinct path(s)); applying '%s' to the whole batch — other rows "
                "will get the wrong voice (per-row multi-voice batching is not yet "
                "supported).",
                n_rows,
                len(paths),
                len(distinct),
                distinct[0] if distinct else "__default__",
            )

        # Pick this batch's voice from its own requests — fall back to the
        # default conditioning when none of them supplied reference audio.
        key = distinct[0] if distinct else "__default__"
        cached = self._ref_dict_cache.get(key)
        if cached is None:
            cached = self._build_ref_dict_from_audio(key, device) if distinct else self._load_default_ref_dict(device)
            self._ref_dict_cache[key] = cached
            # Bound the cache: evict the least-recently-used voice, but never
            # the reusable default entry.
            while len(self._ref_dict_cache) > _REF_DICT_CACHE_MAX:
                oldest = next(iter(self._ref_dict_cache))
                if oldest == "__default__":
                    self._ref_dict_cache.move_to_end(oldest)
                    continue
                self._ref_dict_cache.pop(oldest, None)
        else:
            self._ref_dict_cache.move_to_end(key)
        return cached

    # Warm-up silence tokens prepended to the first chunk of each utterance.
    # S3Gen's causal UpsampleConformerEncoder + CausalConditionalCFM need
    # ~20 mel frames (~10 speech tokens) of context before the causal
    # attention stack has accumulated enough history to produce clean output.
    # With only 3 tokens of warm-up the first 2-3 spoken words came out
    # mumbled. 10 tokens = 400 ms of prepended silence; trim_fade eats 40 ms
    # of that, leaving ~20 mel frames of warm-up for the decoder. The
    # prepended silence is trimmed from the final waveform (see Phase 4's
    # ``n_prepend * samples_per_token`` cut).
    N_WARMUP_SILENCE_TOKENS = 10

    @staticmethod
    def _filter_tokens(
        speech_tokens: torch.Tensor,
        is_first: bool,
        is_last: bool,
    ) -> tuple[torch.Tensor, int]:
        """Filter out EOS/SOS/OOV tokens; pad only at utterance edges.

        Chatterbox S3Gen's ``inference()`` zero-fades the first 40 ms of its
        output (``trim_fade``) to mask reference-audio spillover. That fade
        must land on silence, not speech — and beyond the fade the causal
        decoder needs additional history before it produces faithful speech.
        So on the first chunk of a request we prepend ``N_WARMUP_SILENCE_TOKENS``
        silence tokens; the prepended samples are cut from the output later.
        Prepending on every chunk would sprinkle warm-up silences through
        the middle of the stream.

        Similarly, an abrupt cut at the very end produces a click; we append
        3 silence tokens only on the final chunk (``is_last``). Intermediate
        chunks are handled by the next chunk's left-context overlap.

        Returns the padded token tensor and the number of silence tokens
        prepended (so the caller can adjust its ctx-overlap trim).
        """
        valid = speech_tokens[speech_tokens < SPEECH_VOCAB_SIZE]
        n_prepend = ChatterboxS3Gen.N_WARMUP_SILENCE_TOKENS if is_first else 0
        pieces: list[torch.Tensor] = []
        if n_prepend:
            pieces.append(
                torch.full(
                    (n_prepend,),
                    S3GEN_SIL,
                    dtype=valid.dtype,
                    device=valid.device,
                )
            )
        pieces.append(valid)
        if is_last:
            pieces.append(
                torch.full(
                    (3,),
                    S3GEN_SIL,
                    dtype=valid.dtype,
                    device=valid.device,
                )
            )
        return torch.cat(pieces) if pieces else valid, n_prepend

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        return None

    def _split_request_ids(self, ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        """Split concatenated input_ids into per-request segments."""
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

    @staticmethod
    def _expand_ref_dict(ref_dict: dict[str, Any], batch_size: int) -> dict[str, Any]:
        """Broadcast a single-speaker ``ref_dict`` to ``batch_size`` entries.

        Chatterbox conditions on a single speaker per request in this
        integration — all concurrent requests in one batch share the same
        cached speaker. For multi-speaker batching (future work) this would
        be swapped for indexed assembly keyed by ``request_id``.
        """
        if batch_size == 1:
            return ref_dict
        expanded: dict[str, Any] = {}
        for k, v in ref_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                if v.size(0) == batch_size:
                    expanded[k] = v
                elif v.size(0) == 1:
                    expanded[k] = v.expand(batch_size, *v.shape[1:]).contiguous()
                else:
                    # Unexpected — keep as-is and let downstream fail loudly.
                    expanded[k] = v
            else:
                expanded[k] = v
        return expanded

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode speech tokens into audio waveform (batched across requests).

        input_ids layout per request: [ctx_frames, *speech_tokens]. All
        concurrent requests are padded to the longest length and submitted
        in a single ``model.inference()`` call; per-request trim and
        overlap-add crossfade run after on the CPU slice.
        """
        model = self._ensure_model_loaded()
        device = self._module_device(model)
        sr_tensor = torch.tensor(S3GEN_SR, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))
        n_reqs = len(request_ids_list)

        # Get speaker conditioning (ref_dict) from runtime info.
        ref_dict = self._get_ref_dict(kwargs, device)

        # Per-request finished flag + request_id from runtime info.
        runtime_info_list = kwargs.get("runtime_additional_information") or []
        finished_flags: list[bool] = []
        request_ids: list[str | None] = []
        for info in runtime_info_list:
            if not isinstance(info, dict):
                finished_flags.append(False)
                request_ids.append(None)
                continue
            # The chunk adapter strips ``meta.finished`` when merging payloads
            # into request info, so the last-chunk signal arrives as
            # ``meta.is_segment_finished``; accept the legacy flat key too.
            fin = info.get("finished", False)
            if not self._truthy_flag(fin):
                meta = info.get("meta")
                fin = meta.get("is_segment_finished", False) if isinstance(meta, dict) else False
            finished_flags.append(self._truthy_flag(fin))
            rid = info.get("request_id")
            request_ids.append(str(rid) if rid is not None else None)
        while len(finished_flags) < n_reqs:
            finished_flags.append(False)
        while len(request_ids) < n_reqs:
            request_ids.append(None)

        # Pre-allocate output slots so per-request skips don't shift indices.
        audios: list[torch.Tensor] = [empty] * n_reqs
        srs: list[torch.Tensor] = [sr_tensor] * n_reqs

        # Phase 1: collect jobs (no model work yet).
        # Each job: (req_idx, tokens_1d, ctx_frames, is_first, is_last, n_prepend, rid)
        jobs: list[tuple[int, torch.Tensor, int, bool, bool, int, str | None]] = []
        for req_idx, req_ids in enumerate(request_ids_list):
            if req_ids.numel() < 2:
                continue
            ctx_frames = int(req_ids[0].item())
            speech_tokens = req_ids[1:]
            if speech_tokens.numel() == 0:
                continue
            is_first = ctx_frames == 0
            is_last = finished_flags[req_idx]
            speech_tokens, n_prepend = self._filter_tokens(
                speech_tokens,
                is_first=is_first,
                is_last=is_last,
            )
            if speech_tokens.numel() == 0:
                continue
            jobs.append(
                (
                    req_idx,
                    speech_tokens,
                    ctx_frames,
                    is_first,
                    is_last,
                    n_prepend,
                    request_ids[req_idx],
                )
            )

        if not jobs:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": audios, "sr": srs},
            )

        # Phase 2: pad tokens to (B, T_max).
        B = len(jobs)
        T_max = max(tokens.numel() for _, tokens, *_ in jobs)
        tokens_batch = torch.full(
            (B, T_max),
            S3GEN_SIL,
            dtype=torch.long,
            device=device,
        )
        lens = torch.empty(B, dtype=torch.long, device=device)
        for i, (_, tokens, *_) in enumerate(jobs):
            tokens_batch[i, : tokens.numel()] = tokens.to(device=device)
            lens[i] = tokens.numel()

        if not self._logged_stats:
            self._logged_stats = True
            logger.info(
                "S3Gen batched forward: B=%d T_max=%d meanflow=%s ref_dict=%s",
                B,
                T_max,
                self._meanflow,
                ref_dict is not None,
            )

        # Phase 3: single batched S3Gen call.
        try:
            batched_ref = self._expand_ref_dict(ref_dict, B) if ref_dict else ref_dict
            wav_batch, _ = model.inference(
                speech_tokens=tokens_batch,
                speech_token_lens=lens,
                ref_dict=batched_ref,
                n_cfm_timesteps=self._n_cfm_timesteps,
            )
        except Exception:
            # Surface vocoder failures instead of returning success-shaped empty
            # audio — silently emitting silence hides real CFM/HiFi-GAN errors
            # (OOM, shape bugs) from the caller. Fail loudly.
            logger.error("S3Gen batched inference failed (B=%d)", B, exc_info=True)
            raise

        # wav_batch is (B, T_audio) on GPU. Move once, slice per-request.
        if isinstance(wav_batch, torch.Tensor):
            wav_batch_np = wav_batch.float().detach().cpu().numpy()
        else:
            wav_batch_np = np.asarray(wav_batch, dtype=np.float32)
        if wav_batch_np.ndim == 1:
            wav_batch_np = wav_batch_np.reshape(1, -1)

        samples_per_token = S3GEN_SR // 25  # 960
        K = self._crossfade_samples

        # Phase 4: per-request trim + crossfade into pre-allocated slots.
        for i, (req_idx, tokens, ctx_frames, is_first, is_last, n_prepend, rid) in enumerate(jobs):
            # The valid audio for sample i corresponds to lens[i] tokens ×
            # samples_per_token. Any positions past that are vocoded padding
            # and must be discarded.
            valid_audio = int(tokens.numel()) * samples_per_token
            wav_np = wav_batch_np[i, :valid_audio]

            # Trim left-context overlap plus any silence we prepended.
            use_crossfade = rid is not None and K > 0 and ctx_frames > 0
            overlap = K if use_crossfade else 0
            cut = (ctx_frames + n_prepend) * samples_per_token - overlap
            if cut > 0:
                if cut < wav_np.shape[0]:
                    wav_np = wav_np[cut:]
                else:
                    logger.warning(
                        "Context trim %d >= decoded length %d; returning empty.",
                        cut,
                        wav_np.shape[0],
                    )
                    continue

            # Overlap-add crossfade with held-back tail from prior chunk.
            emit = wav_np
            if rid is not None and K > 0:
                prev_tail = self._chunk_tails.pop(rid, None)
                if prev_tail is not None and emit.shape[0] >= K:
                    n = min(K, prev_tail.shape[0])
                    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
                    fade_out = 1.0 - fade_in
                    mixed = (prev_tail[-n:] * fade_out + emit[:n] * fade_in).astype(np.float32)
                    emit = np.concatenate([mixed, emit[n:]])

                if not is_last and emit.shape[0] > K:
                    self._chunk_tails[rid] = emit[-K:].copy()
                    emit = emit[:-K]
                    # Bound the buffer: tails from requests that abort mid-stream
                    # never reach the is_last cleanup below, so evict oldest.
                    while len(self._chunk_tails) > _CHUNK_TAILS_MAX:
                        self._chunk_tails.popitem(last=False)
                elif is_last:
                    self._chunk_tails.pop(rid, None)

            if emit.shape[0] > 0:
                audios[req_idx] = torch.from_numpy(np.ascontiguousarray(emit)).to(dtype=torch.float32)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(f"ChatterboxS3Gen expected (audio_tensor, sr) outputs, got {type(model_outputs)}")

        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audio_tensor, "sr": sr},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # All S3Gen weights are loaded lazily from checkpoint files.
        return set()
