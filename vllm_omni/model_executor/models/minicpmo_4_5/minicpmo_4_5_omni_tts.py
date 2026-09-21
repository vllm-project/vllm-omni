# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
"""MiniCPM-o 4.5 native autoregressive Talker.

Pipeline:
  1. Receive thinker hidden_states + full token IDs via additional_information
  2. Extract tts_bos..tts_eos region
  3. Build condition: emb_text(tokens) + projector_semantic(hidden) (hidden_text_merge)
  4. Project last hidden through head_code; vLLM Sampler picks the codec id
  5. Next decode embeds that id with emb_code and emits it to Code2Wav
"""

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

try:
    import torch_npu  # noqa: F401  (NPU platform guarantee)
except Exception:  # pragma: no cover
    torch_npu = None
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.sampler import Sampler

from vllm_omni.engine.duplex.intermediate import get_tts_handoff
from vllm_omni.model_executor.models.minicpmo_4_5 import (
    MINICPMO45_DUPLEX_CODEC_TOKENS_PER_CHUNK,
    MINICPMO45_DUPLEX_TURN_END_CODEC_TOKENS,
)
from vllm_omni.model_executor.models.minicpmo_4_5.talker_codec_sample import (
    TalkerCodecDeviceState,
    TalkerCodecSampleResult,
    codec_sample_result,
    greedy_codec_sample,
    make_device_state,
    prepare_codec_logits,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.step_prof import span as _prof_span

logger = init_logger(__name__)

# Codec-sampling fallbacks the multi-frame path reads directly. The deploy YAML
# and the checkpoint's tts_config override them per request; these match what
# the single-frame path falls back to.
_REPETITION_WINDOW = 16
_MIN_AUDIO_TOKENS = 64
_CODEC_SEED = 42
_CODEC_TEMPERATURE = 0.8
_CODEC_TOP_K = 25
_CODEC_TOP_P = 0.85
_CODEC_REPETITION_PENALTY = 1.05
_CODEC_MIN_TOKENS = 50
_CODEC_MAX_TOKENS = 2048
_FAST_CODEC_PENALTY = os.getenv("MINICPMO_FAST_CODEC_PENALTY", "1") == "1"
# YAML key -> (tts_config attribute, hardcoded fallback, type)
_CODEC_SAMPLING_SOURCES: tuple[tuple[str, str, Any, Any], ...] = (
    ("seed", "seed", _CODEC_SEED, int),
    ("temperature", "temperature", _CODEC_TEMPERATURE, float),
    ("top_k", "top_k", _CODEC_TOP_K, int),
    ("top_p", "top_p", _CODEC_TOP_P, float),
    ("repetition_penalty", "repetition_penalty", _CODEC_REPETITION_PENALTY, float),
    ("min_tokens", "min_new_tokens", _CODEC_MIN_TOKENS, int),
    ("max_tokens", "max_new_tokens", _CODEC_MAX_TOKENS, int),
)


_REPETITION_PENALTY_CHUNK_SIZE = 16
# ``past_window`` of MiniCPMTTS's codec repetition penalty: both generate() and
# generate_chunk() build it through gen_logits(), which hardcodes
# CustomRepetitionPenaltyLogitsProcessorRepeat(penalty, num_code, 16).
_CODEC_PENALTY_WINDOW = 16
# MiniCPMTTS.generate's max_new_token. The Talker context bounds this further;
# without it a request that never samples codec EOS keeps emitting frames for
# twice as long as upstream would, which is audible as a long silent tail.
_OFFLINE_CODEC_MAX_NEW_TOKENS = 2048
# Native duplex Talker must finish after one MiniCPMTTS.generate_chunk:
# 25 codec frames (``codec_chunk_frames``) plus the terminating sample.
# Without this, the single-vocab Sampler keeps the stage-1 request alive
# until codec EOS / 4096 and Thinker never starts the next model turn.
_DUPLEX_CODEC_TOKENS_PER_CHUNK = MINICPMO45_DUPLEX_CODEC_TOKENS_PER_CHUNK
_DUPLEX_TURN_END_CODEC_TOKENS = MINICPMO45_DUPLEX_TURN_END_CODEC_TOKENS
#: Frames the Talker forwards per generate_chunk before its cadence EOS.
_DUPLEX_CODEC_FRAMES_PER_CHUNK = MINICPMO45_DUPLEX_CODEC_TOKENS_PER_CHUNK - 1
#: On a turn-end chunk the Talker's EOS is masked for this many steps after
#: each 25-frame boundary: the model emits a cadence EOS there whether or not
#: its text is spoken, while a genuine end of text shows up as an EOS anywhere
#: else in the window.
_DUPLEX_TURN_END_BOUNDARY_MASK_STEPS = 5


def _native_duplex_chunk_budget(meta: Mapping[str, Any] | None) -> tuple[int, int]:
    """Return ``(max_tokens, min_tokens)`` for one native-duplex Talker request."""
    turn_start = isinstance(meta, Mapping) and bool(meta.get("turn_start"))
    turn_end = isinstance(meta, Mapping) and bool(meta.get("turn_end"))
    if turn_end:
        # The turn-end chunk drains the text the Talker still owes: no floor
        # (an early EOS means the text is spoken) and a multi-unit ceiling.
        return _DUPLEX_TURN_END_CODEC_TOKENS, 0
    ceiling = _DUPLEX_CODEC_TOKENS_PER_CHUNK
    return ceiling, 0 if turn_start else ceiling


def _turn_end_boundary_eos_masked(step: int) -> bool:
    """Whether a turn-end chunk masks codec EOS at ``step`` forwarded frames.

    The Talker emits a cadence EOS after every 25 frames regardless of the text
    left, so a turn-end chunk ignores EOS in a short window after each boundary
    and lets the model continue; EOS elsewhere ends the chunk as usual.
    """
    if step < _DUPLEX_CODEC_FRAMES_PER_CHUNK:
        return False
    return step % _DUPLEX_CODEC_FRAMES_PER_CHUNK < _DUPLEX_TURN_END_BOUNDARY_MASK_STEPS


def blank_scheduler_prompt_for_penalties(
    prompt_token_ids: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Return a penalty prompt whose every position is the pad id (``vocab_size``).

    No Talker prompt position is codec history: prefill conditioning arrives as
    embeddings from ``preprocess`` and decode embeds sampled ids with
    ``emb_code``. The scheduler ids are placeholders (``llm2tts`` fills them
    with ``0``, or with thinker token ids on the non-handoff path), so scoring
    them would tax unrelated codec tokens.
    """
    return torch.full_like(prompt_token_ids, int(vocab_size))


def _restore_weight_norm_weight(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """Materialize ``weight_norm(..., dim=0)`` checkpoint parameters."""
    return torch._weight_norm(weight_v, weight_g, dim=0)


def _apply_batched_repetition_penalty(
    logits: torch.Tensor,
    histories: Sequence[torch.Tensor],
    *,
    penalty: float | torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Apply request-local frequency penalties to a batch of codec logits.

    ``penalty`` may be a scalar or one value per row, mirroring upstream's
    per-request ``sampling_params.repetition_penalty``.
    """
    if logits.ndim != 2:
        raise ValueError(f"batched codec logits must be 2D, got shape {tuple(logits.shape)}")
    batch_size, vocab_size = logits.shape
    if len(histories) != batch_size:
        raise ValueError(f"expected {batch_size} codec histories, got {len(histories)}")
    if batch_size == 0:
        return logits

    penalties = torch.as_tensor(penalty, device=logits.device, dtype=logits.dtype).reshape(-1)
    if penalties.numel() == 1:
        penalties = penalties.expand(batch_size)
    elif penalties.numel() != batch_size:
        raise ValueError(f"expected 1 or {batch_size} codec repetition penalties, got {penalties.numel()}")
    if not bool((penalties != 1.0).any()):
        return logits

    penalized = logits.clone()
    for start in range(0, batch_size, _REPETITION_PENALTY_CHUNK_SIZE):
        end = min(start + _REPETITION_PENALTY_CHUNK_SIZE, batch_size)
        chunk_logits = logits[start:end]
        encoded_rows: list[torch.Tensor] = []
        for local_row, history in enumerate(histories[start:end]):
            recent = history.reshape(-1)[-window_size:].to(device=logits.device, dtype=torch.long)
            if recent.numel() > 0:
                encoded_rows.append(recent + local_row * vocab_size)
        if not encoded_rows:
            continue

        # Bound the int64 bincount workspace independently of request concurrency.
        encoded = encoded_rows[0] if len(encoded_rows) == 1 else torch.cat(encoded_rows)
        frequencies = torch.bincount(
            encoded,
            minlength=(end - start) * vocab_size,
        ).reshape(end - start, vocab_size)
        alpha = torch.pow(penalties[start:end].unsqueeze(1), frequencies.to(dtype=logits.dtype))
        penalized[start:end] = torch.where(chunk_logits < 0, chunk_logits * alpha, chunk_logits / alpha)

    return penalized


def _apply_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
    min_tokens_to_keep: int = 3,
    inplace: bool = False,
) -> torch.Tensor:
    """Reference warper: same candidate floors as the upstream warpers."""
    filtered = logits if inplace else logits.clone()
    vocab_size = filtered.shape[-1]
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -min_tokens_to_keep:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        filtered.masked_fill_(remove, float("-inf"))
    if top_k is not None and top_k > 0:
        keep = min(vocab_size, max(int(top_k), min_tokens_to_keep))
        threshold = torch.topk(filtered, keep, dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, float("-inf"))
    return filtered


_NPU_TOPK_CACHE: dict = {}
_NPU_TOP_K_TOP_P = os.environ.get("MINICPMO_TTS_NPU_TOPK_TOPP", "1") != "0"
# npu_top_k_top_p wins on launches at small batch but loses to the two-op
# torch path at large batch (128/8 A/B: 1.391 gated-off vs on).
_NPU_TOPK_MAX_BATCH = int(os.environ.get("MINICPMO_TTS_NPU_TOPK_TOPP_MAX_BATCH", "64"))


def _npu_top_k_top_p_warp(logits, *, top_k, top_p):
    """Fused top-k/top-p floor via torch_npu.npu_top_k_top_p.

    Kernel semantics: keep top-k by value, then keep the top-p probability
    mass over the retained set. Falls back to the exact PyTorch warper when
    the kernel is unavailable or fails.
    """
    if top_k is None or top_p is None or not 0.0 < top_p < 1.0:
        return logits
    npu = getattr(torch_npu, "npu_top_k_top_p", None)
    if npu is None:
        return logits
    key = (str(logits.device), str(logits.dtype), float(top_p), int(top_k))
    cached = _NPU_TOPK_CACHE.get(key)
    if cached is None:
        dev = torch.full((1,), float(top_p), device=logits.device, dtype=logits.dtype)
        dk = torch.full((1,), int(top_k), device=logits.device, dtype=torch.int32)
        _NPU_TOPK_CACHE[key] = (dev, dk)
    else:
        dev, dk = cached
    try:
        return npu(
            logits,
            dev.expand(logits.shape[0]).contiguous(),
            dk.expand(logits.shape[0]).contiguous(),
        )
    except Exception:
        return _apply_top_k_top_p(logits, top_k=top_k, top_p=top_p, min_tokens_to_keep=3, inplace=True)


def _maybe_prewarp_top_k_top_p(logits, sampling_metadata):
    """Pre-apply the fused floor and neutralize the sampler's own warpers.

    Only engages when every row shares one (top_k, top_p) at temperature 1.0,
    where the floor on raw logits is exactly the floor the sampler would
    compute itself (top-k is rank based, so temperature invariance holds and
    top-p mass matches at T=1). Any structural surprise returns None and the
    native sampler path runs untouched.
    """
    if not _NPU_TOP_K_TOP_P:
        return None
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] == 0:
        return None
    if logits.shape[0] > _NPU_TOPK_MAX_BATCH:
        return None
    if getattr(torch_npu, "npu_top_k_top_p", None) is None:
        return None
    try:
        params = getattr(sampling_metadata, "sampling_params", None)
        if not isinstance(params, (list, tuple)) or len(params) != logits.shape[0]:
            return None
        top_k = top_p = None
        for sp in params:
            k = getattr(sp, "top_k", None)
            pp = getattr(sp, "top_p", None)
            t = getattr(sp, "temperature", None)
            if t is None or float(t) != 1.0:
                return None
            if k is None or int(k) <= 0:
                return None
            if pp is None or not 0.0 < float(pp) < 1.0:
                return None
            if top_k is None:
                top_k, top_p = int(k), float(pp)
            elif int(k) != top_k or float(pp) != top_p:
                return None
        logits = _npu_top_k_top_p_warp(logits, top_k=top_k, top_p=top_p)
        new_params = [replace(sp, top_p=1.0, top_k=-1) for sp in params]
        sampling_metadata = replace(sampling_metadata, sampling_params=list(new_params))
        return logits, sampling_metadata
    except Exception:
        return None


def resolve_codec_sampling_params(
    yaml_params: Mapping[str, Any] | None,
    tts_config: Any | None = None,
) -> dict[str, Any]:
    """Resolve Talker codec knobs: deploy YAML, then checkpoint, then defaults.

    The deploy YAML that starts a ranked run comes from the organizer's
    baseline branch and carries no ``codec_sampling_params`` block, so
    requiring one there is a startup failure rather than a configuration
    error: the Talker stage would not come up at all under the official
    launch. Falling back to the checkpoint's ``tts_config`` is what the
    upstream Talker does, and it is what makes this tree drop into the
    official deploy config unchanged.

    A YAML block still wins key by key, so our own configs keep overriding.
    """
    provided = yaml_params if isinstance(yaml_params, Mapping) else {}
    resolved: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key, attribute, fallback, cast in _CODEC_SAMPLING_SOURCES:
        value = provided.get(key)
        source = "yaml"
        if value is None:
            value = getattr(tts_config, attribute, None) if tts_config is not None else None
            source = "tts_config"
        if value is None:
            value, source = fallback, "default"
        resolved[key] = cast(value)
        sources[key] = source
    if resolved["min_tokens"] < 0:
        raise ValueError("codec_sampling_params.min_tokens must be >= 0")
    if resolved["max_tokens"] <= 0:
        raise ValueError("codec_sampling_params.max_tokens must be > 0")
    logger.info("MiniCPM-o Talker codec sampling %s (from %s)", resolved, sources)
    return resolved


# ---------------------------------------------------------------------------
# Per-frame trace for the multi-frame loop (off by default; complements the
# per-step timers in talker_multiframe).
#
#   VLLM_OMNI_MINICPMO_KSTEP_FRAME_TRACE=1  enable
#   prints at most _KSTEP_FRAME_TRACE_FRAMES frames, then goes quiet
#
# Answers what the per-step timers cannot: whether the model's own sampler
# reaches EOS inside the K loop, and whether each frame is fed the previous
# frame's sampled code (frame-order invariant). Diagnostics never take the
# engine down: on any error they disable themselves and log one warning.
# ---------------------------------------------------------------------------
_KSTEP_FRAME_TRACE_ENV = "VLLM_OMNI_MINICPMO_KSTEP_FRAME_TRACE"
_KSTEP_FRAME_TRACE_LINES = 128  # max printed lines
_KSTEP_FRAME_TRACE_HEAD = 16  # first N frames printed one by one (frame-order check)
_KSTEP_FRAME_TRACE_STRIDE = 16  # then one line every N frames (covers ~1800 frames)
_KSTEP_FRAME_TRACE: dict[str, Any] = {"reqs": {}, "lines": 0, "off": False}


def kstep_frame_trace_enabled() -> bool:
    """Off unless the env turns it on; latched off for good after a failure."""
    if _KSTEP_FRAME_TRACE["off"]:
        return False
    raw = os.environ.get(_KSTEP_FRAME_TRACE_ENV, "").strip().lower()
    return bool(raw) and raw not in ("0", "false", "no", "off")


def _trace_kstep_frame(model: Any, **fields: Any) -> None:
    """One line per codec frame while the multi-frame loop owns sampling.

    Logs what the *model* decided: the code each frame took as its input (the
    previous frame's sample, per the contract), what it sampled, whether that
    was the codec EOS, and where the sampler's own step counter stood. The first
    ``HEAD`` frames are logged one per line (that is what verifies the frame
    chain), then one line every ``STRIDE`` frames so the window still covers the
    frames where EOS first becomes eligible -- `min_tokens` masks it before
    then, so a head-only trace can never see a stop. An EOS sample is always
    logged regardless of the stride. Never raises: a failure disables the trace
    for good.
    """
    state = _KSTEP_FRAME_TRACE
    if state["off"]:
        return
    reqs = state["reqs"]
    request_id = str(fields.get("request_id"))
    seen = reqs.get(request_id, 0)
    reqs[request_id] = seen + 1
    is_eos = bool(fields.get("is_eos"))
    interesting = seen < _KSTEP_FRAME_TRACE_HEAD or (
        seen >= _KSTEP_FRAME_TRACE_HEAD and (seen - _KSTEP_FRAME_TRACE_HEAD) % _KSTEP_FRAME_TRACE_STRIDE == 0
    )
    if not is_eos and (not interesting or state["lines"] >= _KSTEP_FRAME_TRACE_LINES):
        return
    try:
        device_states = getattr(model, "_request_codec_device_states", None)
        device_state = device_states.get(fields.get("request_id")) if isinstance(device_states, dict) else None
        device_step = int(device_state.step.reshape(-1)[0].item()) if device_state is not None else -1
        state["lines"] += 1  # cap printed lines only; per-request windows count separately
        logger.info(
            "[kstep-frame] frame=%d req=%s step=%s->%s device_step=%s in_code=%s row=%s "
            "sampled=%s is_eos=%s limit=%s finished=%s ctx=%s min=%s max=%s",
            seen,
            str(fields.get("request_id"))[:12],
            fields.get("step_before"),
            fields.get("step_after"),
            device_step,
            fields.get("prev_code"),
            fields.get("row"),
            fields.get("sampled_id"),
            fields.get("is_eos"),
            fields.get("reached_limit"),
            fields.get("finished"),
            fields.get("ctx_len"),
            fields.get("min_tokens"),
            fields.get("max_tokens"),
        )
    except Exception as exc:  # pragma: no cover - diagnostics never raise
        state["off"] = True
        logger.warning("[kstep-frame] trace disabled after failure: %r", exc)


def _codec_int_param(state: Any, key: str, fallback: int) -> int:
    """Read an integer codec knob from a request state, None meaning unset.

    ``dict.get(key, fallback)`` does not help when the key is present but holds
    None -- which the offline duplex path uses to mean "the deploy YAML owns
    this knob" -- so None is mapped to the fallback explicitly here. Kept in
    one place so every reader agrees on the rule.
    """
    value = state.get(key) if isinstance(state, Mapping) else None
    return int(fallback if value is None else value)


class _MiniCPMTTSProjector(nn.Module):
    """Checkpoint-compatible hidden-state projector used by MiniCPMTTS."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


class MiniCPMO45OmniTTSForConditionalGeneration(nn.Module, SupportsPP):
    """Runner-owned MiniCPM-o 4.5 Talker that emits codec tokens only."""

    requires_request_sample_eligibility = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMOConfig

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self._penalty_histories: list[torch.Tensor] | None = None
        # Per-step codec EOS routing decided in make_omni_output: rows to force
        # to EOS (finished segments / chunk budget hit) and rows to mask EOS on
        # (min_tokens / duplex turn-end drain window). Both are consumed by
        # compute_logits and cleared there; _pending_force_eos_rows carries the
        # force decision through to sample() so vLLM's min_tokens processor
        # cannot blank the EOS we just forced.
        self._force_eos_rows: list[bool] | None = None
        self._mask_eos_rows: list[bool] | None = None
        self._pending_force_eos_rows: list[bool] | None = None
        self._request_audio_states: dict[str, dict[str, Any]] = {}
        # Mirrors upstream TTSStreamingGenerator._chunk_info: one committed
        # condition plus, during a rollover, one immutable recompute recipe.
        self._request_condition_states: dict[str, dict[str, Any]] = {}
        self._deferred_cleanup_ids: set[str] = set()

        tts_config = getattr(config, "tts_config", None)
        if tts_config is None and getattr(config, "model_type", None) == "minicpmtts":
            tts_config = config
        if tts_config is not None:
            self._tts_config = tts_config
            self._tts_bos_id = getattr(tts_config, "audio_bos_token_id", 151687)
            self._text_eos_id = getattr(tts_config, "text_eos_token_id", 151692)
            self._num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            self._codec_eos_id = int(getattr(tts_config, "eos_token_id", self._num_audio_tokens - 1))
            self._hidden_size = getattr(tts_config, "hidden_size", 768)
            self._normalize = getattr(tts_config, "normalize_projected_hidden", True)
            # Codec sampling knobs: the deploy YAML's codec_sampling_params
            # block wins key by key, then the checkpoint's tts_config, then the
            # module fallbacks. The multi-frame path reads the resolved values
            # straight off the model.
            yaml_codec = getattr(getattr(vllm_config, "model_config", None), "codec_sampling_params", None)
            resolved = resolve_codec_sampling_params(yaml_codec, tts_config)
            self._codec_seed = resolved["seed"]
            self._codec_temperature = resolved["temperature"]
            self._codec_top_k = resolved["top_k"]
            self._codec_top_p = resolved["top_p"]
            self._codec_repetition_penalty = resolved["repetition_penalty"]
            self._codec_min_tokens = resolved["min_tokens"]
            self._codec_max_tokens = resolved["max_tokens"]
        else:
            self._tts_config = None
            self._codec_eos_id = 0

        # Multi-frame decode: K codec frames per Talker decode step. A value
        # >= 2 engages the loop -- the scheduler hands each request K query
        # positions through the vLLM V1 speculative path (constant `continue`
        # drafts), talker_multiframe.run() replays the decode graph K times,
        # and this model samples one codec frame per replay in-model (the
        # vLLM-level head degenerates to a one-hot stop/continue row so the
        # rejection sampler verifies without touching the codec stream). K is
        # read back from the injected stage-1 speculative_config.
        self._k_step_frames = self._parse_k_step_frames(vllm_config)
        self.supports_multi_frame_decode = self._k_step_frames > 0
        # Per-request device RNG streams for in-model codec sampling (the
        # multi-frame loop cannot run the vLLM host sampler per frame).
        self._k_step_rngs: dict[str, torch.Generator] = {}
        self._k_step_base_seed = 0x5EEDC0DE
        # Last frame's stop/continue rows and the flattened per-token stop
        # sequence the merged step presents to compute_logits.
        self._k_last_stop_rows: list[bool] | None = None

        self.has_preprocess = True
        self.has_postprocess = False
        # Same-step codes travel through make_omni_output from the previous
        # sampled id (decode preprocess embeds that id via emb_code).
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {("codes", "audio")}
        # K-step (multi-frame Talker) request state. The loop keeps one codec
        # sampling graph, one Generator and one device-side conversation state
        # per in-flight request; _flush_deferred_cleanup drops them when the
        # engine reports the request finished.
        self._batch_stop_logits: torch.Tensor | None = None
        self._request_generators: dict[str, torch.Generator] = {}
        self._request_audio_states: dict[str, dict[str, Any]] = {}
        self._request_codec_device_states: dict[str, TalkerCodecDeviceState] = {}
        self._request_codec_device_inputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        self._init_native_talker(prefix)

    @staticmethod
    def _probe_soc_name() -> str:
        """torch_npu device-name string, or "" when the probe cannot run.

        The integer SoCVersion encoding is not a documented ordering (a local
        910B3 reports 223, which defeats any threshold guess), so the guard
        matches on the name prefix instead -- exactly what torch_npu itself
        does (utils/_module.py lists "Ascend910B" and "Ascend910_93").
        """
        try:
            import torch_npu

            return str(torch_npu.npu.get_device_name(torch_npu.npu.current_device()))
        except Exception:
            return ""

    @classmethod
    def _parse_k_step_frames(cls, vllm_config: Any = None) -> int:
        """K codec frames per Talker step, or 0 when the loop must not engage.

        Read back from the engine-side speculative_config that stage_config.py
        injects for stage 1 (method ``ngram`` with ``num_speculative_tokens ==
        K - 1``). Deriving K from the engine config rather than from a separate
        channel is what keeps the runner and the scheduler from disagreeing: a
        mismatch either arms the loop with no drafts to consume, or leaves it
        off while the scheduler reserves K positions per request.
        """
        from vllm_omni.config.stage_config import _MINICPMO_TALKER_FRAMES_MAX

        model_cfg = getattr(vllm_config, "model_config", None)
        if getattr(model_cfg, "model_stage", None) != "tts":
            return 0
        spec = getattr(vllm_config, "speculative_config", None)
        if isinstance(spec, dict):
            method = spec.get("method")
            num_spec = spec.get("num_speculative_tokens", 0) or 0
        else:
            method = getattr(spec, "method", None)
            num_spec = getattr(spec, "num_speculative_tokens", 0) or 0
        if method != "ngram":
            return 0
        frames = int(num_spec) + 1
        if frames < 2:
            # K=1 degenerates to the ordinary one-frame path; treat it as off
            # so the flag never silently half-arms the pipeline.
            return 0
        if frames > _MINICPMO_TALKER_FRAMES_MAX:
            raise ValueError(f"K-step frame count is capped at {_MINICPMO_TALKER_FRAMES_MAX}, got {frames}")
        logger.info("[minicpmo] Talker multi-frame decode armed: %d codec frames per step", frames)
        return frames

    def _init_native_talker(self, prefix: str) -> None:
        if self._tts_config is None:
            raise ValueError("MiniCPM-o continuous Talker requires tts_config")
        cfg = self._tts_config
        if int(getattr(cfg, "num_vq", 1)) != 1:
            raise ValueError(
                "MiniCPM-o continuous Talker currently requires num_vq=1; "
                f"checkpoint reports {getattr(cfg, 'num_vq', None)}"
            )
        llama_config = LlamaConfig(
            vocab_size=32000,
            hidden_size=int(cfg.hidden_size),
            intermediate_size=int(cfg.intermediate_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            hidden_act=getattr(cfg, "hidden_act", "silu"),
            max_position_embeddings=int(cfg.max_position_embeddings),
            rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
            tie_word_embeddings=False,
        )
        talker_config = self.vllm_config.with_hf_config(llama_config, architectures=["LlamaForCausalLM"])
        talker_config.model_config.hf_text_config = llama_config
        self.tts_model = LlamaModel(
            vllm_config=talker_config,
            prefix=maybe_prefix(prefix, "tts_obj.model"),
        )
        self.emb_text = nn.Embedding(int(cfg.num_text_tokens), int(cfg.hidden_size))
        self.projector_semantic = _MiniCPMTTSProjector(int(cfg.llm_dim), int(cfg.hidden_size))
        self.emb_code = nn.ModuleList(
            [nn.Embedding(int(cfg.num_audio_tokens), int(cfg.hidden_size)) for _ in range(int(cfg.num_vq))]
        )
        self.head_code = nn.ModuleList(
            [nn.Linear(int(cfg.hidden_size), int(cfg.num_audio_tokens), bias=False) for _ in range(int(cfg.num_vq))]
        )
        self.make_empty_intermediate_tensors = self.tts_model.make_empty_intermediate_tensors

    def _boundary_embeddings(self) -> torch.Tensor:
        """Embed the ``<text_eos><audio_bos>`` tail every condition ends with."""
        ids = torch.tensor(
            [self._text_eos_id, self._tts_bos_id],
            device=self.emb_text.weight.device,
            dtype=torch.long,
        )
        return self.emb_text(ids)

    def _build_condition_embeddings(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
        *,
        native_duplex: bool = False,
    ) -> torch.Tensor:
        if tts_token_ids.numel() == 0 or tts_hidden_states.numel() == 0:
            # The thinker can legally emit an empty speech segment (<|tts_bos|>
            # immediately followed by a boundary token) when it decides not to
            # speak. Condition on the boundary tokens alone, which matches the
            # 2-token scheduler prompt the stage bridge builds for an empty
            # handoff.
            return self._boundary_embeddings()
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        token_ids = tts_token_ids.to(device=device, dtype=torch.long).reshape(-1)
        hidden = tts_hidden_states.to(device=device, dtype=dtype)
        if hidden.shape[0] != token_ids.shape[0] and token_ids.shape[0] != 1:
            raise ValueError(
                "MiniCPM-o Talker condition length mismatch: "
                f"token_ids={token_ids.shape[0]} hidden_states={hidden.shape[0]}"
            )
        text_embeds = self.emb_text(token_ids)
        hidden_embeds = self.projector_semantic(hidden)
        if self._normalize:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        audio_bos = self.emb_text(torch.tensor([self._tts_bos_id], device=device, dtype=torch.long))
        condition = text_embeds + hidden_embeds
        if native_duplex:
            # Match MiniCPMTTS.generate_chunk's streaming condition.
            return torch.cat([condition, audio_bos], dim=0)
        return torch.cat([condition, self._boundary_embeddings()], dim=0)

    def _build_streaming_recompute_embeddings(
        self,
        current_condition: torch.Tensor,
        *,
        request_id: str,
        info_dict: Mapping[str, Any],
        meta: Mapping[str, Any],
    ) -> torch.Tensor:
        """Return the official one-previous-chunk sliding-recompute window."""
        condition_seq = meta.get("streaming_condition_seq")
        if not isinstance(condition_seq, int) or isinstance(condition_seq, bool):
            if meta.get("streaming_prompt_recompute") is True:
                raise ValueError("streaming prompt recompute is missing streaming_condition_seq")
            # Direct model tests and non-connector callers do not participate in
            # the persistent async-chunk lifecycle, so they need no window state.
            return current_condition

        turn_start = bool(meta.get("turn_start"))
        recompute = meta.get("streaming_prompt_recompute") is True
        states = self._request_condition_states
        state = states.get(request_id)
        if turn_start:
            if recompute:
                raise ValueError("streaming prompt recompute cannot cross a native duplex turn boundary")
            states[request_id] = {
                "condition_seq": condition_seq,
                "condition": current_condition.detach().clone(),
                "base_recent_codes": (),
            }
            return current_condition

        if state is None:
            if recompute:
                raise ValueError("streaming prompt recompute is missing the previous Talker condition")
            states[request_id] = {
                "condition_seq": condition_seq,
                "condition": current_condition.detach().clone(),
                "base_recent_codes": (),
            }
            return current_condition

        previous_seq = state.get("condition_seq")
        if not isinstance(previous_seq, int) or condition_seq < previous_seq:
            raise ValueError(
                f"stale native duplex Talker condition sequence: current={condition_seq}, previous={previous_seq}"
            )
        if condition_seq > previous_seq + 1:
            raise ValueError(
                f"native duplex Talker skipped a condition sequence: current={condition_seq}, previous={previous_seq}"
            )

        if not recompute:
            if condition_seq > previous_seq:
                attention_type = getattr(self._tts_config, "attention_type", "full_attention")
                if attention_type == "sliding_recompute":
                    raise ValueError(
                        "a native duplex Talker condition advanced without its streaming recompute marker: "
                        f"current={condition_seq}, previous={previous_seq}"
                    )
                audio_state = self._request_audio_states.get(request_id)
                recent_codes = audio_state.get("recent_codes") if isinstance(audio_state, dict) else None
                if isinstance(recent_codes, list):
                    base_recent_codes = tuple(int(code_id) for code_id in recent_codes[-_CODEC_PENALTY_WINDOW:])
                else:
                    base_recent_codes = state.get("base_recent_codes")
                    if not isinstance(base_recent_codes, tuple):
                        raise ValueError("streaming Talker condition lost its frozen codec history")
                states[request_id] = {
                    "condition_seq": condition_seq,
                    "condition": current_condition.detach().clone(),
                    "base_recent_codes": base_recent_codes,
                }
                return current_condition
            if "active_embeddings" in state:
                raise ValueError("an active streaming recompute was replayed without its recompute marker")
            return current_condition

        if condition_seq == previous_seq:
            active_embeddings = state.get("active_embeddings")
            if not isinstance(active_embeddings, torch.Tensor):
                raise ValueError("streaming prompt window lost its cached recompute embeddings")
            return active_embeddings

        if condition_seq != previous_seq + 1:
            raise ValueError(
                "streaming prompt recompute skipped a Talker condition: "
                f"previous={previous_seq}, current={condition_seq}"
            )

        previous_condition = state.get("condition")
        if not isinstance(previous_condition, torch.Tensor):
            raise ValueError("streaming prompt recompute lost the previous Talker condition")

        ids = info_dict.get("ids")
        previous_codes = ids.get("streaming_prompt_previous_codes") if isinstance(ids, Mapping) else None
        if isinstance(previous_codes, torch.Tensor):
            code_ids = previous_codes.to(device=self.emb_code[0].weight.device, dtype=torch.long).reshape(-1)
        elif isinstance(previous_codes, (list, tuple)):
            code_ids = torch.as_tensor(previous_codes, device=self.emb_code[0].weight.device, dtype=torch.long)
        else:
            raise ValueError("streaming prompt recompute is missing confirmed codec ids")
        if code_ids.numel() > _DUPLEX_TURN_END_CODEC_TOKENS - 1:
            raise ValueError(f"streaming prompt recompute has too many codec ids: {code_ids.numel()}")
        if code_ids.numel() and bool(((code_ids < 0) | (code_ids >= self._codec_eos_id)).any()):
            raise ValueError("streaming prompt recompute codec ids include an invalid or terminal token")

        parts = [previous_condition]
        if code_ids.numel():
            parts.append(self.emb_code[0](code_ids))
        parts.append(current_condition)
        full_embeddings = torch.cat(parts, dim=0)
        previous_code_ids = tuple(int(code_id) for code_id in code_ids.tolist())
        previous_base_codes = state.get("base_recent_codes")
        if not isinstance(previous_base_codes, tuple):
            raise ValueError("streaming Talker condition lost its frozen codec history")
        states[request_id] = {
            "condition_seq": condition_seq,
            "condition": current_condition.detach().clone(),
            "active_embeddings": full_embeddings.detach().clone(),
            # Official generate_with_buffer keeps all_generated_tokens across
            # sliding recomputes, so the first sample in this chunk still sees
            # the previous chunk's repetition-penalty window.
            "base_recent_codes": (*previous_base_codes, *previous_code_ids)[-_CODEC_PENALTY_WINDOW:],
        }
        return full_embeddings

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build request-local prefill/decode embeddings for the vLLM runner."""
        del input_embeds
        span_len = int(input_ids.shape[0])
        is_prefill = bool(info_dict.get("_omni_is_prefill", False))
        state = info_dict.get("audio_state")
        first_call = not isinstance(state, dict)
        request_id = str(info_dict.get("request_id", "0"))

        if is_prefill or first_call:
            token_ids, hidden_states = get_tts_handoff(info_dict)
            # Cross-process stage transport serializes CPU tensors as lists.
            # Normalize both local tensor handoffs and transported payloads
            # before validating/building the Talker condition.
            if isinstance(token_ids, (list, tuple)):
                token_ids = torch.as_tensor(token_ids, dtype=torch.long)
            if isinstance(hidden_states, (list, tuple)):
                hidden_states = torch.as_tensor(hidden_states, dtype=torch.float32)
            if not isinstance(token_ids, torch.Tensor) or not isinstance(hidden_states, torch.Tensor):
                available = sorted(key for key in info_dict if not key.startswith("_"))
                raise ValueError(
                    "MiniCPM-o Talker requires tensor tts_token_ids and "
                    "tts_hidden_states conditioning; "
                    f"received token_ids={type(token_ids).__name__}, "
                    f"hidden_states={type(hidden_states).__name__}, "
                    f"available_keys={available}"
                )
            # An empty condition means the thinker chose not to speak: finish the
            # request up front so it emits zero audio codes instead of killing
            # the stage engine.
            empty_condition = token_ids.numel() == 0 or hidden_states.numel() == 0
            if empty_condition:
                logger.warning_once(
                    "MiniCPM-o Talker received an empty condition (request %s); this request produces no audio.",
                    info_dict.get("request_id"),
                )
            native_duplex = bool(info_dict.get("native_duplex", False))
            meta = info_dict.get("meta")
            full_embeds = self._build_condition_embeddings(
                token_ids,
                hidden_states,
                native_duplex=native_duplex,
            )
            if native_duplex:
                full_embeds = self._build_streaming_recompute_embeddings(
                    full_embeds,
                    request_id=request_id,
                    info_dict=info_dict,
                    meta=meta if isinstance(meta, Mapping) else {},
                )
            retained_codes: list[int] = []
            condition_seq = meta.get("streaming_condition_seq") if isinstance(meta, Mapping) else None
            if native_duplex and isinstance(condition_seq, int) and not isinstance(condition_seq, bool):
                condition_state = self._request_condition_states.get(request_id)
                base_recent_codes = (
                    condition_state.get("base_recent_codes") if isinstance(condition_state, dict) else None
                )
                if not isinstance(base_recent_codes, tuple):
                    raise ValueError("streaming Talker condition lost its frozen codec history")
                retained_codes = list(base_recent_codes)
            offset = max(0, int(info_dict.get("_omni_num_computed_tokens", 0) or 0))
            # The handoff rebuilds only the tail-aligned Talker condition.
            # Materialize zero-token embeddings for any scheduler prompt
            # prefix so chunked prefill can slice from a non-zero offset.
            prompt_len = info_dict.get("_omni_prompt_len")
            target_len = int(prompt_len) if prompt_len is not None else offset + span_len
            if native_duplex and isinstance(meta, Mapping) and meta.get("streaming_prompt_recompute") is True:
                if target_len != full_embeds.shape[0]:
                    raise ValueError(
                        "streaming prompt recompute length mismatch: "
                        f"scheduler={target_len}, model={full_embeds.shape[0]}"
                    )
            prefix_len = target_len - full_embeds.shape[0]
            if prefix_len > 0:
                placeholder_ids = torch.zeros(
                    prefix_len,
                    dtype=torch.long,
                    device=self.emb_text.weight.device,
                )
                full_embeds = torch.cat([self.emb_text(placeholder_ids), full_embeds], dim=0)
            embeds = full_embeds[offset : offset + span_len]
            if embeds.shape[0] != span_len:
                raise ValueError(
                    "MiniCPM-o Talker prefill span exceeds condition: "
                    f"request_id={info_dict.get('request_id')} offset={offset} "
                    f"span={span_len} condition={full_embeds.shape[0]} "
                    f"tts_ids={token_ids.shape[0]} tts_hidden={hidden_states.shape[0]} "
                    f"prompt_len={info_dict.get('_omni_prompt_len')}"
                )
            if native_duplex:
                max_tokens, min_tokens = _native_duplex_chunk_budget(meta if isinstance(meta, Mapping) else None)
            else:
                # MiniCPMTTS.generate()'s max_new_token, clamped to what the
                # Talker context can still hold. Sampler min_tokens (upstream's
                # min_new_token=50) comes from the deploy YAML.
                remaining = int(self._tts_config.max_position_embeddings) - target_len
                max_tokens = max(min(_OFFLINE_CODEC_MAX_NEW_TOKENS, remaining), 1)
                # None leaves the sampler floor owned by the deploy YAML; the
                # multi-frame path resolves it to self._codec_min_tokens when
                # the request state carries no explicit value.
                min_tokens = None
            state: dict[str, Any] = {
                "finished": empty_condition,
                "step": 0,
                "max_tokens": max_tokens,
                "min_tokens": min_tokens,
                "turn_end_drain": bool(native_duplex and isinstance(meta, Mapping) and bool(meta.get("turn_end"))),
            }
            if retained_codes:
                state["recent_codes"] = retained_codes
                # Rebuild the device-side penalty window once per chunk from the
                # carried-over ids; the per-frame path below then appends on
                # device without touching the host again.
                retained = [int(code_id) for code_id in retained_codes][-_CODEC_PENALTY_WINDOW:]
                if retained:
                    penalty_windows = getattr(self, "_penalty_windows_dev", None)
                    if not isinstance(penalty_windows, dict):
                        penalty_windows = {}
                        self._penalty_windows_dev = penalty_windows
                    penalty_windows[request_id] = torch.tensor(retained, dtype=torch.long, device=embeds.device)
                    pf_freq = torch.zeros(
                        int(getattr(self, "_num_audio_tokens", 0) or 0),
                        device=embeds.device,
                        dtype=torch.float32,
                    )
                    if pf_freq.numel() > 0:
                        pf_freq.index_add_(
                            0,
                            penalty_windows[request_id],
                            torch.ones(len(retained), device=embeds.device, dtype=torch.float32),
                        )
                        pf_dict = getattr(self, "_penalty_freqs_dev", None)
                        if not isinstance(pf_dict, dict):
                            pf_dict = {}
                            self._penalty_freqs_dev = pf_dict
                        pf_dict[request_id] = pf_freq
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            request_states[request_id] = state
            empty_codes = torch.empty(0, dtype=torch.long, device=embeds.device)
            return (
                input_ids,
                embeds,
                {
                    "audio_state": state,
                    # Prefill has no previous codec id. vLLM samples the first
                    # code after this forward; the next decode emits it.
                    "codes": {"audio": empty_codes},
                },
            )

        stored = self._request_audio_states.get(request_id)
        if isinstance(stored, dict):
            state = stored
        if isinstance(state, dict) and state.get("finished"):
            # An empty speech segment can still be scheduled until EOS is
            # eligible. The sampler is forced to EOS; any shape-correct
            # embedding is enough for these leftover decode rows.
            weight = self.emb_code[0].weight
            empty_codes = torch.empty(0, dtype=torch.long, device=weight.device)
            return input_ids, weight.new_zeros((span_len, weight.shape[1])), {"codes": {"audio": empty_codes}}

        # Decode: vLLM's previous sampled codec id is this step's input.
        # Embed it with the codec table (not the 32k Llama embed_tokens) and
        # hand the same id to make_omni_output so Code2Wav sees it this step.
        # K-step: once the multi-frame loop owns sampling, the rows vLLM
        # schedules carry placeholder continue ids, not real codec ids -- the
        # true previous frame's sample lives in the request state.
        k_last = state.get("last_code") if isinstance(state, dict) else None
        if isinstance(k_last, int) and k_last >= 0:
            code = torch.tensor(
                [k_last],
                dtype=torch.long,
                device=self.emb_code[0].weight.device,
            )
        else:
            code = input_ids.to(device=self.emb_code[0].weight.device, dtype=torch.long).reshape(-1)[-1:]
        embeds = self.emb_code[0](code)
        code_id = int(code.item())
        if code_id == int(self._codec_eos_id):
            if isinstance(state, dict):
                state["finished"] = True
            elif stored is None:
                self._request_audio_states[request_id] = {"finished": True, "step": 0}
            delta = torch.empty(0, dtype=torch.long, device=code.device)
        else:
            delta = code.reshape(1, 1)
        return input_ids, embeds, {"codes": {"audio": delta}}

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        infos = kwargs.get("model_intermediate_buffer") or []
        spans = kwargs.get("request_token_spans")
        if spans is None or len(spans) != len(infos):
            raise RuntimeError("MiniCPM-o continuous Talker requires one request_token_span per request")
        if self._k_step_frames <= 0:
            # Single-frame path (the default): one codec frame per step, sampled
            # by vLLM. Keeps the host-state contract consumed by compute_logits
            # and sample: state step/recent_codes/finished bookkeeping plus the
            # force/mask codec-EOS routing.
            return self._make_omni_output_single_frame(hidden, infos, spans)
        sample_eligible = kwargs.get("request_sample_eligible")
        if sample_eligible is None:
            sample_eligible = [True] * len(infos)
        if len(sample_eligible) != len(infos):
            raise RuntimeError(
                f"MiniCPM-o continuous Talker received {len(sample_eligible)} sampling flags for {len(infos)} requests"
            )
        emit_duplex_metadata = any(isinstance(info, dict) and info.get("native_duplex") is True for info in infos)

        stop_rows: list[torch.Tensor] = []
        codec_deltas: list[torch.Tensor] = []
        terminal_flags: list[torch.Tensor] = []
        native_duplex_flags: list[torch.Tensor] = []
        duplex_epochs: list[torch.Tensor] = []
        duplex_turn_ids: list[torch.Tensor] = []
        segment_texts_utf8: list[torch.Tensor] = []
        turn_end_flags: list[torch.Tensor] = []
        row_continue, row_stop, flag_false, flag_true, empty_delta = self._step_constants(hidden)
        for index, info in enumerate(infos):
            info_dict = info if isinstance(info, dict) else {}
            native_duplex = info_dict.get("native_duplex") is True
            if emit_duplex_metadata:
                duplex_info = info_dict.get("duplex")
                if not isinstance(duplex_info, dict):
                    duplex_info = {}
                epoch = duplex_info.get("epoch", -1)
                turn_id = duplex_info.get("turn_id", -1)
                if native_duplex and not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (epoch, turn_id)
                ):
                    raise RuntimeError(
                        "MiniCPM-o native duplex Talker requires non-negative integer "
                        f"epoch and turn_id, got epoch={epoch!r}, turn_id={turn_id!r}"
                    )
                meta_info = info_dict.get("meta")
                if not isinstance(meta_info, dict):
                    meta_info = {}
                segment_text = meta_info.get("native_duplex_segment_text", "") if native_duplex else ""
                if not isinstance(segment_text, str):
                    segment_text = ""
                turn_eos_id = meta_info.get("turn_eos_token_id")
                ids_info = info_dict.get("ids")
                tts_ids = ids_info.get("tts") if native_duplex and isinstance(ids_info, dict) else None
                if isinstance(tts_ids, torch.Tensor):
                    contains_turn_eos = isinstance(turn_eos_id, int) and bool(
                        torch.any(tts_ids.reshape(-1) == turn_eos_id).item()
                    )
                elif isinstance(tts_ids, (list, tuple)):
                    contains_turn_eos = isinstance(turn_eos_id, int) and turn_eos_id in tts_ids
                else:
                    contains_turn_eos = False
                native_duplex_flags.append(torch.tensor(native_duplex, dtype=torch.bool))
                duplex_epochs.append(torch.tensor(epoch if isinstance(epoch, int) else -1, dtype=torch.long))
                duplex_turn_ids.append(torch.tensor(turn_id if isinstance(turn_id, int) else -1, dtype=torch.long))
                segment_texts_utf8.append(
                    torch.tensor(
                        list(segment_text.encode("utf-8")),
                        dtype=torch.uint8,
                    )
                )
                turn_end_flags.append(torch.tensor(native_duplex and contains_turn_eos, dtype=torch.bool))

            if not isinstance(info, dict):
                stop_rows.append(row_continue)
                codec_deltas.append(empty_delta)
                terminal_flags.append(flag_false)
                continue
            start, end = spans[index]
            end = min(int(end), int(hidden.shape[0]))
            if int(start) >= end:
                stop_rows.append(row_continue)
                codec_deltas.append(empty_delta)
                terminal_flags.append(flag_false)
                continue
            request_id = str(info.get("request_id", index))
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            state = request_states.get(request_id)
            if not isinstance(state, dict):
                state = dict(info.get("audio_state", {}) or {})
                request_states[request_id] = state
            if state.get("finished"):
                stop_rows.append(row_stop)
                codec_deltas.append(empty_delta)
                terminal_flags.append(flag_false)
                continue
            if not sample_eligible[index]:
                # vLLM computes a logit row for incomplete chunked prefills but
                # discards its sampled token. Advancing codec/RNG state here
                # would make output depend on prefill chunking and compaction.
                stop_rows.append(row_continue)
                codec_deltas.append(empty_delta)
                terminal_flags.append(flag_false)
                continue
            codes = state.get("codes")
            if not isinstance(codes, torch.Tensor):
                codes = (info.get("audio_codes", {}) or {}).get("accumulated")
            if not isinstance(codes, torch.Tensor):
                codes = torch.empty(0, dtype=torch.long, device=hidden.device)
            else:
                codes = codes.to(device=hidden.device, dtype=torch.long).reshape(-1)
            step = _codec_int_param(state, "step", 0)
            # Note: main's turn-end semantics require masking codec EOS within 5
            # steps after each 25-frame cadence boundary (see
            # _turn_end_boundary_eos_masked and state["turn_end_drain"];
            # MINICPMO45_DUPLEX_TURN_END_CODEC_TOKENS is 4 chunks, so the window
            # lands on steps 25-29 / 50-54 / 75-79 / 100-103). The EOS masking
            # moved into the sampling core (talker_codec_sample.
            # prepare_codec_logits / greedy_codec_sample), whose criterion is
            # state.step < min_tokens with min_tokens a per-request cached
            # tensor, so that periodic window cannot be expressed per frame here
            # and is left unwired. It only affects duplex turn-end; the
            # offline/TTS path is unaffected. The constants, budget and
            # turn_end_drain flag from main are kept in this file.
            min_tokens = _codec_int_param(state, "min_tokens", self._codec_min_tokens)
            max_tokens = _codec_int_param(state, "max_tokens", self._codec_max_tokens)
            if self._codec_temperature == 0.0:
                sampled = self._sample_audio_code_greedy(
                    hidden[end - 1 : end],
                    codes,
                    request_id,
                    step,
                    min_tokens,
                    max_tokens,
                )
            else:
                stochastic_result = self._sample_audio_code(hidden[end - 1 : end], codes, request_id, step)
                if isinstance(stochastic_result, TalkerCodecSampleResult):
                    sampled = stochastic_result.sampled_token.reshape(()).to(torch.long)
                else:
                    # CPU unit tests and downstream subclasses historically
                    # stub this method with the sampled tensor itself.
                    sampled = stochastic_result.reshape(()).to(torch.long)
            sampled_id = int(sampled.item())
            prev_code = state.get("last_code") if isinstance(state, Mapping) else None
            is_eos = sampled_id == self._codec_eos_id
            state["step"] = _codec_int_param(state, "step", 0) + 1
            reached_limit = int(state["step"]) >= _codec_int_param(state, "max_tokens", self._codec_max_tokens)
            finished = is_eos or reached_limit
            state["finished"] = finished
            if self._k_step_frames > 0 and kstep_frame_trace_enabled():
                _trace_kstep_frame(
                    self,
                    request_id=request_id,
                    step_before=step,
                    step_after=int(state["step"]),
                    prev_code=prev_code,
                    row=int(end) - 1,
                    sampled_id=sampled_id,
                    is_eos=is_eos,
                    reached_limit=reached_limit,
                    finished=finished,
                    ctx_len=int(codes.numel()) if isinstance(codes, torch.Tensor) else -1,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                )
            # MiniCPMTTS.generate_chunk consumes the boundary sample but
            # returns only codes that were fed into the retained KV state.
            if not is_eos and not reached_limit:
                codes = torch.cat([codes[-(_REPETITION_WINDOW - 1) :], sampled.reshape(1)])
                delta = sampled.reshape(1, 1)
                if self._k_step_frames > 0:
                    # K-step: the multi-frame loop owns sampling, so the rows
                    # vLLM schedules for the next step carry placeholder
                    # continue ids. Record this frame's sample so the decode
                    # preprocess embeds the real previous frame's codec id
                    # (the k_last branch) instead of the placeholder.
                    state["last_code"] = sampled_id
            else:
                delta = empty_delta
            state["codes"] = codes
            info["audio_state"] = state
            info["audio_codes"] = {
                "current": sampled.reshape(1),
                "accumulated": codes,
            }
            codec_deltas.append(delta)
            terminal_flags.append(flag_true if finished else flag_false)
            stop_rows.append(row_stop if finished else row_continue)

        self._batch_stop_logits = torch.stack(stop_rows, dim=0) if stop_rows else hidden.new_empty((0, 2))
        # Lists are deliberate: the runner routes element i to request i,
        # preserving compaction alignment while emitting only this step's code.
        meta_outputs = {"finished": terminal_flags}
        if emit_duplex_metadata:
            meta_outputs.update(
                {
                    "native_duplex": native_duplex_flags,
                    "duplex_epoch": duplex_epochs,
                    "duplex_turn_id": duplex_turn_ids,
                    "llm_output_text_utf8": segment_texts_utf8,
                    "turn_end": turn_end_flags,
                }
            )
        multimodal_outputs: dict[str, Any] = {
            "codes": {"audio": codec_deltas},
            "meta": meta_outputs,
        }
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_outputs,
        )

    # This model always emits exactly one codec frame per query position, which
    # is what makes a multi-frame step decomposable into per-frame calls.
    supports_multi_frame_decode = True

    def take_batch_stop_logits(self) -> torch.Tensor | None:
        """Hand over this frame's stop rows and clear them.

        ``compute_logits`` normally consumes them at the end of the step. A
        multi-frame step calls ``make_omni_output`` once per frame, so the
        runner has to collect each frame's rows before the next call replaces
        them, and hand the merged tensor back through
        ``set_batch_stop_logits``.
        """
        logits = self._batch_stop_logits
        self._batch_stop_logits = None
        return logits

    def merge_frame_outputs(
        self,
        frame_outputs: list[OmniOutput],
        frame_stop_logits: list[torch.Tensor],
    ) -> OmniOutput:
        """Fold a multi-frame step's per-frame outputs into one step output.

        Codec deltas concatenate per request -- the connector already takes a
        variable number of frames per step and drops the ``-1`` sentinel, so a
        request that ended mid-step contributes only the frames it actually
        emitted. Stop rows interleave to the layout ``logits_indices`` reads:
        every query position of request 0, then of request 1, and so on.
        """
        if not frame_outputs:
            raise RuntimeError("MiniCPM-o multi-frame step produced no frames")
        if len(frame_outputs) == 1:
            self.set_batch_stop_logits(frame_stop_logits[0])
            return frame_outputs[0]

        per_frame_codes = [output.multimodal_outputs["codes"]["audio"] for output in frame_outputs]
        num_reqs = len(per_frame_codes[0])
        codec_deltas = [torch.cat([frame[index] for frame in per_frame_codes], dim=0) for index in range(num_reqs)]

        meta_outputs: dict[str, Any] = {}
        for key in frame_outputs[0].multimodal_outputs["meta"]:
            per_frame = [output.multimodal_outputs["meta"][key] for output in frame_outputs]
            if key == "finished":
                meta_outputs[key] = [
                    self._merge_frame_finished([frame[index] for frame in per_frame]) for index in range(num_reqs)
                ]
            else:
                # Duplex metadata is per request and constant across the step's
                # frames; the last frame is as good as any and matches what a
                # single-frame step would have reported.
                meta_outputs[key] = per_frame[-1]

        self.set_batch_stop_logits(torch.stack(frame_stop_logits, dim=1).reshape(-1, 2))
        return OmniOutput(
            text_hidden_states=frame_outputs[-1].text_hidden_states,
            multimodal_outputs={"codes": {"audio": codec_deltas}, "meta": meta_outputs},
        )

    def _request_generator(self, request_id: str, device: torch.device) -> torch.Generator:
        generator = self._request_generators.get(request_id)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._codec_seed)
            self._request_generators[request_id] = generator
        return generator

    def _sample_audio_code(
        self,
        hidden_state: torch.Tensor,
        history: torch.Tensor,
        request_id: str,
        step: int,
    ) -> TalkerCodecSampleResult:
        device_states = getattr(self, "_request_codec_device_states", None)
        if device_states is None:
            device_states = {}
            self._request_codec_device_states = device_states
        device_state = device_states.get(request_id)
        if device_state is None:
            request_states = getattr(self, "_request_audio_states", {})
            request_state = request_states.get(request_id, {})
            device_state = make_device_state(
                history,
                step=step,
                max_tokens=_codec_int_param(request_state, "max_tokens", self._codec_max_tokens),
                finished=False,
            )
            device_states[request_id] = device_state
        device_inputs_by_request = getattr(self, "_request_codec_device_inputs", None)
        if device_inputs_by_request is None:
            device_inputs_by_request = {}
            self._request_codec_device_inputs = device_inputs_by_request
        device_inputs = device_inputs_by_request.get(request_id)
        if device_inputs is None:
            request_states = getattr(self, "_request_audio_states", {})
            request_state = request_states.get(request_id, {})
            device_inputs = (
                torch.tensor(
                    [_codec_int_param(request_state, "min_tokens", self._codec_min_tokens)],
                    dtype=torch.int32,
                    device=hidden_state.device,
                ),
                torch.tensor([self._codec_temperature], dtype=torch.float32, device=hidden_state.device),
                torch.tensor(
                    [self._codec_repetition_penalty],
                    dtype=torch.float32,
                    device=hidden_state.device,
                ),
            )
            device_inputs_by_request[request_id] = device_inputs
        min_tokens_tensor, temperature_tensor, penalty_tensor = device_inputs
        eos_id = self._codec_eos_id
        logits = prepare_codec_logits(
            self.head_code[0](hidden_state).float(),
            device_state,
            min_tokens_tensor,
            temperature_tensor,
            penalty_tensor,
            eos_token_id=eos_id,
            top_k=self._codec_top_k,
            top_p=self._codec_top_p,
            min_tokens_to_keep=3,
        )
        probabilities = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=self._request_generator(request_id, probabilities.device),
        ).reshape(())
        result = codec_sample_result(
            device_state,
            sampled,
            eos_token_id=eos_id,
        )
        device_states[request_id] = result.state
        return result

    def _sample_audio_code_greedy(
        self,
        hidden_state: torch.Tensor,
        history: torch.Tensor,
        request_id: str,
        step: int,
        min_tokens: int,
        max_tokens: int,
    ) -> torch.Tensor:
        """Run the greedy codec boundary while keeping sampler state on device.

        Host stop routing still consumes ``sampled.item()`` below; removing that
        sync requires a runner/connector state refactor and is not hidden here.
        """
        device_states = getattr(self, "_request_codec_device_states", None)
        if device_states is None:
            device_states = {}
            self._request_codec_device_states = device_states
        device_state = device_states.get(request_id)
        if device_state is None:
            device_state = make_device_state(
                history,
                step=step,
                max_tokens=max_tokens,
                finished=False,
            )
            device_states[request_id] = device_state

        device_inputs_by_request = getattr(self, "_request_codec_device_inputs", None)
        if device_inputs_by_request is None:
            device_inputs_by_request = {}
            self._request_codec_device_inputs = device_inputs_by_request
        device_inputs = device_inputs_by_request.get(request_id)
        if device_inputs is None:
            device_inputs = (
                torch.tensor([min_tokens], dtype=torch.int32, device=hidden_state.device),
                torch.tensor([1.0], dtype=torch.float32, device=hidden_state.device),
                torch.tensor(
                    [self._codec_repetition_penalty],
                    dtype=torch.float32,
                    device=hidden_state.device,
                ),
            )
            device_inputs_by_request[request_id] = device_inputs
        min_tokens_tensor, _, penalty_tensor = device_inputs
        result = greedy_codec_sample(
            self.head_code[0](hidden_state).float(),
            device_state,
            min_tokens_tensor,
            penalty_tensor,
            top_k=self._codec_top_k,
            eos_token_id=self._codec_eos_id,
        )
        device_states[request_id] = result.state
        # The kernel ABI is int32, while the existing connector/audio-code
        # payload is int64.  Keep the compatibility cast outside the fused op.
        return result.sampled_token.to(torch.long).reshape(())

    def _step_constants(self, hidden: torch.Tensor):
        """Constant per-step tensors, cached per (device, dtype).

        These never change and downstream consumers only read them, so the
        per-step H2D copies from ``new_tensor``/``torch.tensor`` are avoidable.
        """
        cache = getattr(self, "_step_constants_cache", None)
        key = (hidden.device, hidden.dtype)
        if cache is None or cache[0] != key:
            neg_inf = float("-inf")
            cache = (
                key,
                (
                    hidden.new_tensor([0.0, neg_inf]),
                    hidden.new_tensor([neg_inf, 0.0]),
                    torch.tensor(False, dtype=torch.bool),
                    torch.tensor(True, dtype=torch.bool),
                    hidden.new_empty((0, 1), dtype=torch.long),
                ),
            )
            self._step_constants_cache = cache
        return cache[1]

    def set_batch_stop_logits(self, logits: torch.Tensor | None) -> None:
        self._batch_stop_logits = logits

    @staticmethod
    def _merge_frame_finished(flags: list[torch.Tensor]) -> torch.Tensor:
        """OR the per-frame terminal flags without forcing a device sync.

        ``make_omni_output`` reports the flag on the frame that ended the codec
        sequence and reports ``False`` for every frame after it, so the step's
        answer is the OR. The flags are device tensors on the device-state
        path and CPU constants on the native-sampler path, and one step can
        produce both --
        the frame that finishes samples on device, the frames after it take the
        early return and get the CPU constant. Reading the CPU ones costs
        nothing; the device ones are OR'd on device and travel out in the
        runner's single coalesced D2H.
        """
        merged: torch.Tensor | None = None
        for flag in flags:
            if flag.device.type == "cpu":
                if bool(flag.reshape(()).item()):
                    return flag
                continue
            flag = flag.reshape(1)
            merged = flag if merged is None else torch.logical_or(merged, flag)
        if merged is None:
            return flags[-1]
        return merged.reshape(())

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        self._deferred_cleanup_ids.update(str(req_id) for req_id in finished_req_ids)

    def _flush_deferred_cleanup(self) -> None:
        """Drop every per-request cache a finished request left behind.

        The sampler's own caches belong here as much as the codec history does:
        ``_request_codec_device_states`` carries the step counter and the
        penalty window the EOS mask is derived from, and
        ``_request_codec_device_inputs`` carries the ``min_tokens`` copied at
        first use. Leaving them behind means a recycled request id resumes with
        another request's counters -- the EOS mask then holds or lifts at the
        wrong frame, and ``_request_generators`` leaks a generator per request
        on top of that. All three key off the same request id, so they are
        dropped together with the rest.
        """
        request_audio_states = getattr(self, "_request_audio_states", {})
        request_condition_states = getattr(self, "_request_condition_states", {})
        penalty_windows = getattr(self, "_penalty_windows_dev", None)
        penalty_freqs = getattr(self, "_penalty_freqs_dev", None)
        codec_device_states = getattr(self, "_request_codec_device_states", None)
        codec_device_inputs = getattr(self, "_request_codec_device_inputs", None)
        request_generators = getattr(self, "_request_generators", None)
        for request_id in self._deferred_cleanup_ids:
            request_audio_states.pop(request_id, None)
            request_condition_states.pop(request_id, None)
            if isinstance(penalty_windows, dict):
                penalty_windows.pop(request_id, None)
            if isinstance(penalty_freqs, dict):
                penalty_freqs.pop(request_id, None)
            if isinstance(codec_device_states, dict):
                codec_device_states.pop(request_id, None)
            if isinstance(codec_device_inputs, dict):
                codec_device_inputs.pop(request_id, None)
            if isinstance(request_generators, dict):
                request_generators.pop(request_id, None)
        self._deferred_cleanup_ids.clear()

    def _dummy_hidden_states(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shape-correct zero tensor for vllm KV cache profiling.

        vllm's gpu_model_runner._dummy_run takes forward()'s return value as
        ``hidden_states`` and does ``hidden_states[logit_indices_device]``;
        returning None on the dummy path crashes with
        ``TypeError: 'NoneType' object is not subscriptable``.
        """
        for ref in (input_ids, positions, inputs_embeds):
            if isinstance(ref, torch.Tensor):
                num_tokens = int(ref.shape[0]) if ref.ndim >= 1 else 1
                device = ref.device
                break
        else:
            num_tokens = 1
            device = current_omni_platform.get_torch_device()
        hidden_size = int(getattr(self, "_hidden_size", 768) or 768)
        return torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        self._flush_deferred_cleanup()
        if input_ids is None and inputs_embeds is None:
            return self._dummy_hidden_states(input_ids, positions, inputs_embeds)
        return self.tts_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def _make_omni_output_single_frame(
        self,
        hidden: torch.Tensor,
        infos: list[Any],
        spans: list[Any],
    ) -> OmniOutput:
        """Single-frame Talker output: bookkeeping plus codec-EOS routing.

        Emits the codec delta sampled on the previous step and decides which
        rows ``compute_logits`` must force to EOS (finished segments, chunk
        budget reached) or mask EOS on (min_tokens, duplex turn-end drain
        window).
        """
        emit_duplex_metadata = any(isinstance(info, dict) and info.get("native_duplex") is True for info in infos)
        native_duplex_flags: list[torch.Tensor] = []
        duplex_epochs: list[torch.Tensor] = []
        duplex_turn_ids: list[torch.Tensor] = []
        segment_texts_utf8: list[torch.Tensor] = []
        turn_end_flags: list[torch.Tensor] = []
        empty_delta = hidden.new_empty((0, 1), dtype=torch.long)
        codec_deltas = [empty_delta for _ in infos]
        terminal_flags = [torch.tensor(False, dtype=torch.bool) for _ in infos]
        force_eos_rows = [False] * len(infos)
        mask_eos_rows = [False] * len(infos)
        empty_history = hidden.new_empty((0,), dtype=torch.long)
        penalty_histories = [empty_history for _ in infos]
        for index, info in enumerate(infos):
            info_dict = info if isinstance(info, dict) else {}
            native_duplex = info_dict.get("native_duplex") is True
            if emit_duplex_metadata:
                duplex_info = info_dict.get("duplex")
                if not isinstance(duplex_info, dict):
                    duplex_info = {}
                epoch = duplex_info.get("epoch", -1)
                turn_id = duplex_info.get("turn_id", -1)
                if native_duplex and not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (epoch, turn_id)
                ):
                    raise RuntimeError(
                        "MiniCPM-o native duplex Talker requires non-negative integer "
                        f"epoch and turn_id, got epoch={epoch!r}, turn_id={turn_id!r}"
                    )
                meta_info = info_dict.get("meta")
                if not isinstance(meta_info, dict):
                    meta_info = {}
                segment_text = meta_info.get("native_duplex_segment_text", "") if native_duplex else ""
                if not isinstance(segment_text, str):
                    segment_text = ""
                turn_eos_id = meta_info.get("turn_eos_token_id")
                ids_info = info_dict.get("ids")
                tts_ids = ids_info.get("tts") if native_duplex and isinstance(ids_info, dict) else None
                if isinstance(tts_ids, torch.Tensor):
                    contains_turn_eos = isinstance(turn_eos_id, int) and bool(
                        torch.any(tts_ids.reshape(-1) == turn_eos_id).item()
                    )
                elif isinstance(tts_ids, (list, tuple)):
                    contains_turn_eos = isinstance(turn_eos_id, int) and turn_eos_id in tts_ids
                else:
                    contains_turn_eos = False
                native_duplex_flags.append(torch.tensor(native_duplex, dtype=torch.bool))
                duplex_epochs.append(torch.tensor(epoch if isinstance(epoch, int) else -1, dtype=torch.long))
                duplex_turn_ids.append(torch.tensor(turn_id if isinstance(turn_id, int) else -1, dtype=torch.long))
                segment_texts_utf8.append(
                    torch.tensor(
                        list(segment_text.encode("utf-8")),
                        dtype=torch.uint8,
                    )
                )
                turn_end_flags.append(torch.tensor(native_duplex and contains_turn_eos, dtype=torch.bool))

            if not isinstance(info, dict):
                continue
            request_id = str(info.get("request_id", index))
            state = self._request_audio_states.get(request_id)
            if not isinstance(state, dict):
                state = dict(info.get("audio_state", {}) or {})
                self._request_audio_states[request_id] = state
            empty_speech = bool(state.get("finished"))
            codes = info.get("codes", {})
            audio = codes.get("audio") if isinstance(codes, Mapping) else None
            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                codec_deltas[index] = audio.to(device=hidden.device, dtype=torch.long).reshape(-1, 1)
                state["step"] = int(state.get("step", 0)) + 1
                # ``audio`` is the id sampled last step, i.e. exactly the
                # history the logits computed below are scored against.
                recent = state.get("recent_codes")
                recent = (recent if isinstance(recent, list) else []) + codec_deltas[index].reshape(-1).tolist()
                state["recent_codes"] = recent[-_CODEC_PENALTY_WINDOW:]
            recent_codes = state.get("recent_codes")
            if recent_codes:
                penalty_histories[index] = torch.tensor(recent_codes, dtype=torch.long, device=hidden.device)
            max_tokens = state.get("max_tokens")
            min_tokens = state.get("min_tokens")
            step = int(state.get("step", 0))
            # Duplex: 26 samples include the terminating EOS, so force it after
            # 25 forwarded frames -- the same cadence as generate_chunk.
            # Offline: force-stop at the remaining Talker context rather than
            # waiting for a sampled EOS.
            hit_chunk_limit = max_tokens is not None and step >= int(max_tokens) - 1
            chunk_done = empty_speech or hit_chunk_limit
            if hit_chunk_limit:
                # The next sampled id is forced EOS and will finish the request;
                # mark the chunk done on this step so the data plane does not
                # wait for a follow-up empty decode.
                state["finished"] = True
            force_eos_rows[index] = chunk_done
            mask_eos_rows[index] = not force_eos_rows[index] and (
                (min_tokens is not None and step < int(min_tokens))
                or (bool(state.get("turn_end_drain")) and _turn_end_boundary_eos_masked(step))
            )
            terminal_flags[index] = torch.tensor(chunk_done, dtype=torch.bool)

        # Empty-speech rows, finished duplex chunks, and offline requests that
        # fill the remaining Talker context must sample codec EOS so the
        # scheduler releases the request. Mid-chunk rows mask EOS until
        # min_tokens.
        self._force_eos_rows = force_eos_rows
        self._mask_eos_rows = mask_eos_rows
        self._penalty_histories = penalty_histories
        meta_outputs: dict[str, Any] = {"finished": terminal_flags}
        if emit_duplex_metadata:
            meta_outputs.update(
                {
                    "native_duplex": native_duplex_flags,
                    "duplex_epoch": duplex_epochs,
                    "duplex_turn_id": duplex_turn_ids,
                    "llm_output_text_utf8": segment_texts_utf8,
                    "turn_end": turn_end_flags,
                }
            )
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={
                "codes": {"audio": codec_deltas},
                "meta": meta_outputs,
            },
        )

    def compute_logits(self, hidden_states, *args, **kwargs):
        if not isinstance(hidden_states, torch.Tensor):
            return None
        if self._k_step_frames > 0:
            # K-step: the codec stream was already sampled in-model, one frame
            # per replay, so this head carries no codec information at all.
            # All it decides is stop/continue per scheduled position, over a
            # two-id vocabulary -- continue (0) and stop (1) -- which is what
            # the stage's ``stop_token_ids: [1]`` names. merge_frame_outputs
            # publishes this step's rows in logits_indices order; a step with
            # none gets zeros, whose argmax is 0 and therefore "continue".
            if self._batch_stop_logits is None:
                return torch.zeros(
                    hidden_states.shape[0],
                    2,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
            logits = self._batch_stop_logits
            self._batch_stop_logits = None
            return logits
        if hidden_states.numel() == 0:
            return hidden_states.new_empty((0, int(self._num_audio_tokens)))
        # Non-K-step: the real codec head. vLLM samples the codec stream here,
        # and the stage's stop_token_ids ends the request when it produces it.
        logits = self.head_code[0](hidden_states).float()
        force_eos = self._force_eos_rows
        mask_eos = self._mask_eos_rows
        self._force_eos_rows = None
        self._mask_eos_rows = None
        need_force = bool(force_eos and len(force_eos) == logits.shape[0] and any(force_eos))
        need_mask = bool(mask_eos and len(mask_eos) == logits.shape[0] and any(mask_eos))
        # sample() re-applies the decision on the sampled ids: vLLM's
        # MinTokensLogitsProcessor runs after this and would blank the codec EOS
        # we just forced (it is in the stage's ``stop_token_ids``), leaving an
        # all -inf row and a request that never releases.
        self._pending_force_eos_rows = force_eos if need_force else None
        if need_force or need_mask:
            logits = logits.clone()
            eos_id = int(self._codec_eos_id)
            if need_force:
                assert force_eos is not None
                forced = torch.tensor(force_eos, dtype=torch.bool, device=logits.device)
                logits[forced] = float("-inf")
                logits[forced, eos_id] = 0.0
            if need_mask:
                assert mask_eos is not None
                masked = torch.tensor(mask_eos, dtype=torch.bool, device=logits.device)
                logits[masked, eos_id] = float("-inf")
        return logits

    def sample(self, logits, sampling_metadata):
        # Two-column rows are the K-step stop/continue heads compute_logits
        # built: they are one-hot, so a full sampler pass is ~10 host-dispatched
        # kernels for a deterministic pick while argmax is bit-identical on
        # them. int32 matches what the runner's async path scatters into its
        # input buffers. Anything else (the non-K-step codec head) takes the
        # ordinary path below, untouched.
        if (
            isinstance(logits, torch.Tensor)
            and logits.ndim == 2
            and logits.shape[-1] == 2
            and not getattr(sampling_metadata, "max_num_logprobs", None)
        ):
            with _prof_span("tts_sampler"):
                return SamplerOutput(
                    sampled_token_ids=logits.argmax(dim=-1, keepdim=True).to(torch.int32),
                    logprobs_tensors=None,
                )
        prompt_ids = getattr(sampling_metadata, "prompt_token_ids", None)
        if isinstance(logits, torch.Tensor) and isinstance(prompt_ids, torch.Tensor):
            # Copy rather than mutate: the runner may hand us the input batch's
            # own persistent SamplingMetadata.
            sampling_metadata = replace(
                sampling_metadata,
                prompt_token_ids=blank_scheduler_prompt_for_penalties(prompt_ids, logits.shape[-1]),
            )
        with _prof_span("tts_penalty"):
            logits, sampling_metadata = self._apply_codec_repetition_penalty(logits, sampling_metadata)
        prewarped = _maybe_prewarp_top_k_top_p(logits, sampling_metadata)
        if prewarped is not None:
            logits, sampling_metadata = prewarped
        force_eos = self._pending_force_eos_rows
        self._pending_force_eos_rows = None
        with _prof_span("tts_sampler"):
            output = Sampler()(logits, sampling_metadata)
        return self._force_eos_on_sampled_ids(output, force_eos)

    def _force_eos_on_sampled_ids(self, output: Any, force_eos: list[bool] | None) -> Any:
        """Overwrite sampled ids for rows the model terminated this step.

        The codec EOS is a stage ``stop_token_ids`` entry, so vLLM's
        ``min_tokens`` processor masks it for the first ``min_tokens`` steps.
        A row the model forced to EOS therefore reaches the sampler as all
        -inf and comes back as an arbitrary codec id, which keeps an
        already-finished request decoding until its length cap.
        """
        if not force_eos or not any(force_eos):
            return output
        sampled = getattr(output, "sampled_token_ids", None)
        if not isinstance(sampled, torch.Tensor) or sampled.shape[0] != len(force_eos):
            return output
        rows = torch.tensor(force_eos, dtype=torch.bool, device=sampled.device)
        sampled[rows] = int(self._codec_eos_id)
        return output

    def _apply_codec_repetition_penalty(self, logits, sampling_metadata):
        """Score MiniCPMTTS.generate's windowed codec penalty, not vLLM's.

        Upstream taxes a code by ``penalty ** frequency`` over the last
        ``past_window`` frames only (``gen_logits`` builds
        ``CustomRepetitionPenaltyLogitsProcessorRepeat(penalty, num_code, 16)``).
        vLLM's is presence-based over the whole stream, so on a codec stream
        thousands of frames long every code ever sampled ends up taxed by the
        same flat factor while never-sampled codes stay untouched, and the tail
        of a long answer drifts off the speech manifold into near-silence.

        Runs before ``Sampler`` so the penalty lands ahead of top-k/top-p, as
        upstream does. Upstream scores it after dividing by temperature, but
        the penalty only rescales and preserves sign, so the two orders agree.
        """
        histories = self._penalty_histories
        self._penalty_histories = None
        penalties = getattr(sampling_metadata, "repetition_penalties", None)
        if (
            not isinstance(logits, torch.Tensor)
            or histories is None
            or len(histories) != logits.shape[0]
            or not isinstance(penalties, torch.Tensor)
            or getattr(sampling_metadata, "no_penalties", False)
        ):
            self._penalty_freq_rows = None
            return logits, sampling_metadata
        freq_rows = getattr(self, "_penalty_freq_rows", None)
        self._penalty_freq_rows = None
        if (
            isinstance(freq_rows, list)
            and len(freq_rows) == logits.shape[0]
            and all(
                isinstance(f, torch.Tensor) and f.device == logits.device and f.numel() == logits.shape[-1]
                for f in freq_rows
            )
            and bool((penalties.reshape(-1) != 1.0).any())
        ):
            # Incremental histogram: freq rows are maintained on device by
            # make_omni_output (+1 append / -1 evict), identical to a full
            # scatter_add rebuild of the sliding window.
            freqs = torch.stack([f.to(dtype=torch.float32) for f in freq_rows], dim=0)
            alpha = torch.pow(penalties.to(device=logits.device, dtype=torch.float32).reshape(-1, 1), freqs).to(
                dtype=logits.dtype
            )
            logits = torch.where(logits < 0, logits * alpha, logits / alpha)
            return logits, replace(sampling_metadata, repetition_penalties=torch.ones_like(penalties))
        logits = _apply_batched_repetition_penalty(
            logits,
            histories,
            penalty=penalties.to(device=logits.device, dtype=logits.dtype),
            window_size=_CODEC_PENALTY_WINDOW,
        )
        # Neutralize the sampler's own pass so the penalty is scored once.
        return logits, replace(sampling_metadata, repetition_penalties=torch.ones_like(penalties))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return self._load_native_weights(weights)

    def _load_native_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        direct_params = dict(self.named_parameters())
        head_g = head_v = None

        for name, tensor in weights:
            if not name.startswith("tts."):
                continue
            stripped = name[len("tts.") :]
            if stripped.startswith("model."):
                backbone_weights.append((stripped[len("model.") :], tensor))
                continue
            if stripped == "head_code.0.parametrizations.weight.original0":
                head_g = tensor
                continue
            if stripped == "head_code.0.parametrizations.weight.original1":
                head_v = tensor
                continue
            target = stripped
            parameter = direct_params.get(target)
            if parameter is None:
                continue
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(target)

        for name in self.tts_model.load_weights(backbone_weights):
            loaded.add(f"tts_model.{name}")

        if head_g is None or head_v is None:
            raise ValueError("MiniCPM-o checkpoint is missing weight-norm Talker head parameters")
        restored = _restore_weight_norm_weight(head_g, head_v)
        self.head_code[0].weight.data.copy_(
            restored.to(
                device=self.head_code[0].weight.device,
                dtype=self.head_code[0].weight.dtype,
            )
        )
        loaded.add("head_code.0.weight")
        return loaded

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        del multimodal_embeddings
        # Decode tokens live in the codec table. Prefill overwrites these
        # embeddings in preprocess with emb_text + projected thinker hidden.
        return self.emb_code[0](input_ids)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
