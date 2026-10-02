# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CUDA multi-frame (K-step) decode for the MiniCPM-o Talker.

Stage 1's n-gram ``speculative_config`` (``num_speculative_tokens == K - 1``)
schedules K positions per request, and one ``execute_model`` produces up to K
codec frames, so the per-step host work is paid once per K frames. Unlike the
NPU runner, the codec head, the scheduler's codec ids and vLLM's ``Sampler``
stay as in the single-frame path:

* frame 0 is the ordinary single-frame step (the drafts repeat the last id, so
  preprocess embeds it at the span's first row);
* frames 1..K-1 replay the same forward after writing the previous frame's
  embedding into its row (attention is causal, so later stale rows are
  harmless), with the stop decision kept on device; the step reads its K ids
  back once and returns a rejection-sampler-shaped output (``-1`` padded).

A declined step is exactly one single-frame step: it returns frame 0's id as a
``(B, 1)`` output and the scheduler rejects the drafts. ``input_batch.vocab_size``
must cover the codec head (``ensure_codec_vocab``), since vLLM drops ids at or
above it from multi-token outputs.

The GPU runners reach this module through the model's
``multi_frame_decode_hook`` (``None`` on every other model and on NPU, whose
in-model K-step lives in ``platforms/npu/worker/talker_multiframe.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker.sampling_utils import sanitize_min_tokens_stop_ids

logger = init_logger(__name__)

_LOGGED_BLOCKS: set[str] = set()


def _log_block_once(reason: str, detail: str = "") -> None:
    """Log once per distinct reason (with its first numbers) why the loop did not engage."""
    if reason in _LOGGED_BLOCKS:
        return
    _LOGGED_BLOCKS.add(reason)
    suffix = f" ({detail})" if detail else ""
    logger.info("[minicpmo] multi-frame Talker decode not engaged: %s%s", reason, suffix)


def _block(reason: str, detail: str = "") -> int:
    _log_block_once(reason, detail)
    return 0


def applies(
    model: Any,
    model_kwargs_extra: dict[str, Any],
    *,
    require_sampled_frame: bool = True,
) -> int:
    """The uniform per-request query length of a pure Talker decode step, or 0.

    A prefill, a mixed step or non-uniform spans return 0 (the single-forward
    path). ``require_sampled_frame`` is the NPU contract (frame 0 embeds the
    model's recorded previous code); the CUDA runner passes ``False``. A
    multi-token step that is refused must not take the single-forward path;
    the runners raise on it (``is_multi_token_decode``).
    """
    if not getattr(model, "supports_multi_frame_decode", False):
        return 0
    spans = model_kwargs_extra.get("request_token_spans")
    infos = model_kwargs_extra.get("model_intermediate_buffer")
    if not spans or not infos or len(spans) != len(infos):
        return _block(
            "no request_token_spans/model_intermediate_buffer for this step",
            f"{len(spans or ())} spans, {len(infos or ())} buffers",
        )
    frames = int(spans[0][1]) - int(spans[0][0])
    if frames <= 1:
        return 0
    for start, end in spans:
        if int(end) - int(start) != frames:
            return _block(
                "request token spans are not uniform",
                f"frames {frames}, spans {[(int(s), int(e)) for s, e in spans]}",
            )
    for index, info in enumerate(infos):
        if not isinstance(info, dict):
            return _block("a request carries no intermediate buffer", f"request {index} of {len(infos)}")
        if bool(info.get("_omni_is_prefill", False)):
            return 0
        state = info.get("audio_state")
        if not isinstance(state, dict):
            return _block("a request carries no audio_state", f"request {info.get('request_id', index)}")
        if require_sampled_frame and int(state.get("step", 0)) <= 0:
            return _block("a request has not sampled a codec token yet", f"request {info.get('request_id', index)}")
    return frames


def is_multi_token_decode(model: Any, model_kwargs_extra: dict[str, Any]) -> bool:
    """True when this Talker step schedules several tokens for a decoding request."""
    if not getattr(model, "supports_multi_frame_decode", False):
        return False
    spans = model_kwargs_extra.get("request_token_spans")
    infos = model_kwargs_extra.get("model_intermediate_buffer")
    if not spans or not infos or len(spans) != len(infos):
        return False
    for (start, end), info in zip(spans, infos):
        if int(end) - int(start) <= 1:
            continue
        if isinstance(info, dict) and not bool(info.get("_omni_is_prefill", False)):
            return True
    return False


def drafts_this_step(runner: Any) -> int:
    """Frames the *next* step is scheduled for (``num_spec_tokens + 1``), or 0.

    Speculative tokens are vLLM V1's only way to reserve KV for several
    positions; the frames are generated sequentially, not verified.
    """
    if not getattr(getattr(runner, "model", None), "supports_multi_frame_decode", False):
        return 0
    num_spec = int(getattr(runner, "num_spec_tokens", 0) or 0)
    if num_spec <= 0:
        return 0
    return num_spec + 1


def constant_drafts(
    valid_sampled_token_ids: Any,
    frames: int,
    num_reqs: int,
    *,
    draft_token: int | None,
) -> list[list[int]]:
    """``frames - 1`` drafts per request -- for the whole batch, or for none of it.

    The loop needs uniform spans, so one request that sampled nothing this step
    means no drafts at all (an ordinary one-frame step). ``draft_token=None``
    repeats each request's last sampled id (the CUDA runner embeds it at frame 0).
    """
    rows: list[list[int]] = []
    for index in range(num_reqs):
        sampled = None
        if isinstance(valid_sampled_token_ids, list) and index < len(valid_sampled_token_ids):
            sampled = valid_sampled_token_ids[index]
        if not sampled:
            _log_block_once(
                "a request in this batch sampled nothing; no drafts this step",
                f"request {index} of {num_reqs}",
            )
            return [[] for _ in range(num_reqs)]
        token = int(sampled[-1]) if draft_token is None else int(draft_token)
        rows.append([token] * (frames - 1))
    return rows


_STASH_ATTR = "_talker_frames_sampler_output"
_BUILTIN_LOGITSPROCS = (MinTokensLogitsProcessor, LogitBiasLogitsProcessor, MinPLogitsProcessor)
_SAMPLER: Sampler | None = None


@dataclass(frozen=True)
class Decline:
    """Why a step runs one frame; logs dedupe on ``kind``, ``detail`` has numbers."""

    kind: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.kind} ({self.detail})" if self.detail else self.kind


_LOGGED: set[tuple[str, str]] = set()


def _log_once(site: str, decline: Decline) -> None:
    """Log each distinct decline once per site, with its first numbers."""
    key = (site, decline.kind)
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info("[minicpmo] CUDA multi-frame Talker decode: %s: %s", site, decline)


def _codec_sampler() -> Sampler:
    # Same default ``Sampler()`` the Talker's own sample() uses for frame 0.
    global _SAMPLER
    if _SAMPLER is None:
        _SAMPLER = Sampler()
    return _SAMPLER


def supports(model: Any) -> bool:
    return bool(getattr(model, "supports_multi_frame_decode", False)) and callable(
        getattr(model, "plan_codec_frames", None)
    )


def codec_vocab_size(model: Any) -> int:
    """Width of the Talker's codec head, or 0 when the model does not say."""
    return int(getattr(model, "codec_vocab_size", 0) or 0)


def ensure_codec_vocab(runner: Any) -> None:
    """Make stage 1 report the codec head's width (6562) as its vocab size.

    The remote-code ``MiniCPMTTSConfig`` has no ``vocab_size``, so vLLM reports
    0: ``RejectionSampler.parse_output`` then empties every multi-token row and
    ``InputBatch.add_request`` silently drops the stage's ``top_k``. Called at
    ``load_model``, before ``initialize_kv_cache`` rebuilds the ``InputBatch``
    from ``model_config``.
    """
    model = getattr(runner, "model", None)
    head = codec_vocab_size(model)
    if not supports(model) or head <= 0:
        return
    model_config = getattr(runner, "model_config", None)
    reported = int(model_config.get_vocab_size() or 0) if model_config is not None else head
    arch = getattr(model_config, "model_arch_config", None)
    if reported < head and arch is not None:
        arch.vocab_size = head
    batch = getattr(runner, "input_batch", None)
    batch_vocab = int(getattr(batch, "vocab_size", 0) or 0) if batch is not None else head
    if batch is not None and batch_vocab < head:
        batch.vocab_size = head
    if reported < head or batch_vocab < head:
        logger.info(
            "[minicpmo] Talker stage reported vocab_size %d (input batch %d); using the %d-wide codec head, "
            "so multi-token steps keep their ids and the stage's top_k applies",
            reported,
            batch_vocab,
            head,
        )


def _stop_ids(sampling_params: Any) -> set[int]:
    """The ids ``check_stop`` ends a request on (EOS and stop ids)."""
    ids = {int(t) for t in (getattr(sampling_params, "stop_token_ids", None) or ())}
    eos = getattr(sampling_params, "eos_token_id", None)
    if eos is not None:
        ids.add(int(eos))
    return ids


def vocab_decline(runner: Any, head_vocab: int) -> Decline | None:
    """Decline when ``parse_output`` would drop the frames (batch vocab narrower than the head)."""
    batch_vocab = int(getattr(runner.input_batch, "vocab_size", 0) or 0)
    if head_vocab > 0 and batch_vocab < head_vocab:
        return Decline(
            "input batch vocab is narrower than the codec head",
            f"input batch vocab {batch_vocab}, codec head {head_vocab}",
        )
    return None


def ineligible_reason(runner: Any, eos_token_id: int) -> Decline | None:
    """Why this batch cannot run several frames per step, or None."""
    if getattr(runner, "use_async_scheduling", False):
        # The drafts are this step's last ids and the accepted count must be
        # known before the next step is scheduled.
        return Decline("async scheduling is on", "stage 1 needs async_scheduling: false")
    batch = runner.input_batch
    md = batch.sampling_metadata
    if md.max_num_logprobs is not None or md.logprob_token_ids:
        return Decline("logprobs requested", f"max_num_logprobs {md.max_num_logprobs}")
    if md.bad_words_token_ids or md.allowed_token_ids_mask is not None:
        return Decline("bad words / allowed token ids")
    holder = getattr(md, "thinking_budget_state_holder", None)
    if holder is not None and holder.has_tracked_requests():
        return Decline("thinking budget")
    custom = [type(proc).__name__ for proc in md.logitsprocs.all if not isinstance(proc, _BUILTIN_LOGITSPROCS)]
    if custom:
        return Decline("custom logits processor", ", ".join(custom))
    for req_id in batch.req_ids:
        req_state = runner.requests.get(req_id)
        params = getattr(req_state, "sampling_params", None)
        if params is None:
            return Decline("request without sampling params", f"request {req_id}")
        if params.frequency_penalty or params.presence_penalty:
            return Decline(
                "frequency/presence penalty",
                f"request {req_id}: frequency {params.frequency_penalty}, presence {params.presence_penalty}",
            )
        if params.structured_outputs is not None:
            return Decline("structured output", f"request {req_id}")
        if getattr(params, "repetition_detection", None) is not None:
            return Decline("repetition detection", f"request {req_id}")
        stops = _stop_ids(params)
        if eos_token_id not in stops:
            return Decline(
                "codec EOS is not a stop id",
                f"request {req_id}: codec EOS {eos_token_id}, stop ids {sorted(stops)}",
            )
    return None


def draft_decline(runner: Any, model: Any) -> Decline | None:
    """What ``maybe_run`` would decline on that is known before drafting (a drafted
    step forwards K rows per request whether or not the loop runs)."""
    if not bool(getattr(model, "requires_request_sample_eligibility", False)):
        # The runner builds request_max_tokens_remaining only for such models.
        return Decline("no per-request token budget", "the model does not declare requires_request_sample_eligibility")
    return vocab_decline(runner, codec_vocab_size(model)) or ineligible_reason(runner, int(model.codec_eos_token_id))


def propose_drafts(runner: Any, sampled_token_ids: Any) -> list[list[int]] | None:
    """The Talker's next-step drafts (last id K-1 times, whole batch or none), or
    None for any other stage."""
    model: Any = getattr(runner, "model", None)
    if not supports(model):
        return None
    frames = drafts_this_step(runner)
    num_reqs = runner.input_batch.num_reqs
    if frames <= 1:
        return None
    decline = draft_decline(runner, model)
    if decline is None and not isinstance(sampled_token_ids, list):
        decline = Decline("sampled ids are not a list", type(sampled_token_ids).__name__)
    if decline is not None:
        _log_once("no drafts", decline)
        return [[] for _ in range(num_reqs)]
    return constant_drafts(sampled_token_ids, frames, num_reqs, draft_token=None)


def take_sampler_output(runner: Any) -> SamplerOutput | None:
    output = getattr(runner, _STASH_ATTR, None)
    setattr(runner, _STASH_ATTR, None)
    return output


@dataclass
class FrameControls:
    """Device tensors driving frames 1..K-1 (``(K, B)`` rows are per frame)."""

    eos_token_id: int
    force_eos: torch.Tensor  # (K, B) bool
    mask_eos: torch.Tensor  # (K, B) bool
    allowed: torch.Tensor  # (K, B) bool: frame k is inside the token budget
    min_tokens: torch.Tensor  # (K, B) bool: vLLM min_tokens masks stop ids
    min_tokens_stop: torch.Tensor | None  # (B, V) bool: the stop ids it masks
    stop_ids: torch.Tensor  # (B, S) long, -1 padded: check_stop's ids
    window: torch.Tensor  # (B, W) long, right-aligned, padded with V
    penalties: torch.Tensor | None  # (B,) repetition penalty, None when off
    forced_row: torch.Tensor  # (V,) float: -inf except 0.0 at EOS
    eos_col: torch.Tensor  # (V,) bool


def _upload(tensors: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    """One pinned host-to-device copy for all of a step's small int tensors."""
    flat = torch.cat([t.reshape(-1).to(torch.long) for t in tensors])
    if device.type == "cuda":
        flat = flat.pin_memory().to(device, non_blocking=True)
    else:
        flat = flat.to(device)
    out, offset = [], 0
    for t in tensors:
        n = t.numel()
        out.append(flat[offset : offset + n].view(t.shape))
        offset += n
    return out


def build_controls(
    plan: Any,
    *,
    frames: int,
    vocab_size: int,
    budgets: list[int],
    stop_ids: list[set[int]],
    min_tokens_state: dict[int, tuple[Any, ...]],
    penalties: torch.Tensor | None,
    device: torch.device,
    extra: list[torch.Tensor] | None = None,
) -> tuple[FrameControls, list[torch.Tensor]]:
    """Build the loop controls from host state; ``extra`` rides the same upload."""
    from vllm_omni.model_executor.models.minicpmo_4_5.talker_frame_plan import CODEC_PENALTY_WINDOW as WINDOW

    num_reqs = len(budgets)
    force = torch.tensor(plan.force_eos, dtype=torch.bool)
    mask = torch.tensor(plan.mask_eos, dtype=torch.bool)
    allowed = torch.tensor([[k < int(b) for b in budgets] for k in range(frames)], dtype=torch.bool)
    # vLLM's MinTokensLogitsProcessor state: batch index -> (min_tokens,
    # output ids, stop ids, structured). Its mask is computed once per step
    # from len(output ids); frame k has k more outputs than that.
    min_mask = torch.zeros((frames, num_reqs), dtype=torch.bool)
    pairs: list[tuple[int, int]] = []
    for index, (min_toks, out_ids, stop_tok_ids, _) in min_tokens_state.items():
        if index >= num_reqs:
            continue
        for k in range(frames):
            min_mask[k, index] = len(out_ids) + k < int(min_toks)
        pairs.extend((index, int(t)) for t in stop_tok_ids if 0 <= int(t) < vocab_size)
    width = max(1, max((len(s) for s in stop_ids), default=1))
    stop_table = torch.full((num_reqs, width), -1, dtype=torch.long)
    for index, ids in enumerate(stop_ids):
        kept = sorted(t for t in ids if 0 <= t < vocab_size)
        stop_table[index, : len(kept)] = torch.tensor(kept, dtype=torch.long)
    window = torch.full((num_reqs, WINDOW), vocab_size, dtype=torch.long)
    for index, codes in enumerate(plan.recent_codes):
        codes = codes[-WINDOW:]
        if codes:
            window[index, WINDOW - len(codes) :] = torch.tensor(codes, dtype=torch.long)
    pair_tensor = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
    host = [force, mask, allowed, min_mask, stop_table, window, pair_tensor, *(extra or [])]
    dev = _upload(host, device)
    force_d, mask_d, allowed_d, min_d, stop_d, window_d, pairs_d = dev[:7]
    min_stop = None
    if pairs:
        min_stop = torch.zeros((num_reqs, vocab_size), dtype=torch.bool, device=device)
        min_stop[pairs_d[:, 0], pairs_d[:, 1]] = True
    eos = int(plan.eos_token_id)
    eos_col = torch.arange(vocab_size, device=device) == eos
    forced_row = torch.where(
        eos_col,
        torch.zeros((), dtype=torch.float32, device=device),
        torch.full((), float("-inf"), dtype=torch.float32, device=device),
    )
    controls = FrameControls(
        eos_token_id=eos,
        force_eos=force_d.bool(),
        mask_eos=mask_d.bool(),
        allowed=allowed_d.bool(),
        min_tokens=min_d.bool(),
        min_tokens_stop=min_stop,
        stop_ids=stop_d,
        window=window_d,
        penalties=penalties,
        forced_row=forced_row,
        eos_col=eos_col,
    )
    return controls, dev[7:]


def adjust_logits(logits: torch.Tensor, k: int, window: torch.Tensor, c: FrameControls) -> torch.Tensor:
    """Frame k's logits as the single-frame path shapes them before ``Sampler``:
    EOS force/mask, the windowed repetition penalty (same ops as
    ``_apply_batched_repetition_penalty``, bit-identical), then min_tokens."""
    neg_inf = float("-inf")
    logits = torch.where(c.force_eos[k].unsqueeze(1), c.forced_row, logits)
    logits = torch.where(c.mask_eos[k].unsqueeze(1) & c.eos_col, neg_inf, logits)
    if c.penalties is not None:
        num_reqs, vocab = logits.shape
        counts = torch.zeros((num_reqs, vocab + 1), dtype=torch.long, device=logits.device)
        counts.scatter_add_(1, window, torch.ones_like(window))
        alpha = torch.pow(c.penalties.to(logits.dtype).unsqueeze(1), counts[:, :vocab].to(logits.dtype))
        logits = torch.where(logits < 0, logits * alpha, logits / alpha)
    if c.min_tokens_stop is not None:
        logits = logits.masked_fill(c.min_tokens_stop & c.min_tokens[k].unsqueeze(1), neg_inf)
    return logits


def decode_frames(
    first: torch.Tensor,
    frames: int,
    controls: FrameControls,
    *,
    embed: Callable[[torch.Tensor], torch.Tensor],
    forward: Callable[[int, torch.Tensor], torch.Tensor],
    codec_logits: Callable[[torch.Tensor], torch.Tensor],
    sample: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run frames 1..frames-1 after frame 0 sampled ``first`` ((B,) long).

    Returns ``(sampled, emitted)``, both ``(B, frames)``; a frame is emitted
    until a stop id or the token budget, where ``check_stop`` would end it.
    Nothing here reads the device.
    """
    c = controls
    sampled = [first]
    emitted = [torch.ones_like(first, dtype=torch.bool)]
    window = c.window
    prev = first
    for k in range(1, frames):
        stopped = (prev.unsqueeze(1) == c.stop_ids).any(dim=1)
        emitted.append(emitted[-1] & ~stopped & c.allowed[k])
        window = torch.cat([window[:, 1:], prev.unsqueeze(1)], dim=1)
        logits = adjust_logits(codec_logits(forward(k, embed(prev))), k, window, c)
        prev = torch.where(c.force_eos[k], c.eos_token_id, sample(logits).to(torch.long))
        sampled.append(prev)
    return torch.stack(sampled, dim=1), torch.stack(emitted, dim=1)


def frame_sampling_metadata(md: Any) -> Any:
    """Frames 1..K-1 metadata: min_tokens moves to ``adjust_logits`` and vLLM's
    penalty pass is dropped (the Talker scores its own window; eligibility
    requires zero frequency/presence penalties)."""
    procs = LogitsProcessors(p for p in md.logitsprocs.all if not isinstance(p, MinTokensLogitsProcessor))
    return replace(md, no_penalties=True, logitsprocs=procs)


def sample_frame(logits: torch.Tensor, md: Any) -> torch.Tensor:
    return _codec_sampler()(logits, md).sampled_token_ids[:, 0]


def frame_budgets(runner: Any, model_kwargs_extra: dict[str, Any], num_reqs: int) -> tuple[list[int], Decline | None]:
    """Frames each request may still emit (``max_tokens``, else the context)."""
    given = model_kwargs_extra.get("request_max_tokens_remaining")
    if given is None or len(given) != num_reqs:
        got = "none" if given is None else len(given)
        return [], Decline("no per-request token budget", f"{got} budgets for {num_reqs} requests")
    budgets: list[int] = []
    for req_id, budget in zip(runner.input_batch.req_ids[:num_reqs], given):
        if budget is None:
            req_state = runner.requests.get(req_id)
            num_tokens = getattr(req_state, "num_tokens", None)
            if num_tokens is None:
                return [], Decline("no per-request token budget", f"request {req_id} has no max_tokens or state")
            budget = max(int(runner.max_model_len) - int(num_tokens), 0)
        budgets.append(int(budget))
    return budgets, None


def _hidden(output: Any) -> torch.Tensor:
    if isinstance(output, OmniOutput):
        return output.text_hidden_states
    if isinstance(output, tuple):
        return output[0]
    return output


def maybe_run(
    runner: Any,
    model_output: Any,
    *,
    run_model: Callable[[], Any],
    inputs_embeds: torch.Tensor | None,
    model_kwargs_extra: dict[str, Any],
) -> Any:
    """Run a Talker K-frame step on CUDA; any other step returns unchanged."""
    setattr(runner, _STASH_ATTR, None)
    model: Any = getattr(runner, "model", None)
    if not supports(model) or drafts_this_step(runner) <= 1:
        return model_output
    frames = applies(model, model_kwargs_extra, require_sampled_frame=False)
    if frames <= 1:
        if is_multi_token_decode(model, model_kwargs_extra):
            spans = [(int(s), int(e)) for s, e in model_kwargs_extra.get("request_token_spans") or ()]
            raise RuntimeError(
                "MiniCPM-o scheduled a multi-token Talker decode step the CUDA "
                f"multi-frame loop cannot run (request token spans {spans}); "
                "see the '[minicpmo]' log for the reason"
            )
        return model_output
    if not isinstance(model_output, OmniOutput) or inputs_embeds is None:
        raise RuntimeError("MiniCPM-o CUDA multi-frame decode needs the Talker's OmniOutput and inputs_embeds")

    spans = [(int(s), int(e)) for s, e in model_kwargs_extra["request_token_spans"]]
    infos = model_kwargs_extra["model_intermediate_buffer"]
    num_reqs = len(spans)
    device = inputs_embeds.device
    hidden = _hidden(model_output)
    rows_host = torch.tensor([[s + k for s, _ in spans] for k in range(frames)], dtype=torch.long)

    # Frame 0: the single-frame step, verbatim.
    rows0 = rows_host[0].to(device, non_blocking=True)
    logits0 = model.compute_logits(hidden.index_select(0, rows0))
    md = runner.input_batch.sampling_metadata
    vocab = int(logits0.shape[-1])
    sanitize_min_tokens_stop_ids(md.logitsprocs, vocab)
    first = runner._sample(logits0, None).sampled_token_ids[:, 0].to(torch.long)

    eos_id = int(model.codec_eos_token_id)
    decline = vocab_decline(runner, vocab) or ineligible_reason(runner, eos_id)
    budgets: list[int] = []
    if decline is None:
        budgets, decline = frame_budgets(runner, model_kwargs_extra, num_reqs)
    if decline is not None:
        _log_once("one frame this step", decline)
        # Frame 0 is the whole step, as (B, 1): the one-token bookkeeping branch
        # skips parse_output's vocab filter and the scheduler rejects the drafts.
        # An empty row would roll the span back and re-send the codec id.
        output = SamplerOutput(sampled_token_ids=first.to(torch.int32).unsqueeze(1), logprobs_tensors=None)
        setattr(runner, _STASH_ATTR, output)
        return model_output

    plan = model.plan_codec_frames(infos, frames)
    req_ids = runner.input_batch.req_ids[:num_reqs]
    min_state: dict[int, tuple[Any, ...]] = {}
    for proc in md.logitsprocs.non_argmax_invariant:
        if isinstance(proc, MinTokensLogitsProcessor):
            min_state.update(proc.min_toks)
    penalties = None if md.no_penalties else md.repetition_penalties[:num_reqs]
    controls, (rows,) = build_controls(
        plan,
        frames=frames,
        vocab_size=vocab,
        budgets=budgets,
        stop_ids=[_stop_ids(runner.requests[r].sampling_params) for r in req_ids],
        min_tokens_state=min_state,
        penalties=penalties,
        device=device,
        extra=[rows_host],
    )
    frame_md = frame_sampling_metadata(md)
    last = [model_output]

    def forward(k: int, embeds: torch.Tensor) -> torch.Tensor:
        inputs_embeds.index_copy_(0, rows[k], embeds.to(inputs_embeds.dtype))
        last[0] = run_model()
        return _hidden(last[0]).index_select(0, rows[k])

    def replay() -> torch.Tensor:
        last[0] = run_model()
        return _hidden(last[0])

    # Opt-in model hook: the model may run frames 1..K-1 itself (MiniCPM-o
    # replays captured per-frame tails) and must then return exactly what
    # decode_frames would; None, or no hook, keeps the loop below.
    model_frames = getattr(model, "kstep_decode_frames", None)
    graphed = None
    if callable(model_frames):
        graphed = model_frames(
            first=first,
            frames=frames,
            controls=controls,
            rows=rows,
            inputs_embeds=inputs_embeds,
            hidden=hidden,
            replay=replay,
            embed=model.embed_input_ids,
            codec_logits=model.compute_logits,
            sampler=_codec_sampler(),
            sampling_metadata=frame_md,
        )
    if graphed is not None:
        sampled, emitted = graphed
    else:
        sampled, emitted = decode_frames(
            first,
            frames,
            controls,
            embed=model.embed_input_ids,
            forward=forward,
            codec_logits=model.compute_logits,
            sample=lambda logits: sample_frame(logits, frame_md),
        )
    ids = torch.where(emitted, sampled, -1)
    ids_host = ids.cpu().tolist()  # the step's one device read
    forwarded = []
    for row in ids_host:
        n = sum(1 for t in row if t >= 0)
        forwarded.append(row[: n - 1])
    flags = model.commit_codec_frames(plan, forwarded)

    multimodal: Any = model_output.multimodal_outputs
    audio = multimodal["codes"]["audio"]
    finished = multimodal["meta"]["finished"]
    for index, codes in enumerate(forwarded):
        if not codes:
            continue
        tail = sampled[index, : len(codes)].reshape(-1, 1).to(device=audio[index].device)
        audio[index] = torch.cat([audio[index].to(torch.long), tail], dim=0)
        if flags[index]:
            finished[index] = torch.tensor(True, dtype=torch.bool)
    setattr(runner, _STASH_ATTR, SamplerOutput(sampled_token_ids=ids.to(torch.int32), logprobs_tensors=None))
    return model_output._replace(text_hidden_states=_hidden(last[0]))
