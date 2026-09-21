"""Several codec frames inside one stage-1 ``execute_model``.

Measured on A3 / 910C (``server-env/design/RTF_CEILING_20260828.md``): one
MiniCPM-o Talker decode step is ~2.9 ms of vLLM host work wrapped around a
0.83 ms device forward, for a 190M-parameter model that emits exactly one codec
frame. Over ~118 frames per request that host work is ~340 ms of the ~540 ms
that follows TTFT -- by a wide margin the largest single term left in the score.

None of it is per *frame*. The scheduler step, input preparation, attention
metadata, sampling, output assembly and engine-core IPC are per *step*, and a
step could just as well carry K frames. This module runs them.

## How a frame advances without the host

The step is scheduled with K query positions per request (see
``_apply_minicpmo_talker_multiframe_default``). vLLM prepares all K of them --
positions, KV slots, block table, sequence lengths -- and the captured graph is
a K-token uniform decode. What vLLM cannot do is fill in the *inputs*: frame
k+1's embedding is the codec token sampled at frame k, so the K positions are
sequential, not speculative.

So the loop replays the same captured K-token graph K times. Before replay k it
writes frame k's embedding into row k of the persistent ``inputs_embeds``
buffer the graph captured; after replay k it samples frame k from row k of the
hidden states. Rows above k still hold stale embeddings during replay k, and
their KV is stale too -- but the attention bias is causal, so row k never reads
them, and the next replay overwrites them with the real thing. After the K'th
replay every row and every KV entry is what a K-step decode would have written.

A replay of the 20-layer forward is 0.83 ms of device and 8 us of host
(``server-env/tools/dispatch_cost.py``), so K replays cost what K frames cost
and the ~2.9 ms is paid once. At K=4 that is a cadence of ~1.9 ms against ~4.0.

Nothing in the loop touches the host: the embedding lookup, the codec sample
(the captured codec-sampling step), the stop row and the emitted delta are all device
tensors enqueued in order on one stream, and the runner's existing coalesced
D2H carries them out at the end of the step.

## What makes it correct

* **Stop is still per frame.** ``make_omni_output`` evaluates the codec EOS and
  the token limit on every frame exactly as it does today. A request that ends
  at frame j reports its stop row there; frames j+1..K-1 take the "already
  finished" early return, emit an empty delta and report a stop row too. vLLM's
  rejection sampler sees the first stop and truncates the request to j+1
  tokens, which is the same sequence a one-frame-per-step run would produce.
* **The emitted codes are unchanged.** The extra frames after a stop contribute
  nothing: their deltas are empty or the ``-1`` sentinel the connector already
  drops (``_codec_scalars``). So the audio is expected to be bit-identical, not
  merely close -- which is a far stronger check than the CER gate.

Off with ``VLLM_OMNI_MINICPMO_TALKER_FRAMES=1``.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_LOGGED_ENGAGE = False

# ---------------------------------------------------------------------------
# K 步逐帧计时（默认关，关掉时 run() 里一个 timer 都不取）
#
#   VLLM_OMNI_MINICPMO_KSTEP_PROF=1        打开
#   VLLM_OMNI_MINICPMO_KSTEP_PROF_STEPS=3  前 N 个 step 逐帧打点并做设备同步取样
#   VLLM_OMNI_MINICPMO_KSTEP_PROF_EVERY=25 每 N 个 step 打一行累计摘要
#
# 要回答的问题只有一个：一步 8 帧的墙钟时间，到底花在 host 入队（replay/after_forward）
# 还是花在设备执行上。replay/after_forward 是入队耗时（异步），synced 是在其后加一次
# 设备同步测到的真实帧耗时 —— 两者相差多少，就是设备侧的账。
# 诊断代码永不许抛异常：任何异常只把本 profiling 永久关掉并记一条 warning。
# ---------------------------------------------------------------------------
_PROF_ENV = "VLLM_OMNI_MINICPMO_KSTEP_PROF"
_PROF_STEPS_ENV = "VLLM_OMNI_MINICPMO_KSTEP_PROF_STEPS"
_PROF_EVERY_ENV = "VLLM_OMNI_MINICPMO_KSTEP_PROF_EVERY"
_PROF_OFF = ("0", "false", "no", "off")
_PROF_STEPS_DEFAULT = 3
_PROF_EVERY_DEFAULT = 25


class _StepProf:
    """One step's timers. Never raises: profiling is best-effort by contract."""

    __slots__ = ("frames", "replay_ms", "after_ms", "detail")

    def __init__(self, frames: int, *, detail: bool) -> None:
        self.frames = frames
        self.replay_ms = 0.0
        self.after_ms = 0.0
        self.detail = detail


def prof_enabled() -> bool:
    """Off unless the env explicitly turns it on (``1``/``true``/``yes``/``on``)."""
    try:
        raw = os.environ.get(_PROF_ENV, "").strip().lower()
        return bool(raw) and raw not in _PROF_OFF
    except Exception:  # pragma: no cover - env is a dict lookup, but be total
        return False


def _prof_steps() -> int:
    try:
        raw = os.environ.get(_PROF_STEPS_ENV, "").strip()
        steps = int(raw) if raw else _PROF_STEPS_DEFAULT
    except Exception:
        steps = _PROF_STEPS_DEFAULT
    return max(steps, 0)


def _prof_every() -> int:
    try:
        raw = os.environ.get(_PROF_EVERY_ENV, "").strip()
        every = int(raw) if raw else _PROF_EVERY_DEFAULT
    except Exception:
        every = _PROF_EVERY_DEFAULT
    return max(every, 1)


def _prof_sync() -> None:
    """Best-effort device drain; no-op on a build without the op."""
    for fn in (
        getattr(getattr(torch, "npu", None), "synchronize", None),
        getattr(getattr(torch, "cuda", None), "synchronize", None),
    ):
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                return


def _prof_frame_line(step: int, frame: int, replay_ms: float, after_ms: float, synced_ms: float) -> None:
    try:
        logger.info(
            "[kstep-prof] step=%d frame=%d replay_ms=%.3f after_forward_ms=%.3f synced_frame_ms=%s",
            step,
            frame,
            replay_ms,
            after_ms,
            f"{synced_ms:.3f}" if synced_ms >= 0.0 else "n/a",
        )
    except Exception:
        pass


def _prof_summary(steps: int, frames_total: int, last_frames: int, *, detail: bool) -> None:
    """累计一行：每步/每帧墙钟，以及 host 入队与设备执行各自的账。"""
    try:
        per_step = _PROF_TOTAL_MS / steps if steps else 0.0
        per_frame = _PROF_TOTAL_MS / frames_total if frames_total else 0.0
        replay_per_frame = _PROF_REPLAY_MS / frames_total if frames_total else 0.0
        after_per_frame = _PROF_AFTER_MS / frames_total if frames_total else 0.0
        synced = (
            f" synced_frame_ms={_PROF_SYNCED_MS / _PROF_SYNCED_FRAMES:.3f}"
            f" (n={_PROF_SYNCED_FRAMES})"
            if _PROF_SYNCED_FRAMES
            else ""
        )
        logger.info(
            "[kstep-prof] steps=%d frames=%d last_step_frames=%d step_ms=%.2f frame_ms=%.3f "
            "replay_enqueue_ms_per_frame=%.3f after_forward_ms_per_frame=%.3f (totals: %.3f/%.3f)%s%s",
            steps,
            frames_total,
            last_frames,
            per_step,
            per_frame,
            replay_per_frame,
            after_per_frame,
            _PROF_REPLAY_MS,
            _PROF_AFTER_MS,
            synced,
            "  [per-frame lines above]" if detail else "",
        )
    except Exception:
        pass


_PROF_WARNED = False
_PROF_STEPS = 0
_PROF_FRAMES = 0
_PROF_REPLAY_MS = 0.0
_PROF_AFTER_MS = 0.0
_PROF_TOTAL_MS = 0.0
_PROF_SYNCED_MS = 0.0
_PROF_SYNCED_FRAMES = 0


def _prof_disabled(reason: str) -> None:
    """Turn the profiling off for the rest of the process, once."""
    global _PROF_WARNED
    if _PROF_WARNED:
        return
    _PROF_WARNED = True
    try:
        logger.warning("[kstep-prof] disabled after failure: %s", reason)
    except Exception:
        pass
    os.environ[_PROF_ENV] = "0"


def _prof_step_begin(frames: int) -> "_StepProf | None":
    """``None`` when off -- and then ``run()`` takes no timer at all."""
    try:
        if not prof_enabled():
            return None
        return _StepProf(frames, detail=_PROF_STEPS < _prof_steps())
    except Exception:
        _prof_disabled("step_begin")
        return None


def _prof_frame(prof: "_StepProf", frame: int, t0: float, t1: float, t2: float, t3: float) -> None:
    """Book one replay. ``replay``/``after`` are enqueue times; ``synced`` is the
    frame's real wall clock (drained after the replay), and the gap between them
    is the device side of the account."""
    try:
        global _PROF_SYNCED_MS, _PROF_SYNCED_FRAMES
        replay_ms = (t1 - t0) * 1000.0
        after_ms = (t2 - t1) * 1000.0
        prof.replay_ms += replay_ms
        prof.after_ms += after_ms
        synced_ms = -1.0
        if prof.detail and t3:
            synced_ms = (t3 - t0) * 1000.0
            _PROF_SYNCED_MS += synced_ms
            _PROF_SYNCED_FRAMES += 1
        if prof.detail:
            _prof_frame_line(_PROF_STEPS + 1, frame, replay_ms, after_ms, synced_ms)
    except Exception:
        _prof_disabled("frame")


def _prof_step_end(prof: "_StepProf | None", step_start: float) -> None:
    if prof is None:
        return
    try:
        global _PROF_STEPS, _PROF_FRAMES, _PROF_REPLAY_MS, _PROF_AFTER_MS, _PROF_TOTAL_MS
        _PROF_STEPS += 1
        _PROF_FRAMES += prof.frames
        _PROF_REPLAY_MS += prof.replay_ms
        _PROF_AFTER_MS += prof.after_ms
        _PROF_TOTAL_MS += (perf_counter() - step_start) * 1000.0
        if prof.detail or _PROF_STEPS % _prof_every() == 0:
            _prof_summary(_PROF_STEPS, _PROF_FRAMES, prof.frames, detail=prof.detail)
    except Exception:
        _prof_disabled("step_end")
# ---------------------------------------------------------------------------
# K 步 stop 取证（默认关）
#
#   VLLM_OMNI_MINICPMO_KSTEP_STOP_TRACE=1
#
# 只回答一个问题：一步 K 帧里，模型到底有没有产出 stop 行。有 stop 行而请求
# 仍跑到 max_tokens，丢在 vLLM 的 spec 记账侧；一行都没有，丢在多帧的采样/状态
# 传递侧。前 50 步各打一行，之后静默。诊断组件永不抛异常。
# ---------------------------------------------------------------------------
_STOP_TRACE_ENV = "VLLM_OMNI_MINICPMO_KSTEP_STOP_TRACE"
_STOP_TRACE_STEPS = 50
_STOP_TRACE_COUNT = 0


def stop_trace_enabled() -> bool:
    """Off unless the env explicitly turns it on (``1``/``true``/``yes``/``on``)."""
    try:
        raw = os.environ.get(_STOP_TRACE_ENV, "").strip().lower()
        return bool(raw) and raw not in _PROF_OFF
    except Exception:  # pragma: no cover - env is a dict lookup, but be total
        return False


def _trace_stop_rows(frame_stop_logits: list[torch.Tensor]) -> None:
    """每步一行：每个请求的 K 帧里各有几帧的 stop 行是 stop。"""
    global _STOP_TRACE_COUNT
    try:
        if _STOP_TRACE_COUNT >= _STOP_TRACE_STEPS:
            return
        _STOP_TRACE_COUNT += 1
        flags = torch.stack([row.argmax(dim=-1) for row in frame_stop_logits], dim=1)
        logger.info(
            "[kstep-stop] step=%d frames=%d stops_per_req=%s",
            _STOP_TRACE_COUNT,
            len(frame_stop_logits),
            flags.sum(dim=1).cpu().tolist(),
        )
    except Exception as exc:
        logger.warning("[kstep-stop] trace disabled after failure: %r", exc)
        os.environ[_STOP_TRACE_ENV] = "0"


_STOP_KB_COUNT = 0


def trace_kstep_bookkeeping(runner: Any, valid_sampled_token_ids: Any, logits: Any) -> None:
    """[kstep-stop] 第二层：stop 走完 spec 记账之后还剩什么。

    两条事实一起打，一次就能把"stop 去哪了"钉死在某一层：

    * ``carried`` -- rejection sampler 交给请求的 token 序列（前 4 个请求）。
      里面有 1 说明 stop 进了请求自己的 token 表，后面就只剩 engine 侧的结束判定；
      一排 0 而没有 1 说明 stop 在到这里之前就没了。
    * ``min_toks`` -- vLLM 的 MinTokens 处理器里还有哪些请求在 mask 名单上，
      以及它看到的 output 长度（``(index, min_tokens, len(output_token_ids),
      sorted(stop_ids))``）。**只要某个请求还在这张名单上，它的 stop 列就是
      -inf**；如果 ``len(output_token_ids)`` 始终不涨，这张名单就永远不会被
      清掉，请求也就永远停不下来。

    诊断组件永不抛异常：出错只关掉自己。
    """
    global _STOP_KB_COUNT
    try:
        if _STOP_KB_COUNT >= _STOP_TRACE_STEPS:
            return
        _STOP_KB_COUNT += 1
        if isinstance(valid_sampled_token_ids, list):
            carried: Any = [list(row) for row in valid_sampled_token_ids[:4]]
        else:
            carried = "<%s>" % type(valid_sampled_token_ids).__name__

        masks: list[Any] = []
        sampling_metadata = getattr(getattr(runner, "input_batch", None), "sampling_metadata", None)
        for proc in getattr(getattr(sampling_metadata, "logitsprocs", None), "non_argmax_invariant", None) or []:
            min_toks = getattr(proc, "min_toks", None)
            if isinstance(min_toks, dict) and min_toks:
                masks.append(
                    [
                        (index, int(min_tok), len(out_ids), sorted(stop_ids))
                        for index, (min_tok, out_ids, stop_ids) in list(min_toks.items())[:4]
                    ]
                )
        logger.info(
            "[kstep-stop] carried=%s min_toks=%s logits=%s",
            carried,
            masks,
            tuple(getattr(logits, "shape", ()) or ()),
        )
    except Exception as exc:
        logger.warning("[kstep-stop] bookkeeping trace disabled after failure: %r", exc)
        os.environ[_STOP_TRACE_ENV] = "0"


def neutralize_kstep_min_tokens(logitsprocs: Any) -> None:
    """K 步下让 vLLM 层的 ``min_tokens`` 不再 censoring 停止信号。

    yaml 的 ``min_tokens: 50`` 在 K 步下是通过 **把 stop token（id 1）的 logit
    置成 -inf** 实现的（``MinTokensLogitsProcessor``）。它自己的解除条件是
    ``len(output_token_ids) >= min_tokens``，而这个长度归 vLLM 的 spec 记账管；
    只要那个计数没有推进到位，请求唯一的停止信号就被永久 mask，
    ``finished_reason`` 只能是 ``length``（910C 现场：stage1 全 length、0 stop）。

    真正的"最小帧数"保护不在这里：``talker_codec_sample.greedy_codec_sample``
    用 ``state.step < min_tokens`` 把 codec EOS 本身压成 -inf，用的是模型自己
    的帧计数，与 vLLM 的 token 记账无关。所以这一层是重复的、且在 K 步下会
    把停止信号一起吃掉 —— 清空它的名单，把最小长度交还给模型内的那道保护。

    ``min_toks`` 每个 step 都会被 ``update_state`` 重新登记，所以本函数必须
    每步调用（调用点在 runner 的采样前）。永不抛异常。
    """
    try:
        for proc in getattr(logitsprocs, "non_argmax_invariant", None) or []:
            if not type(proc).__name__.endswith("MinTokensLogitsProcessor"):
                # Suffix rather than equality: the processor can arrive
                # subclassed or wrapped (a platform patch, a profiling shim),
                # and a mismatch here is silent -- the mask stays on and the
                # only stop signal a K-step request has is censored forever.
                continue
            min_toks = getattr(proc, "min_toks", None)
            if isinstance(min_toks, dict) and min_toks:
                min_toks.clear()
    except Exception as exc:
        logger.warning("[minicpmo] K-step min_tokens neutralization skipped: %r", exc)


_LOGGED_BLOCK: str | None = None
_LOGGED_NARROW = False
_LOGGED_NARROW_BLOCK: str | None = None

_NARROW_ENV = "VLLM_OMNI_MINICPMO_NARROW_REPLAY"
_NARROW_OFF = frozenset({"0", "off", "false", "no"})
_NARROW_ON = frozenset({"1", "on", "true", "yes"})
# SoC families whose one-query capture is known to fault. The 910_93 part
# trips a vector-core error (acl 507035) inside the packaged codec operator
# while that graph is being captured, so the narrow path stays off there
# unless an operator opts in explicitly -- see narrow_replay_enabled.
# "ascend910c" guards the device-name fallback: on some A3 containers the
# driver reports the part as "Ascend910C" rather than its 910_93 SoC code,
# and an unblocked unknown would let the capture brick the deployment.
# "ascend910b" is fail-safe rather than observed: no K-step attempt there
# ever reached a capture (all 8 combos faulted earlier, inside the reject
# kernel), so the one-query capture is unverified on that family.
_NARROW_BLOCKED_SOC_PREFIXES = ("ascend910_93", "ascend910c", "ascend910b")
_NARROW_SOC_LOGGED: str | None = None


def _narrow_soc_allows() -> bool:
    """Whether this part may capture the one-query graph by default.

    An explicit ``VLLM_OMNI_MINICPMO_NARROW_REPLAY=1/on`` always wins, and an
    explicit ``0/off`` always loses. With no setting the answer depends on the
    part: the environment is consulted first (the same variables the deploy
    config reads, so both sides agree), and the device name is the fallback for
    a worker launched without them.
    """
    global _NARROW_SOC_LOGGED

    raw = os.environ.get(_NARROW_ENV, "").strip().lower()
    if raw in _NARROW_ON:
        return True
    if raw in _NARROW_OFF:
        return False

    name = ""
    for variable in ("SOC_VERSION", "ASCEND_SOC_VERSION"):
        value = os.environ.get(variable, "").strip().lower()
        if value:
            name = value
            break
    if not name:
        try:
            import torch_npu

            name = str(torch_npu.npu.get_device_name(torch_npu.npu.current_device())).lower()
        except Exception:
            name = ""
    allows = True if not name else not name.startswith(_NARROW_BLOCKED_SOC_PREFIXES)
    if _NARROW_SOC_LOGGED != name:
        _NARROW_SOC_LOGGED = name
        logger.info(
            "[minicpmo] narrow replay SoC gate: name='%s' -> %s",
            name or "<unidentified>",
            "allowed" if allows else "blocked",
        )
    return allows


def _log_block_once(reason: str) -> None:
    """Say once why the loop never engaged; silence is the worse failure."""
    global _LOGGED_BLOCK
    if _LOGGED_BLOCK == reason:
        return
    _LOGGED_BLOCK = reason
    logger.info("[minicpmo] multi-frame Talker decode not engaged: %s", reason)


def applies(model: Any, model_kwargs_extra: dict[str, Any]) -> int:
    """Frames this step should run, or 0 when the loop must not engage.

    Returns the uniform per-request query length of a pure decode step over a
    Talker that can build its own decode embedding. Anything else -- a prefill,
    a mixed step, a batch whose requests were scheduled different token counts
    -- returns 0 and the caller takes the ordinary single-forward path.

    Every refusal says so once. A step that schedules several tokens per
    request and then takes the single-forward path is not merely slow: that
    path samples one frame from the *last* row of each span and reports one
    stop row for a step that needs one per position, so it corrupts the codec
    stream and the scheduler's accounting alike. `_model_forward` turns those
    into an error rather than letting them run.
    """
    if not getattr(model, "supports_multi_frame_decode", False):
        # Every other stage: not a refusal, just not this model.
        return 0
    spans = model_kwargs_extra.get("request_token_spans")
    infos = model_kwargs_extra.get("model_intermediate_buffer")
    if not spans or not infos or len(spans) != len(infos):
        return _block("no request_token_spans/model_intermediate_buffer for this step")
    frames = int(spans[0][1]) - int(spans[0][0])
    if frames <= 1:
        return 0
    for start, end in spans:
        if int(end) - int(start) != frames:
            # A mixed step (one request prefilling, another decoding) has no
            # single frame count, and the captured graph is not a uniform
            # decode either. Dump the spans so a real-world refusal can be
            # attributed to an exact scheduling state instead of guessed at.
            logger.info(
                "[minicpmo] refusing non-uniform spans: frames=%d spans=%s",
                frames,
                [(int(s), int(e)) for s, e in spans],
            )
            return _block("request token spans are not uniform")
    for info in infos:
        if not isinstance(info, dict):
            return _block("a request carries no intermediate buffer")
        if bool(info.get("_omni_is_prefill", False)):
            return 0
        state = info.get("audio_state")
        if not isinstance(state, dict):
            return _block("a request carries no audio_state")
        if int(state.get("step", 0)) <= 0:
            # The request has not sampled a codec token yet, so there is no
            # previous code for frame 0 to embed and this is not a decode.
            return _block("a request has not sampled a codec token yet")
    return frames


def is_multi_token_decode(model: Any, model_kwargs_extra: dict[str, Any]) -> bool:
    """True when this step schedules several tokens for a request that is decoding.

    Scoped to the Talker. Stage 0 drafts the Thinker's text with n-grams and so
    schedules multi-token decode steps of its own, which vLLM handles perfectly
    well -- it is only the Talker's one-frame-per-position sampler that cannot.
    """
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


@dataclass
class NarrowStep:
    """What one codec frame has to change before the one-query graph replays.

    A K-query graph replay costs 1.33-1.47 ms of device time against the
    one-query graph's 0.83 ms, and it is a step change rather than a per-row
    cost -- a two-query graph costs the same as a four-query one. The loop only
    ever reads one row of the wide graph, so it replays the narrow one instead
    and moves the five things a frame changes:

    * ``inputs_embeds[r]`` -- request r's embedding for this frame
    * ``positions[r]``     -- its position, +frame
    * ``slot_mapping[r]``  -- the KV slot vLLM reserved for this frame
    * ``seq_lens[r]``      -- its length, +frame+1, in the buffer the graph reads
    * the block table      -- unchanged for the whole step

    The step's own K-wide layouts are cloned at ``begin`` because the narrow
    views alias them: ``slot_mapping[:rows]`` is the first row of
    ``slot_mapping.view(rows, K)``.
    """

    rows: int
    frames: int
    inputs_embeds: Any
    positions: Any
    slot_mapping: Any
    seq_lens: Any
    step_slots: Any
    step_positions: Any
    step_seq_lens: Any
    spans: list[tuple[int, int]]
    saved_descriptor: Any = None
    saved_runtime_mode: Any = None

    def select(self, frame: int) -> None:
        """Put frame ``frame``'s inputs where the captured replay reads them.

        Three copies and nothing else: everything a frame needs was computed
        once, at the start of the step, because these run between two graph
        replays and an allocation there costs more than the values do.
        """
        rows = self.rows
        self.slot_mapping[:rows].copy_(self.step_slots[:, frame])
        self.positions[:rows].copy_(self.step_positions[:, frame])
        self.seq_lens[:rows].copy_(self.step_seq_lens[:, frame])


def narrow_replay_enabled(runner: Any) -> bool:
    """Whether this worker should capture and replay the one-query graph.

    Off with ``VLLM_OMNI_MINICPMO_NARROW_REPLAY=off``, which restores the
    K-query captures and the wide replay. On a part whose one-query capture is
    known to fault (``_NARROW_BLOCKED_SOC_PREFIXES``) it is off as well unless
    explicitly enabled, so the capture can never brick a deployment.
    """
    if not _narrow_soc_allows():
        return False
    model = getattr(runner, "model", None)
    if not getattr(model, "supports_multi_frame_decode", False):
        return False
    return int(getattr(runner, "num_spec_tokens", 0) or 0) > 0


def begin_narrow_step(
    *,
    runner: Any,
    forward_context: Any,
    frames: int,
    positions: Any,
    inputs_embeds: Any,
    spans: list[tuple[int, int]],
) -> "NarrowStep | None":
    """Point the forward context at the one-query graph, or decline.

    Declines -- and the caller falls back to the wide replay -- whenever the
    narrow graph for this row count is not captured, which is what makes this
    safe to land before every path that reaches it has been exercised.
    """
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor

    # 迁移注记（2026-09-21）：narrow replay 所需的 KV 几何由第 6 项
    # （platforms/npu/attention/fixed_kv_decode）提供，本分支未迁第 6 项。
    # 这里容错导入：模块缺席时按本函数既有承诺 decline，调用方回落 wide replay
    # （见 docstring），K 步解码与停止链不受影响。
    try:
        from vllm_omni.platforms.npu.attention import fixed_kv_decode
    except Exception:
        fixed_kv_decode = None

    if fixed_kv_decode is None:
        return _decline("fixed-KV decode (item 6) is not present in this tree")
    if not narrow_replay_enabled(runner) or inputs_embeds is None or positions is None:
        return None
    rows = len(spans)
    if rows <= 0:
        return None
    seq_lens = fixed_kv_decode.captured_seq_lens(rows)
    if seq_lens is None:
        return _decline("no one-query graph captured for %d rows" % rows)
    metadata = getattr(forward_context, "attn_metadata", None)
    if isinstance(metadata, dict):
        metadata = next(iter(metadata.values()), None)
    step_slot_mapping = getattr(metadata, "slot_mapping", None)
    live_seq_lens = getattr(metadata, "seq_lens_device", None)
    graph_slot_mapping = fixed_kv_decode.captured_slot_mapping(rows)
    if step_slot_mapping is None or live_seq_lens is None or graph_slot_mapping is None:
        return _decline("this step's attention metadata carries no slot mapping")
    if step_slot_mapping.shape[0] < rows * frames:
        return _decline("slot mapping is shorter than the step's own frames")

    narrow = NarrowStep(
        rows=rows,
        frames=frames,
        inputs_embeds=inputs_embeds,
        positions=positions,
        slot_mapping=graph_slot_mapping,
        seq_lens=seq_lens,
        step_slots=step_slot_mapping[: rows * frames].view(rows, frames).clone(),
        step_positions=positions[: rows * frames].view(rows, frames).clone(),
        # `seq_lens` already counts all K of this step's frames, and by frame k
        # only k + 1 have been written -- so row r, column k is
        # `live - frames + k + 1`.
        step_seq_lens=(
            live_seq_lens[:rows, None]
            - frames
            + torch.arange(1, frames + 1, device=live_seq_lens.device, dtype=live_seq_lens.dtype)
        ),
        spans=[(index, index + 1) for index in range(rows)],
    )
    descriptor = getattr(forward_context, "batch_descriptor", None)
    narrow.saved_descriptor = descriptor
    if descriptor is not None:
        # `uniform` too: a failed dispatch clears it on the descriptor it hands
        # back, and the captured key has it set. What this step is has not
        # changed -- it is one token per request against a graph captured for
        # exactly that -- only which graph replays it.
        forward_context.batch_descriptor = dataclasses.replace(
            descriptor, num_tokens=rows, num_reqs=rows, uniform=True
        )
    else:
        forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=rows, num_reqs=rows, uniform=True
        )
    # The dispatcher builds its lookup descriptor from its own query length --
    # 4 at runtime, which is right for the step -- so it finds no key for the
    # narrow capture and hands back NONE, and the wrapper then runs the model
    # uncaptured without saying so. This step *is* a full-graph uniform decode;
    # only which graph it replays changed, which is what this function decides.
    narrow.saved_runtime_mode = getattr(forward_context, "cudagraph_runtime_mode", None)
    forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    global _LOGGED_NARROW
    if not _LOGGED_NARROW:
        _LOGGED_NARROW = True
        logger.info(
            "[minicpmo] multi-frame Talker decode replays the one-query graph: "
            "%d rows, descriptor %s, dispatched mode was %s",
            rows,
            forward_context.batch_descriptor,
            narrow.saved_runtime_mode,
        )
    return narrow


def end_narrow_step(forward_context: Any, narrow: "NarrowStep | None") -> None:
    if narrow is None:
        return
    forward_context.batch_descriptor = narrow.saved_descriptor
    if narrow.saved_runtime_mode is not None:
        forward_context.cudagraph_runtime_mode = narrow.saved_runtime_mode


def _decline(reason: str) -> None:
    global _LOGGED_NARROW_BLOCK
    if _LOGGED_NARROW_BLOCK != reason:
        _LOGGED_NARROW_BLOCK = reason
        logger.info("[minicpmo] one-query Talker replay not engaged: %s", reason)
    return None


def _block(reason: str) -> int:
    _log_block_once(reason)
    return 0


def run(
    *,
    model: Any,
    run_model: Callable[[], Any],
    after_forward: Callable[[], None],
    inputs_embeds: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    frames: int,
    model_kwargs: dict[str, Any],
    model_kwargs_extra: dict[str, Any],
    narrow: "NarrowStep | None" = None,
) -> Any:
    """Run ``frames`` codec frames and return the merged step output.

    ``after_forward`` is the runner's post-forward graph bookkeeping. It is
    called after every replay because that is where it sits in the single-frame
    path; under fixed-KV decode it is two pointer comparisons, and the frames of
    one step share their sequence lengths so there is nothing for it to rebind.
    """
    global _LOGGED_ENGAGE
    spans = model_kwargs_extra["request_token_spans"]
    infos = model_kwargs_extra["model_intermediate_buffer"]

    step_prof = _prof_step_begin(frames)
    step_start = perf_counter() if step_prof is not None else 0.0

    frame_outputs = []
    frame_stop_logits = []
    frame_hidden = []
    for frame in range(frames):
        if narrow is not None:
            # Frame 0's embeddings are already in the buffer, but at the rows a
            # K-query step laid them out on: request r's is at row r*K, and the
            # one-query graph reads row r. r <= r*K, so compacting in request
            # order never overwrites a row it has still to read.
            narrow.select(frame)
            if frame == 0:
                _compact_frame_zero_embeddings(inputs_embeds, spans)
            else:
                _write_frame_embeddings(
                    model, inputs_embeds, input_ids, spans, infos, frame, narrow=narrow
                )
        elif frame > 0:
            _write_frame_embeddings(model, inputs_embeds, input_ids, spans, infos, frame)
        t0 = perf_counter() if step_prof is not None else 0.0
        hidden = run_model()
        t1 = perf_counter() if step_prof is not None else 0.0
        after_forward()
        t2 = perf_counter() if step_prof is not None else 0.0
        if step_prof is not None and step_prof.detail:
            _prof_sync()
        t3 = perf_counter() if step_prof is not None else 0.0
        if step_prof is not None:
            _prof_frame(step_prof, frame, t0, t1, t2, t3)
        frame_kwargs = dict(model_kwargs_extra)
        frame_kwargs["request_token_spans"] = (
            list(narrow.spans)
            if narrow is not None
            else [(int(start) + frame, int(start) + frame + 1) for start, _ in spans]
        )
        if narrow is not None:
            # The graph writes one row per request and the next frame overwrites
            # it, so this frame's hidden state has to be kept before the replay
            # that follows.
            frame_hidden.append(hidden.clone())
        output = model.make_omni_output(hidden, **model_kwargs, **frame_kwargs)
        stop_logits = model.take_batch_stop_logits()
        if stop_logits is None:
            raise RuntimeError("MiniCPM-o multi-frame decode expects stop rows from every frame")
        frame_outputs.append(output)
        frame_stop_logits.append(stop_logits)

    if not _LOGGED_ENGAGE:
        _LOGGED_ENGAGE = True
        logger.info(
            "[minicpmo] multi-frame Talker decode engaged: %d codec frames per step", frames
        )
    if stop_trace_enabled():
        _trace_stop_rows(frame_stop_logits)
    merged = model.merge_frame_outputs(frame_outputs, frame_stop_logits)
    if narrow is not None and frame_hidden:
        # (frames, rows, hidden) -> (rows * frames, hidden), request-major --
        # the layout `logits_indices` reads. OmniOutput is a NamedTuple.
        merged = merged._replace(
            text_hidden_states=torch.stack(frame_hidden, dim=1).reshape(
                -1, frame_hidden[0].shape[-1]
            )
        )
    # Whole step, frames + merge: this is the number to compare against the
    # scheduler's step cadence.
    _prof_step_end(step_prof, step_start)
    return merged


def _compact_frame_zero_embeddings(inputs_embeds: torch.Tensor | None, spans) -> None:
    """Move each request's frame-0 embedding to the row the narrow graph reads."""
    if inputs_embeds is None:
        raise RuntimeError("MiniCPM-o narrow Talker replay requires an inputs_embeds buffer")
    for row, (start, _) in enumerate(spans):
        if int(start) != row:
            inputs_embeds[row : row + 1].copy_(inputs_embeds[int(start) : int(start) + 1])


def _write_frame_embeddings(
    model: Any,
    inputs_embeds: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    spans: list[tuple[int, int]],
    infos: list[dict[str, Any]],
    frame: int,
    narrow: "NarrowStep | None" = None,
) -> None:
    """Put frame ``frame``'s decode embedding into the row the graph reads.

    ``model.preprocess`` is the same call the runner's ``_preprocess`` makes for
    frame 0; it reads the codec token the previous frame wrote into
    ``info["audio_codes"]["current"]`` and returns its embedding. Writing into
    ``inputs_embeds`` in place is what makes the next replay see it -- the
    buffer is the one the graph captured, and vLLM does not copy inputs into
    captured graphs (``cudagraph_copy_inputs`` is False).
    """
    if inputs_embeds is None:
        raise RuntimeError("MiniCPM-o multi-frame decode requires an inputs_embeds buffer")
    for index, ((start, _), info) in enumerate(zip(spans, infos)):
        # The token to embed always sits at the step's own row; only where the
        # embedding goes changes when the narrow graph is the one replaying.
        source = int(start) + frame
        row = index if narrow is not None else source
        row_ids = input_ids[source : source + 1] if input_ids is not None else None
        if row_ids is None:
            raise RuntimeError("MiniCPM-o multi-frame decode requires input_ids for preprocess")
        _, embeds, _ = model.preprocess(input_ids=row_ids, input_embeds=None, **info)
        inputs_embeds[row : row + 1].copy_(embeds[:1])


# The Talker's vLLM-level vocabulary is two tokens wide: `compute_logits`
# returns one-hot stop rows ([0, -inf] / [-inf, 0]) and `sample` takes their
# argmax, so 0 means "keep going" and 1 is the stop token the deploy config
# names in `stop_token_ids`.
CONTINUE_TOKEN_ID = 0

# The id that same row emits to mean stop: the stop column wins the argmax,
# `parse_output` keeps ids below the two-wide vocab, and vLLM's `check_stop`
# only fires if this id is in the request's `stop_token_ids` -- which stage 1's
# pipeline constraints derive from this constant.
STOP_TOKEN_ID = 1

# Width of the Talker vLLM-level head: the two-wide continue/stop row.
# input_batch.vocab_size must report the same, because InputBatch stores
# top_k = vocab_size as its no-top-k sentinel and that holds only for 0 or 2.
STOP_ROW_WIDTH = 2


def drafts_this_step(runner: Any) -> int:
    """Frames the *next* step should be scheduled for, or 0 to stay generic.

    The count travels to the scheduler as speculative tokens because that is
    vLLM V1's only way of saying "this request advanced by more than one token",
    which is what grows the block table and reserves the KV slots the loop
    writes into. Nothing about it is speculative: the drafts are always
    `continue`, and the frames are generated sequentially rather than verified.
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
) -> list[list[int]]:
    """`continue` repeated -- for the whole batch, or for none of it.

    The drafts are how the next step gets K query positions per request, and
    the multi-frame loop can only run a step whose requests all have the *same*
    span: it replays one captured uniform-decode graph and samples one codec
    frame per position. So drafting per request is not an option. A request
    that sampled nothing this step (discarded, or finished on this step's stop
    token) would be scheduled one token while its neighbours got K, and
    `applies` would refuse the resulting step -- which `_model_forward` turns
    into a fatal error rather than let the single-forward path sample one frame
    from the last row of a K-row span.

    So: every request drafts, or nobody does. A step with no drafts is an
    ordinary one-frame-per-request decode, and the frames resume on the step
    after it.
    """
    for index in range(num_reqs):
        sampled = None
        if isinstance(valid_sampled_token_ids, list) and index < len(valid_sampled_token_ids):
            sampled = valid_sampled_token_ids[index]
        if not sampled:
            _log_block_once("a request in this batch sampled nothing; no drafts this step")
            return [[] for _ in range(num_reqs)]
    return [[CONTINUE_TOKEN_ID] * (frames - 1) for _ in range(num_reqs)]


def ensure_stop_token_vocab(runner: Any, logits: Any) -> None:
    """Let vLLM's rejection-sampler output survive the Talker's narrow head.

    `RejectionSampler.parse_output` keeps a sampled token only while it is both
    not the placeholder and `< vocab_size`, and stage 1's `input_batch` reports
    a vocab size of **0** -- the Talker's vLLM-level head is the two-wide
    continue/stop row `compute_logits` builds, not a tokenizer vocabulary, and
    nothing had ever needed the number. With 0 the mask is empty, every
    accepted token is dropped, and the request comes back having generated
    nothing: the scheduler then never rolls back the tokens it advanced for the
    step and schedules a negative count two steps later.

    A single-frame step never reached this. It returns one token per request,
    which `_bookkeeping_sync` takes down the `max_gen_len == 1` branch, and that
    branch does not consult `vocab_size` at all.

    Raising it back to the width of the row the model actually emits is the
    whole fix -- and that width is `STOP_ROW_WIDTH` (two), *not* the width of
    whatever tensor the caller happens to be holding. The Talker's vLLM-level
    head is the two-wide continue/stop row `compute_logits` builds, so two is
    what every `vocab_size` reader has to see.

    The caller used to hand over `text_hidden_states`, so this landed on the
    hidden width (768). That passes the `parse_output` filter by accident (0
    and 1 are both below 768) and breaks the one reader that repays checking:
    `InputBatch.add_request` stores `top_k = vocab_size` as its "no top-k"
    sentinel, which only holds for 0 and 2. At 768 `top_k_reqs` stops being
    empty, `sampling_metadata.top_k` stops being None, and the Ascend
    `enable_reduce_sample` branch starts treating a two-column distribution as
    a 768-way one. The stop row then does not survive into the request's token
    list, so the request runs to `max_tokens` instead of stopping on EOS -- and
    the frame stream that keeps arriving after the codec sequence ended is what
    eventually hands the scheduler a step it cannot merge.
    """
    if logits is None or not getattr(runner.model, "supports_multi_frame_decode", False):
        return
    batch = getattr(runner, "input_batch", None)
    width = STOP_ROW_WIDTH
    if batch is None or int(getattr(batch, "vocab_size", 0) or 0) >= width:
        return
    logger.info(
        "[minicpmo] Talker input batch reported vocab_size=%s; raising it to the "
        "%d-wide stop row so parse_output keeps the accepted tokens",
        getattr(batch, "vocab_size", None),
        width,
    )
    batch.vocab_size = width
