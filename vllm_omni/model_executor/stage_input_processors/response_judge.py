# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Response-judge stage: decide after ASR whether a committed turn needs a reply.

VAD commits a turn on every pause, so a backchannel ("uh-huh"), a cough or
background speech runs the whole downstream pipeline. A pipeline can place a
``response_judge`` stage right after its ASR stage: a small judge model reads
the transcript, and a turn that needs no reply ends there through the
engine's existing no-output path (``[]`` from the next bridge, or a listen
decision in duplex sessions).

The judge model is configured on its own stage, as ``hf_overrides``:

    response_judge:
      format: chat_yes_no      # generative model, one token (YES / NO)
      system_prompt: "..."
    # or
    response_judge:
      format: laya             # LAYA decision model (pooling runner)
      instructions: "..."
      options: {yes: "...", no: "..."}
      reply_option: yes
    # or format: clm           # CLM encoder + heads (pooling runner)

Anything other than a clear "no reply" lets the turn through unchanged.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any
from weakref import WeakValueDictionary

from vllm.logger import init_logger

logger = init_logger(__name__)

RESPONSE_JUDGE_STAGE = "response_judge"
RESPONSE_JUDGE_CONFIG_KEY = "response_judge"
_STATE_KEY = "response_judge"

DEFAULT_CHAT_SYSTEM_PROMPT = (
    "你是语音助手前面的门控判断器，只判断语音助手要不要回应用户刚说的话。"
    "只输出 YES 或 NO，不要输出其他内容。\n"
    "YES：用户在提问、提出请求或指令，或者在回答助手刚才的问题。\n"
    "NO：用户只是在附和（嗯、好的、哦、对对对）、在笑、在咳嗽，这段文字是噪声或无意义的字幕，"
    "或者用户在和身边的其他人说话。"
)
DEFAULT_CHAT_USER_TEMPLATE = "对话记录：\n（还没有对话）\n用户刚刚说：「{transcript}」"


@dataclass(frozen=True, slots=True)
class JudgeSpec:
    """How to prompt one judge model and how to read its answer."""

    format: str
    options: Mapping[str, Any]

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> JudgeSpec:
        raw = getattr(hf_config, RESPONSE_JUDGE_CONFIG_KEY, None)
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        fmt = str(raw.pop("format", "chat_yes_no"))
        if fmt not in _FORMATS:
            raise ValueError(f"unknown response_judge format {fmt!r}; expected one of {sorted(_FORMATS)}")
        return cls(format=fmt, options=raw)


@dataclass
class _PendingTurn:
    # No slots: weak references must also work on Python 3.10.
    spec: JudgeSpec | None
    transcript: str
    source_output: Any
    # The ASR stage's token decoder, which the orchestrator swaps for the
    # judge's while it forwards the judge output.
    source_token_decoder: Any = None
    # A streaming request forwards one finished judge output twice (update,
    # then terminal); both forwards must get the same answer. The decision is
    # kept with the output it was made for, so a different judge output (e.g.
    # an earlier turn of the same request id arriving late) is judged on its own.
    decided_for: Any = None
    rejected: bool = False


_PENDING: WeakValueDictionary[str, _PendingTurn] = WeakValueDictionary()
_LOCK = Lock()


def _owned(streaming_context: Any) -> dict[str, _PendingTurn]:
    bridge_states = getattr(streaming_context, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        raise RuntimeError("response judge requires request-owned streaming bridge state")
    return bridge_states.setdefault(_STATE_KEY, {})


def _remember(request_id: str, pending: _PendingTurn, streaming_context: Any) -> None:
    # The request's bridge state owns the entry; the index only holds a weak
    # reference, so cancel / failure / close release it with the request.
    owned = _owned(streaming_context)
    with _LOCK:
        owned[request_id] = pending
        _PENDING[request_id] = pending


def _peek(request_id: str) -> _PendingTurn | None:
    with _LOCK:
        return _PENDING.get(request_id)


def _owned_turn(request_id: str, streaming_context: Any) -> _PendingTurn | None:
    # The turn stays with its request until the request ends (or the next turn
    # of the same request replaces it), so a repeated forward still finds it.
    owned = _owned(streaming_context)
    with _LOCK:
        return owned.get(request_id)


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def _first_completion(output: Any) -> Any:
    outputs = getattr(output, "outputs", None)
    if isinstance(outputs, list) and outputs:
        return outputs[0]
    return outputs


def _completion_text(output: Any) -> str:
    completion = _first_completion(output)
    for attr in ("cumulative_text", "text"):
        value = getattr(completion, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _pooled_tensor(output: Any) -> Any:
    """Pooling output of a judge request, whichever carrier the engine used."""
    completion = _first_completion(output)
    for obj, attr in ((completion, "data"), (output, "pooling_output"), (completion, "pooling_output")):
        value = getattr(obj, attr, None)
        if value is not None and hasattr(value, "tolist"):
            return value
    # A stage with a detokenizer moves the pooling tensor under its output
    # modality key (``engine_output_type``); the judge emits exactly one.
    for mm in (getattr(output, "multimodal_output", None), getattr(completion, "multimodal_output", None)):
        if isinstance(mm, Mapping):
            tensors = [value for value in mm.values() if value is not None and hasattr(value, "tolist")]
            if len(tensors) == 1:
                return tensors[0]
    return None


def default_transcript(source_output: Any, source_prompt: Mapping[str, Any]) -> str:
    del source_prompt
    return _completion_text(source_output).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _prompt_for(source_outputs: list[Any], prompt: Any) -> dict[str, Mapping[str, Any]]:
    prompts = _as_list(prompt)
    mapping: dict[str, Mapping[str, Any]] = {}
    for idx, source_output in enumerate(source_outputs):
        item = prompts[idx] if idx < len(prompts) else (prompts[0] if len(prompts) == 1 else None)
        mapping[str(getattr(source_output, "request_id", idx))] = item if isinstance(item, Mapping) else {}
    return mapping


# --------------------------------------------------------------------------- #
# Judge formats: prompt building (orchestrator side) and decision reading
# --------------------------------------------------------------------------- #


def _chat_prompt(spec: JudgeSpec, transcript: str, model_config: Any) -> dict[str, Any]:
    from vllm.tokenizers import cached_tokenizer_from_config

    tokenizer = cached_tokenizer_from_config(model_config)
    user = str(spec.options.get("user_template", DEFAULT_CHAT_USER_TEMPLATE)).format(transcript=transcript)
    messages = [
        {"role": "system", "content": str(spec.options.get("system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT))},
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # Qwen3-style templates: no thinking block, the first token is the answer.
        enable_thinking=False,
    )
    return {"prompt": text}


def _chat_rejects(spec: JudgeSpec, output: Any) -> bool:
    answer = _completion_text(output).strip().upper()
    return bool(answer) and answer == str(spec.options.get("reject_label", "NO")).upper()


def _laya_prompt(spec: JudgeSpec, transcript: str, model_config: Any) -> dict[str, Any]:
    """Token ids in LAYA's layout (laya.common.build_sequence, Apache-2.0):

    ``[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]``
    """
    from vllm.tokenizers import cached_tokenizer_from_config

    tok = cached_tokenizer_from_config(model_config)
    opts = spec.options
    # The model's head scores this question type too (laya_question_type).
    qtype = str(opts.get("question_type", getattr(model_config.hf_config, "laya_question_type", "choice")))
    max_len = int(opts.get("max_len", 1024))
    head_max_len = int(opts.get("head_max_len", 256))
    mask = tok.mask_token
    options: Mapping[str, str] = opts.get("options") or {"yes": "a reply is needed", "no": "no reply is needed"}
    rendered = [k if not v else f"{k}: {v}" for k, v in options.items()]
    head_ids = tok.encode(
        f"{qtype} question: {str(opts.get('instructions', '')).replace(mask, ' ')}", add_special_tokens=False
    )
    opt_ids = []
    for text in rendered:
        ids = tok.encode(" " + text.replace(mask, " "), add_special_tokens=False)[:48]
        opt_ids.append([tok.mask_token_id, *ids])
    budget = head_max_len - sum(len(o) for o in opt_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        budget = head_max_len - sum(len(o) for o in opt_ids)
    ids = [tok.cls_token_id, *head_ids[: max(8, budget)], tok.sep_token_id]
    for o in opt_ids:
        ids.extend(o)
    ids.append(tok.sep_token_id)
    state = str(opts.get("state_template", "{transcript}")).format(transcript=transcript).replace(mask, " ")
    room = max(0, max_len - len(ids) - 1)
    ids = ids + tok.encode(state, add_special_tokens=False)[:room] + [tok.sep_token_id]
    return {"prompt_token_ids": ids[:max_len]}


def _option_scores_reject(spec: JudgeSpec, output: Any) -> bool:
    """Pooling judges (LAYA, CLM): one logit per option -> P(reply) vs ``threshold``.

    ``option_keys`` (or the keys of ``options``) name the logits in order;
    ``reply_option`` is one key or a list of keys whose probabilities add up.
    """
    import torch

    logits = _pooled_tensor(output)
    if logits is None:
        return False
    keys = list(spec.options.get("option_keys") or (spec.options.get("options") or {"yes": None, "no": None}))
    reply = spec.options.get("reply_option", keys[0])
    reply_keys = [str(k) for k in (reply if isinstance(reply, list) else [reply])]
    scores = torch.as_tensor(logits)
    if scores.dim() == 2 and scores.shape[0] == 1:
        scores = scores[0]
    # One finite real score per option; anything else is unreadable.
    if scores.dim() != 1 or scores.dtype == torch.bool or scores.is_complex():
        return False
    scores = scores.float()
    if not bool(torch.isfinite(scores).all()):
        return False
    probs = torch.softmax(scores, dim=-1).tolist()
    if len(probs) != len(keys) or not reply_keys or any(k not in keys for k in reply_keys):
        return False
    p_reply = sum(probs[keys.index(k)] for k in reply_keys)
    return p_reply < float(spec.options.get("threshold", 0.5))


def _clm_prompt(spec: JudgeSpec, transcript: str, model_config: Any) -> dict[str, Any]:
    """CLM state text (clm.schema.state_text): context, a blank line, then the question."""
    del model_config
    state = str(spec.options.get("state_template", "{transcript}")).format(transcript=transcript).strip()
    instructions = str(spec.options.get("instructions", "")).strip()
    return {"prompt": f"{state}\n\n{instructions}" if state and instructions else (state or instructions)}


_FORMATS: dict[str, tuple[Callable[..., dict[str, Any]], Callable[[JudgeSpec, Any], bool]]] = {
    "chat_yes_no": (_chat_prompt, _chat_rejects),
    "laya": (_laya_prompt, _option_scores_reject),
    # Option projections are baked into the prepared CLM model directory, whose
    # config.json also carries the matching response_judge options.
    "clm": (_clm_prompt, _option_scores_reject),
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _rejects(pending: _PendingTurn | None, output: Any) -> bool:
    # No recorded turn, an empty transcript or an unreadable answer: let it through.
    if pending is None or pending.spec is None or not pending.transcript:
        return False
    try:
        return _FORMATS[pending.spec.format][1](pending.spec, output)
    except Exception:
        logger.exception(
            "response judge output could not be read; letting request %s through",
            getattr(output, "request_id", None),
        )
        return False


def judge_rejects(output: Any) -> bool:
    """True only when the judge clearly said this turn needs no reply.

    Called by the engine on a finished ``response_judge`` stage output.
    """
    request_id = getattr(output, "request_id", None)
    return _rejects(_peek(request_id) if isinstance(request_id, str) else None, output)


def judge_input(
    transcript_fn: Callable[[Any, Mapping[str, Any]], str] = default_transcript,
) -> Callable[..., list[dict[str, Any]]]:
    """Bridge ASR -> judge. ``transcript_fn`` lets a pipeline normalize its ASR text."""

    def asr2judge(
        source_outputs: list[Any],
        prompt: Any = None,
        requires_multimodal_data: bool = False,
        streaming_context: Any = None,
        *,
        target_model_config: Any,
    ) -> list[dict[str, Any]]:
        del requires_multimodal_data
        spec = JudgeSpec.from_hf_config(target_model_config.hf_config)
        build = _FORMATS[spec.format][0]
        prompts = _prompt_for(source_outputs, prompt)
        next_inputs = []
        for idx, source_output in enumerate(source_outputs):
            request_id = str(getattr(source_output, "request_id", idx))
            transcript = transcript_fn(source_output, prompts.get(request_id, {}))
            # Build first: a prompt that cannot be built must not leave a turn behind.
            next_input = build(spec, transcript, target_model_config)
            decoder = getattr(streaming_context, "source_token_decoder", None)
            _remember(request_id, _PendingTurn(spec, transcript, source_output, decoder), streaming_context)
            next_inputs.append(next_input)
        return next_inputs

    return asr2judge


def after_judge(bridge: Callable[..., list[Any]]) -> Callable[..., list[Any]]:
    """Wrap a pipeline's original ASR -> main-model bridge to run after the judge.

    The wrapped bridge sees the ASR output exactly as without the judge. A turn
    the judge rejected yields no input, so the request ends through the
    engine's existing empty-output path.
    """
    # The caller (StageEngineCoreClientBase._call_custom_process_input) passes
    # the request's streaming context as the 4th positional argument when the
    # processor declares ``streaming_context`` or ``_streaming_context``, and
    # keyword-only extras (``target_model_config`` ...) by name. The judge
    # always needs the context; the wrapped bridge gets it only if it asked.
    signature = inspect.signature(bridge)
    wants_context = any(name in signature.parameters for name in ("streaming_context", "_streaming_context"))
    params = list(signature.parameters.values())
    if not wants_context:
        positional_kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        split = max((i + 1 for i, p in enumerate(params) if p.kind in positional_kinds), default=0)
        context = inspect.Parameter("streaming_context", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)
        params = [*params[:split], context, *params[split:]]

    @functools.wraps(bridge)
    def judged(source_outputs, prompt=None, requires_multimodal_data=False, streaming_context=None, **kwargs):
        asr_outputs = []
        asr_decoder = None
        for idx, judge_output in enumerate(source_outputs):
            request_id = str(getattr(judge_output, "request_id", idx))
            pending = _owned_turn(request_id, streaming_context)
            if pending is None:
                raise RuntimeError(f"response judge has no ASR output for request {request_id}")
            if pending.decided_for is not judge_output:
                pending.rejected = _rejects(pending, judge_output)
                pending.decided_for = judge_output
            if pending.rejected:
                continue
            asr_outputs.append(pending.source_output)
            asr_decoder = asr_decoder or pending.source_token_decoder
        if not asr_outputs:
            return []
        if not wants_context:
            return bridge(asr_outputs, prompt, requires_multimodal_data, **kwargs)
        # The wrapped bridge reads ASR outputs, so it gets the ASR decoder back.
        judge_decoder = getattr(streaming_context, "source_token_decoder", None)
        if asr_decoder is not None:
            streaming_context.source_token_decoder = asr_decoder
        try:
            return bridge(asr_outputs, prompt, requires_multimodal_data, streaming_context, **kwargs)
        finally:
            if asr_decoder is not None:
                streaming_context.source_token_decoder = judge_decoder

    judged.__signature__ = signature.replace(parameters=params)  # type: ignore[attr-defined]
    return judged


asr2judge = judge_input()

__all__ = [
    "DEFAULT_CHAT_SYSTEM_PROMPT",
    "RESPONSE_JUDGE_STAGE",
    "JudgeSpec",
    "after_judge",
    "asr2judge",
    "default_transcript",
    "judge_input",
    "judge_rejects",
]
