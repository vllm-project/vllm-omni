# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the model-agnostic response-judge bridges."""

from __future__ import annotations

import gc
import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors import response_judge as rj

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|assistant:"


class _LayaTokenizer:
    """Whitespace tokenizer with LAYA's special ids (cls=1, sep=1, mask=4)."""

    mask_token = "[MASK]"
    mask_token_id = 4
    cls_token_id = 1
    sep_token_id = 1

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.vocab.setdefault(word, 100 + len(self.vocab)) for word in text.split()]


def _model_config(response_judge, tokenizer, monkeypatch):
    monkeypatch.setattr("vllm.tokenizers.cached_tokenizer_from_config", lambda config: tokenizer)
    return SimpleNamespace(hf_config=SimpleNamespace(response_judge=response_judge))


def _asr(request_id: str, text: str):
    return SimpleNamespace(request_id=request_id, finished=True, outputs=[SimpleNamespace(text=text)])


def _judge_text(request_id: str, text: str):
    return SimpleNamespace(request_id=request_id, finished=True, outputs=[SimpleNamespace(text=text)])


def _judge_pooled(request_id: str, logits):
    return SimpleNamespace(request_id=request_id, finished=True, outputs=SimpleNamespace(data=torch.tensor(logits)))


def _owner():
    return SimpleNamespace(bridge_states={})


def test_unknown_format_is_rejected_at_prompt_time():
    with pytest.raises(ValueError, match="unknown response_judge format"):
        rj.JudgeSpec.from_hf_config(SimpleNamespace(response_judge={"format": "nope"}))


def test_default_format_is_a_one_token_chat_judge():
    spec = rj.JudgeSpec.from_hf_config(SimpleNamespace())
    assert spec.format == "chat_yes_no"


@pytest.mark.parametrize(
    ("answer", "rejected"),
    [("NO", True), (" no\n", True), ("YES", False), ("", False), ("NO.", False), ("maybe", False)],
)
def test_chat_judge_rejects_only_a_clear_no(monkeypatch, answer, rejected):
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)
    owner = _owner()
    [judge_input] = rj.asr2judge([_asr("r1", "嗯嗯")], None, False, owner, target_model_config=config)
    assert "用户刚刚说：「嗯嗯」" in judge_input["prompt"]
    assert rj.judge_rejects(_judge_text("r1", answer)) is rejected


def test_chat_judge_uses_configured_prompts(monkeypatch):
    config = _model_config(
        {"format": "chat_yes_no", "system_prompt": "RULES", "user_template": "U<{transcript}>"},
        _ChatTokenizer(),
        monkeypatch,
    )
    [judge_input] = rj.asr2judge([_asr("r1", "hi")], None, False, _owner(), target_model_config=config)
    assert judge_input == {"prompt": "system:RULES|user:U<hi>|assistant:"}


def test_empty_transcript_is_never_rejected(monkeypatch):
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)
    owner = _owner()
    rj.asr2judge([_asr("r1", "   ")], None, False, owner, target_model_config=config)
    assert rj._peek("r1") is not None
    assert rj.judge_rejects(_judge_text("r1", "NO")) is False


def test_unknown_request_is_let_through():
    assert rj.judge_rejects(_judge_text("never-judged", "NO")) is False


def test_after_judge_forwards_the_asr_output_and_drops_rejected_turns(monkeypatch):
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)
    seen = []

    def asr2main(source_outputs, prompt=None, requires_multimodal_data=False):
        seen.append((source_outputs, prompt, requires_multimodal_data))
        return [{"prompt": source_outputs[0].outputs[0].text}]

    judged = rj.after_judge(asr2main)
    owner = _owner()
    asr = _asr("r1", "今天天气怎么样")
    rj.asr2judge([asr], None, False, owner, target_model_config=config)
    assert judged([_judge_text("r1", "YES")], {"p": 1}, True, owner) == [{"prompt": "今天天气怎么样"}]
    assert seen == [([asr], {"p": 1}, True)]

    rj.asr2judge([_asr("r2", "嗯嗯")], None, False, owner, target_model_config=config)
    assert judged([_judge_text("r2", "NO")], None, True, owner) == []
    assert owner.bridge_states["response_judge"] == {}


def test_after_judge_keeps_the_wrapped_signature_for_extra_context():
    def bridge(source_outputs, prompt=None, requires_multimodal_data=False, *, target_model_config):
        return [target_model_config]

    params = inspect.signature(rj.after_judge(bridge)).parameters
    assert "streaming_context" in params
    assert params["target_model_config"].kind is inspect.Parameter.KEYWORD_ONLY


def test_after_judge_without_a_judged_turn_raises():
    judged = rj.after_judge(lambda source_outputs, prompt=None, requires_multimodal_data=False: [])
    with pytest.raises(RuntimeError, match="no ASR output"):
        judged([_judge_text("missing", "YES")], None, False, _owner())


def test_pending_turn_is_released_with_the_request_owner(monkeypatch):
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)
    owner = _owner()
    rj.asr2judge([_asr("r-owned", "嗯嗯")], None, False, owner, target_model_config=config)
    assert rj.judge_rejects(_judge_text("r-owned", "NO")) is True
    del owner
    gc.collect()
    assert rj.judge_rejects(_judge_text("r-owned", "NO")) is False


def test_judge_requires_request_owned_bridge_state(monkeypatch):
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)
    with pytest.raises(RuntimeError, match="bridge state"):
        rj.asr2judge([_asr("r1", "hi")], None, False, None, target_model_config=config)


def test_laya_prompt_follows_the_laya_sequence_layout(monkeypatch):
    tok = _LayaTokenizer()
    options = {"yes": "reply", "no": "stay quiet"}
    config = _model_config(
        {
            "format": "laya",
            "instructions": "should we reply?",
            "options": options,
            "state_template": "said {transcript}",
        },
        tok,
        monkeypatch,
    )
    [judge_input] = rj.asr2judge([_asr("r1", "hello there")], None, False, _owner(), target_model_config=config)
    ids = judge_input["prompt_token_ids"]
    v = tok.vocab
    head = [v["choice"], v["question:"], v["should"], v["we"], v["reply?"]]
    opt0 = [4, v["yes:"], v["reply"]]
    opt1 = [4, v["no:"], v["stay"], v["quiet"]]
    state = [v["said"], v["hello"], v["there"]]
    assert ids == [1, *head, 1, *opt0, *opt1, 1, *state, 1]


@pytest.mark.parametrize(("logits", "rejected"), [([2.0, 0.0], False), ([0.0, 2.0], True), ([0.0], False)])
def test_laya_judge_rejects_when_the_reply_option_is_unlikely(monkeypatch, logits, rejected):
    config = _model_config(
        {"format": "laya", "options": {"yes": "reply", "no": "quiet"}, "reply_option": "yes"},
        _LayaTokenizer(),
        monkeypatch,
    )
    owner = _owner()
    rj.asr2judge([_asr("r1", "嗯嗯")], None, False, owner, target_model_config=config)
    assert rj._peek("r1") is not None
    assert rj.judge_rejects(_judge_pooled("r1", logits)) is rejected


def test_unreadable_pooling_output_is_let_through(monkeypatch):
    config = _model_config({"format": "laya"}, _LayaTokenizer(), monkeypatch)
    owner = _owner()
    rj.asr2judge([_asr("r1", "嗯嗯")], None, False, owner, target_model_config=config)
    assert rj._peek("r1") is not None
    assert rj.judge_rejects(SimpleNamespace(request_id="r1", outputs=[SimpleNamespace(text="")])) is False


def test_clm_prompt_is_context_blank_line_question(monkeypatch):
    config = _model_config(
        {"format": "clm", "instructions": "Does the assistant need to answer?", "state_template": "User: {transcript}"},
        _ChatTokenizer(),
        monkeypatch,
    )
    owner = _owner()
    [judge_input] = rj.asr2judge([_asr("r1", "嗯嗯")], None, False, owner, target_model_config=config)
    assert judge_input == {"prompt": "User: 嗯嗯\n\nDoes the assistant need to answer?"}


@pytest.mark.parametrize(("logits", "rejected"), [([0.0, 0.0, 3.0], True), ([1.0, 1.0, 0.0], False)])
def test_several_reply_options_add_up(monkeypatch, logits, rejected):
    config = _model_config(
        {
            "format": "clm",
            "option_keys": ["request", "answer", "backchannel"],
            "reply_option": ["request", "answer"],
            "threshold": 0.5,
        },
        _ChatTokenizer(),
        monkeypatch,
    )
    owner = _owner()
    rj.asr2judge([_asr("r1", "好的")], None, False, owner, target_model_config=config)
    assert rj.judge_rejects(_judge_pooled("r1", logits)) is rejected


def test_multimodal_carrier_with_one_tensor_is_read(monkeypatch):
    config = _model_config({"format": "laya", "options": {"yes": "", "no": ""}}, _LayaTokenizer(), monkeypatch)
    owner = _owner()
    rj.asr2judge([_asr("r1", "嗯嗯")], None, False, owner, target_model_config=config)
    output = SimpleNamespace(
        request_id="r1", outputs=[SimpleNamespace(text="")], multimodal_output={"text": torch.tensor([0.0, 3.0])}
    )
    assert rj.judge_rejects(output) is True


def _call_like_the_engine(processor, source_outputs, prompt, owner, **extras):
    """Invoke through the real StageEngineCoreClientBase._call_custom_process_input."""
    from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase

    client = SimpleNamespace(
        custom_process_input_func=processor,
        requires_multimodal_data=True,
        _stage_hf_config=extras.get("hf_config"),
        vllm_config=SimpleNamespace(model_config=extras.get("model_config")),
    )
    return StageEngineCoreClientBase._call_custom_process_input(client, source_outputs, prompt, owner)


def _bridge_shapes():
    def plain(source_outputs, prompt=None, requires_multimodal_data=False):
        return [("plain", source_outputs[0].outputs[0].text, None)]

    def underscore_context(source_outputs, prompt, requires_multimodal_data, _streaming_context=None):
        return [("underscore", source_outputs[0].outputs[0].text, _streaming_context)]

    def varkw_with_model_config(
        source_outputs, prompt=None, requires_multimodal_data=False, *, target_model_config, **kw
    ):
        return [("varkw", source_outputs[0].outputs[0].text, target_model_config)]

    return {"plain": plain, "underscore": underscore_context, "varkw": varkw_with_model_config}


@pytest.mark.parametrize("shape", ["plain", "underscore", "varkw"])
def test_after_judge_forwards_through_the_engine_caller(monkeypatch, shape):
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)
    owner = _owner()
    rj.asr2judge([_asr("r1", "今天天气怎么样")], None, False, owner, target_model_config=config)
    judged = rj.after_judge(_bridge_shapes()[shape])
    [(name, text, extra)] = _call_like_the_engine(judged, [_judge_text("r1", "YES")], None, owner, model_config="MC")
    assert (name, text) == (shape, "今天天气怎么样")
    assert extra == {"plain": None, "underscore": owner, "varkw": "MC"}[shape]
