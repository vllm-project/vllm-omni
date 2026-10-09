# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the model-agnostic response-judge bridges."""

from __future__ import annotations

import gc
import inspect
import weakref
from pathlib import Path
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


def test_unknown_format_is_rejected_when_the_spec_is_constructed():
    with pytest.raises(ValueError, match="unknown response_judge format"):
        rj.JudgeSpec.from_hf_config(SimpleNamespace(response_judge={"format": "nope"}))


def test_default_format_is_a_one_token_chat_judge():
    spec = rj.JudgeSpec.from_hf_config(SimpleNamespace())
    assert spec.format == "chat_yes_no"


def _startup_model_config(response_judge, monkeypatch, model_stage=rj.RESPONSE_JUDGE_STAGE):
    from vllm_omni.config.model import OmniModelConfig

    # The HF config is already loaded; avoid unrelated text-config/model inspection.
    monkeypatch.setattr(OmniModelConfig, "_maybe_override_text_config", lambda self: None)
    base = SimpleNamespace(hf_config=SimpleNamespace(response_judge=response_judge))
    return OmniModelConfig.from_vllm_model_config(base, model_stage=model_stage, model_arch="ClmDecisionModel")


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ({"format": "clm", "threshold": 0}, None),
        ({"format": "clm", "threshold": 1}, None),
        ({"format": "clm", "threshold": "0.0173"}, None),
        ({"format": "chat_yes_no"}, None),
        ({"format": "clm", "threshold": 1.01}, "threshold"),
        ({"format": "clm", "threshold": float("nan")}, "threshold"),
        ({"format": "clm", "threshold": "invalid"}, "threshold"),
        ({"format": "clm", "options": {}}, "options"),
        ({"format": "clm", "option_keys": []}, "option_keys"),
        ({"format": "clm", "reply_option": "missing"}, "reply_option"),
        ({"format": "clm", "reply_option": []}, "reply_option"),
    ],
)
def test_judge_configuration_is_checked_at_startup(monkeypatch, raw, error):
    if error is None:
        _startup_model_config(raw, monkeypatch)
        return
    with pytest.raises(ValueError, match=f"response_judge.{error}"):
        _startup_model_config(raw, monkeypatch)


def test_startup_spec_is_reused_for_later_transcripts(monkeypatch):
    config = _startup_model_config({"format": "clm", "option_keys": ["request", "quiet"]}, monkeypatch)
    spec = rj.JudgeSpec.for_model_config(config)

    def unexpected_reload(cls, hf_config):
        pytest.fail("judge configuration must be validated at startup, not per transcript")

    monkeypatch.setattr(rj.JudgeSpec, "from_hf_config", classmethod(unexpected_reload))
    owner = _owner()
    for request_id in ("first", "second"):
        rj.asr2judge([_asr(request_id, "hi")], None, False, owner, target_model_config=config)
        assert rj._owned_turn(request_id, owner).spec is spec


def test_other_stages_do_not_load_judge_configuration(monkeypatch):
    config = _startup_model_config({"format": "clm", "threshold": 2}, monkeypatch, model_stage="other")
    assert not hasattr(config, "_response_judge_spec")


@pytest.fixture
def prepared_clm_options():
    # response_judge from a prepared CLM directory's config.json.
    return {
        "format": "clm",
        "instructions": "What should the voice assistant do next?",
        "option_keys": ["answer", "act", "ack", "other", "noise"],
        "reply_option": ["answer", "act"],
        "threshold": 0.300773,
        "state_template": "{transcript}",
    }


@pytest.mark.parametrize(
    ("name", "fmt"),
    [("aura_omni_judged", "chat_yes_no"), ("aura_omni_judged_laya", "laya"), ("aura_omni_judged_clm", "clm")],
)
def test_example_deploy_judge_overrides_remain_valid(monkeypatch, prepared_clm_options, name, fmt):
    import yaml

    path = Path(rj.__file__).parents[2] / "deploy" / f"{name}.yaml"
    deploy = yaml.safe_load(path.read_text())
    stage = next(s for s in deploy["stages"] if s["stage_id"] == 1)
    raw = stage["hf_overrides"].get("response_judge", prepared_clm_options)
    config = _startup_model_config(raw, monkeypatch)
    spec = rj.JudgeSpec.for_model_config(config)
    assert spec.format == fmt
    if fmt == "clm":
        assert spec.options["option_keys"] == ["answer", "act", "ack", "other", "noise"]
        assert spec.options["reply_option"] == ["answer", "act"]


@pytest.mark.parametrize(
    ("answer", "rejected"),
    [("NO", True), (" no\n", True), ("YES", False), ("", False), ("NO.", False), ("maybe", False)],
)
def test_chat_judge_rejects_only_a_clear_no(monkeypatch, answer, rejected):
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)
    owner = _owner()
    [judge_input] = rj.asr2judge([_asr("r1", "嗯嗯")], None, False, owner, target_model_config=config)
    assert "用户刚刚说：「嗯嗯」" in judge_input["prompt"]
    assert rj.judge_rejects(_judge_text("r1", answer), owner) is rejected


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
    assert rj._owned_turn("r1", owner) is not None
    assert rj.judge_rejects(_judge_text("r1", "NO"), owner) is False


def test_unknown_request_is_let_through():
    assert rj.judge_rejects(_judge_text("never-judged", "NO"), _owner()) is False


def test_judge_decisions_do_not_cross_request_owners_with_the_same_id(monkeypatch):
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)
    owner, other = _owner(), _owner()
    rj.asr2judge([_asr("same", "嗯嗯")], None, False, owner, target_model_config=config)
    # An empty transcript must pass through even when another owner records NO.
    rj.asr2judge([_asr("same", "")], None, False, other, target_model_config=config)
    output = _judge_text("same", "NO")
    assert rj.judge_rejects(output, owner) is True
    assert rj.judge_rejects(output, other) is False
    assert rj.judge_rejects(output, _owner()) is False


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
    assert len(seen) == 1


def test_a_repeated_forward_of_one_judge_output_gets_the_same_answer(monkeypatch):
    """Streaming requests forward a finished output twice (update, then terminal).

    Both calls must see the turn: the host bridge runs twice on the original
    ASR output, exactly as it would without a judge, and a rejected turn stays
    rejected.
    """
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)
    calls = []

    def asr2main(source_outputs, prompt=None, requires_multimodal_data=False):
        calls.append(source_outputs)
        return [{"prompt": source_outputs[0].outputs[0].text}]

    judged = rj.after_judge(asr2main)
    owner = _owner()
    asr = _asr("r1", "今天天气怎么样")
    rj.asr2judge([asr], None, False, owner, target_model_config=config)
    for _ in range(2):
        assert judged([_judge_text("r1", "YES")], None, False, owner) == [{"prompt": "今天天气怎么样"}]
    assert calls == [[asr], [asr]]

    rj.asr2judge([_asr("r2", "嗯嗯")], None, False, owner, target_model_config=config)
    for _ in range(2):
        assert judged([_judge_text("r2", "NO")], None, False, owner) == []
    assert len(calls) == 2


def test_the_wrapped_bridge_decodes_with_the_asr_decoder(monkeypatch):
    """The orchestrator swaps in the judge's decoder to forward the judge output;
    the wrapped bridge reads ASR outputs and must see the ASR decoder."""
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)

    def asr_decode(ids):
        return "asr"

    def judge_decode(ids):
        return "judge"

    seen = []

    def bridge(source_outputs, prompt=None, requires_multimodal_data=False, streaming_context=None):
        seen.append(streaming_context.source_token_decoder([1]))
        return [source_outputs[0].outputs[0].text]

    owner = SimpleNamespace(bridge_states={}, source_token_decoder=asr_decode)
    rj.asr2judge([_asr("r1", "今天天气怎么样")], None, False, owner, target_model_config=config)
    owner.source_token_decoder = judge_decode
    assert rj.after_judge(bridge)([_judge_text("r1", "YES")], None, False, owner) == ["今天天气怎么样"]
    assert seen == ["asr"]
    assert owner.source_token_decoder is judge_decode


def test_laya_prompt_follows_the_model_question_type(monkeypatch):
    tok = _LayaTokenizer()
    monkeypatch.setattr("vllm.tokenizers.cached_tokenizer_from_config", lambda config: tok)
    config = SimpleNamespace(hf_config=SimpleNamespace(response_judge={"format": "laya"}, laya_question_type="score"))
    [judge_input] = rj.asr2judge([_asr("r1", "hi")], None, False, _owner(), target_model_config=config)
    assert judge_input["prompt_token_ids"][1] == tok.vocab["score"]


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


def test_a_cached_decision_belongs_to_its_judge_output(monkeypatch):
    """Overlapping turns of one request id: ASR 1, ASR 2, judge 1 (YES), judge 2 (NO).

    A request id carries one turn at a time, so judge 1 decides the turn it
    finds (ASR 2). Judge 2 is a different output and gets its own decision
    instead of reusing judge 1's.
    """
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)
    judged = rj.after_judge(
        lambda source_outputs, prompt=None, requires_multimodal_data=False: [o.outputs[0].text for o in source_outputs]
    )
    owner = _owner()
    rj.asr2judge([_asr("r1", "第一句")], None, False, owner, target_model_config=config)
    rj.asr2judge([_asr("r1", "嗯嗯")], None, False, owner, target_model_config=config)
    assert judged([_judge_text("r1", "YES")], None, False, owner) == ["嗯嗯"]
    no = _judge_text("r1", "NO")
    assert judged([no], None, False, owner) == []
    assert judged([no], None, False, owner) == []


def test_a_prompt_that_cannot_be_built_records_no_turn(monkeypatch):
    config = _model_config({"format": "clm", "state_template": "User: {missing}"}, _ChatTokenizer(), monkeypatch)
    owner = _owner()
    with pytest.raises(KeyError):
        rj.asr2judge([_asr("r-bad", "hi")], None, False, owner, target_model_config=config)
    assert rj._owned_turn("r-bad", owner) is None
    assert owner.bridge_states.get("response_judge", {}) == {}


def test_pending_turn_is_released_with_the_request_owner(monkeypatch):
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)
    owner = _owner()
    rj.asr2judge([_asr("r-owned", "嗯嗯")], None, False, owner, target_model_config=config)
    assert rj.judge_rejects(_judge_text("r-owned", "NO"), owner) is True
    pending = weakref.ref(rj._owned_turn("r-owned", owner))
    del owner
    gc.collect()
    assert pending() is None


def test_judge_requires_request_owned_bridge_state(monkeypatch):
    config = _model_config({"format": "chat_yes_no"}, _ChatTokenizer(), monkeypatch)
    with pytest.raises(RuntimeError, match="bridge state"):
        rj.asr2judge([_asr("r1", "hi")], None, False, None, target_model_config=config)


def test_judge_decision_requires_request_owned_bridge_state():
    with pytest.raises(RuntimeError, match="bridge state"):
        rj.judge_rejects(_judge_text("r1", "NO"), None)


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
    assert rj._owned_turn("r1", owner) is not None
    assert rj.judge_rejects(_judge_pooled("r1", logits), owner) is rejected


def test_unreadable_pooling_output_is_let_through(monkeypatch):
    config = _model_config({"format": "laya"}, _LayaTokenizer(), monkeypatch)
    owner = _owner()
    rj.asr2judge([_asr("r1", "嗯嗯")], None, False, owner, target_model_config=config)
    assert rj._owned_turn("r1", owner) is not None
    assert rj.judge_rejects(SimpleNamespace(request_id="r1", outputs=[SimpleNamespace(text="")]), owner) is False


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([float("nan"), 3.0]),
        torch.tensor([0.0, float("inf")]),
        torch.tensor([[0.0], [3.0]]),
        torch.tensor([0.0, 3.0], dtype=torch.complex64),
        torch.tensor([False, True]),
    ],
    ids=["nan", "inf", "column", "complex", "bool"],
)
def test_pooled_scores_that_cannot_be_read_let_the_turn_through(monkeypatch, logits):
    config = _model_config(
        {"format": "laya", "options": {"yes": "reply", "no": "quiet"}, "reply_option": "yes"},
        _LayaTokenizer(),
        monkeypatch,
    )
    owner = _owner()
    rj.asr2judge([_asr("r1", "嗯嗯")], None, False, owner, target_model_config=config)
    output = SimpleNamespace(request_id="r1", finished=True, outputs=SimpleNamespace(data=logits))
    assert rj.judge_rejects(output, owner) is False


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
    assert rj.judge_rejects(_judge_pooled("r1", logits), owner) is rejected


def test_multimodal_carrier_with_one_tensor_is_read(monkeypatch):
    config = _model_config({"format": "laya", "options": {"yes": "", "no": ""}}, _LayaTokenizer(), monkeypatch)
    owner = _owner()
    rj.asr2judge([_asr("r1", "嗯嗯")], None, False, owner, target_model_config=config)
    output = SimpleNamespace(
        request_id="r1", outputs=[SimpleNamespace(text="")], multimodal_output={"text": torch.tensor([0.0, 3.0])}
    )
    assert rj.judge_rejects(output, owner) is True


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
