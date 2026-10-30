# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU checks of the pooling judges' heads against their reference maths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.response_judge.laya import (
    LayaDecisionModel,
    LayaDecisionPooler,
    laya_question_type,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _metadata(lengths, token_ids=None):
    lens = torch.tensor(lengths)
    cursor = SimpleNamespace(
        num_scheduled_tokens_cpu=lens,
        last_token_indices_gpu=torch.cumsum(lens, 0) - 1,
        is_partial_prefill=lambda: False,
    )
    return SimpleNamespace(get_pooling_cursor=lambda: cursor, prompt_token_ids=None, prompt_token_ids_cpu=token_ids)


def _weight_loader():
    backbone_calls = []

    def load_backbone(weights):
        weights = list(weights)
        backbone_calls.append(weights)
        return {name for name, _ in weights}

    model = object.__new__(LayaDecisionModel)
    torch.nn.Module.__init__(model)
    model.pooler = LayaDecisionPooler(hidden_size=64, head_layers=1, mask_token_id=4, qtype=0)
    model.encoder = SimpleNamespace(load_weights=load_backbone)
    return model, backbone_calls


def test_laya_loader_copies_exact_shapes_and_forwards_backbone_weights():
    model, backbone_calls = _weight_loader()
    params = dict(model.pooler.named_parameters())
    expected = {
        name: torch.full(param.shape, i + 1.0, dtype=torch.float64) for i, (name, param) in enumerate(params.items())
    }
    backbone = torch.ones(2, 2)
    weights = [*expected.items(), ("encoder.layer.weight", backbone)]
    loaded = model.load_weights(iter(weights))
    assert loaded == {f"pooler.{name}" for name in params} | {"encoder.layer.weight"}
    assert len(backbone_calls) == 1
    [(name, tensor)] = backbone_calls[0]
    assert name == "layer.weight" and tensor is backbone
    for name, param in params.items():
        torch.testing.assert_close(param, expected[name].to(param.dtype))


@pytest.mark.parametrize(("name", "shape"), [("type_emb.weight", (1, 64)), ("scorer.3.bias", ())])
def test_laya_loader_rejects_reshaped_or_broadcastable_head_weights(name, shape):
    model, backbone_calls = _weight_loader()
    param = dict(model.pooler.named_parameters())[name]
    before = param.detach().clone()
    tensor = torch.ones(shape)
    with pytest.raises(ValueError) as exc:
        model.load_weights(iter([(name, tensor)]))
    message = str(exc.value)
    assert name in message
    assert f"has shape {shape}" in message
    assert f"expected {tuple(param.shape)}" in message
    torch.testing.assert_close(param, before)
    assert backbone_calls == []


def test_laya_pooler_scores_each_request_at_its_own_mask_markers():
    torch.manual_seed(0)
    pooler = LayaDecisionPooler(hidden_size=64, head_layers=1, mask_token_id=4, qtype=0).eval()
    token_ids = torch.tensor([[1, 4, 7, 4, 8, 1, 0], [1, 9, 4, 5, 4, 6, 1]])
    hidden = torch.randn(13, 64)
    got = pooler(hidden, _metadata([6, 7], token_ids))
    assert [tuple(o.shape) for o in got] == [(2,), (2,)]
    with torch.no_grad():
        x = pooler.head((hidden[:6] + pooler.type_emb.weight[0]).unsqueeze(0))[0]
        torch.testing.assert_close(got[0], pooler.scorer(x[[1, 3]]).squeeze(-1))


def _vllm_metadata(prompt_lens, scheduled, seq_lens, token_ids_cpu=None):
    """PoolingMetadata as vLLM 0.30's input batch builds it (token ids only on the CPU copy)."""
    import numpy as np
    from vllm.pooling_params import PoolingParams
    from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

    md = PoolingMetadata(
        prompt_lens=torch.tensor(prompt_lens),
        prompt_token_ids=None,
        prompt_token_ids_cpu=token_ids_cpu,
        pooling_params=[PoolingParams(task="classify") for _ in prompt_lens],
        pooling_states=[PoolingStates() for _ in prompt_lens],
    )
    md.build_pooling_cursor(np.array(scheduled), torch.tensor(seq_lens), torch.device("cpu"))
    return md


def test_laya_pooler_reads_the_cpu_token_ids_vllm_provides():
    torch.manual_seed(0)
    pooler = LayaDecisionPooler(hidden_size=64, head_layers=1, mask_token_id=4, qtype=0).eval()
    token_ids = torch.tensor([[1, 4, 7, 4, 8, 1, 99], [1, 9, 4, 5, 4, 6, 1]])
    md = _vllm_metadata([6, 7], [6, 7], [6, 7], token_ids_cpu=token_ids)
    assert md.prompt_token_ids is None
    got = pooler(torch.randn(13, 64), md)
    assert [tuple(o.shape) for o in got] == [(2,), (2,)]


@pytest.mark.parametrize(
    ("model_qtype", "judge_qtype", "expected"),
    [(None, None, "choice"), ("score", None, "score"), (None, "score", "score"), ("choice", "choice", "choice")],
)
def test_laya_question_type_has_one_value(model_qtype, judge_qtype, expected):
    config = SimpleNamespace(response_judge={} if judge_qtype is None else {"question_type": judge_qtype})
    if model_qtype is not None:
        config.laya_question_type = model_qtype
    assert laya_question_type(config) == expected


def test_laya_question_types_that_disagree_fail_at_load():
    config = SimpleNamespace(laya_question_type="choice", response_judge={"question_type": "score"})
    with pytest.raises(ValueError, match="does not match"):
        laya_question_type(config)


@pytest.mark.parametrize("qtype", [0, 1, True, "unknown"])
def test_model_question_type_rejects_non_names(qtype):
    with pytest.raises(ValueError, match="question_type"):
        laya_question_type(SimpleNamespace(laya_question_type=qtype))


@pytest.mark.parametrize(
    ("budget", "chunked", "valid"),
    [(63, True, False), (63, False, False), (64, True, False), (64, False, True), (128, False, True)],
)
def test_laya_checks_prefill_budget_before_constructing_encoder(monkeypatch, budget, chunked, valid):
    import vllm_omni.model_executor.models.response_judge.laya as module

    class EncoderReachedError(Exception):
        pass

    def encoder(**kwargs):
        raise EncoderReachedError

    monkeypatch.setattr(module, "ModernBertModel", encoder)
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=64, hf_config=SimpleNamespace()),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=budget, enable_chunked_prefill=chunked),
    )
    if valid:
        with pytest.raises(EncoderReachedError):
            LayaDecisionModel(vllm_config=config)
    else:
        with pytest.raises(ValueError, match="enable_chunked_prefill" if chunked else "max_num_batched_tokens"):
            LayaDecisionModel(vllm_config=config)
