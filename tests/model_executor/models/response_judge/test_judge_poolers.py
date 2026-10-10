# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU checks of the pooling judges' heads against their reference maths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.response_judge.clm import ClmDecisionModel, ClmDecisionPooler, ClmHead
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


def test_clm_pooler_matches_the_reference_contrastive_score():
    torch.manual_seed(0)
    cfg = {"hidden": 16, "width": 8, "depth": 3, "proj": 4, "activation": "gelu", "layernorm": True, "residual": True}
    pooler = ClmDecisionPooler(cfg, num_options=3)
    action_head = ClmHead(**cfg)
    options = torch.randn(3, 16)
    with torch.no_grad():
        pooler.option_proj.copy_(F.normalize(action_head(F.normalize(options, dim=-1)), dim=-1))
        pooler.scale.fill_(20.0)
    hidden = torch.randn(7, 16)  # two requests: 3 and 4 tokens
    got = pooler(hidden, _metadata([3, 4]))
    for out, last in zip(got, (hidden[2], hidden[6])):
        zs = F.normalize(pooler.state_head(F.normalize(last, dim=-1)), dim=-1)
        expected = 20.0 * (pooler.option_proj @ zs)
        torch.testing.assert_close(out, expected)


def test_clm_head_parameter_names_match_the_checkpoint_layout():
    names = set(dict(ClmHead(hidden=8, width=4, depth=3, layernorm=True).named_parameters()))
    assert {"inp.weight", "hidden.0.weight", "norms.0.weight", "out.weight"} <= names


def _weight_loader(kind, monkeypatch):
    backbone_calls = []

    def load_backbone(weights):
        weights = list(weights)
        backbone_calls.append(weights)
        return {name for name, _ in weights}

    if kind == "clm":
        model = object.__new__(ClmDecisionModel)
        torch.nn.Module.__init__(model)
        model.pooler = ClmDecisionPooler({"hidden": 8, "width": 4, "depth": 3, "proj": 4, "layernorm": True}, 2)
        # Exercise the real head loader without constructing the Qwen backbone.
        monkeypatch.setattr(ClmDecisionModel.__mro__[1], "load_weights", lambda self, weights: load_backbone(weights))
    else:
        model = object.__new__(LayaDecisionModel)
        torch.nn.Module.__init__(model)
        model.pooler = LayaDecisionPooler(hidden_size=64, head_layers=1, mask_token_id=4, qtype=0)
        model.encoder = SimpleNamespace(load_weights=load_backbone)
    return model, backbone_calls


@pytest.mark.parametrize("kind", ["clm", "laya"])
def test_judge_loaders_copy_exact_shapes_and_forward_backbone_weights(monkeypatch, kind):
    model, backbone_calls = _weight_loader(kind, monkeypatch)
    params = dict(model.pooler.named_parameters())
    expected = {
        name: torch.full(param.shape, i + 1.0, dtype=torch.float64) for i, (name, param) in enumerate(params.items())
    }
    prefix = "clm." if kind == "clm" else ""
    backbone = torch.ones(2, 2)
    weights = [(prefix + name, tensor) for name, tensor in expected.items()]
    weights.append(("encoder.layer.weight", backbone))
    loaded = model.load_weights(iter(weights))
    assert loaded == {f"pooler.{name}" for name in params} | {"encoder.layer.weight"}
    expected_backbone_name = "encoder.layer.weight" if kind == "clm" else "layer.weight"
    assert len(backbone_calls) == 1
    [(name, tensor)] = backbone_calls[0]
    assert name == expected_backbone_name and tensor is backbone
    for name, param in params.items():
        torch.testing.assert_close(param, expected[name].to(param.dtype))
    if kind == "clm":
        # The preparation script writes clm.scale as torch.tensor(scale), shape ().
        assert model.pooler.scale.shape == torch.Size([])


@pytest.mark.parametrize(
    ("kind", "name", "shape"),
    [
        ("clm", "state_head.inp.weight", (8, 4)),
        ("clm", "scale", (1,)),
        ("laya", "type_emb.weight", (1, 64)),
        ("laya", "scorer.3.bias", ()),
    ],
)
def test_judge_loaders_reject_reshaped_or_broadcastable_head_weights(monkeypatch, kind, name, shape):
    model, backbone_calls = _weight_loader(kind, monkeypatch)
    param = dict(model.pooler.named_parameters())[name]
    before = param.detach().clone()
    weight_name = f"clm.{name}" if kind == "clm" else name
    tensor = torch.ones(shape)
    with pytest.raises(ValueError) as exc:
        model.load_weights(iter([(weight_name, tensor)]))
    message = str(exc.value)
    assert weight_name in message
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


def test_clm_pooler_scores_a_prefix_cache_hit_at_the_prompts_last_token():
    torch.manual_seed(0)
    cfg = {"hidden": 16, "width": 8, "depth": 2, "proj": 4}
    pooler = ClmDecisionPooler(cfg, num_options=2)
    # 64-token prompt, 48 cached: only 16 scheduled this step, request finished.
    md = _vllm_metadata([64], [16], [64])
    cursor = md.get_pooling_cursor()
    assert cursor.is_partial_prefill() and cursor.get_finished_mask() == [True]
    hidden = torch.randn(16, 16)
    [out] = pooler(hidden, md)
    zs = F.normalize(pooler.state_head(F.normalize(hidden[15], dim=-1)), dim=-1)
    torch.testing.assert_close(out, pooler.scale * (F.normalize(pooler.option_proj, dim=-1) @ zs))


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
