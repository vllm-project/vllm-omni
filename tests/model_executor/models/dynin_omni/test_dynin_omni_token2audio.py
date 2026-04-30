from __future__ import annotations

from transformers import PreTrainedModel

from vllm_omni.model_executor.models.dynin_omni.dynin_omni_token2audio import (
    _ensure_transformers_tied_weights_compat,
)


def test_transformers_tied_weights_compat(monkeypatch):
    monkeypatch.delattr(PreTrainedModel, "all_tied_weights_keys", raising=False)

    _ensure_transformers_tied_weights_compat()

    model = PreTrainedModel.__new__(PreTrainedModel)
    model._tied_weights_keys = ["decoder.weight"]
    model._dynamic_tied_weights_keys = {"lm_head.weight": "decoder.weight"}

    assert model.all_tied_weights_keys == {
        "decoder.weight": None,
        "lm_head.weight": "decoder.weight",
    }
