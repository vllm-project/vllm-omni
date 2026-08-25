from __future__ import annotations

import pytest
from transformers import PreTrainedModel

from vllm_omni.model_executor.models.dynin_omni.dynin_omni_token2audio import (
    _ensure_transformers_tied_weights_compat,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_transformers_tied_weights_compat(monkeypatch):
    """Compat shim must not permanently mutate ``PreTrainedModel``.

    ``monkeypatch.delattr(..., raising=False)`` is a no-op undo when the attr
    was absent. ``_ensure_transformers_tied_weights_compat`` then assigns a
    read-only ``property`` on the class that later PreTrainedModel subclasses
    (e.g. qwen3 TTS mask-cache) can trip over with
    ``AttributeError: property ... has no setter``.
    """
    monkeypatch.delattr(PreTrainedModel, "all_tied_weights_keys", raising=False)

    _ensure_transformers_tied_weights_compat()
    installed = PreTrainedModel.__dict__.get("all_tied_weights_keys")
    assert installed is not None

    # Re-own the installed descriptor through monkeypatch so teardown always
    # removes/restores it, even when the attr did not exist before this test.
    delattr(PreTrainedModel, "all_tied_weights_keys")
    monkeypatch.setattr(PreTrainedModel, "all_tied_weights_keys", installed, raising=False)

    model = PreTrainedModel.__new__(PreTrainedModel)
    model._tied_weights_keys = ["decoder.weight"]
    model._dynamic_tied_weights_keys = {"lm_head.weight": "decoder.weight"}

    assert model.all_tied_weights_keys == {
        "decoder.weight": None,
        "lm_head.weight": "decoder.weight",
    }
