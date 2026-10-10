# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Talker-stage ``CosyVoice3Model.load_weights``: iterator, partial reloads, file fallback."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model

_HIDDEN = 4
_SPEECH_VOCAB = 6


@functools.lru_cache(maxsize=1)
def _cosyvoice3_model_cls():
    """Defer heavy Omni/vLLM imports until a test runs (avoids duplicate CustomOp init)."""
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model

    return CosyVoice3Model


class _FakeQwen2(nn.Module):
    """Stands in for vLLM's Qwen2 model: records what its loader is handed."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(_HIDDEN)
        self.load_calls: list[list[str]] = []

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        names = []
        for name, tensor in weights:
            names.append(name)
            params[name].data.copy_(tensor)
        self.load_calls.append(names)


class _FakeTalker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.speech_embedding = nn.Embedding(_SPEECH_VOCAB, _HIDDEN)
        self.llm_decoder = nn.Linear(_HIDDEN, _SPEECH_VOCAB)
        self.llm = nn.Module()
        self.llm.model = _FakeQwen2()


def _make_talker_model(model_dir) -> CosyVoice3Model:
    model = object.__new__(_cosyvoice3_model_cls())
    nn.Module.__init__(model)
    model.model_stage = "cosyvoice3_talker"
    model.model_dir = str(model_dir)
    model.model = _FakeTalker()
    return model


def _checkpoint(fill: float) -> dict[str, torch.Tensor]:
    """A full ``llm.pt``-schema checkpoint with every tensor set to ``fill``."""
    return {
        "speech_embedding.weight": torch.full((_SPEECH_VOCAB, _HIDDEN), fill),
        "llm_decoder.weight": torch.full((_SPEECH_VOCAB, _HIDDEN), fill),
        "llm_decoder.bias": torch.full((_SPEECH_VOCAB,), fill),
        "llm.model.model.norm.weight": torch.full((_HIDDEN,), fill),
        "llm.model.model.norm.bias": torch.full((_HIDDEN,), fill),
        "llm.model.lm_head.weight": torch.full((3, _HIDDEN), fill),
    }


def _write_llm_pt(model_dir, fill: float) -> None:
    torch.save(_checkpoint(fill), model_dir / "llm.pt")


def _count_torch_load(monkeypatch) -> list[str]:
    """Record every ``torch.load`` path while still performing the load."""
    calls: list[str] = []
    real_load = torch.load

    def _spy(path, *args, **kwargs):
        calls.append(str(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy)
    return calls


def test_iterator_weights_are_loaded_without_reading_the_checkpoint_file(tmp_path, monkeypatch):
    # No llm.pt exists: reaching the file fallback would raise.
    model = _make_talker_model(tmp_path)
    load_calls = _count_torch_load(monkeypatch)

    # A generator, as the default loader hands over, not a list.
    assert model.load_weights(item for item in _checkpoint(2.0).items()) is None

    talker = model.model
    assert load_calls == []
    assert torch.all(talker.speech_embedding.weight == 2.0)
    assert torch.all(talker.llm_decoder.weight == 2.0)
    assert torch.all(talker.llm_decoder.bias == 2.0)
    assert torch.all(talker.llm.model.norm.weight == 2.0)
    # The transformer prefix is stripped and the unused text lm_head is skipped.
    assert talker.llm.model.load_calls == [["norm.weight", "norm.bias"]]
    assert not talker.training


def test_partial_reload_updates_only_the_named_tensors_in_place(tmp_path):
    model = _make_talker_model(tmp_path)
    model.load_weights(_checkpoint(1.0).items())
    talker = model.model
    embedding_param = talker.speech_embedding.weight
    embedding_storage = embedding_param.data_ptr()

    model.load_weights([("speech_embedding.weight", torch.full((_SPEECH_VOCAB, _HIDDEN), 5.0))])

    assert torch.all(talker.speech_embedding.weight == 5.0)
    # Same Parameter and same storage, so captured CUDA graphs stay valid.
    assert talker.speech_embedding.weight is embedding_param
    assert talker.speech_embedding.weight.data_ptr() == embedding_storage
    # Everything not named in the update keeps its earlier value.
    assert torch.all(talker.llm_decoder.weight == 1.0)
    assert torch.all(talker.llm_decoder.bias == 1.0)
    assert torch.all(talker.llm.model.norm.weight == 1.0)
    # A talker-only update never calls the transformer loader again.
    assert len(talker.llm.model.load_calls) == 1


def test_partial_reload_can_split_one_module_across_calls(tmp_path):
    model = _make_talker_model(tmp_path)
    model.load_weights(_checkpoint(1.0).items())
    talker = model.model

    model.load_weights([("llm.model.model.norm.weight", torch.full((_HIDDEN,), 7.0))])
    model.load_weights([("llm_decoder.bias", torch.full((_SPEECH_VOCAB,), 8.0))])

    assert torch.all(talker.llm.model.norm.weight == 7.0)
    assert torch.all(talker.llm.model.norm.bias == 1.0)
    assert torch.all(talker.llm_decoder.bias == 8.0)
    assert torch.all(talker.llm_decoder.weight == 1.0)
    assert talker.llm.model.load_calls[1:] == [["norm.weight"]]


def test_empty_iterator_falls_back_to_the_checkpoint_file_once(tmp_path, monkeypatch):
    _write_llm_pt(tmp_path, 3.0)
    model = _make_talker_model(tmp_path)
    load_calls = _count_torch_load(monkeypatch)

    # Dummy load format: the loader hands over an empty iterator.
    model.load_weights(iter(()))

    talker = model.model
    assert load_calls == [str(tmp_path / "llm.pt")]
    assert torch.all(talker.speech_embedding.weight == 3.0)
    assert torch.all(talker.llm.model.norm.bias == 3.0)

    # A runtime update, then another empty call: the update must survive and
    # the on-disk checkpoint must not be read a second time.
    model.load_weights([("speech_embedding.weight", torch.full((_SPEECH_VOCAB, _HIDDEN), 9.0))])
    model.load_weights(iter(()))

    assert load_calls == [str(tmp_path / "llm.pt")]
    assert torch.all(talker.speech_embedding.weight == 9.0)
    assert torch.all(talker.llm_decoder.weight == 3.0)


def test_empty_iterator_after_an_iterator_load_does_not_read_the_file(tmp_path, monkeypatch):
    # llm.pt holds different values; reading it would overwrite the loaded ones.
    _write_llm_pt(tmp_path, 3.0)
    model = _make_talker_model(tmp_path)
    load_calls = _count_torch_load(monkeypatch)

    model.load_weights(_checkpoint(4.0).items())
    model.load_weights(iter(()))

    assert load_calls == []
    assert torch.all(model.model.speech_embedding.weight == 4.0)
    assert torch.all(model.model.llm.model.norm.weight == 4.0)


def test_first_load_missing_decoder_tensors_is_rejected_and_stays_incomplete(tmp_path, monkeypatch):
    _write_llm_pt(tmp_path, 3.0)
    model = _make_talker_model(tmp_path)
    load_calls = _count_torch_load(monkeypatch)
    talker = model.model
    embedding_before = talker.speech_embedding.weight.detach().clone()

    with pytest.raises(ValueError, match=r"missing required tensors: \['llm_decoder.bias', 'llm_decoder.weight'\]"):
        model.load_weights([("speech_embedding.weight", torch.full((_SPEECH_VOCAB, _HIDDEN), 5.0))])

    # Nothing was copied and the load is not marked complete, so a later
    # empty call still initializes from the checkpoint file.
    assert torch.equal(talker.speech_embedding.weight, embedding_before)
    assert not getattr(model, "_talker_weights_loaded", False)

    model.load_weights(iter(()))

    assert load_calls == [str(tmp_path / "llm.pt")]
    assert torch.all(talker.speech_embedding.weight == 3.0)
    assert torch.all(talker.llm_decoder.weight == 3.0)
    assert torch.all(talker.llm_decoder.bias == 3.0)


def test_checkpoint_file_missing_decoder_tensors_is_rejected(tmp_path):
    checkpoint = _checkpoint(3.0)
    del checkpoint["llm_decoder.weight"]
    torch.save(checkpoint, tmp_path / "llm.pt")
    model = _make_talker_model(tmp_path)

    with pytest.raises(ValueError, match=r"missing required tensors: \['llm_decoder.weight'\]"):
        model.load_weights(iter(()))

    assert not getattr(model, "_talker_weights_loaded", False)


def test_broadcastable_but_unequal_shape_is_rejected_without_copying(tmp_path):
    model = _make_talker_model(tmp_path)
    model.load_weights(_checkpoint(1.0).items())
    talker = model.model

    # [1, 4] broadcasts into the [6, 4] embedding, so copy_ alone would
    # accept it and overwrite every row.
    with pytest.raises(ValueError, match=r"shape mismatch: speech_embedding.weight: checkpoint \(1, 4\) vs"):
        model.load_weights(
            [
                ("llm_decoder.bias", torch.full((_SPEECH_VOCAB,), 8.0)),
                ("speech_embedding.weight", torch.full((1, _HIDDEN), 5.0)),
            ]
        )

    assert torch.all(talker.speech_embedding.weight == 1.0)
    # The valid tensor in the same call is not applied either.
    assert torch.all(talker.llm_decoder.bias == 1.0)


def test_first_load_with_unequal_shape_is_rejected(tmp_path):
    model = _make_talker_model(tmp_path)
    checkpoint = _checkpoint(2.0)
    checkpoint["llm_decoder.bias"] = torch.full((1,), 2.0)

    with pytest.raises(ValueError, match=r"shape mismatch: llm_decoder.bias: checkpoint \(1,\) vs parameter \(6,\)"):
        model.load_weights(checkpoint.items())

    assert not getattr(model, "_talker_weights_loaded", False)


def test_unexpected_tensor_name_is_rejected(tmp_path):
    model = _make_talker_model(tmp_path)

    with pytest.raises(ValueError, match="unexpected CosyVoice3 talker checkpoint tensor 'input_embedding.weight'"):
        model.load_weights([("input_embedding.weight", torch.zeros(2, 2))])
