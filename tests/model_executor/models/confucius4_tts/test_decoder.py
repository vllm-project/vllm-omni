# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.confucius4_tts import (
    confucius4_tts_decoder as decoder_module,
)
from vllm_omni.model_executor.models.confucius4_tts.configuration_confucius4_tts import (
    Confucius4TTSConfig,
)
from vllm_omni.model_executor.models.confucius4_tts.confucius4_tts_decoder import (
    Confucius4TTSS2ADecoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _decoder() -> Confucius4TTSS2ADecoder:
    decoder = object.__new__(Confucius4TTSS2ADecoder)
    nn.Module.__init__(decoder)
    decoder.config = Confucius4TTSConfig()
    return decoder


class _FakeRecompute(nn.Module):
    """Records its inputs and returns a latent of the right shape."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.call: tuple[Any, ...] | None = None

    def forward(self, text_ids, condition, semantic, bos, eos):
        self.call = (text_ids, condition, semantic, bos, eos)
        return torch.ones(semantic.shape[0], 1280)


def _conditioning() -> dict[str, torch.Tensor]:
    return {
        "text_token_ids": torch.tensor([5, 6, 7]),
        "condition_vector": torch.ones(1, 4, 1024),
        "ref_mel": torch.ones(1, 3, 80),
        "style": torch.ones(1, 192),
    }


def test_seed_context_supports_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.accelerator,
        "current_device_index",
        lambda: pytest.fail("CPU seed context queried the accelerator"),
    )
    decoder = object.__new__(Confucius4TTSS2ADecoder)
    before = torch.random.get_rng_state()
    with decoder._seed_context(torch.device("cpu"), 7):
        torch.manual_seed(7)
        first = torch.rand(3)
    after = torch.random.get_rng_state()

    torch.random.set_rng_state(before)
    with decoder._seed_context(torch.device("cpu"), 7):
        torch.manual_seed(7)
        second = torch.rand(3)
    assert torch.equal(first, second)
    assert torch.equal(after, before)


def test_prepare_request_recomputes_latent_from_final_sequence() -> None:
    decoder = _decoder()
    decoder.t2s_recompute = _FakeRecompute()
    valid = {"semantic_codes": torch.tensor([1, 2]), **_conditioning()}

    semantic, latent, ref_mel, style, seed = decoder._prepare_request(
        valid,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert semantic.shape == (1, 2)
    assert latent.shape == (1, 2, 1280)
    assert ref_mel.shape == (1, 3, 80)
    assert style.shape == (1, 192)
    assert seed is None
    text_ids, condition, codes, bos, eos = decoder.t2s_recompute.call
    assert text_ids.tolist() == [5, 6, 7]
    assert condition.shape == (1, 4, 1024)
    assert codes.tolist() == [1, 2]
    assert (bos, eos) == (8192, 8193)


def test_prepare_request_validates_payload_ranges_and_conditioning() -> None:
    decoder = _decoder()
    decoder.t2s_recompute = _FakeRecompute()
    valid = {"semantic_codes": torch.tensor([1, 2]), **_conditioning()}

    for bad in (torch.tensor([1, 8192]), torch.tensor([102, -1])):
        decoder.t2s_recompute.call = None
        with pytest.raises(ValueError, match="outside"):
            decoder._prepare_request(
                {**valid, "semantic_codes": bad},
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        # Range checks run before the latent recompute embeds the ids; on CUDA
        # an out-of-range lookup is a device-side assert that kills the engine.
        assert decoder.t2s_recompute.call is None
    for key in ("text_token_ids", "condition_vector", "ref_mel", "style"):
        with pytest.raises(ValueError, match=f"requires non-empty {key}"):
            decoder._prepare_request(
                {k: v for k, v in valid.items() if k != key},
                device=torch.device("cpu"),
                dtype=torch.float32,
            )


def test_forward_without_payload_returns_profile_output() -> None:
    decoder = _decoder()

    output = decoder.forward(torch.tensor([0]))

    assert output.text_hidden_states.shape == (1,)
    assert output.multimodal_outputs["audio"].shape == (1,)
    assert output.multimodal_outputs["sr"] == 22050


class _FakeS2A(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.call: dict[str, Any] | None = None

    def inference(self, **kwargs):
        self.call = kwargs
        return torch.ones(1, 80, 5)


class _FakeBigVGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = None

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        self.input = mel
        return torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)


def test_forward_invokes_s2a_and_bigvgan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _decoder()
    decoder.model_path = "/model"
    decoder._resolved_vocoder_source = None
    decoder.s2a_model = _FakeS2A()
    decoder.t2s_recompute = _FakeRecompute()
    vocoder = _FakeBigVGAN()
    monkeypatch.setattr(
        decoder_module,
        "_resolve_bigvgan_source",
        lambda *_args: "/bigvgan",
    )
    monkeypatch.setattr(
        decoder_module,
        "_load_bigvgan",
        lambda *_args: vocoder,
    )
    info = {"semantic_codes": torch.tensor([1, 2]), **_conditioning(), "seed": 7}

    output = decoder.forward(
        torch.tensor([0]),
        model_intermediate_buffer=[info],
    )

    assert decoder._resolved_vocoder_source == "/bigvgan"
    assert decoder.s2a_model.call is not None
    assert decoder.s2a_model.call["target_feat_len"].tolist() == [3]
    assert vocoder.input is not None
    assert output.multimodal_outputs["model_outputs"][0].tolist() == list(range(8))
    assert output.multimodal_outputs["sr"][0].item() == 22050


def test_load_weights_uses_safe_tensor_only_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _decoder()
    decoder.model_path = "/model"
    decoder.s2a_model = nn.Linear(1, 1, bias=False)
    recompute = Mock()
    recompute.load_state.return_value = {"semantic_embedding.weight"}
    decoder.t2s_recompute = recompute
    load_call: dict[str, Any] = {}
    talker_state = {"semantic_embedding.weight": torch.ones(1)}

    monkeypatch.setattr(
        decoder_module,
        "resolve_model_file",
        lambda _path, name: f"/model/{name}",
    )

    def fake_torch_load(path: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        load_call["path"] = path
        load_call.update(kwargs)
        return {"weight": torch.ones(1, 1)}

    monkeypatch.setattr(torch, "load", fake_torch_load)
    monkeypatch.setattr(decoder_module, "load_file", lambda *_a, **_k: talker_state)

    loaded = decoder.load_weights([])

    assert load_call == {
        "path": "/model/s2a_model.pt",
        "map_location": "cpu",
        "weights_only": True,
    }
    recompute.load_state.assert_called_once_with(talker_state)
    assert loaded == {"s2a_model.weight", "t2s_recompute.semantic_embedding.weight"}


def test_load_weights_requires_talker_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _decoder()
    decoder.model_path = "/model"
    decoder.s2a_model = nn.Linear(1, 1, bias=False)
    monkeypatch.setattr(
        decoder_module,
        "resolve_model_file",
        lambda _path, name: None if name.endswith(".safetensors") else f"/model/{name}",
    )
    monkeypatch.setattr(torch, "load", lambda *_a, **_k: {"weight": torch.ones(1, 1)})

    with pytest.raises(FileNotFoundError, match="t2s_model.safetensors"):
        decoder.load_weights([])
