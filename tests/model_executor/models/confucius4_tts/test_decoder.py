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


def test_forward_returns_empty_audio_for_an_empty_semantic_sequence_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An empty request must not abort the batch or touch the S2A model, and a
    # normal request batched with it must still be synthesized.
    decoder = _decoder()
    decoder.model_path = "/model"
    decoder._resolved_vocoder_source = None
    decoder.s2a_model = _FakeS2A()
    decoder.t2s_recompute = _FakeRecompute()
    monkeypatch.setattr(decoder_module, "_resolve_bigvgan_source", lambda *_args: "/bigvgan")
    monkeypatch.setattr(decoder_module, "_load_bigvgan", lambda *_args: _FakeBigVGAN())
    empty = {"semantic_codes": torch.empty(0, dtype=torch.long)}
    normal = {"semantic_codes": torch.tensor([1, 2]), **_conditioning(), "seed": 7}

    output = decoder.forward(torch.tensor([0, 0]), model_intermediate_buffer=[empty, normal])

    waveforms = output.multimodal_outputs["model_outputs"]
    assert waveforms[0].numel() == 0
    assert waveforms[1].tolist() == list(range(8))
    assert len(output.multimodal_outputs["sr"]) == 2
    assert decoder.s2a_model.call["target_feat_len"].tolist() == [3]


def _forward_with(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_codes: int,
    ref_frames: int,
    max_positions: int | None = None,
):
    decoder = _decoder()
    if max_positions is not None:
        decoder.config.s2a_max_positions = max_positions
    decoder.model_path = "/model"
    decoder._resolved_vocoder_source = None
    decoder.s2a_model = _FakeS2A()
    decoder.t2s_recompute = _FakeRecompute()
    monkeypatch.setattr(decoder_module, "_resolve_bigvgan_source", lambda *_args: "/bigvgan")
    monkeypatch.setattr(decoder_module, "_load_bigvgan", lambda *_args: _FakeBigVGAN())
    ref_mel = torch.arange(ref_frames, dtype=torch.float32).reshape(1, ref_frames, 1).expand(1, ref_frames, 80)
    info = {
        **_conditioning(),
        "semantic_codes": torch.zeros(num_codes, dtype=torch.long),
        "ref_mel": ref_mel.contiguous(),
    }
    output = decoder.forward(torch.tensor([0]), model_intermediate_buffer=[info])
    return decoder, output


def test_reference_is_trimmed_to_fit_the_s2a_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 30 s reference (~2585 frames) next to the full 1518-token budget (2610 frames)
    # overflows the 4096-row rotary table; that crashed the stage in a worker.
    decoder, output = _forward_with(monkeypatch, num_codes=1518, ref_frames=2585)

    prompt = decoder.s2a_model.call["prompt_feat"]
    target = decoder.s2a_model.call["target_feat_len"].item()
    assert target == 2610
    assert prompt.shape[1] == 4096 - 2610
    assert prompt.shape[1] + target <= decoder.config.s2a_max_positions
    # The frames adjacent to the target are kept.
    assert prompt[0, -1, 0].item() == 2584
    assert prompt[0, 0, 0].item() == 2585 - (4096 - 2610)
    assert output.multimodal_outputs["model_outputs"][0].numel() == 8


@pytest.mark.parametrize(
    ("ref_frames", "expected_prompt"),
    [
        (100, 100),  # well inside the table
        (4096 - 1720, 4096 - 1720),  # exactly full: 1000 codes -> 1720 frames
        (4096 - 1720 + 1, 4096 - 1720),  # one frame over is trimmed by one
    ],
)
def test_reference_trim_boundary(monkeypatch: pytest.MonkeyPatch, ref_frames: int, expected_prompt: int) -> None:
    decoder, _ = _forward_with(monkeypatch, num_codes=1000, ref_frames=ref_frames)

    assert decoder.s2a_model.call["target_feat_len"].item() == 1720
    assert decoder.s2a_model.call["prompt_feat"].shape[1] == expected_prompt


def test_target_that_cannot_fit_returns_empty_audio_for_that_request(monkeypatch: pytest.MonkeyPatch) -> None:
    decoder, output = _forward_with(monkeypatch, num_codes=10, ref_frames=5, max_positions=10)

    assert decoder.s2a_model.call is None
    assert output.multimodal_outputs["model_outputs"][0].numel() == 0


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
