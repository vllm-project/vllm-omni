# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import functools
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

N_ACOUSTIC_CODEBOOK = 3


@dataclass(frozen=True)
class _ModelArgsStub:
    n_acoustic_codebook: int = N_ACOUSTIC_CODEBOOK
    semantic_codebook_size: int = 2


@dataclass(frozen=True)
class _GeneratorDeviceStub:
    device: torch.device


@functools.lru_cache(maxsize=1)
def _voxtral_classes():
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    import vllm.model_executor.models.utils  # noqa: F401

    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts import (
        VoxtralTTSForConditionalGeneration,
    )
    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_generation import (
        AudioSpecialTokens,
        FlowMatchingAudioTransformer,
        VoxtralTTSAudioGenerationForConditionalGeneration,
    )

    return (
        VoxtralTTSForConditionalGeneration,
        VoxtralTTSAudioGenerationForConditionalGeneration,
        FlowMatchingAudioTransformer,
        AudioSpecialTokens,
    )


class _CaptureAcousticTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_args = _ModelArgsStub()
        self.calls: list[dict[str, torch.Tensor | None]] = []

    def forward(self, llm_hidden, cfg_alpha, noise=None):
        self.calls.append(
            {
                "hidden_states": llm_hidden.clone(),
                "cfg_alpha": cfg_alpha.clone(),
                "noise": None if noise is None else noise.clone(),
            }
        )
        semantic = torch.zeros((llm_hidden.shape[0], 1), dtype=torch.long, device=llm_hidden.device)
        acoustic = torch.zeros(
            (llm_hidden.shape[0], N_ACOUSTIC_CODEBOOK),
            dtype=torch.long,
            device=llm_hidden.device,
        )
        return torch.cat([semantic, acoustic], dim=1)


class _CaptureAudioGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.acoustic_transformer = _CaptureAcousticTransformer()
        self.calls: list[dict[str, torch.Tensor | None]] = []

    def compute_mm_logits(self, hidden_states, cfg_alpha, noise=None):
        self.calls.append(
            {
                "hidden_states": hidden_states.clone(),
                "cfg_alpha": cfg_alpha.clone(),
                "noise": None if noise is None else noise.clone(),
            }
        )
        fake_eos = torch.zeros(hidden_states.shape[0], dtype=hidden_states.dtype)
        return fake_eos, {"codes": {"audio": []}}


class _CaptureAcousticGraph:
    def __init__(self):
        self.calls: list[dict[str, torch.Tensor | None]] = []

    def __call__(self, hidden_states, cfg_alpha, noise=None):
        self.calls.append(
            {
                "hidden_states": hidden_states.clone(),
                "cfg_alpha": cfg_alpha.clone(),
                "noise": None if noise is None else noise.clone(),
            }
        )
        fake_eos = torch.zeros(hidden_states.shape[0], dtype=hidden_states.dtype)
        return fake_eos, {"codes": {"audio": []}}


def _make_voxtral_model():
    model_cls, _, _, _ = _voxtral_classes()
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.model_stage = "audio_generation"
    model.model = _CaptureAudioGeneration()
    model._cudagraph_acoustic_transformer = None
    model._tts_local_generators = {}
    return model


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _request_info(req_id, acoustic_seed=None, *, token_seed=None):
    extra_args = {"tts_local_seed": acoustic_seed} if acoustic_seed is not None else None
    return {"req_id": req_id, "sampling_params": SimpleNamespace(extra_args=extra_args, seed=token_seed)}


def test_request_acoustic_stream_follows_request_across_batch_changes():
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    model = _make_voxtral_model()
    model.postprocess_requires_multimodal_outputs = True
    state = object.__new__(OmniModelState)
    state.model = model
    state.have_multimodal_outputs = True
    state.intermediate_buffer = OmniIntermediateBuffer(2)

    request_a = SimpleNamespace(**_request_info("a", 31), mm_features=[])
    request_b = SimpleNamespace(**_request_info("b", 47), mm_features=[])
    state.intermediate_buffer.add_request(0, request_a)
    state.intermediate_buffer.add_request(1, request_b)
    expected_a, expected_b = _generator(31), _generator(47)

    def run_rows(*indices):
        size = len(indices)
        batch = SimpleNamespace(
            idx_mapping_np=indices,
            num_reqs=size,
            logits_indices=torch.arange(size),
            query_start_loc_np=range(size),
            num_scheduled_tokens=[1] * size,
        )
        state.postprocess_model_output(torch.zeros((size, 4)), batch, SimpleNamespace())
        return model.model.calls[-1]["noise"]

    def check_rows(indices, expected_generators):
        noise = run_rows(*indices)
        assert noise is not None
        for row, generator in enumerate(expected_generators):
            expected = torch.empty(N_ACOUSTIC_CODEBOOK).normal_(generator=generator)
            torch.testing.assert_close(noise[row], expected, atol=0, rtol=0)
        return noise

    first = check_rows((0, 1), (expected_a, expected_b))
    state.intermediate_buffer.remove_request(0)
    check_rows((1,), (expected_b,))

    state.intermediate_buffer.add_request(0, request_a)
    check_rows((1, 0), (expected_b, expected_a))

    model.on_requests_finished({"a", "unknown"})
    state.intermediate_buffer.remove_request(0)
    assert "a" not in model._tts_local_generators
    assert "b" in model._tts_local_generators

    state.intermediate_buffer.add_request(0, request_a)
    restarted = run_rows(0)
    assert restarted is not None
    torch.testing.assert_close(restarted[0], first[0], atol=0, rtol=0)


def test_default_token_seed_does_not_create_an_acoustic_generator():
    model = _make_voxtral_model()
    model.make_omni_output(
        torch.zeros((1, 4)),
        logits_index=torch.arange(1),
        model_intermediate_buffer=[_request_info("unmarked", token_seed=42)],
    )
    assert model.model.calls[-1]["noise"] is None
    assert not model._tts_local_generators


def test_same_seed_produces_identical_multi_frame_noise_streams():
    model = _make_voxtral_model()
    hidden = torch.zeros((1, 4))
    generators_a = {0: _generator(42)}
    generators_b = {0: _generator(42)}

    frames_a = [model._make_acoustic_noise(hidden, generators_a) for _ in range(3)]
    frames_b = [model._make_acoustic_noise(hidden, generators_b) for _ in range(3)]

    for frame_a, frame_b in zip(frames_a, frames_b):
        assert frame_a is not None
        assert frame_b is not None
        torch.testing.assert_close(frame_a, frame_b, atol=0, rtol=0)
    assert not torch.equal(frames_a[0], frames_a[1])


def test_different_seeds_produce_different_noise():
    model = _make_voxtral_model()
    hidden = torch.zeros((1, 4))

    noise_a = model._make_acoustic_noise(hidden, {0: _generator(1)})
    noise_b = model._make_acoustic_noise(hidden, {0: _generator(2)})

    assert noise_a is not None
    assert noise_b is not None
    assert not torch.equal(noise_a, noise_b)


def test_row_reordering_does_not_change_request_noise():
    model = _make_voxtral_model()
    hidden = torch.zeros((2, 4))

    noise_ab = model._make_acoustic_noise(
        hidden,
        {0: _generator(11), 1: _generator(22)},
    )
    noise_ba = model._make_acoustic_noise(
        hidden,
        {0: _generator(22), 1: _generator(11)},
    )

    assert noise_ab is not None
    assert noise_ba is not None
    torch.testing.assert_close(noise_ab[0], noise_ba[1], atol=0, rtol=0)
    torch.testing.assert_close(noise_ab[1], noise_ba[0], atol=0, rtol=0)


def test_seeded_row_is_independent_of_unseeded_batch_rows():
    model = _make_voxtral_model()
    standalone = model._make_acoustic_noise(
        torch.zeros((1, 4)),
        {0: _generator(7)},
    )
    batched = model._make_acoustic_noise(
        torch.zeros((3, 4)),
        {1: _generator(7)},
    )

    assert standalone is not None
    assert batched is not None
    torch.testing.assert_close(standalone[0], batched[1], atol=0, rtol=0)


def test_missing_active_generator_uses_legacy_noise_path():
    model = _make_voxtral_model()

    assert model._make_acoustic_noise(torch.zeros((2, 4)), None) is None
    assert model._make_acoustic_noise(torch.zeros((2, 4)), {}) is None
    assert model._make_acoustic_noise(torch.zeros((2, 4)), {4: _generator(3)}) is None


def test_generator_device_mismatch_fails_with_row_context():
    model = _make_voxtral_model()
    incompatible_generator = _GeneratorDeviceStub(device=torch.device("cuda:1"))

    with pytest.raises(RuntimeError, match=r"row 0.*cuda:1.*cpu"):
        model._make_acoustic_noise(
            torch.zeros((1, 4)),
            {0: incompatible_generator},
        )


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_make_omni_output_maps_generators_to_selected_rows(use_cuda_graph):
    model = _make_voxtral_model()
    graph = _CaptureAcousticGraph() if use_cuda_graph else None
    model._cudagraph_acoustic_transformer = graph
    hidden_states = torch.arange(20, dtype=torch.float32).view(5, 4)
    original_hidden_states = hidden_states.clone()
    logits_index = torch.tensor([1, 4])
    model.make_omni_output(
        hidden_states,
        logits_index=logits_index,
        model_intermediate_buffer=[_request_info("a", 31), _request_info("b", 47)],
        sampling_extra_args=[{}, {}],
    )

    calls = graph.calls if graph is not None else model.model.calls
    assert len(calls) == 1
    torch.testing.assert_close(calls[0]["hidden_states"], original_hidden_states[logits_index])
    expected_row_0 = torch.empty(N_ACOUSTIC_CODEBOOK).normal_(generator=_generator(31))
    expected_row_1 = torch.empty(N_ACOUSTIC_CODEBOOK).normal_(generator=_generator(47))
    torch.testing.assert_close(calls[0]["noise"][0], expected_row_0, atol=0, rtol=0)
    torch.testing.assert_close(calls[0]["noise"][1], expected_row_1, atol=0, rtol=0)


def test_compute_mm_logits_forwards_explicit_noise():
    _, audio_generation_cls, _, _ = _voxtral_classes()
    model = audio_generation_cls.__new__(audio_generation_cls)
    nn.Module.__init__(model)
    model.acoustic_transformer = _CaptureAcousticTransformer()
    model._end_audio_token_id = -1
    hidden = torch.zeros((2, 4))
    cfg_alpha = torch.tensor([1.1, 1.3])
    noise = torch.randn((2, N_ACOUSTIC_CODEBOOK))

    model.compute_mm_logits(hidden, cfg_alpha=cfg_alpha, noise=noise)

    calls = model.acoustic_transformer.calls
    assert len(calls) == 1
    torch.testing.assert_close(calls[0]["noise"], noise, atol=0, rtol=0)


def test_flow_matching_forward_forwards_explicit_noise():
    _, _, flow_cls, audio_special_tokens = _voxtral_classes()
    hidden = torch.zeros((2, 4))
    cfg_alpha = torch.tensor([1.1, 1.3])
    noise = torch.randn((2, N_ACOUSTIC_CODEBOOK))
    captured = {}

    class SemanticProjection:
        def __call__(self, value):
            return torch.zeros((value.shape[0], len(audio_special_tokens) + 2))

    def decode_one_frame(semantic_code, llm_hidden, cfg_alpha, noise=None):
        captured["noise"] = noise
        return torch.zeros((llm_hidden.shape[0], N_ACOUSTIC_CODEBOOK), dtype=torch.long)

    transformer = flow_cls.__new__(flow_cls)
    nn.Module.__init__(transformer)
    transformer.semantic_codebook_output = SemanticProjection()
    transformer._empty_audio_token_id = 1
    transformer.model_args = _ModelArgsStub()
    transformer.decode_one_frame = decode_one_frame

    flow_cls.forward(transformer, hidden, cfg_alpha=cfg_alpha, noise=noise)

    assert captured["noise"] is noise


def _make_minimal_flow_transformer():
    _, _, flow_cls, audio_special_tokens = _voxtral_classes()
    transformer = flow_cls.__new__(flow_cls)
    nn.Module.__init__(transformer)
    transformer.model_args = _ModelArgsStub(n_acoustic_codebook=2)
    transformer._end_audio_token_id = -1
    transformer._empty_audio_token_id = audio_special_tokens.id(audio_special_tokens.empty_audio)
    transformer._noise_scale = 0.5
    transformer._timesteps_cache = {}
    transformer._timesteps = torch.tensor([0.0, 1.0])
    transformer.time_embedding = nn.Identity()
    transformer.time_projection = nn.Identity()
    transformer.llm_projection = nn.Identity()
    transformer.acoustic_embeddings_levels = 5

    def zero_velocity(
        *,
        x_t: torch.Tensor,
        llm_proj: torch.Tensor,
        t_proj: torch.Tensor,
    ) -> torch.Tensor:
        del llm_proj, t_proj
        return torch.zeros_like(x_t)

    transformer._predict_velocity = zero_velocity
    return transformer


def test_decode_one_frame_uses_supplied_noise_without_global_randn(monkeypatch):
    transformer = _make_minimal_flow_transformer()
    _, _, _, audio_special_tokens = _voxtral_classes()
    noise = torch.tensor([[-0.5, 0.5]])

    def fail_randn(*shape: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        del shape, dtype, device
        pytest.fail("global torch.randn was called")

    monkeypatch.setattr(torch, "randn", fail_randn)

    output = transformer.decode_one_frame(
        semantic_code=torch.tensor([0]),
        llm_hidden=torch.zeros((1, 2)),
        cfg_alpha=torch.tensor([1.2]),
        noise=noise,
    )

    sampled = transformer._noise_scale * noise
    expected = (((sampled + 1) / 2) * (transformer.acoustic_embeddings_levels - 1)).round().long()
    expected += len(audio_special_tokens)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_decode_one_frame_without_noise_retains_global_randn_fallback(monkeypatch):
    transformer = _make_minimal_flow_transformer()
    calls: list[tuple[tuple[int, ...], torch.dtype, torch.device]] = []

    def fake_randn(*shape: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        calls.append((shape, dtype, device))
        return torch.zeros(shape, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "randn", fake_randn)

    transformer.decode_one_frame(
        semantic_code=torch.tensor([0]),
        llm_hidden=torch.zeros((1, 2)),
        cfg_alpha=torch.tensor([1.2]),
    )

    assert len(calls) == 1
    assert calls[0][0] == (1, 2)
