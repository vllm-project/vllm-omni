# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU acceptance for the diffusion-loop CUDA-graph executor.

Replay must be bitwise identical to the eager loop for the same inputs,
across consecutive tokens (proving no cross-replay state leakage) and across
active-batch sizes. Capture failure must fall back to eager permanently.
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm_omni.model_executor.models.vibevoice.diffusion import (
    VibeVoiceDiffusionGraphExecutor,
    VibeVoiceDiffusionHead,
    VibeVoiceDiffusionSampler,
    _DiffusionGraphCaptureError,
)
from vllm_omni.model_executor.models.vibevoice.vibevoice import VibeVoiceModel
from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

_MODEL = os.getenv("VIBEVOICE_TEST_MODEL", "microsoft/VibeVoice-1.5B")


def _build() -> tuple[VibeVoiceDiffusionHead, VibeVoiceDiffusionSampler]:
    config = VibeVoiceConfig.from_pretrained(_MODEL)
    torch.manual_seed(0)
    head = VibeVoiceDiffusionHead(config).to(device="cuda", dtype=torch.bfloat16).eval()
    sampler = VibeVoiceDiffusionSampler.from_model_config(config)
    return head, sampler


def _inputs(batch: int, hidden: int, latent: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    positive = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16, generator=generator)
    negative = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16, generator=generator)
    noise = torch.randn(2 * batch, latent, device="cuda", dtype=torch.bfloat16, generator=generator)
    return positive, negative, noise


def _build_graph_model() -> VibeVoiceModel:
    head, sampler = _build()
    model = VibeVoiceModel.__new__(VibeVoiceModel)
    torch.nn.Module.__init__(model)
    model.diffusion_head = head
    model.diffusion_sampler = sampler
    model.diffusion_graph_enabled = True
    model.cuda_graph_capture_failure_fatal = False
    model._diffusion_graph_executor = None
    model._shared_graph_pool = None
    return model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_diffusion_graph_startup_warmup_captures_all_batches_without_consuming_rng() -> None:
    model = _build_graph_model()

    model.diffusion_graph_enabled = False
    model.warmup_diffusion_graphs(
        (1, 2, 3, 4),
        num_inference_steps=10,
        guidance_scale=1.3,
    )
    assert model._diffusion_graph_executor is None
    model.diffusion_graph_enabled = True
    model.warmup_diffusion_graphs(
        (),
        num_inference_steps=10,
        guidance_scale=1.3,
    )
    assert model._diffusion_graph_executor is None

    torch.manual_seed(1234)
    expected_cpu_random = torch.randn(8)
    expected_cuda_random = torch.randn(8, device="cuda")
    torch.manual_seed(1234)

    model.warmup_diffusion_graphs(
        (4, 1, 3, 2, 3),
        num_inference_steps=10,
        guidance_scale=1.3,
    )

    executor = model._diffusion_graph_executor
    assert executor is not None
    assert executor.disabled is False
    assert set(executor._entries) == {
        (1, 10, 1.3),
        (2, 10, 1.3),
        (3, 10, 1.3),
        (4, 10, 1.3),
    }
    assert torch.equal(torch.randn(8), expected_cpu_random)
    assert torch.equal(torch.randn(8, device="cuda"), expected_cuda_random)

    entry_ids = {key: id(entry) for key, entry in executor._entries.items()}
    model.warmup_diffusion_graphs(
        (1, 2, 3, 4),
        num_inference_steps=10,
        guidance_scale=1.3,
    )
    assert {key: id(entry) for key, entry in executor._entries.items()} == entry_ids

    positive, negative, noise = _inputs(
        3,
        model.diffusion_sampler.condition_size,
        model.diffusion_sampler.latent_size,
        seed=901,
    )
    model.sample_audio_latent(
        positive,
        negative,
        noise,
        guidance_scale=1.0,
        num_inference_steps=5,
    )
    assert (3, 5, 1.0) not in executor._entries


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_diffusion_graph_startup_warmup_stops_after_capture_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_graph_model()
    executor = VibeVoiceDiffusionGraphExecutor(
        model.diffusion_sampler,
        model.diffusion_head,
    )
    model._diffusion_graph_executor = executor
    batch_sizes: list[int] = []
    sample_audio_latent = model.sample_audio_latent

    def record_sample(*args, **kwargs):
        batch_sizes.append(int(args[0].shape[0]))
        return sample_audio_latent(*args, **kwargs)

    def fail_capture(*_args, **_kwargs):
        raise _DiffusionGraphCaptureError("capture failed")

    monkeypatch.setattr(model, "sample_audio_latent", record_sample)
    monkeypatch.setattr(executor, "_capture", fail_capture)

    model.warmup_diffusion_graphs(
        (1, 2, 3, 4),
        num_inference_steps=10,
        guidance_scale=1.3,
    )

    assert executor.disabled is True
    assert batch_sizes == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_diffusion_graph_replay_is_bitwise_identical_across_tokens(batch_size: int) -> None:
    head, sampler = _build()
    executor = VibeVoiceDiffusionGraphExecutor(sampler, head)
    hidden = sampler.condition_size
    latent = sampler.latent_size

    with torch.inference_mode():
        for token_index in range(3):
            positive, negative, noise = _inputs(batch_size, hidden, latent, seed=100 + token_index)
            expected = sampler.sample_audio_latent(
                head,
                positive,
                negative,
                noise.clone(),
                guidance_scale=1.3,
                num_inference_steps=10,
            )
            actual = executor.sample(
                positive,
                negative,
                noise.clone(),
                guidance_scale=1.3,
                num_inference_steps=10,
            )
            assert actual is not None
            assert torch.equal(actual, expected), (
                f"token {token_index}: graph replay diverged from eager "
                f"(max diff {(actual.float() - expected.float()).abs().max().item()})"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_diffusion_graph_cache_is_bounded_to_official_control_keys() -> None:
    head, sampler = _build()
    executor = VibeVoiceDiffusionGraphExecutor(sampler, head)
    hidden = sampler.condition_size
    latent = sampler.latent_size

    with torch.inference_mode():
        for batch_size in (1, 2, 3, 4):
            positive, negative, noise = _inputs(batch_size, hidden, latent, seed=7 + batch_size)
            expected = sampler.sample_audio_latent(
                head,
                positive,
                negative,
                noise.clone(),
                guidance_scale=1.3,
                num_inference_steps=10,
            )
            actual = executor.sample(
                positive,
                negative,
                noise.clone(),
                guidance_scale=1.3,
                num_inference_steps=10,
            )
            assert actual is not None
            assert torch.equal(actual, expected)

        assert executor.num_captured_graphs == 4

        positive, negative, noise = _inputs(2, hidden, latent, seed=17)
        for guidance, steps in ((1.0, 10), (1.3, 5), (2.5, 7), (1.3, 50)):
            assert (
                executor.sample(
                    positive,
                    negative,
                    noise.clone(),
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                )
                is None
            )
            assert executor.num_captured_graphs == 4

        positive5, negative5, noise5 = _inputs(5, hidden, latent, seed=18)
        assert (
            executor.sample(
                positive5,
                negative5,
                noise5,
                guidance_scale=1.3,
                num_inference_steps=10,
            )
            is None
        )
        assert executor.num_captured_graphs == 4

        expected_replay = sampler.sample_audio_latent(
            head,
            positive,
            negative,
            noise.clone(),
            guidance_scale=1.3,
            num_inference_steps=10,
        )
        replayed = executor.sample(
            positive,
            negative,
            noise.clone(),
            guidance_scale=1.3,
            num_inference_steps=10,
        )
        assert replayed is not None
        assert torch.equal(replayed, expected_replay)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_diffusion_graph_capture_failure_falls_back_to_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head, sampler = _build()
    executor = VibeVoiceDiffusionGraphExecutor(sampler, head)
    hidden = sampler.condition_size
    latent = sampler.latent_size
    positive, negative, noise = _inputs(1, hidden, latent, seed=11)

    def fail_capture(*_args, **_kwargs):
        raise _DiffusionGraphCaptureError("capture failed")

    monkeypatch.setattr(executor, "_capture", fail_capture)
    with torch.inference_mode():
        assert executor.sample(positive, negative, noise, guidance_scale=1.3, num_inference_steps=10) is None
        assert executor.sample(positive, negative, noise, guidance_scale=1.3, num_inference_steps=10) is None
        assert executor.disabled is True

        strict_executor = VibeVoiceDiffusionGraphExecutor(
            sampler,
            head,
            capture_failure_fatal=True,
        )
        monkeypatch.setattr(strict_executor, "_capture", fail_capture)
        with pytest.raises(RuntimeError, match="Required VibeVoice diffusion CUDA-graph capture failed"):
            strict_executor.sample(
                positive,
                negative,
                noise,
                guidance_scale=1.3,
                num_inference_steps=10,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_diffusion_graph_oom_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head, sampler = _build()
    executor = VibeVoiceDiffusionGraphExecutor(sampler, head)
    positive, negative, noise = _inputs(1, sampler.condition_size, sampler.latent_size, seed=12)

    def fail_capture(*_args, **_kwargs):
        raise torch.OutOfMemoryError("injected OOM")

    monkeypatch.setattr(executor, "_capture", fail_capture)
    with pytest.raises(torch.OutOfMemoryError, match="injected OOM"):
        executor.sample(
            positive,
            negative,
            noise,
            guidance_scale=1.3,
            num_inference_steps=10,
        )
    assert executor.disabled is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_diffusion_graph_non_capture_errors_fail_fast() -> None:
    _, sampler = _build()

    class _BrokenHead:
        def parameters(self):
            return iter([])

    executor = VibeVoiceDiffusionGraphExecutor(sampler, _BrokenHead())
    positive, negative, noise = _inputs(1, sampler.condition_size, sampler.latent_size, seed=13)

    with pytest.raises(RuntimeError, match="requires a CUDA diffusion head"):
        executor.sample(
            positive,
            negative,
            noise,
            guidance_scale=1.3,
            num_inference_steps=10,
        )
    assert executor.disabled is False
