# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.models.lingbot_world import test_lingbot_world_attention as attention_tests
from tests.diffusion.models.lingbot_world import test_lingbot_world_transformer as model_tests
from tests.diffusion.models.lingbot_world import test_pipeline_lingbot_world as pipeline_tests

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _cache(embedding, **kwargs):
    path = Path(__file__).parents[4] / "vllm_omni/diffusion/models/lingbot_world/npu_timestep.py"
    spec = importlib.util.spec_from_file_location("_host_timestep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HostTimestepFrequencyCache(embedding, **kwargs)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("frames", [1, 3])
@torch.inference_mode()
def test_host_frequencies_preserve_fp32_timestep_then_fp64_expression(dtype, frames):
    module = attention_tests._load_module()
    cache = _cache(module._sinusoidal_embedding)
    # Fractional labels distinguish FP32 materialization from passing FP64 host
    # values straight into the sinusoidal expression; include clean-x0 t=0.
    for value in [0.0, 1000.0, 977.124938147, 482.00319291, -0.0]:
        t = torch.full((frames,), value, dtype=torch.float32)
        expected = module._sinusoidal_embedding(256, t).to(dtype)
        actual = cache.get(value, dim=256, batch_size=1, frames=frames, device="cpu", dtype=dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_host_cache_reuses_equal_fp32_values_and_partitions_dtype_shape():
    module = attention_tests._load_module()
    calls = []

    def embedding(dim, t):
        assert t.device.type == "cpu" and t.dtype == torch.float32
        calls.append(t.clone())
        return module._sinusoidal_embedding(dim, t)

    cache = _cache(embedding, max_entries=2)
    kw = dict(dim=256, batch_size=1, frames=3, device="cpu", dtype=torch.bfloat16)
    a = cache.get(1.0, **kw)
    assert cache.get(1.0 + 1e-10, **kw) is a
    b = cache.get(1.0, **{**kw, "dtype": torch.float32})
    assert b.dtype == torch.float32 and len(calls) == 2
    cache.get(1.0, **{**kw, "frames": 1})
    assert len(cache) == 2
    cache.get(1.0, **kw)
    assert len(calls) == 4  # LRU eviction, no stale shape/dtype reuse.


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@torch.inference_mode()
def test_host_cache_rejects_nonfinite_timestep(value):
    cache = _cache(lambda dim, t: t)
    with pytest.raises(ValueError, match="finite"):
        cache.get(value, dim=256, batch_size=1, frames=3, device="cpu", dtype=torch.bfloat16)


@torch.inference_mode()
def test_host_cache_rejects_tensor_metadata_instead_of_reading_it_back():
    cache = _cache(lambda dim, t: t)
    with pytest.raises(TypeError, match="host scalar"):
        cache.get(torch.tensor(1.0), dim=256, batch_size=1, frames=3, device="cpu", dtype=torch.bfloat16)


def test_host_cache_rejects_grad_enabled_execution():
    cache = _cache(lambda dim, t: t)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference"):
        cache.get(1.0, dim=256, batch_size=1, frames=3, device="cpu", dtype=torch.bfloat16)


@torch.inference_mode()
def test_prepared_frequencies_preserve_both_learned_timestep_outputs():
    module = attention_tests._load_module()
    model = model_tests._tiny_model(module, num_layers=1, num_frames_per_block=3, sliding_window_num_frames=6)
    cache = _cache(module._sinusoidal_embedding)
    value = 977.124938147
    t = torch.full((1,), value, dtype=torch.float32)
    kw = dict(batch_size=1, frames=3, dtype=torch.float32)
    expected = model._timestep_embeddings(t, **kw)
    frequency = cache.get(value, dim=model.config.freq_dim, device="cpu", **kw)
    actual = model._timestep_embeddings(t, frequency_embedding=frequency, **kw)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Prepared timestep"):
        model._timestep_embeddings(t, frequency_embedding=frequency[:1], **kw)


@torch.inference_mode()
def test_dmd_routes_host_metadata_for_both_noisy_probe_and_clean_commit(monkeypatch):
    module = pipeline_tests._load_pipeline_module()
    reference = attention_tests._load_module()
    calls = []

    class Consumer:
        config = SimpleNamespace(freq_dim=256, patch_size=(1, 2, 2))

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return torch.zeros(1, 2, 3, 4, 4)

    runner = module.LingBotDMDBlockRunner(Consumer(), device=torch.device("cpu"), enforce_eager=True)
    # Exercise metadata routing on CPU without claiming a CPU deployment option.
    runner._host_timestep_cache = _cache(reference._sinusoidal_embedding)
    monkeypatch.setattr(runner, "_block_cache", lambda **kwargs: None)
    condition = torch.zeros(1, 34, 3, 4, 4)
    common = dict(
        condition=condition, camera=torch.empty(0), prompt_embeds=torch.empty(0), cache=None, ar=None, start_frame=0
    )
    runner.probe_step(current_latents=torch.zeros(1, 2, 3, 4, 4), timestep_value=937.5, step_index=1, **common)
    runner.commit_block_kv(latents=torch.zeros(1, 2, 3, 4, 4), **common)
    assert len(calls) == 2
    for call, value in zip(calls, [937.5, 0.0], strict=True):
        expected = reference._sinusoidal_embedding(256, torch.full((3,), value, dtype=torch.float32)).float()
        torch.testing.assert_close(call["timestep_frequency_embedding"], expected, atol=0, rtol=0)
