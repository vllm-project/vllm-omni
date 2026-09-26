# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import cache_dit
import pytest
import torch
from cache_dit import BlockAdapter
from torch import nn

from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend, RequestScopedCacheDiTRuntime
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import (
    AttentionConfig,
    DiffusionCacheConfig,
    DiffusionOutput,
    OmniDiffusionConfig,
    TransformerConfig,
)
from vllm_omni.diffusion.models.interface import adopt_request_scoped_cache_dit
from vllm_omni.diffusion.models.mammoth_moda2 import pipeline_mammothmoda2_dit
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
    _build_mammoth_config,
    _MammothRequest,
    _pad_cond_sequence,
    _root_weight_source,
    get_mammoth_moda2_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

from .helpers import initialize_block_weights

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _raw_config() -> dict:
    return {
        "model_type": "mammothmoda2",
        "llm_config": {
            "model_type": "mammothmoda2_qwen2_5_vl",
            "text_config": {
                "model_type": "mammothmoda2_qwen2_5_vl_text",
                "hidden_size": 8,
                "gen_vocab_start_index": 100,
            },
        },
        "gen_vae_config": {"block_out_channels": [8, 8]},
        "gen_dit_config": {"hidden_size": 8, "in_channels": 4},
    }


def _od_config() -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model="/models/MammothModa2-Preview",
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(_raw_config()),
    )


def test_build_mammoth_config_uses_shared_transformer_projection() -> None:
    config = _build_mammoth_config(_od_config())
    assert config.gen_dit_config["hidden_size"] == 8


def test_build_mammoth_config_rejects_empty_shared_projection() -> None:
    config = _od_config()
    config.tf_model_config = TransformerConfig()
    with pytest.raises(ValueError, match="root checkpoint config"):
        _build_mammoth_config(config)


def test_root_weight_source_loads_combined_checkpoint_once() -> None:
    source = _root_weight_source(_od_config())
    assert source.model_or_path == "/models/MammothModa2-Preview"
    assert source.subfolder is None
    assert source.prefix == ""
    assert source.fall_back_to_pt is True


def test_root_weight_source_forwards_revision() -> None:
    config = _od_config()
    config.revision = "rev-7"
    assert _root_weight_source(config).revision == "rev-7"


def test_pipeline_declares_native_components_and_batch_request_mode() -> None:
    assert MammothModa2DiTPipeline._dit_modules == ["gen_transformer"]
    assert MammothModa2DiTPipeline._encoder_modules == ["gen_image_condition_refiner"]
    assert MammothModa2DiTPipeline._vae_modules == ["gen_vae"]
    assert MammothModa2DiTPipeline.supports_request_batch is True
    assert MammothModa2DiTPipeline.supports_step_execution is False


def test_mammoth_postprocess_denormalizes_nonnegative_raw_vae_output() -> None:
    factory = getattr(pipeline_mammothmoda2_dit, "get_mammoth_moda2_post_process_func", None)
    assert factory is not None

    images = factory(_od_config())(torch.zeros(1, 3, 2, 2))

    assert len(images) == 1
    assert images[0].getpixel((0, 0)) == (128, 128, 128)


def test_mammoth_postprocess_is_registered() -> None:
    from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

    assert _DIFFUSION_POST_PROCESS_FUNCS["MammothModa2DiTPipeline"] == "get_mammoth_moda2_post_process_func"


def test_root_weight_source_rejects_missing_model_path() -> None:
    config = _od_config()
    config.model = None
    with pytest.raises(ValueError, match="model path"):
        _root_weight_source(config)


def _pipeline_shell() -> MammothModa2DiTPipeline:
    pipeline = object.__new__(MammothModa2DiTPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.config = _build_mammoth_config(_od_config())
    pipeline._llm_hidden_size = 8
    # Cache-DiT runtime state that __init__ normally creates (disabled here,
    # matching a pipeline built without a cache_backend).
    pipeline._cache_dit_runtime = RequestScopedCacheDiTRuntime(pipeline)
    pipeline._cache_dit_config = None
    return pipeline


def _batch(
    *,
    request_id: str = "req-7",
    prompt: object | None = None,
    sampling: OmniDiffusionSamplingParams | None = None,
    height: int = 32,
    width: int = 48,
) -> DiffusionRequestBatch:
    if prompt is None:
        prompt = {
            "prompt": "",
            "height": height,
            "width": width,
            "additional_information": {
                "full_hidden_states": torch.arange(32, dtype=torch.float32).reshape(4, 8),
                "full_token_ids": [10, 11, 100, 101],
                "answer_start_index": 2,
            },
        }
    if sampling is None:
        sampling = OmniDiffusionSamplingParams(
            height=32,
            width=48,
            seed=42,
            guidance_scale=4.0,
            num_inference_steps=7,
            extra_args={"cfg_range": [0.2, 0.8]},
        )
    return DiffusionRequestBatch([OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id=request_id)])


def test_parse_request_uses_standard_sampling_fields() -> None:
    parsed = _pipeline_shell()._parse_request(_batch())
    assert parsed.request_id == "req-7"
    assert (parsed.height, parsed.width) == (32, 48)
    assert parsed.num_inference_steps == 7
    assert parsed.text_guidance_scale == 4.0
    assert parsed.cfg_range == (0.2, 0.8)
    assert parsed.seed == 42
    assert parsed.answer_start_index == 2


def test_parse_request_prefers_legacy_sampling_overrides() -> None:
    sampling = OmniDiffusionSamplingParams(
        guidance_scale=3.0,
        num_inference_steps=5,
        seed=9,
        extra_args={"text_guidance_scale": 6.0, "num_inference_steps": 11, "cfg_range": [0.0, 0.5]},
    )
    parsed = _pipeline_shell()._parse_request(_batch(sampling=sampling))
    assert parsed.text_guidance_scale == 6.0
    assert parsed.num_inference_steps == 11
    assert parsed.cfg_range == (0.0, 0.5)


def test_parse_request_falls_back_to_request_level_sampling_values() -> None:
    prompt = _batch().prompts[0]
    prompt["additional_information"].update(
        text_guidance_scale=[1.5],
        num_inference_steps=[3],
        cfg_range=[0.25, 0.75],
    )

    parsed = _pipeline_shell()._parse_request(_batch(prompt=prompt, sampling=OmniDiffusionSamplingParams()))

    assert parsed.text_guidance_scale == 1.5
    assert parsed.num_inference_steps == 3
    assert parsed.cfg_range == (0.25, 0.75)


def test_parse_request_standard_fields_precede_request_level_fallbacks() -> None:
    prompt = _batch().prompts[0]
    prompt["additional_information"].update(
        text_guidance_scale=[1.5],
        num_inference_steps=[3],
    )
    sampling = OmniDiffusionSamplingParams(guidance_scale=4.0, num_inference_steps=7)

    parsed = _pipeline_shell()._parse_request(_batch(prompt=prompt, sampling=sampling))

    assert parsed.text_guidance_scale == 4.0
    assert parsed.num_inference_steps == 7


def test_parse_request_parses_each_request_and_rejects_multi_output() -> None:
    pipeline = _pipeline_shell()
    batch = _batch()
    batch.requests.append(_batch(request_id="req-8").requests[0])
    req0 = pipeline._parse_request(batch, index=0)
    req1 = pipeline._parse_request(batch, index=1)
    assert req0.request_id == "req-7"
    assert req1.request_id == "req-8"
    assert req0.index == 0
    assert req1.index == 1
    with pytest.raises(ValueError, match="num_outputs_per_prompt == 1"):
        pipeline._parse_request(_batch(sampling=OmniDiffusionSamplingParams(num_outputs_per_prompt=2)))


@pytest.mark.parametrize(
    ("cfg_range", "message"),
    [
        ([0.5], "two values"),
        ([-0.1, 0.5], "0 <= start <= end <= 1"),
        ([0.7, 0.2], "0 <= start <= end <= 1"),
        ([0.2, 1.1], "0 <= start <= end <= 1"),
    ],
)
def test_parse_request_rejects_invalid_cfg_range(cfg_range, message) -> None:
    batch = _batch(sampling=OmniDiffusionSamplingParams(extra_args={"cfg_range": cfg_range}))
    with pytest.raises(ValueError, match=message):
        _pipeline_shell()._parse_request(batch)


def test_parse_request_requires_real_ar_conditions() -> None:
    with pytest.raises(ValueError, match="req-missing"):
        _pipeline_shell()._parse_request(_batch(request_id="req-missing", prompt={"prompt": "draw a cat"}))


def test_parse_request_rejects_hidden_state_token_count_mismatch() -> None:
    batch = _batch()
    batch.prompts[0]["additional_information"]["full_hidden_states"] = torch.zeros(3, 8)
    with pytest.raises(ValueError, match="hidden-state/token-count mismatch"):
        _pipeline_shell()._parse_request(batch)


@pytest.mark.parametrize(
    ("height", "width", "message"),
    [
        (0, 32, "Invalid image size"),
        (32, -1, "Invalid image size"),
        (30, 32, "multiples of 16"),
        (32, 31, "multiples of 16"),
    ],
)
def test_parse_request_rejects_invalid_dimensions(height, width, message) -> None:
    batch = _batch()
    batch.prompts[0].update(height=height, width=width)
    with pytest.raises(ValueError, match=message):
        _pipeline_shell()._parse_request(batch)


def test_parse_request_rejects_explicit_zero_steps() -> None:
    with pytest.raises(ValueError, match="num_inference_steps must be positive"):
        _pipeline_shell()._parse_request(_batch(sampling=OmniDiffusionSamplingParams(num_inference_steps=0)))


@pytest.mark.parametrize(("standard_steps", "expected_steps"), [(7, 7), (None, 50)])
def test_parse_request_falls_through_null_legacy_step_count(standard_steps, expected_steps) -> None:
    sampling = OmniDiffusionSamplingParams(num_inference_steps=standard_steps, extra_args={"num_inference_steps": None})
    parsed = _pipeline_shell()._parse_request(_batch(sampling=sampling))
    assert parsed.num_inference_steps == expected_steps


@pytest.mark.parametrize(("standard_guidance", "expected_guidance"), [(4.0, 4.0), (None, 9.0)])
def test_parse_request_falls_through_null_legacy_guidance(standard_guidance, expected_guidance) -> None:
    sampling = OmniDiffusionSamplingParams(guidance_scale=standard_guidance, extra_args={"text_guidance_scale": None})
    parsed = _pipeline_shell()._parse_request(_batch(sampling=sampling))
    assert parsed.text_guidance_scale == expected_guidance


def test_parse_request_defaults_null_cfg_range() -> None:
    sampling = OmniDiffusionSamplingParams(extra_args={"cfg_range": None})
    parsed = _pipeline_shell()._parse_request(_batch(sampling=sampling))
    assert parsed.cfg_range == (0.0, 1.0)


@pytest.mark.parametrize("generators", [[], [torch.Generator(), torch.Generator()]])
def test_parse_request_rejects_invalid_generator_list_cardinality(generators) -> None:
    sampling = OmniDiffusionSamplingParams(generator=generators)
    with pytest.raises(ValueError, match="exactly one generator.*req-generators"):
        _pipeline_shell()._parse_request(_batch(request_id="req-generators", sampling=sampling))


def test_parse_request_synthesizes_dummy_ar_conditions() -> None:
    batch = _batch(
        request_id="dummy_req_id",
        prompt={"prompt": "dummy run"},
        sampling=OmniDiffusionSamplingParams(height=512, width=512, seed=1, guidance_scale=0.0, num_inference_steps=2),
    )
    parsed = _pipeline_shell()._parse_request(batch)
    assert parsed.full_hidden_states.shape == (2, 8)
    assert parsed.full_token_ids == [0, 100]
    assert parsed.answer_start_index == 1


@dataclass
class _FakeTransformerConfig:
    in_channels: int = 4


@dataclass
class _FakeTimeCaptionEmbed:
    image_embedder: nn.Module | None = None


class _FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = _FakeTransformerConfig()
        self.time_caption_embed = _FakeTimeCaptionEmbed()
        self.calls = 0

    def forward(self, *, hidden_states, **kwargs):
        self.calls += 1
        return torch.zeros_like(hidden_states)


@dataclass
class _FakeVaeConfig:
    scaling_factor: float | None = None
    shift_factor: float | None = None


class _FakeVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = _FakeVaeConfig()
        self.last_decode_dtype: torch.dtype | None = None

    def decode(self, latents, return_dict=False):
        assert return_dict is False
        self.last_decode_dtype = latents.dtype
        b = latents.shape[0]
        base = latents[:, :1, :1, :1]
        return (torch.zeros(b, 3, 32, 48, dtype=latents.dtype) + base,)


class _FakeScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([])
        self.requested_steps = None

    def set_timesteps(self, *, num_inference_steps, device, num_tokens):
        self.requested_steps = num_inference_steps
        self.timesteps = torch.arange(num_inference_steps, device=device, dtype=torch.float32)

    def step(self, model_pred, timestep, latents, return_dict=False):
        assert return_dict is False
        return (latents - model_pred,)


def test_forward_returns_diffusion_output_with_request_sampling(mocker) -> None:
    pipeline = _pipeline_shell()
    pipeline.gen_transformer = _FakeTransformer()
    pipeline.gen_image_condition_refiner = None
    pipeline.gen_vae = _FakeVae()
    pipeline.gen_freqs_cis = torch.zeros(1)
    scheduler = _FakeScheduler()
    captured = {}

    def fake_randn_tensor(shape, *, generator, device, dtype):
        captured["shape"] = shape
        captured["seed"] = generator.initial_seed()
        return torch.zeros(shape, device=device, dtype=dtype)

    module = "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit"
    mocker.patch(f"{module}.FlowMatchEulerDiscreteScheduler", return_value=scheduler)
    mocker.patch(f"{module}.randn_tensor", side_effect=fake_randn_tensor)
    result = pipeline.forward(
        _batch(
            sampling=OmniDiffusionSamplingParams(
                height=32, width=48, seed=42, guidance_scale=1.0, num_inference_steps=2
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], DiffusionOutput)
    assert result[0].output.shape == (1, 3, 32, 48)
    assert captured["shape"] == (1, 4, 4, 6)
    assert captured["seed"] == 42
    assert scheduler.requested_steps == 2
    assert pipeline.gen_transformer.calls == 2


def test_forward_rejects_missing_visual_tokens_before_model_access() -> None:
    prompt = {
        "prompt": "",
        "additional_information": {
            "full_hidden_states": torch.zeros(3, 8),
            "full_token_ids": [10, 11, 12],
            "answer_start_index": 2,
        },
    }
    with pytest.raises(ValueError, match="no visual-token hidden states.*req-empty"):
        _pipeline_shell().forward(_batch(request_id="req-empty", prompt=prompt))


@contextmanager
def _force_torch_sdpa():
    """Pin TORCH_SDPA so CPU shape tests do not pick CUDA-only backends (FA3)."""
    od_config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        parallel_config=SimpleNamespace(ring_degree=1),
    )
    with set_current_diffusion_config(od_config):
        yield


def _constructible_raw_config() -> dict:
    """Checkpoint config whose DiT/VAE are small enough to actually run on CPU."""
    return {
        "model_type": "mammothmoda2",
        "llm_config": {
            "model_type": "mammothmoda2_qwen2_5_vl",
            "text_config": {
                "model_type": "mammothmoda2_qwen2_5_vl_text",
                "hidden_size": 8,
                "gen_vocab_start_index": 100,
            },
        },
        "gen_vae_config": {"block_out_channels": [8, 8], "norm_num_groups": 8},
        "gen_dit_config": {
            "patch_size": 2,
            "in_channels": 4,
            "hidden_size": 96,
            "num_layers": 4,
            "num_refiner_layers": 1,
            "num_attention_heads": 2,
            "num_kv_heads": 2,
            "multiple_of": 8,
            "axes_dim_rope": [16, 16, 16],
            "axes_lens": [300, 512, 512],
            "text_feat_dim": 16,
        },
        # Top-level axes feed the pipeline-owned rotary table (see pipeline __init__).
        "gen_axes_dim_rope": [16, 16, 16],
        "gen_axes_lens": [300, 512, 512],
    }


def _cache_dit_od_config() -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model="/models/MammothModa2-Preview",
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(_constructible_raw_config()),
        cache_backend="cache_dit",
        cache_config=DiffusionCacheConfig(),
    )


@pytest.mark.skipif(
    current_omni_platform.is_rocm(),
    reason="vLLM ROCm custom ops lack CPU fallback",
)
def test_forward_transitions_request_scoped_cache_dit_across_requests(mock_tp1, request: pytest.FixtureRequest) -> None:
    """Cache-DiT request lifecycle on one production-built pipeline instance.

    Mirrors the diffusion runner startup (``get_cache_backend`` → ``enable`` →
    ``adopt_request_scoped_cache_dit``) on a tiny pipeline built through the
    real ``__init__``, then drives three requests through the real ``forward()``:

    1. a partial-CFG request installs the request-scoped cache context:
       cond/uncond forwards stay paired on every denoise step (parity
       accounting), even outside ``cfg_range``, and blocks record cached steps;
    2. a guidance=1.0 request runs with hooks disabled (``prepare(None)``) and
       its output is bit-identical to a reference captured before the backend
       was ever enabled, proving a clean uninstall with no stale module state;
    3. a CFG request with a different step count and image size re-installs the
       hooks through a fresh context, so no stale shape or step count survives.
    """
    torch.manual_seed(0)  # Deterministic random init → deterministic cache hits.
    with _force_torch_sdpa():
        pipeline = MammothModa2DiTPipeline(od_config=_cache_dit_od_config())
    initialize_block_weights(pipeline.gen_transformer)
    pipeline.eval()
    assert pipeline._cache_dit_config is not None
    request.addfinalizer(pipeline._cache_dit_runtime.disable)

    # The default prompt carries 2 text + 2 visual condition tokens, so the
    # conditional pass sees 4 tokens and the uncond pass (empty prompt) sees 0.
    cond_tokens, uncond_tokens = 4, 0
    seq_lens: list[int] = []

    def record_call(_module, _args, kwargs):
        text = kwargs.get("text_hidden_states")
        seq_lens.append(int(text.shape[1]) if text is not None else -1)

    hook = pipeline.gen_transformer.register_forward_pre_hook(record_call, with_kwargs=True)

    unguided_sampling = OmniDiffusionSamplingParams(
        height=32, width=48, seed=4321, guidance_scale=1.0, num_inference_steps=8
    )
    partial_cfg_sampling = OmniDiffusionSamplingParams(
        height=32,
        width=48,
        seed=1234,
        guidance_scale=4.0,
        num_inference_steps=8,
        extra_args={"cfg_range": [0.2, 0.8]},
    )
    changed_cfg_sampling = OmniDiffusionSamplingParams(
        height=48,
        width=64,
        seed=777,
        guidance_scale=6.0,
        num_inference_steps=10,
        extra_args={"cfg_range": [0.1, 0.9]},
    )

    # Reference run for phase 2: identical request while hooks were never
    # installed on this pipeline.
    assert not pipeline.is_cache_dit_enabled()
    reference = pipeline.forward(_batch(sampling=unguided_sampling))[0].output
    assert seq_lens == [cond_tokens] * 8
    assert torch.isfinite(reference.float()).all()

    # Runner startup: enable once, then the pipeline owns all transitions.
    backend = get_cache_backend("cache_dit", DiffusionCacheConfig())
    assert isinstance(backend, CacheDiTBackend)
    backend.enable(pipeline)
    assert adopt_request_scoped_cache_dit(pipeline, backend)
    assert pipeline.is_cache_dit_enabled()

    # Phase 1: partial CFG keeps cond/uncond paired on every step, including
    # the steps whose normalized index falls outside cfg_range [0.2, 0.8].
    seq_lens.clear()
    pipeline.forward(_batch(sampling=partial_cfg_sampling))
    assert seq_lens == [cond_tokens, uncond_tokens] * 8
    assert pipeline.is_cache_dit_enabled()
    assert BlockAdapter.is_cached(pipeline.gen_transformer)
    hits = cache_dit.summary(pipeline.gen_transformer, logging=False)[0]
    assert hits.cached_steps or hits.cfg_cached_steps

    # Phase 2: no-CFG request disables the hooks and must reproduce the
    # never-enabled reference bit-for-bit.
    seq_lens.clear()
    unguided = pipeline.forward(_batch(sampling=unguided_sampling))[0].output
    assert seq_lens == [cond_tokens] * 8
    assert not pipeline.is_cache_dit_enabled()
    assert not BlockAdapter.is_cached(pipeline.gen_transformer)
    assert torch.equal(unguided, reference)

    # Phase 3: CFG with a new step count and image size re-installs the hooks
    # on a fresh context; output shape follows the new request geometry.
    seq_lens.clear()
    regenerated = pipeline.forward(_batch(sampling=changed_cfg_sampling, height=48, width=64))[0].output
    assert seq_lens == [cond_tokens, uncond_tokens] * 10
    assert pipeline.is_cache_dit_enabled()
    assert BlockAdapter.is_cached(pipeline.gen_transformer)
    hits = cache_dit.summary(pipeline.gen_transformer, logging=False)[0]
    assert hits.cached_steps or hits.cfg_cached_steps
    assert regenerated.shape == (1, 3, 12, 16)
    assert torch.isfinite(regenerated.float()).all()

    hook.remove()


def test_pre_process_registers_batch_compatibility_key() -> None:
    pre_process = get_mammoth_moda2_pre_process_func(_od_config())
    req = _batch().requests[0]
    pre_process(req)
    assert req.batch_compatibility_key == ("mammoth_moda2_dit", 32, 48, 7)


def test_pre_process_rejects_missing_ar_conditions() -> None:
    pre_process = get_mammoth_moda2_pre_process_func(_od_config())
    req = _batch(request_id="req-bad-ar", prompt={"prompt": "test"}).requests[0]
    with pytest.raises(ValueError, match="Missing additional_information AR conditions.*req-bad-ar"):
        pre_process(req)


@pytest.mark.parametrize(
    ("height", "width", "message"),
    [
        (0, 32, "Invalid image size.*req-dim"),
        (32, -1, "Invalid image size.*req-dim"),
        (30, 32, "multiples of 16.*req-dim"),
        (32, 31, "multiples of 16.*req-dim"),
    ],
)
def test_pre_process_rejects_invalid_dimensions(height, width, message) -> None:
    pre_process = get_mammoth_moda2_pre_process_func(_od_config())
    batch = _batch(request_id="req-dim")
    batch.prompts[0].update(height=height, width=width)
    with pytest.raises(ValueError, match=message):
        pre_process(batch.requests[0])


def test_pre_process_rejects_non_positive_steps() -> None:
    pre_process = get_mammoth_moda2_pre_process_func(_od_config())
    req = _batch(
        request_id="req-zero-steps",
        sampling=OmniDiffusionSamplingParams(num_inference_steps=0),
    ).requests[0]
    with pytest.raises(ValueError, match="num_inference_steps must be positive.*req-zero-steps"):
        pre_process(req)


def test_pre_process_allows_dummy_run() -> None:
    pre_process = get_mammoth_moda2_pre_process_func(_od_config())
    req = _batch(
        request_id="dummy_req_id",
        prompt={"prompt": "dummy run"},
        sampling=OmniDiffusionSamplingParams(height=512, width=512, num_inference_steps=20),
    ).requests[0]
    pre_process(req)
    assert req.batch_compatibility_key == ("mammoth_moda2_dit", 512, 512, 20)


def test_forward_casts_latents_to_vae_dtype() -> None:
    pipeline = _pipeline_shell()
    pipeline.gen_transformer = _FakeTransformer()
    pipeline.gen_image_condition_refiner = None
    vae = _FakeVae()
    vae.anchor = nn.Parameter(torch.zeros(1, dtype=torch.float16))
    pipeline.gen_vae = vae
    pipeline.gen_freqs_cis = torch.zeros(1)
    scheduler = _FakeScheduler()

    module = "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit"
    with (
        patch(f"{module}.FlowMatchEulerDiscreteScheduler", return_value=scheduler),
        patch(
            f"{module}.randn_tensor",
            side_effect=lambda s, **kw: torch.zeros(s, device=kw.get("device"), dtype=kw.get("dtype")),
        ),
    ):
        result = pipeline.forward(_batch())

    assert len(result) == 1
    assert vae.last_decode_dtype == torch.float16


def test_group_requests_by_geometry_and_steps() -> None:
    def make_req(idx: int, h: int, w: int, s: int) -> _MammothRequest:
        return _MammothRequest(
            index=idx,
            request_id=f"r{idx}",
            full_hidden_states=torch.zeros(1, 8),
            full_token_ids=[1],
            answer_start_index=0,
            height=h,
            width=w,
            text_guidance_scale=1.0,
            cfg_range=(0.0, 1.0),
            num_inference_steps=s,
            seed=None,
            generator=None,
            generator_device=None,
        )

    reqs = [make_req(0, 32, 32, 5), make_req(1, 32, 32, 5), make_req(2, 64, 64, 5)]
    groups = MammothModa2DiTPipeline._group_requests(reqs)
    assert sorted(groups) == [[0, 1], [2]]


def test_pad_cond_sequence_right_pads() -> None:
    embeds = [torch.ones(1, 2, 8), torch.ones(1, 4, 8) * 2]
    masks = [torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool)]
    padded, mask = _pad_cond_sequence(embeds, masks)
    assert padded.shape == (2, 4, 8)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, True]]


def test_forward_batched_matches_solo_output() -> None:
    pipeline = _pipeline_shell()
    pipeline.gen_transformer = _FakeTransformer()
    pipeline.gen_image_condition_refiner = None
    pipeline.gen_vae = _FakeVae()
    pipeline.gen_freqs_cis = torch.zeros(1)
    scheduler = _FakeScheduler()

    def fake_randn_tensor(shape, *, generator, device, dtype):
        if isinstance(generator, list):
            tensors = [
                torch.full((1, *shape[1:]), float(g.initial_seed() if g else 0), device=device, dtype=dtype)
                for g in generator
            ]
            return torch.cat(tensors, dim=0)
        val = float(generator.initial_seed() if generator else 0)
        return torch.full(shape, val, device=device, dtype=dtype)

    module = "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit"
    with (
        patch(f"{module}.FlowMatchEulerDiscreteScheduler", return_value=scheduler),
        patch(f"{module}.randn_tensor", side_effect=fake_randn_tensor),
    ):
        req1 = _batch(
            request_id="r1",
            sampling=OmniDiffusionSamplingParams(
                height=32, width=48, seed=10, guidance_scale=1.0, num_inference_steps=2
            ),
        ).requests[0]
        req2 = _batch(
            request_id="r2",
            sampling=OmniDiffusionSamplingParams(
                height=32, width=48, seed=20, guidance_scale=1.0, num_inference_steps=2
            ),
        ).requests[0]

        batched_out = pipeline.forward(DiffusionRequestBatch([req1, req2]))
        solo1 = pipeline.forward(DiffusionRequestBatch([req1]))
        solo2 = pipeline.forward(DiffusionRequestBatch([req2]))

    assert len(batched_out) == 2
    torch.testing.assert_close(batched_out[0].output, solo1[0].output)
    torch.testing.assert_close(batched_out[1].output, solo2[0].output)
    assert not torch.equal(batched_out[0].output, batched_out[1].output)


def test_refiner_output_length_preserved_when_input_len_differs_from_queries() -> None:
    pipeline = _pipeline_shell()
    pipeline.gen_transformer = _FakeTransformer()

    class _MockRefiner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

        def forward(self, features, mask=None):
            b, _, d = features.shape
            return torch.zeros(b, 8, d, device=features.device, dtype=features.dtype)

    pipeline.gen_image_condition_refiner = _MockRefiner()
    pipeline.gen_vae = _FakeVae()
    pipeline.gen_freqs_cis = torch.zeros(1)
    scheduler = _FakeScheduler()

    captured_cond_lens = []

    def fake_forward(*, hidden_states, text_hidden_states, **kwargs):
        captured_cond_lens.append(text_hidden_states.shape[1])
        return torch.zeros_like(hidden_states)

    module = "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit"
    with (
        patch(f"{module}.FlowMatchEulerDiscreteScheduler", return_value=scheduler),
        patch(
            f"{module}.randn_tensor",
            side_effect=lambda s, **kw: torch.zeros(s, device=kw.get("device"), dtype=kw.get("dtype")),
        ),
        patch.object(pipeline.gen_transformer, "forward", fake_forward),
    ):
        prompt = {
            "prompt": "",
            "additional_information": {
                "full_hidden_states": torch.zeros(6, 8),
                "full_token_ids": [10, 11, 100, 101, 102, 103],
                "answer_start_index": 2,
            },
        }
        batch = _batch(
            prompt=prompt,
            sampling=OmniDiffusionSamplingParams(height=32, width=48, guidance_scale=1.0, num_inference_steps=1),
        )
        pipeline.forward(batch)

    # 2 text tokens + 8 refined queries = 10 total conditioning tokens seen by transformer
    assert captured_cond_lens == [10]


def test_nested_image_embedder_compacts_unequal_text_lengths() -> None:
    from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import (
        Transformer2DModel,
    )

    model = object.__new__(Transformer2DModel)
    nn.Module.__init__(model)
    object.__setattr__(model, "_internal_dict", SimpleNamespace(patch_size=2))

    dim = 8
    batch_size = 2
    max_text_len = 4

    class _MockTimeCaptionEmbed(nn.Module):
        def forward(self, timestep, text_hidden_states, dtype):
            return torch.zeros(batch_size, dim), text_hidden_states

        def image_embedder(self, feats, mask=None):
            return torch.full((batch_size, 3, dim), 99.0)

    model.time_caption_embed = _MockTimeCaptionEmbed()
    model.x_embedder = nn.Identity()

    def mock_rope(freqs, mask, *args):
        return None, None, None, None, mask.sum(dim=-1).tolist(), []

    model.rope_embedder = mock_rope

    hidden_states = torch.zeros(batch_size, 4, 4, 4)
    timestep = torch.tensor([1.0, 1.0])
    text_hidden_states = torch.zeros(batch_size, max_text_len, dim)
    text_hidden_states[0, 0, :] = 1.0
    text_hidden_states[0, 1, :] = 2.0
    text_hidden_states[1, 0, :] = 10.0
    text_hidden_states[1, 1, :] = 20.0
    text_hidden_states[1, 2, :] = 30.0
    text_hidden_states[1, 3, :] = 40.0

    text_attention_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, True],
        ]
    )

    ar_image_hidden_states = torch.zeros(batch_size, 5, dim)

    res = model._prepare_embeddings(
        hidden_states=hidden_states,
        timestep=timestep,
        text_hidden_states=text_hidden_states,
        text_attention_mask=text_attention_mask,
        freqs_cis=torch.zeros(1),
        batch_size=batch_size,
        height=4,
        width=4,
        ar_image_hidden_states=ar_image_hidden_states,
    )

    compacted_text = res[1]
    compacted_mask = res[2]
    encoder_seq_lengths = res[9]

    assert encoder_seq_lengths == [5, 7]
    assert compacted_text.shape == (2, 7, dim)
    assert compacted_text[0, 0, 0].item() == 1.0
    assert compacted_text[0, 1, 0].item() == 2.0
    assert compacted_text[0, 2, 0].item() == 99.0
    assert compacted_text[0, 3, 0].item() == 99.0
    assert compacted_text[0, 4, 0].item() == 99.0
    assert compacted_mask[0].tolist() == [True, True, True, True, True, False, False]
    assert compacted_mask[1].tolist() == [True, True, True, True, True, True, True]


def test_pre_process_rejects_missing_visual_tokens_and_multi_output() -> None:
    pre_process = get_mammoth_moda2_pre_process_func(_od_config())

    multi_out = _batch(
        request_id="req-multi",
        sampling=OmniDiffusionSamplingParams(num_outputs_per_prompt=2),
    ).requests[0]
    with pytest.raises(ValueError, match="num_outputs_per_prompt == 1.*req-multi"):
        pre_process(multi_out)

    no_vis_prompt = {
        "prompt": "",
        "additional_information": {
            "full_hidden_states": torch.zeros(3, 8),
            "full_token_ids": [10, 11, 12],
            "answer_start_index": 1,
        },
    }
    no_vis_req = _batch(request_id="req-no-vis", prompt=no_vis_prompt).requests[0]
    with pytest.raises(ValueError, match="no visual-token hidden states.*req-no-vis"):
        pre_process(no_vis_req)

    valid_prompt = {
        "prompt": "",
        "additional_information": {
            "full_hidden_states": torch.zeros(3, 8),
            "full_token_ids": [10, 100, 101],
            "answer_start_index": 1,
        },
    }
    valid_req = _batch(request_id="req-valid", prompt=valid_prompt).requests[0]
    pre_process(valid_req)
    assert valid_req.batch_compatibility_key is not None


def test_cfg_precomputation_uses_cpu_booleans_without_device_sync() -> None:
    pipeline = _pipeline_shell()
    pipeline.gen_transformer = _FakeTransformer()
    pipeline.gen_image_condition_refiner = None
    pipeline.gen_vae = _FakeVae()
    pipeline.gen_freqs_cis = torch.zeros(1)
    scheduler = _FakeScheduler()

    raw_any = torch.Tensor.any
    tensor_any_called = []

    def wrapped_any(self, *args, **kwargs):
        tensor_any_called.append(self)
        return raw_any(self, *args, **kwargs)

    module = "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit"
    with (
        patch(f"{module}.FlowMatchEulerDiscreteScheduler", return_value=scheduler),
        patch(
            f"{module}.randn_tensor",
            side_effect=lambda s, **kw: torch.zeros(s, device=kw.get("device"), dtype=kw.get("dtype")),
        ),
        patch.object(torch.Tensor, "any", wrapped_any),
    ):
        req1 = _batch(
            request_id="r1",
            sampling=OmniDiffusionSamplingParams(
                height=32,
                width=48,
                seed=1,
                guidance_scale=4.0,
                num_inference_steps=5,
                extra_args={"cfg_range": [0.2, 0.8]},
            ),
        ).requests[0]
        req2 = _batch(
            request_id="r2",
            sampling=OmniDiffusionSamplingParams(
                height=32, width=48, seed=2, guidance_scale=1.0, num_inference_steps=5
            ),
        ).requests[0]

        pipeline.forward(DiffusionRequestBatch([req1, req2]))

    assert len(tensor_any_called) == 0


def test_dummy_request_splits_valid_ar_conditions_without_error() -> None:
    """Startup warmup dummy run must split into valid non-empty AR conditions."""
    pipeline = _pipeline_shell()
    batch = _batch(
        request_id="dummy_req_id",
        prompt={"prompt": "dummy run"},
        sampling=OmniDiffusionSamplingParams(height=512, width=512, seed=1, guidance_scale=0.0, num_inference_steps=2),
    )
    parsed = pipeline._parse_request(batch)
    text_cond, image_cond = pipeline._split_request_conditions(parsed)
    assert text_cond.shape == (1, pipeline._llm_hidden_size)
    assert image_cond.shape == (1, pipeline._llm_hidden_size)


@pytest.mark.parametrize(
    ("llm_cfg_patch", "expected_threshold"),
    [
        # Parent-level llm_config threshold
        ({"gen_vocab_start_index": 200}, 200),
        # Nested text_config threshold
        ({"text_config": {"model_type": "mammothmoda2_qwen2_5_vl_text", "gen_vocab_start_index": 350}}, 350),
        # Default-derived threshold (fallback to vocab_size / 152064)
        ({}, 152064),
    ],
)
def test_admission_and_inference_threshold_resolution_matches(llm_cfg_patch: dict, expected_threshold: int) -> None:
    """Verify identical threshold resolution for parent-level, nested, and default configs."""
    raw = _raw_config()
    raw["llm_config"] = {"model_type": "mammothmoda2_qwen2_5_vl", **llm_cfg_patch}
    od_cfg = OmniDiffusionConfig(
        model="/models/MammothModa2-Preview",
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(raw),
    )
    # 1. Check admission preprocessor resolution
    pre_process = get_mammoth_moda2_pre_process_func(od_cfg)

    # Token just below threshold -> rejected
    below_prompt = {
        "prompt": "",
        "additional_information": {
            "full_hidden_states": torch.zeros(2, 8),
            "full_token_ids": [10, expected_threshold - 1],
            "answer_start_index": 1,
        },
    }
    below_req = _batch(request_id="req-below", prompt=below_prompt).requests[0]
    with pytest.raises(ValueError, match="no visual-token hidden states.*req-below"):
        pre_process(below_req)

    # Token at or above threshold -> accepted
    at_prompt = {
        "prompt": "",
        "additional_information": {
            "full_hidden_states": torch.zeros(2, 8),
            "full_token_ids": [10, expected_threshold],
            "answer_start_index": 1,
        },
    }
    at_req = _batch(request_id="req-at", prompt=at_prompt).requests[0]
    pre_process(at_req)
    assert at_req.batch_compatibility_key is not None

    # 2. Check inference pipeline model config resolution
    pipeline = _pipeline_shell()
    pipeline.config = _build_mammoth_config(od_cfg)

    # Token below threshold rejected by pipeline
    parsed_below = pipeline._parse_request(_batch(request_id="req-inf-below", prompt=below_prompt))
    with pytest.raises(ValueError, match="no visual-token hidden states.*req-inf-below"):
        pipeline._split_request_conditions(parsed_below)

    # Token at threshold accepted by pipeline
    parsed_at = pipeline._parse_request(_batch(request_id="req-inf-at", prompt=at_prompt))
    text_cond, image_cond = pipeline._split_request_conditions(parsed_at)
    assert text_cond.shape == (1, 8)
    assert image_cond.shape == (1, 8)
