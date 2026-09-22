# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request batching: independent prefixes/RNG, fused denoise and ordered outputs."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from vllm_omni.diffusion.models.sensenova_u1.batching import image_count, merge_conditioning
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
    SenseNovaU1Pipeline,
    get_sensenova_u1_pre_process_func,
)
from vllm_omni.diffusion.output_formatter import format_diffusion_outputs, normalize_diffusion_postprocess_output
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _request(name="first", *, count=1, seed=42, extra=None, mode="t2i"):
    prompt = {"prompt": name}
    if mode == "text":
        prompt["modalities"] = ["text"]
    elif mode == "it2i":
        prompt["multi_modal_data"] = {"image": [object()]}
    return OmniDiffusionRequest(
        request_id=name,
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(
            width=32, height=32, num_inference_steps=2, num_outputs_per_prompt=count, seed=seed, extra_args=extra or {}
        ),
    )


def _prefix(length, count=1, offset=0, branches=("cond", "uncond")):
    result = {}
    for branch in branches:
        cache = DynamicCache()
        for layer in range(2):
            keys = torch.randn(1, 2, length, 4).expand(count, -1, -1, -1)
            values = torch.randn(1, 2, length, 4).expand(count, -1, -1, -1)
            cache.update(keys, values, layer)
        result[branch] = cache
        result[f"idx_{branch}"] = torch.arange(3).repeat(3, 1) + offset
    return result


@pytest.mark.parametrize("lengths", [(2, 5), (4, 4)])
@pytest.mark.parametrize("branches", [("cond", "uncond"), ("cond", "img_cond", "uncond")])
def test_padded_attention_matches_independent_requests(lengths, branches):
    torch.manual_seed(10)
    prefixes = [_prefix(length, 2, offset=length, branches=branches) for length in lengths]
    merged = merge_conditioning(prefixes, [2, 2], image_tokens=3)
    query, image_k, image_v = [torch.randn(4, 2, 3, 4) for _ in range(3)]
    for branch in branches:
        mask = merged[f"mask_{branch}"]["full_attention"]
        assert (mask is None) == (lengths[0] == lengths[1])
        if mask is not None:
            assert mask.dtype == torch.bool
            assert mask.shape == (4, max(lengths) + 3)
        for layer_idx in range(2):
            layer = merged[branch].layers[layer_idx]
            actual = F.scaled_dot_product_attention(
                query,
                torch.cat([layer.keys, image_k], dim=2),
                torch.cat([layer.values, image_v], dim=2),
                attn_mask=None if mask is None else mask[:, None, None, :],
            )
            expected = []
            for i, prefix in enumerate(prefixes):
                rows = slice(i * 2, (i + 1) * 2)
                source = prefix[branch].layers[layer_idx]
                expected.append(
                    F.scaled_dot_product_attention(
                        query[rows],
                        torch.cat([source.keys, image_k[rows]], dim=2),
                        torch.cat([source.values, image_v[rows]], dim=2),
                    )
                )
                torch.testing.assert_close(
                    merged[f"idx_{branch}"][:, rows], prefix[f"idx_{branch}"].unsqueeze(1).expand(-1, 2, -1)
                )
            torch.testing.assert_close(actual, torch.cat(expected), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("count,legacy,expected", [(1, None, 1), (2, None, 2), (1, 3, 3), (3, 3, 3)])
def test_image_count_alias(count, legacy, expected):
    req = _request(count=count, extra={"batch_size": legacy} if legacy is not None else {})
    assert image_count(req.sampling_params) == expected
    pre = get_sensenova_u1_pre_process_func(SimpleNamespace(step_execution=False))
    assert pre(req).sampling_params.num_outputs_per_prompt == expected


@pytest.mark.parametrize("count,legacy", [(2, 3), (0, None), (1, -1), (1, True), (1, 1.5)])
def test_invalid_image_counts(count, legacy):
    with pytest.raises(ValueError):
        image_count(SimpleNamespace(num_outputs_per_prompt=count, extra_args={"batch_size": legacy}))


def test_admission_separates_cfg_and_modes_but_not_seed_or_think():
    pre = get_sensenova_u1_pre_process_func(SimpleNamespace(step_execution=False))
    first = pre(_request()).batch_compatibility_key
    assert pre(_request("other", seed=3, extra={"think": True})).batch_compatibility_key == first
    for req in [_request(extra={"cfg_scale": 5}), _request(mode="it2i"), _request(mode="text")]:
        assert pre(req).batch_compatibility_key != first
    assert (
        pre(_request("a", mode="text")).batch_compatibility_key
        != pre(_request("b", mode="text")).batch_compatibility_key
    )
    step_request = _request()
    get_sensenova_u1_pre_process_func(SimpleNamespace(step_execution=True))(step_request)
    assert step_request.batch_compatibility_key is None


def _pipeline():
    pipe = object.__new__(SenseNovaU1Pipeline)
    pipe.device = torch.device("cpu")
    pipe.patch_size = 8
    pipe.merge_size = 2
    pipe.od_config = SimpleNamespace(dtype=torch.float32, cache_backend="none")
    pipe.model_cfg = SimpleNamespace(noise_scale=1.0, noise_scale_mode="constant", noise_scale_max_value=1.0)
    pipe._apply_time_schedule = lambda timesteps, *args: timesteps
    return pipe


def test_noise_is_seeded_per_request_and_supports_generator_lists():
    pipe = _pipeline()
    for seed in (42, 73):
        p = pipe._parse_request(DiffusionRequestBatch([_request(count=2, seed=seed)]))
        actual = pipe._init_noise_and_schedule(p).image_prediction
        expected = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(seed))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    p.image_generator = [torch.Generator().manual_seed(seed) for seed in (42, 73)]
    actual = pipe._init_noise_and_schedule(p).image_prediction
    expected = torch.cat(
        [torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(seed)) for seed in (42, 73)]
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    p.image_generator.pop()
    with pytest.raises(ValueError, match="generator lists"):
        pipe._init_noise_and_schedule(p)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("mode", ["t2i", "it2i"])
def test_batch_finishes_each_ar_prefix_then_fuses_denoise_and_splits_outputs(count, mode):
    pipe = _pipeline()
    events = []
    requests = [_request("first", count=count, mode=mode), _request("second", count=count, seed=73, mode=mode)]

    def prefix(p, ns, *args):
        events.append(("prefix", p.prompt))
        return SimpleNamespace(cursor=SimpleNamespace(finished=False, name=p.prompt))

    def think(cursor):
        events.append(("think", cursor.name))
        cursor.finished = True

    def caches(p, ns, ctx, *, prepare_flash):
        assert ctx.cursor.finished and not prepare_flash
        events.append(("save", p.prompt))
        cache = _prefix(2 if p.prompt == "first" else 5, count)
        for branch in ("cond", "uncond"):
            cache[f"idx_{branch}"] = torch.zeros(3, ns.token_h * ns.token_w, dtype=torch.long)
        return cache, p.prompt

    def denoise(images, ns, caches, p, step, is_it2i):
        assert is_it2i == (mode == "it2i")
        assert images.shape[0] == p.batch_size == 2 * count
        assert caches["cond"].layers[0].keys.shape[0] == 2 * count
        events.append(("denoise", step))
        return images, torch.zeros_like(images)

    pipe._t2i_prefix = pipe._it2i_prefix = prefix
    pipe._t2i_caches = pipe._it2i_caches = caches
    pipe._think_step = think
    pipe._denoise_one = denoise
    pipe._advance_latents = lambda z, *args: z
    # Input encoding isn't part of this orchestration test.
    pipe._extract_input_images = lambda prompt: [object()] if mode == "it2i" else None
    outputs = pipe.forward(DiffusionRequestBatch(requests))
    assert events == [(stage, name) for name in ("first", "second") for stage in ("prefix", "think", "save")] + [
        ("denoise", 0),
        ("denoise", 1),
    ]
    assert len(outputs) == 2
    for req, output in zip(requests, outputs, strict=True):
        assert output.output["metadata"]["text"]["think_text"] == req.request_id
        ns = pipe._init_noise_and_schedule(pipe._parse_request(DiffusionRequestBatch([req])))
        expected = pipe._denoising_output({}, ns.image_prediction).output["payload"]["image"]
        actual = output.output["payload"]["image"]
        if count == 1:
            assert actual.tobytes() == expected.tobytes()
        else:
            assert len(actual) == count
            assert [im.tobytes() for im in actual] == [im.tobytes() for im in expected]


def test_incompatible_batch_rejected_before_preparation():
    pipe = _pipeline()
    with pytest.raises(ValueError, match="compatible"):
        pipe.forward(DiffusionRequestBatch([_request(), _request("other", extra={"cfg_scale": 9})]))
    with pytest.raises(ValueError, match="text output"):
        pipe.forward(DiffusionRequestBatch([_request(mode="text"), _request("other", mode="text")]))
    pipe.od_config.cache_backend = "cache_dit"
    with pytest.raises(ValueError, match="cache_backend"):
        pipe.forward(DiffusionRequestBatch([_request(), _request("other")]))


@pytest.mark.parametrize("count", [1, 2])
def test_single_request_uses_list_contract_and_formatter_preserves_all_images(count, monkeypatch):
    pipe = _pipeline()
    req = _request(count=count)
    pipe._forward_t2i = lambda p: pipe._denoising_output({}, pipe._init_noise_and_schedule(p).image_prediction)
    outputs = pipe.forward(DiffusionRequestBatch([req]))
    assert isinstance(outputs, list) and len(outputs) == 1
    monkeypatch.setattr("vllm_omni.diffusion.output_formatter.supports_audio_output", lambda _: False)
    [result] = format_diffusion_outputs(
        request=req,
        od_config=SimpleNamespace(model_class_name="SenseNovaU1Pipeline"),
        diffusion_output=outputs[0],
        output_data=outputs[0].output,
        postprocess_output=normalize_diffusion_postprocess_output(outputs[0].output),
    )
    assert result.request_id == req.request_id
    assert len(result.images) == count
    assert result.metrics["image_num"] == count
