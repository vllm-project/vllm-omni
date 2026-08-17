from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.cache.teacache.extractors import extract_mammoth_moda2_context
from vllm_omni.diffusion.models.mammoth_moda2 import pipeline_mammothmoda2_dit as mammoth_pipeline_module
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import MammothModa2DiTPipeline
from vllm_omni.model_executor.models.mammoth_moda2 import mammoth_moda2 as mammoth_model_module
from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import MammothModa2ForConditionalGeneration

pytestmark = [pytest.mark.cpu]


class _FakeScheduler:
    def set_timesteps(self, num_inference_steps, device, num_tokens):  # noqa: ARG002
        self.timesteps = torch.arange(num_inference_steps, 0, -1, device=device, dtype=torch.float32)

    def step(self, model_pred, t, latents, return_dict=False):  # noqa: ARG002
        return (latents,)


class _FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=4)
        self.time_caption_embed = SimpleNamespace(image_embedder=None)
        self.param = nn.Parameter(torch.zeros(()))
        self.branches: list[str | None] = []
        self.calls: list[dict] = []

    def forward(self, hidden_states, **kwargs):
        self.branches.append(kwargs.get("teacache_branch"))
        self.calls.append(kwargs)
        return torch.zeros_like(hidden_states)


class _FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=None, shift_factor=None)

    def decode(self, latents, return_dict=False):  # noqa: ARG002
        return (latents,)


def _build_pipeline(monkeypatch):
    monkeypatch.setattr(mammoth_pipeline_module, "FlowMatchEulerDiscreteScheduler", _FakeScheduler)

    pipe = MammothModa2DiTPipeline.__new__(MammothModa2DiTPipeline)
    nn.Module.__init__(pipe)
    pipe.gen_transformer = _FakeTransformer()
    pipe.gen_vae = _FakeVAE()
    pipe.gen_image_condition_refiner = None
    pipe.gen_freqs_cis = []
    pipe._llm_hidden_size = 8
    pipe.cache_backend = None
    return pipe


def _runtime_info():
    return {
        "text_prompt_embeds": torch.randn(2, 8),
        "image_prompt_embeds": torch.randn(1, 8),
        "negative_prompt_embeds": torch.randn(1, 8),
        "negative_prompt_attention_mask": [True],
        "image_height": [32],
        "image_width": [32],
        "text_guidance_scale": [1.0],
        "cfg_range": [0.0, 1.0],
        "num_inference_steps": [4],
    }


def _run_pipeline(pipe, *, text_guidance_scale, cfg_range, num_inference_steps=4):
    pipe(
        inputs_embeds=torch.zeros(1, 8),
        runtime_additional_information=[_runtime_info()],
        sampling_extra_args=[
            {
                "text_guidance_scale": text_guidance_scale,
                "cfg_range": cfg_range,
                "num_inference_steps": num_inference_steps,
            }
        ],
    )
    return pipe.gen_transformer.branches


def test_mammoth_moda2_non_cfg_passes_positive_teacache_branch(monkeypatch):
    pipe = _build_pipeline(monkeypatch)

    branches = _run_pipeline(pipe, text_guidance_scale=1.0, cfg_range=[0.0, 1.0])

    assert branches == ["positive", "positive", "positive", "positive"]


def test_mammoth_moda2_dev_forwards_ar_image_conditioning_to_positive_branch(monkeypatch):
    """Dev keeps AR image tokens separate and forwards them to the positive DiT call."""
    pipe = _build_pipeline(monkeypatch)
    pipe.gen_transformer.time_caption_embed.image_embedder = object()

    _run_pipeline(
        pipe,
        text_guidance_scale=1.0,
        cfg_range=[0.0, 1.0],
        num_inference_steps=1,
    )

    assert len(pipe.gen_transformer.calls) == 1
    call = pipe.gen_transformer.calls[0]
    assert call["teacache_branch"] == "positive"
    assert call["text_hidden_states"].shape == (1, 2, 8)
    assert call["ar_image_hidden_states"].shape == (1, 1, 8)
    assert torch.equal(call["ar_image_attention_mask"], torch.ones(1, 1, dtype=torch.bool))


def test_mammoth_moda2_cfg_passes_positive_then_negative_teacache_branch(monkeypatch):
    pipe = _build_pipeline(monkeypatch)

    branches = _run_pipeline(pipe, text_guidance_scale=4.0, cfg_range=[0.0, 1.0])

    assert branches == [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
    ]


def test_mammoth_moda2_cfg_range_only_uses_negative_inside_range(monkeypatch):
    pipe = _build_pipeline(monkeypatch)

    branches = _run_pipeline(pipe, text_guidance_scale=4.0, cfg_range=[0.5, 1.0])

    assert branches == [
        "positive",
        "positive",
        "positive",
        "negative",
        "positive",
        "negative",
    ]


def test_mammoth_moda2_teacache_forwards_ar_image_conditioning():
    """The TeaCache extractor follows the full embedding contract, including the expanded text mask."""
    hidden_size = 4
    hidden_states = torch.zeros(1, hidden_size, 1, 1)
    timestep = torch.zeros(1)
    text_hidden_states = torch.zeros(1, 1, hidden_size)
    text_attention_mask = torch.ones(1, 1, dtype=torch.bool)
    freqs_cis = object()
    ar_image_hidden_states = torch.randn(1, 2, hidden_size)
    ar_image_attention_mask = torch.ones(1, 2, dtype=torch.bool)

    temb = torch.zeros(1, hidden_size)
    prepared_text_hidden_states = torch.randn(1, 3, hidden_size)
    prepared_text_attention_mask = torch.ones(1, 3, dtype=torch.bool)
    img_tokens = torch.randn(1, 1, hidden_size)
    img_mask = torch.ones(1, 1, dtype=torch.bool)
    context_rotary_emb = object()
    noise_rotary_emb = object()
    rotary_emb = object()

    first_layer = SimpleNamespace(
        norm1=Mock(side_effect=lambda joint_hidden_states, _temb: (joint_hidden_states,)),
    )
    module = SimpleNamespace(
        layers=[first_layer],
        config=SimpleNamespace(hidden_size=hidden_size, patch_size=1),
        _validate_inputs=Mock(return_value=(1, 1, 1)),
        _prepare_embeddings=Mock(
            return_value=(
                temb,
                prepared_text_hidden_states,
                prepared_text_attention_mask,
                img_tokens,
                img_mask,
                1,
                context_rotary_emb,
                noise_rotary_emb,
                rotary_emb,
                [3],
                [4],
            )
        ),
        _apply_refiners=Mock(return_value=(prepared_text_hidden_states, img_tokens)),
    )

    ctx = extract_mammoth_moda2_context(
        module,
        hidden_states=hidden_states,
        timestep=timestep,
        text_hidden_states=text_hidden_states,
        text_attention_mask=text_attention_mask,
        ref_image_hidden_states=None,
        ar_image_hidden_states=ar_image_hidden_states,
        ar_image_attention_mask=ar_image_attention_mask,
        freqs_cis=freqs_cis,
        teacache_branch="positive",
    )

    module._prepare_embeddings.assert_called_once_with(
        hidden_states,
        timestep,
        text_hidden_states,
        text_attention_mask,
        freqs_cis,
        1,
        1,
        1,
        ar_image_hidden_states,
        ar_image_attention_mask,
    )
    module._apply_refiners.assert_called_once_with(
        prepared_text_hidden_states,
        prepared_text_attention_mask,
        context_rotary_emb,
        img_tokens,
        img_mask,
        noise_rotary_emb,
        temb,
    )
    assert ctx.hidden_states.shape == (1, 4, hidden_size)
    assert ctx.extra_states["teacache_branch"] == "positive"


def test_mammoth_moda2_dit_stage_enables_cache_backend(monkeypatch):
    pipe = _build_pipeline(monkeypatch)
    fake_backend = SimpleNamespace(
        enable=Mock(),
        is_enabled=Mock(return_value=True),
        refresh=Mock(),
    )
    get_cache_backend = Mock(return_value=fake_backend)
    monkeypatch.setattr(mammoth_model_module, "get_cache_backend", get_cache_backend)
    wrapper = object.__new__(MammothModa2ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.model_stage = "dit"
    wrapper.dit = pipe
    wrapper.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            cache_backend="tea_cache",
            cache_config='{"rel_l1_thresh": 0.1}',
        )
    )

    wrapper._maybe_enable_dit_cache_backend()

    get_cache_backend.assert_called_once_with("tea_cache", {"rel_l1_thresh": 0.1})
    fake_backend.enable.assert_called_once_with(pipe)
    assert pipe.cache_backend is fake_backend


def test_mammoth_moda2_dit_stage_rejects_unsupported_cache_backend(monkeypatch):
    pipe = _build_pipeline(monkeypatch)
    get_cache_backend = Mock()
    monkeypatch.setattr(mammoth_model_module, "get_cache_backend", get_cache_backend)
    wrapper = object.__new__(MammothModa2ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.model_stage = "dit"
    wrapper.dit = pipe
    wrapper.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            cache_backend="cache_dit",
            cache_config={},
        )
    )

    with pytest.raises(ValueError, match="MammothModa2.*only supports.*tea_cache.*cache_dit"):
        wrapper._maybe_enable_dit_cache_backend()

    get_cache_backend.assert_not_called()


def test_mammoth_moda2_ar_stage_does_not_consume_diffusion_cache_config(monkeypatch):
    get_cache_backend = Mock()
    monkeypatch.setattr(mammoth_model_module, "get_cache_backend", get_cache_backend)
    wrapper = object.__new__(MammothModa2ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.model_stage = "ar"
    wrapper.dit = None
    wrapper.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            cache_backend="tea_cache",
            cache_config={"rel_l1_thresh": 0.1},
        )
    )

    wrapper._maybe_enable_dit_cache_backend()

    get_cache_backend.assert_not_called()


def test_mammoth_moda2_dit_stage_rejects_cache_config_without_backend(monkeypatch):
    pipe = _build_pipeline(monkeypatch)
    get_cache_backend = Mock()
    monkeypatch.setattr(mammoth_model_module, "get_cache_backend", get_cache_backend)
    wrapper = object.__new__(MammothModa2ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.model_stage = "dit"
    wrapper.dit = pipe
    wrapper.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            cache_backend="none",
            cache_config={"rel_l1_thresh": 0.1},
        )
    )

    with pytest.raises(ValueError, match="cache_config requires cache_backend='tea_cache'"):
        wrapper._maybe_enable_dit_cache_backend()

    get_cache_backend.assert_not_called()


@pytest.mark.parametrize(
    ("cache_config", "message"),
    [
        ("not-json", "valid JSON object"),
        ("[]", "JSON object"),
        ({"Fn_compute_blocks": 2}, "not valid for cache_backend='tea_cache'"),
    ],
)
def test_mammoth_moda2_dit_stage_rejects_invalid_teacache_config(monkeypatch, cache_config, message):
    pipe = _build_pipeline(monkeypatch)
    get_cache_backend = Mock()
    monkeypatch.setattr(mammoth_model_module, "get_cache_backend", get_cache_backend)
    wrapper = object.__new__(MammothModa2ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.model_stage = "dit"
    wrapper.dit = pipe
    wrapper.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            cache_backend="tea_cache",
            cache_config=cache_config,
        )
    )

    with pytest.raises(ValueError, match=message):
        wrapper._maybe_enable_dit_cache_backend()

    get_cache_backend.assert_not_called()


def test_mammoth_moda2_dit_stage_refreshes_cache_from_sampling_steps(monkeypatch):
    pipe = _build_pipeline(monkeypatch)
    fake_backend = SimpleNamespace(
        is_enabled=Mock(return_value=True),
        refresh=Mock(),
    )
    pipe.cache_backend = fake_backend

    _run_pipeline(pipe, text_guidance_scale=1.0, cfg_range=[0.0, 1.0], num_inference_steps=3)

    fake_backend.refresh.assert_called_once_with(pipe, 3, verbose=False)
