from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import vllm_omni.diffusion.models.nextstep_1_1.pipeline_nextstep_1_1 as nextstep_pipeline_module
from vllm_omni.diffusion.models.nextstep_1_1.modeling_nextstep_heads import FlowMatchingHead
from vllm_omni.diffusion.models.nextstep_1_1.pipeline_nextstep_1_1 import NextStep11Pipeline


class _DummyImageHead:
    def __init__(self, token_dim: int):
        self.token_dim = token_dim
        self.calls = []

    def sample(
        self,
        c: torch.Tensor,
        cfg: float,
        cfg_img: float,
        cfg_mult: int,
        timesteps_shift: float,
        num_sampling_steps: int,
        noise_repeat: int,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "batch": c.shape[0],
                "cfg": cfg,
                "cfg_img": cfg_img,
                "cfg_mult": cfg_mult,
                "noise_repeat": noise_repeat,
            }
        )
        batch_per_prompt = c.shape[0] // cfg_mult
        return torch.ones(batch_per_prompt, self.token_dim, dtype=c.dtype, device=c.device)


class _DummyModel:
    def __init__(self, hidden_dim: int, token_dim: int):
        self.hidden_dim = hidden_dim
        self.image_head = _DummyImageHead(token_dim)
        self.forward_batches = []

    def image_out_projector(self, c: torch.Tensor) -> torch.Tensor:
        return c

    def image_in_projector(self, sampled_tokens: torch.Tensor) -> torch.Tensor:
        bsz = sampled_tokens.shape[0]
        return torch.zeros(bsz, 1, self.hidden_dim, dtype=sampled_tokens.dtype, device=sampled_tokens.device)

    def forward_model(self, inputs_embeds: torch.Tensor, attention_mask, past_key_values, use_cache: bool):
        del attention_mask, use_cache
        self.forward_batches.append(inputs_embeds.shape[0])
        return SimpleNamespace(
            last_hidden_state=torch.zeros(
                inputs_embeds.shape[0],
                1,
                self.hidden_dim,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            ),
            past_key_values=past_key_values,
        )


def _make_minimal_pipeline_for_decoding(hidden_dim: int = 8, token_dim: int = 4):
    pipeline = object.__new__(NextStep11Pipeline)
    pipeline.config = SimpleNamespace(latent_channels=token_dim, latent_patch_size=1, use_gen_pos_embed=False)
    pipeline.model = _DummyModel(hidden_dim=hidden_dim, token_dim=token_dim)
    return pipeline


@pytest.mark.parametrize(
    ("cfg", "cfg_img", "has_image_conditions", "expected_cfg_mult", "expected_cfg_img"),
    [
        (1.0, 1.0, False, 1, 1.0),
        (7.5, 1.0, False, 2, 1.0),
        (7.5, 8.0, False, 2, 1.0),
        (7.5, 1.5, True, 3, 1.5),
    ],
)
def test_resolve_cfg_layout(cfg, cfg_img, has_image_conditions, expected_cfg_mult, expected_cfg_img):
    cfg_mult, effective_cfg_img = NextStep11Pipeline._resolve_cfg_layout(cfg, cfg_img, has_image_conditions)
    assert cfg_mult == expected_cfg_mult
    assert effective_cfg_img == expected_cfg_img


def test_build_captions_ignores_image_cfg_without_image_conditions():
    pipeline = object.__new__(NextStep11Pipeline)
    pipeline._image_str = lambda hw: f"<image:{hw}>"

    captions, images, cfg_mult, effective_cfg_img = pipeline._build_captions(
        captions=["a prompt"],
        images=None,
        num_images_per_caption=1,
        positive_prompt=None,
        negative_prompt="bad quality",
        cfg=7.5,
        cfg_img=8.0,
    )

    assert cfg_mult == 2
    assert effective_cfg_img == 1.0
    assert images is None
    assert captions == ["a prompt", "bad quality"]


def test_build_captions_enables_three_way_cfg_when_image_conditions_exist():
    pipeline = object.__new__(NextStep11Pipeline)
    pipeline._image_str = lambda hw: f"<image:{hw}>"

    image = Image.new("RGB", (64, 32))
    captions, images, cfg_mult, effective_cfg_img = pipeline._build_captions(
        captions=["a prompt"],
        images=[image],
        num_images_per_caption=1,
        positive_prompt=None,
        negative_prompt="bad quality",
        cfg=7.5,
        cfg_img=1.5,
    )

    assert cfg_mult == 3
    assert effective_cfg_img == 1.5
    assert len(captions) == 3
    assert captions[1].startswith("<image:")
    assert captions[2] == "bad quality"
    assert len(images) == 2


def test_decoding_non_parallel_uses_cfg_mult_for_sampling_and_duplication(monkeypatch):
    pipeline = _make_minimal_pipeline_for_decoding()
    monkeypatch.setattr(nextstep_pipeline_module, "get_classifier_free_guidance_world_size", lambda: 1)

    c = torch.zeros(2, 1, 8)
    attention_mask = torch.ones(2, 3, dtype=torch.long)

    tokens = pipeline.decoding(
        c=c,
        attention_mask=attention_mask,
        past_key_values=None,
        max_new_len=1,
        num_images_per_caption=1,
        cfg=7.5,
        cfg_img=1.0,
        cfg_mult=2,
        progress=False,
    )

    assert tokens.shape == (1, 1, 4)
    assert pipeline.model.image_head.calls[0]["cfg_mult"] == 2
    assert pipeline.model.forward_batches == [2]


def test_decoding_cfg_parallel_mismatch_falls_back_to_non_parallel(monkeypatch):
    pipeline = _make_minimal_pipeline_for_decoding()

    monkeypatch.setattr(nextstep_pipeline_module, "get_classifier_free_guidance_world_size", lambda: 2)

    def _unexpected(*args, **kwargs):
        del args, kwargs
        raise AssertionError("CFG rank/group should not be queried on mismatch fallback.")

    monkeypatch.setattr(nextstep_pipeline_module, "get_classifier_free_guidance_rank", _unexpected)
    monkeypatch.setattr(nextstep_pipeline_module, "get_cfg_group", _unexpected)

    c = torch.zeros(3, 1, 8)
    attention_mask = torch.ones(3, 3, dtype=torch.long)

    tokens = pipeline.decoding(
        c=c,
        attention_mask=attention_mask,
        past_key_values=None,
        max_new_len=1,
        num_images_per_caption=1,
        cfg=7.5,
        cfg_img=1.5,
        cfg_mult=3,
        progress=False,
    )

    assert tokens.shape == (1, 1, 4)
    assert pipeline.model.image_head.calls[0]["cfg_mult"] == 3
    assert pipeline.model.forward_batches == [3]


def test_decoding_rejects_incompatible_batch_and_cfg_mult(monkeypatch):
    pipeline = _make_minimal_pipeline_for_decoding()
    monkeypatch.setattr(nextstep_pipeline_module, "get_classifier_free_guidance_world_size", lambda: 1)

    with pytest.raises(ValueError, match="not divisible"):
        pipeline.decoding(
            c=torch.zeros(5, 1, 8),
            attention_mask=torch.ones(5, 3, dtype=torch.long),
            past_key_values=None,
            max_new_len=1,
            num_images_per_caption=1,
            cfg=7.5,
            cfg_img=1.0,
            cfg_mult=2,
            progress=False,
        )


def test_flow_matching_head_sample_validates_cfg_mult_divisibility():
    head = FlowMatchingHead(input_dim=4, cond_dim=8, dim=8, layers=1)

    with pytest.raises(ValueError, match="not divisible"):
        head.sample(
            c=torch.zeros(5, 8),
            cfg=2.0,
            cfg_img=1.0,
            cfg_mult=2,
            num_sampling_steps=1,
        )
