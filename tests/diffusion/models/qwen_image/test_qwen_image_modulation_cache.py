# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer
from vllm_omni.diffusion.models.qwen_image.cfg_parallel import QwenImageCFGParallelMixin
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenImageTransformerBlock

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CountingProjection(nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return value + self.offset


def _block(*, zero_cond_t: bool = False) -> QwenImageTransformerBlock:
    block = QwenImageTransformerBlock.__new__(QwenImageTransformerBlock)
    nn.Module.__init__(block)
    block.img_mod = _CountingProjection(1.0)
    block.txt_mod = _CountingProjection(2.0)
    block.zero_cond_t = zero_cond_t
    block._modulation_cache = None
    block.eval()
    return block


def _run_cache(
    block: QwenImageTransformerBlock,
    temb: torch.Tensor,
    timestep: torch.Tensor,
    guidance: torch.Tensor | None = None,
    additional_t_cond: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states is None:
        hidden_states = torch.zeros(1, 2, temb.shape[-1], dtype=temb.dtype)
    return block._get_modulation_params(temb, True, timestep, guidance, additional_t_cond, hidden_states)


def test_modulation_cache_reuses_once_then_recomputes() -> None:
    block = _block()
    temb = torch.randn(1, 8)
    timestep = torch.tensor([0.5])
    guidance = torch.tensor([1.0])

    with torch.no_grad():
        first = _run_cache(block, temb, timestep, guidance)
        second = _run_cache(block, temb, timestep, guidance)
        third = _run_cache(block, temb, timestep, guidance)

    assert block.img_mod.calls == 2
    assert block.txt_mod.calls == 2
    assert second[0] is first[0]
    assert second[1] is first[1]
    assert third[0] is not first[0]


@pytest.mark.parametrize("changed", ["timestep", "guidance", "additional_t_cond", "dtype"])
def test_modulation_cache_invalidates_changed_inputs(changed: str) -> None:
    block = _block()
    temb = torch.randn(1, 8)
    timestep = torch.tensor([0.5])
    guidance = torch.tensor([1.0])
    additional_t_cond = torch.tensor([0])
    hidden_states = torch.zeros(1, 2, 8)

    with torch.no_grad():
        _run_cache(block, temb, timestep, guidance, additional_t_cond, hidden_states)
        if changed == "timestep":
            timestep = timestep.clone()
        elif changed == "guidance":
            guidance = guidance.clone()
        elif changed == "additional_t_cond":
            additional_t_cond = additional_t_cond.clone()
        else:
            hidden_states = hidden_states.double()
        _run_cache(block, temb, timestep, guidance, additional_t_cond, hidden_states)

    assert block.img_mod.calls == 2
    assert block.txt_mod.calls == 2


def test_modulation_cache_invalidates_in_place_change() -> None:
    block = _block()
    temb = torch.randn(1, 8)
    timestep = torch.tensor([0.5])

    with torch.no_grad():
        _run_cache(block, temb, timestep)
        timestep.add_(1)
        _run_cache(block, temb, timestep)

    assert block.img_mod.calls == 2
    assert block.txt_mod.calls == 2


def test_modulation_cache_supports_inference_tensors() -> None:
    block = _block()
    with torch.inference_mode():
        temb = torch.randn(1, 8)
        timestep = torch.tensor([0.5])
        first = _run_cache(block, temb, timestep)
        second = _run_cache(block, temb, timestep)

    assert block.img_mod.calls == 1
    assert block.txt_mod.calls == 1
    assert second[0] is first[0]
    assert second[1] is first[1]


@pytest.mark.parametrize("fallback", ["training", "grad", "compile", "cuda_graph"])
def test_modulation_cache_falls_back(monkeypatch: pytest.MonkeyPatch, fallback: str) -> None:
    block = _block()
    temb = torch.randn(1, 8)
    timestep = torch.tensor([0.5])
    context = torch.no_grad()

    if fallback == "training":
        block.train()
    elif fallback == "grad":
        context = nullcontext()
    elif fallback == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    else:
        monkeypatch.setattr(qwen_image_transformer, "_is_cuda_graph_capturing", lambda: True)

    with context:
        _run_cache(block, temb, timestep)
        _run_cache(block, temb, timestep)

    assert block.img_mod.calls == 2
    assert block.txt_mod.calls == 2
    assert block._modulation_cache is None


def test_modulation_cache_compile_fallback_is_fullgraph() -> None:
    block = _block()
    block.img_mod = nn.Identity()
    block.txt_mod = nn.Identity()
    compiled = torch.compile(block._get_modulation_params, backend="eager", fullgraph=True)
    temb = torch.randn(1, 8)
    timestep = torch.tensor([0.5])
    hidden_states = torch.zeros(1, 2, 8)

    with torch.no_grad():
        img_mod_params, txt_mod_params = compiled(temb, True, timestep, None, None, hidden_states)

    torch.testing.assert_close(img_mod_params, temb)
    torch.testing.assert_close(txt_mod_params, temb)
    assert block._modulation_cache is None


def test_modulation_cache_disabled_without_serial_cfg() -> None:
    block = _block()
    temb = torch.randn(1, 8)
    timestep = torch.tensor([0.5])
    hidden_states = torch.zeros(1, 2, 8)

    with torch.no_grad():
        block._get_modulation_params(temb, False, timestep, None, None, hidden_states)
        block._get_modulation_params(temb, False, timestep, None, None, hidden_states)

    assert block.img_mod.calls == 2
    assert block.txt_mod.calls == 2
    assert block._modulation_cache is None


def test_zero_condition_text_projection_uses_half_temb() -> None:
    block = _block(zero_cond_t=True)
    temb = torch.randn(2, 8)
    timestep = torch.tensor([0.5])

    with torch.no_grad():
        img_mod_params, txt_mod_params = _run_cache(block, temb, timestep)

    assert img_mod_params.shape[0] == 2
    assert txt_mod_params.shape[0] == 1


def test_step_builder_shares_model_timestep_across_cfg_branches() -> None:
    latents = torch.zeros(1, 4, 8)
    positive_kwargs, negative_kwargs, _ = QwenImagePipeline._build_denoise_kwargs(
        None,
        latents=latents,
        timestep=torch.tensor(500.0),
        guidance=torch.tensor([1.0]),
        prompt_embeds=torch.zeros(1, 2, 8),
        prompt_embeds_mask=torch.ones(1, 2, dtype=torch.bool),
        img_shapes=[(1, 2, 2)],
        txt_seq_lens=[2],
        do_true_cfg=True,
        negative_prompt_embeds=torch.zeros(1, 2, 8),
        negative_prompt_embeds_mask=torch.ones(1, 2, dtype=torch.bool),
        negative_txt_seq_lens=[2],
    )

    assert negative_kwargs is not None
    assert positive_kwargs["timestep"] is negative_kwargs["timestep"]


class _Progress:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def update(self) -> None:
        pass


class _Scheduler:
    def set_begin_index(self, _index: int) -> None:
        pass


class _Transformer:
    do_true_cfg = False


class _DummyCFG(QwenImageCFGParallelMixin):
    def __init__(self):
        self.scheduler = _Scheduler()
        self.transformer = _Transformer()
        self.interrupt = False
        self.seen = None

    def progress_bar(self, total: int):
        return _Progress()

    def predict_noise_maybe_with_cfg(self, _do_true_cfg, _scale, positive, negative, *_args):
        self.seen = (positive, negative)
        return torch.zeros(1)

    def scheduler_step_maybe_with_cfg(self, _noise, _timestep, latents, _do_true_cfg):
        return latents


def test_streaming_diffuse_shares_model_timestep_across_cfg_branches() -> None:
    pipeline = _DummyCFG()
    tensor = torch.zeros(1, 2, 8)
    mask = torch.ones(1, 2, dtype=torch.bool)
    pipeline.diffuse(
        prompt_embeds=tensor,
        prompt_embeds_mask=mask,
        negative_prompt_embeds=tensor,
        negative_prompt_embeds_mask=mask,
        latents=torch.zeros(1, 4, 8),
        img_shapes=[(1, 2, 2)],
        txt_seq_lens=torch.tensor([2]),
        negative_txt_seq_lens=torch.tensor([2]),
        timesteps=torch.tensor([500.0]),
        do_true_cfg=True,
        guidance=torch.tensor([1.0]),
        true_cfg_scale=4.0,
    )

    assert pipeline.seen is not None
    positive_kwargs, negative_kwargs = pipeline.seen
    assert negative_kwargs is not None
    assert positive_kwargs["timestep"] is negative_kwargs["timestep"]
