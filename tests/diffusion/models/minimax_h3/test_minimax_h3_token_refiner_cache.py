# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The token-refiner output is request-static, so the denoise loop avoids
recomputing it every step. How it does so depends on the offload mode:

* layerwise / no offload -- the refiner weights are resident on GPU for the
  whole request, so the loop precomputes the refined text once above the step
  loop and every ``forward`` receives it via ``refined_text_embed``.
* model (cpu) offload -- the model-level hook only onloads the transformer
  inside ``forward``; hoisting the refiner above the loop would run it while its
  weights are still on CPU, forcing an extra CPU<->GPU round-trip that a
  two-layer refiner can't pay back. So the loop does *not* hoist: it passes the
  raw text inputs and ``forward`` refines internally every step (weights already
  onloaded, no extra transfer), exactly matching the pre-optimization path and
  guaranteeing parity rather than a regression.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _RefinerCountingModel:
    """Minimal stand-in for MiniMaxH3DiTModel used by the denoise loop.

    ``compute_refined_text`` is the sole wrapper around ``condition_proj`` +
    ``token_refiner``; counting its calls counts the refiner runs. ``forward``
    mirrors the real contract: it uses the cached ``refined_text_embed`` when
    present, otherwise refines from the raw ``prompt_embeds`` inputs — exactly
    what the model does under model cpu offload.
    """

    def __init__(self, *, img_rows: int, audio_rows: int, hidden: int, enable_cpu_offload: bool) -> None:
        self.refine_calls = 0
        self.forward_calls = 0
        self._img_rows = img_rows
        self._audio_rows = audio_rows
        self._hidden = hidden
        self.od_config = SimpleNamespace(enable_cpu_offload=enable_cpu_offload)

    def compute_refined_text(self, *, prompt_embeds, refiner_packed_seq_params):
        self.refine_calls += 1
        text_len = int(prompt_embeds.shape[0])
        return torch.zeros(text_len, self._hidden, dtype=torch.bfloat16)

    def __call__(self, **kwargs):
        self.forward_calls += 1
        # Raw inputs are always present so either path works.
        assert "prompt_embeds" in kwargs
        assert "refiner_packed_seq_params" in kwargs
        if kwargs.get("refined_text_embed") is None:
            # Model cpu offload path: forward refines internally each step.
            self.compute_refined_text(
                prompt_embeds=kwargs["prompt_embeds"],
                refiner_packed_seq_params=kwargs["refiner_packed_seq_params"],
            )
        video_logits = torch.zeros(self._img_rows, 96, dtype=torch.float32)
        audio_logits = torch.zeros(self._audio_rows, 32, dtype=torch.float32)
        return video_logits, audio_logits


def _build_branch():
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        MiniMaxH3DenoiseBranch,
    )
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    packed = minimax_h3_packed_sequence(
        text_len=4,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        include_keyframe_cond=False,
    )
    text_len = int(packed["text_pos"].view(-1).shape[0])
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(text_len, 5120, dtype=torch.float32),
        token_tags=packed["token_tags"].clone(),
        device=torch.device("cpu"),
    )
    return branch


def _run_loop(branch, model, num_steps):
    from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
        minimax_h3_denoise_loop,
    )

    img_rows = int(branch.img_pos.shape[0])
    audio_rows = int(branch.audio_pos.shape[0])
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1).tolist()
    minimax_h3_denoise_loop(
        model=model,
        positive=branch,
        initial_video_rows=torch.zeros(img_rows, 96, dtype=torch.float32),
        initial_audio_rows=torch.zeros(audio_rows, 32, dtype=torch.float32),
        keyframe_cond_rows=None,
        sigmas_video=sigmas,
        sigmas_audio=sigmas,
        device=torch.device("cpu"),
    )


def _model_for(branch, *, enable_cpu_offload):
    return _RefinerCountingModel(
        img_rows=int(branch.img_pos.shape[0]),
        audio_rows=int(branch.audio_pos.shape[0]),
        hidden=5120,
        enable_cpu_offload=enable_cpu_offload,
    )


def test_refiner_hoisted_once_without_model_offload():
    """layerwise / no offload: refiner runs once, forward gets the cached value."""
    num_steps = 5
    branch = _build_branch()
    model = _model_for(branch, enable_cpu_offload=False)

    _run_loop(branch, model, num_steps)

    assert model.forward_calls == num_steps
    assert model.refine_calls == 1
    # The cached embedding is what every forward consumed.
    assert branch.static_kwargs.get("refined_text_embed") is not None


def test_refiner_refined_per_step_under_model_offload():
    """model cpu offload: no hoist. ``forward`` refines internally every step
    (matching the pre-optimization path), so there is no cached
    ``refined_text_embed`` and no extra CPU<->GPU transfer to regress on."""
    num_steps = 5
    branch = _build_branch()
    model = _model_for(branch, enable_cpu_offload=True)

    _run_loop(branch, model, num_steps)

    assert model.forward_calls == num_steps
    # The loop never hoists under model offload, so every forward refines
    # internally: one refine per step, and nothing cached on the branch.
    assert model.refine_calls == num_steps
    assert branch.static_kwargs.get("refined_text_embed") is None
