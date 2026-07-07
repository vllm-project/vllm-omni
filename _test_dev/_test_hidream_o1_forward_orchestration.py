# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L3 orchestration test for HiDreamO1ImagePipeline.forward() denoise math.

Usage:
    python _test_dev/_test_hidream_o1_forward_orchestration.py
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from vllm_omni.diffusion.models.hidream_o1_image import pipeline_hidream_o1_image as pmod
from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    PATCH_SIZE,
    HiDreamO1ImagePipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

IMAGE_LEN = 6                            # h=64, w=96 -> 2x3 patches
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE  # = 3072


class _FakeScheduler:
    """Captures the model_output tensor passed to step() and returns sample unchanged."""
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.seen_model_output = None
        self.step_calls = 0

    def step(self, model_output, timestep, sample, return_dict):
        del timestep, return_dict
        self.step_calls += 1
        self.seen_model_output = model_output.detach().clone()
        return (sample,)


def _fake_build_t2i_text_sample(*, prompt, height, width, tokenizer, processor, model_config):
    """Returns a fake sample whose vinput_mask has exactly (h/32)*(w/32) True slots.
    Both cond ('a cat') and uncond (' ') build the same-shape sample here since
    the L3 test only cares about mask sizing, not the actual token ids."""
    del prompt, tokenizer, processor, model_config
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
    text_len = 3
    all_len = text_len + image_len
    vinput_mask = torch.zeros((1, all_len), dtype=torch.bool)
    vinput_mask[0, text_len:] = True
    return {
        'input_ids':    torch.zeros((1, text_len), dtype=torch.long),
        'position_ids': torch.zeros((3, 1, all_len), dtype=torch.long),
        'token_types':  torch.zeros((1, all_len), dtype=torch.long),
        'vinput_mask':  vinput_mask,
    }


def _make_pipe_stub(x_pred_values, constant_z):
    """Bypass HiDreamO1ImagePipeline.__init__ (no weight load), but still call
    nn.Module.__init__ so _parameters/_buffers/_modules exist. Otherwise
    patch.object's delattr on exit blows up in nn.Module.__delattr__.

    x_pred_values is a list; _forward_once returns the value at the current
    call index (cond call 0, uncond call 1). Wrapped in a list so the
    closure can mutate it."""
    pipe = HiDreamO1ImagePipeline.__new__(HiDreamO1ImagePipeline)
    nn.Module.__init__(pipe)
    pipe.device = torch.device('cpu')
    pipe.dtype = torch.bfloat16
    pipe.tokenizer = None
    pipe.processor = None
    pipe.model = SimpleNamespace(config=None)

    call_idx = [0]
    pipe._forward_once_call_count = call_idx

    def _mock_prepare_noise(height, width, seed, dtype, device):
        del height, width, seed
        return constant_z.to(device, dtype)

    def _mock_forward_once(sample, z_in, t_pixeldit):
        del sample, t_pixeldit
        val = x_pred_values[call_idx[0]]
        call_idx[0] += 1
        return torch.full_like(z_in, val, dtype=torch.float32)

    pipe._prepare_noise_and_patchify = _mock_prepare_noise
    pipe._forward_once = _mock_forward_once
    return pipe


def _make_req(seed=42, guidance_scale=1.0):
    """64x96 T2I request. Note: forward() calls _resolve_generation_params which
    would snap this via find_closest_resolution, so we also mock resolve to
    keep the fake dims (64, 96) end-to-end."""
    sp = OmniDiffusionSamplingParams(
        height=64, width=96, num_inference_steps=1, seed=seed, guidance_scale=guidance_scale,
    )
    return OmniDiffusionRequest(prompts=['a cat'], sampling_params=sp, request_id='orch')


def _run_case(x_pred_values, guidance_scale):
    """Run one forward() with a fresh stub + scheduler; return the fake_sched
    (for model_output inspection) and pipe (for call count)."""
    constant_z = torch.ones((1, IMAGE_LEN, PATCH_DIM), dtype=torch.bfloat16)
    pipe = _make_pipe_stub(x_pred_values=x_pred_values, constant_z=constant_z)
    fake_sched = _FakeScheduler(timesteps=torch.tensor([500.0]))

    with patch.object(pmod, 'build_hidream_o1_scheduler', lambda **kw: fake_sched), \
         patch.object(pmod, 'build_t2i_text_sample', _fake_build_t2i_text_sample), \
         patch.object(pipe, '_resolve_generation_params',
                      lambda req: ('a cat', 64, 96, 1, 42, guidance_scale)):
        out = pipe.forward(_make_req())
    return out, fake_sched, pipe


def _check_output_envelope(out, expected_model_output_scalar, fake_sched, expected_calls, pipe):
    z_out, h_out, w_out = out.output
    assert (h_out, w_out) == (64, 96), f'envelope hw: ({h_out}, {w_out})'
    assert z_out.shape == (1, IMAGE_LEN, PATCH_DIM), f'z_out shape {z_out.shape}'
    assert z_out.dtype == torch.bfloat16, f'z_out dtype {z_out.dtype}'
    assert fake_sched.step_calls == 1, f'expected 1 sched.step call, got {fake_sched.step_calls}'
    actual_calls = pipe._forward_once_call_count[0]
    assert actual_calls == expected_calls, \
        f'expected exactly {expected_calls} _forward_once calls, got {actual_calls}'

    expected = torch.full((1, IMAGE_LEN, PATCH_DIM), expected_model_output_scalar, dtype=torch.float32)
    assert fake_sched.seen_model_output.shape == expected.shape, \
        f'model_output shape {fake_sched.seen_model_output.shape}'
    assert fake_sched.seen_model_output.dtype == torch.float32, \
        f'model_output must be fp32 for scheduler, got {fake_sched.seen_model_output.dtype}'
    torch.testing.assert_close(
        fake_sched.seen_model_output, expected, rtol=1e-4, atol=1e-4,
    )


# Case 1: cond-only 1-step (guidance=1.0, no CFG branch)
#   z=1, x_pred_cond=3, timestep=500 -> sigma=0.5, t_pixeldit=0.5
#   v_cond = (3-1)/0.5 = 4
#   v_guided = v_cond = 4
#   model_output = -4
out1, sched1, pipe1 = _run_case(x_pred_values=[3.0], guidance_scale=1.0)
_check_output_envelope(out1, -4.0, sched1, expected_calls=1, pipe=pipe1)
print(f'orch cond-only 1-step    : guidance=1.0 model_output=-4 (v_cond=(3-1)/0.5=4), forward_once_calls=1')

# Case 2: CFG 1-step (guidance=5.0 triggers second forward)
#   z=1, x_pred_cond=3, x_pred_uncond=1, timestep=500 -> sigma=0.5
#   v_cond   = (3-1)/0.5 = 4
#   v_uncond = (1-1)/0.5 = 0
#   v_guided = 0 + 5*(4-0) = 20
#   model_output = -20
out2, sched2, pipe2 = _run_case(x_pred_values=[3.0, 1.0], guidance_scale=5.0)
_check_output_envelope(out2, -20.0, sched2, expected_calls=2, pipe=pipe2)
print(f'orch CFG guidance=5.0    : model_output=-20 (v_uncond=0 + 5*(4-0)=20), forward_once_calls=2')

# Case 3: guidance=1.0 boundary -- CFG trigger is strict `> 1.0`, so 1.0 stays cond-only.
#   Same math as Case 1 -- verifies boundary behavior explicitly.
out3, sched3, pipe3 = _run_case(x_pred_values=[3.0], guidance_scale=1.0)
_check_output_envelope(out3, -4.0, sched3, expected_calls=1, pipe=pipe3)
print(f'orch CFG boundary=1.0    : model_output=-4 (guidance=1.0 stays cond-only), forward_once_calls=1')

print('pass')


# output:
# orch cond-only 1-step    : guidance=1.0 model_output=-4 (v_cond=(3-1)/0.5=4), forward_once_calls=1
# orch CFG guidance=5.0    : model_output=-20 (v_uncond=0 + 5*(4-0)=20), forward_once_calls=2
# orch CFG boundary=1.0    : model_output=-4 (guidance=1.0 stays cond-only), forward_once_calls=1
# pass
