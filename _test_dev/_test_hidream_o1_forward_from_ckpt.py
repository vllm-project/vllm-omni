# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L4 real-checkpoint end-to-end HiDreamO1ImagePipeline.forward() test.

Usage:
    export HIDREAM_O1_MODEL_DIR=/path/to/HiDream-O1-Image
    python _test_dev/_test_hidream_o1_forward_from_ckpt.py
"""
from __future__ import annotations

import os

import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    HiDreamO1ImagePipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL_DIR = os.environ.get('HIDREAM_O1_MODEL_DIR')
assert MODEL_DIR, 'set HIDREAM_O1_MODEL_DIR to the HiDream-O1-Image checkpoint dir'

od_config = OmniDiffusionConfig(
    model=MODEL_DIR,
    dtype=torch.bfloat16,
    custom_pipeline_args={'model_type': 'full'},
    enable_diffusion_pipeline_profiler=False,
)

print(f'config                   : model_dir={MODEL_DIR}')

pipe = HiDreamO1ImagePipeline(od_config=od_config)


# 2048x2048: unique square in PREDEFINED_RESOLUTIONS, so any square input snaps here.
# We pass 2048x2048 explicitly so the test name matches actual dims (avoid the
# "1024x1024 named, 2048x2048 executed" trap).
def _run_forward(prompt, guidance_scale, label):
    """One forward call + envelope/dtype/NaN check; prints tensor stats."""
    req = OmniDiffusionRequest(
        prompts=[prompt],
        sampling_params=OmniDiffusionSamplingParams(
            height=2048, width=2048,
            num_inference_steps=4,
            seed=42, guidance_scale=guidance_scale,
        ),
        request_id=f'ckpt-{label}',
    )
    print(f'request {label:<8}         : prompt={prompt!r} h=2048 w=2048 steps=4 seed=42 guidance_scale={guidance_scale}')

    out = pipe.forward(req)

    assert isinstance(out.output, tuple) and len(out.output) == 3, \
        f'envelope must be a 3-tuple (z, h, w), got {type(out.output).__name__}'
    z, snapped_h, snapped_w = out.output
    assert (snapped_h, snapped_w) == (2048, 2048), f'envelope hw: ({snapped_h}, {snapped_w})'
    assert isinstance(z, torch.Tensor), f'z must be torch.Tensor, got {type(z).__name__}'
    assert z.shape == (1, 4096, 3072), f'z shape {z.shape}, expected (1, 4096, 3072)'
    assert z.dtype == torch.bfloat16, f'z dtype {z.dtype}, expected bfloat16'
    assert not torch.isnan(z).any().item(), 'z contains NaN'
    assert not torch.isinf(z).any().item(), 'z contains Inf'

    z_f = z.float()
    print(f'output {label:<8}          : shape={tuple(z.shape)} dtype={z.dtype} '
          f'min={z_f.min().item():.3f} max={z_f.max().item():.3f} '
          f'mean={z_f.mean().item():.4f} std={z_f.std().item():.4f}')
    return z


# Cond-only (guidance=1.0, single forward per step)
z_cond = _run_forward('a red cat sitting on a wooden chair', guidance_scale=1.0, label='cond')

# CFG (guidance=5.0 triggers uncond=' ' second forward per step)
z_cfg = _run_forward('a red cat sitting on a wooden chair', guidance_scale=5.0, label='cfg')

# Sanity: CFG output must differ from cond-only output (uncond branch actually ran).
# Not a numerical parity check, just a "the two paths are not the same" guard.
z_diff = (z_cond.float() - z_cfg.float()).abs().max().item()
assert z_diff > 1e-3, f'CFG output identical to cond-only (max_abs_diff={z_diff}); uncond branch likely no-op'
print(f'cond vs cfg diff         : max_abs_diff={z_diff:.4f} (must be > 1e-3, confirms CFG branch ran)')

print('pass')


# output (H100, HiDream-O1-Image 8.8B, torch.bfloat16):
# config                   : model_dir=/workspace/vllm-omni/.hf_models_cache/HiDream-O1-Image
# request cond             : prompt='a red cat sitting on a wooden chair' h=2048 w=2048 steps=4 seed=42 guidance_scale=1.0
# output cond              : shape=(1, 4096, 3072) dtype=torch.bfloat16 min=-1.008 max=0.988 mean=-0.2483 std=0.3932
# request cfg              : prompt='a red cat sitting on a wooden chair' h=2048 w=2048 steps=4 seed=42 guidance_scale=5.0
# output cfg               : shape=(1, 4096, 3072) dtype=torch.bfloat16 min=-2.234 max=2.891 mean=-0.5029 std=0.5392
# cond vs cfg diff         : max_abs_diff=3.1406 (must be > 1e-3, confirms CFG branch ran)
# pass
