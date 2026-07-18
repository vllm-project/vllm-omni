# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L4 real-checkpoint end-to-end HiDreamO1ImagePipeline.forward() test.

Usage:
    python _test_dev/_test_hidream_o1_forward_from_ckpt.py

Optional:
    export HIDREAM_O1_MODEL_DIR=/path/to/HiDream-O1-Image
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    HiDreamO1ImagePipeline,
    get_hidream_o1_image_post_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def _resolve_model_dir() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    env_model_dir = os.environ.get("HIDREAM_O1_MODEL_DIR")

    candidates: list[Path] = []
    if env_model_dir:
        candidates.append(Path(env_model_dir))

    candidates.extend(
        [
            Path("/workspace/.hf_models_cache/HiDream-O1-Image"),
            repo_root / ".hf_models_cache/HiDream-O1-Image",
            Path("/workspace/vllm-omni/.hf_models_cache/HiDream-O1-Image"),
        ]
    )

    seen: set[str] = set()
    deduped_candidates: list[Path] = []
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        deduped_candidates.append(candidate)

    for candidate in deduped_candidates:
        if candidate.exists():
            return str(candidate)

    checked = ", ".join(repr(str(candidate)) for candidate in deduped_candidates)
    raise AssertionError(
        "model_dir not found. Checked: "
        f"{checked}. "
        "Set HIDREAM_O1_MODEL_DIR explicitly, or download the checkpoint to "
        "/workspace/vllm-omni/.hf_models_cache/HiDream-O1-Image."
    )


MODEL_DIR = _resolve_model_dir()

od_config = OmniDiffusionConfig(
    model=MODEL_DIR,
    dtype=torch.bfloat16,
    enable_diffusion_pipeline_profiler=False,
)

print(f'config                   : model_dir={MODEL_DIR}')

pipe = HiDreamO1ImagePipeline(od_config=od_config)
postprocess = get_hidream_o1_image_post_process_func(od_config)

def _run_forward(prompt, guidance_scale, label):
    req = OmniDiffusionRequest(
        prompts=[prompt],
        sampling_params=OmniDiffusionSamplingParams(
            height=2048,
            width=2048,
            num_inference_steps=4,
            seed=42,
            guidance_scale=guidance_scale,
        ),
        request_id=f'ckpt-{label}',
    )

    print(
        f'request {label:<8} : '
        f'prompt={prompt!r} '
        f'h=2048 w=2048 steps=4 seed=42 '
        f'guidance_scale={guidance_scale}'
    )

    out = pipe.forward(req)

    # ---- raw forward envelope checks ----
    assert isinstance(out.output, tuple) and len(out.output) == 3, (
        'envelope must be a 3-tuple (z, h, w), '
        f'got {type(out.output).__name__}'
    )

    z, snapped_h, snapped_w = out.output

    assert (snapped_h, snapped_w) == (2048, 2048), (
        f'envelope hw: ({snapped_h}, {snapped_w})'
    )
    assert isinstance(z, torch.Tensor), (
        f'z must be torch.Tensor, got {type(z).__name__}'
    )
    assert z.shape == (1, 4096, 3072), (
        f'z shape {z.shape}, expected (1, 4096, 3072)'
    )
    assert z.dtype == torch.bfloat16, (
        f'z dtype {z.dtype}, expected bfloat16'
    )
    assert not torch.isnan(z).any().item(), 'z contains NaN'
    assert not torch.isinf(z).any().item(), 'z contains Inf'

    z_f = z.float()

    print(
        f'output {label:<8} : '
        f'shape={tuple(z.shape)} '
        f'dtype={z.dtype} '
        f'min={z_f.min().item():.3f} '
        f'max={z_f.max().item():.3f} '
        f'mean={z_f.mean().item():.4f} '
        f'std={z_f.std().item():.4f}'
    )

    img = postprocess(out.output)

    assert isinstance(img, Image.Image), (
        f'postprocess must return PIL.Image.Image, '
        f'got {type(img).__name__}'
    )
    assert img.mode == 'RGB', (
        f'image mode {img.mode!r}, expected RGB'
    )
    assert img.size == (2048, 2048), (
        f'image size {img.size}, expected (2048, 2048)'
    )

    output_path = f'hidream_o1_{label}.png'
    img.save(output_path)

    print(
        f'image {label:<9}: '
        f'type={type(img).__name__} '
        f'mode={img.mode} '
        f'size={img.size} '
        f'saved={output_path!r}'
    )

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
# image cond               : type=Image mode=RGB size=(2048, 2048) saved='hidream_o1_cond.png'
# request cfg              : prompt='a red cat sitting on a wooden chair' h=2048 w=2048 steps=4 seed=42 guidance_scale=5.0
# output cfg               : shape=(1, 4096, 3072) dtype=torch.bfloat16 min=-2.234 max=2.891 mean=-0.5029 std=0.5392
# image cfg                : type=Image mode=RGB size=(2048, 2048) saved='hidream_o1_cfg.png'
# cond vs cfg diff         : max_abs_diff=3.1406 (must be > 1e-3, confirms CFG branch ran)
# pass
