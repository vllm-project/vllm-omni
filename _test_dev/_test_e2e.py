# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vllm_omni.diffusion.data import resolve_model_class_name
from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    find_closest_resolution,
)
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

REF_IMAGE = os.environ.get('HIDREAM_O1_REF_IMAGE')
OUTPUT_PATH = Path(os.environ.get('HIDREAM_O1_E2E_OUTPUT', 'hidream_o1_e2e.png'))
PROMPT = 'a red cat sitting on a wooden chair'
REQUEST_HEIGHT = 1920
REQUEST_WIDTH = 1080
NUM_STEPS = 4
GUIDANCE_SCALE = 5.0
SEED = 42

def _resolve_model_dir() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        os.environ.get('HIDREAM_O1_MODEL_DIR'),
        '/workspace/.hf_models_cache/HiDream-O1-Image',
        str(repo_root / '.hf_models_cache/HiDream-O1-Image'),
        '/workspace/vllm-omni/.hf_models_cache/HiDream-O1-Image',
    ]
    checked = []
    for raw in candidates:
        if raw is None or raw in checked:
            continue
        checked.append(raw)
        if Path(raw).exists():
            return raw

    raise AssertionError(
        f'model_dir not found: {checked}. '
        'set HIDREAM_O1_MODEL_DIR or download to '
        '/workspace/vllm-omni/.hf_models_cache/HiDream-O1-Image'
    )

def _calc_psnr(img: Image.Image, ref_img: Image.Image) -> float:
    arr = np.asarray(img.convert('RGB'), dtype=np.float32)
    ref_arr = np.asarray(ref_img.convert('RGB'), dtype=np.float32)
    assert arr.shape == ref_arr.shape, f'psnr shape mismatch: arr={arr.shape} ref={ref_arr.shape}'
    mse = float(np.mean((arr - ref_arr) ** 2))
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)

def main() -> None:
    model_dir = _resolve_model_dir()
    expected_size = find_closest_resolution(REQUEST_WIDTH, REQUEST_HEIGHT)
    resolved_model_class = resolve_model_class_name(model_dir)
    assert resolved_model_class == 'Qwen3VLForConditionalGeneration', (
        f'unexpected resolved model class: {resolved_model_class!r}'
    )
    pipeline_cls = DiffusionModelRegistry._try_load_model_cls(
        resolved_model_class
    )
    assert pipeline_cls is not None, 'diffusion registry could not load the resolved HiDream-O1 pipeline'

    print(
        f'model={model_dir!r} dtype=bf16 '
        f'resolved={resolved_model_class!r} pipeline={pipeline_cls.__name__!r}'
    )
    print(
        f'prompt={PROMPT!r} h={REQUEST_HEIGHT} w={REQUEST_WIDTH} '
        f'steps={NUM_STEPS} seed={SEED} guidance_scale={GUIDANCE_SCALE}'
    )

    omni = Omni(model=model_dir, dtype=torch.bfloat16)
    try:
        outputs = omni.generate(
            prompts=[{'prompt': PROMPT}],
            sampling_params_list=[
                OmniDiffusionSamplingParams(
                    height=REQUEST_HEIGHT,
                    width=REQUEST_WIDTH,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    seed=SEED,
                )
            ],
        )
        result = OmniRequestOutput.unwrap_result(outputs)
    finally:
        omni.close()

    assert result.final_output_type == 'image', (
        f"expected final_output_type='image', got {result.final_output_type!r}"
    )
    assert result.finished is True, 'expected finished=True'
    assert result.num_images == 1, f'expected num_images=1, got {result.num_images}'

    img = result.images[0]
    assert isinstance(img, Image.Image), f'expected PIL.Image.Image, got {type(img).__name__}'
    assert img.mode == 'RGB', f'image mode {img.mode!r}, expected RGB'
    assert img.size == expected_size, (
        f'image size {img.size}, expected {expected_size}'
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT_PATH)

    print(
        f'output={str(OUTPUT_PATH)!r} type={type(img).__name__} '
        f'mode={img.mode} size={img.size}'
    )

    if REF_IMAGE is None:
        print('psnr=skipped')
    else:
        ref_path = Path(REF_IMAGE)
        assert ref_path.exists(), f'reference image not found: {ref_path!r}'
        with Image.open(ref_path) as ref_file:
            ref_img = ref_file.convert('RGB')
        psnr = _calc_psnr(img, ref_img)
        assert psnr >= 40.0, f'expected PSNR >= 40 dB, got {psnr:.3f} dB'
        print(f'psnr={psnr:.3f} ref={str(ref_path)!r}')

    print('pass')

if __name__ == '__main__':
    main()


# output (H100, HiDream-O1-Image 8.8B, torch.bfloat16; HIDREAM_O1_REF_IMAGE unset):
# model='/workspace/vllm-omni/.hf_models_cache/HiDream-O1-Image' dtype=bf16 resolved='Qwen3VLForConditionalGeneration' pipeline='HiDreamO1ImagePipeline'
# prompt='a red cat sitting on a wooden chair' h=1920 w=1080 steps=4 seed=42 guidance_scale=5.0
# output='hidream_o1_e2e.png' type=Image mode=RGB size=(1440, 2560)
# psnr=skipped
# pass
