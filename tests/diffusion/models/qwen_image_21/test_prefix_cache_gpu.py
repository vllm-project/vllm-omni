# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise real Worker COW and partial-prefix parity on eight CUDA GPUs."""

import gc
import json
import math
import os

import numpy as np
import pytest
from PIL import Image

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.slow]


@hardware_test(res={"cuda": "H100"}, num_cards=8)
def test_partial_prefix_hits_match_cache_off(monkeypatch, tmp_path):
    import torch
    import vllm

    from vllm_omni.diffusion.diffusion_kv.manager import DiffusionKVCacheManager
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    model = os.environ.get("QWEN_IMAGE21_TEST_MODEL", "Qwen/Qwen-Image-2.1")
    images = [
        Image.fromarray(np.random.default_rng(seed).integers(0, 256, (128, 128, 3), dtype=np.uint8))
        for seed in (20, 21, 22, 23)
    ]
    prompt = "Combine the reference images into a watercolor painting."
    cases = [
        ("cold", images[:3], prompt),
        ("changed_text", images[:3], "Combine the reference images into a charcoal drawing."),
        ("changed_later_image", [*images[:2], images[3]], prompt),
        ("return_to_original", images[:3], prompt),
    ]
    config = dict(
        model=model,
        enforce_eager=True,
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=2,
        max_num_batched_tokens=65536,
        max_model_len=32768,
        additional_config={"diffusion_kv_profile_max_sequence_length": 512},
        kv_cache_memory_bytes=10 * 1024**3,
        num_gpus=8,
        use_hsdp=True,
        hsdp_shard_size=8,
        hsdp_replicate_size=1,
        ulysses_degree=8,
        mask_sp_padding=True,
    )
    allocations = []
    original = DiffusionKVCacheManager.reserve_request

    def reserve(manager, request_id, requests):
        metadata = original(manager, request_id, requests)
        if metadata is not None:
            allocations.append(
                {
                    "hits": [row.cached_prefix_len for row in metadata.sequences],
                    "copies": [list(pair) for pair in metadata.block_copies],
                }
            )
        return metadata

    monkeypatch.setattr(DiffusionKVCacheManager, "reserve_request", reserve)
    expected = {}
    metrics = []
    for caching in (False, True):
        omni = Omni(enable_prefix_caching=caching, **config)
        try:
            for name, refs, text in cases:
                allocations.clear()
                params = OmniDiffusionSamplingParams(
                    height=256,
                    width=256,
                    num_inference_steps=2,
                    true_cfg_scale=4.0,
                    seed=42,
                    max_sequence_length=512,
                )
                output = list(
                    omni.generate(
                        {
                            "prompt": text,
                            "negative_prompt": "blurry, low quality",
                            "multi_modal_data": {"image": [image.copy() for image in refs]},
                        },
                        params,
                        py_generator=True,
                    )
                )
                array = np.asarray(output[0].images[0]).copy()
                if not caching:
                    expected[name] = array
                    continue
                assert allocations
                allocation = allocations[-1]
                if name != "cold":
                    assert min(allocation["hits"]) > 0
                if name == "changed_later_image":
                    assert allocation["copies"], "Partial image hits must exercise Worker tail-page COW"
                assert array.shape == expected[name].shape
                delta = array[..., :3].astype(np.float64) - expected[name][..., :3].astype(np.float64)
                mse = float(np.mean(delta**2))
                psnr = None if mse == 0 else 10 * math.log10(255**2 / mse)
                alpha_mse: float | None = None
                alpha_psnr: float | None = None
                alpha_max_abs: float | None = None
                if array.shape[-1] == 4:
                    alpha_delta = array[..., 3].astype(np.float64) - expected[name][..., 3].astype(np.float64)
                    alpha_mse = float(np.mean(alpha_delta**2))
                    alpha_psnr = None if alpha_mse == 0 else 10 * math.log10(255**2 / alpha_mse)
                    alpha_max_abs = float(np.abs(alpha_delta).max())
                metrics.append(
                    {
                        "case": name,
                        "bitwise": bool(np.array_equal(array, expected[name])),
                        "max_abs": float(np.abs(delta).max()),
                        "psnr_db": psnr,
                        "alpha_max_abs": alpha_max_abs,
                        "alpha_psnr_db": alpha_psnr,
                        **allocation,
                    }
                )
                # Bitwise equality is recorded where observed; other CUDA
                # kernels must still meet this explicit image-level tolerance.
                assert mse == 0 or (psnr is not None and psnr >= 40), metrics[-1]
                assert alpha_mse is None or alpha_mse == 0 or (alpha_psnr is not None and alpha_psnr >= 40), metrics[-1]
        finally:
            omni.close()
            del omni
            gc.collect()
    (tmp_path / "prefix_cache_parity.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "vllm": vllm.__version__,
                "gpu": torch.cuda.get_device_name(),
                "config": config,
                "cases": metrics,
            },
            indent=2,
        )
        + "\n"
    )
