# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in, full-request A/B for the Ascend Qwen-Image-2.1 adapter."""

import functools
import json
import math
import os
import statistics
import time
from contextlib import contextmanager

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.npu, pytest.mark.diffusion]


def _assert_route_hits(hits: dict[str, int], fused: bool) -> None:
    if fused:
        assert hits["prefill"] > 0, "MindIE-SD prefill did not execute"
        assert hits["decode"] > 0, "MindIE-SD decode did not execute"
        assert hits["concat"] > 0, "Head-major prefix concatenation did not execute"
    else:
        assert not any(hits.values()), f"Disabled arm used optimized routes: {hits}"


@contextmanager
def _observe_routes(monkeypatch, mindiesd):
    from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as module

    hits = dict(prefill=0, decode=0, concat=0)
    phase = None
    forward = module.QwenImage21Attention.forward
    norm_rope = mindiesd.norm_rope_concat
    concat = module.concat_prefix_bnsd

    @functools.wraps(forward)
    def observe_attention(self, *args, **kwargs):
        nonlocal phase
        previous = phase
        phase = "prefill" if kwargs.get("cache_write_len") is not None else "decode"
        try:
            return forward(self, *args, **kwargs)
        finally:
            phase = previous

    @functools.wraps(norm_rope)
    def observe_norm_rope(*args, **kwargs):
        result = norm_rope(*args, **kwargs)
        assert phase in ("prefill", "decode"), "Operator executed outside model attention"
        hits[phase] += 1
        return result

    def observe_concat(prefix, current):
        result = concat(prefix, current)
        assert phase == "decode"
        assert result.transpose(1, 2).is_contiguous(), "Concat lost head-major storage"
        hits["concat"] += 1
        return result

    with monkeypatch.context() as patch:
        patch.setattr(module.QwenImage21Attention, "forward", observe_attention)
        patch.setattr(mindiesd, "norm_rope_concat", observe_norm_rope)
        patch.setattr(module, "concat_prefix_bnsd", observe_concat)
        yield hits


@pytest.mark.skipif(
    not os.environ.get("QWEN_IMAGE_21_TEST_MODEL"),
    reason="Set QWEN_IMAGE_21_TEST_MODEL to a provisioned checkpoint for full-request validation.",
)
def test_mindiesd_qkv_full_request_ab(monkeypatch, tmp_path):
    pytest.importorskip("torch_npu")
    mindiesd = pytest.importorskip("mindiesd")
    from PIL import Image

    from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms.npu import is_a5

    if not is_a5() or not callable(getattr(mindiesd, "norm_rope_concat", None)):
        pytest.skip("Requires A5 and the MindIE-SD norm_rope_concat Python API")

    pipelines = []
    init = QwenImage21Pipeline.__init__

    @functools.wraps(init)
    def capture_pipeline(self, *args, **kwargs):
        init(self, *args, **kwargs)
        pipelines.append(self)

    monkeypatch.setattr(QwenImage21Pipeline, "__init__", capture_pipeline)
    model = Omni(
        model=os.environ["QWEN_IMAGE_21_TEST_MODEL"],
        enforce_eager=True,
        dtype=torch.bfloat16,
        enable_cpu_offload=False,
        log_stats=False,
        additional_config={"qwen21_mindiesd_qkv_fusion": True, "qwen21_head_major_kv_concat": True},
    )
    rows = []
    try:
        assert len(pipelines) == 1
        transformer = pipelines[0].transformer
        assert transformer.mindiesd_qkv_fusion, "Requested fusion was rejected by its eligibility checks"
        prompts: list[tuple[str, dict[str, object]]] = [
            ("t2i", {"prompt": "A ceramic teapot on a wooden table", "modalities": ["image"]})
        ]
        input_image = os.environ.get("QWEN_IMAGE_21_TEST_IMAGE")
        if input_image:
            with Image.open(input_image) as image:
                prompts.append(
                    (
                        "i2i",
                        {
                            "prompt": "Turn this image into a warm oil painting, preserving its subject and pose.",
                            "modalities": ["image"],
                            "multi_modal_data": {"image": image.convert("RGB")},
                        },
                    )
                )
        for scenario, prompt in prompts:
            outputs = {}
            for index in range(5):
                for fused in [False, True] if index % 2 == 0 else [True, False]:
                    transformer.mindiesd_qkv_fusion = fused
                    for block in transformer.transformer_blocks:
                        block.attn.head_major_kv_concat = fused
                    params = OmniDiffusionSamplingParams(
                        height=1024,
                        width=1024,
                        seed=42,
                        true_cfg_scale=1.0,
                        num_inference_steps=50,
                        num_outputs_per_prompt=1,
                    )
                    if index < 2:
                        # Observe complete warmups only; measured requests have no spies.
                        with _observe_routes(monkeypatch, mindiesd) as hits:
                            output = model.generate(prompt, sampling_params_list=[params], use_tqdm=False)
                            torch.npu.synchronize()
                            _assert_route_hits(hits, fused)
                            print("QWEN21_ROUTES " + json.dumps({"scenario": scenario, "fused": fused, **hits}))
                        elapsed = None
                    else:
                        torch.npu.synchronize()
                        torch.npu.reset_peak_memory_stats()
                        started = time.perf_counter()
                        output = model.generate(prompt, sampling_params_list=[params], use_tqdm=False)
                        torch.npu.synchronize()
                        elapsed = time.perf_counter() - started
                    images = extract_images_from_outputs(output)
                    assert len(images) == 1 and images[0].size == (1024, 1024)
                    images[0].save(tmp_path / f"{scenario}-{fused}-{index}.png")
                    outputs[fused] = torch.tensor(list(images[0].convert("RGB").tobytes()), dtype=torch.float32)
                    row = {
                        "scenario": scenario,
                        "fused": fused,
                        "warmup": index < 2,
                        "index": index,
                        "wall_seconds": elapsed,
                        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                        "peak_reserved_bytes": torch.npu.max_memory_reserved(),
                    }
                    rows.append(row)
                    print("QWEN21_AB " + json.dumps(row), flush=True)
            mse = (outputs[False] - outputs[True]).square().mean().item()
            psnr = math.inf if mse == 0 else 10 * math.log10(255**2 / mse)
            assert psnr >= 30, f"{scenario}: RGB PSNR {psnr:.3f} dB is below the single-input screen"
            medians = {
                str(fused): statistics.median(
                    row["wall_seconds"]
                    for row in rows
                    if row["scenario"] == scenario and row["fused"] == fused and not row["warmup"]
                )
                for fused in (False, True)
            }
            print("QWEN21_SUMMARY " + json.dumps({"scenario": scenario, "median_seconds": medians, "psnr_db": psnr}))
    finally:
        (tmp_path / "requests.json").write_text(json.dumps(rows, indent=2))
        model.close()
