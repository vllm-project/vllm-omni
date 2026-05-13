# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import gc
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from tests.e2e.accuracy.helpers import CLIPScorer, assert_similarity, model_output_dir
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import build_prompt_tokens

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct"
SEED = 42
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 0
HEIGHT, WIDTH = 1024, 1024
PSNR_THRESHOLD = 0.0
SSIM_THRESHOLD = 0.0
CLIP_SCORE_THRESHOLD = 20.0
QUANT_CLIP_SCORE_DROP_THRESHOLD = float(os.environ.get("HUNYUAN_IMAGE3_QUANT_CLIP_SCORE_DROP_THRESHOLD", "5.0"))
PROMPT = "A brown and white dog is running on the grass."
QUANT_BF16_ENV = "HUNYUAN_IMAGE3_BF16_MODEL"
QUANT_FP8_ENV = "HUNYUAN_IMAGE3_FP8_MODEL"
QUANT_NVFP4_ENV = "HUNYUAN_IMAGE3_NVFP4_MODEL"

# fmt: off
_BASE_CONFIG = {
    "stage_args": [
        {
            "stage_id": 0, "stage_type": "llm",
            "runtime": {"process": True, "devices": "0,1", "max_batch_size": 1, "requires_multimodal_data": True},
            "engine_args": {
                "model_stage": "AR", "model_arch": "HunyuanImage3ForCausalMM",
                "worker_cls": "vllm_omni.worker.gpu_ar_worker.GPUARWorker",
                "scheduler_cls": "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler",
                "gpu_memory_utilization": 0.95, "enforce_eager": True, "trust_remote_code": True,
                "engine_output_type": "latent", "enable_prefix_caching": False,
                "max_num_batched_tokens": 32768, "tensor_parallel_size": 2, "pipeline_parallel_size": 1,
                "hf_overrides": {"rope_parameters": {"mrope_section": [0, 32, 32], "rope_type": "default"}},
            },
            "is_comprehension": False, "final_output": True, "final_output_type": "text",
            "default_sampling_params": {
                "temperature": 0.0, "top_p": 1, "top_k": -1, "max_tokens": 8192,
                "stop_token_ids": [128025], "detokenize": True, "skip_special_tokens": False,
            },
            "output_connectors": {"to_stage_1": "rdma_connector"},
        },
        {
            "stage_id": 1, "stage_type": "diffusion",
            "runtime": {"process": True, "devices": "2,3", "max_batch_size": 1, "requires_multimodal_data": True},
            "engine_args": {
                "model_stage": "dit", "model_arch": "HunyuanImage3ForCausalMM",
                "enforce_eager": True, "trust_remote_code": True, "distributed_executor_backend": "mp",
                "parallel_config": {"tensor_parallel_size": 2, "enable_expert_parallel": True},
            },
            "engine_input_source": [0],
            "custom_process_input_func": "vllm_omni.model_executor.stage_input_processors.hunyuan_image3.ar2diffusion",
            "final_output": True, "final_output_type": "image",
            "default_sampling_params": {"num_inference_steps": NUM_INFERENCE_STEPS, "guidance_scale": GUIDANCE_SCALE},
            "input_connectors": {"from_stage_0": "rdma_connector"},
        },
    ],
    "runtime": {
        "enabled": True,
        "connectors": {"rdma_connector": {
            "name": "MooncakeTransferEngineConnector",
            "extra": {"host": "auto", "zmq_port": 50051, "protocol": "rdma", "device_name": "", "memory_pool_size": 4294967296, "memory_pool_device": "cpu"},
        }},
        "edges": [{"from": 0, "to": 1}],
    },
}
# fmt: on

_QUANT_DIT_CONFIG = {
    "stage_args": [
        {
            "stage_id": 0,
            "stage_type": "diffusion",
            "runtime": {
                "devices": "0,1",
                "max_batch_size": 1,
            },
            "engine_args": {
                "model_stage": "dit",
                "enforce_eager": True,
                "trust_remote_code": True,
                "distributed_executor_backend": "mp",
                "force_cutlass_fp8": True,
                "moe_backend": "cutlass",
                "parallel_config": {
                    "tensor_parallel_size": 2,
                    "enable_expert_parallel": True,
                },
                "omni_kv_config": {"need_recv_cache": True},
            },
            "final_output": True,
            "final_output_type": "image",
            "is_comprehension": False,
            "default_sampling_params": {"seed": SEED},
        }
    ],
    "runtime": {"enabled": True},
}


@dataclass(frozen=True)
class _QuantAccuracyCase:
    name: str
    model_env: str
    nvfp4_backend: str | None = None


def _quant_accuracy_cases() -> list[pytest.ParameterSet]:
    cases = [
        _QuantAccuracyCase(name="fp8", model_env=QUANT_FP8_ENV),
        _QuantAccuracyCase(name="mixed_nvfp4", model_env=QUANT_NVFP4_ENV, nvfp4_backend="cutlass"),
    ]
    params: list[pytest.ParameterSet] = []
    for case in cases:
        marks = []
        if not os.environ.get(case.model_env):
            marks.append(pytest.mark.skip(reason=f"Set {case.model_env} to run this quant accuracy case."))
        params.append(pytest.param(case, id=case.name, marks=marks))
    return params


def _make_config(enable_kv_reuse: bool, path: Path) -> None:
    config = copy.deepcopy(_BASE_CONFIG)
    config["stage_args"][0]["engine_args"]["omni_kv_config"] = {"need_send_cache": enable_kv_reuse}
    config["stage_args"][1]["engine_args"]["omni_kv_config"] = {"need_recv_cache": enable_kv_reuse}
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _quant_devices() -> str:
    return os.environ.get("HUNYUAN_IMAGE3_QUANT_DEVICES", "0,1")


def _quant_tensor_parallel_size() -> int:
    return int(os.environ.get("HUNYUAN_IMAGE3_QUANT_TP", str(len(_quant_devices().split(",")))))


def _make_quant_dit_config(path: Path) -> None:
    config = copy.deepcopy(_QUANT_DIT_CONFIG)
    config["stage_args"][0]["runtime"]["devices"] = _quant_devices()
    config["stage_args"][0]["engine_args"]["parallel_config"]["tensor_parallel_size"] = _quant_tensor_parallel_size()
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _run(stage_config_path: str, output_path: Path) -> tuple[Image.Image, str, float]:
    from transformers import AutoTokenizer

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
    from vllm_omni.platforms import current_omni_platform

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    result = build_prompt_tokens(PROMPT, tokenizer, task="it2i_recaption", sys_type="en_unified")
    token_ids = result.token_ids
    system_prompt_type = result.system_prompt_type

    with OmniRunner(MODEL_NAME, stage_configs_path=stage_config_path) as runner:
        params_list = list(runner.omni.default_sampling_params_list)
        for sp in params_list:
            if isinstance(sp, OmniDiffusionSamplingParams):
                sp.num_inference_steps = NUM_INFERENCE_STEPS
                sp.guidance_scale = GUIDANCE_SCALE
                sp.seed = SEED
                sp.generator = torch.Generator(device=current_omni_platform.device_type or "cuda").manual_seed(SEED)

        prompts: list[OmniPromptType] = [
            {
                "prompt_token_ids": token_ids,
                "prompt": PROMPT,
                "use_system_prompt": system_prompt_type,
                "height": HEIGHT,
                "width": WIDTH,
            }
        ]
        t0 = time.perf_counter()
        outputs = list(runner.omni.generate(prompts=prompts, sampling_params_list=params_list))
        elapsed = time.perf_counter() - t0

    assert outputs, "Pipeline produced no outputs"
    images = None
    cot_text = ""
    for out in outputs:
        ro = getattr(out, "request_output", None)
        if ro and getattr(ro, "outputs", None):
            txt = "".join(getattr(o, "text", "") or "" for o in ro.outputs)
            if txt:
                cot_text = txt

        imgs = getattr(out, "images", None)
        if not imgs and ro and hasattr(ro, "images"):
            imgs = ro.images
        if imgs:
            images = imgs

    assert images, "Pipeline output had no images"

    image = images[0].convert("RGB")
    image.save(output_path)
    gc.collect()
    if torch.accelerator.is_available():
        torch.accelerator.empty_cache()
    return image, cot_text, elapsed


def _extract_image(outputs) -> Image.Image:
    assert outputs, "Pipeline produced no outputs"
    for output in outputs:
        images = getattr(output, "images", None)
        request_output = getattr(output, "request_output", None)
        if not images and request_output is not None:
            images = getattr(request_output, "images", None)
        if images:
            image = images[0].convert("RGB")
            image.load()
            return image
    raise AssertionError("Pipeline output had no images")


def _run_dit_model(
    model: str,
    stage_config_path: str,
    output_path: Path,
    *,
    nvfp4_backend: str | None = None,
) -> tuple[Image.Image, float]:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    old_backend = os.environ.get("VLLM_NVFP4_GEMM_BACKEND")
    if nvfp4_backend is not None:
        os.environ["VLLM_NVFP4_GEMM_BACKEND"] = nvfp4_backend

    try:
        with OmniRunner(model, stage_configs_path=stage_config_path, mode="text-to-image") as runner:
            generator = torch.Generator(device=current_omni_platform.device_type or "cuda").manual_seed(SEED)
            params = OmniDiffusionSamplingParams(
                height=HEIGHT,
                width=WIDTH,
                seed=SEED,
                generator=generator,
                num_inference_steps=20,
                guidance_scale=4.0,
                guidance_scale_provided=True,
            )
            t0 = time.perf_counter()
            outputs = runner.omni.generate({"prompt": PROMPT}, params)
            elapsed = time.perf_counter() - t0
            image = _extract_image(outputs)
            image.save(output_path)
            return image, elapsed
    finally:
        if nvfp4_backend is not None:
            if old_backend is None:
                os.environ.pop("VLLM_NVFP4_GEMM_BACKEND", None)
            else:
                os.environ["VLLM_NVFP4_GEMM_BACKEND"] = old_backend
        gc.collect()
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()


@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="Needs 4+ GPUs (2 AR + 2 DiT)")
def test_text_to_image_alignment(accuracy_artifact_root: Path) -> None:
    """KV reuse ON vs OFF: same pipeline, same seed → PSNR >= 40 dB."""
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME + "-kv-reuse")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _make_config(False, tmp / "off.yaml")
        image_no_reuse_1, cot_no_reuse_1, time_no_reuse_1 = _run(
            str(tmp / "off.yaml"), output_dir / "without_kv_reuse_1.png"
        )
        image_no_reuse_2, cot_no_reuse_2, time_no_reuse_2 = _run(
            str(tmp / "off.yaml"), output_dir / "without_kv_reuse_2.png"
        )
        _make_config(True, tmp / "on.yaml")
        image_reuse, cot_reuse, time_reuse = _run(str(tmp / "on.yaml"), output_dir / "with_kv_reuse.png")

    print("\n--- End-to-end time ---")
    print(f"  WITHOUT KV reuse 1: {time_no_reuse_1:.2f}s")
    print(f"  WITHOUT KV reuse 2: {time_no_reuse_2:.2f}s")
    print(f"  WITH KV reuse:    {time_reuse:.2f}s")

    (output_dir / "cot_with_kv_reuse.txt").write_text(cot_reuse, encoding="utf-8")
    (output_dir / "cot_without_kv_reuse_1.txt").write_text(cot_no_reuse_1, encoding="utf-8")
    (output_dir / "cot_without_kv_reuse_2.txt").write_text(cot_no_reuse_2, encoding="utf-8")

    print(f"\n--- CoT WITH KV reuse (len={len(cot_reuse)}) ---\n{cot_reuse}")
    print(f"\n--- CoT WITHOUT KV reuse 1 (len={len(cot_no_reuse_1)}) ---\n{cot_no_reuse_1}")
    print(f"\n--- CoT WITHOUT KV reuse 2 (len={len(cot_no_reuse_2)}) ---\n{cot_no_reuse_2}")

    clip_scorer = CLIPScorer()
    clip_scorer.assert_score(
        model_name=f"{MODEL_NAME} with-reuse",
        image=image_reuse,
        text=PROMPT,
        threshold=CLIP_SCORE_THRESHOLD,
    )
    clip_scorer.assert_score(
        model_name=f"{MODEL_NAME} without-reuse 1",
        image=image_no_reuse_1,
        text=PROMPT,
        threshold=CLIP_SCORE_THRESHOLD,
    )
    clip_scorer.assert_score(
        model_name=f"{MODEL_NAME} without-reuse 2",
        image=image_no_reuse_2,
        text=PROMPT,
        threshold=CLIP_SCORE_THRESHOLD,
    )

    assert_similarity(
        model_name=f"{MODEL_NAME} non-reuse run1 vs run2",
        vllm_image=image_no_reuse_1,
        diffusers_image=image_no_reuse_2,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
        width=WIDTH,
        height=HEIGHT,
    )

    assert_similarity(
        model_name=f"{MODEL_NAME} KV-reuse vs no-reuse",
        vllm_image=image_reuse,
        diffusers_image=image_no_reuse_1,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
        width=WIDTH,
        height=HEIGHT,
    )


@pytest.mark.parametrize("case", _quant_accuracy_cases())
@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="Needs 2+ GPUs for HunyuanImage3 DiT")
def test_quantized_dit_matches_bf16_accuracy(
    case: _QuantAccuracyCase,
    accuracy_artifact_root: Path,
) -> None:
    """Quantized DiT checkpoints should preserve prompt-aligned image quality."""
    output_dir = model_output_dir(accuracy_artifact_root, MODEL_NAME + "-quant")
    bf16_model = os.environ.get(QUANT_BF16_ENV, MODEL_NAME)
    quant_model = os.environ[case.model_env]

    with tempfile.TemporaryDirectory() as tmpdir:
        stage_config_path = Path(tmpdir) / "hunyuan_image3_quant_dit.yaml"
        _make_quant_dit_config(stage_config_path)

        bf16_image, bf16_time = _run_dit_model(
            bf16_model,
            str(stage_config_path),
            output_dir / "bf16.png",
        )
        quant_image, quant_time = _run_dit_model(
            quant_model,
            str(stage_config_path),
            output_dir / f"{case.name}.png",
            nvfp4_backend=case.nvfp4_backend,
        )

    assert_similarity(
        model_name=f"{MODEL_NAME} {case.name} vs bf16",
        vllm_image=quant_image,
        diffusers_image=bf16_image,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
        width=WIDTH,
        height=HEIGHT,
    )

    clip_scorer = CLIPScorer()
    bf16_clip_score = clip_scorer.score(bf16_image, PROMPT)
    quant_clip_score = clip_scorer.score(quant_image, PROMPT)
    clip_score_drop = bf16_clip_score - quant_clip_score

    metrics = {
        "case": case.name,
        "bf16_model": bf16_model,
        "quant_model": quant_model,
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "num_inference_steps": 20,
        "guidance_scale": 4.0,
        "bf16_elapsed_s": bf16_time,
        "quant_elapsed_s": quant_time,
        "bf16_clip_score": bf16_clip_score,
        "quant_clip_score": quant_clip_score,
        "clip_score_drop": clip_score_drop,
    }
    metrics_path = output_dir / f"{case.name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nHunyuanImage3 quant accuracy ({case.name})")
    print(f"  bf16 model:       {bf16_model}")
    print(f"  quant model:      {quant_model}")
    print(f"  BF16 CLIP score:  {bf16_clip_score:.4f}")
    print(f"  quant CLIP score: {quant_clip_score:.4f} threshold>={CLIP_SCORE_THRESHOLD:.4f}")
    print(f"  CLIP score drop:  {clip_score_drop:.4f} threshold<={QUANT_CLIP_SCORE_DROP_THRESHOLD:.4f}")
    print(f"  metrics:          {metrics_path}")

    assert quant_clip_score >= CLIP_SCORE_THRESHOLD, (
        f"{case.name} CLIP score below threshold: got {quant_clip_score:.4f}, expected >= {CLIP_SCORE_THRESHOLD:.4f}"
    )
    assert clip_score_drop <= QUANT_CLIP_SCORE_DROP_THRESHOLD, (
        f"{case.name} CLIP score drop too large: got {clip_score_drop:.4f}, "
        f"expected <= {QUANT_CLIP_SCORE_DROP_THRESHOLD:.4f}"
    )
