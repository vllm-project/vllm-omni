# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Expanded end-to-end tests for BAGEL in offline mode.

Coverage:
- LoRA scale / deactivation on Stage 1 DiT
- Sleep / wake gaps not covered by small-model entrypoint suites:
  BagelPipeline TP=2 VRAM reclaim, and (skipped) dual-engine coordination
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest
import pytest_asyncio
import torch
from PIL import Image
from safetensors.torch import save_file
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger("BagelSleepExpansion")

MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
BAGEL_STAGE_CONFIG = get_deploy_config_path("ci/bagel.yaml")
DEFAULT_PROMPT = "<|im_start|>A cute cat<|im_end|>"

# (model, deploy_config_path) for ``@pytest.mark.parametrize("omni_runner", ..., indirect=True)``
_OMNI_RUNNER_PARAM = (MODEL, BAGEL_STAGE_CONFIG)


# ---------------------------------------------------------------------------
# Helpers (reused from test_bagel_text2img.py patterns)
# ---------------------------------------------------------------------------


def _configure_sampling_params(omni: Omni, num_inference_steps: int = 10) -> list[OmniSamplingParams]:
    params_list = omni.default_sampling_params_list
    if len(params_list) > 1:
        params_list[1].num_inference_steps = num_inference_steps
        params_list[1].extra_args = {
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.5,
        }
    return params_list


def _extract_generated_image(omni_outputs: list[OmniRequestOutput]) -> Image.Image | None:
    for req_output in omni_outputs:
        if req_output.images:
            return req_output.images[0]
    return None


def _generate_bagel_image(omni: Omni) -> Image.Image:
    params_list = _configure_sampling_params(omni)
    params_list[1].lora_request = None
    outputs = list(
        omni.generate(
            prompts=[{"prompt": DEFAULT_PROMPT, "modalities": ["image"]}],
            sampling_params_list=params_list,
        )
    )
    img = _extract_generated_image(outputs)
    assert img is not None, "No image generated"
    return img


def _generate_bagel_image_with_lora(
    omni: Omni,
    lora_request: LoRARequest,
    lora_scale: float = 1.0,
) -> Image.Image:
    params_list = _configure_sampling_params(omni)
    params_list[1].lora_request = lora_request
    params_list[1].lora_scale = lora_scale
    outputs = list(
        omni.generate(
            prompts=[{"prompt": DEFAULT_PROMPT, "modalities": ["image"]}],
            sampling_params_list=params_list,
        )
    )
    img = _extract_generated_image(outputs)
    assert img is not None, "No image generated with LoRA"
    return img


# BAGEL uses GQA: hidden_size=3584, 28 Q heads, 4 KV heads, head_dim=128
# QKV packed dim = 28*128 + 4*128 + 4*128 = 3584 + 512 + 512 = 4608
_LORA_DIM = 3584
_LORA_QKV_DIM = 4608
_LORA_MODULE = "bagel.language_model.model.layers.0.self_attn.qkv_proj"
_LORA_RANK = 4


def _make_file_lora_request(adapter_dir: Path) -> LoRARequest:
    """Write synthetic adapter to disk and return a file-backed LoRARequest."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(42)
    lora_a = torch.randn((_LORA_RANK, _LORA_DIM), dtype=torch.float32, generator=gen) * 0.1
    lora_b = torch.randn((_LORA_QKV_DIM, _LORA_RANK), dtype=torch.float32, generator=gen) * 0.5
    save_file(
        {
            f"base_model.model.{_LORA_MODULE}.lora_A.weight": lora_a,
            f"base_model.model.{_LORA_MODULE}.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": _LORA_RANK, "lora_alpha": _LORA_RANK, "target_modules": [_LORA_MODULE]}),
        encoding="utf-8",
    )
    lora_dir = str(adapter_dir)
    return LoRARequest(lora_name="test_file", lora_int_id=stable_lora_int_id(lora_dir), lora_path=lora_dir)


# ---------------------------------------------------------------------------
# LoRA scale / deactivation
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_bagel_lora_scale_and_deactivation(omni_runner: OmniRunner, tmp_path) -> None:
    """Validate LoRA effect, bounded perturbation, and clean deactivation."""
    omni = omni_runner.omni
    lora_request = _make_file_lora_request(tmp_path / "bagel_lora")

    # 1) Baseline (no LoRA)
    baseline = _generate_bagel_image(omni)

    # 2) LoRA with scale=1.0
    img_1x = _generate_bagel_image_with_lora(omni, lora_request, lora_scale=1.0)

    # 3) LoRA with scale=2.0
    img_2x = _generate_bagel_image_with_lora(omni, lora_request, lora_scale=2.0)

    # 4) No LoRA again (deactivation)
    restored = _generate_bagel_image(omni)

    baseline_arr = np.array(baseline, dtype=np.int16)
    img_1x_arr = np.array(img_1x, dtype=np.int16)
    img_2x_arr = np.array(img_2x, dtype=np.int16)
    restored_arr = np.array(restored, dtype=np.int16)

    diff_1x = np.abs(baseline_arr - img_1x_arr).mean()
    diff_2x = np.abs(baseline_arr - img_2x_arr).mean()
    diff_restored = np.abs(baseline_arr - restored_arr).mean()

    # (a) Adapter has visible effect at both scales
    assert diff_1x > 0.5, f"LoRA scale=1.0 had no visible effect: diff={diff_1x}"
    assert diff_2x > 0.5, f"LoRA scale=2.0 had no visible effect: diff={diff_2x}"

    # (b) Different scales produce different outputs
    assert not np.isclose(diff_1x, diff_2x, atol=1.0), (
        f"LoRA scale has no effect: diff_1x={diff_1x:.2f}, diff_2x={diff_2x:.2f}"
    )

    # (c) Output is not corrupted (scale=2.0 can produce ~2x the diff of scale=1.0)
    assert diff_1x < 80, f"LoRA output looks corrupted: diff_1x={diff_1x}"
    assert diff_2x < 120, f"LoRA output looks corrupted: diff_2x={diff_2x}"

    # (d) Deactivation fully restores base model
    assert diff_restored == 0.0, f"Base model not restored after LoRA deactivation: diff={diff_restored}"


# ---------------------------------------------------------------------------
# Sleep / wake — BAGEL-only gaps not covered by small-model entrypoint suites:
# BagelPipeline TP=2 VRAM reclaim + (skipped) dual-engine coordination.
# ---------------------------------------------------------------------------


def _clean_device_envs() -> None:
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
        "ONEAPI_DEVICE_SELECTOR",
        "ASCEND_RT_VISIBLE_DEVICES",
    ):
        os.environ.pop(key, None)


def _get_device_global_memory_used_gib(device_id: int) -> float:
    """GPU-wide memory in use (GiB), includes all processes (driver view)."""
    try:
        with current_omni_platform.device(device_id):
            current_omni_platform.synchronize()
            free_b, total_b = current_omni_platform.get_device_memory()
        return (total_b - free_b) / 1024**3
    except Exception as e:
        logger.warning("get_device_global_memory_used_gib(%s): %s", device_id, e)
        return 0.0


def _get_ack_info(ack, key, default=None):
    if hasattr(ack, key):
        return getattr(ack, key)
    if isinstance(ack, dict):
        return ack.get(key, default)
    return default


async def _ensure_awake(engine: AsyncOmni, stage_ids: list[int]) -> None:
    try:
        await engine.wake_up(stage_ids=stage_ids)
    except Exception as e:
        logger.warning("ensure_awake failed (stage_ids=%s): %s", stage_ids, e)
    try:
        await engine.resume_generation(stage_ids=stage_ids)
    except Exception as e:
        logger.warning("ensure_resume failed (stage_ids=%s): %s", stage_ids, e)


@pytest_asyncio.fixture(scope="class", loop_scope="class")
async def bagel_diffusion_engine():
    """Shared BAGEL BagelPipeline TP=2 engine for sleep/wake + generate."""
    if current_omni_platform.is_rocm():
        _clean_device_envs()
    stages = [
        {
            "stage_id": 0,
            "stage_type": "diffusion",
            "runtime": {"process": True, "devices": "0,1", "max_batch_size": 1},
            "engine_args": {
                "model_stage": "base",
                "gpu_memory_utilization": 0.1,
                "model_class_name": "BagelPipeline",
                "enable_sleep_mode": True,
                "enforce_eager": True,
                "max_num_batched_tokens": 8192,
                "parallel_config": {"tensor_parallel_size": 2},
            },
            "final_output": True,
            "final_output_type": "image",
        }
    ]
    engine = AsyncOmni(model=MODEL, stages=stages, init_timeout=600, enable_sleep_mode=True)
    yield engine
    engine.shutdown()
    await asyncio.sleep(1.5)


class TestBagelDiffusionSleepMode:
    """BAGEL diffusion sleep/wake on a class-scoped TP=2 BagelPipeline."""

    @pytest.mark.slow
    @pytest.mark.diffusion
    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    async def test_diffusion_sleep_handshake(self, bagel_diffusion_engine: AsyncOmni):
        try:
            acks = await bagel_diffusion_engine.sleep(stage_ids=[0], level=1)
            assert len(acks) >= 1
            assert all(_get_ack_info(ack, "status") == "SUCCESS" for ack in acks)
            await bagel_diffusion_engine.wake_up(stage_ids=[0])
        finally:
            await _ensure_awake(bagel_diffusion_engine, [0])

    @pytest.mark.slow
    @pytest.mark.diffusion
    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    async def test_cross_device_cleanup(self, bagel_diffusion_engine: AsyncOmni):
        try:
            used_before = _get_device_global_memory_used_gib(0) + _get_device_global_memory_used_gib(1)
            acks = await bagel_diffusion_engine.sleep(stage_ids=[0], level=1)
            await asyncio.sleep(1.5)
            used_after = _get_device_global_memory_used_gib(0) + _get_device_global_memory_used_gib(1)
            drop_gib = used_before - used_after
            freed_gb = sum(_get_ack_info(ack, "freed_bytes", 0) for ack in acks) / 1024**3
            assert freed_gb > 14.0 or drop_gib > 8.0, f"ACK={freed_gb:.2f} GiB, global_drop={drop_gib:.2f} GiB"
        finally:
            await _ensure_awake(bagel_diffusion_engine, [0])

    @pytest.mark.slow
    @pytest.mark.diffusion
    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    async def test_diffusion_sleep_wake_generate(self, bagel_diffusion_engine: AsyncOmni):
        import gc

        device_id = 1
        try:
            prompt = "A huge swimming pool, with many people swimming."
            sp = OmniDiffusionSamplingParams(num_inference_steps=4, height=512, width=512, seed=42)
            llm_sp = SamplingParams()

            base_output = None
            async for output in bagel_diffusion_engine.generate(
                prompt, request_id="base", sampling_params_list=[llm_sp, sp]
            ):
                base_output = output
            assert base_output is not None and len(base_output.images) > 0

            current_omni_platform.empty_cache()
            vram_initial = _get_device_global_memory_used_gib(device_id)

            acks = await bagel_diffusion_engine.sleep(stage_ids=[0], level=1)
            statuses = [_get_ack_info(ack, "status") for ack in acks]
            assert all(s == "SUCCESS" for s in statuses), f"Sleep failed. Statuses: {statuses}"

            reported_freed_gib = sum(_get_ack_info(ack, "freed_bytes", 0) for ack in acks) / 1024**3
            await asyncio.sleep(2)
            current_omni_platform.empty_cache()
            vram_sleeping = _get_device_global_memory_used_gib(device_id)
            assert reported_freed_gib > 14.0 or vram_sleeping < 5.0, (
                f"Reported: {reported_freed_gib:.2f}G, Measured: {vram_sleeping:.2f}G"
            )

            await bagel_diffusion_engine.wake_up(stage_ids=[0])
            await bagel_diffusion_engine.resume_generation(stage_ids=[0])
            await asyncio.sleep(2.0)
            gc.collect()
            current_omni_platform.empty_cache()
            vram_restored = _get_device_global_memory_used_gib(device_id)
            assert abs(vram_restored - vram_initial) < 3.0, "VRAM failed to restore to initial levels"

            post_output = None
            async for output in bagel_diffusion_engine.generate(
                prompt, request_id="post", sampling_params_list=[llm_sp, sp]
            ):
                post_output = output
            assert post_output is not None
            assert len(base_output.images) == len(post_output.images)
            assert post_output.images[0] is not None
        finally:
            await _ensure_awake(bagel_diffusion_engine, [0])


def _build_bagel_llm_stages() -> tuple[list[dict], list[dict]]:
    common_args = {
        "worker_type": "ar",
        "enable_sleep_mode": True,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": 2048,
        "max_num_batched_tokens": 8192,
        "enforce_eager": True,
    }
    stages = [
        {
            "stage_id": 0,
            "stage_type": "llm",
            "runtime": {"process": True, "devices": "0", "max_batch_size": 1},
            "engine_args": {**common_args, "model_stage": "thinker", "gpu_memory_utilization": 0.1},
        },
        {
            "stage_id": 1,
            "stage_type": "llm",
            "engine_input_source": [0],
            "runtime": {"process": True, "devices": "1", "max_batch_size": 1, "connector_type": "queue"},
            "engine_args": {**common_args, "model_stage": "talker", "gpu_memory_utilization": 0.1},
        },
    ]
    connectors = [{"src_stage_id": 0, "dst_stage_id": 1, "connector_type": "queue"}]
    return stages, connectors


class TestBagelCoordinatedSleepMode:
    """Dual-engine coordination (kept skipped; do not delete)."""

    @pytest.mark.skip(
        reason=(
            "Flaky/CI: dual AsyncOmni can fail with "
            "RuntimeError: Orchestrator init failed, StageDiffusionProc died during handshake. "
            "Re-enable when stable (no OOM on coordinated talker+diffusion)."
        )
    )
    @pytest.mark.slow
    @pytest.mark.diffusion
    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    async def test_coordinated_cross_device(self):
        """Heterogeneous coordinated cleanup (talker + diffusion on GPU 1)."""
        if current_omni_platform.is_rocm():
            _clean_device_envs()

        llm_stages, llm_connectors = _build_bagel_llm_stages()
        llm_engine = AsyncOmni(
            model=MODEL, stages=llm_stages, connectors=llm_connectors, init_timeout=600, enable_sleep_mode=True
        )
        diffusion_stages = [
            {
                "stage_id": 0,
                "stage_type": "diffusion",
                "runtime": {"process": True, "devices": "0,1", "max_batch_size": 1},
                "engine_args": {
                    "model_stage": "base",
                    "gpu_memory_utilization": 0.1,
                    "model_class_name": "BagelPipeline",
                    "enable_sleep_mode": True,
                    "enforce_eager": True,
                    "max_num_batched_tokens": 8192,
                    "parallel_config": {"tensor_parallel_size": 2},
                },
                "final_output": True,
                "final_output_type": "image",
            }
        ]
        diffusion_engine = AsyncOmni(model=MODEL, stages=diffusion_stages, init_timeout=600, enable_sleep_mode=True)
        device_id = 1
        try:
            await llm_engine.wake_up(stage_ids=[1])
            await diffusion_engine.wake_up(stage_ids=[0])
            current_omni_platform.empty_cache()
            await asyncio.sleep(2)
            initial_vram = _get_device_global_memory_used_gib(device_id)

            await llm_engine.sleep(stage_ids=[1], level=2)
            await asyncio.sleep(1.0)
            await diffusion_engine.sleep(stage_ids=[0], level=2)
            await asyncio.sleep(3.0)
            current_omni_platform.empty_cache()
            final_vram = _get_device_global_memory_used_gib(device_id)
            assert initial_vram - final_vram > 15.0 or final_vram < 8.0
        finally:
            llm_engine.shutdown()
            diffusion_engine.shutdown()
