# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Entrypoint sleep-mode coverage on small models.

Layering:
1. AR protocol (#4473) — ``Qwen/Qwen2.5-Omni-7B`` thinker-only on L4
2. Diffusion sleep/wake/generate — ``riverclouds/qwen_image_random`` on L4
3. Light multistage orchestration — thinker-only AR + tiny DiT on H100×2

BAGEL BagelPipeline TP=2 / coordinated dual-engine stay in
``tests/e2e/offline_inference/test_bagel_expansion.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os

import pytest
import pytest_asyncio
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OmniTest")

MODEL_DIFF = "riverclouds/qwen_image_random"
MODEL_AR = "Qwen/Qwen2.5-Omni-7B"
AR_STAGE_CONFIG = modify_stage_config(
    get_deploy_config_path("ci/qwen2_5_omni_thinker_only.yaml"),
    updates={"stages": {0: {"enable_sleep_mode": True}}},
)


def clean_device_envs():
    """Clear device-visibility env vars so tests see all available devices."""
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
        "ONEAPI_DEVICE_SELECTOR",
        "ASCEND_RT_VISIBLE_DEVICES",
    ):
        os.environ.pop(key, None)


def get_device_global_memory_used_gib(device_id: int) -> float:
    """GPU-wide memory in use (GiB), includes all processes (driver view)."""
    try:
        with current_omni_platform.device(device_id):
            current_omni_platform.synchronize()
            free_b, total_b = current_omni_platform.get_device_memory()
        return (total_b - free_b) / 1024**3
    except Exception as e:
        logger.warning("get_device_global_memory_used_gib(%s): %s", device_id, e)
        return 0.0


def get_ack_info(ack, key, default=None):
    if hasattr(ack, key):
        return getattr(ack, key)
    if isinstance(ack, dict):
        return ack.get(key, default)
    return default


async def _ensure_awake(engine: AsyncOmni, stage_ids: list[int] | None = None) -> None:
    try:
        await engine.wake_up(stage_ids=stage_ids)
    except Exception as e:
        logger.warning("ensure_awake failed (stage_ids=%s): %s", stage_ids, e)
    try:
        await engine.resume_generation(stage_ids=stage_ids)
    except Exception as e:
        logger.warning("ensure_resume failed (stage_ids=%s): %s", stage_ids, e)


@pytest.fixture(scope="module", autouse=True)
def _module_device_cleanup():
    from tests.helpers.clean import cleanup_test_environment

    print("\n=== PRE-MODULE DEVICE CLEANUP (sleep_mode) ===")
    cleanup_test_environment()
    yield
    print("\n=== POST-MODULE DEVICE CLEANUP (sleep_mode) ===")
    cleanup_test_environment()


# ---------------------------------------------------------------------------
# 1) AR protocol — Omni thinker-only (L4)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="class", loop_scope="class")
async def ar_engine():
    """Shared thinker-only AR engine for #4473 protocol checks on L4."""
    if current_omni_platform.is_rocm():
        clean_device_envs()
    engine = AsyncOmni(
        model=MODEL_AR,
        deploy_config=AR_STAGE_CONFIG,
        enable_sleep_mode=True,
        stage_init_timeout=1200,
    )
    yield engine
    engine.shutdown()
    await asyncio.sleep(1.5)


class TestOmniArSleepMode:
    """AR sleep protocol on Qwen2.5-Omni thinker-only."""

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
    async def test_llm_sleep_ack(self, ar_engine: AsyncOmni):
        device_id = 0
        try:
            used_before = get_device_global_memory_used_gib(device_id)
            acks = await ar_engine.sleep(stage_ids=[0], level=1)
            await asyncio.sleep(1.5)
            used_after = get_device_global_memory_used_gib(device_id)
            drop_gib = used_before - used_after
            assert all(get_ack_info(ack, "status") == "SUCCESS" for ack in acks)
            freed_gib = sum(get_ack_info(ack, "freed_bytes", 0) for ack in acks) / 1024**3
            logger.info(
                "AR: ACK freed=%.2f GiB, global drop=%.2f GiB (before=%.2f, after=%.2f)",
                freed_gib,
                drop_gib,
                used_before,
                used_after,
            )
            assert freed_gib > 1.0 or drop_gib > 0.5, (
                "Expected ACK freed_bytes or global VRAM drop after sleep. "
                f"ACK={freed_gib:.2f} GiB, global_drop={drop_gib:.2f} GiB"
            )
        finally:
            await _ensure_awake(ar_engine, [0])

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
    async def test_partial_wake_blocks_generate(self, ar_engine: AsyncOmni):
        """#4473 Repro B: generate() rejected while kv_cache stays asleep."""
        try:
            await ar_engine.sleep(stage_ids=[0], level=1)
            await ar_engine.wake_up(stage_ids=[0], tags=["weights"])
            await ar_engine.resume_generation(stage_ids=[0])
            with pytest.raises(RuntimeError, match="partially or fully asleep"):
                async for _ in ar_engine.generate("test", sampling_params=SamplingParams(max_tokens=4)):
                    pass
        finally:
            await _ensure_awake(ar_engine, [0])

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
    async def test_duplicate_wake_is_idempotent(self, ar_engine: AsyncOmni):
        """#4473 Repro C: duplicate wake_up(tags=None) is a safe no-op."""
        try:
            await ar_engine.sleep(stage_ids=[0], level=1)
            first_acks = await ar_engine.wake_up(stage_ids=[0])
            assert len(first_acks) > 0, "First wake_up() should return ACKs"
            second_acks = await ar_engine.wake_up(stage_ids=[0])
            assert second_acks == [], f"Duplicate wake_up() should return [] but got {second_acks}"
        finally:
            await _ensure_awake(ar_engine, [0])


# ---------------------------------------------------------------------------
# 2) Diffusion sleep/wake/generate — qwen_image_random (L4)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="class", loop_scope="class")
async def diffusion_engine():
    """Shared tiny diffusion engine on L4."""
    if current_omni_platform.is_rocm():
        clean_device_envs()
    engine = AsyncOmni(
        model=MODEL_DIFF,
        enable_sleep_mode=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        stage_init_timeout=1200,
    )
    yield engine
    engine.shutdown()
    await asyncio.sleep(1.5)


class TestOmniDiffusionSleepMode:
    """Diffusion worker sleep/wake on ``qwen_image_random`` (TP=1)."""

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
    async def test_diffusion_sleep_handshake(self, diffusion_engine: AsyncOmni):
        try:
            acks = await diffusion_engine.sleep(level=1)
            assert acks is not None
            assert all(get_ack_info(ack, "status") == "SUCCESS" for ack in acks)
            await diffusion_engine.wake_up()
        finally:
            await _ensure_awake(diffusion_engine)

    @pytest.mark.omni
    @pytest.mark.core_model
    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
    async def test_diffusion_sleep_wake_generate(self, diffusion_engine: AsyncOmni):
        try:
            acks = await diffusion_engine.sleep(level=1)
            assert acks is not None
            await diffusion_engine.wake_up()
            await diffusion_engine.resume_generation()
            async for _ in diffusion_engine.generate(
                "test",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2, height=256, width=256),
            ):
                pass
        finally:
            await _ensure_awake(diffusion_engine)


# ---------------------------------------------------------------------------
# 3) Light multistage — small AR + small DiT (H100×2; Entrypoint L4 is cards_1)
# ---------------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.asyncio
async def test_multistage_ar_diffusion_sleep_wake():
    """Orchestration: sleep/wake both stages on thinker-only AR + tiny DiT.

    Covers joint ``sleep(stage_ids=[0, 1])`` / wake / resume that single-stage
    suites do not. End-to-end cross-model generate is not required here —
    BAGEL BagelPipeline TP=2 remains the heavy product path in expansion.
    """
    if current_omni_platform.is_rocm():
        clean_device_envs()
    if current_omni_platform.get_device_count() < 2:
        pytest.skip("Need 2 GPUs for light multistage sleep/wake")

    stages = [
        {
            "stage_id": 0,
            "stage_type": "llm",
            "runtime": {"process": True, "devices": "0", "max_batch_size": 1},
            "engine_args": {
                "model": MODEL_AR,
                "model_stage": "thinker",
                "gpu_memory_utilization": 0.45,
                "dtype": "bfloat16",
                "enable_sleep_mode": True,
                "trust_remote_code": True,
                "enforce_eager": True,
                "max_model_len": 2048,
                "max_num_batched_tokens": 2048,
            },
        },
        {
            "stage_id": 1,
            "stage_type": "diffusion",
            "runtime": {"process": True, "devices": "1", "max_batch_size": 1},
            "engine_args": {
                "model": MODEL_DIFF,
                "gpu_memory_utilization": 0.4,
                "dtype": "bfloat16",
                "enable_sleep_mode": True,
                "enforce_eager": True,
                "tensor_parallel_size": 1,
            },
            "final_output": True,
            "final_output_type": "image",
        },
    ]
    connectors = [{"src_stage_id": 0, "dst_stage_id": 1, "connector_type": "queue"}]
    engine = AsyncOmni(
        model=MODEL_AR,
        stages=stages,
        connectors=connectors,
        enable_sleep_mode=True,
        stage_init_timeout=1200,
    )
    try:
        acks = await engine.sleep(stage_ids=[0, 1], level=1)
        assert len(acks) == 2
        assert all(get_ack_info(ack, "status") == "SUCCESS" for ack in acks)

        await engine.wake_up(stage_ids=[0, 1])
        await engine.resume_generation(stage_ids=[0, 1])
        logger.info("Light multistage joint sleep/wake/resume OK")
    finally:
        engine.shutdown()
