# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Entrypoint sleep-mode coverage on small models.

Layering (tiny DiT before 7B on the same L4 so residual thinker weights cannot
OOM the diffusion suite):
1. Diffusion sleep/wake/generate — ``riverclouds/qwen_image_random`` on L4
2. AR protocol (#4473) — ``Qwen/Qwen2.5-Omni-7B`` thinker-only on L4
3. Light multistage orchestration — thinker-only AR + tiny DiT on L4×2

BAGEL BagelPipeline TP=2 / coordinated dual-engine stay in
``tests/e2e/offline_inference/test_bagel_expansion.py``.
"""

from __future__ import annotations

import asyncio
import logging

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
# Sleep/wake on 24 GiB L4 needs CPU-offload headroom. The thinker-only CI overlay
# is the abort-test profile (util 0.90 / max_model_len 16384); 16.78 GiB weights
# already left 0 KV at util 0.85. Match the L4×2 fixture below.
_AR_SLEEP_GPU_MEMORY_UTILIZATION = 0.45
_AR_SLEEP_MAX_MODEL_LEN = 2048
_AR_SLEEP_MAX_NUM_BATCHED_TOKENS = 2048
AR_STAGE_CONFIG = modify_stage_config(
    get_deploy_config_path("ci/qwen2_5_omni_thinker_only.yaml"),
    updates={
        "stages": {
            0: {
                "enable_sleep_mode": True,
                "gpu_memory_utilization": _AR_SLEEP_GPU_MEMORY_UTILIZATION,
                "max_model_len": _AR_SLEEP_MAX_MODEL_LEN,
                "max_num_batched_tokens": _AR_SLEEP_MAX_NUM_BATCHED_TOKENS,
            }
        }
    },
)


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


async def _tiny_dit_multistage_generate(engine: AsyncOmni, prompt: str, request_id: str):
    """2-step tiny-DiT generate through the joint AR + DiT pipeline.

    Restores the old H100 ``test_multistage_llm_diffusion_sleep_wake`` contract
    (generate before and after ``sleep(stage_ids=[0, 1])``) without BAGEL.
    AR is capped so this stays a sleep/wake orchestration check, not a 7B
    completion.
    """
    params_list = [
        SamplingParams(max_tokens=4),
        OmniDiffusionSamplingParams(num_inference_steps=2, height=256, width=256),
    ]
    output = None
    async for item in engine.generate(prompt, request_id=request_id, sampling_params_list=params_list):
        output = item
    assert output is not None, f"generate({request_id!r}) produced no output"
    return output


async def _shutdown_engine_and_clear_gpu(engine: AsyncOmni) -> None:
    """Drop the engine, then wait until this L4 is free for the next class.

    ``shutdown()`` + a short sleep is not enough: residual thinker weights can
    still occupy the device when the next class loads tiny DiT or another 7B.
    """
    from tests.helpers.clean import cleanup_test_environment, wait_for_gpu_memory_to_clear

    engine.shutdown()
    await asyncio.sleep(1.5)
    cleanup_test_environment()
    n = current_omni_platform.device_count()
    if n <= 0:
        return
    # Fail closed. Module autouse cleanup only logs a note on timeout.
    wait_for_gpu_memory_to_clear(
        devices=list(range(n)),
        threshold_ratio=0.15,
        timeout_s=120,
    )


@pytest.fixture(scope="module", autouse=True)
def _module_device_cleanup():
    from tests.helpers.clean import cleanup_test_environment

    print("\n=== PRE-MODULE DEVICE CLEANUP (sleep_mode) ===")
    cleanup_test_environment()
    yield
    print("\n=== POST-MODULE DEVICE CLEANUP (sleep_mode) ===")
    cleanup_test_environment()


# ---------------------------------------------------------------------------
# 1) Diffusion sleep/wake/generate — qwen_image_random (L4)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="class", loop_scope="class")
async def diffusion_engine():
    """Shared tiny diffusion engine on L4."""
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
    await _shutdown_engine_and_clear_gpu(engine)


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
# 2) AR protocol — Omni thinker-only (L4)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="class", loop_scope="class")
async def ar_engine():
    """Shared thinker-only AR engine for #4473 protocol checks on L4."""
    engine = AsyncOmni(
        model=MODEL_AR,
        deploy_config=AR_STAGE_CONFIG,
        enable_sleep_mode=True,
        stage_init_timeout=1200,
    )
    yield engine
    await _shutdown_engine_and_clear_gpu(engine)


class TestOmniArSleepMode:
    """AR sleep protocol on Qwen2.5-Omni thinker-only."""

    @pytest.mark.advanced_model
    @pytest.mark.omni
    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
    async def test_llm_sleep_ack(self, ar_engine: AsyncOmni):
        """AR sleep reports EngineCore SUCCESS; generate works after a full wake.

        ``AsyncOmni.sleep()`` synthesizes OmniACK with ``freed_bytes=0`` (no
        worker handshake), so VRAM / ``freed_bytes`` is not a real contract.
        """
        try:
            acks = await ar_engine.sleep(stage_ids=[0], level=1)
            assert acks
            assert all(get_ack_info(ack, "status") == "SUCCESS" for ack in acks)
            for ack in acks:
                meta = get_ack_info(ack, "metadata") or {}
                assert meta.get("path") == "engine_core", f"expected EngineCore ACK, got metadata={meta}"

            await ar_engine.wake_up(stage_ids=[0])
            await ar_engine.resume_generation(stage_ids=[0])
            output = None
            async for item in ar_engine.generate("test", sampling_params=SamplingParams(max_tokens=4)):
                output = item
            assert output is not None, "generate after full wake produced no output"
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
# 3) Light multistage — small AR + small DiT (L4×2)
# ---------------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=2)
@pytest.mark.asyncio
async def test_multistage_ar_diffusion_sleep_wake():
    """Orchestration: generate → joint sleep/wake/resume → generate.

    Covers the old H100 path that ``TestBagelCoordinatedSleepMode`` (still
    skipped) and BAGEL TP=2 expansion (diffusion-only) do not: a 2-step
    tiny-DiT generate after ``sleep(stage_ids=[0, 1])`` + ``resume_generation``.
    """
    stages = [
        {
            "stage_id": 0,
            "stage_type": "llm",
            "runtime": {"process": True, "devices": "0", "max_batch_size": 1},
            "engine_args": {
                "model": MODEL_AR,
                "model_stage": "thinker",
                "gpu_memory_utilization": _AR_SLEEP_GPU_MEMORY_UTILIZATION,
                "dtype": "bfloat16",
                "enable_sleep_mode": True,
                "trust_remote_code": True,
                "enforce_eager": True,
                "max_model_len": _AR_SLEEP_MAX_MODEL_LEN,
                "max_num_batched_tokens": _AR_SLEEP_MAX_NUM_BATCHED_TOKENS,
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
        await _tiny_dit_multistage_generate(engine, "warmup", "warmup")

        acks = await engine.sleep(stage_ids=[0, 1], level=1)
        assert len(acks) == 2
        assert all(get_ack_info(ack, "status") == "SUCCESS" for ack in acks)

        await engine.wake_up(stage_ids=[0, 1])
        await engine.resume_generation(stage_ids=[0, 1])
        post_output = await _tiny_dit_multistage_generate(engine, "verify", "verify")
        assert post_output is not None
        logger.info("Light multistage joint sleep/wake/resume generate OK")
    finally:
        await _shutdown_engine_and_clear_gpu(engine)
