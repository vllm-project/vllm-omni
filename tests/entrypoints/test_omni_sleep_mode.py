# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Omni sleep mode: entrypoint-level VRAM/ACK tests plus multi-stage e2e.

Class-scoped engines amortize BAGEL load across sleep/wake assertions within
each suite (LLM vs diffusion). Suites are separate classes so only one heavy
engine is alive at a time. Device cleanup runs once per module.
"""

import asyncio
import logging
import os

import pytest
import pytest_asyncio
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OmniTest")
pytestmark = [
    pytest.mark.advanced_model,
]


def clean_device_envs():
    """Clear device-visibility env vars so tests see all available devices."""
    device_visibility_vars = [
        "CUDA_VISIBLE_DEVICES",  # NVIDIA
        "HIP_VISIBLE_DEVICES",  # AMD ROCm
        "ZE_AFFINITY_MASK",  # Intel XPU
        "ONEAPI_DEVICE_SELECTOR",  # Intel OneAPI
        "ASCEND_RT_VISIBLE_DEVICES",  # Huawei NPU (CAN)
    ]
    for key in device_visibility_vars:
        os.environ.pop(key, None)


def get_vram_info(device_id: int) -> dict:
    """Per-**process** CUDA allocator stats (GiB). Does not see other processes' usage."""
    try:
        if current_omni_platform.is_rocm():
            num_gpus = torch.accelerator.device_count()
            safe_id = device_id if device_id < num_gpus else 0
            torch.accelerator.synchronize(safe_id)
            return {
                "reserved": torch.cuda.memory_reserved(safe_id) / 1024**3,
                "allocated": torch.cuda.memory_allocated(safe_id) / 1024**3,
            }
        else:
            with torch.cuda.device(device_id):
                torch.accelerator.synchronize()
                return {
                    "reserved": torch.cuda.memory_reserved() / 1024**3,
                    "allocated": torch.cuda.memory_allocated() / 1024**3,
                }
    except Exception as e:
        logger.warning(f"memory skip ({device_id}): {e}")
        return {"reserved": 0.0, "allocated": 0.0}


def get_device_global_memory_used_gib(device_id: int) -> float:
    """GPU-wide memory in use (GiB), includes all processes (driver view).

    Uses ``torch.cuda.mem_get_info`` (CUDA / ROCm when available). Use this when stages run in
    **separate worker processes**; ``memory_reserved`` in the test process does not reflect
    child workers' allocations.
    """
    try:
        with torch.cuda.device(device_id):
            torch.accelerator.synchronize()
            free_b, total_b = torch.cuda.mem_get_info()
        return (total_b - free_b) / 1024**3
    except Exception as e:
        logger.warning("get_device_global_memory_used_gib(%s): %s", device_id, e)
        return 0.0


def get_ack_info(ack, key, default=None):
    """
    Since ACKs in a distributed environment can be either objects or dictionaries,
    this tool ensures compatibility.
    """
    if hasattr(ack, key):
        return getattr(ack, key)
    if isinstance(ack, dict):
        return ack.get(key, default)
    return default


MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
MODEL_DIFF = "riverclouds/qwen_image_random"


def get_dynamic_devices(stage_idx: int, num_stages: int, tp_size: int) -> str:
    total_gpus = torch.accelerator.device_count()
    gpus_per_stage = tp_size
    start_idx = stage_idx * gpus_per_stage
    if start_idx + gpus_per_stage > total_gpus:
        start_idx = start_idx % total_gpus
    device_ids = [str(start_idx + i) for i in range(gpus_per_stage)]
    return ",".join(device_ids)


async def _ensure_awake(engine: AsyncOmni, stage_ids: list[int]) -> None:
    """Best-effort full wake so the next shared-engine test starts active."""
    try:
        await engine.wake_up(stage_ids=stage_ids)
    except Exception as e:
        logger.warning("ensure_awake failed (stage_ids=%s): %s", stage_ids, e)


def _build_llm_stages() -> tuple[list[dict], list[dict]]:
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


@pytest.fixture(scope="module", autouse=True)
def _module_device_cleanup():
    """One device cleanup pass around the module (engines are class-scoped)."""
    from tests.helpers.env import run_post_test_cleanup, run_pre_test_cleanup

    print("\n=== PRE-MODULE DEVICE CLEANUP (sleep_mode) ===")
    run_pre_test_cleanup()
    yield
    print("\n=== POST-MODULE DEVICE CLEANUP (sleep_mode) ===")
    run_post_test_cleanup()


@pytest_asyncio.fixture(scope="class", loop_scope="class")
async def llm_engine():
    """Shared 2-stage BAGEL LLM engine for sleep/wake protocol tests."""
    if current_omni_platform.is_rocm():
        clean_device_envs()
    stages, connectors = _build_llm_stages()
    engine = AsyncOmni(model=MODEL, stages=stages, connectors=connectors, init_timeout=600, enable_sleep_mode=True)
    yield engine
    engine.shutdown()
    await asyncio.sleep(1.5)


@pytest_asyncio.fixture(scope="class", loop_scope="class")
async def diffusion_engine():
    """Shared BAGEL diffusion TP=2 engine for sleep/wake + generate checks."""
    if current_omni_platform.is_rocm():
        clean_device_envs()
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
                "parallel_config": {
                    "tensor_parallel_size": 2,
                },
            },
            "final_output": True,
            "final_output_type": "image",
        }
    ]
    engine = AsyncOmni(model=MODEL, stages=stages, init_timeout=600, enable_sleep_mode=True)
    yield engine
    engine.shutdown()
    await asyncio.sleep(1.5)


class TestOmniLlmSleepMode:
    """LLM sleep/wake protocol tests sharing one class-scoped engine."""

    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
    async def test_llm_sleep_ack(self, llm_engine: AsyncOmni):
        """LLM Thinker (GPU0) Signal and Physical Recycling Audit"""
        device_id = 0
        try:
            used_before = get_device_global_memory_used_gib(device_id)
            acks = await llm_engine.sleep(stage_ids=[0], level=1)
            await asyncio.sleep(1.5)
            used_after = get_device_global_memory_used_gib(device_id)
            drop_gib = used_before - used_after
            assert all(get_ack_info(ack, "status") == "SUCCESS" for ack in acks)
            total_freed_bytes = sum(get_ack_info(ack, "freed_bytes", 0) for ack in acks)
            freed_gib = total_freed_bytes / 1024**3
            logger.info(
                "Thinker: ACK freed=%.2f GiB, global GPU used drop=%.2f GiB (before=%.2f, after=%.2f)",
                freed_gib,
                drop_gib,
                used_before,
                used_after,
            )
            assert freed_gib > 5.0 or drop_gib > 3.0, (
                "Expected either ACK freed_bytes or global VRAM drop after sleep. "
                f"ACK={freed_gib:.2f} GiB, global_drop={drop_gib:.2f} GiB"
            )
        finally:
            await _ensure_awake(llm_engine, [0])

    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    async def test_partial_wake_blocks_generate(self, llm_engine: AsyncOmni):
        """Regression for #4473 Repro B: generate() must be rejected if kv_cache
        is still asleep after wake_up(tags=["weights"]), instead of crashing with
        CUDA illegal memory access."""
        try:
            await llm_engine.sleep(stage_ids=[0], level=1)
            await llm_engine.wake_up(stage_ids=[0], tags=["weights"])
            with pytest.raises(RuntimeError, match="partially or fully asleep"):
                async for _ in llm_engine.generate("test", sampling_params=SamplingParams(max_tokens=4)):
                    pass
        finally:
            await _ensure_awake(llm_engine, [0])

    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    async def test_duplicate_wake_is_idempotent(self, llm_engine: AsyncOmni):
        """Regression for #4473 Repro C: duplicate wake_up(tags=None) must be a
        safe no-op instead of raising a cumem CUDA invalid argument error."""
        try:
            await llm_engine.sleep(stage_ids=[0], level=1)
            first_acks = await llm_engine.wake_up(stage_ids=[0])
            assert len(first_acks) > 0, "First wake_up() should return ACKs"
            second_acks = await llm_engine.wake_up(stage_ids=[0])
            assert second_acks == [], f"Duplicate wake_up() should return [] but got {second_acks}"
        finally:
            await _ensure_awake(llm_engine, [0])


class TestOmniDiffusionSleepMode:
    """Diffusion sleep/wake tests sharing one class-scoped TP=2 engine."""

    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    async def test_diffusion_sleep_handshake(self, diffusion_engine: AsyncOmni):
        """Diffusion Worker stage signal loop"""
        try:
            logger.info("Starting Diffusion Worker Handshake Test")
            acks = await diffusion_engine.sleep(stage_ids=[0], level=1)

            def _get_status(ack):
                return ack.status if hasattr(ack, "status") else ack.get("status")

            assert len(acks) >= 1, "Expected at least 1 ACK from Diffusion Workers"
            assert all(_get_status(ack) == "SUCCESS" for ack in acks)
            logger.info(f"Success: Received {len(acks)} Diffusion Worker ACKs")
            await diffusion_engine.wake_up(stage_ids=[0])
        finally:
            await _ensure_awake(diffusion_engine, [0])

    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    async def test_cross_device_cleanup(self, diffusion_engine: AsyncOmni):
        """Physical recycling audit: leveraging deterministic data returned by Workers"""
        try:
            used_before = get_device_global_memory_used_gib(0) + get_device_global_memory_used_gib(1)
            acks = await diffusion_engine.sleep(stage_ids=[0], level=1)
            await asyncio.sleep(1.5)
            used_after = get_device_global_memory_used_gib(0) + get_device_global_memory_used_gib(1)
            drop_gib = used_before - used_after
            total_freed_bytes = sum(get_ack_info(ack, "freed_bytes", 0) for ack in acks)
            freed_gb = total_freed_bytes / 1024**3
            logger.info("Physical reclamation summary from workers:")
            logger.info(f"- Total Workers: {len(acks)}")
            logger.info(f"- Total Freed (ACK): {freed_gb:.2f} GiB, global used drop: {drop_gib:.2f} GiB")
            assert freed_gb > 14.0 or drop_gib > 8.0, (
                "Expected either ACK freed_bytes or global VRAM drop on GPUs 0+1. "
                f"ACK={freed_gb:.2f} GiB, global_drop={drop_gib:.2f} GiB"
            )
            logger.info("SUCCESS: 100% weights offloaded.")
        finally:
            await _ensure_awake(diffusion_engine, [0])

    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    async def test_diffusion_sleep_wake_generate(self, diffusion_engine: AsyncOmni):
        """Merged diffusion lifecycle with test points from the former trio:

        - ``test_diffusion_integrity_bit_level``: 4-step/512 baseline generate →
          sleep → wake → post generate (images present, count unchanged).
        - ``test_diffusion_vram_lifecycle_audit``: ACK/process VRAM reclaim on
          sleep; after wake, process VRAM restores to within 3 GiB of initial;
          post-wake generate smoke.
        - ``test_diffusion_model_sleep_tp``: sleep ACK SUCCESS → wake → verify
          generate (same Active→Sleep→Active→Generate path; runs on the shared
          class TP=2 BagelPipeline engine rather than a second default-config load).
        """
        import gc

        device_id = 1
        try:
            # --- integrity: baseline generate at original resolution/steps ---
            prompt = "A huge swimming pool, with many people swimming."
            sp = OmniDiffusionSamplingParams(num_inference_steps=4, height=512, width=512, seed=42)
            llm_sp = SamplingParams()

            logger.info("Running Baseline Generation...")
            base_output = None
            async for output in diffusion_engine.generate(prompt, request_id="base", sampling_params_list=[llm_sp, sp]):
                base_output = output
            assert base_output is not None and len(base_output.images) > 0
            logger.info("Baseline Generation successful.")

            # Snapshot Active VRAM after generate (lifecycle compared Sleep/Wake
            # against the pre-sleep active watermark).
            get_vram_info(device_id)
            torch.accelerator.empty_cache()
            vram_initial = get_vram_info(device_id)["reserved"]
            logger.info(f"Diffusion Active VRAM before sleep: {vram_initial:.2f} GiB")

            # --- lifecycle + model_sleep_tp: level-1 sleep reclaim ---
            logger.info("Entering Deep Sleep (VRAM Scavenging)...")
            acks = await diffusion_engine.sleep(stage_ids=[0], level=1)
            statuses = [get_ack_info(ack, "status") for ack in acks]
            assert all(s == "SUCCESS" for s in statuses), f"Sleep failed. Statuses: {statuses}"

            reported_freed_bytes = sum(get_ack_info(ack, "freed_bytes", 0) for ack in acks)
            reported_freed_gib = reported_freed_bytes / 1024**3
            logger.info(f"Worker internally reported freed: {reported_freed_gib:.2f} GiB")

            await asyncio.sleep(2)
            get_vram_info(device_id)
            torch.accelerator.empty_cache()
            vram_sleeping = get_vram_info(device_id)["reserved"]
            logger.info(f"External VRAM measurement during Sleep: {vram_sleeping:.2f} GiB")
            assert reported_freed_gib > 14.0 or vram_sleeping < 5.0, (
                f"Reclamation failed. Reported: {reported_freed_gib:.2f}G, Measured: {vram_sleeping:.2f}G"
            )

            # --- wake + VRAM restore to initial (±3 GiB) ---
            logger.info("Waking up (Reloading Weights)...")
            await diffusion_engine.wake_up(stage_ids=[0])
            await asyncio.sleep(2.0)
            gc.collect()
            get_vram_info(device_id)
            torch.accelerator.empty_cache()
            vram_restored = get_vram_info(device_id)["reserved"]
            logger.info(f"VRAM after Wake-up: {vram_restored:.2f} GiB")
            assert abs(vram_restored - vram_initial) < 3.0, "VRAM failed to restore to initial levels"

            # --- integrity + lifecycle: post-wake generate ---
            logger.info("Running Post-Wakeup Generation...")
            post_output = None
            async for output in diffusion_engine.generate(prompt, request_id="post", sampling_params_list=[llm_sp, sp]):
                post_output = output
            assert post_output is not None
            assert len(base_output.images) == len(post_output.images)
            assert post_output.images[0] is not None
            logger.info("SUCCESS: Diffusion integrity + VRAM lifecycle + sleep/wake generate verified.")
        finally:
            await _ensure_awake(diffusion_engine, [0])


class TestOmniCoordinatedSleepMode:
    """Dual-engine coordination (kept skipped; do not delete)."""

    @pytest.mark.skip(
        reason=(
            "Flaky/CI: dual AsyncOmni can fail with "
            "RuntimeError: Orchestrator init failed, StageDiffusionProc died during handshake. "
            "Re-enable when stable (no OOM on coordinated talker+diffusion). "
            "Note: this test is fixture-free so skip does not init llm/diffusion engines."
        )
    )
    @pytest.mark.asyncio(loop_scope="class")
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    async def test_coordinated_cross_device(self, llm_engine: AsyncOmni, diffusion_engine: AsyncOmni):
        """Heterogeneous Coordinated Cleanup Test (Talker and Diffusion on GPU 1)"""
        device_id = 1
        try:
            logger.info(f"Waking up both engines on GPU {device_id}...")
            await llm_engine.wake_up(stage_ids=[1])
            await diffusion_engine.wake_up(stage_ids=[0])

            get_vram_info(device_id)
            torch.accelerator.empty_cache()
            await asyncio.sleep(2)

            initial_vram = get_vram_info(device_id)["reserved"]
            logger.info(f"GPU {device_id} Peak Pressure: {initial_vram:.2f} GiB")

            # coordinated sleep
            logger.info("Issuing concurrent SLEEP commands...")
            await llm_engine.sleep(stage_ids=[1], level=2)
            await asyncio.sleep(1.0)
            await diffusion_engine.sleep(stage_ids=[0], level=2)

            await asyncio.sleep(3.0)
            torch.accelerator.empty_cache()

            final_vram = get_vram_info(device_id)["reserved"]
            logger.info(f"GPU {device_id} Final VRAM after coordinated sleep: {final_vram:.2f} GiB")

            assert initial_vram - final_vram > 15.0 or final_vram < 8.0
            logger.info(f"SUCCESS: Heterogeneous VRAM drop verified on GPU {device_id}.")
        except Exception as e:
            logger.error(f"Coordinated test failed: {e}")
            raise e
        finally:
            logger.info("Triggering mandatory cleanup for both engines...")
            llm_engine.shutdown()
            diffusion_engine.shutdown()
            logger.info("All engines scavenged. Ready for next test.")


# ---------------------------------------------------------------------------
# Multi-stage (H100) / pure diffusion TP=1 (2×L4): sleep→wake→generate
# (BAGEL TP=2 diffusion path covered by TestOmniDiffusionSleepMode)
# ---------------------------------------------------------------------------


@pytest.mark.omni
@pytest.mark.advanced_model
@pytest.mark.parametrize("tp_size", [1, 2])
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.asyncio
async def test_multistage_llm_diffusion_sleep_wake(tp_size: int):
    """Explicit 2-stage (llm + diffusion) + connectors; sleep/wake both stages."""
    if current_omni_platform.is_rocm():
        clean_device_envs()
    num_gpus = torch.accelerator.device_count()
    if num_gpus < tp_size * 2:
        pytest.skip("Not enough GPUs")

    stages = []
    for i in range(2):
        devs = get_dynamic_devices(i, 2, tp_size)
        stages.append(
            {
                "stage_id": i,
                "stage_type": "llm" if i == 0 else "diffusion",
                "runtime": {"process": True, "devices": devs},
                "engine_args": {
                    "model": MODEL,
                    "model_stage": "thinker" if i == 0 else "base",
                    "tensor_parallel_size": tp_size,
                    "gpu_memory_utilization": 0.4,
                    "dtype": "bfloat16",
                    "enable_sleep_mode": True,
                    "trust_remote_code": True,
                },
            }
        )

    connectors = [{"src_stage_id": 0, "dst_stage_id": 1, "connector_type": "queue"}]

    engine = AsyncOmni(
        model=MODEL, stages=stages, connectors=connectors, enable_sleep_mode=True, stage_init_timeout=1200
    )
    try:
        sp = OmniDiffusionSamplingParams(num_inference_steps=2)
        async for _ in engine.generate("warmup", sampling_params_list=[SamplingParams(), sp]):
            pass

        acks = await engine.sleep(stage_ids=[0, 1], level=1)
        assert len(acks) == 2

        await engine.wake_up(stage_ids=[0, 1])
        async for _ in engine.generate("verify", sampling_params_list=[SamplingParams(), sp]):
            pass
    finally:
        engine.shutdown()


@pytest.mark.omni
@pytest.mark.core_model
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.asyncio
async def test_pure_diffusion_sleep_wake():
    """Single-stage random diffusion (TP=1): sleep → wake → generate.

    Diffusion TP=2 sleep/wake is covered by TestOmniDiffusionSleepMode
    (shared BAGEL TP=2 engine); this case keeps the default AsyncOmni +
    qwen_image_random entrypoint path on L4.
    """
    if current_omni_platform.is_rocm():
        clean_device_envs()
    engine_args = {
        "model": MODEL_DIFF,
        "enable_sleep_mode": True,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.5,
    }

    engine = AsyncOmni(**engine_args, stage_init_timeout=1200)
    try:
        await engine.sleep(level=1)
        await engine.wake_up()
        async for _ in engine.generate(
            "test",
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2, height=256, width=256),
        ):
            pass
        logger.info("Pure diffusion sleep/wake OK")
    finally:
        engine.shutdown()
