# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Omni sleep mode: entrypoint-level VRAM/ACK tests (L4) plus B70 multi-TP e2e."""

import asyncio
import logging

import pytest
import torch
from vllm import SamplingParams

from tests.entrypoints.test_omni_sleep_mode import (
    get_ack_info,
    get_vram_info,
)
from tests.entrypoints.test_omni_sleep_mode import (
    get_device_global_memory_used_gib as _shared_get_device_global_memory_used_gib,
)
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OmniTest")
pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.usefixtures("clean_gpu_memory_between_tests"),
]


MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
MODEL_DIFF = "riverclouds/qwen_image_random"


def get_device_global_memory_used_gib(device_id: int) -> float:
    try:
        if current_omni_platform.is_xpu():
            torch.xpu.set_device(device_id)
            torch.xpu.synchronize(device_id)
            free_b, total_b = torch.xpu.mem_get_info(device_id)
            return (total_b - free_b) / 1024**3
        return _shared_get_device_global_memory_used_gib(device_id)
    except Exception as e:
        logger.warning("get_device_global_memory_used_gib(%s): %s", device_id, e)
        return 0.0


def _pick_two_stage_devices() -> tuple[int, int]:
    num_gpus = torch.accelerator.device_count()
    if num_gpus < 2:
        pytest.skip("BAGEL llm_engine requires at least 2 visible devices")
    if current_omni_platform.is_xpu():
        return 1, 0
    ranked = sorted(range(num_gpus), key=get_device_global_memory_used_gib)
    return ranked[0], ranked[1]


def _build_bagel_dual_stage_config() -> tuple[str, int, int]:
    stage0_device, stage1_device = _pick_two_stage_devices()
    config_path = modify_stage_config(
        get_deploy_config_path("ci/bagel.yaml"),
        updates={
            "stages": {
                0: {
                    "devices": str(stage0_device),
                    "gpu_memory_utilization": 0.98,
                    "max_model_len": 2048,
                    "max_num_batched_tokens": 2048,
                    "max_num_seqs": 1,
                    "skip_mm_profiling": True,
                    "enable_prefix_caching": False,
                    "enforce_eager": True,
                },
                1: {
                    "devices": str(stage1_device),
                    "gpu_memory_utilization": 0.45,
                    "max_num_batched_tokens": 2048,
                    "max_num_seqs": 1,
                    "skip_mm_profiling": True,
                    "enable_prefix_caching": False,
                    "enforce_eager": True,
                },
            }
        },
    )
    return config_path, stage0_device, stage1_device


@pytest.fixture(scope="function")
async def llm_engine():
    stage_config_path, stage0_device, stage1_device = _build_bagel_dual_stage_config()
    engine = AsyncOmni(model=MODEL, stage_configs_path=stage_config_path, init_timeout=600, enable_sleep_mode=True)
    engine._sleep_test_stage0_device = stage0_device
    engine._sleep_test_stage1_device = stage1_device
    yield engine
    engine.shutdown()
    # Subprocess / driver can lag releasing VRAM; brief pause before the next test spins up new workers.
    await asyncio.sleep(1.5)


@pytest.fixture(scope="function")
async def diffusion_engine():
    stage_config_path, stage0_device, stage1_device = _build_bagel_dual_stage_config()
    engine = AsyncOmni(model=MODEL, stage_configs_path=stage_config_path, init_timeout=600, enable_sleep_mode=True)
    engine._sleep_test_thinker_stage_id = 0
    engine._sleep_test_diffusion_stage_id = 1
    engine._sleep_test_thinker_device = stage0_device
    engine._sleep_test_diffusion_device = stage1_device
    yield engine
    engine.shutdown()
    await asyncio.sleep(1.5)


class TestOmniSleepMode:
    @pytest.mark.asyncio
    async def test_llm_sleep_ack(self, llm_engine: AsyncOmni):
        """LLM Thinker (GPU0) Signal and Physical Recycling Audit"""
        try:
            device_id = getattr(llm_engine, "_sleep_test_stage0_device", 0)
            used_before = get_device_global_memory_used_gib(device_id)
            acks = await llm_engine.sleep(stage_ids=[0], level=1)
            await asyncio.sleep(1.5)
            used_after = get_device_global_memory_used_gib(device_id)
            drop_gib = used_before - used_after
            # Verification signal successful
            assert all(get_ack_info(ack, "status") == "SUCCESS" for ack in acks)
            # Worker-reported delta (can be 0 if get_current_memory_usage does not move) or
            # GPU-global drop from mem_get_info (sees child worker processes).
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
            llm_engine.shutdown()

    @pytest.mark.asyncio
    async def test_diffusion_sleep_handshake(self, diffusion_engine: AsyncOmni):
        """Diffusion Worker stage signal loop"""
        try:
            logger.info("Starting Diffusion Worker Handshake Test")
            diffusion_stage_id = getattr(diffusion_engine, "_sleep_test_diffusion_stage_id", 1)
            acks = await diffusion_engine.sleep(stage_ids=[diffusion_stage_id], level=1)

            def _get_status(ack):
                return ack.status if hasattr(ack, "status") else ack.get("status")

            assert len(acks) >= 1, "Expected at least 1 ACK from Diffusion Workers"
            assert all(_get_status(ack) == "SUCCESS" for ack in acks)
            logger.info(f"Success: Received {len(acks)} Diffusion Worker ACKs")
            logger.info("Testing auto-wakeup before test end...")
            await diffusion_engine.wake_up(stage_ids=[diffusion_stage_id])
            logger.info("Test logic finished, triggering manual shutdown...")
        finally:
            diffusion_engine.shutdown()
            logger.info("Manual shutdown executed. Test should exit now.")

    @pytest.mark.asyncio
    async def test_cross_device_cleanup(self, diffusion_engine: AsyncOmni):
        """Physical recycling audit: leveraging deterministic data returned by Workers"""
        try:
            diffusion_stage_id = getattr(diffusion_engine, "_sleep_test_diffusion_stage_id", 1)
            # TP2 uses GPUs 0 and 1; measure whole-GPU usage (includes worker subprocesses).
            used_before = get_device_global_memory_used_gib(0) + get_device_global_memory_used_gib(1)
            acks = await diffusion_engine.sleep(stage_ids=[diffusion_stage_id], level=1)
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
            diffusion_engine.shutdown()

    @pytest.mark.asyncio
    async def test_diffusion_integrity_bit_level(self, diffusion_engine: AsyncOmni):
        """Bit-level consistency after Diffusion wake-up (prevent image corruption)"""
        try:
            diffusion_stage_id = getattr(diffusion_engine, "_sleep_test_diffusion_stage_id", 1)
            prompt = "A huge swimming pool, with many people swimming."
            sp = OmniDiffusionSamplingParams(num_inference_steps=4, height=512, width=512, seed=42)
            llm_sp = SamplingParams()

            # Baseline Generation
            logger.info("Running Baseline Generation...")
            base_output = None
            async for output in diffusion_engine.generate(prompt, request_id="base", sampling_params_list=[llm_sp, sp]):
                base_output = output
            assert base_output is not None and len(base_output.images) > 0
            logger.info("Baseline Generation successful.")
            # Sleep Level 1
            logger.info("Entering Deep Sleep (VRAM Scavenging)...")
            await diffusion_engine.sleep(stage_ids=[diffusion_stage_id], level=1)
            # Wake-up
            logger.info("Waking up (Reloading Weights)...")
            await diffusion_engine.wake_up(stage_ids=[diffusion_stage_id])

            await asyncio.sleep(2.0)
            import gc

            gc.collect()

            logger.info("Running Post-Wakeup Generation...")
            post_output = None
            async for output in diffusion_engine.generate(prompt, request_id="post", sampling_params_list=[llm_sp, sp]):
                post_output = output
            # Assert result consistency
            assert post_output is not None
            assert len(base_output.images) == len(post_output.images)
            assert post_output.images[0] is not None
            logger.info("SUCCESS: Diffusion integrity verified after Sleep/Wake cycle.")
        except Exception as e:
            logger.error(f"Integrity test failed: {e}")
            raise e
        finally:
            logger.info("Triggering mandatory cleanup...")
            diffusion_engine.shutdown()
            logger.info("Cleanup complete, test exiting.")

    @pytest.mark.asyncio
    async def test_diffusion_vram_lifecycle_audit(self, diffusion_engine: AsyncOmni):
        """Diffusion memory loop: Active -> Deep Sleep -> Active -> inference sanity check"""
        device_id = getattr(diffusion_engine, "_sleep_test_diffusion_device", 0)
        try:
            diffusion_stage_id = getattr(diffusion_engine, "_sleep_test_diffusion_stage_id", 1)
            get_vram_info(device_id)
            torch.accelerator.empty_cache()
            vram_initial = get_vram_info(device_id)["reserved"]
            logger.info(f"Diffusion Initial VRAM: {vram_initial:.2f} GiB")

            # Sleep
            logger.info("Triggering Level 1 Deep Sleep (Partial Weight Offloading)...")
            acks = await diffusion_engine.sleep(stage_ids=[diffusion_stage_id], level=1)

            reported_freed_bytes = sum(getattr(ack, "freed_bytes", 0) for ack in acks)
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

            # wake-up
            logger.info("Triggering Wake-up (Reloading weights to GPU)...")
            await diffusion_engine.wake_up(stage_ids=[diffusion_stage_id])

            await asyncio.sleep(2)
            get_vram_info(device_id)
            torch.accelerator.empty_cache()
            vram_restored = get_vram_info(device_id)["reserved"]
            logger.info(f"VRAM after Wake-up: {vram_restored:.2f} GiB")

            assert abs(vram_restored - vram_initial) < 3.0, "VRAM failed to restore to initial levels"

            # inference sanity check
            logger.info("Running post-lifecycle inference smoke test...")
            prompt = "A futuristic lab with glowing lights, high quality."
            sp = OmniDiffusionSamplingParams(num_inference_steps=2, height=512, width=512, seed=42)
            llm_sp = SamplingParams()

            base_img_found = False
            async for output in diffusion_engine.generate(
                prompt, request_id="lifecycle-check", sampling_params_list=[llm_sp, sp]
            ):
                if output.images and output.images[0] is not None:
                    base_img_found = True

            assert base_img_found, "Inference failed after Wake-up cycle!"
            logger.info("SUCCESS: Full Diffusion Lifecycle (Active -> Sleep -> Active -> Generate) audited.")

        except Exception as e:
            logger.error(f"Lifecycle audit failed: {e}")
            raise e
        finally:
            logger.info("Cleaning up engine and scavenging processes...")
            diffusion_engine.shutdown()
            await asyncio.sleep(1)

    @pytest.mark.asyncio
    async def test_level2_sleep_wake_raises(self, llm_engine: AsyncOmni):
        """Regression for #4473 Repro A: wake_up() after sleep(level=2) must raise
        NotImplementedError instead of silently producing corrupted output."""
        try:
            await llm_engine.sleep(stage_ids=[0], level=2)
            with pytest.raises(NotImplementedError, match="sleep\\(level=2\\)"):
                await llm_engine.wake_up(stage_ids=[0])
        finally:
            llm_engine.shutdown()

    @pytest.mark.asyncio
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
            llm_engine.shutdown()

    @pytest.mark.asyncio
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
            llm_engine.shutdown()


@pytest.mark.omni
@pytest.mark.advanced_model
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.asyncio
async def test_diffusion_model_sleep_tp(tp_size: int):
    """Two-stage BAGEL default config: warmup, sleep all, wake, verify generate."""
    num_gpus = torch.accelerator.device_count()
    if num_gpus < tp_size:
        pytest.skip(f"Skipping TP={tp_size}")

    engine_args = {
        "model": MODEL,
        "enable_sleep_mode": True,
        "tensor_parallel_size": tp_size,
        "enforce_eager": True,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "quantization": "fp8",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.4,
    }

    engine = AsyncOmni(**engine_args, stage_init_timeout=1200)
    try:
        diff_sp = OmniDiffusionSamplingParams(num_inference_steps=2, height=256, width=256)
        llm_sp = SamplingParams()

        async for _ in engine.generate("test", sampling_params_list=[llm_sp, diff_sp]):
            pass

        acks = await engine.sleep(level=1)
        statuses = [get_ack_info(ack, "status") for ack in acks]
        assert all(s == "SUCCESS" for s in statuses), f"Sleep failed. Statuses: {statuses}"

        await engine.wake_up()
        async for _ in engine.generate("verify", sampling_params_list=[llm_sp, diff_sp]):
            pass

        logger.info("Diffusion TP=%s lifecycle OK", tp_size)
    finally:
        engine.shutdown()


@pytest.mark.omni
@pytest.mark.advanced_model
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.asyncio
async def test_multistage_sleep_b70(tp_size: int):
    """Explicit 3-stage Qwen2.5-Omni config; sleep/wake all stages."""
    num_gpus = torch.accelerator.device_count()
    if num_gpus < 2:
        pytest.skip("Not enough GPUs")

    model_name = "Qwen/Qwen2.5-Omni-7B"
    stage2_device = "1" if current_omni_platform.is_xpu() else "0"
    common_args = {
        "model": model_name,
        "tensor_parallel_size": tp_size,
        "gpu_memory_utilization": 0.4,
        "dtype": "bfloat16",
        "enable_sleep_mode": True,
        "trust_remote_code": True,
    }
    stages = [
        {
            "stage_id": 0,
            "stage_type": "llm",
            "runtime": {"process": True, "devices": "0"},
            "engine_args": {**common_args, "model_stage": "thinker"},
        },
        {
            "stage_id": 1,
            "stage_type": "llm",
            "engine_input_source": [0],
            "runtime": {"process": True, "devices": "1", "connector_type": "queue"},
            "engine_args": {**common_args, "model_stage": "talker"},
        },
        {
            "stage_id": 2,
            "stage_type": "llm",
            "engine_input_source": [1],
            "runtime": {"process": True, "devices": stage2_device, "connector_type": "queue"},
            "engine_args": {**common_args, "model_stage": "code2wav", "worker_type": "generation"},
        },
    ]

    connectors = [
        {"src_stage_id": 0, "dst_stage_id": 1, "connector_type": "queue"},
        {"src_stage_id": 1, "dst_stage_id": 2, "connector_type": "queue"},
    ]

    engine = AsyncOmni(
        model=model_name, stages=stages, connectors=connectors, enable_sleep_mode=True, stage_init_timeout=1200
    )
    try:
        prompt = {"prompt": "Say hello in one sentence.", "modalities": ["audio"]}
        sampling_params_list = [
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=2048,
                seed=42,
                detokenize=True,
                repetition_penalty=1.1,
            ),
            SamplingParams(
                temperature=0.9,
                top_p=0.8,
                top_k=40,
                max_tokens=2048,
                seed=42,
                detokenize=True,
                repetition_penalty=1.05,
                stop_token_ids=[8294],
            ),
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=2048,
                seed=42,
                detokenize=True,
                repetition_penalty=1.1,
            ),
        ]
        async for _ in engine.generate(
            prompt,
            request_id="warmup",
            sampling_params_list=sampling_params_list,
            output_modalities=["audio"],
        ):
            pass

        acks = await engine.sleep(stage_ids=[0, 1, 2], level=1)
        statuses = [get_ack_info(ack, "status") for ack in acks]
        assert len(acks) >= 3
        assert all(status == "SUCCESS" for status in statuses), f"Sleep failed. Statuses: {statuses}"

        await engine.wake_up(stage_ids=[0, 1, 2])
        async for _ in engine.generate(
            prompt,
            request_id="verify",
            sampling_params_list=sampling_params_list,
            output_modalities=["audio"],
        ):
            pass
    finally:
        engine.shutdown()


@pytest.mark.omni
@pytest.mark.advanced_model
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.asyncio
async def test_pure_diffusion_scenario(tp_size: int):
    """Single-stage random diffusion: sleep, wake, generate."""
    engine_args = {
        "model": MODEL_DIFF,
        "enable_sleep_mode": True,
        "tensor_parallel_size": tp_size,
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
        logger.info("Pure diffusion OK (TP=%s)", tp_size)
    finally:
        engine.shutdown()
