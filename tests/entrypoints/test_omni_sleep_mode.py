# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Entrypoint sleep-mode coverage on small models plus H100 BAGEL.

Layering (AR before tiny DiT on the same L4). Thinker lives in a child
``StageEngineCoreProc`` that shutdown can reap; tiny DiT is an inline
UniProc ``DiffusionWorker`` in pytest and leftover VRAM cannot be killed
(#8016). DiT-then-7B OOMs (6.78 GiB leftover + 16.78 GiB weights > 22 GiB).
1. AR protocol (#4473) — ``Qwen/Qwen2.5-Omni-7B`` thinker-only on L4
2. Diffusion sleep/wake/generate — ``riverclouds/qwen_image_random`` on L4
3. Light multistage orchestration — thinker-only AR + tiny DiT on L4×2
4. BAGEL BagelPipeline TP=2 / coordinated dual-engine — H100, ``full_model``
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
MODEL_BAGEL = "ByteDance-Seed/BAGEL-7B-MoT"
# Thinker weights are 16.78 GiB. On L4 (22 GiB) util 0.45 only budgets 9.9 GiB
# so device-level KV profiling reports 0.0 GiB and StageEngineCoreProc dies.
# Keep abort-test util 0.90; cap len/batch at 2048 so the leftover ~3 GiB KV fits.
_AR_SLEEP_GPU_MEMORY_UTILIZATION = 0.90
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
_MULTISTAGE_SLEEP_PIPELINE = "qwen2_5_omni_thinker_tiny_dit"


def _tiny_dit_multistage_deploy_config() -> str:
    """Thinker-only AR + tiny DiT deploy for joint sleep/wake.

    ``AsyncOmni(stages=...)`` is ignored by ``resolve_omni_config``: a
    ``model=Qwen2.5-Omni-7B`` call otherwise loads the native 3-stage
    thinker/talker/code2wav pipeline (``Expected 3 sampling params, got 2``).
    """
    import atexit
    import os
    import tempfile
    from pathlib import Path

    import yaml

    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, register_pipeline
    from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

    if _MULTISTAGE_SLEEP_PIPELINE not in OMNI_PIPELINES:
        register_pipeline(
            PipelineConfig(
                model_type=_MULTISTAGE_SLEEP_PIPELINE,
                hf_architectures=(),
                stages=(
                    StagePipelineConfig(
                        stage_id=0,
                        model_stage="thinker",
                        execution_type=StageExecutionType.LLM_AR,
                        input_sources=(),
                        owns_tokenizer=True,
                        requires_multimodal_data=True,
                        hf_config_name="thinker_config",
                        model_arch="Qwen2_5OmniForConditionalGeneration",
                        engine_output_type="latent",
                        sampling_constraints={"detokenize": True},
                    ),
                    StagePipelineConfig(
                        stage_id=1,
                        model_stage="dit",
                        execution_type=StageExecutionType.DIFFUSION,
                        input_sources=(0,),
                        final_output=True,
                        final_output_type="image",
                    ),
                ),
            )
        )

    config = {
        "pipeline": _MULTISTAGE_SLEEP_PIPELINE,
        "async_chunk": False,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "connectors": {
            "shared_memory_connector": {"name": "SharedMemoryConnector"},
        },
        "stages": [
            {
                "stage_id": 0,
                "devices": "0",
                "enable_sleep_mode": True,
                "gpu_memory_utilization": _AR_SLEEP_GPU_MEMORY_UTILIZATION,
                "enforce_eager": True,
                "max_model_len": _AR_SLEEP_MAX_MODEL_LEN,
                "max_num_batched_tokens": _AR_SLEEP_MAX_NUM_BATCHED_TOKENS,
                "skip_mm_profiling": True,
                "mm_processor_cache_gb": 0,
                "max_num_seqs": 1,
            },
            {
                "stage_id": 1,
                "devices": "1",
                "model": MODEL_DIFF,
                "enable_sleep_mode": True,
                "gpu_memory_utilization": 0.4,
                "enforce_eager": True,
                "tensor_parallel_size": 1,
                "max_num_seqs": 1,
                "input_connectors": {"from_stage_0": "shared_memory_connector"},
            },
        ],
    }
    fd, path = tempfile.mkstemp(prefix="sleep_thinker_tiny_dit_", suffix=".yaml")
    atexit.register(Path(path).unlink, missing_ok=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


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


_ENGINE_CHILD_MARKERS = (
    "stagediffusionproc",
    "enginecore",
    "diffusionworker",
    "vllm::worker",
    "vllm-omni:",
)


def _is_engine_worker_proc(proc) -> bool:
    blob = " ".join([proc.name(), *proc.cmdline()]).lower()
    return any(marker in blob for marker in _ENGINE_CHILD_MARKERS)


def _snapshot_engine_child_pids() -> list[int]:
    """PIDs of engine workers under this pytest process (before shutdown).

    ``DiffusionWorker`` is started ``daemon=True``. After StagePool joins
    ``StageDiffusionProc``, that worker is reparented to PID 1 and a later
    ``children(recursive=True)`` scan misses it while it still holds VRAM.
    """
    try:
        import psutil
    except ImportError:
        return []
    try:
        children = psutil.Process().children(recursive=True)
    except psutil.Error:
        return []
    pids: list[int] = []
    for proc in children:
        try:
            if _is_engine_worker_proc(proc):
                pids.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def _reap_engine_pids(pids: list[int]) -> None:
    """SIGKILL snapshot PIDs even if they were reparented after shutdown."""
    try:
        import psutil
    except ImportError:
        return

    procs = []
    for pid in pids:
        try:
            procs.append(psutil.Process(pid))
        except psutil.NoSuchProcess:
            continue
    print(f"[sleep_mode] engine worker snapshot pids={pids} still_alive={[p.pid for p in procs]}")
    if not procs:
        return
    # shutdown() already tried a graceful exit. SIGTERM + 5s here just
    # delays SIGKILL on CUDA workers that ignore it.
    for proc in procs:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    psutil.wait_procs(procs, timeout=3)


async def _shutdown_engine_and_clear_gpu(engine: AsyncOmni) -> None:
    """Shut down the engine, reap leftover workers, then run shared cleanup."""
    from tests.helpers.clean import cleanup_test_environment

    leftover_pids = _snapshot_engine_child_pids()
    engine.shutdown()
    _reap_engine_pids(leftover_pids)
    cleanup_test_environment()


@pytest.fixture(scope="module", autouse=True)
def _module_device_cleanup():
    from tests.helpers.clean import cleanup_test_environment

    print("\n=== PRE-MODULE DEVICE CLEANUP (sleep_mode) ===")
    cleanup_test_environment()
    yield
    # Fixture / test teardown already ran cleanup_test_environment(); a second
    # POST wait just repeats the same 5%×60s on leftover worker VRAM.


# ---------------------------------------------------------------------------
# 1) AR protocol — Omni thinker-only (L4)
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
# 2) Diffusion sleep/wake/generate — qwen_image_random (L4)
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
            assert all(get_ack_info(ack, "status") == "SUCCESS" for ack in acks)
            await diffusion_engine.wake_up()
            await diffusion_engine.resume_generation()
            output = None
            async for item in diffusion_engine.generate(
                "test",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2, height=256, width=256),
            ):
                output = item
            assert output is not None, "generate after sleep/wake produced no output"
            assert output.images and output.images[0] is not None, "generate after sleep/wake produced no image"
        finally:
            await _ensure_awake(diffusion_engine)


# ---------------------------------------------------------------------------
# 3) Light multistage — small AR + small DiT (L4×2)
# ---------------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=2)
@pytest.mark.asyncio
async def test_multistage_ar_diffusion_sleep_wake():
    """Orchestration: generate → joint sleep/wake/resume → generate.

    Covers the L4 path that ``TestBagelCoordinatedSleepMode`` (still skipped)
    and BAGEL TP=2 (diffusion-only, ``full_model``) do not: a 2-step tiny-DiT
    generate after ``sleep(stage_ids=[0, 1])`` + ``resume_generation``.
    """
    engine = AsyncOmni(
        model=MODEL_AR,
        deploy_config=_tiny_dit_multistage_deploy_config(),
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
        await _tiny_dit_multistage_generate(engine, "verify", "verify")
        logger.info("Light multistage joint sleep/wake/resume generate OK")
    finally:
        await _shutdown_engine_and_clear_gpu(engine)


# ---------------------------------------------------------------------------
# 4) BAGEL — H100 ``full_model`` (nightly Entrypoints Test)
# ---------------------------------------------------------------------------


def _get_device_global_memory_used_gib(device_id: int) -> float:
    """GPU-wide memory in use (GiB), including all processes (driver view).

    Fail closed: a swallowed query that returned 0.0 used to inflate
    ``drop_gib`` and false-pass VRAM assertions.
    """
    with current_omni_platform.device(device_id):
        current_omni_platform.synchronize()
        free_b, total_b = current_omni_platform.get_device_memory()
    return (total_b - free_b) / 1024**3


def _bagel_diffusion_deploy_config() -> str:
    """Official single-stage BAGEL DiT, TP=2 on cards 0,1.

    ``AsyncOmni(stages=...)`` is ignored and leaked as an unknown diffusion
    field (``Unknown diffusion config field(s) for stage 1: 'stages'``).
    """
    return modify_stage_config(
        get_deploy_config_path("bagel_single_stage.yaml"),
        updates={
            "stages": {
                0: {
                    "devices": "0,1",
                    "tensor_parallel_size": 2,
                    "enable_sleep_mode": True,
                    "enforce_eager": True,
                    "dtype": "bfloat16",
                    "gpu_memory_utilization": 0.4,
                }
            }
        },
    )


@pytest_asyncio.fixture(scope="class", loop_scope="class")
async def bagel_diffusion_engine():
    """Shared BAGEL single-stage DiT TP=2 engine for sleep/wake + generate."""
    engine = AsyncOmni(
        model=MODEL_BAGEL,
        deploy_config=_bagel_diffusion_deploy_config(),
        init_timeout=600,
        enable_sleep_mode=True,
    )
    yield engine
    await _shutdown_engine_and_clear_gpu(engine)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
class TestBagelDiffusionSleepMode:
    """BAGEL diffusion sleep/wake on a class-scoped TP=2 BagelPipeline."""

    @pytest.mark.asyncio(loop_scope="class")
    async def test_diffusion_sleep_handshake(self, bagel_diffusion_engine: AsyncOmni):
        try:
            acks = await bagel_diffusion_engine.sleep(stage_ids=[0], level=1)
            assert len(acks) >= 1
            assert all(get_ack_info(ack, "status") == "SUCCESS" for ack in acks)
            await bagel_diffusion_engine.wake_up(stage_ids=[0])
        finally:
            await _ensure_awake(bagel_diffusion_engine, [0])

    @pytest.mark.asyncio(loop_scope="class")
    async def test_cross_device_cleanup(self, bagel_diffusion_engine: AsyncOmni):
        try:
            used_before = _get_device_global_memory_used_gib(0) + _get_device_global_memory_used_gib(1)
            acks = await bagel_diffusion_engine.sleep(stage_ids=[0], level=1)
            await asyncio.sleep(1.5)
            used_after = _get_device_global_memory_used_gib(0) + _get_device_global_memory_used_gib(1)
            drop_gib = used_before - used_after
            freed_gb = sum(get_ack_info(ack, "freed_bytes", 0) for ack in acks) / 1024**3
            assert freed_gb > 14.0 or drop_gib > 8.0, f"ACK={freed_gb:.2f} GiB, global_drop={drop_gib:.2f} GiB"
        finally:
            await _ensure_awake(bagel_diffusion_engine, [0])

    @pytest.mark.asyncio(loop_scope="class")
    async def test_diffusion_sleep_wake_generate(self, bagel_diffusion_engine: AsyncOmni):
        import gc

        device_id = 1
        try:
            prompt = "A huge swimming pool, with many people swimming."
            sp = OmniDiffusionSamplingParams(num_inference_steps=4, height=512, width=512, seed=42)

            base_output = None
            async for output in bagel_diffusion_engine.generate(prompt, request_id="base", sampling_params=sp):
                base_output = output
            assert base_output is not None and len(base_output.images) > 0

            current_omni_platform.empty_cache()
            vram_initial = _get_device_global_memory_used_gib(device_id)

            acks = await bagel_diffusion_engine.sleep(stage_ids=[0], level=1)
            statuses = [get_ack_info(ack, "status") for ack in acks]
            assert all(s == "SUCCESS" for s in statuses), f"Sleep failed. Statuses: {statuses}"

            reported_freed_gib = sum(get_ack_info(ack, "freed_bytes", 0) for ack in acks) / 1024**3
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
            async for output in bagel_diffusion_engine.generate(prompt, request_id="post", sampling_params=sp):
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


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
class TestBagelCoordinatedSleepMode:
    """Dual-engine coordination (kept skipped; do not delete)."""

    @pytest.mark.skip(
        reason=(
            "Flaky/CI: dual AsyncOmni can fail with "
            "RuntimeError: Orchestrator init failed, StageDiffusionProc died during handshake. "
            "Re-enable when stable (no OOM on coordinated talker+diffusion)."
        )
    )
    @pytest.mark.asyncio(loop_scope="class")
    async def test_coordinated_cross_device(self):
        """Heterogeneous coordinated cleanup (talker + diffusion on GPU 1)."""
        llm_stages, llm_connectors = _build_bagel_llm_stages()
        llm_engine = AsyncOmni(
            model=MODEL_BAGEL, stages=llm_stages, connectors=llm_connectors, init_timeout=600, enable_sleep_mode=True
        )
        diffusion_engine = AsyncOmni(
            model=MODEL_BAGEL,
            deploy_config=_bagel_diffusion_deploy_config(),
            init_timeout=600,
            enable_sleep_mode=True,
        )
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
            await _shutdown_engine_and_clear_gpu(llm_engine)
            await _shutdown_engine_and_clear_gpu(diffusion_engine)
