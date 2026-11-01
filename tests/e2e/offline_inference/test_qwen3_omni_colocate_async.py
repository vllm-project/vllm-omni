# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Isolated AsyncOmni abort / sleep-admission regression for Qwen3-Omni.

This lives in its own module so the L3 merge job can run it *after*
``test_qwen3_omni.py`` (offline) and the online serving file. Those modules keep
module-scoped OmniRunner / OmniServer instances until teardown; overlapping a
CuMem sleep-mode engine with them kills StageEngineCoreProc at handshake
(``Failed core proc(s): {}``).
"""

import asyncio
import os
from contextlib import ExitStack, suppress

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from tests.helpers.clean import wait_for_gpu_memory_to_clear
from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.platforms import current_omni_platform

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def _output_token_ids(output) -> list[int]:
    if not getattr(output, "outputs", None):
        return []
    completion = output.outputs[0]
    token_ids = list(completion.token_ids or [])
    if token_ids:
        return token_ids
    return list(getattr(completion, "cumulative_token_ids", None) or [])


def _finish_reason(output) -> str | None:
    if not getattr(output, "outputs", None):
        return None
    return getattr(output.outputs[0], "finish_reason", None)


def _colocate_async_deploy() -> str:
    """Thinker-only Instruct deploy with sleep mode on a single GPU.

    Abort / sleep admission is a thinker control-plane contract. The 3-stage
    CI yaml plus CuMem on every stage fails engine-core startup in the L3
    merge job: Stage 1 resolves ``devices: "1"`` while only GPU 0 is visible.
    """
    return get_deploy_config_path("ci/qwen3_omni_moe_colocate_async.yaml")


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.usefixtures("clean_gpu_memory_between_tests")
@pytest.mark.asyncio
async def test_colocate_async_abort_tokens_and_sleep_admission() -> None:
    """**required** regression. Control-plane APIs (``abort`` / ``sleep`` /
    ``wake_up`` / ``resume_generation``) are not in ``send_omni_request``.

    On ``origin/main`` this test fails:
    - abort: ``generate()`` never yields ``finish_reason="abort"`` with the
      tokens produced so far, so resume has an empty prefix (or the stream
      hangs after frontend state is dropped).
    - sleep: ``generate()`` is admitted into EngineCore while weights/KV are
      offloaded, so the task errors (asleep / corrupted ADD frame) instead of
      waiting for ``resume_generation()``.

    On this PR both legs pass: abort returns a resumeable prefix, and sleep
    holds ``generate()`` until wake + resume.
    """
    prompt = "What color is the sky? Write a long, detailed explanation."
    device_count = current_omni_platform.device_count()
    if device_count > 0:
        # Fail closed if a prior module-scoped runner is still on the GPU.
        # ``clean_gpu_memory_between_tests`` only logs a note on timeout.
        wait_for_gpu_memory_to_clear(
            devices=list(range(device_count)),
            threshold_ratio=0.15,
            timeout_s=120,
        )

    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            deploy_config=_colocate_async_deploy(),
            enable_sleep_mode=True,
        )
        after.callback(engine.shutdown)

        request_id = "qwen3-abort-partial"
        outputs: list = []

        async def _generate(max_tokens: int, req_id: str, gen_prompt) -> None:
            async for output in engine.generate(
                prompt=gen_prompt,
                request_id=req_id,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
                output_modalities=["text"],
            ):
                outputs.append(output)

        abort_task = asyncio.create_task(_generate(256, request_id, prompt))
        prefix: list[int] = []
        prompt_token_ids: list[int] = []
        for _ in range(600):
            if abort_task.done():
                break
            if outputs:
                latest = outputs[-1]
                prefix = _output_token_ids(latest)
                prompt_token_ids = list(getattr(latest, "prompt_token_ids", None) or [])
                if prefix:
                    break
            await asyncio.sleep(0.1)

        assert prefix, "generate produced no tokens before abort"
        assert not abort_task.done(), "generate finished before abort; raise max_tokens"
        await engine.abort(request_id)
        await asyncio.wait_for(abort_task, timeout=90)

        final = outputs[-1]
        assert final.finished
        assert _finish_reason(final) == "abort"
        abort_prefix = _output_token_ids(final)
        assert abort_prefix, "abort dropped the generated prefix (main-branch behavior)"

        resume_prompt: str | TokensPrompt = (
            TokensPrompt(prompt_token_ids=prompt_token_ids + abort_prefix) if prompt_token_ids else prompt
        )
        outputs.clear()
        async for output in engine.generate(
            prompt=resume_prompt,
            request_id="qwen3-abort-resume",
            sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
            output_modalities=["text"],
        ):
            outputs.append(output)
        assert outputs
        assert any(_output_token_ids(out) for out in outputs)

        # Trainer order is pause → abort → sleep. Pause here after the
        # resume generate so EngineCore is idle before CuMem offload.
        await engine.pause_generation(mode="abort", clear_cache=True)
        await engine.sleep(level=1)
        outputs.clear()
        sleep_task = asyncio.create_task(_generate(8, "qwen3-sleep-admission", prompt))
        await asyncio.sleep(1.0)
        assert not sleep_task.done(), "generate() ran while EngineCore was sleeping (main-branch admission race)"
        await engine.wake_up()
        await asyncio.sleep(0.5)
        assert not sleep_task.done(), "generate() resumed before resume_generation()"
        await engine.resume_generation()
        await asyncio.wait_for(sleep_task, timeout=180)
        assert outputs
        assert _finish_reason(outputs[-1]) != "abort"
    await asyncio.sleep(5)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.usefixtures("clean_gpu_memory_between_tests")
@pytest.mark.asyncio
async def test_colocate_async_resubmit_window_pause_sleep_cycle() -> None:
    """#8678 e2e: the abort -> resubmit -> pause -> sleep RL cycle must not wedge.

    Reproduces the issue's trigger on a real engine: an in-flight request is
    aborted, the abort-retry client resubmits it (the documented ~1 s retry)
    while frontend admission is still open, and only then do
    ``pause_generation(mode="abort")`` and ``sleep(level=1, mode="abort")``
    land against the still-draining intake. The wait_for timeouts bound
    ordinary (loop-alive) stalls only.

    On ``origin/main`` this test fails at the partial-wake step: after
    ``wake_up(tags=["weights"])`` + ``resume_generation()`` the resubmitted
    ``generate()`` raises ``Generation rejected: Engine is partially or fully
    asleep`` instead of waiting for the full wake (the reporter measured ~20 s
    of ~1 s-interval retry churn per training step). This test owns that
    partial-wake coverage; it does not cover the full-queue freeze legs:
    the two resubmissions cannot saturate the intake queue, and if the
    event loop itself wedges, every in-loop ``wait_for`` timer freezes with
    it. That freeze is covered deterministically by
    ``test_add_request_async_survives_saturated_queue_on_orchestrator_loop``
    in tests/entrypoints/test_async_omni_pause_sleep_routing.py, whose
    watchdog enforces the deadline outside the blocked loop.
    """
    prompt = "What color is the sky? Write a long, detailed explanation."
    device_count = current_omni_platform.device_count()
    if device_count > 0:
        wait_for_gpu_memory_to_clear(
            devices=list(range(device_count)),
            threshold_ratio=0.15,
            timeout_s=120,
        )

    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            deploy_config=_colocate_async_deploy(),
            enable_sleep_mode=True,
        )
        after.callback(engine.shutdown)

        async def _generate(sink: list, max_tokens: int, req_id: str, gen_prompt) -> None:
            async for output in engine.generate(
                prompt=gen_prompt,
                request_id=req_id,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
                output_modalities=["text"],
            ):
                sink.append(output)

        # 1) In-flight rollout request (the CI overlay caps max_num_seqs=1).
        first_outputs: list = []
        first_task = asyncio.create_task(_generate(first_outputs, 256, "qwen3-8678-first", prompt))
        for _ in range(600):
            if first_task.done():
                break
            if first_outputs and _output_token_ids(first_outputs[-1]):
                break
            await asyncio.sleep(0.1)
        assert first_outputs and _output_token_ids(first_outputs[-1]), "generate produced no tokens before abort"
        prompt_token_ids = list(getattr(first_outputs[-1], "prompt_token_ids", None) or [])
        assert not first_task.done(), "generate finished before abort; raise max_tokens"

        # 2) Trainer abort; the abort-retry client keeps the partial tokens.
        await engine.abort("qwen3-8678-first")
        await asyncio.wait_for(first_task, timeout=90)
        final = first_outputs[-1]
        assert final.finished
        assert _finish_reason(final) == "abort"
        abort_prefix = _output_token_ids(final)
        assert abort_prefix, "abort dropped the generated prefix"

        # 3) Client resubmits ~1 s later — before pause_generation() closes
        #    admission. This is the issue's race window: the resubmission is
        #    admitted into the orchestrator intake while a fresh request
        #    (still-being-admitted traffic in the repro) queues behind it.
        resume_prompt: str | TokensPrompt = (
            TokensPrompt(prompt_token_ids=prompt_token_ids + abort_prefix) if prompt_token_ids else prompt
        )
        resubmit_outputs: list = []
        fresh_outputs: list = []
        resubmit_task = asyncio.create_task(_generate(resubmit_outputs, 256, "qwen3-8678-resubmit", resume_prompt))
        fresh_task = asyncio.create_task(_generate(fresh_outputs, 256, "qwen3-8678-fresh", prompt))
        await asyncio.sleep(1.0)

        # 4) Pause and sleep against the still-draining intake — both calls
        #    are #8678 deadlock guards. On the unfixed frontend a full
        #    request queue froze the event loop here and sleep() never
        #    returned.
        await asyncio.wait_for(engine.pause_generation(mode="abort", clear_cache=True), timeout=120)
        await asyncio.wait_for(engine.sleep(level=1, mode="abort"), timeout=180)
        assert await engine.is_sleeping()

        # 5) Partial wake (weights only, kv_cache still asleep) + resume:
        #    the next resubmission must PARK at the admission gate instead of
        #    erroring — the deterministic main-branch failure.
        await engine.wake_up(tags=["weights"])
        await engine.resume_generation()
        parked_outputs: list = []
        parked_task = asyncio.create_task(_generate(parked_outputs, 8, "qwen3-8678-after-partial-wake", prompt))
        await asyncio.sleep(1.0)
        assert not parked_task.done(), (
            "generate() after wake_up(tags=['weights']) should wait for the kv_cache wake, "
            "not error with 'partially or fully asleep' (main-branch retry churn)"
        )

        # 6) Full wake releases the parked generate; the in-window
        #    resubmissions must also terminate — abort-finished if the pause
        #    killed them, completed if they were parked at the gate.
        await engine.wake_up(tags=["kv_cache"])
        await asyncio.wait_for(parked_task, timeout=180)
        assert parked_outputs
        assert any(_output_token_ids(out) for out in parked_outputs)
        assert _finish_reason(parked_outputs[-1]) != "abort"

        results = await asyncio.wait_for(
            asyncio.gather(resubmit_task, fresh_task, return_exceptions=True),
            timeout=240,
        )
        for sink, result in zip((resubmit_outputs, fresh_outputs), results):
            assert not isinstance(result, BaseException), f"in-window resubmission failed: {result!r}"
            assert sink, "resubmission produced no outputs"
            assert sink[-1].finished

        with suppress(Exception):  # noqa: BLE001 - best-effort cleanup for the shared GPU
            await engine.wake_up()
            await engine.resume_generation()
    await asyncio.sleep(5)
