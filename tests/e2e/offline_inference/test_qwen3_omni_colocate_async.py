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
from contextlib import ExitStack

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from tests.helpers.clean import wait_for_gpu_memory_to_clear
from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
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
    return modify_stage_config(
        get_deploy_config_path("qwen3_omni_moe_thinking.yaml"),
        updates={
            "stages": {
                0: {
                    "enable_sleep_mode": True,
                    "enforce_eager": True,
                    "trust_remote_code": True,
                    "tensor_parallel_size": 1,
                    "devices": "0",
                    "gpu_memory_utilization": 0.9,
                    "max_num_seqs": 1,
                    "max_model_len": 8192,
                    "max_num_batched_tokens": 8192,
                    "enable_prefix_caching": False,
                    "skip_mm_profiling": True,
                },
            },
        },
    )


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
