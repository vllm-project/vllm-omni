import asyncio
import os
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest
from vllm import SamplingParams
from vllm.inputs import PromptType

from tests.utils import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

SEED = 42

stage_config = str(Path(__file__).parent.parent / "e2e" / "stage_configs" / "qwen3_omni_thinker_ci.yaml")
model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


async def generate(
    engine: AsyncOmni,
    request_id: str,
    prompt: PromptType,
    max_tokens: int,
) -> tuple[int, str]:
    # Ensure generate doesn't complete too fast for cancellation test.
    await asyncio.sleep(0.2)
    thinker_sampling_params = SamplingParams(
        temperature=0.4,  # Deterministic
        top_p=0.9,
        top_k=1,
        max_tokens=max_tokens,
        repetition_penalty=1.05,
        stop_token_ids=[151645],  # Qwen EOS token <|im_end|>
        seed=SEED,
    )

    sampling_params_list = [
        thinker_sampling_params,
    ]
    count = 0
    async for omni_output in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=sampling_params_list,
        output_modalities=["text"],
    ):
        stage_id = omni_output.stage_id
        out = omni_output.request_output
        if stage_id == 0:
            num_tokens = sum(len(output.token_ids) for output in out.outputs)
            count = num_tokens

        await asyncio.sleep(0.0)

    return count, request_id


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.asyncio
async def test_abort():
    with ExitStack() as after:
        # Avoid SHM IPC in tests to prevent /dev/shm exhaustion and SIGBUS.
        engine = AsyncOmni(
            model=model,
            stage_configs_path=stage_config,
            shm_threshold_bytes=sys.maxsize,
        )
        after.callback(engine.shutdown)

        # Keep token counts modest to reduce flakiness on slow test hardware.
        NUM_REQUESTS = 3
        NUM_EXPECTED_TOKENS = 64
        NUM_EXPECTED_TOKENS_LONG = 256
        REQUEST_IDS_TO_ABORT = [1]

        prompt = "Hello my name is Robert and "

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests.
        tasks: list[asyncio.Task] = []
        for idx, request_id in enumerate(request_ids):
            max_tokens = NUM_EXPECTED_TOKENS_LONG if (idx in REQUEST_IDS_TO_ABORT) else NUM_EXPECTED_TOKENS
            tasks.append(asyncio.create_task(generate(engine, request_id, prompt, max_tokens)))

        # API server cancels requests when they disconnect.
        # Explicitly abort in the engine to avoid orphaned requests hanging.
        for idx in REQUEST_IDS_TO_ABORT:
            tasks[idx].cancel()
            await engine.abort(request_ids[idx])
            await asyncio.sleep(0.1)

        # Confirm the other requests are okay.
        for idx, task in enumerate(tasks):
            # Confirm that it was actually canceled.
            if idx in REQUEST_IDS_TO_ABORT:
                with pytest.raises((asyncio.CancelledError, GeneratorExit)):
                    await asyncio.wait_for(task, timeout=60)
            else:
                # Otherwise, make sure the request was not impacted.
                num_generated_tokens, request_id = await asyncio.wait_for(task, timeout=180)
                expected_tokens = NUM_EXPECTED_TOKENS
                assert num_generated_tokens == expected_tokens, (
                    f"{request_id} generated {num_generated_tokens} but expected {expected_tokens}"
                )

        # Confirm we can do another generation.
        request_id = f"request-{REQUEST_IDS_TO_ABORT[0]}"
        task = asyncio.create_task(generate(engine, request_id, prompt, NUM_EXPECTED_TOKENS))
        num_generated_tokens, request_id = await task
        assert num_generated_tokens == NUM_EXPECTED_TOKENS
    await asyncio.sleep(5)
