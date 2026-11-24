# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import os
import random
from contextlib import aclosing
from pathlib import Path

from vllm import SamplingParams
from vllm.logger import init_logger

from vllm_omni.platforms.npu.worker.prefill_graph import THINKER_STARTUP_LENGTHS, THINKER_STARTUP_PREFIX

logger = init_logger(__name__)


async def precapture_thinker_prefill(engine):
    """Capture and verify bounded single-row prompt lengths before HTTP serving."""
    marker = Path(os.environ["MINICPMO_NPU_THINKER_PREFILL_STARTUP"])
    if "MiniCPMO45" not in str(engine.model_config.architectures):
        raise ValueError("Thinker prefill precapture requires MiniCPM-o 4.5")
    tokenizer = await engine.get_tokenizer()
    # Exercise varying hidden states without service traffic or benchmark prompts.
    tokens = tokenizer.encode(
        "A harbor pilot watches the wind turn while crews check the channel markers and write their notes.",
        add_special_tokens=False,
    )
    randomizer = random.Random(19)
    marker.touch(exist_ok=False)
    try:
        lengths = list(THINKER_STARTUP_LENGTHS) + [128, 144, 160]
        for index, length in enumerate(lengths):
            prompt = {"prompt_token_ids": randomizer.choices(tokens, k=length)}
            request_id = f"{THINKER_STARTUP_PREFIX}{index}"
            stream = engine.generate(
                prompt=prompt,
                sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
                request_id=request_id,
                output_modalities=["text"],
            )
            async with aclosing(stream) as outputs:
                async with asyncio.timeout(30):
                    async for _ in outputs:
                        pass
    finally:
        marker.unlink()
    logger.info("[thinker-prefill-startup] finished synthetic prefill requests")
