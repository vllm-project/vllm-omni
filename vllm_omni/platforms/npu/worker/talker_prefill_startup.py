# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import os
import random
from contextlib import aclosing
from pathlib import Path

import torch
from vllm import SamplingParams
from vllm.logger import init_logger

from vllm_omni.model_executor.models.minicpmo_4_5.handoff import pack_handoff_tensor
from vllm_omni.platforms.npu.worker.prefill_graph import TALKER_STARTUP_LENGTHS, TALKER_STARTUP_PREFIX

logger = init_logger(__name__)


async def precapture_talker_prefill(engine):
    """Qualify both Talker replicas on synthetic handoffs before HTTP serving."""
    marker = Path(os.environ["MINICPMO_NPU_TALKER_PREFILL_STARTUP"])
    if "MiniCPMO45" not in str(engine.model_config.architectures):
        raise ValueError("Talker prefill precapture requires MiniCPM-o 4.5")
    tts_config = engine.model_config.hf_config.tts_config
    llm_dim = int(tts_config["llm_dim"] if isinstance(tts_config, dict) else tts_config.llm_dim)
    tokenizer = await engine.get_tokenizer()
    token_pool = tokenizer.encode(
        "The station reports a quiet afternoon as the boats return with their crew and supplies.",
        add_special_tokens=False,
    )
    rnd = random.Random(23)
    generator = torch.Generator(device="cpu").manual_seed(23)
    params = list(engine.default_sampling_params_list)
    params[1] = SamplingParams(temperature=0.0, min_tokens=0, max_tokens=16, detokenize=False)
    marker.touch(exist_ok=False)
    try:
        lengths = list(TALKER_STARTUP_LENGTHS) + list(reversed(TALKER_STARTUP_LENGTHS))
        for index, length in enumerate(lengths):
            for copy in range(2):
                prompt = {
                    "prompt_token_ids": [0] * length,
                    "model_intermediate_buffer": {
                        "ids": {"tts": rnd.choices(token_pool, k=length - 2)},
                        "hidden_states": {
                            "tts": pack_handoff_tensor(
                                torch.randn(length - 2, llm_dim, generator=generator).mul_(0.125)
                            )
                        },
                    },
                }
                request_id = f"{TALKER_STARTUP_PREFIX}{index}-{copy}"
                stream = engine.generate(
                    prompt=prompt,
                    sampling_params_list=params,
                    request_id=request_id,
                    output_modalities=["audio"],
                    entry_stage_id=1,
                )
                async with aclosing(stream) as outputs:
                    async with asyncio.timeout(30):
                        async for _ in outputs:
                            pass
    finally:
        marker.unlink()
    logger.info("[talker-prefill-startup] finished synthetic handoffs")
