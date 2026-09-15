# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sequential generation of complete, non-streaming speech segments."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.entrypoints.openai.speech_usage import SpeechOutputTokenCounter
from vllm_omni.inputs.data import OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.entrypoints.async_omni import AsyncOmni


async def generate_speech_segments(
    engine_client: AsyncOmni,
    *,
    prompts: list[dict[str, Any]],
    request_id: str,
    sampling_params_list: list[OmniSamplingParams],
    silence_ms: int,
    extract_audio: Callable[[OmniRequestOutput], tuple[dict | None, str | None]],
    arrival_time: float | None = None,
) -> AsyncIterator[OmniRequestOutput]:
    """Yield one logical result after every segment has finished successfully.

    Each prompt gets an independent generation budget and a copy of sampling
    parameters (including the seed). Sum per-segment codec usage; intermediate
    cumulative snapshots must not duplicate either audio or token counts.
    The caller encodes the concatenated mono waveform once, in the requested
    response format. Streaming and timestamp aggregation are not supported.
    """
    if not prompts or silence_ms < 0:
        raise ValueError("Segmented speech requires prompts and a nonnegative silence interval")
    waves: list[torch.Tensor] = []
    sample_rate: int | None = None
    output_tokens = 0
    for index, prompt in enumerate(prompts):
        segment_id = f"{request_id}-segment-{index}"
        complete = False
        audio_output = None
        audio_key = None
        counter = SpeechOutputTokenCounter()
        generator = engine_client.generate(
            prompt=prompt,
            request_id=segment_id,
            # Engine preprocessing may mutate parameters. Use each parameter
            # type's cloning contract so later segments keep the original seed.
            sampling_params_list=[params.clone() for params in sampling_params_list],
            output_modalities=["audio"],
            arrival_time=arrival_time,
        )
        try:
            finished = False
            async for result in generator:
                if result.error:
                    raise ValueError(f"Speech segment {index + 1} failed: {result.error}")
                counter.observe(result)
                finished = result.finished
                candidate, key = extract_audio(result)
                if key is not None:
                    audio_output, audio_key = candidate, key
            if not finished or audio_output is None or audio_key is None:
                raise ValueError(f"Speech segment {index + 1} did not produce complete audio")
            if counter.stage0_finish_reason == "length":
                raise ValueError(f"Speech segment {index + 1} reached its generation token limit")
            audio = audio_output[audio_key]
            # Non-streaming history lists contain cumulative waveforms. Retain
            # only the latest nonempty snapshot, as the regular serving path does.
            if isinstance(audio, list):
                audio = next((item for item in reversed(audio) if torch.as_tensor(item).numel()), [])
            wave = torch.as_tensor(audio).detach().to(device="cpu", dtype=torch.float32).squeeze()
            if wave.ndim == 0:
                wave = wave.reshape(1)
            if wave.ndim != 1 or not wave.numel() or not bool(torch.isfinite(wave).all()):
                raise ValueError(f"Speech segment {index + 1} did not produce finite nonempty mono audio")
            sr = audio_output.get("sr")
            if isinstance(sr, list):
                sr = sr[-1] if sr else None
            if sr is None or int(sr) <= 0:
                raise ValueError(f"Speech segment {index + 1} has no valid sample rate")
            if sample_rate is not None and int(sr) != sample_rate:
                raise ValueError("Speech segments have inconsistent sample rates")
            sample_rate = int(sr)
            waves.append(wave)
            output_tokens += counter.total()
            complete = True
        finally:
            if not complete:
                await engine_client.abort(segment_id)
            await generator.aclose()

    assert sample_rate is not None
    silence = torch.zeros(round(sample_rate * silence_ms / 1000), dtype=torch.float32)
    pieces: list[torch.Tensor] = []
    for wave in waves:
        if pieces:
            pieces.append(silence)
        pieces.append(wave)
    yield OmniRequestOutput(
        request_id=request_id,
        finished=True,
        final_output_type="audio",
        _multimodal_output={"audio": torch.cat(pieces), "sr": sample_rate},
        # These are aggregate usage counts, not the last segment's timing data.
        metrics={"stage_metrics": {"0": {"num_tokens_out": output_tokens}}},
    )
