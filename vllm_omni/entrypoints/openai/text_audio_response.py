# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Combine paired stage outputs for pipelines with one spoken text per choice."""

from collections import defaultdict

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponseChoice


def combine_text_audio_choices(
    choices: list[ChatCompletionResponseChoice],
    spoken_text: dict[int, str],
) -> list[ChatCompletionResponseChoice]:
    """Pair by completion index, never by list order or across requests.

    ``spoken_text`` is the actual completed Thinker text passed to the Talker,
    before response-only echo/parsing. It is intended speech text, not an ASR
    measurement of the waveform. Preserve a transcript supplied by the audio
    stage. Unexpected duplicate outputs remain visible rather than being lost.
    """
    grouped = defaultdict(list)
    for choice in choices:
        grouped[choice.index].append(choice)
    result = []
    for index, group in grouped.items():
        audio = [choice for choice in group if choice.message.audio is not None]
        text = [choice for choice in group if choice.message.audio is None]
        if len(group) == 2 and len(audio) == len(text) == 1:
            combined = text[0].model_copy(deep=True)
            combined.message.audio = audio[0].message.audio.model_copy(deep=True)
            # Do not hide truncation of either stage behind the other's stop.
            if combined.finish_reason == "stop" and audio[0].finish_reason != "stop":
                combined.finish_reason = audio[0].finish_reason
                combined.stop_reason = audio[0].stop_reason
            group = [combined]
        else:
            group = [choice.model_copy(deep=True) for choice in group]
        for choice in group:
            if choice.message.audio is not None and not choice.message.audio.transcript:
                choice.message.audio.transcript = spoken_text.get(index, "")
        result.extend(group)
    return result
