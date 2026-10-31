# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise final session prompts with the supported, real Qwen processor.

Only processor/tokenizer assets are needed; no model weights or forward pass
are loaded. Populate the pinned checkpoint's HF cache before running, or set
QWEN3_OMNI_PROCESSOR_PATH to its ``snapshots/<revision>`` directory. Missing
optional cached assets skip; an explicit path must load successfully. All HF
loads are local-only, and an explicit path must name the pinned revision.
The ready pipeline's Model Executor CPU job prefetches these assets and sets
QWEN3_OMNI_PROCESSOR_REQUIRED=1, making cache failures errors instead of skips.
"""

from __future__ import annotations

import io
import json
import os
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pybase64 as base64
import pytest
import regex as re
import torch
from PIL import Image

from tests.engine.duplex.test_session_runner import Harness, close_harness, open_harness, tts_output
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.commands import AppendAudio, Commit, CreateItem, CreateResponse
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin
from vllm_omni.protocol.realtime.items import normalize_conversation_item

if TYPE_CHECKING:
    from transformers import Qwen3OmniMoeProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]

PROCESSOR_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
PROCESSOR_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
_MEDIA_BLOCKS = {
    "audio": "<|audio_start|><|audio_pad|><|audio_end|>",
    "image": "<|vision_start|><|image_pad|><|vision_end|>",
}


@pytest.fixture(scope="module")
def qwen_processor(record_testsuite_property) -> Qwen3OmniMoeProcessor:
    import transformers
    from transformers import AutoProcessor, Qwen3OmniMoeProcessor

    override = os.environ.get("QWEN3_OMNI_PROCESSOR_PATH")
    source = PROCESSOR_MODEL
    if override is not None:
        checkpoint = Path(override).expanduser().resolve()
        if not checkpoint.is_dir() or checkpoint.name != PROCESSOR_REVISION:
            pytest.fail("QWEN3_OMNI_PROCESSOR_PATH must be the pinned snapshots/" + PROCESSOR_REVISION)
        source = str(checkpoint)
    try:
        processor = AutoProcessor.from_pretrained(
            source,
            revision=PROCESSOR_REVISION,
            local_files_only=True,
            trust_remote_code=False,
        )
    except OSError:
        if override is not None or os.environ.get("QWEN3_OMNI_PROCESSOR_REQUIRED") == "1":
            raise
        pytest.skip(
            f"Optional processor assets for {PROCESSOR_MODEL}@{PROCESSOR_REVISION} are not cached; "
            "set QWEN3_OMNI_PROCESSOR_PATH to the pinned snapshot to require this integration test"
        )
    assert isinstance(processor, Qwen3OmniMoeProcessor)
    assert processor.audio_token == "<|audio_pad|>"
    assert processor.image_token == "<|image_pad|>"
    record_testsuite_property("qwen3_omni_processor_model", PROCESSOR_MODEL)
    record_testsuite_property("qwen3_omni_processor_revision", PROCESSOR_REVISION)
    record_testsuite_property("qwen3_omni_processor_source", source)
    record_testsuite_property("qwen3_omni_transformers_version", transformers.__version__)
    return processor


def _image_part(image: Image.Image) -> dict[str, object]:
    with io.BytesIO() as stream:
        image.save(stream, format="PNG")
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}


def _pcm(index: int) -> np.ndarray:
    # Different durations and frequencies expose swapped or duplicated clips in
    # both decoded PCM and the real processor's padded features/token spans.
    time = np.arange(16000 + 800 * index, dtype=np.float64) / 16000
    return (0.25 * np.sin(2 * np.pi * (220 + 37 * index) * time)).astype("<f4")


async def _open(processor: Qwen3OmniMoeProcessor, keep: int | None) -> Harness:
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = processor
    runtime_config = (
        DuplexSessionRuntimeConfig() if keep is None else DuplexSessionRuntimeConfig(context_keep_media_turns=keep)
    )
    return await open_harness(
        plugin=plugin,
        auto_response=False,
        stage_count=3,
        modalities=("text",),
        model=PROCESSOR_MODEL,
        runtime_config=runtime_config,
    )


async def _commit_audio(h: Harness, audio: np.ndarray, transcript: str) -> None:
    await h.run(AppendAudio(audio=audio.tobytes(), format="pcm_f32le", sample_rate_hz=16000, is_speech=True))
    events = await h.run(Commit(final=True, create_response=False))
    assert not [event for event in events if event.type == "error"]
    item_id = h.session.history_snapshot().item_ids[-1]
    assert isinstance(item_id, str)
    assert h.session.update_input_transcript(item_id, transcript, audio_part=h.session.input_audio_part(item_id))


async def _respond(h: Harness) -> None:
    before = len(h.port.submissions)
    events = await h.run(CreateResponse())
    assert len(h.port.submissions) == before + 1, [event.type for event in events]


def _media_spans(input_ids: torch.Tensor, processor: Qwen3OmniMoeProcessor) -> list[tuple[str, int]]:
    names = {
        processor.tokenizer.convert_tokens_to_ids(processor.audio_token): "audio",
        processor.tokenizer.convert_tokens_to_ids(processor.image_token): "image",
    }
    return [(names[token], sum(1 for _ in span)) for token, span in groupby(input_ids.tolist()) if token in names]


def _assert_preprocessing(
    processor: Qwen3OmniMoeProcessor,
    prompt: dict,
    *,
    audio: list[np.ndarray],
    images: list[Image.Image],
    order: list[str],
) -> None:
    text = prompt["prompt"]
    media = prompt["multi_modal_data"]
    assert text.startswith("<|im_start|>system\nYou are a concise assistant.")
    assert text.endswith("<|im_start|>assistant\n")
    assert re.findall(r"<\|(audio|image)_pad\|>", text) == order
    actual_audio = media.get("audio", [])
    actual_images = media.get("image", [])
    assert len(actual_audio) == len(audio)
    assert len(actual_images) == len(images)
    for (actual, sample_rate), expected in zip(actual_audio, audio, strict=True):
        assert sample_rate == 16000
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(actual_images, images, strict=True):
        assert actual.size == expected.size
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    processed = processor(
        text=text,
        audio=[clip for clip, _ in actual_audio] or None,
        images=actual_images or None,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    # Build an independent, literal media-only input from the expected fixture
    # order. Matching its token spans and feature rows checks actual template
    # expansion and positional pairing, beyond placeholder/array counts.
    reference = processor(
        text=" ".join(_MEDIA_BLOCKS[kind] for kind in order),
        audio=audio or None,
        images=images or None,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    assert processed["input_ids"].shape[0] == 1
    assert processed["input_ids"].shape == processed["attention_mask"].shape
    assert _media_spans(processed["input_ids"][0], processor) == _media_spans(reference["input_ids"][0], processor)
    for key in ("input_features", "feature_attention_mask", "pixel_values", "image_grid_thw"):
        assert (key in processed) == (key in reference)
        if key in reference:
            assert processed[key].device.type == "cpu"
            assert torch.isfinite(processed[key]).all()
            torch.testing.assert_close(processed[key], reference[key])
    assert processed["input_features"].shape[0] == len(audio)
    if images:
        assert processed["image_grid_thw"].shape[0] == len(images)


@pytest.mark.parametrize("keep", [1, None], ids=["keep-one", "default-off"])
async def test_six_audio_turns_reach_real_processor_with_expected_media(qwen_processor, keep):
    h = await _open(qwen_processor, keep)
    old_image = Image.new("RGB", (64, 64), color="red")
    clips = [_pcm(turn) for turn in range(6)]
    try:
        h.session.register_history_item(
            "old-picture",
            {"role": "user", "content": [{"type": "text", "text": "old picture question"}, _image_part(old_image)]},
        )
        h.session.register_history_item("old-picture-answer", {"role": "assistant", "content": "old picture answer"})
        for turn, clip in enumerate(clips):
            await _commit_audio(h, clip, f"spoken question {turn + 1}")
            await _respond(h)
            if turn < 5:
                request_id = h.port.submissions[-1].context.request_id
                await h.deliver_and_settle(
                    tts_output(request_id, samples=0, text=f"spoken answer {turn + 1}", finished=True), stage_id=0
                )
                await h.deliver_and_settle(tts_output(request_id, samples=0, finished=True), stage_id=2)

        assert len(h.port.submissions) == 6
        prompt = h.port.submissions[-1].prompt
        text = prompt["prompt"]
        assert text.count("old picture question") == text.count("old picture answer") == 1
        if keep == 1:
            expected_audio, expected_images, order = clips[-1:], [], ["audio"]
            for turn in range(1, 6):
                assert text.count(f"spoken question {turn}") == 1
                assert text.count(f"spoken answer {turn}") == 1
            assert text.index("old picture answer") < text.index("spoken question 1")
            assert text.index("spoken question 5") < text.index(_MEDIA_BLOCKS["audio"])
        else:
            # Default-off preserves the existing four retained-audio slots;
            # expired unbound audio turns do not acquire L1 transcript text.
            expected_audio, expected_images = clips[2:], [old_image]
            order = ["image", "audio", "audio", "audio", "audio"]
            assert "spoken question 1" not in text and "spoken question 2" not in text
            assert "spoken answer 1" not in text and "spoken answer 2" not in text
        _assert_preprocessing(qwen_processor, prompt, audio=expected_audio, images=expected_images, order=order)
        stored = h.session.history_snapshot().messages
        assert stored[0]["content"][1] == _image_part(old_image)
        assert [message["transcript"] for message in stored if message.get("transcript")] == [
            f"spoken question {turn}" for turn in range(1, 7)
        ]
    finally:
        await close_harness(h)


@pytest.mark.parametrize(
    ("keep", "tool_result"),
    [(0, False), (0, True), (None, True)],
    ids=["fresh-system-keep-zero", "tool-continuation-keep-zero", "tool-continuation-default-off"],
)
async def test_current_media_survives_system_and_tool_continuation(qwen_processor, keep, tool_result):
    h = await _open(qwen_processor, keep)
    old_image = Image.new("RGB", (64, 64), color="red")
    current_images = [Image.new("RGB", (64, 64), color=color) for color in ("green", "blue")]
    clips = [_pcm(7), _pcm(8)]
    try:
        h.session.register_history_item(
            "old-picture",
            {"role": "user", "content": [{"type": "text", "text": "old picture question"}, _image_part(old_image)]},
        )
        h.session.register_history_item("old-answer", {"role": "assistant", "content": "old picture answer"})
        h.session.register_history_item(
            "current-pictures",
            {
                "role": "user",
                "content": [{"type": "text", "text": "compare current pictures"}]
                + [_image_part(image) for image in current_images],
            },
        )
        for index, clip in enumerate(clips):
            await _commit_audio(h, clip, f"current speech part {index + 1}")
        if tool_result:
            # A model-produced call follows a real media submission: its PCM
            # has already been bound before the continuation's history anchor.
            await _respond(h)
            initial = h.port.submissions[-1].prompt
            _assert_preprocessing(
                qwen_processor,
                initial,
                audio=[np.concatenate(clips)],
                images=current_images if keep == 0 else [old_image, *current_images],
                order=[*(["image"] * (2 if keep == 0 else 3)), "audio"],
            )
            before_call = h.session.history_snapshot()
            request_id = h.port.submissions[-1].context.request_id
            events = await h.deliver_and_settle(
                tts_output(
                    request_id,
                    samples=0,
                    text='<tool_call>{"name":"compare_pictures","arguments":{"units":"metric"}}</tool_call>',
                    finished=True,
                ),
                stage_id=0,
            )
            calls = [event for event in events if event.type == "response.function_call_arguments.done"]
            assert len(calls) == 1
            call_id = calls[0].call_id
            call = h.runner.tool_calls.record(call_id)
            assert call is not None and call.origin.name == "compare_pictures"
            await h.deliver_and_settle(tts_output(request_id, samples=0, text="", finished=True), stage_id=2)
            assert h.session.active_response_id is None
            assert h.session.history_snapshot() == before_call
        await h.run(
            CreateItem(
                item=normalize_conversation_item(
                    {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "answer briefly"}]}
                )
            )
        )
        if tool_result:
            before_tool = h.session.history_snapshot()
            events = await h.run(
                CreateItem(
                    item=normalize_conversation_item(
                        {"type": "function_call_output", "call_id": call_id, "output": "comparison complete"}
                    )
                )
            )
            assert not [event for event in events if event.type == "error"]
            # Supported tool calls/results live in plugin runtime state, not
            # assistant/tool history entries that could end the current input.
            assert h.session.history_snapshot() == before_tool
            assert len(h.session.runtime_config["qwen_tool_turns"]) == 1

        await _respond(h)
        assert len(h.port.submissions) == (2 if tool_result else 1)

        if tool_result:
            assert h.session.history_snapshot() == before_tool
            assert len(h.session.runtime_config["qwen_tool_turns_inflight"]) == 1

        prompt = h.port.submissions[-1].prompt
        text = prompt["prompt"]
        assert text.count("compare current pictures") == text.count("answer briefly") == 1
        assert text.count("old picture question") == text.count("old picture answer") == 1
        assert "<|im_start|>system\nanswer briefly<|im_end|>" in text
        if tool_result:
            assert text.count("<tool_call>") == text.count("</tool_call>") == 1
            (rendered_call,) = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
            assert json.loads(rendered_call) == {"name": "compare_pictures", "arguments": {"units": "metric"}}
            assert text.count("<tool_response>\ncomparison complete\n</tool_response>") == 1
            assert text.index("compare current pictures") < text.index("<tool_call>") < text.index("<tool_response>")
        images = current_images if keep == 0 else [old_image, *current_images]
        _assert_preprocessing(
            qwen_processor,
            prompt,
            audio=[np.concatenate(clips)],
            images=images,
            order=[*(["image"] * len(images)), "audio"],
        )
    finally:
        await close_harness(h)
