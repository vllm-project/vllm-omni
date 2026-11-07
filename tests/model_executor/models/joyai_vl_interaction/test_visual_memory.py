# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from tests.model_executor.models.joyai_vl_interaction.test_duplex_plugin import Processor, frame
from vllm_omni.experimental.fullduplex.joyvl.memory.memory import MidTermSummary, Summarizer
from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex import memory as memory_module
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.memory import (
    EncodedVisualImage,
    JoyAIVisualMemory,
    VisualMemoryCapacityError,
)
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.plugin import JoyAIDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def append(memory, index=0, image=None):
    return memory.append(
        source_id=f"observation-{index}",
        image=image if image is not None else Image.new("RGB", (8, 8)),
        token_count=4,
        time_s=index * 0.75,
    )


def summarize(memory, text="A red cup is visible."):
    source = memory.frame_source()
    assert source is not None
    assert memory.apply(source, text=text, token_count=8)
    return source


def test_frames_retain_lossless_encoded_pixels_and_distinct_observations():
    memory = JoyAIVisualMemory()
    image = Image.new("RGB", (13, 7), "red")
    first = append(memory, 1, image)
    second = append(memory, 2, image)
    assert first.image.tobytes() == second.image.tobytes() == image.tobytes()
    image.paste("blue", (0, 0, 13, 7))
    assert first.image.getpixel((0, 0)) == (255, 0, 0)
    assert (first.id, second.id) == (0, 1)
    assert first.source_id != second.source_id
    assert memory.pixel_bytes == 2 * 13 * 7 * 3
    assert memory.encoded_bytes == len(first.encoded_image.data) + len(second.encoded_image.data)


def test_encoded_capacity_is_atomic_and_released_by_summary():
    encoded = EncodedVisualImage.from_image(Image.new("RGB", (32, 32), "red"))
    memory = JoyAIVisualMemory(max_encoded_bytes=len(encoded.data))
    first = memory.append(source_id="one", image=encoded, token_count=4)
    with pytest.raises(VisualMemoryCapacityError):
        memory.append(source_id="two", image=encoded, token_count=4)
    assert memory.pending == [first] and memory.next_id == 1
    summarize(memory)
    assert memory.encoded_bytes == 0
    memory.append(source_id="two", image=encoded, token_count=4)
    memory.clear()
    assert memory.encoded_bytes == 0


def test_prepared_encoded_frames_admit_without_encoding_or_decoding(monkeypatch):
    encoded = EncodedVisualImage.from_image(Image.new("RGB", (512, 512), "red"))

    def unexpected(*args, **kwargs):
        raise AssertionError("Session admission must only bind prepared bytes")

    monkeypatch.setattr(EncodedVisualImage, "from_image", unexpected)
    monkeypatch.setattr(EncodedVisualImage, "decode", unexpected)
    memory = JoyAIVisualMemory()
    memory.check_capacity([encoded])
    retained = memory.append(source_id="one", image=encoded, token_count=256)
    assert retained.encoded_image is encoded
    assert memory.encoded_bytes < memory.pixel_bytes / 10
    assert memory.frame_source().covered_ids == (0,)


def test_default_timestamp_uses_elapsed_monotonic_time(monkeypatch):
    memory = JoyAIVisualMemory(start_time=100.0)
    monkeypatch.setattr(memory_module, "monotonic", lambda: 103.125)
    frame = memory.append(source_id="camera", image=Image.new("RGB", (8, 8)), token_count=4)
    assert frame.time_s == 3.125


@pytest.mark.parametrize("limit", ["frames", "bytes"])
def test_capacity_failure_is_atomic_and_successful_summary_releases_capacity(limit):
    memory = JoyAIVisualMemory(
        max_pending=1 if limit == "frames" else 32, max_pixel_bytes=10000 if limit == "frames" else 192
    )
    first = append(memory)
    source = memory.frame_source()
    with pytest.raises(VisualMemoryCapacityError):
        append(memory, 1)
    assert memory.pending == [first]
    assert memory.next_id == 1
    assert memory.matches(source)
    summarize(memory)
    assert memory.pixel_bytes == 0
    append(memory, 1)


def test_batch_admission_checks_total_size_without_mutation():
    memory = JoyAIVisualMemory(max_pixel_bytes=300)
    images = [Image.new("RGB", (8, 8)) for _ in range(2)]
    with pytest.raises(VisualMemoryCapacityError):
        memory.check_capacity(images)
    assert not memory.pending and memory.next_id == 0
    with pytest.raises(ValueError, match="normalized RGB"):
        memory.check_capacity([Image.new("RGBA", (8, 8))])


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_frame_summary_replaces_exact_prefix_preserving_new_tail_and_times(count):
    memory = JoyAIVisualMemory()
    frames = [append(memory, index) for index in range(4)]
    source = memory.frame_source(count)
    tail = append(memory, 4)
    assert memory.matches(source)
    assert memory.apply(source, text="  A red cup moved left.  ", token_count=8)
    assert memory.pending == frames[count:] + [tail]
    summary = memory.mid_term[0]
    assert (summary.first_id, summary.last_id) == (0, count - 1)
    assert (summary.start_s, summary.end_s) == (0, (count - 1) * 0.75)
    assert summary.text == "A red cup moved left."
    assert not memory.matches(source)
    assert not memory.apply(source, text="duplicate", token_count=1)


@pytest.mark.parametrize("kind", ["frames", "long"])
@pytest.mark.parametrize("text,tokens", [("", 1), ("   ", 1), ("too long", 4001), ("invalid", 0)])
def test_invalid_summary_retains_source_and_previous_memory(kind, text, tokens):
    memory = JoyAIVisualMemory()
    append(memory)
    if kind == "long":
        summarize(memory)
        source = memory.long_source()
    else:
        source = memory.frame_source()
    before = (memory.pending.copy(), memory.mid_term.copy(), memory.long_term.copy())
    assert not memory.apply(source, text=text, token_count=tokens)
    assert (memory.pending, memory.mid_term, memory.long_term) == before
    assert memory.matches(source)


def test_long_term_rollover_preserves_uncovered_mid_prefix_and_bounds_retention():
    memory = JoyAIVisualMemory(mid_group=2, max_long=2)
    for index in range(3):
        append(memory, index)
        summarize(memory)
        source = memory.long_source()
        append(memory, index + 10)
        summarize(memory, "A blue ball appeared.")
        tail = memory.mid_term[-1]
        assert memory.matches(source)
        assert memory.apply(source, text=f"Visual event {index}", token_count=4)
        assert memory.mid_term == [tail]
        remainder = memory.long_source()
        assert memory.apply(remainder, text="A blue ball appeared.", token_count=6)
    assert len(memory.long_term) == 2
    assert memory.long_term[0].text == "Visual event 2"
    assert memory.long_term[0].start_s == 1.5
    assert memory.long_term[1].text == "A blue ball appeared."
    assert not memory.mid_term


def test_full_mid_term_keeps_raw_evidence_until_long_term_compression_succeeds():
    memory = JoyAIVisualMemory(mid_group=1)
    append(memory)
    summarize(memory)
    pending = append(memory, 1)
    source = memory.frame_source()
    assert not memory.apply(source, text="New event", token_count=3)
    assert memory.pending == [pending]
    assert memory.apply(memory.long_source(), text="Old event", token_count=3)
    assert memory.apply(source, text="New event", token_count=3)


def test_clear_revokes_all_sources_and_keeps_monotonic_identity_and_clock():
    memory = JoyAIVisualMemory(start_time=50.0)
    append(memory)
    summarize(memory)
    assert memory.apply(memory.long_source(), text="Archived event", token_count=2)
    append(memory, 1)
    summarize(memory)
    long_source = memory.long_source()
    append(memory, 2)
    frame_source = memory.frame_source()
    memory.clear()
    assert not memory.pending and not memory.mid_term and not memory.long_term
    assert memory.generation == 1 and memory.start_time == 50.0
    assert append(memory, 3).id == 3
    for source in (long_source, frame_source):
        assert not memory.matches(source)
        assert not memory.apply(source, text="stale", token_count=1)


def test_no_work_has_no_source():
    memory = JoyAIVisualMemory()
    assert memory.frame_source() is None
    assert memory.long_source() is None


@pytest.mark.parametrize("chunk_frames", [4, 100])
def test_rollover_preserves_frame_and_time_range(chunk_frames):
    memory = JoyAIVisualMemory(chunk_frames=chunk_frames)
    for index in range(chunk_frames * 5):
        append(memory, index, Image.new("RGB", (1, 1)))
        if len(memory.pending) == chunk_frames:
            summarize(memory)
    assert len(memory.mid_term) == 5
    source = memory.long_source()
    assert len(source.covered_ids) == 5
    assert memory.apply(source, text="A sequence of visual events", token_count=8)
    summary = memory.long_term[0]
    last_id = chunk_frames * 5 - 1
    assert (summary.first_id, summary.last_id, summary.start_s, summary.end_s) == (0, last_id, 0.0, last_id * 0.75)
    assert not memory.pending and not memory.mid_term


def test_experimental_defaults_and_independent_summary_token_budgets():
    memory = JoyAIVisualMemory()
    experimental = InteractionConfig()
    assert memory.chunk_frames == experimental.chunk_frames
    assert memory.mid_group == experimental.long_term_every_n_chunks
    assert memory.max_long == experimental.long_term_memory_window
    assert memory.mid_max_tokens == experimental.mid_term_max_tokens
    assert memory.long_max_tokens == experimental.long_term_max_tokens
    assert (memory.max_pending, memory.max_pixel_bytes) == (200, 200 * experimental.max_pixels * 3)
    memory = JoyAIVisualMemory(mid_max_tokens=9, long_max_tokens=3)
    append(memory)
    summarize(memory)
    source = memory.long_source()
    assert not memory.apply(source, text="Too many tokens", token_count=4)
    assert memory.matches(source)
    assert memory.apply(source, text="Short block", token_count=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["frames", "long"])
async def test_visual_summary_request_matches_experimental(mocker, kind):
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    template = mocker.spy(plugin.processor, "apply_chat_template")
    plugin.max_model_len = 16384
    plugin.max_images = 999
    memory = JoyAIVisualMemory()
    backend = Mock(generate=AsyncMock(return_value=("A visual summary.", None)))
    experimental = Summarizer(backend)
    if kind == "frames":
        data_urls = [f"data:image/png;base64,{frame(color)}" for color in ("red", "green", "blue")]
        images = plugin._decode_images(data_urls)
        for index, image in enumerate(images):
            append(memory, index, image)
        entries = tuple(memory.pending)
        source = memory.frame_source()
        await experimental.summarize_chunk(
            source.chunk_index,
            f"{entries[0].time_s:.1f} seconds-{entries[-1].time_s:.1f} seconds",
            [(f"{entry.time_s:.1f} seconds", url) for entry, url in zip(entries, data_urls)],
        )
    else:
        for index in range(memory.mid_group):
            append(memory, index)
            summarize(memory, f"Object {index} moved left.")
        entries = tuple(memory.mid_term)
        source = memory.long_source()
        await experimental.compress_to_long_term(
            [
                MidTermSummary(entry.chunk_index, f"{entry.start_s:.1f} seconds-{entry.end_s:.1f} seconds", entry.text)
                for entry in entries
            ]
        )
    backend.generate.assert_awaited_once()
    expected_messages = backend.generate.call_args.args[0]
    if kind == "frames":
        parts = expected_messages[0]["content"]
        assert [part["image_url"]["url"] for part in parts if part["type"] == "image_url"] == data_urls
        # HTTP uses image URLs; the native stage receives the same images separately.
        expected_messages = [
            {"role": "user", "content": [{"type": "image"} if part["type"] == "image_url" else part for part in parts]}
        ]
    request = plugin._prepare_visual_summary(source, entries, 4000)
    assert request is not None and request.model_source == source
    assert template.call_args.args[0] == expected_messages
    if kind == "frames":
        assert request.prompt["multi_modal_data"]["image"] == images
    expected_sampling = backend.generate.call_args.kwargs
    for name in ("temperature", "top_p", "max_tokens"):
        assert getattr(request.sampling_params, name) == expected_sampling[name]
    for name in ("top_k", "repetition_penalty", "presence_penalty"):
        assert getattr(request.sampling_params, name) == expected_sampling["extra_body"][name]
    assert expected_sampling["extra_body"]["greedy"] is False
    assert request.sampling_params.temperature > 0


@pytest.mark.parametrize("reason", ["context", "image_limit"])
def test_capacity_closed_chunks_keep_number_and_reason_across_retry_and_clear(reason):
    memory = JoyAIVisualMemory()
    for index in range(3):
        append(memory, index)
    source = memory.frame_source(2, closure_reason=reason)
    assert source.chunk_index == 1
    assert not memory.apply(source, text="", token_count=0)
    assert memory.next_chunk_index == 1
    assert memory.apply(source, text="First short chunk", token_count=3)
    assert memory.mid_term[0].chunk_index == 1
    assert memory.mid_term[0].closure_reason == reason
    assert memory.pending[0].id == 2
    summarize(memory)
    assert memory.mid_term[1].chunk_index == 2
    long_source = memory.long_source(2, closure_reason="context")
    assert memory.apply(long_source, text="Merged chunks", token_count=2)
    assert memory.long_term[0].chunk_index is None
    assert memory.long_term[0].closure_reason == "context"
    memory.clear()
    append(memory, 3)
    assert memory.frame_source().chunk_index == 3
