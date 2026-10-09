# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU-only Realtime history projection benchmark (no inference or model).

Run from the repository root in a vLLM-Omni environment:
    python tests/engine/duplex/benchmark_realtime_history.py --chunks 256 1024 4096

Includes delta projection, one final retrieval and response completion. Each
audio chunk carries one ordered alignment mark; text-only streams carry none.
Copy this file into the baseline checkout to compare identical workloads.
"""

import argparse
import json
import platform
import statistics
import time

from vllm_omni.engine.duplex.realtime_events import (
    RealtimeProjectionState,
    project_internal_event,
    retrieve_item_events,
)


def run_stream(chunks: int, *, audio: bool) -> float:
    state = RealtimeProjectionState(session_id="benchmark")
    project_internal_event(
        state, {"type": "response.created", "response_id": "r", "modalities": ["audio" if audio else "text"]}
    )
    start = time.perf_counter()
    for index in range(1, chunks + 1):
        event: dict[str, object] = {"response_id": "r"}
        if audio:
            event.update(
                type="response.output_audio.delta",
                audio="AAAA",
                text="word ",
                audio_duration_ms=index * 80,
                audio_text_marks=[{"audio_end_ms": index * 80, "text_chars": index * 5}],
            )
        else:
            event.update(type="response.text.delta", delta="word ")
        project_internal_event(state, event)
    retrieved = retrieve_item_events(state, {"item_id": "item_r"})[0].to_realtime()["item"]
    project_internal_event(state, {"type": "response.done", "response_id": "r"})
    elapsed = time.perf_counter() - start
    part = retrieved["content"][0]
    assert part["transcript" if audio else "text"] == "word " * chunks
    if audio:
        assert len(part["audio_text_marks"]) == chunks
    return elapsed * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if min(args.chunks) < 1 or args.warmup < 0 or args.repeat < 1:
        parser.error("chunks/repeat must be positive and warmup non-negative")
    results = []
    for audio in (False, True):
        for chunks in args.chunks:
            for _ in range(args.warmup):
                run_stream(chunks, audio=audio)
            samples = [run_stream(chunks, audio=audio) for _ in range(args.repeat)]
            results.append(
                {
                    "mode": "audio_text_marks" if audio else "text",
                    "chunks": chunks,
                    "median_ms": statistics.median(samples),
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                    "samples_ms": samples,
                }
            )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "args": vars(args),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
