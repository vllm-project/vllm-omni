# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import json
from pathlib import Path

import av
import pytest

from benchmarks.socialomni import dataset as socialomni
from benchmarks.socialomni.dataset import (
    build_ffmpeg_prefix_command,
    create_video_prefix,
    inspect_socialomni_dataset,
    load_socialomni_level1_samples,
    load_socialomni_level2_samples,
    parse_socialomni_timestamp,
    resolve_ffmpeg_executable,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _level1(sample_id: str, video: str, consistency: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "video_path": video,
        "question": "Who is speaking?",
        "options": ["A. one", "B. two", "C. three", "D. four"],
        "correct_answer": "A",
        "metadata": {"consistency": consistency},
    }


def _level2(sample_id: str, video: str, answer: str) -> dict[str, object]:
    return {
        "video_id": sample_id,
        "video_file": video,
        "full_asr": "Reference text for the judge only.",
        "question_1": {
            "question": "Should Alex speak now?",
            "timestamp": "00:03:25",
            "correct_answer": "A" if answer == "YES" else "B",
            "option_A": "YES",
            "option_B": "NO",
        },
        "question_2": {
            "question": "What should Alex say?",
            "answer": "Hello" if answer == "YES" else "",
        },
        "metadata": {},
    }


def test_loaders_preserve_nested_paths_and_mini_groups(tmp_path: Path) -> None:
    level1 = tmp_path / "data" / "level_1"
    level2 = tmp_path / "data" / "level_2"
    for path in (
        level1 / "videos" / "nested" / "visible.mp4",
        level1 / "videos" / "mismatch.mp4",
        level2 / "videos" / "yes.mp4",
        level2 / "videos" / "nested" / "no.mp4",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    _write(
        level1 / "dataset.json",
        [
            _level1("visible", "nested/visible.mp4", "consistent"),
            _level1("mismatch", "mismatch.mp4", "inconsistent"),
        ],
    )
    _write(
        level2 / "annotations.json",
        {
            "total_samples": 2,
            "data": [
                _level2("yes", "yes.mp4", "YES"),
                _level2("no", "nested/no.mp4", "NO"),
            ],
        },
    )

    first = load_socialomni_level1_samples(tmp_path, mini=True)
    second = load_socialomni_level2_samples(tmp_path, mini=True)

    assert [sample.sample_id for sample in first] == ["visible", "mismatch"]
    assert Path(first[0].video_path).relative_to(tmp_path).as_posix().endswith("videos/nested/visible.mp4")
    assert [sample.gold_when for sample in second] == ["YES", "NO"]
    assert second[0].timestamp_s == 3.25


def test_inspect_dataset_matches_expected_metadata_hashes(tmp_path: Path, monkeypatch) -> None:
    level1 = tmp_path / "data" / "level_1" / "dataset.json"
    level2 = tmp_path / "data" / "level_2" / "annotations.json"
    _write(level1, [])
    _write(level2, {"total_samples": 0, "data": []})
    expected_metadata = {
        "level1": socialomni._sha256(level1),
        "level2": socialomni._sha256(level2),
    }
    monkeypatch.setattr(socialomni, "SOCIALOMNI_METADATA_SHA256", expected_metadata)

    identity = inspect_socialomni_dataset(tmp_path, ("level1", "level2"))

    assert identity["metadata_sha256"] == expected_metadata
    assert identity["verification_scope"] == "metadata_only"
    assert identity["metadata_matches_expected_revision"] is True


def test_inspect_dataset_rejects_modified_metadata_as_expected_revision(tmp_path: Path, monkeypatch) -> None:
    metadata = tmp_path / "data" / "level_1" / "dataset.json"
    _write(metadata, [])
    monkeypatch.setattr(socialomni, "SOCIALOMNI_METADATA_SHA256", {"level1": "0" * 64})

    identity = inspect_socialomni_dataset(tmp_path, ("level1",))

    assert identity["metadata_matches_expected_revision"] is False


def test_inspect_dataset_checks_only_requested_levels(tmp_path: Path) -> None:
    _write(tmp_path / "data" / "level_1" / "dataset.json", [])
    _write(tmp_path / "data" / "level_2" / "annotations.json", "invalid")

    identity = inspect_socialomni_dataset(tmp_path, ("level1",))

    assert set(identity["metadata_sha256"]) == {"level1"}


@pytest.mark.parametrize("video", ["../escape.mp4", "/tmp/escape.mp4", "level_2/x.mp4"])
def test_level1_rejects_path_escape(tmp_path: Path, video: str) -> None:
    level = tmp_path / "data" / "level_1"
    _write(level / "dataset.json", [_level1("bad", video, "consistent")])
    with pytest.raises(ValueError, match="unsafe|wrong"):
        load_socialomni_level1_samples(tmp_path)


def test_level1_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.mp4"
    outside.touch()
    level = tmp_path / "data" / "level_1"
    videos = level / "videos"
    videos.mkdir(parents=True)
    (videos / "escape.mp4").symlink_to(outside)
    _write(level / "dataset.json", [_level1("bad", "escape.mp4", "consistent")])
    with pytest.raises(ValueError, match="escapes"):
        load_socialomni_level1_samples(tmp_path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(3, 3.0), ("3.5", 3.5), ("01:03.5", 63.5), ("00:17:25", 17.25)],
)
def test_parse_timestamp(raw: object, expected: float) -> None:
    assert parse_socialomni_timestamp(raw) == expected


@pytest.mark.parametrize("raw", [True, "", "bad", 0, -1, float("nan"), float("inf")])
def test_parse_timestamp_rejects_invalid_values(raw: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_socialomni_timestamp(raw)


def test_prefix_command_reencodes_video_and_audio(tmp_path: Path) -> None:
    command = build_ffmpeg_prefix_command("ffmpeg", tmp_path / "source.mp4", 1.25, tmp_path / "prefix.mp4")
    assert "-c:v" in command and "libx264" in command
    assert "-c:a" in command and "aac" in command
    assert "copy" not in command
    assert "-nostdin" in command
    assert command[command.index("-t") + 1] == "1.250000"


@pytest.mark.asyncio
async def test_prefix_media_ends_at_query_time(tmp_path: Path, monkeypatch) -> None:
    ffmpeg = resolve_ffmpeg_executable()
    if not ffmpeg:
        pytest.skip("ffmpeg is unavailable")
    source = tmp_path / "source.mp4"
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=64x64:rate=10:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-shortest",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-y",
        str(source),
        stdin=asyncio.subprocess.DEVNULL,
    )
    try:
        assert await asyncio.wait_for(process.wait(), timeout=30) == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    monkeypatch.chdir(tmp_path)
    prefix = await create_video_prefix(source, 0.75, "cache")
    assert prefix.is_absolute()
    assert await create_video_prefix(source, 0.75, "cache") == prefix
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    monkeypatch.chdir(server_dir)
    with av.open(str(prefix)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.duration is not None
        assert container.duration / av.time_base <= 0.85


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_prefix_stops_process_and_removes_partial_file(tmp_path: Path, monkeypatch, cancel: bool) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    started = asyncio.Event()
    stopped = asyncio.Event()

    class Process:
        returncode = None

        async def communicate(self):
            started.set()
            await stopped.wait()
            return b"", b""

        def kill(self):
            self.returncode = -9
            stopped.set()

    process = Process()

    async def start_process(*command, **kwargs):
        assert kwargs["stdin"] == asyncio.subprocess.DEVNULL
        Path(command[-1]).write_bytes(b"partial")
        return process

    monkeypatch.setattr(socialomni, "resolve_ffmpeg_executable", lambda: "ffmpeg")
    monkeypatch.setattr(socialomni.asyncio, "create_subprocess_exec", start_process)
    task = asyncio.create_task(create_video_prefix(source, 1, tmp_path / "cache", timeout=0.01))
    await started.wait()
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(RuntimeError, match="timed out"):
            await task
    assert stopped.is_set()
    assert not list((tmp_path / "cache").iterdir())
