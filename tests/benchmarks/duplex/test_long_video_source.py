# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.data_modules import long_video_source as source

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def _archive(root, *, name="src_videos/Ego4D/example.mp4", symlink=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo(name)
        if symlink:
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            archive.addfile(member)
        else:
            member.size = 5
            archive.addfile(member, io.BytesIO(b"video"))
    data = stream.getvalue()
    # Split through a header, as download boundaries need not match tar records.
    (root / "src_videos.tar.partaa").write_bytes(data[:517])
    (root / "src_videos.tar.partab").write_bytes(data[517:])


def test_local_ovo_split_tar_and_cached_extraction(tmp_path):
    annotation = tmp_path / "ovo_bench_new.json"
    annotation.write_text("[]")
    _archive(tmp_path)
    resolved, media = source.resolve_long_video_source(str(tmp_path), dataset="ovo-bench")
    assert resolved == annotation
    assert (media / "Ego4D/example.mp4").read_bytes() == b"video"
    for part in tmp_path.glob("src_videos.tar.part*"):
        part.unlink()
    assert source.resolve_long_video_source(str(annotation), dataset="ovo-bench") == (annotation, media)


@pytest.mark.parametrize("name,symlink", [("../escape", False), ("/escape", False), ("src_videos/link", True)])
def test_ovo_extraction_rejects_unsafe_members(tmp_path, name, symlink):
    _archive(tmp_path, name=name, symlink=symlink)
    with pytest.raises(ValueError, match="Unsafe"):
        source._extract_ovo(tmp_path)
    assert not (tmp_path / ".vllm_omni_src_videos").exists()


def test_hub_ovo_uses_pinned_revision_and_offline_cache(tmp_path, monkeypatch):
    annotation = tmp_path / "annotations.json"
    annotation.write_text("[]")
    _archive(tmp_path)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(source, "_ovo_annotation", lambda: annotation)
    monkeypatch.setattr(source, "hf_api", lambda: SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(source.hf_constants, "HF_HUB_OFFLINE", True)
    result = source.resolve_long_video_source("JoeLeelyf/OVO-Bench@fixed-revision", dataset="ovo-bench")
    assert result[0] == annotation
    assert calls == [
        dict(
            repo_id="JoeLeelyf/OVO-Bench",
            repo_type="dataset",
            revision="fixed-revision",
            allow_patterns=["src_videos.tar.part*"],
            local_files_only=True,
        )
    ]


def test_ovo_offline_annotation_miss_and_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(source.hf_constants, "HF_HOME", str(tmp_path))
    monkeypatch.setattr(source.hf_constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(FileNotFoundError, match="not cached"):
        source._ovo_annotation()
    cached = tmp_path / "vllm_omni/ovo-bench" / source.OVO_ANNOTATION_REVISION / "ovo_bench_new.json"
    cached.write_text("[]")
    assert source._ovo_annotation() == cached


def test_videomme_local_root_reuses_shared_layout(tmp_path):
    (tmp_path / "videomme").mkdir()
    annotation = tmp_path / "videomme/test-00000-of-00001.parquet"
    annotation.touch()
    (tmp_path / "video/subfolder").mkdir(parents=True)
    (tmp_path / "video/subfolder/example.mp4").touch()
    assert source.resolve_long_video_source(str(tmp_path), dataset="videomme-streaming") == (
        annotation,
        tmp_path / "video",
    )


def test_local_regression_client_and_server_capacity_match():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "tests/dfx/perf/tests/test_joyai_long_video_qa.json").read_text())[0]
    assert config["server_params"]["update"]["duplex_session.max_sessions"] == 8
    for benchmark in config["benchmark_params"]:
        assert benchmark["max_concurrency"] == [8]
        assert not Path(benchmark["dataset_path"]).is_absolute()
        assert "long_video_media_root" not in benchmark


def test_ovo_download_is_cached_after_valid_response(tmp_path, monkeypatch):
    import requests

    monkeypatch.setattr(source.hf_constants, "HF_HOME", str(tmp_path))
    monkeypatch.setattr(source.hf_constants, "HF_HUB_OFFLINE", False)
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b'[{"id": 37}]'

    def get(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(requests, "get", get)
    path = source._ovo_annotation()
    assert json.loads(path.read_text()) == [{"id": 37}]
    assert source._ovo_annotation() == path
    assert len(calls) == 1
    assert source.OVO_ANNOTATION_REVISION in calls[0]
