# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The manifest contract can be checked without importing the inference runtime."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("cosmos3_contract", _ROOT / "vllm_omni/model_extras/cosmos3.py")
assert _spec is not None and _spec.loader is not None
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def manifest(vision=False):
    return {
        "wsm": True,
        "multiview": {
            "views": [
                {
                    "camera_key": camera,
                    "control_reference_index": index,
                    **({"vision_reference_index": index + 11} if vision else {}),
                }
                for index, camera in enumerate(contract.COSMOS3_MADS_CAMERAS)
            ]
        },
    }


@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("suffix", [".mp4", ".png"])
def test_resolves_camera_roles_without_mutating_manifest(vision, suffix):
    extra = manifest(vision)
    before = copy.deepcopy(extra)
    paths = [f"upload-{index}{suffix}" for index in range(22 if vision else 11)]
    resolved = contract.resolve_multiview_uploads(extra, paths)
    for index, view in enumerate(resolved["multiview"]["views"]):
        assert view["control_path"] == paths[index]
        if vision:
            assert view["vision_path"] == paths[index + 11]
        assert not any(key.endswith("_reference_index") for key in view)
    assert extra == before


def test_upload_order_is_explicit_and_path_inputs_remain_compatible():
    extra = manifest()
    extra["multiview"]["views"][0] = {"camera_key": contract.COSMOS3_MADS_CAMERAS[0], "control_path": "/owned.mp4"}
    for index, view in enumerate(extra["multiview"]["views"][1:]):
        view["control_reference_index"] = 9 - index
    paths = [f"upload-{index}.mp4" for index in range(10)]
    resolved = contract.resolve_multiview_uploads(extra, paths)
    assert resolved["multiview"]["views"][0]["control_path"] == "/owned.mp4"
    assert [view["control_path"] for view in resolved["multiview"]["views"][1:]] == paths[::-1]
    contract.validate_multiview_request(resolved)


@pytest.mark.parametrize("index", [-1, 11, True, 0.0, "0", None])
def test_rejects_invalid_indexes(index):
    extra = manifest()
    extra["multiview"]["views"][0]["control_reference_index"] = index
    with pytest.raises(ValueError, match="integer index"):
        contract.resolve_multiview_uploads(extra, ["upload.mp4"] * 11)


@pytest.mark.parametrize(
    "case,match",
    [
        ("duplicate", "more than once"),
        ("unused", "exactly once"),
        ("too_many", "at most 22"),
        ("conflict", "cannot be combined"),
        ("missing_control", "control input for every camera"),
        ("partial_vision", "every camera or none"),
        ("camera_order", "camera order"),
        ("duplicate_camera", "unique"),
        ("missing_camera", "camera order"),
        ("mixed_media", "all images or all videos"),
        ("no_wsm", "exactly one"),
        ("top_level_control", "supplied per view"),
        ("unknown_field", "Unsupported"),
    ],
)
def test_rejects_invalid_manifests(case, match):
    extra = manifest()
    paths = ["upload.mp4"] * 11
    views = extra["multiview"]["views"]
    if case == "duplicate":
        views[1]["control_reference_index"] = 0
    elif case == "unused":
        paths.append("unused.mp4")
    elif case == "too_many":
        paths *= 3
    elif case == "conflict":
        views[0]["control"] = "existing.mp4"
    elif case == "missing_control":
        views[-1].pop("control_reference_index")
        paths.pop()
    elif case == "partial_vision":
        views[0]["vision_reference_index"] = len(paths)
        paths.append("vision.mp4")
    elif case == "camera_order":
        views.reverse()
    elif case == "duplicate_camera":
        views[1]["camera_key"] = views[0]["camera_key"]
    elif case == "missing_camera":
        views.pop()
        paths.pop()
    elif case == "mixed_media":
        paths[0] = "image.png"
    elif case == "no_wsm":
        extra.pop("wsm")
    elif case == "top_level_control":
        extra["wsm"] = {"control_path": "control.mp4"}
    elif case == "unknown_field":
        views[0]["typo"] = 1
    with pytest.raises(ValueError, match=match):
        contract.resolve_multiview_uploads(extra, paths)


def test_indexes_without_uploads_and_malformed_views():
    assert contract.has_multiview_upload_indexes(manifest())
    assert not contract.has_multiview_upload_indexes({"multiview": {"views": None}})
    with pytest.raises(ValueError, match="integer index"):
        contract.resolve_multiview_uploads(manifest(), [])
    with pytest.raises(ValueError, match="at least one"):
        contract.resolve_multiview_uploads({"multiview": {"views": {}}}, [])


def test_pipeline_validation_accepts_in_memory_media_with_its_classifier():
    extra = {"wsm": {}, "multiview": {"views": [{"camera_key": "front", "control": object()}]}}
    _, views = contract.validate_multiview_request(extra, ("front",), media_kind=lambda _: "video")
    assert len(views) == 1


@pytest.mark.parametrize("envelope", [False, True])
def test_client_converts_local_manifest_to_upload_indexes(tmp_path, envelope):
    pytest.importorskip("httpx")
    spec = importlib.util.spec_from_file_location(
        "multiview_client", _ROOT / "examples/online_serving/multiview_video/cosmos3_multiview_client.py"
    )
    assert spec is not None and spec.loader is not None
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    extra = manifest(True)
    for index, view in enumerate(extra["multiview"]["views"]):
        for role in ("control", "vision"):
            view.pop(f"{role}_reference_index")
            filename = f"{role}-{index}.mp4"
            (tmp_path / filename).write_bytes(b"media")
            view[f"{role}_path"] = filename
    request = {"prompt": "drive", "fps": 30, "num_steps": 4, **({"extra_params": extra} if envelope else extra)}
    before = copy.deepcopy(request)
    data, paths = client.prepare_request(request, tmp_path)
    resolved = contract.resolve_multiview_uploads(json.loads(data["extra_params"]), [str(path) for path in paths])
    assert len(paths) == 22
    assert resolved["multiview"]["views"][0]["vision_path"] == str(tmp_path / "vision-0.mp4")
    assert data["num_inference_steps"] == "4"
    assert data["fps"] == "30"
    assert request == before


@pytest.mark.parametrize("mode", ["async", "sync", "failed"])
def test_client_uploads_and_downloads_or_reports_failure(tmp_path, monkeypatch, mode):
    httpx = pytest.importorskip("httpx")
    spec = importlib.util.spec_from_file_location(
        "multiview_client", _ROOT / "examples/online_serving/multiview_video/cosmos3_multiview_client.py"
    )
    assert spec is not None and spec.loader is not None
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    extra = manifest()
    for index, view in enumerate(extra["multiview"]["views"]):
        view.pop("control_reference_index")
        filename = f"control-{index}.mp4"
        (tmp_path / filename).write_bytes(f"media-{index}".encode())
        view["control_path"] = filename
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"prompt": "drive", **extra}))
    output = tmp_path / "output.mp4"
    requests = []

    def respond(request):
        requests.append(request.url.path)
        if request.method == "POST":
            body = request.read()
            assert body.count(b'name="input_references"') == 11
            assert b"control_reference_index" in body
            assert b"control_path" not in body
            for index in range(11):
                assert f"media-{index}".encode() in body
            if mode == "sync":
                return httpx.Response(200, content=b"generated-video")
            return httpx.Response(200, json={"id": "video-test", "status": "queued"})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"generated-video")
        if mode == "failed":
            return httpx.Response(500, json={"status": "failed", "error": {"message": "corrupt input"}})
        return httpx.Response(200, json={"status": "completed"})

    real_client = httpx.Client
    monkeypatch.setattr(
        client.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs)
    )
    monkeypatch.setattr(client.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        sys, "argv", ["client", str(manifest_path), "--output", str(output)] + (["--sync"] if mode == "sync" else [])
    )
    if mode == "failed":
        with pytest.raises(RuntimeError, match="corrupt input"):
            client.main()
        assert not output.exists()
    else:
        client.main()
        assert output.read_bytes() == b"generated-video"
        assert requests[-1] == ("/v1/videos/sync" if mode == "sync" else "/v1/videos/video-test/content")
