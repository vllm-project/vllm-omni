# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The manifest contract can be checked without importing the inference runtime."""

import ast
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


def legacy_prompt(captions):
    return json.dumps(
        {
            "view_order": "The view captions are listed in the same order as the generated video views.",
            "num_views": len(captions),
            "views": [
                {"view_index": index, "camera_role": "front", "caption": caption}
                for index, caption in enumerate(captions)
            ],
        }
    )


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
        ("too_many", "at most 23"),
        ("conflict", "cannot be combined"),
        ("missing_control", "control input for every camera"),
        ("partial_vision", "complete RGB videos"),
        ("duplicate_camera", "unique"),
        ("unknown_camera", "subset"),
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
        paths.append("vision.png")
    elif case == "duplicate_camera":
        views[1]["camera_key"] = views[0]["camera_key"]
    elif case == "unknown_camera":
        views[0]["camera_key"] = "unknown"
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
@pytest.mark.parametrize("resolution", [None, "480", "720"])
@pytest.mark.parametrize("aspect_ratio", [None, "auto", "3:4", "9:16"])
@pytest.mark.parametrize("null_lidar", [False, True])
def test_client_uploads_and_downloads_or_reports_failure(
    tmp_path, monkeypatch, mode, resolution, aspect_ratio, null_lidar
):
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
    captions = [f"Raw camera caption {index}." for index in range(11)]
    request_manifest = {"prompt": legacy_prompt(captions), **extra}
    if null_lidar:
        request_manifest["lidar"] = None
    if aspect_ratio is not None:
        request_manifest["aspect_ratio"] = "1:1"
        request_manifest["multiview"]["aspect_ratio"] = "16:9"
    if resolution is not None:
        # The CLI must replace conflicting fields and stale dimensions together.
        request_manifest.update(resolution="480", width=832, height=480)
        request_manifest["multiview"]["resolution"] = "720"
    manifest_path.write_text(json.dumps(request_manifest))
    output = tmp_path / "output.mp4"
    requests = []

    def respond(request):
        requests.append(request.url.path)
        if request.method == "POST":
            body = request.read()
            assert body.count(b'name="input_references"') == 11
            assert b"control_reference_index" in body
            assert b"control_path" not in body
            for caption in captions:
                assert f'"prompt": {json.dumps(caption)}'.encode() in body
            expected_resolution = resolution
            if expected_resolution is None:
                assert b'"resolution":' not in body
            else:
                assert f'"resolution": "{expected_resolution}"'.encode() in body
            expected_ratio = (aspect_ratio or "auto").replace(":", ",")
            assert f'"aspect_ratio": "{expected_ratio}"'.encode() in body
            if expected_ratio == "auto" or expected_resolution is None:
                assert b'name="width"' not in body
                assert b'name="height"' not in body
            else:
                width, height = client.SUPPORTED_RESOLUTIONS[expected_resolution][expected_ratio]
                assert f'name="width"\r\n\r\n{width}\r\n'.encode() in body
                assert f'name="height"\r\n\r\n{height}\r\n'.encode() in body
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
        sys,
        "argv",
        ["client", str(manifest_path), "--output", str(output)]
        + (["--sync"] if mode == "sync" else [])
        + (["--resolution", resolution] if resolution is not None else [])
        + (["--aspect-ratio", aspect_ratio] if aspect_ratio is not None else []),
    )
    if mode == "failed":
        with pytest.raises(RuntimeError, match="corrupt input"):
            client.main()
        assert not output.exists()
    else:
        client.main()
        assert output.read_bytes() == b"generated-video"
        assert requests[-1] == ("/v1/videos/sync" if mode == "sync" else "/v1/videos/video-test/content")


@pytest.fixture
def multiview_client():
    pytest.importorskip("httpx")
    spec = importlib.util.spec_from_file_location(
        "multiview_client", _ROOT / "examples/online_serving/multiview_video/cosmos3_multiview_client.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_client_expands_legacy_captions_through_upload_validation(
    tmp_path, multiview_client, envelope, shuffled, explicit
):
    (tmp_path / "control.mp4").write_bytes(b"media")
    captions = [f'Camera {index}: a "STANDARD" truck passes.\nTrees line the road.' for index in range(11)]
    payload = json.loads(legacy_prompt(captions))
    if shuffled:
        payload["views"].reverse()
    extra = {
        "wsm": {},
        "multiview": {
            "views": [{"camera_key": camera, "control_path": "control.mp4"} for camera in contract.COSMOS3_MADS_CAMERAS]
        },
    }
    expected = captions.copy()
    if explicit:
        expected[3] = extra["multiview"]["views"][3]["prompt"] = "An explicit camera caption."
    request = {"prompt": json.dumps(payload), **({"extra_params": extra} if envelope else extra)}
    before = copy.deepcopy(request)
    data, paths = multiview_client.prepare_request(request, tmp_path)
    resolved = contract.resolve_multiview_uploads(json.loads(data["extra_params"]), [str(path) for path in paths])
    _, views = contract.validate_multiview_request(resolved, separate_view_text_tokenization=True)
    assert [view["prompt"] for view in views] == expected
    # Legacy checkpoints still receive their original aggregate top-level text.
    contract.validate_multiview_request(resolved, separate_view_text_tokenization=False)
    assert data["prompt"] == request["prompt"]
    assert request == before


@pytest.mark.parametrize(
    "failure,match",
    [
        ("count", "must match"),
        ("boolean_count", "must match"),
        ("missing_view", "must match"),
        ("non_list", "must match"),
        ("non_object", "must be objects"),
        ("duplicate_index", "unique integers"),
        ("out_of_range", "unique integers"),
        ("negative_index", "unique integers"),
        ("boolean_index", "unique integers"),
        ("missing_index", "unique integers"),
        ("empty_caption", "nonempty string"),
        ("object_caption", "nonempty string"),
    ],
)
def test_client_rejects_ambiguous_legacy_captions_before_upload(tmp_path, multiview_client, failure, match):
    payload = json.loads(legacy_prompt(["First caption.", "Second caption."]))
    if failure == "count":
        payload["num_views"] = 3
    elif failure == "boolean_count":
        payload["num_views"] = True
    elif failure == "missing_view":
        payload["views"].pop()
    elif failure == "non_list":
        payload["views"] = {}
    elif failure == "non_object":
        payload["views"][0] = "caption"
    elif failure == "duplicate_index":
        payload["views"][0]["view_index"] = 1
    elif failure == "out_of_range":
        payload["views"][0]["view_index"] = 2
    elif failure == "negative_index":
        payload["views"][0]["view_index"] = -1
    elif failure == "boolean_index":
        payload["views"][0]["view_index"] = False
    elif failure == "missing_index":
        payload["views"][0].pop("view_index")
    elif failure == "empty_caption":
        payload["views"][0]["caption"] = " \n"
    elif failure == "object_caption":
        payload["views"][0]["caption"] = {"caption": "nested"}
    request = {
        "prompt": json.dumps(payload),
        "multiview": {"views": [{"camera_key": camera, "control_path": "missing.mp4"} for camera in ("front", "rear")]},
    }
    with pytest.raises(ValueError, match=match):
        multiview_client.prepare_request(request, tmp_path)


@pytest.mark.parametrize("prompt", ["Drive.", "", '{"caption": "Drive."}', '{"views": []}', "[]"])
def test_client_does_not_broadcast_unstructured_prompts(tmp_path, multiview_client, prompt):
    request = {"prompt": prompt, "multiview": {"views": [{"camera_key": contract.COSMOS3_MADS_CAMERAS[0]}]}}
    data, _ = multiview_client.prepare_request(request, tmp_path)
    assert data["prompt"] == prompt
    assert "prompt" not in json.loads(data["extra_params"])["multiview"]["views"][0]


def test_client_explicit_prompts_do_not_require_matching_legacy_payload(tmp_path, multiview_client):
    request = {
        "prompt": legacy_prompt(["First.", "Second."]),
        "multiview": {"views": [{"camera_key": contract.COSMOS3_MADS_CAMERAS[1], "prompt": "Explicit."}]},
    }
    data, _ = multiview_client.prepare_request(request, tmp_path)
    assert json.loads(data["extra_params"])["multiview"]["views"][0]["prompt"] == "Explicit."


@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("location", ["top", "extra", "nested", "all", "omitted"])
def test_client_preserves_resolution_and_manifest(tmp_path, multiview_client, envelope, location):
    (tmp_path / "control.mp4").write_bytes(b"media")
    extra = {"multiview": {"views": [{"camera_key": "front", "control_path": "control.mp4"}]}, "wsm": {}}
    request = {"prompt": "drive", **({"extra_params": extra} if envelope else extra)}
    if location in ("top", "all"):
        request["resolution"] = 720
    if location in ("extra", "all"):
        request.setdefault("extra_params", {})["resolution"] = "720"
    if location in ("nested", "all"):
        extra["multiview"]["resolution"] = 720
    before = copy.deepcopy(request)
    data, paths = multiview_client.prepare_request(request, tmp_path)
    resolved = json.loads(data["extra_params"])
    if location == "omitted":
        assert "resolution" not in resolved and "resolution" not in resolved["multiview"]
    else:
        assert resolved["resolution"] == resolved["multiview"]["resolution"] == "720"
    assert "width" not in data and "height" not in data
    assert resolved["multiview"]["aspect_ratio"] == "auto"
    assert paths == [tmp_path / "control.mp4"]
    assert request == before


@pytest.mark.parametrize("resolution", [256, 704, "1080", "720p", True])
def test_client_rejects_unsupported_resolution(tmp_path, multiview_client, resolution):
    request = {"multiview": {"views": [], "resolution": resolution}}
    with pytest.raises(ValueError, match="Unsupported Cosmos3 multiview resolution"):
        multiview_client.prepare_request(request, tmp_path)


@pytest.mark.parametrize("location", ["top", "extra"])
def test_client_rejects_resolution_conflicts(tmp_path, multiview_client, location):
    request = {"multiview": {"views": [], "resolution": "720"}}
    target = request if location == "top" else request.setdefault("extra_params", {})
    target["resolution"] = "480"
    with pytest.raises(ValueError, match="Conflicting Cosmos3 multiview resolutions"):
        multiview_client.prepare_request(request, tmp_path)


@pytest.mark.parametrize("dimensions", [{"width": 832}, {"height": 480}])
def test_client_rejects_dimensions_that_disagree_with_resolution(tmp_path, multiview_client, dimensions):
    request = {"multiview": {"views": [], "resolution": "720", "aspect_ratio": "16:9"}, **dimensions}
    with pytest.raises(ValueError, match="resolution='720' requires"):
        multiview_client.prepare_request(request, tmp_path)


def test_client_buckets_match_runtime_without_importing_inference(multiview_client):
    tree = ast.parse((_ROOT / "vllm_omni/diffusion/models/cosmos3/utils.py").read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "VIDEO_RES_SIZE_INFO"
    )
    canonical = ast.literal_eval(assignment.value)
    assert multiview_client.SUPPORTED_RESOLUTIONS == {key: canonical[key] for key in ("480", "720")}
    assert set(contract.COSMOS3_MULTIVIEW_ASPECT_RATIOS) == set(canonical["480"]) == set(canonical["720"])


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "auto"),
        ("auto", "auto"),
        ("1:1", "1,1"),
        ("4,3", "4,3"),
        (" 6 : 8 ", "3,4"),
        ("32:18", "16,9"),
        ("9:16", "9,16"),
    ],
)
def test_ratio_normalization_matches_client(multiview_client, value, expected):
    assert contract.normalize_multiview_aspect_ratio(value) == expected
    assert multiview_client.normalize_aspect_ratio(value) == expected


@pytest.mark.parametrize("value", ["21:9", "0:1", "-1:1", "4:0", "", "square", True, 1.5, "1.5:1", [4, 3]])
def test_invalid_ratios_rejected_by_contract_and_client(multiview_client, value):
    for normalize in (contract.normalize_multiview_aspect_ratio, multiview_client.normalize_aspect_ratio):
        with pytest.raises(ValueError, match="aspect_ratio"):
            normalize(value)
    extra = manifest()
    extra["multiview"]["aspect_ratio"] = value
    with pytest.raises(ValueError, match="aspect_ratio"):
        contract.resolve_multiview_uploads(extra, [f"{index}.mp4" for index in range(11)])


@pytest.mark.parametrize("resolution", ["480", "720"])
@pytest.mark.parametrize("ratio", ["1:1", "4:3", "3:4", "16:9", "9:16"])
@pytest.mark.parametrize("location", ["top", "extra", "nested"])
def test_client_explicit_aspect_ratios(tmp_path, multiview_client, resolution, ratio, location):
    extra = {"resolution": resolution, "multiview": {"views": []}}
    request = {"extra_params": extra}
    target = {"top": request, "extra": extra, "nested": extra["multiview"]}[location]
    target["aspect_ratio"] = ratio
    before = copy.deepcopy(request)
    data, _ = multiview_client.prepare_request(request, tmp_path)
    normalized = ratio.replace(":", ",")
    assert (int(data["width"]), int(data["height"])) == multiview_client.SUPPORTED_RESOLUTIONS[resolution][normalized]
    assert json.loads(data["extra_params"])["multiview"]["aspect_ratio"] == normalized
    assert request == before


def test_client_ratio_conflicts_and_independent_overrides(tmp_path, multiview_client):
    request = {
        "aspect_ratio": "1:1",
        "multiview": {"views": [], "aspect_ratio": "9:16", "resolution": "720"},
        "width": 832,
        "height": 480,
    }
    with pytest.raises(ValueError, match="Conflicting.*aspect ratios"):
        multiview_client.prepare_request(request, tmp_path)
    # A resolution override cannot hide an unrelated aspect-ratio conflict.
    with pytest.raises(ValueError, match="Conflicting.*aspect ratios"):
        multiview_client.prepare_request(request, tmp_path, resolution_override="480")
    explicit, _ = multiview_client.prepare_request(request, tmp_path, aspect_ratio_override="3:4")
    assert (explicit["width"], explicit["height"]) == ("832", "1104")
    automatic, _ = multiview_client.prepare_request(request, tmp_path, aspect_ratio_override="auto")
    assert "width" not in automatic and "height" not in automatic
    assert json.loads(automatic["extra_params"])["resolution"] == "720"
    request["aspect_ratio"] = "18:32"
    resolved, _ = multiview_client.prepare_request(request, tmp_path, resolution_override="480")
    assert (resolved["width"], resolved["height"]) == ("480", "832")


@pytest.mark.parametrize("dimensions", [{"width": 640}, {"height": 640}, {"width": 640, "height": 640}])
def test_client_auto_preserves_only_explicit_dimension_constraints(tmp_path, multiview_client, dimensions):
    request = {"multiview": {"views": []}, **dimensions}
    data, _ = multiview_client.prepare_request(request, tmp_path)
    assert {key: int(data[key]) for key in ("width", "height") if key in data} == dimensions
    overridden, _ = multiview_client.prepare_request(request, tmp_path, resolution_override="720")
    assert "width" not in overridden and "height" not in overridden


@pytest.mark.parametrize("mode", ["ordinary", "transfer", "completion", "joint"])
@pytest.mark.parametrize("vision", ["none", "images", "prefix"])
def test_request_mode_matrix_preserves_subset_order(mode, vision):
    keys = contract.COSMOS3_MADS_CAMERAS[2::-1]
    views = [{"camera_key": key, "prompt": f"Raw caption {i}."} for i, key in enumerate(keys)]
    extra = {"multiview": {"views": views}}
    if mode != "ordinary":
        extra["wsm"] = {}
        for view in views:
            view["control_path"] = "control.mp4"
    if mode == "joint":
        extra["lidar"] = {"control_path": "map.safetensors"}
    if mode == "completion":
        views[0]["vision_path"] = "complete.mp4"
    elif vision != "none":
        for view in views:
            view["vision_path"] = "image.png" if vision == "images" else "prefix.mp4"
    _, resolved = contract.validate_multiview_request(
        extra, separate_view_text_tokenization=True, variable_view_count=True
    )
    assert [view["camera_key"] for view in resolved] == list(keys)


@pytest.mark.parametrize(
    "failure", ["partial_joint", "extra_hint", "missing_control", "numeric_camera", "missing_caption"]
)
def test_joint_request_mode_rejections(failure):
    extra = contract.resolve_multiview_uploads(manifest(), ["control.mp4"] * 11)
    extra["lidar"] = {"control_path": "map.safetensors"}
    for view in extra["multiview"]["views"]:
        view["prompt"] = "Raw caption."
    view = extra["multiview"]["views"][0]
    if failure == "partial_joint":
        view["vision_path"] = "prefix.mp4"
    elif failure == "extra_hint":
        extra["depth"] = {}
    elif failure == "missing_control":
        view.pop("control_path")
    elif failure == "numeric_camera":
        view["control_path"] = "map.safetensors"
    else:
        view.pop("prompt")
    with pytest.raises(ValueError):
        contract.validate_multiview_request(extra, separate_view_text_tokenization=True)


@pytest.mark.parametrize(
    "caption",
    [
        '{"camera_view": "front", "caption": "drive"}',
        "The video is captured from a camera mounted on a car. Drive.",
        "This video is of 480x832 resolution.",
        "Follow the depth control video precisely: drive",
        "The video is 6.7 seconds long and is of 30 FPS.",
    ],
)
def test_caption_rejects_runtime_framing(caption):
    with pytest.raises(ValueError, match="runtime"):
        contract.validate_camera_caption(caption)


@pytest.mark.parametrize("caption", ['{"caption": "Driving."}', ' {"weather": "rain", "objects": []} ', "{}"])
def test_per_camera_json_objects_are_rejected_before_prompt_formatting(caption):
    extra = {"multiview": {"views": [{"camera_key": contract.COSMOS3_MADS_CAMERAS[0], "prompt": caption}]}}
    with pytest.raises(ValueError, match="JSON-object camera prompts"):
        contract.validate_multiview_request(extra, separate_view_text_tokenization=True, variable_view_count=True)


@pytest.mark.parametrize("aspect_ratio", ["auto", "16:9"])
def test_http_client_leaves_omitted_defaults_for_checkpoint(tmp_path, multiview_client, aspect_ratio):
    data, _ = multiview_client.prepare_request({"multiview": {"views": [], "aspect_ratio": aspect_ratio}}, tmp_path)
    extra = json.loads(data["extra_params"])
    assert "resolution" not in extra and "resolution" not in extra["multiview"]
    assert "width" not in data and "height" not in data and "fps" not in data
    assert "emphasize_control_in_prompt" not in extra


@pytest.mark.parametrize("index", [True, -1, "11", 0, 12])
def test_numeric_upload_rejects_invalid_or_reused_index(index):
    extra = manifest()
    extra["lidar"] = {"control_reference_index": index}
    with pytest.raises(ValueError):
        contract.resolve_multiview_uploads(extra, ["camera.mp4"] * 11 + ["map.safetensors"])


@pytest.mark.parametrize("return_output", [None, False, True])
def test_joint_client_keeps_captions_and_resolves_numeric_upload(tmp_path, multiview_client, return_output):
    camera = contract.COSMOS3_MADS_CAMERAS[3]
    for name in ("camera.mp4", "map.safetensors"):
        (tmp_path / name).write_bytes(b"input")
    request = {
        "wsm": {},
        "num_frames": 201,
        "fps": 30,
        "emphasize_control_in_prompt": False,
        "lidar": {"control_path": "map.safetensors"},
        "multiview": {"views": [{"camera_key": camera, "control_path": "camera.mp4", "prompt": "Raw."}]},
    }
    if return_output is not None:
        request["lidar"]["return_output"] = return_output
    data, paths = multiview_client.prepare_request(request, tmp_path)
    extra = json.loads(data["extra_params"])
    resolved = contract.resolve_multiview_uploads(extra, [str(path) for path in paths])
    assert resolved["lidar"] == {**request["lidar"], "control_path": str(tmp_path / "map.safetensors")}
    assert resolved["multiview"]["views"][0]["prompt"] == "Raw."
    assert resolved["emphasize_control_in_prompt"] is False
    assert data["num_frames"] == "201"


@pytest.mark.parametrize("flag", [None, 0, 1, "true", [], {}])
def test_numeric_upload_rejects_nonboolean_output_flag(flag):
    extra = manifest()
    extra["lidar"] = {"control_reference_index": 11, "return_output": flag}
    with pytest.raises(ValueError, match="return_output must be boolean"):
        contract.resolve_multiview_uploads(extra, ["camera.mp4"] * 11 + ["map.safetensors"])


@pytest.mark.parametrize("sync", [False, True])
def test_client_downloads_lidar_or_rejects_sync(tmp_path, multiview_client, monkeypatch, sync):
    from types import SimpleNamespace

    (tmp_path / "map.safetensors").write_bytes(b"control")
    manifest_path = tmp_path / "input.json"
    manifest_path.write_text(
        json.dumps(
            {
                "multiview": {"views": []},
                "lidar": {"control_path": "map.safetensors", "return_output": True},
            }
        )
    )
    output = tmp_path / "output.mp4"
    calls = []
    job = {"id": "test", "status": "completed", "lidar": {"url": "/v1/videos/test/lidar", "fps": 10}}

    class Client:
        def __init__(self, **kwargs):
            assert not sync, "Sync rejection must happen before creating an HTTP client"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            assert json.loads(kwargs["data"]["extra_params"])["lidar"]["return_output"] is True
            calls.append(url)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: job)

        def get(self, url):
            calls.append(url)
            return SimpleNamespace(raise_for_status=lambda: None, content=b"lidar" if url.endswith("lidar") else b"mp4")

    monkeypatch.setattr(multiview_client.httpx, "Client", Client)
    argv = ["client", str(manifest_path), "--output", str(output)]
    if sync:
        argv.append("--sync")
    monkeypatch.setattr(sys, "argv", argv)
    if sync:
        with pytest.raises(SystemExit) as error:
            multiview_client.main()
        assert error.value.code == 2
        assert not output.exists()
    else:
        multiview_client.main()
        assert output.read_bytes() == b"mp4"
        assert output.with_suffix(".lidar.safetensors").read_bytes() == b"lidar"
        assert json.loads(output.with_suffix(".lidar.json").read_text())["fps"] == 10
        assert calls[-2].endswith("/lidar") and calls[-1].endswith("/content")
