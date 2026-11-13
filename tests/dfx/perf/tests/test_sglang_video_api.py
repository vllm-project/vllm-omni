# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check the comparison image's video API without model/CUDA dependencies."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def sglang_source():
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        pytest.skip("Run in the SGLang comparison environment")
    assert spec is not None and spec.origin is not None
    return Path(spec.origin).parent


@pytest.fixture(scope="module")
def video_api(sglang_source):
    source = sglang_source / "multimodal_gen/runtime/entrypoints/openai/video_api.py"
    return ast.parse(source.read_text())


def test_common_client_image_reference_preserves_input(video_api):
    helper = next(n for n in video_api.body if isinstance(n, ast.FunctionDef) and n.name == "_image_reference_url")
    namespace: dict[str, object] = {"json": json}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "video_api", "exec"), namespace)
    parse = namespace["_image_reference_url"]
    assert callable(parse)
    image = "data:image/png;base64,unchanged-by-adapter"
    assert parse(json.dumps({"image_url": image})) == image
    for invalid in ("[]", "{}", '{"image_url": ""}', '{"image_url": 42}', "malformed"):
        with pytest.raises(ValueError):
            parse(invalid)


@pytest.mark.parametrize("content_type", ["application/x-www-form-urlencoded", "multipart/form-data; boundary=x"])
def test_common_client_form_encoding_uses_form_path(video_api, content_type):
    route = next(n for n in video_api.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "create_video")
    assignment = next(
        n
        for n in route.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "is_multipart" for t in n.targets)
    )
    namespace: dict[str, object] = {"content_type": content_type}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "video_api", "exec"), namespace)
    assert namespace["is_multipart"] is True


@pytest.mark.parametrize("content_type", ["application/x-www-form-urlencoded", "multipart/form-data; boundary=x"])
def test_common_client_image_reference_reaches_route_image_input(video_api, content_type):
    route = next(n for n in video_api.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "create_video")
    assert "image_reference" in {argument.arg for argument in route.args.args}
    helper = next(n for n in video_api.body if isinstance(n, ast.FunctionDef) and n.name == "_image_reference_url")
    form_assignment = next(
        n
        for n in route.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "is_multipart" for t in n.targets)
    )
    conversion = next(
        n
        for n in route.body
        if isinstance(n, ast.If)
        and any(isinstance(value, ast.Name) and value.id == "image_reference" for value in ast.walk(n.test))
    )
    image_sources = next(
        n
        for n in ast.walk(route)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "image_sources" for t in n.targets)
    )
    image = "data:image/png;base64,unchanged-by-adapter"
    namespace = {
        "json": json,
        "content_type": content_type,
        "input_reference": None,
        "reference_url": None,
        "image_reference": json.dumps({"image_url": image}),
        "merge_image_input_list": lambda *values: [value for value in values if value is not None],
    }
    exec(
        compile(
            ast.Module(body=[helper, form_assignment, conversion, image_sources], type_ignores=[]), "video_api", "exec"
        ),
        namespace,
    )
    assert namespace["image_sources"] == [image]


def test_wan_request_declares_its_expert_boundary(sglang_source):
    tree = ast.parse((sglang_source / "multimodal_gen/configs/sample/wan.py").read_text())
    sampling = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Wan2_2_I2V_A14B_SamplingParam")
    declaration = next(
        n for n in sampling.body if isinstance(n, ast.FunctionDef) and n.name == "video_request_extra_fields"
    )
    declaration.decorator_list = []
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), "wan_sampling", "exec"), namespace)
    fields = namespace["video_request_extra_fields"]
    assert callable(fields)
    assert fields(None) == frozenset({"boundary_ratio"})
