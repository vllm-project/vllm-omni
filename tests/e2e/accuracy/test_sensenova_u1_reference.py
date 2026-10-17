# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regression coverage for the SenseNova reference comparison contract."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from tests.e2e.accuracy import sensenova_u1 as accuracy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_committed_reference_has_valid_cases_and_pixels():
    reference = json.loads(accuracy.REFERENCE_PATH.read_text())
    assert isinstance(reference, dict), "Reference must be a JSON object"
    assert type(reference.get("schema_version")) is int and reference["schema_version"] == 1, (
        "Reference schema_version must be integer 1"
    )
    cases = reference.get("cases")
    assert isinstance(cases, dict) and set(cases) == {"t2i", "i2i"}, "Reference must contain exactly t2i and i2i"
    for case, expected_size in {"t2i": [1536, 2720], "i2i": [2560, 1664]}.items():
        record = cases[case]
        assert isinstance(record, dict) and record.get("case") == case, f"{case}: case identity mismatch"
        contract = record.get("contract")
        assert isinstance(contract, dict), f"{case}: missing input contract"
        assert contract.get("model_id") == accuracy.MODEL_ID, f"{case}: unexpected model ID"
        assert contract.get("model_revision") == accuracy.MODEL_REVISION, f"{case}: unexpected model revision"
        size = contract.get("output_size")
        assert isinstance(size, list) and len(size) == 2 and all(type(value) is int for value in size), (
            f"{case}: output size must contain two integers"
        )
        assert size == expected_size, f"{case}: expected output size {expected_size}, got {size}"
        pixels = record.get("reference_pixels")
        assert isinstance(pixels, list) and len(pixels) == 10, f"{case}: expected ten reference pixels"
        positions: set[tuple[int, int]] = set()
        for index, pixel in enumerate(pixels):
            label = f"{case} pixel {index}"
            assert isinstance(pixel, dict), f"{label}: expected a JSON object"
            position = pixel.get("position")
            assert (
                isinstance(position, list) and len(position) == 2 and all(type(value) is int for value in position)
            ), f"{label}: position must contain two integers"
            x, y = position
            assert 0 <= x < size[0] and 0 <= y < size[1], f"{label}: position {position} is outside {size}"
            assert (x, y) not in positions, f"{label}: duplicate position {position}"
            positions.add((x, y))
            rgb = pixel.get("rgb")
            assert (
                isinstance(rgb, list)
                and len(rgb) == 3
                and all(type(channel) is int and 0 <= channel <= 255 for channel in rgb)
            ), f"{label}: RGB must contain three integers in [0, 255]"


@pytest.fixture
def recorded_case(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSENOVA_U1_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(
        accuracy, "_environment_metadata", lambda: {"gpu": "NVIDIA H100", "packages": {"torch": "test"}}
    )
    monkeypatch.setattr(accuracy, "_source_metadata", lambda: {"git_commit": "unit-test"})
    params = SimpleNamespace(
        height=1, width=10, seed=42, num_inference_steps=50, extra_args={"cfg_interval": (0.0, 1.0)}
    )
    record = accuracy.record_accuracy(
        Image.new("RGB", (10, 1), (100, 120, 140)),
        case="i2i",
        prompt="unit-test prompt",
        sampling_params=params,
        pixel_positions=[(x, 0) for x in range(10)],
        input_image=Image.new("RGB", (10, 1), (30, 40, 50)),
    )
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"schema_version": 1, "cases": {"i2i": record}}))
    return record, path


@pytest.mark.parametrize("delta", [-10, 10])
def test_reference_accepts_tolerance_boundary(recorded_case, delta):
    record, path = recorded_case
    record["reference_pixels"][-1]["rgb"][1] += delta
    accuracy.assert_accuracy(record, path)


@pytest.mark.parametrize("delta", [-11, 11])
def test_reference_rejects_one_channel_outside_tolerance(recorded_case, delta):
    record, path = recorded_case
    record["reference_pixels"][-1]["rgb"][1] += delta
    with pytest.raises(AssertionError, match=r"Pixel mismatch at \[9, 0\].*tolerance 10"):
        accuracy.assert_accuracy(record, path)


@pytest.mark.parametrize(
    "field",
    ["model_id", "model_revision", "prompt", "sampling_params", "output_size", "input_image"],
)
def test_reference_rejects_changed_inputs(recorded_case, field):
    record, path = recorded_case
    record["contract"][field] = "changed"
    with pytest.raises(AssertionError, match=f"input mismatch: {field}"):
        accuracy.assert_accuracy(record, path)


def test_reference_records_input_pixels_and_output_hash(recorded_case):
    record, path = recorded_case
    assert record["contract"]["input_image"] == {
        "size": [10, 1],
        "rgb_sha256": hashlib.sha256(bytes([30, 40, 50]) * 10).hexdigest(),
    }
    assert record["contract"]["sampling_params"]["extra_args"]["cfg_interval"] == [0.0, 1.0]
    assert record["output_png_sha256"] == hashlib.sha256(path.with_name("i2i.png").read_bytes()).hexdigest()
    assert json.loads(path.with_name("i2i.json").read_text()) == record


def test_reference_allows_testing_changed_software_and_source(recorded_case):
    record, path = recorded_case
    record["environment"]["packages"]["torch"] = "new version"
    record["source"]["git_commit"] = "new commit"
    accuracy.assert_accuracy(record, path)


def test_reference_missing_does_not_create_a_golden(recorded_case):
    record, path = recorded_case
    path.unlink()
    with pytest.raises(AssertionError, match="Missing reviewed H100 reference"):
        accuracy.assert_accuracy(record, path)
    assert not path.exists()
    assert path.with_name("i2i.png").is_file()
    assert path.with_name("i2i.json").is_file()


def test_reference_rejects_missing_or_reordered_points(recorded_case):
    record, path = recorded_case
    incomplete = copy.deepcopy(record)
    incomplete["reference_pixels"].pop()
    with pytest.raises(AssertionError, match="Expected ten reference pixels"):
        accuracy.assert_accuracy(incomplete, path)
    record["reference_pixels"].reverse()
    with pytest.raises(AssertionError, match="reference positions differ"):
        accuracy.assert_accuracy(record, path)
