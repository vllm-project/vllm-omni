# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for diffusion dataset loaders (Issue #6873)."""

import json
import tempfile
from argparse import Namespace
from unittest.mock import patch

import pytest

from vllm_omni.benchmarks.data_modules.diffusion_dataset import (
    load_diffusion_samples,
)
from vllm_omni.benchmarks.patch.patch import get_samples

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cpu,
]


def _make_args(**overrides):
    defaults = dict(
        diffusion_dataset="random",
        num_prompts=3,
        seed=42,
        dataset_path=None,
        random_request_config=None,
        width=None,
        height=None,
        num_inference_steps=None,
        num_frames=None,
        fps=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_load_diffusion_samples_returns_zero_len_requests():
    samples = load_diffusion_samples(_make_args())
    assert len(samples) == 3
    for s in samples:
        assert s.prompt_len == 0
        assert s.expected_output_len == 0


def test_random_respects_num_prompts():
    samples = load_diffusion_samples(_make_args(num_prompts=7))
    assert len(samples) == 7


def test_custom_requires_dataset_path():
    with pytest.raises(ValueError, match="--dataset-path"):
        load_diffusion_samples(_make_args(diffusion_dataset="custom"))


def test_vbench_download_failure_raises(tmp_path, monkeypatch):
    """A failed VBench download must fail loudly, not substitute canned prompts."""
    args = _make_args(diffusion_dataset="vbench", dataset_path=None, num_prompts=3)
    monkeypatch.setattr(
        "vllm_omni.benchmarks.data_modules.diffusion_dataset._CACHE_DIR",
        str(tmp_path),
    )

    def _fail(url, timeout):
        raise OSError("network down")

    monkeypatch.setattr(
        "vllm_omni.benchmarks.data_modules.diffusion_dataset.http_requests.get",
        _fail,
    )
    with pytest.raises(RuntimeError, match="Failed to download VBench prompts"):
        load_diffusion_samples(args)


def test_custom_loads_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"prompt": "a red car"}) + "\n")
        f.write(json.dumps({"prompt": "a blue sky"}) + "\n")
        f.flush()
        samples = load_diffusion_samples(_make_args(diffusion_dataset="custom", dataset_path=f.name, num_prompts=0))
    assert len(samples) == 2
    assert samples[0].prompt == "a red car"


def test_dataset_name_selects_loader_when_diffusion_dataset_unset():
    """An explicit --dataset-name that names a diffusion loader selects it
    when the --diffusion-dataset flag is absent."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"prompt": "a red car"}) + "\n")
        f.flush()
        args = _make_args(
            diffusion_dataset=None,
            dataset_name="custom",
            dataset_path=f.name,
            num_prompts=0,
        )
        samples = load_diffusion_samples(args)
    assert len(samples) == 1
    assert samples[0].prompt == "a red car"


def test_unknown_dataset_name_falls_back_to_random():
    """A --dataset-name that is not a diffusion loader keeps the random
    default (e.g. the documented random / random-mm recipes)."""
    args = _make_args(diffusion_dataset=None, dataset_name="random-mm", num_prompts=3)
    samples = load_diffusion_samples(args)
    assert len(samples) == 3
    assert samples[0].prompt.startswith("Random prompt")


def test_get_samples_routes_to_diffusion_loader():
    args = _make_args(endpoint="/v1/images/generations", backend="")
    samples = get_samples(args, tokenizer=None)
    assert len(samples) == 3
    assert all(s.prompt_len == 0 for s in samples)


@pytest.mark.parametrize("endpoint", ["/v1/images/generations", "/v1/videos"])
def test_get_samples_diffusion_routing_overrides_explicit_dataset_name(endpoint):
    """Diffusion endpoints route to the diffusion loader even when an
    explicit --dataset-name is present (the documented recipes pair
    --dataset-name random / random-mm with these endpoints)."""
    args = _make_args(endpoint=endpoint, backend="", dataset_name="random")
    with patch(
        "vllm_omni.benchmarks.data_modules.diffusion_dataset.load_diffusion_samples",
        return_value=["diffusion-sentinel"],
    ) as mock_loader:
        samples = get_samples(args, tokenizer=None)
    assert samples == ["diffusion-sentinel"]
    mock_loader.assert_called_once_with(args)
