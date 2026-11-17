# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for diffusion benchmark tokenizer bypass (Issue #6873)."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.benchmarks.serve import ASYNC_REQUEST_FUNCS

from vllm_omni.benchmarks.serve import (
    _DIFFUSION_BACKENDS,
    _DIFFUSION_ENDPOINTS,
    _prepare_diffusion_args,
    is_diffusion_benchmark,
    main,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.benchmark,
]


def test_diffusion_backends_are_registered():
    """Every diffusion dispatch key (endpoint paths and backend aliases)
    must exist in ASYNC_REQUEST_FUNCS, or benchmark dispatch fails with
    'Unknown backend' at request time."""
    for backend in sorted(_DIFFUSION_ENDPOINTS | _DIFFUSION_BACKENDS):
        assert backend in ASYNC_REQUEST_FUNCS, f"{backend!r} not registered in patch.py"


@pytest.mark.parametrize("endpoint", sorted(_DIFFUSION_ENDPOINTS))
def test_diffusion_endpoints_detected(endpoint):
    assert is_diffusion_benchmark(Namespace(endpoint=endpoint, backend=""))


@pytest.mark.parametrize("endpoint", sorted(_DIFFUSION_ENDPOINTS))
def test_diffusion_endpoints_detected_without_leading_slash(endpoint):
    assert is_diffusion_benchmark(Namespace(endpoint=endpoint.lstrip("/"), backend=""))


@pytest.mark.parametrize("backend", sorted(_DIFFUSION_BACKENDS))
def test_diffusion_backends_detected(backend):
    assert is_diffusion_benchmark(Namespace(endpoint="", backend=backend))


@pytest.mark.parametrize("endpoint", sorted(_DIFFUSION_ENDPOINTS))
def test_diffusion_endpoints_detected_as_backend_spelling(endpoint):
    """Endpoint paths are also valid --backend spellings (registered adapters)."""
    assert is_diffusion_benchmark(Namespace(endpoint="", backend=endpoint))


@pytest.mark.parametrize(
    "endpoint,backend",
    [
        ("/v1/chat/completions", ""),
        ("", "openai-chat-omni"),
        ("", ""),
    ],
)
def test_non_diffusion_not_detected(endpoint, backend):
    assert not is_diffusion_benchmark(Namespace(endpoint=endpoint, backend=backend))


def test_prepare_diffusion_args_sets_skip_tokenizer():
    args = Namespace()
    _prepare_diffusion_args(args)
    assert args.skip_tokenizer_init is True


def test_prepare_diffusion_args_respects_explicit_tokenizer():
    args = Namespace(tokenizer="gpt2")
    _prepare_diffusion_args(args)
    assert not getattr(args, "skip_tokenizer_init", False)


def _make_bench_args(**overrides):
    """Build a minimal Namespace that survives up to the tokenizer check in upstream main_async."""
    defaults = dict(
        endpoint="/v1/images/generations",
        backend="",
        seed_tts_wer_eval=False,
        seed_tts_wer_save_items=False,
        daily_omni_save_eval_items=False,
        videomme_save_eval_items=False,
        omni_request_timeout_s=None,
        print_stage=False,
        extra_body=None,
        dataset_name=None,
        dataset_path=None,
        tokenizer=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
        # Fields needed by upstream main_async before the tokenizer check
        max_concurrency=None,
        seed=0,
        plot_timeline=False,
        ramp_up_strategy=None,
        base_url="http://localhost:8000",
        model="test-model",
        served_model_name="test-model",
        skip_tokenizer_init=False,
        header=None,
        insecure=False,
        label=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_main_skips_tokenizer_init_for_diffusion():
    """Verify get_tokenizer is never called when main routes a diffusion endpoint."""
    args = _make_bench_args()
    with patch("vllm.benchmarks.serve.get_tokenizer") as mock_tokenizer:
        mock_tokenizer.side_effect = AssertionError("get_tokenizer should not be called")
        with pytest.raises(ValueError, match="dataset-name"):
            main(args)
    mock_tokenizer.assert_not_called()


def test_main_does_not_skip_tokenizer_when_explicit():
    """With --tokenizer set, get_tokenizer must be called even for diffusion endpoints."""
    args = _make_bench_args(tokenizer="gpt2")
    with patch("vllm.benchmarks.serve.get_tokenizer", return_value="fake") as mock_tokenizer:
        with pytest.raises(ValueError, match="dataset-name"):
            main(args)
    mock_tokenizer.assert_called_once()


def test_main_slashless_endpoint_skips_tokenizer():
    """Slashless endpoints like 'v1/videos' must also bypass tokenizer init."""
    args = _make_bench_args(endpoint="v1/videos")
    with patch("vllm.benchmarks.serve.get_tokenizer") as mock_tokenizer:
        mock_tokenizer.side_effect = AssertionError("get_tokenizer should not be called")
        with pytest.raises(ValueError, match="dataset-name"):
            main(args)
    mock_tokenizer.assert_not_called()


def test_main_backend_spelling_skips_tokenizer():
    """--backend /v1/videos (endpoint path as backend) must also bypass tokenizer init."""
    args = _make_bench_args(endpoint="/v1/completions", backend="/v1/videos")
    with patch("vllm.benchmarks.serve.get_tokenizer") as mock_tokenizer:
        mock_tokenizer.side_effect = AssertionError("get_tokenizer should not be called")
        with pytest.raises(ValueError, match="dataset-name"):
            main(args)
    mock_tokenizer.assert_not_called()


def test_main_image_edit_keeps_tokenizer_init():
    """Image edits with random-mm and no explicit --tokenizer must not skip
    tokenizer init: the retained dataset path samples via tokenizer.vocab_size."""
    args = _make_bench_args(endpoint="/v1/images/edits", dataset_name="random-mm")
    with patch("vllm.benchmarks.serve.get_tokenizer", return_value=MagicMock()) as mock_tokenizer:
        # Sampling proceeds with the mock tokenizer and eventually fails on it;
        # any exception past the tokenizer check is fine — the point is that
        # the tokenizer was loaded and the bypass did not fire.
        with pytest.raises(Exception):
            main(args)
    mock_tokenizer.assert_called_once()
    assert not args.skip_tokenizer_init


def test_main_image_edit_backend_keeps_tokenizer_init():
    """Same regression via the backend name instead of the endpoint."""
    args = _make_bench_args(backend="openai-image-edits-omni", dataset_name="random-mm")
    with patch("vllm.benchmarks.serve.get_tokenizer", return_value=MagicMock()) as mock_tokenizer:
        with pytest.raises(Exception):
            main(args)
    mock_tokenizer.assert_called_once()
    assert not args.skip_tokenizer_init


def test_main_image_edit_endpoint_as_backend_keeps_tokenizer_init():
    """--backend /v1/images/edits must keep tokenizer init like the endpoint spelling."""
    args = _make_bench_args(backend="/v1/images/edits", dataset_name="random-mm")
    with patch("vllm.benchmarks.serve.get_tokenizer", return_value=MagicMock()) as mock_tokenizer:
        with pytest.raises(Exception):
            main(args)
    mock_tokenizer.assert_called_once()
    assert not args.skip_tokenizer_init


class _ReachedBenchmarkError(Exception):
    """Sentinel raised by the mocked benchmark() to stop main_async."""


def test_main_tokenizer_less_diffusion_run_reaches_sampling(tmp_path):
    """A tokenizer-less diffusion run must get past tokenizer init, dataset
    validation, and sampling, and reach the benchmark loop with tokenizer=None.

    This exercises the full region of upstream main_async between
    ``skip_tokenizer_init`` and ``benchmark()`` — the previously untested
    stretch behind the ``--dataset-name`` stop point of the other tests.
    """
    dataset_path = tmp_path / "prompts.jsonl"
    dataset_path.write_text(
        '{"prompt": "a cat sitting on a mat"}\n{"prompt": "a red barn in a field"}\n',
        encoding="utf-8",
    )
    args = _make_bench_args(
        dataset_name="custom",
        dataset_path=str(dataset_path),
        num_prompts=2,
        custom_output_len=8,
        skip_chat_template=True,
        disable_shuffle=True,
        no_oversample=True,
        # Post-dataset fields read by upstream main_async before benchmark()
        input_len=None,
        output_len=None,
        self_timed=None,
        goodput=None,
        top_p=None,
        top_k=None,
        min_p=None,
        temperature=None,
        frequency_penalty=None,
        presence_penalty=None,
        repetition_penalty=None,
        percentile_metrics=None,
        metric_percentiles="99",
        logprobs=None,
        request_rate=1.0,
        burstiness=1.0,
        disable_tqdm=True,
        num_warmups=0,
        profile=False,
        ignore_eos=False,
        lora_modules=None,
        lora_assignment=None,
        ramp_up_start_rps=None,
        ramp_up_end_rps=None,
        ready_check_timeout_sec=None,
        probe_request_rate=None,
    )

    captured: dict = {}

    async def _fake_benchmark(**kwargs):
        captured.update(kwargs)
        raise _ReachedBenchmarkError

    with patch("vllm.benchmarks.serve.get_tokenizer") as mock_tokenizer:
        mock_tokenizer.side_effect = AssertionError("get_tokenizer should not be called")
        with patch("vllm.benchmarks.serve.benchmark", side_effect=_fake_benchmark):
            with pytest.raises(_ReachedBenchmarkError):
                main(args)

    mock_tokenizer.assert_not_called()
    assert args.skip_tokenizer_init is True
    # The run reached the benchmark loop with a tokenizer-less config and
    # sampled requests from the dataset on the way.
    assert captured["tokenizer"] is None
    assert len(captured["input_requests"]) == 2
    # Content check: the custom JSONL prompts must be what reaches the
    # benchmark, not synthetic random filler.
    assert captured["input_requests"][0].prompt == "a cat sitting on a mat"
