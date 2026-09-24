# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import json
import os
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

from vllm_omni.entrypoints.cli.benchmark.cli_args import (
    add_omni_args,
    extend_omni_choices,
    preprocess_serve_args,
    update_omni_help,
)
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def test_it2i_dataset_uses_upstream_warmups(monkeypatch, tmp_path):
    """Exercise real dataset/benchmark code; replace only the HTTP backend."""
    from vllm_omni.benchmarks import serve
    from vllm_omni.benchmarks.patch import patch
    from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand

    dataset_path = Path(__file__).parents[1] / "assets/hunyuan_image3/it2i.jsonl"
    sample = json.loads(dataset_path.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--model",
            "test-model",
            "--endpoint",
            "/v1/images/edits",
            "--dataset-name",
            "custom_image",
            "--dataset-path",
            str(dataset_path),
            "--skip-tokenizer-init",
            "--disable-shuffle",
            "--disable-tqdm",
            "--num-prompts",
            "8",
            "--num-warmups",
            "2",
            "--max-concurrency",
            "1",
            "--ready-check-timeout-sec",
            "0",
            "--percentile-metrics",
            "e2el",
            "--extra-body",
            '{"seed":42,"guidance_scale":2.5,"bot_task":"think_recaption"}',
            "--save-result",
            "--result-dir",
            str(tmp_path),
            "--result-filename",
            "result.json",
        ]
    )
    requests = []

    async def request_func(request_func_input, session, pbar=None):
        requests.append(request_func_input)
        if pbar is not None:
            pbar.update(1)
        return patch.MixRequestFuncOutput(
            success=True,
            prompt_len=1,
            output_tokens=1,
            image_count=1,
            start_time=time.perf_counter(),
            latency=99.0 if len(requests) <= 2 else 0.1,
        )

    monkeypatch.setitem(patch.ASYNC_REQUEST_FUNCS, "/v1/images/edits", request_func)
    result = serve.main(args)
    assert len(requests) == 10
    for request in requests:
        assert request.prompt == sample["prompt"]
        assert [image["image_url"]["url"] for image in request.multi_modal_content] == sample["image_files"]
        assert request.extra_body == args.extra_body
    assert result["completed"] == 8
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["completed"] == 8
    assert saved["mean_e2el_ms"] == pytest.approx(100.0)  # Excludes both 99-second warmups.


@pytest.mark.parametrize(
    "dataset_name",
    [
        "daily-omni",
        "omniinteract",
        "seed-tts",
        "seed-tts-text",
        "seed-tts-design",
        "ttsd",
        "sound-effect",
    ],
)
def test_extend_omni_choices_updates_tracking_parser_shadow(dataset_name: str) -> None:
    parser = TrackingArgumentParser()
    parser.add_argument("--dataset-name", choices=["random"])
    parser.add_argument("--backend", choices=["openai-chat-omni"])

    extend_omni_choices(parser)

    args = parser.parse_args(
        [
            "--dataset-name",
            dataset_name,
            "--backend",
            "openai-image-edits-omni",
        ]
    )

    assert args.dataset_name == dataset_name
    assert args.backend == "openai-image-edits-omni"
    assert args.explicit_keys == {"dataset_name", "backend"}


@pytest.mark.parametrize(
    ("argv", "dest", "expected"),
    [
        (["--print-stage"], "print_stage", True),
        (["--daily-omni-input-mode", "audio"], "daily_omni_input_mode", "audio"),
        (["--omniinteract-subsets", "1qna"], "omniinteract_subsets", ["1qna"]),
        (["--seed-tts-locale", "zh"], "seed_tts_locale", "zh"),
    ],
)
def test_add_omni_args_registers_arguments_on_tracking_parser(
    argv: list[str],
    dest: str,
    expected: object,
) -> None:
    parser = TrackingArgumentParser()

    add_omni_args(parser)

    args = parser.parse_args(argv)
    assert getattr(args, dest) == expected
    assert args.explicit_keys == {dest}


def test_add_omni_args_preserves_implicit_defaults() -> None:
    parser = TrackingArgumentParser()

    add_omni_args(parser)

    args = parser.parse_args([])
    assert args.print_stage is False
    assert args.daily_omni_input_mode == "all"
    assert args.omniinteract_subsets == ["1q1a", "1q1a_math", "1qna"]
    assert args.seed_tts_locale == "en"
    assert args.explicit_keys == set()


def test_update_omni_help_updates_upstream_actions() -> None:
    parser = TrackingArgumentParser()
    parser.add_argument("--num-prompts", default=1000, help="Number of prompts.")
    parser.add_argument("--percentile-metrics", help="Upstream percentile help.")
    parser.add_argument("--random-mm-limit-mm-per-prompt", help="Upstream limit help.")
    parser.add_argument("--random-mm-bucket-config", help="Upstream bucket help.")

    update_omni_help(parser)

    actions = {action.dest: action for action in parser._actions}
    assert "OmniInteract uses 3" in actions["num_prompts"].help
    assert all(metric in actions["percentile_metrics"].help for metric in ("ttfc", "tpop", "audio_rtf"))
    assert "probabilities are renormalized" in actions["random_mm_limit_mm_per_prompt"].help
    assert "Currently allows for 3 modalities" in actions["random_mm_bucket_config"].help


@pytest.mark.parametrize(
    ("backend", "extra_body", "bot_task", "expected"),
    [
        ("openai-chat-omni", None, "think", {}),
        ("openai-image-edits-omni", None, "think", {"bot_task": "think"}),
        ("openai-image-edits-omni", {"bot_task": "recaption"}, "think", {"bot_task": "recaption"}),
        ("openai-image-edits-omni", {"custom": True}, None, {"custom": True}),
    ],
)
def test_preprocess_serve_args_merges_bot_task_without_overriding_extra_body(
    backend: str,
    extra_body: dict | None,
    bot_task: str | None,
    expected: dict,
) -> None:
    args = argparse.Namespace(backend=backend, extra_body=extra_body, bot_task=bot_task)

    preprocess_serve_args(args)

    assert args.extra_body == expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], 3),
        (["--num-prompts", "0"], 0),
        (["--num-prompts", "12"], 12),
    ],
)
def test_preprocess_serve_args_applies_safe_omniinteract_prompt_default(
    argv: list[str],
    expected: int,
) -> None:
    parser = TrackingArgumentParser()
    parser.add_argument("--dataset-name", default="omniinteract")
    parser.add_argument("--backend", default="openai-realtime-duplex")
    parser.add_argument("--endpoint", default="/v1/realtime")
    parser.add_argument("--omniinteract-ref-audio", default="reference.wav")
    parser.add_argument("--num-prompts", type=int, default=1000)
    args = parser.parse_args(argv)

    preprocess_serve_args(args)

    assert args.num_prompts == expected


@pytest.mark.parametrize(
    ("argv", "expected_extra_body", "expected_explicit"),
    [
        (
            [
                "--backend",
                "openai-image-edits-omni",
                "--print-stage",
                "--image-edits-bot-task",
                "recaption",
            ],
            {"bot_task": "recaption"},
            {"backend", "print_stage", "bot_task"},
        ),
        (
            [
                "--endpoint",
                "/v1/images/edits",
                "--image-edits-bot-task",
                "think",
            ],
            {"bot_task": "think"},
            {"endpoint", "bot_task"},
        ),
        (
            [
                "--backend",
                "/v1/images/edits",
                "--image-edits-bot-task",
                "recaption",
            ],
            {"bot_task": "recaption"},
            {"backend", "bot_task"},
        ),
        (
            ["--extra-body", '{"bot_task":"vanilla"}'],
            {"bot_task": "vanilla"},
            {"extra_body"},
        ),
    ],
)
def test_omni_args_parse_and_preprocess(
    argv: list[str],
    expected_extra_body: dict,
    expected_explicit: set[str],
) -> None:
    parser = TrackingArgumentParser()
    parser.add_argument("--extra-body", type=json.loads, default=None)
    parser.add_argument("--backend", default="openai-chat-omni")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    add_omni_args(parser)

    args = parser.parse_args(argv)
    preprocess_serve_args(args)

    assert args.extra_body == expected_extra_body
    assert args.explicit_keys == expected_explicit


def test_bench_serve_cli_mocks_http_request(tmp_path: Path):
    num_prompts = 5
    port = 18000
    result_filename = "bench-result.json"
    calls_filename = "http-post-calls.json"
    result_path = tmp_path / result_filename
    calls_path = tmp_path / calls_filename

    sitecustomize_path = tmp_path / "sitecustomize.py"
    sitecustomize_path.write_text(
        textwrap.dedent(
            """
            import atexit
            import json
            import os

            POST_CALLS = []
            SSE_CHUNKS = [
                b'data: {"choices":[{"delta":{"content":"hi"}}],"modality":"text"}\\n\\n',
                b'data: {"choices":[{"delta":{"content":" there"}}],"modality":"text","metrics":{"num_tokens_out":4,"num_tokens_in":5}}\\n\\n',
                b"data: [DONE]\\n\\n",
            ]

            class _Content:
                def __init__(self, chunks):
                    self._chunks = chunks

                async def iter_any(self):
                    for chunk in self._chunks:
                        yield chunk

            class MockResponse:
                def __init__(self, status=200, reason="OK", chunks=None):
                    self.status = status
                    self.reason = reason
                    self.content = _Content(chunks or SSE_CHUNKS)
                    self._json_data = None
                    self._text_data = None

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def text(self):
                    if self._text_data is not None:
                        return self._text_data
                    return json.dumps({"tokens": [1, 2, 3, 4, 5]})

                async def json(self):
                    if self._json_data is not None:
                        return self._json_data
                    return {"tokens": [1, 2, 3, 4, 5]}

                def raise_for_status(self):
                    pass

            class MockClientSession:
                def __init__(self, *args, **kwargs):
                    pass

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                def post(self, url=None, *args, **kwargs):
                    if url is not None:
                        POST_CALLS.append(url)
                    return MockResponse(status=200)

                async def get(self, url=None, *args, **kwargs):
                    if url is not None:
                        POST_CALLS.append(url)
                    return MockResponse(status=200)

                async def close(self):
                    return None

            # Patch globally so modules loaded after this also see the mock
            import aiohttp
            aiohttp.ClientSession = MockClientSession
            aiohttp.TCPConnector = lambda *args, **kwargs: object()

            # Also patch in the patch module
            import vllm_omni.benchmarks.patch.patch as patch_mod
            patch_mod.aiohttp.ClientSession = MockClientSession
            patch_mod.aiohttp.TCPConnector = lambda *args, **kwargs: object()

            calls_file = os.environ.get("VLLM_OMNI_TEST_POST_CALLS_FILE")

            if calls_file:
                def _write_calls():
                    with open(calls_file, "w", encoding="utf-8") as f:
                        json.dump({"requested_urls": POST_CALLS}, f)

                atexit.register(_write_calls)
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    env["VLLM_OMNI_TEST_POST_CALLS_FILE"] = str(calls_path)

    cmd = [
        "vllm",
        "bench",
        "serve",
        "--omni",
        "--model",
        "Qwen/Qwen2.5-Omni-7B",
        "--port",
        str(port),
        "--dataset-name",
        "random",
        "--random-input-len",
        "32",
        "--random-output-len",
        "4",
        "--num-prompts",
        str(num_prompts),
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat-omni",
        "--disable-tqdm",
        "--num-warmups",
        "0",
        "--ready-check-timeout-sec",
        "0",
        "--save-result",
        "--result-dir",
        str(tmp_path),
        "--result-filename",
        result_filename,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, f"CLI failed: stdout={proc.stdout}\nstderr={proc.stderr}"
    print(f"CLI output: {proc.stdout}")
    assert result_path.exists()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert calls_path.exists()
    calls = json.loads(calls_path.read_text(encoding="utf-8"))

    expected_url = f"http://127.0.0.1:{port}/v1/chat/completions"
    requested_urls = calls["requested_urls"]
    # Count only benchmark requests (to the chat completions endpoint),
    # excluding tokenize/detokenize alignment calls from the upstream.
    bench_requests = [url for url in requested_urls if url == expected_url]
    sent_bench_requests = len(bench_requests)

    assert result["completed"] == sent_bench_requests == num_prompts, (
        f"completed={result['completed']}, bench_requests={sent_bench_requests}, all_requested_urls={requested_urls}"
    )
    assert bench_requests
    assert all(url == expected_url for url in bench_requests), f"Unexpected target URLs: {bench_requests}"


def test_omni_request_timeout_s_flag_defaults_and_parses() -> None:
    parser = TrackingArgumentParser()
    add_omni_args(parser)

    args = parser.parse_args([])
    assert args.omni_request_timeout_s is None

    args = parser.parse_args(["--omni-request-timeout-s", "0"])
    assert args.omni_request_timeout_s == 0.0

    args = parser.parse_args(["--omni-request-timeout-s", "3600"])
    assert args.omni_request_timeout_s == 3600.0
