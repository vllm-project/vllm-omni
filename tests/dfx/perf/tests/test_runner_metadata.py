"""Tests for DFX runner metadata field exclusion."""

import json

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_task_excluded_from_cli_args():
    """'task' field must not become --task CLI arg."""
    params = {
        "task": "voice_clone",
        "dataset_name": "seed-tts",
        "backend": "openai-audio-speech",
        "endpoint": "/v1/audio/speech",
        "percentile-metrics": "audio_rtf,audio_ttfp",
        "baseline": {"mean_audio_rtf": [0.5]},
    }
    exclude_keys = {"request_rate", "baseline", "num_prompts", "max_concurrency", "task", "enabled", "eval_phase"}
    args = []
    for key, value in params.items():
        if key in exclude_keys or value is None:
            continue
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool) and value:
            args.append(arg_name)
        elif isinstance(value, dict):
            args.extend([arg_name, json.dumps(value)])
        elif not isinstance(value, bool):
            args.extend([arg_name, str(value)])
    assert "--task" not in args
    assert "--enabled" not in args
    assert "--dataset-name" in args


def test_enabled_false_entry_is_skipped():
    """benchmark_params entry with enabled=false should be skipped."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from tests.dfx.conftest import create_test_parameter_mapping

    configs = [
        {
            "test_name": "test_model",
            "server_params": {"model": "some/model"},
            "benchmark_params": [
                {
                    "task": "voice_clone",
                    "enabled": True,
                    "dataset_name": "seed-tts",
                    "backend": "openai-audio-speech",
                    "endpoint": "/v1/audio/speech",
                    "num_prompts": [10],
                    "max_concurrency": [1],
                    "percentile-metrics": "audio_rtf",
                    "baseline": {},
                },
                {
                    "task": "voice_design",
                    "enabled": False,
                    "dataset_name": "seed-tts-design",
                    "backend": "openai-audio-speech",
                    "endpoint": "/v1/audio/speech",
                    "num_prompts": [5],
                    "max_concurrency": [1],
                    "percentile-metrics": "audio_rtf",
                    "baseline": {},
                },
            ],
        }
    ]
    mapping = create_test_parameter_mapping(configs)
    params = mapping["test_model"]["benchmark_params"]
    # Only the enabled=True entry should appear
    assert len(params) == 1
    assert params[0].get("task") == "voice_clone"
