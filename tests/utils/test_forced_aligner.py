import numpy as np
import pytest

from vllm_omni.utils import forced_aligner


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_build_prompt_has_boundary_timestamp_markers():
    prompt = forced_aligner._build_prompt("hello world.")

    assert prompt.count("<timestamp>") == 4
    assert "hello<timestamp><timestamp>world" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_decode_timestamps_maps_boundary_bins_to_words():
    logits = np.zeros((4, 5), dtype=np.float32)
    logits[0, 0] = 1.0
    logits[1, 2] = 1.0
    logits[2, 2] = 1.0
    logits[3, 4] = 1.0

    timestamps = forced_aligner._decode_timestamps(
        logits=logits,
        text="hello world",
        timestamp_positions=[0, 1, 2, 3],
        classify_num=5,
        audio_duration_ms=1000,
    )

    assert timestamps == [
        forced_aligner.WordTimestamp("hello", 0, 400),
        forced_aligner.WordTimestamp("world", 400, 800),
    ]


def test_decode_timestamps_rejects_marker_count_mismatch():
    logits = np.zeros((2, 5), dtype=np.float32)

    timestamps = forced_aligner._decode_timestamps(
        logits=logits,
        text="hello world",
        timestamp_positions=[0, 1],
        classify_num=5,
        audio_duration_ms=1000,
    )

    assert timestamps == []


def test_build_config_from_yaml(tmp_path):
    cfg = tmp_path / "forced_aligner.yaml"
    cfg.write_text(
        """
forced_aligner:
  model: Qwen/Qwen3-ForcedAligner-0.6B
  gpu_memory_utilization: 0.42
  dtype: float16
  max_model_len: 2048
  trust_remote_code: false
""",
        encoding="utf-8",
    )
    args = type("Args", (), {"forced_aligner": None, "forced_aligner_config": str(cfg)})()

    out = forced_aligner.build_forced_aligner_config(args)

    assert out == forced_aligner.ForcedAlignerConfig(
        model="Qwen/Qwen3-ForcedAligner-0.6B",
        runner="pooling",
        architecture="Qwen3ASRForcedAlignerForTokenClassification",
        pooling_task="token_classify",
        gpu_memory_utilization=0.42,
        dtype="float16",
        max_model_len=2048,
        trust_remote_code=False,
    )


def test_build_config_cli_model_overrides_yaml(tmp_path):
    cfg = tmp_path / "forced_aligner.yaml"
    cfg.write_text(
        "forced_aligner:\n  model: old\n  gpu_memory_utilization: 0.2\n  dtype: float16\n",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "forced_aligner": "new",
            "forced_aligner_config": str(cfg),
            "forced_aligner_gpu_memory_utilization": 0.55,
        },
    )()

    out = forced_aligner.build_forced_aligner_config(args)

    assert out is not None
    assert out.model == "new"
    assert out.gpu_memory_utilization == 0.55
    assert out.dtype == "float16"
    assert out.runner == "pooling"


def test_build_config_from_cli_model_uses_default_yaml():
    args = type(
        "Args",
        (),
        {
            "forced_aligner": "local-aligner",
            "forced_aligner_config": None,
            "forced_aligner_gpu_memory_utilization": None,
        },
    )()

    out = forced_aligner.build_forced_aligner_config(args)

    assert out is not None
    assert out.model == "local-aligner"
    assert out.runner == "pooling"
    assert out.architecture == "Qwen3ASRForcedAlignerForTokenClassification"


def test_build_config_cli_device_override():
    args = type(
        "Args",
        (),
        {
            "forced_aligner": "local-aligner",
            "forced_aligner_config": None,
            "forced_aligner_gpu_memory_utilization": None,
            "forced_aligner_device": "7",
        },
    )()

    out = forced_aligner.build_forced_aligner_config(args)

    assert out is not None
    assert out.device == "7"


def test_build_config_device_defaults_to_none():
    args = type(
        "Args",
        (),
        {
            "forced_aligner": "local-aligner",
            "forced_aligner_config": None,
            "forced_aligner_gpu_memory_utilization": None,
        },
    )()

    out = forced_aligner.build_forced_aligner_config(args)

    assert out is not None
    assert out.device is None
