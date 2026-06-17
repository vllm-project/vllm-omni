import numpy as np
import pytest

from vllm_omni.utils import forced_aligner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_decode_timestamps_maps_boundary_bins_to_words():
    logits = np.zeros((4, 5), dtype=np.float32)
    logits[0, 0] = 1.0
    logits[1, 2] = 1.0
    logits[2, 2] = 1.0
    logits[3, 4] = 1.0

    timestamps = forced_aligner._decode_timestamps(
        logits=logits,
        words=["hello", "world"],
        timestamp_positions=[0, 1, 2, 3],
        classify_num=5,
        audio_duration_ms=1000,
    )

    assert timestamps == [
        forced_aligner.WordTimestamp("hello", 0, 400),
        forced_aligner.WordTimestamp("world", 400, 800),
    ]


def test_decode_timestamps_repairs_non_monotonic_bins():
    # Word 1's end bin (1) dips below its start bin (2); fix_timestamp should
    # snap it back so the frame never carries end_ms < start_ms.
    logits = np.zeros((4, 5), dtype=np.float32)
    logits[0, 0] = 1.0  # word0 start -> bin 0
    logits[1, 2] = 1.0  # word0 end   -> bin 2
    logits[2, 1] = 1.0  # word1 start -> bin 1 (out of order)
    logits[3, 3] = 1.0  # word1 end   -> bin 3

    timestamps = forced_aligner._decode_timestamps(
        logits=logits,
        words=["hello", "world"],
        timestamp_positions=[0, 1, 2, 3],
        classify_num=5,
        timestamp_segment_time_ms=200,
        audio_duration_ms=1000,
    )

    for ts in timestamps:
        assert ts.end_ms >= ts.start_ms
    # bins [0, 2, 1, 3] -> repaired [0, 2, 2, 3] -> ms x200
    assert timestamps == [
        forced_aligner.WordTimestamp("hello", 0, 400),
        forced_aligner.WordTimestamp("world", 400, 600),
    ]


def test_decode_timestamps_clamps_to_audio_duration_and_stays_ordered():
    # Bins run past the 1000 ms audio; the decoder must cap each bound at the
    # audio length and keep words non-overlapping (start <= end, monotonic).
    logits = np.zeros((4, 5), dtype=np.float32)
    logits[0, 0] = 1.0  # word0 start -> bin 0
    logits[1, 3] = 1.0  # word0 end   -> bin 3 (1200 ms, past audio)
    logits[2, 3] = 1.0  # word1 start -> bin 3 (1200 ms, past audio)
    logits[3, 4] = 1.0  # word1 end   -> bin 4 (1600 ms, past audio)

    timestamps = forced_aligner._decode_timestamps(
        logits=logits,
        words=["a", "b"],
        timestamp_positions=[0, 1, 2, 3],
        classify_num=5,
        timestamp_segment_time_ms=400,
        audio_duration_ms=1000,
    )

    assert timestamps == [
        forced_aligner.WordTimestamp("a", 0, 1000),
        forced_aligner.WordTimestamp("b", 1000, 1000),
    ]


def test_decode_timestamps_rejects_marker_count_mismatch():
    logits = np.zeros((2, 5), dtype=np.float32)

    timestamps = forced_aligner._decode_timestamps(
        logits=logits,
        words=["hello", "world"],
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
        },
    )()

    out = forced_aligner.build_forced_aligner_config(args)

    assert out is not None
    # --forced-aligner overrides the YAML model; gpu_memory_utilization/dtype
    # come from the user YAML (there is no longer a CLI flag for gpu mem).
    assert out.model == "new"
    assert out.gpu_memory_utilization == 0.2
    assert out.dtype == "float16"
    assert out.runner == "pooling"


def test_build_config_from_cli_model_uses_default_yaml():
    args = type(
        "Args",
        (),
        {
            "forced_aligner": "local-aligner",
            "forced_aligner_config": None,
        },
    )()

    out = forced_aligner.build_forced_aligner_config(args)

    assert out is not None
    assert out.model == "local-aligner"
    assert out.runner == "pooling"
    assert out.architecture == "Qwen3ASRForcedAlignerForTokenClassification"
