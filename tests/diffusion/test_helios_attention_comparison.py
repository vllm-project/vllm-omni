# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest

from benchmarks.diffusion.compare_helios_attention import compare_metrics

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_exact_outputs_have_json_safe_psnr():
    video = np.zeros((2, 8, 8, 3), dtype=np.float32)
    metrics = compare_metrics(video, video.copy())
    assert metrics == {
        "exact": True,
        "mae": 0.0,
        "max_abs": 0.0,
        "rmse": 0.0,
        "psnr_db": None,
        "temporal_delta_mae": 0.0,
    }


def test_constant_offset_has_known_metrics():
    video = np.zeros((2, 8, 8, 3), dtype=np.float32)
    metrics = compare_metrics(video, video + 0.25)
    assert not metrics["exact"]
    assert metrics["mae"] == metrics["rmse"] == metrics["max_abs"] == 0.25
    assert metrics["psnr_db"] == pytest.approx(12.041199826559248)
    assert metrics["temporal_delta_mae"] == 0.0


def test_temporal_difference():
    video = np.zeros((2, 8, 8, 3), dtype=np.float32)
    changed = video.copy()
    changed[1] = 0.5
    assert compare_metrics(video, changed)["temporal_delta_mae"] == 0.5


def test_rejects_invalid_outputs():
    video = np.zeros((2, 8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="matching FHWC"):
        compare_metrics(video, video[:, :, :, :1])
    invalid = video.copy()
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        compare_metrics(video, invalid)


def sample_runs():
    from benchmarks.diffusion.compare_helios_attention import BACKENDS

    return {
        name: {
            "metadata": {
                "frames": [33],
                "seeds": [42],
                "repeats": 1,
                "backend": name,
                "attention_implementations": {"provider": name},
            },
            "records": [{"warmup": False, "num_frames": 33, "seed": 42, "repeat": 0, "prompt": "train"}],
        }
        for name in BACKENDS
    }


def test_contract_allows_backend_specific_bindings():
    from benchmarks.diffusion.compare_helios_attention import validate_runs

    assert len(validate_runs(sample_runs())) == 3


@pytest.mark.parametrize("change", ["duplicate", "missing", "environment", "prompt"])
def test_contract_rejects_incomparable_runs(change):
    from benchmarks.diffusion.compare_helios_attention import validate_runs

    runs = sample_runs()
    run = runs["FLASH_ATTN"]
    if change == "duplicate":
        run["records"] *= 2
    elif change == "missing":
        run["records"] = []
    elif change == "environment":
        run["metadata"]["torch"] = "different"
    else:
        run["records"][0]["prompt"] = "beach"
    with pytest.raises(ValueError):
        validate_runs(runs)


def test_export_excludes_warmup_and_raw_timelines():
    from benchmarks.diffusion.export_helios_attention import PROVENANCE_KEYS, RECORD_KEYS, build_artifact

    runs = sample_runs()
    for run in runs.values():
        run["metadata"].update(model="/local/model", engine_startup_ms=1)
        run["summary"] = {"33": {"count": 1}}
        row = run["records"][0]
        row.update({key: 1 for key in RECORD_KEYS if key not in row})
        row.update(transformer_timings=[{"gpu_ms": 1}], stage_durations_ms={"decode": 1})
        run["records"].append({"warmup": True})
    provenance = {key: "historical" for key in PROVENANCE_KEYS}
    alignment: dict = {"alignment_vs_torch_sdpa": {}, "self_variance": {}}
    for name in runs:
        alignment["alignment_vs_torch_sdpa"][name] = [{"num_frames": 33, "seed": 42, "repeat": 0}]
        alignment["self_variance"][name] = []
    artifact = build_artifact(runs, alignment, provenance)
    assert artifact["benchmark_script_sha256"] == "historical"
    assert "model" not in artifact["metadata"]
    assert artifact["prompts_by_seed"] == {"42": "train"}
    for backend in artifact["backends"].values():
        assert backend["summary"][33]["median_wall_ms"] == 1
        assert len(backend["measurements"]) == 1
        assert set(backend["measurements"][0]) == set(RECORD_KEYS)


def test_ssim_matches_exact_frames_when_available():
    pytest.importorskip("skimage.metrics")
    from benchmarks.diffusion.compare_helios_attention import compare

    video = np.zeros((2, 8, 8, 3), dtype=np.float32)
    result = compare(video, video.copy())
    assert result["ssim_mean"] == 1.0
    assert result["psnr_db"] is None


@pytest.mark.parametrize("tamper", [False, True])
def test_main_groups_repeats_and_verifies_arrays(tmp_path, monkeypatch, tamper):
    import hashlib
    import json

    from benchmarks.diffusion import compare_helios_attention as comparison

    runs = sample_runs()
    for index, (name, run) in enumerate(runs.items()):
        directory = tmp_path / name
        directory.mkdir()
        run["metadata"]["repeats"] = 2
        run["records"] = []
        for repeat in range(2):
            video = np.full((2, 8, 8, 3), index * 0.25 + repeat * 0.0625, dtype=np.float32)
            row = {
                "warmup": False,
                "num_frames": 33,
                "seed": 42,
                "repeat": repeat,
                "prompt": "train",
                "array": f"{repeat}.npy",
                "sha256": hashlib.sha256(video.tobytes()).hexdigest(),
            }
            run["records"].append(row)
            if tamper and name == "FLASH_ATTN" and repeat == 1:
                video += 0.125
            np.save(directory / row["array"], video)
        (directory / "results.json").write_text(json.dumps(run))
    # Exercise real I/O and grouping without making SSIM a CPU-lane dependency.
    monkeypatch.setattr(comparison, "compare", comparison.compare_metrics)
    monkeypatch.setattr("sys.argv", ["compare", str(tmp_path)])
    if tamper:
        with pytest.raises(ValueError, match="hash mismatch"):
            comparison.main()
        assert not (tmp_path / "alignment.json").exists()
    else:
        comparison.main()
        report = json.loads((tmp_path / "alignment.json").read_text())
        for index, name in enumerate(runs):
            cross = report["alignment_vs_torch_sdpa"][name]
            own = report["self_variance"][name]
            assert len(cross) == len(own) == 1
            assert cross[0]["repeat"] == 0
            assert cross[0]["mae"] == index * 0.25
            assert own[0]["repeat"] == 1
            assert own[0]["mae"] == 0.0625


@pytest.mark.parametrize("backend", ["TORCH_SDPA", None])
def test_rejects_mislabeled_backend(backend):
    from benchmarks.diffusion.compare_helios_attention import validate_runs

    runs = sample_runs()
    runs["FLASH_ATTN"]["metadata"]["backend"] = backend
    with pytest.raises(ValueError, match="backend does not match folder"):
        validate_runs(runs)


def test_committed_h20_artifact_matches_export_contract():
    import json
    from pathlib import Path

    from benchmarks.diffusion.compare_helios_attention import BACKENDS
    from benchmarks.diffusion.export_helios_attention import PROVENANCE_KEYS, RECORD_KEYS

    path = Path(__file__).resolve().parents[2] / "recipes/Helios/Helios-Distilled-H20-results.json"
    artifact = json.loads(path.read_text())
    assert set(PROVENANCE_KEYS) <= artifact.keys()
    assert set(artifact["backends"]) == set(BACKENDS)
    expected = {(frames, seed, repeat) for frames in (33, 66) for seed in (42, 7, 123) for repeat in range(3)}
    implementations = {
        "TORCH_SDPA": "sdpa.SDPAImpl",
        "FLASH_ATTN": "flash_attn.FlashAttentionImpl",
        "CUDNN_ATTN": "cudnn_attn.CuDNNAttentionImpl",
    }
    hashes = {}
    for name, backend in artifact["backends"].items():
        assert backend["attention_implementations"] == {
            f"vllm_omni.diffusion.attention.backends.{implementations[name]}": 80
        }
        rows = backend["measurements"]
        assert len(rows) == len(expected)
        assert {(row["num_frames"], row["seed"], row["repeat"]) for row in rows} == expected
        assert all(set(row) == set(RECORD_KEYS) for row in rows)
        assert all(row["transformer_forward_count"] == {33: 12, 66: 18}[row["num_frames"]] for row in rows)
        grouped: dict[tuple[int, int], set[str]] = {}
        for row in rows:
            digest = row["sha256"]
            assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
            grouped.setdefault((row["num_frames"], row["seed"]), set()).add(digest)
        assert all(len(values) == 1 for values in grouped.values())
        hashes[name] = grouped
    for name in ("FLASH_ATTN", "CUDNN_ATTN"):
        assert hashes[name].keys() == hashes["TORCH_SDPA"].keys()
        assert all(hashes[name][case] != hashes["TORCH_SDPA"][case] for case in hashes[name])


@pytest.mark.parametrize(
    "change",
    ["missing_group", "missing_backend", "missing_cell", "duplicate", "wrong_repeat", "wrong_seed", "extra_backend"],
)
def test_export_rejects_alignment_with_different_cells(change):
    from benchmarks.diffusion.export_helios_attention import PROVENANCE_KEYS, RECORD_KEYS, build_artifact

    runs = sample_runs()
    alignment: dict = {"alignment_vs_torch_sdpa": {}, "self_variance": {}}
    for name, run in runs.items():
        run["metadata"]["repeats"] = 2
        first = run["records"][0]
        first.update({key: 1 for key in RECORD_KEYS if key not in first})
        run["records"].append({**first, "repeat": 1})
        run["summary"] = {"incorrect": "unused"}
        alignment["alignment_vs_torch_sdpa"][name] = [{"num_frames": 33, "seed": 42, "repeat": 0}]
        alignment["self_variance"][name] = [{"num_frames": 33, "seed": 42, "repeat": 1}]
    if change == "missing_group":
        del alignment["self_variance"]
    elif change == "extra_backend":
        alignment["self_variance"]["UNKNOWN"] = []
    elif change == "missing_backend":
        del alignment["self_variance"]["FLASH_ATTN"]
    else:
        rows = alignment["self_variance"]["FLASH_ATTN"]
        if change == "missing_cell":
            rows.clear()
        elif change == "duplicate":
            rows.append(dict(rows[0]))
        elif change == "wrong_repeat":
            rows[0]["repeat"] = 0
        else:
            rows[0]["seed"] = 7
    provenance = {key: "historical" for key in PROVENANCE_KEYS}
    with pytest.raises(ValueError, match="alignment.json"):
        build_artifact(runs, alignment, provenance)
