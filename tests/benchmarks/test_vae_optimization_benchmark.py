from types import SimpleNamespace

import pytest

from benchmarks.diffusion.vae_optimization_benchmark import (
    RunResult,
    _apply_gates,
    _tile_boundaries,
    _vae_time,
    _warm_median,
    parse_rank_timings,
    parse_vae_diagnostics,
)


def _report(*times):
    return {
        "runs": [
            {
                "cold": index == 0,
                "end_to_end_s": value,
                "inference_s": value - 0.1,
                "stage_durations": {
                    "video_vae.decode_latent": value * 0.2,
                    "audio_vae.decode_latent": value * 0.01,
                },
            }
            for index, value in enumerate(times)
        ]
    }


def test_warm_median_excludes_first_request():
    assert _warm_median(_report(20.0, 10.0, 12.0), "end_to_end_s") == 11.0


def test_vae_time_uses_separate_video_and_audio_components():
    run = RunResult(
        index=0,
        cold=True,
        http_status=200,
        end_to_end_s=10.0,
        inference_s=9.0,
        stage_durations={"video_vae.decode_latent": 2.0, "audio_vae.decode_latent": 0.5, "decode": 3.0},
        peak_memory_mb=100.0,
        media_path="output.mp4",
        sha256="abc",
    )

    assert _vae_time(run) == 2.5


@pytest.mark.parametrize(
    ("candidate_time", "psnr", "audio_mae", "expected"),
    [
        (10.4, 80.0, 0.0, []),
        (11.0, 80.0, 0.0, ["end-to-end regression"]),
        (10.0, 40.0, 0.0, ["video PSNR"]),
        (10.0, 80.0, 0.01, ["audio MAE"]),
    ],
)
def test_acceptance_gates(candidate_time, psnr, audio_mae, expected):
    report = _report(candidate_time, candidate_time)
    reference = _report(10.0, 10.0)
    quality = {"video_psnr_db": psnr, "video_seam_band_mae": 0.0, "audio_mae": audio_mae}
    args = SimpleNamespace(
        max_end_to_end_regression_pct=5.0,
        min_video_psnr_db=60.0,
        max_video_mae=1.0,
        max_video_seam_band_mae=1.0,
        max_video_seam_excess_ratio=1.25,
        max_audio_mae=1e-4,
        max_av_sync_delta_s=0.1,
        max_rank_imbalance_pct=15.0,
    )

    failures = _apply_gates(report, reference, quality, args)

    assert len(failures) == len(expected)
    for prefix, failure in zip(expected, failures):
        assert failure.startswith(prefix)
    assert report["amdahl"]["reference_vae_share"] == pytest.approx(0.21)
    assert report["amdahl"]["observed_end_to_end_speedup"] == pytest.approx(10.0 / candidate_time)


def test_rank_timing_parser_reports_median_imbalance(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(
            [
                'INFO [VAE component timing] {"rank":0,"metric":"video_vae.decode_latent","duration_s":2.0}',
                'INFO [VAE component timing] {"rank":0,"metric":"video_vae.decode_latent","duration_s":4.0}',
                'INFO [VAE component timing] {"rank":1,"metric":"video_vae.decode_latent","duration_s":3.3}',
            ]
        ),
        encoding="utf-8",
    )

    timings = parse_rank_timings(log)["video_vae.decode_latent"]

    assert timings["median_s_by_rank"] == {"0": 3.0, "1": 3.3}
    assert timings["imbalance_pct"] == pytest.approx(10.0)


def test_diagnostic_parser_keeps_last_decode_per_rank(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(
            [
                'INFO [VAE diagnostics] {"rank":0,"latent_sha256":"old","cold":true}',
                'INFO [VAE diagnostics] {"rank":1,"latent_sha256":"same","cold":true}',
                'INFO [VAE diagnostics] {"rank":0,"latent_sha256":"same","cold":false}',
            ]
        ),
        encoding="utf-8",
    )

    diagnostics = parse_vae_diagnostics(log)["last_decode_by_rank"]

    assert diagnostics["0"]["latent_sha256"] == "same"
    assert diagnostics["0"]["cold"] is False
    assert diagnostics["1"]["latent_sha256"] == "same"


def test_log_parsers_can_ignore_preexisting_requests(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        'INFO [VAE component timing] {"rank":0,"metric":"video_vae.decode_latent","duration_s":99.0}\n',
        encoding="utf-8",
    )
    start_offset = log.stat().st_size
    with log.open("a", encoding="utf-8") as stream:
        stream.write('INFO [VAE component timing] {"rank":0,"metric":"video_vae.decode_latent","duration_s":3.0}\n')

    timings = parse_rank_timings(log, start_offset)["video_vae.decode_latent"]

    assert timings["median_s_by_rank"] == {"0": 3.0}


def test_gate_rejects_different_latent_inputs():
    report = _report(10.0, 10.0)
    reference = _report(10.0, 10.0)
    report["vae_diagnostics"] = {"last_decode_by_rank": {"0": {"latent_sha256": "candidate"}}}
    reference["vae_diagnostics"] = {"last_decode_by_rank": {"0": {"latent_sha256": "reference"}}}
    quality = {"video_psnr_db": 80.0, "video_seam_band_mae": 0.0, "audio_mae": 0.0}
    args = SimpleNamespace(
        max_end_to_end_regression_pct=5.0,
        min_video_psnr_db=60.0,
        max_video_mae=1.0,
        max_video_seam_band_mae=1.0,
        max_video_seam_excess_ratio=1.25,
        max_audio_mae=1e-4,
        max_av_sync_delta_s=0.1,
        max_rank_imbalance_pct=15.0,
    )

    failures = _apply_gates(report, reference, quality, args)

    assert "reference and candidate VAE latent fingerprints differ" in failures


def test_gate_rejects_audio_video_duration_drift():
    report = _report(10.0, 10.0)
    reference = _report(10.0, 10.0)
    quality = {
        "video_psnr_db": 80.0,
        "video_seam_band_mae": 0.0,
        "audio_mae": 0.0,
        "av_sync_delta_s": 0.2,
    }
    args = SimpleNamespace(
        max_end_to_end_regression_pct=5.0,
        min_video_psnr_db=60.0,
        max_video_mae=1.0,
        max_video_seam_band_mae=1.0,
        max_video_seam_excess_ratio=1.25,
        max_audio_mae=1e-4,
        max_av_sync_delta_s=0.1,
        max_rank_imbalance_pct=15.0,
    )

    failures = _apply_gates(report, reference, quality, args)

    assert any(failure.startswith("audio/video duration delta") for failure in failures)


def test_h3_tile_boundaries_match_native_split_algorithm():
    assert _tile_boundaries(576, 256, 64, 16) == [160, 320]
    assert _tile_boundaries(1024, 256, 64, 16) == [192, 384, 576, 768]
