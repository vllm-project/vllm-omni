import pytest
import torch

import benchmarks.diffusion.h3_vae_student_evaluation as student_eval
from benchmarks.diffusion.h3_vae_student_evaluation import DecoderArtifact, evaluate


def _artifact(**overrides):
    values = {
        "schema_version": 1,
        "base_model": "MiniMaxAI/MiniMax-H3",
        "component": "video_vae.decoder",
        "latent_channels": 24,
        "spatial_ratio": 16,
        "temporal_ratio": 4,
        "runner": "student_package:create_decoder",
        "checkpoint": "/models/student",
        "post_training_provenance": "same-latent distillation run 42",
    }
    values.update(overrides)
    return DecoderArtifact(**values)


def test_h3_student_manifest_accepts_exact_latent_contract():
    _artifact().validate()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_model": "other"}, "base_model"),
        ({"component": "vae"}, "component"),
        ({"latent_channels": 16}, "latent contract"),
        ({"runner": "missing_separator"}, "module:callable"),
        ({"post_training_provenance": ""}, "post_training_provenance"),
    ],
)
def test_h3_student_manifest_rejects_non_drop_in_artifacts(overrides, message):
    with pytest.raises(ValueError, match=message):
        _artifact(**overrides).validate()


def test_same_latent_evaluation_reports_quality_and_speedup(monkeypatch):
    reference = _artifact(runner="reference:create")
    candidate = _artifact(runner="candidate:create")

    def load_runner(artifact):
        if artifact is reference:
            return lambda latent: latent * 2
        return lambda latent: latent * 2 + 0.01

    monkeypatch.setattr(student_eval, "_load_runner", load_runner)
    report = evaluate(
        reference,
        candidate,
        torch.zeros(1, 24, 1, 1, 1),
        warmups=0,
        runs=1,
        data_range=2.0,
    )

    assert report["output_shape"] == [1, 24, 1, 1, 1]
    assert report["mae"] == pytest.approx(0.01)
    assert report["psnr_db"] > 40
    assert report["decoder_speedup"] > 0


def test_exact_student_match_uses_json_safe_psnr(monkeypatch):
    monkeypatch.setattr(student_eval, "_load_runner", lambda artifact: lambda latent: latent * 2)

    report = evaluate(
        _artifact(),
        _artifact(),
        torch.zeros(1, 24, 1, 1, 1),
        warmups=0,
        runs=1,
        data_range=2.0,
    )

    assert report["exact_match"] is True
    assert report["psnr_db"] is None


def test_student_evaluation_rejects_non_finite_decoder_output(monkeypatch):
    def load_runner(artifact):
        if artifact.runner == "reference:create":
            return lambda latent: latent
        return lambda latent: torch.full_like(latent, float("nan"))

    monkeypatch.setattr(student_eval, "_load_runner", load_runner)

    with pytest.raises(ValueError, match="NaN or Infinity"):
        evaluate(
            _artifact(runner="reference:create"),
            _artifact(runner="candidate:create"),
            torch.zeros(1, 24, 1, 1, 1),
            warmups=0,
            runs=1,
            data_range=2.0,
        )


def test_student_evaluation_rejects_non_finite_latent(monkeypatch):
    monkeypatch.setattr(student_eval, "_load_runner", lambda artifact: lambda latent: latent)
    latent = torch.zeros(1, 24, 1, 1, 1)
    latent[0, 0, 0, 0, 0] = float("inf")

    with pytest.raises(ValueError, match="latent contains"):
        evaluate(
            _artifact(),
            _artifact(),
            latent,
            warmups=0,
            runs=1,
            data_range=2.0,
        )


def test_student_evaluation_rejects_empty_measurement_set(monkeypatch):
    monkeypatch.setattr(student_eval, "_load_runner", lambda artifact: lambda latent: latent)

    with pytest.raises(ValueError, match="runs must be"):
        evaluate(
            _artifact(),
            _artifact(),
            torch.zeros(1, 24, 1, 1, 1),
            warmups=0,
            runs=0,
            data_range=2.0,
        )
