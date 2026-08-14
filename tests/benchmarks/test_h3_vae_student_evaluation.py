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
