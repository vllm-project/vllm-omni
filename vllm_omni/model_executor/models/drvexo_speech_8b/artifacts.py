# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Resolve model assets relative to the stage's local model directory."""

import hashlib
from pathlib import Path


def resolve_artifact_path(model_directory: str, artifact: str) -> str:
    """Support relocatable bundles while retaining historical absolute paths.

    For example, a Talker stage in ``bundle/talker`` can use
    ``../assets/speaker.pt``. Final export validation is responsible for
    checking that every resolved asset stays inside the complete bundle.
    """
    if not isinstance(artifact, str) or not artifact.strip():
        raise ValueError("DRVEXO artifact path must be a nonempty local path")
    if "://" in artifact:
        raise ValueError("DRVEXO artifact paths must refer to local bundle assets")
    path = Path(artifact)
    if not path.is_absolute():
        directory = Path(model_directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"DRVEXO local model directory is missing: {directory}")
        path = directory / path
    return str(path.resolve())


def resolve_projector_quantization(model_directory, specification, source_checkpoint):
    """Bind an explicitly selected packed projector to the trained checkpoint."""
    if specification is None:
        return None
    if not isinstance(specification, dict) or specification.get("format") != "drvexo_input_projector_v1":
        raise ValueError("Invalid input projector quantization specification")
    precision = specification.get("precision")
    if precision not in ("fp8", "nvfp4"):
        raise ValueError(f"Unsupported input projector precision: {precision}")
    path = resolve_artifact_path(model_directory, specification.get("path"))
    for artifact, expected, description in [
        (path, specification.get("sha256"), "packed projector"),
        (source_checkpoint, specification.get("source_checkpoint_sha256"), "source projector"),
    ]:
        if not isinstance(expected, str) or len(expected) != 64 or not artifact:
            raise ValueError(f"Missing {description} hash or path")
        with open(artifact, "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"{description} hash mismatch")
    return precision, path


def resolve_text_projection_quantization(model_directory, specification):
    """Resolve packed TTS projection and its exact BF16 source reference."""
    if specification is None:
        return None
    if not isinstance(specification, dict) or specification.get("format") != "drvexo_text_projection_v1":
        raise ValueError("Invalid text projection quantization specification")
    precision = specification.get("precision")
    if precision not in ("fp8", "nvfp4"):
        raise ValueError(f"Unsupported text projection precision: {precision}")
    paths = []
    for key, hash_key in [("path", "sha256"), ("source_path", "source_sha256")]:
        path = resolve_artifact_path(model_directory, specification.get(key))
        expected = specification.get(hash_key)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Missing text projection {hash_key}")
        with open(path, "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Text projection {hash_key} mismatch")
        paths.append(path)
    return precision, *paths


def resolve_code_predictor_quantization(model_directory, specification):
    if specification is None:
        return None
    if not isinstance(specification, dict) or specification.get("format") != "drvexo_code_predictor_v1":
        raise ValueError("Invalid Code Predictor quantization specification")
    # The two components use the same packed/source artifact contract.
    translated = {**specification, "format": "drvexo_text_projection_v1"}
    return resolve_text_projection_quantization(model_directory, translated)


def resolve_audio_encoder_quantization(model_directory, specification):
    if specification is None:
        return None
    if not isinstance(specification, dict) or specification.get("format") != "drvexo_native_audio_encoder_v1":
        raise ValueError("Invalid native audio encoder quantization specification")
    translated = {**specification, "format": "drvexo_text_projection_v1"}
    return resolve_text_projection_quantization(model_directory, translated)


def resolve_codec_quantization(model_directory, specification):
    if specification is None:
        return None
    if not isinstance(specification, dict) or specification.get("format") != "drvexo_codec_linears_v1":
        raise ValueError("Invalid Codec quantization specification")
    translated = {**specification, "format": "drvexo_text_projection_v1"}
    return resolve_text_projection_quantization(model_directory, translated)
