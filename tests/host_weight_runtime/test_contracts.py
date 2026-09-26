# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for semantic identity and immutable artifact schemas."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import get_type_hints

import pytest

from vllm_omni.host_weight_runtime import (
    AdaptationIdentity,
    CanonicalJson,
    ComponentIdentity,
    FileManifestEntry,
    IdentityValidationError,
    ManifestValidationError,
    ProducerIdentity,
    ProductionPolicy,
    PublicationMarker,
    RuntimeWeightLayout,
    TensorKind,
    TensorManifestEntry,
    WeightArtifactIdentity,
    WeightManifest,
    WeightRepresentation,
    WeightRestorePlan,
    WeightRestorer,
    WeightSourceIdentity,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _identity(
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    sp_size: int = 1,
    adaptation: AdaptationIdentity | None = None,
    source_metadata: CanonicalJson | None = None,
) -> WeightArtifactIdentity:
    return WeightArtifactIdentity(
        schema_version=1,
        source=WeightSourceIdentity(
            model_id="test-org/test-model",
            revision="0123456789abcdef",
            fingerprint="source-files-sha256",
            metadata=source_metadata or CanonicalJson.empty(),
        ),
        component=ComponentIdentity(name="transformer", ownership="diffusion.dit"),
        representation=WeightRepresentation(name="runtime-bf16", dtype="torch.bfloat16"),
        layout=RuntimeWeightLayout(
            name="final-module-layout",
            tensor_parallel_size=tp_size,
            tensor_parallel_rank=tp_rank,
            sequence_parallel_size=sp_size,
            sequence_parallel_backend="ulysses" if sp_size > 1 else "none",
        ),
        adaptation=adaptation or AdaptationIdentity(),
        producer=ProducerIdentity(
            producer_id="test.final-layout",
            version="1",
            implementation_fingerprint="implementation-sha256",
            manifest_schema="test-producer-v1",
            restorer_schema="test-restorer-v1",
        ),
    )


def test_canonical_metadata_is_a_stable_immutable_snapshot() -> None:
    items: list[object] = [1, {"value": 2}]
    mutable = {"z": items, "a": True}
    snapshot = CanonicalJson.from_value(mutable)
    items.append(3)

    assert snapshot.encoded == b'{"a":true,"z":[1,{"value":2}]}'
    assert snapshot.to_value() == {"a": True, "z": [1, {"value": 2}]}
    assert CanonicalJson.from_value({"a": True, "z": [1, {"value": 2}]}) == snapshot


@pytest.mark.parametrize("value", [math.inf, math.nan, object(), {1: "non-string-key"}])
def test_canonical_metadata_rejects_process_specific_or_non_json_values(value: object) -> None:
    with pytest.raises(IdentityValidationError):
        CanonicalJson.from_value(value)


@pytest.mark.parametrize("encoded", [b"not-json", b'{"b":1, "a":2}', b'{"value":NaN}'])
def test_canonical_metadata_cannot_bypass_stable_encoding(encoded: bytes) -> None:
    with pytest.raises(IdentityValidationError):
        CanonicalJson(encoded)


def test_identity_round_trip_and_hash_are_deterministic() -> None:
    first = _identity(source_metadata=CanonicalJson.from_value({"files": ["b", "a"], "size": 3}))
    second = _identity(source_metadata=CanonicalJson.from_value({"size": 3, "files": ["b", "a"]}))

    assert first == second
    assert first.key == second.key
    assert WeightArtifactIdentity.from_dict(first.to_dict()) == first
    assert len(first.key) == 64


def test_identity_separates_semantic_parallel_layout_and_static_adaptation() -> None:
    base = _identity()
    tp0 = _identity(tp_size=2, tp_rank=0)
    tp1 = _identity(tp_size=2, tp_rank=1)
    sp2 = _identity(sp_size=2)
    merged_lora = _identity(adaptation=AdaptationIdentity(kind="merged-lora", fingerprint="adapter-sha256"))

    assert len({item.key for item in (base, tp0, tp1, sp2, merged_lora)}) == 5
    assert "data_parallel" not in base.layout.to_dict()
    assert "sequence_parallel_rank" not in sp2.layout.to_dict()


def test_identity_schema_rejects_unknown_fields() -> None:
    document = _identity().to_dict()
    document["transfer_mode"] = "registered"

    with pytest.raises(IdentityValidationError, match="unexpected"):
        WeightArtifactIdentity.from_dict(document)


def test_identity_schema_does_not_coerce_field_types() -> None:
    document = _identity().to_dict()
    document["schema_version"] = "1"

    with pytest.raises(IdentityValidationError, match="must be an integer"):
        WeightArtifactIdentity.from_dict(document)


def test_restoration_contract_has_a_validation_only_one_shot_commit_boundary() -> None:
    assert get_type_hints(WeightRestorePlan.commit)["return"] is type(None)
    assert "one-shot" in (WeightRestorePlan.__doc__ or "")
    assert "without mutating" in (WeightRestorer.__doc__ or "")


def test_post_load_publication_policy_is_explicitly_enabled() -> None:
    policy = ProductionPolicy(allow_post_load_publish=True)

    assert policy.allow_local_build
    assert policy.allow_post_load_publish


def test_manifest_and_publication_marker_round_trip_exactly() -> None:
    identity = _identity()
    manifest = WeightManifest(
        schema_version=1,
        artifact_key=identity.key,
        identity=identity,
        producer_schema="test-producer-v1",
        restorer_schema="test-restorer-v1",
        tensors=(
            TensorManifestEntry(
                name="layer.weight",
                kind=TensorKind.PARAMETER,
                role="weight",
                file_name="weights.safetensors",
                storage_key="layer.weight",
                shape=(2, 4),
                dtype="torch.bfloat16",
                stride=(4, 1),
                nbytes=16,
                sha256=_SHA_A,
            ),
        ),
        files=(FileManifestEntry(file_name="weights.safetensors", size=128, sha256=_SHA_B),),
        artifact_content_sha256=_SHA_C,
        format_metadata=CanonicalJson.from_value({"quantization": "none"}),
    )
    marker = PublicationMarker(
        schema_version=1,
        publication_id="publication-id",
        manifest_sha256=_SHA_A,
        artifact_content_sha256=_SHA_C,
        integrity_mode="full_checksum",
    )

    assert WeightManifest.from_dict(manifest.to_dict()) == manifest
    assert PublicationMarker.from_dict(marker.to_dict()) == marker

    document = manifest.to_dict()
    document["unknown"] = True
    with pytest.raises(ManifestValidationError, match="unexpected"):
        WeightManifest.from_dict(document)

    malformed = manifest.to_dict()
    tensors = malformed["tensors"]
    assert isinstance(tensors, list) and isinstance(tensors[0], dict)
    tensors[0]["nbytes"] = "16"
    with pytest.raises(ManifestValidationError, match="must be an integer"):
        WeightManifest.from_dict(malformed)


@pytest.mark.parametrize("file_name", ["", ".", "..", "../weights.safetensors", "nested/weights.safetensors"])
def test_manifest_rejects_unsafe_payload_names(file_name: str) -> None:
    with pytest.raises(ManifestValidationError, match="safe relative basename"):
        FileManifestEntry(file_name=file_name, size=1, sha256=_SHA_A)


def test_package_source_keeps_the_neutral_dependency_boundary() -> None:
    package_root = Path(__file__).resolve().parents[2] / "vllm_omni" / "host_weight_runtime"
    assert package_root.is_dir()
    forbidden = (
        "vllm_omni.diffusion",
        "vllm_omni.distributed",
        "torch.cuda",
        "torch.distributed",
    )
    violations: list[str] = []
    for source_path in package_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(
                    f"{source_path.name}:{node.lineno}:{alias.name}"
                    for alias in node.names
                    if alias.name.startswith(forbidden)
                )
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module or ""
                if module.startswith(forbidden) or (
                    module == "torch" and any(alias.name in {"cuda", "distributed"} for alias in node.names)
                ):
                    violations.append(f"{source_path.name}:{node.lineno}:{module}")

    assert violations == []
