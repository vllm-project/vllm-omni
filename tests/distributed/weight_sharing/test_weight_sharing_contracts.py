# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import msgspec
import pytest

from tests.distributed.weight_sharing.helpers import FakeCodec, make_context, make_weights

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def manifest(contracts):
    bindings, spec = make_weights(contracts)
    contracts.validate_spec(spec)
    return FakeCodec(contracts).inspect(bindings, spec, make_context(contracts)).manifest


def test_manifest_preserves_shared_storage_and_aliases(contracts, manifest):
    contracts.validate_manifest(manifest, manifest.spec)
    assert len(manifest.storages) == 1
    assert len(manifest.views) == 3
    assert manifest.views[1].offset_bytes == 16


@pytest.mark.parametrize(
    "changes",
    [
        {"rank": -1},
        {"rank": 2, "tp_size": 2},
        {"tp_size": 0},
        {"generation": -1},
        {"protocol_version": 2},
        {"device_uuid": ""},
        {"consumer_id": ""},
    ],
)
def test_context_rejects_invalid_pairing(contracts, changes):
    with pytest.raises(ValueError):
        make_context(contracts, **changes)


def test_context_rejects_negative_device(contracts):
    with pytest.raises(ValueError):
        replace(make_context(contracts), device_index=-1)


@pytest.mark.parametrize("changes", [{"max_entries": 0}, {"max_bundle_bytes": 0}, {"max_bundle_bytes": 2**32}])
def test_metadata_limits(contracts, changes):
    with pytest.raises(ValueError):
        contracts.Limits(**changes)


@pytest.mark.parametrize(
    "tensor",
    [
        {"name": "", "dtype": "float32", "shape": (4,)},
        {"name": "weight", "dtype": "opaque", "shape": (4,)},
        {"name": "weight", "dtype": "float32", "shape": (-1,)},
        {"name": "weight", "dtype": "float32", "shape": (4,), "stride": (-1,)},
        {"name": "weight", "dtype": "float32", "shape": (4,), "stride": (1, 1)},
        {"name": "weight", "dtype": "float32", "shape": (4,), "alias_of": "missing"},
        {"name": "weight", "dtype": "float32", "shape": (4,), "alias_of": "weight"},
    ],
)
def test_spec_rejects_invalid_tensor(contracts, tensor):
    spec = contracts.WeightSpec("model", "revision", "layout", (contracts.TensorSpec(**tensor),))
    with pytest.raises(contracts.CompatibilityError):
        contracts.validate_spec(spec)


def test_spec_rejects_duplicate_names_and_entry_overflow(contracts):
    tensor = contracts.TensorSpec("weight", "float32", (4,))
    spec = contracts.WeightSpec("model", "revision", "layout", (tensor, tensor))
    with pytest.raises(contracts.CompatibilityError, match="unique"):
        contracts.validate_spec(spec)
    spec = msgspec.structs.replace(spec, tensors=(tensor,))
    with pytest.raises(contracts.CompatibilityError, match="bounded"):
        contracts.validate_spec(spec, max_entries=0)


@pytest.mark.parametrize("offset", [-4, 1, 20])
def test_manifest_rejects_misaligned_or_out_of_bounds_view(contracts, manifest, offset):
    bad_view = msgspec.structs.replace(manifest.views[1], offset_bytes=offset)
    manifest = msgspec.structs.replace(manifest, views=(manifest.views[0], bad_view, manifest.views[2]))
    with pytest.raises(contracts.CompatibilityError):
        contracts.validate_manifest(manifest, manifest.spec)


def test_manifest_rejects_alias_with_different_offset(contracts, manifest):
    alias = msgspec.structs.replace(manifest.views[2], offset_bytes=16)
    manifest = msgspec.structs.replace(manifest, views=(*manifest.views[:2], alias))
    with pytest.raises(contracts.CompatibilityError, match="alias"):
        contracts.validate_manifest(manifest, manifest.spec)


@pytest.mark.parametrize("field", ["model_id", "checkpoint_revision", "layout"])
def test_manifest_rejects_different_weight_identity(contracts, manifest, field):
    expected = msgspec.structs.replace(manifest.spec, **{field: "other"})
    with pytest.raises(contracts.CompatibilityError, match="differs"):
        contracts.validate_manifest(manifest, expected)


def test_manifest_accepts_scalar_empty_and_transposed_views(contracts):
    specs = (
        contracts.TensorSpec("scalar", "float32", (), ()),
        contracts.TensorSpec("empty", "float32", (0, 4), (4, 1)),
        contracts.TensorSpec("transposed", "float32", (2, 3), (1, 2)),
    )
    spec = contracts.WeightSpec("model", "revision", "layout", specs)
    contracts.validate_spec(spec)
    manifest = contracts.WeightManifest(
        spec,
        (contracts.StorageSpec("s", 24), contracts.StorageSpec("empty", 0)),
        (
            contracts.TensorViewSpec("scalar", "s", "float32", (), (), 0),
            contracts.TensorViewSpec("empty", "empty", "float32", (0, 4), (4, 1), 0),
            contracts.TensorViewSpec("transposed", "s", "float32", (2, 3), (1, 2), 0),
        ),
    )
    contracts.validate_manifest(manifest, spec)


def test_manifest_rejects_unreferenced_storage(contracts, manifest):
    manifest = msgspec.structs.replace(manifest, storages=(*manifest.storages, contracts.StorageSpec("extra", 8)))
    with pytest.raises(contracts.CompatibilityError, match="unused"):
        contracts.validate_manifest(manifest, manifest.spec)


def test_protocol_decoder_rejects_unknown_fields(contracts):
    payload = msgspec.msgpack.encode(
        {"identity": make_context(contracts).identity, "import_id": "i", "op": "imported", "callable": "bad"}
    )
    with pytest.raises(msgspec.ValidationError, match="unknown field"):
        msgspec.msgpack.decode(payload, type=contracts.SessionReceipt)


def test_independent_package_import_and_backend_construction_are_lazy():
    """Check actual source in a fresh process with enclosing packages isolated."""
    script = """
import sys
import threading
from tests.distributed.weight_sharing.helpers import load_weight_sharing

before = set(threading.enumerate())
package = load_weight_sharing()
backend = package.TorchCudaIpcBackend()
assert "torch" not in sys.modules
assert "vllm_omni" not in sys.modules
assert set(threading.enumerate()) == before
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
