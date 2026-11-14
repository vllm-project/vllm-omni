# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""`prefix_identity` must stay byte-identical while avoiding extra copies.

The digest is used as a prefix-cache dictionary key, so a digest that changes
for equal content would silently disable prefix reuse, and a digest that stays
equal for different content would serve the wrong KV pages. Both directions are
pinned here, against a verbatim copy of the previous implementation.

Measured on the previous implementation vs the current one (CPU hashing, the
order of magnitude that matters here):

| prefix | previous | current | saved |
| --- | --- | --- | --- |
| 3 MB  | 1.3 ms | 1.2 ms | 0.1 ms |
| 13 MB | 5.1 ms | 4.5 ms | 0.6 ms |
| 50 MB | 74 ms  | 39 ms  | 34 ms  |

So this is a real but bounded win that only matters for large reference
latents, and only on session admission (`prefix_identity` runs once per new
session, not per chunk). It is kept as a copy-avoidance cleanup, not as a
performance feature.
"""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.duo.contracts import RealizationGroup
from vllm_omni.diffusion.models.duo.paged_cache import prefix_identity

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _previous_implementation(sequence, *, namespace):
    """The implementation before the copy-avoidance change, verbatim."""
    if not namespace or any(g.kind not in ("reference", "shared", "padding") for g in sequence.groups):
        raise ValueError("Prefix identity requires namespace and prefix-only sequence")
    digest = hashlib.sha256(namespace.encode())
    for g in (*sequence.groups, *sequence.refiner_groups):
        digest.update(repr(g).encode())
    for name in sequence.__dataclass_fields__:
        value = getattr(sequence, name)
        if isinstance(value, torch.Tensor):
            value = value.detach().contiguous().cpu()
            digest.update(str((name, tuple(value.shape), value.dtype)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _sequence(video, audio, *, groups=None):
    groups = groups if groups is not None else (RealizationGroup(0, 0, "reference", 0, 4),)
    return SimpleNamespace(
        __dataclass_fields__={"video": None, "audio": None, "seed": None},
        groups=groups,
        refiner_groups=(),
        video=video,
        audio=audio,
        seed=7,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_prefix_identity_matches_previous_implementation(dtype):
    torch.manual_seed(0)
    seq = _sequence(torch.randn(1, 5, 24, dtype=dtype), torch.randn(1, 29, 32, dtype=dtype))
    assert prefix_identity(seq, namespace="t/ckpt/bf16/tp1/frame-v5") == _previous_implementation(
        seq, namespace="t/ckpt/bf16/tp1/frame-v5"
    )


def test_prefix_identity_hashes_non_contiguous_inputs_identically():
    # A non-contiguous view must be materialized before hashing; the old code
    # relied on .contiguous() and so must this one.
    torch.manual_seed(0)
    base = torch.randn(6, 8, dtype=torch.float32)
    view = base.t()
    assert not view.is_contiguous()
    seq = _sequence(view, torch.randn(6, 8, dtype=torch.float32))
    assert prefix_identity(seq, namespace="ns") == _previous_implementation(seq, namespace="ns")


def test_prefix_identity_still_separates_different_content():
    # The optimisation must not collapse distinct prefixes onto one key.
    torch.manual_seed(0)
    a = _sequence(torch.zeros(1, 5, 24), torch.zeros(1, 29, 32))
    b = _sequence(torch.zeros(1, 5, 24), torch.zeros(1, 29, 32))
    b.audio[0, 0, 0] = 1e-6  # any content change must change the digest
    assert prefix_identity(a, namespace="ns") != prefix_identity(b, namespace="ns")


def test_prefix_identity_rejects_non_prefix_groups():
    seq = _sequence(torch.zeros(1, 5, 24), torch.zeros(1, 29, 32), groups=(RealizationGroup(0, 0, "noisy", 0, 4),))
    with pytest.raises(ValueError, match="prefix-only"):
        prefix_identity(seq, namespace="ns")
