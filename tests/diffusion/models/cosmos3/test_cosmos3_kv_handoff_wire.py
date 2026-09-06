# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Cosmos3 reasoner -> generator K/V payload must survive the stage edge.

Unlike the AR stages, the diffusion path has no ``_ensure_tensor_values`` boundary
flattening ``multimodal_output`` to ``dict[str, torch.Tensor]``, so the handoff is
a *nested* structure: a fingerprint-keyed dict of per-layer ``(K, V)`` pairs, plus
a metadata dict of ints, floats and bools. Everything about the split depends on
that surviving serialization bit-exactly -- the fingerprints are content hashes, so
a payload that decodes to slightly different values does not degrade the image, it
misses the replay table.

bf16 is the case worth pinning: it has no numpy dtype, so the encoder round-trips
it through a ``uint8`` view rather than ``.numpy()``.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_disagg import (
    _ReplayLanguageModel,
    fingerprint_text_ids,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_KV_KEY as KV_KEY,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_META_KEY as META_KEY,
)
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerde

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

NUM_LAYERS = 4
KV_HEADS = 2
HEAD_DIM = 8
SEQ = 3


def _payload(dtype: torch.dtype) -> dict:
    """A two-branch payload shaped like the reasoner's, with distinguishable values."""
    torch.manual_seed(0)
    table = {}
    for branch, ids in enumerate((torch.tensor([[11, 12, 13]]), torch.tensor([[21, 22, 23]]))):
        table[fingerprint_text_ids(ids)] = [
            (
                torch.randn(1, SEQ, KV_HEADS, HEAD_DIM).to(dtype) + branch,
                torch.randn(1, SEQ, KV_HEADS, HEAD_DIM).to(dtype) - branch,
            )
            for _ in range(NUM_LAYERS)
        ]
    return {
        KV_KEY: table,
        META_KEY: {
            "height": 1024,
            "width": 1024,
            "max_sequence_length": 512,
            "use_system_prompt": False,
            "num_branches": 2,
            "payload_mib": 0.1,
            "num_layers": NUM_LAYERS,
            "num_kv_heads_local": KV_HEADS,
            "head_dim": HEAD_DIM,
            "tp_size": 1,
        },
    }


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_kv_payload_survives_the_connector_serde_bit_exactly(dtype: torch.dtype):
    payload = _payload(dtype)
    serde = OmniSerde()

    decoded = serde.deserialize(serde.serialize(payload))

    assert set(decoded[KV_KEY]) == set(payload[KV_KEY])
    for fingerprint, branch in payload[KV_KEY].items():
        replayed = decoded[KV_KEY][fingerprint]
        assert len(replayed) == NUM_LAYERS
        for (k, v), (dk, dv) in zip(branch, replayed, strict=True):
            assert dk.dtype == k.dtype
            assert dk.shape == k.shape
            # Bit-exact, not allclose: the fingerprint is a content hash, so a
            # lossy hop would show up as a replay miss rather than a soft error.
            assert torch.equal(dk, k)
            assert torch.equal(dv, v)


def test_metadata_survives_with_its_types():
    """The generator's layout check compares these with ``!=``, so ints must stay
    ints -- a float 64.0 for num_layers would read as a mismatch."""
    payload = _payload(torch.bfloat16)
    serde = OmniSerde()

    meta = serde.deserialize(serde.serialize(payload))[META_KEY]

    assert meta == payload[META_KEY]
    assert isinstance(meta["num_layers"], int)
    assert isinstance(meta["use_system_prompt"], bool)


def test_the_decoded_payload_is_replayable():
    """The end-to-end invariant: what comes off the wire installs and replays.

    ``install`` validates the layout and ``forward`` looks branches up by
    fingerprint, so this covers the two ways a serialization change could break the
    split -- a reshaped tensor and a perturbed value.
    """
    payload = _payload(torch.bfloat16)
    serde = OmniSerde()
    decoded = serde.deserialize(serde.serialize(payload))

    stub = _ReplayLanguageModel(
        NUM_LAYERS,
        torch.nn.Identity(),
        num_kv_heads_local=KV_HEADS,
        head_dim=HEAD_DIM,
    )
    stub.install(decoded[KV_KEY], dtype=torch.bfloat16)

    text_ids = torch.tensor([[11, 12, 13]])
    replayed = stub(text_ids, freqs=None)

    assert len(replayed) == NUM_LAYERS
    expected = payload[KV_KEY][fingerprint_text_ids(text_ids)]
    for (k, v), (ek, ev) in zip(replayed, expected, strict=True):
        assert torch.equal(k, ek)
        assert torch.equal(v, ev)


def test_tuples_arrive_as_lists_which_the_stub_tolerates():
    """msgpack has no tuple type, so the ``(K, V)`` pairs decode as 2-lists. The
    stub deliberately unpacks by iteration rather than requiring tuples."""
    serde = OmniSerde()
    decoded = serde.deserialize(serde.serialize(_payload(torch.bfloat16)))

    first_branch = next(iter(decoded[KV_KEY].values()))

    assert isinstance(first_branch[0], list)
    assert len(first_branch[0]) == 2
