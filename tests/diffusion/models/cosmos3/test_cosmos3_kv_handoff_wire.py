# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Cosmos3 reasoner -> generator K/V payload must survive the connector serde.

The payload is nested (fingerprint -> per-layer ``(K, V)``) and the branch keys are
content hashes, so it has to round-trip bit-exactly. bf16 is the interesting
dtype: it has no numpy equivalent.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_disagg import (
    Cosmos3TextConditioning,
    _ReplayLanguageModel,
    fingerprint_text_ids,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_KV_KEY as KV_KEY,
)
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerde

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

NUM_LAYERS = 4
KV_HEADS = 2
HEAD_DIM = 8
SEQ = 3


def _payload(dtype: torch.dtype) -> dict:
    torch.manual_seed(0)
    branches = {}
    for branch, ids in enumerate((torch.tensor([[11, 12, 13]]), torch.tensor([[21, 22, 23]]))):
        branches[fingerprint_text_ids(ids)] = [
            (
                torch.randn(1, SEQ, KV_HEADS, HEAD_DIM).to(dtype) + branch,
                torch.randn(1, SEQ, KV_HEADS, HEAD_DIM).to(dtype) - branch,
            )
            for _ in range(NUM_LAYERS)
        ]
    return Cosmos3TextConditioning(
        branches=branches,
        num_layers=NUM_LAYERS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        height=1024,
        width=1024,
        max_sequence_length=512,
        use_system_prompt=False,
        reasoner_tp_size=1,
        payload_mib=0.1,
    ).to_payload()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_decoded_payload_is_replayable_bit_exactly(dtype: torch.dtype):
    """What comes off the wire rebuilds the contract, installs, and replays unchanged."""
    payload = _payload(dtype)
    serde = OmniSerde()
    decoded = serde.deserialize(serde.serialize(payload))

    stub = _ReplayLanguageModel(
        NUM_LAYERS,
        torch.nn.Identity(),
        num_kv_heads=KV_HEADS,
        num_kv_heads_local=KV_HEADS,
        kv_head_offset=0,
        head_dim=HEAD_DIM,
    )
    stub.install(Cosmos3TextConditioning.from_payload(decoded), dtype=dtype)

    text_ids = torch.tensor([[11, 12, 13]])
    replayed = stub(text_ids, freqs=None)

    expected = payload[KV_KEY][fingerprint_text_ids(text_ids)]
    assert len(replayed) == NUM_LAYERS
    for (k, v), (ek, ev) in zip(replayed, expected, strict=True):
        assert k.dtype == dtype
        assert torch.equal(k, ek)
        assert torch.equal(v, ev)
