# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.diffusion_kv.metadata import (
    DiffusionKVContextMetadata,
    DiffusionKVMetadata,
    DiffusionKVSequenceMetadata,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def test_diffusion_kv_metadata_uses_native_cache_group_block_ids() -> None:
    context = DiffusionKVContextMetadata(
        context_id="text",
        cache_role="cross_attention",
        num_tokens=3,
        block_ids=([7], [11]),
    )
    sequence = DiffusionKVSequenceMetadata(
        sequence_id=1,
        prefix_len=4,
        target_len=2,
        seq_len=8,
        block_ids=([1, 2], [5, 6]),
        context_ids=(context.context_id,),
    )
    metadata = DiffusionKVMetadata(
        request_id="req-0",
        allocation_generation=3,
        sequences=(sequence,),
        contexts=(context,),
    )

    assert metadata.sequences[0].block_ids == ([1, 2], [5, 6])
    assert metadata.contexts[0].block_ids == ([7], [11])


def test_cfg_sequences_can_share_one_request_context() -> None:
    shared_context = DiffusionKVContextMetadata(
        context_id="shared-text",
        cache_role="cross_attention",
        num_tokens=3,
        block_ids=([7], [11]),
    )
    sequences = tuple(
        DiffusionKVSequenceMetadata(
            sequence_id=sequence_id,
            prefix_len=4,
            target_len=2,
            seq_len=8,
            block_ids=([sequence_id + 1],),
            context_ids=(shared_context.context_id,),
        )
        for sequence_id in range(2)
    )

    metadata = DiffusionKVMetadata(
        request_id="req-shared-context",
        allocation_generation=1,
        sequences=sequences,
        contexts=(shared_context,),
    )

    assert metadata.contexts == (shared_context,)
    assert metadata.sequences[0].context_ids == (shared_context.context_id,)
    assert metadata.sequences[1].context_ids == (shared_context.context_id,)


def test_cfg_sequences_can_reference_branch_specific_contexts() -> None:
    conditional_context = DiffusionKVContextMetadata(
        context_id="conditional-text",
        cache_role="cross_attention",
        num_tokens=3,
        block_ids=([7], [11]),
    )
    unconditional_context = DiffusionKVContextMetadata(
        context_id="unconditional-text",
        cache_role="cross_attention",
        num_tokens=3,
        block_ids=([8], [12]),
    )
    metadata = DiffusionKVMetadata(
        request_id="req-branch-contexts",
        allocation_generation=1,
        sequences=(
            DiffusionKVSequenceMetadata(
                sequence_id=0,
                prefix_len=4,
                target_len=2,
                seq_len=8,
                block_ids=([1],),
                context_ids=(conditional_context.context_id,),
            ),
            DiffusionKVSequenceMetadata(
                sequence_id=1,
                prefix_len=4,
                target_len=2,
                seq_len=8,
                block_ids=([2],),
                context_ids=(unconditional_context.context_id,),
            ),
        ),
        contexts=(conditional_context, unconditional_context),
    )

    assert metadata.sequences[0].context_ids == (conditional_context.context_id,)
    assert metadata.sequences[1].context_ids == (unconditional_context.context_id,)
