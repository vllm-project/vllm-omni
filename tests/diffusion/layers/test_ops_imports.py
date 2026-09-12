# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_qk_norm_rope_legacy_import_forwards_to_public_surface() -> None:
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        fused_qk_norm_rope as legacy_op,
    )
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        fused_qk_norm_rope_min_tokens as legacy_min_tokens,
    )
    from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
        fused_qk_norm_rope_supported as legacy_supported,
    )
    from vllm_omni.diffusion.layers.ops import (
        fused_qk_norm_rope,
        fused_qk_norm_rope_min_tokens,
        fused_qk_norm_rope_supported,
    )
    from vllm_omni.diffusion.layers.ops.rope.qk_norm_rope import (
        fused_qk_norm_rope as canonical_op,
    )

    assert legacy_op is fused_qk_norm_rope is canonical_op
    assert legacy_min_tokens is fused_qk_norm_rope_min_tokens
    assert legacy_supported is fused_qk_norm_rope_supported


def test_qk_norm_rope_support_query_rejects_cpu_inputs() -> None:
    from vllm_omni.diffusion.layers.ops import fused_qk_norm_rope_supported

    q = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 1, 128, dtype=torch.bfloat16)

    assert not fused_qk_norm_rope_supported(q, k, 128, 96)
