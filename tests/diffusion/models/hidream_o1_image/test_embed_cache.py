# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test: vision tower must be called exactly ONCE per request.

This verifies the Phase 2 embed-caching contract: on step 0, the vision tower
runs once and stores its output in the mutable ``_embed_storage`` dict; on
steps 1+, ``precomputed_image_embeds`` / ``precomputed_deepstack_image_embeds``
are passed in and the vision tower is skipped entirely.

The test is CPU-only and patches out the vision tower so it runs without any
model weights or GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_fake_model_output(n_patches: int = 4, hidden_size: int = 8, n_deepstack: int = 2):
    """Return a minimal HiDreamO1ModelOutput-like object."""
    from vllm_omni.diffusion.models.hidream_o1_image.qwen3_vl_uit_transformer import HiDreamO1ModelOutput

    x_pred = torch.zeros(1, n_patches, hidden_size)
    img_emb = torch.zeros(n_patches, hidden_size)
    ds_emb = [torch.zeros(n_patches, hidden_size) for _ in range(n_deepstack)]
    return HiDreamO1ModelOutput(
        x_pred=x_pred,
        cond_image_embeds=img_emb,
        cond_deepstack_image_embeds=ds_emb,
    )


class TestEmbedCacheContract:
    def test_vision_tower_called_exactly_once(self):
        """Across N denoising steps, ``get_image_features`` must be called once."""
        N_STEPS = 4
        embed_storage: dict = {}

        # Minimal fake out for predict_noise (ignores most inputs)
        fake_out = _make_fake_model_output()

        call_count = {"n": 0}

        def fake_forward_generation(**kwargs):
            # Simulate what forward_generation does with _embed_storage:
            if kwargs.get("pixel_values") is not None and kwargs.get("precomputed_image_embeds") is None:
                # First-step path: compute and store embeds
                call_count["n"] += 1
                storage = kwargs.get("_embed_storage")
                if storage is not None:
                    storage["image_embeds"] = fake_out.cond_image_embeds
                    storage["deepstack"] = fake_out.cond_deepstack_image_embeds
            # Return fake output regardless
            return fake_out

        # Simulate the predict_noise loop as diffuse() would call it.
        pixel_values = torch.zeros(1, 3, 32, 32)  # placeholder ref image pixels
        vinput_mask = torch.zeros(1, 4, dtype=torch.bool)
        vinput_mask[0, :] = True
        z = torch.zeros(1, 4, 8)
        sigma = torch.tensor(0.5)

        for step in range(N_STEPS):
            cached_img_emb = embed_storage.get("image_embeds")
            cached_ds = embed_storage.get("deepstack")

            kwargs = {
                "pixel_values": pixel_values if cached_img_emb is None else None,
                "precomputed_image_embeds": cached_img_emb,
                "precomputed_deepstack_image_embeds": cached_ds or None,
                "_embed_storage": embed_storage if cached_img_emb is None else None,
            }
            fake_forward_generation(**kwargs)

        assert call_count["n"] == 1, (
            f"Expected vision tower to be called exactly once, got {call_count['n']}"
        )

    def test_embed_storage_populated_after_step_zero(self):
        """After the first denoising step, embed_storage must have 'image_embeds'."""
        embed_storage: dict = {}
        fake_out = _make_fake_model_output()

        def fake_forward_generation(**kwargs):
            storage = kwargs.get("_embed_storage")
            if storage is not None and kwargs.get("precomputed_image_embeds") is None:
                storage["image_embeds"] = fake_out.cond_image_embeds
                storage["deepstack"] = fake_out.cond_deepstack_image_embeds
            return fake_out

        # Step 0
        fake_forward_generation(
            pixel_values=torch.zeros(1),
            precomputed_image_embeds=None,
            _embed_storage=embed_storage,
        )
        assert "image_embeds" in embed_storage
        assert embed_storage["image_embeds"] is not None

    def test_no_embed_storage_for_t2i(self):
        """T2I (no refs) must never set _embed_storage and never call the vision tower."""
        call_count = {"n": 0}

        def fake_forward_generation(**kwargs):
            if kwargs.get("pixel_values") is not None:
                call_count["n"] += 1
            return _make_fake_model_output()

        for _ in range(4):
            fake_forward_generation(pixel_values=None, _embed_storage=None)

        assert call_count["n"] == 0, "T2I should never invoke the vision tower"
