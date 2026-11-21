# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Automatic placement must restore metadata across native phase transitions."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.sparsp.config import SparSPSpec
from vllm_omni.diffusion.attention.sparsp.placement import interval_periodic_layout, supports_interval_periodic_layout
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.models.ltx2.sparsp_layout import LTX2SparSPLayout

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_four_rank_paper_geometry_preserves_query_and_key_tiles():
    low, high = (40, 12, 24), (40, 24, 48)
    assert not supports_interval_periodic_layout(low, 4)
    with pytest.raises(ValueError, match="Q128"):
        interval_periodic_layout(low, 4, torch.device("cpu"))
    layout = interval_periodic_layout(high, 4, torch.device("cpu"))
    tokens = torch.arange(46080)
    assert torch.equal(layout.permutation[layout.inverse], tokens)
    query_order = layout.permutation.view(-1, 128)
    torch.testing.assert_close(query_order, query_order[:, :1] + torch.arange(128))
    query_blocks = query_order[:, 0] // 128
    assert torch.equal(query_blocks, torch.arange(360).view(40, 9).T.flatten())
    for rank_order in layout.permutation.view(4, -1):
        key_order = rank_order.view(-1, 64)
        torch.testing.assert_close(key_order, key_order[:, :1] + torch.arange(64))


def test_auto_layout_transitions_keep_tokens_coordinates_and_timestep_together():
    observed = []
    spec = SimpleNamespace(sparsp=SparSPSpec(placement="auto"))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            attention = SimpleNamespace(effective_attention=lambda: (None, None, spec))
            self.transformer_blocks = [SimpleNamespace(attn1=SimpleNamespace(attn=attention))]
            self.fail = False

        def forward(
            self, hidden_states, audio_hidden_states, num_frames, height, width, video_coords, timestep, keyframes_mask
        ):
            observed.append(get_forward_context().sparsp_video_layout is not None)
            torch.testing.assert_close(video_coords[:, 0, :, 0], hidden_states[:, :, 0])
            torch.testing.assert_close(timestep, hidden_states[:, :, 0])
            torch.testing.assert_close(keyframes_mask, hidden_states[:, :, 0] % 2 == 0)
            if self.fail:
                raise RuntimeError("injected model failure")
            return hidden_states, audio_hidden_states

    model = Model()
    adapter = LTX2SparSPLayout(model)
    config = SimpleNamespace(diffusion_attention_schedule=None, parallel_config=SimpleNamespace(ring_degree=2))
    with set_forward_context(omni_diffusion_config=config):
        for grid in ((2, 8, 8), (2, 16, 16), (2, 8, 8)):
            frames, height, width = grid
            tokens = torch.arange(frames * height * width, dtype=torch.float32).view(1, -1, 1)
            audio = torch.ones(1, 9, 4)
            kwargs = dict(
                hidden_states=tokens,
                audio_hidden_states=audio,
                num_frames=frames,
                height=height,
                width=width,
                video_coords=tokens[:, None].expand(-1, 3, -1, 2),
                timestep=tokens[:, :, 0],
                keyframes_mask=tokens[:, :, 0] % 2 == 0,
            )
            video_out, audio_out = model(**kwargs)
            torch.testing.assert_close(video_out, tokens, atol=0, rtol=0)
            assert audio_out is audio
            assert get_forward_context().sparsp_video_layout is None
            assert not adapter.active
        assert observed == [False, True, False]
        model.fail = True
        kwargs.update(
            hidden_states=torch.zeros(1, 512, 1),
            num_frames=2,
            height=16,
            width=16,
            video_coords=torch.zeros(1, 3, 512, 2),
            timestep=torch.zeros(1, 512),
            keyframes_mask=torch.ones(1, 512, dtype=torch.bool),
        )
        with pytest.raises(RuntimeError, match="injected model failure"):
            model(**kwargs)
        assert get_forward_context().sparsp_video_layout is None
        assert not adapter.active
