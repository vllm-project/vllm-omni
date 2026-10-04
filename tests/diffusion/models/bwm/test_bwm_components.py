# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L1 unit tests for Boundless-World-Model components (CPU).

Covers the action encoder's shape contracts and frame grouping, the
checkpoint weight-name conversion map, and the condition embedder's action
modulation and text-pathway bypass.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.bwm.bwm_action_encoder import BWMActionEncoder
from vllm_omni.diffusion.models.bwm.pipeline_bwm import BoundlessWorldModelPipeline, resolve_num_frames

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ACTION_DIM = 14
DIM = 64  # small stand-in for the 3072 DiT dim


def _load_download_module():
    path = Path(__file__).resolve().parents[4] / "examples" / "offline_inference" / "bwm" / "download_bwm.py"
    spec = importlib.util.spec_from_file_location("download_bwm", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_bwm"] = module
    spec.loader.exec_module(module)
    return module


class TestActionEncoder:
    def test_output_shapes(self):
        enc = BWMActionEncoder(action_dim=ACTION_DIM, dim=DIM)
        num_latent_frames = 15
        frames = 1 + 4 * (num_latent_frames - 1)  # 57
        action = torch.randn(1, frames, ACTION_DIM)
        context_emb, mod_emb = enc(action)
        assert context_emb.shape == (1, frames, DIM)
        assert mod_emb.shape == (1, num_latent_frames, DIM)

    def test_leading_group_replicates_first_frame(self):
        """The first latent frame's group is [a0, a0, a0, a0] (frame layout
        1 + 4*(T-1)): a trajectory with distinct first action must produce a
        first modulation equal to encoding that action alone repeated."""
        enc = BWMActionEncoder(action_dim=ACTION_DIM, dim=DIM)
        action = torch.randn(1, 5, ACTION_DIM)  # 2 latent frames
        _, mod = enc(action)
        first_group = action[:, 0:1].repeat(1, 4, 1).reshape(1, 1, ACTION_DIM * 4)
        expected_first = enc.action_mlp2(first_group)
        assert torch.allclose(mod[:, 0], expected_first[:, 0], atol=1e-6)


class TestWeightConversion:
    def test_all_bwm_checkpoint_key_families_map(self):
        m = _load_download_module()
        # Representative keys from the released step-12000.safetensors.
        cases = {
            "patch_embedding.weight": "patch_embedding.weight",
            "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
            "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
            "time_projection.1.weight": "condition_embedder.time_proj.weight",
            "head.head.weight": "proj_out.weight",
            "head.modulation": "scale_shift_table",
            "blocks.0.self_attn.q.weight": "blocks.0.attn1.to_q.weight",
            "blocks.0.self_attn.o.bias": "blocks.0.attn1.to_out.0.bias",
            "blocks.0.self_attn.norm_q.weight": "blocks.0.attn1.norm_q.weight",
            "blocks.29.cross_attn.v.weight": "blocks.29.attn2.to_v.weight",
            "blocks.3.ffn.0.weight": "blocks.3.ffn.net.0.proj.weight",
            "blocks.3.ffn.2.bias": "blocks.3.ffn.net.2.bias",
            "blocks.7.norm3.weight": "blocks.7.norm2.weight",
            "blocks.7.modulation": "blocks.7.scale_shift_table",
        }
        for src, expected in cases.items():
            assert m.convert_dit_key(src) == expected, src


class TestResolveNumFrames:
    def test_derives_from_action_length_when_unset(self):
        # OmniDiffusionSamplingParams.num_frames defaults to 1 (image-model
        # default); both None and 1 must fall back to the action trajectory.
        assert resolve_num_frames(57, None) == 57
        assert resolve_num_frames(57, 1) == 57
        assert resolve_num_frames(59, None) == 57  # snapped to the 4n+1 grid

    def test_honors_explicit_request(self):
        assert resolve_num_frames(100, 57) == 57
        assert resolve_num_frames(100, 58) == 57  # snapped down

    def test_rejects_empty_action(self):
        with pytest.raises(ValueError):
            resolve_num_frames(0, None)


class TestHistoryFrames:
    to_tensor = staticmethod(BoundlessWorldModelPipeline._history_frames_to_tensor)

    def test_uint8_scaled_to_minus_one_one(self):
        frames = np.full((2, 32, 48, 3), 255, dtype=np.uint8)
        video = self.to_tensor(frames)
        assert video.shape == (1, 3, 2, 32, 48)
        assert torch.allclose(video, torch.ones_like(video))

    def test_unit_float_rescaled(self):
        frames = np.zeros((1, 32, 48, 3), dtype=np.float32)  # [0, 1] range
        video = self.to_tensor(frames)
        assert torch.allclose(video, -torch.ones_like(video))

    def test_signed_float_passthrough(self):
        frames = -np.ones((1, 32, 48, 3), dtype=np.float32)  # already [-1, 1]
        video = self.to_tensor(frames)
        assert torch.allclose(video, -torch.ones_like(video))

    def test_resize_to_requested_resolution(self):
        frames = np.random.randint(0, 255, (2, 32, 48, 3), dtype=np.uint8)
        video = self.to_tensor(frames, height=64, width=96)
        assert video.shape == (1, 3, 2, 64, 96)


class TestConditionEmbedder:
    def _make_embedder(self):
        from vllm_omni.diffusion.models.bwm.bwm_condition_embedder import BWMConditionEmbedder
        from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTimeTextImageEmbedding

        inner = WanTimeTextImageEmbedding(
            dim=DIM,
            time_freq_dim=32,
            time_proj_dim=DIM * 6,
            text_embed_dim=DIM,
        )
        inner.__class__ = BWMConditionEmbedder
        inner.action_mod_emb = None
        return inner

    def test_text_projection_bypassed(self):
        emb = self._make_embedder()
        context = torch.randn(1, 57, DIM)
        timestep = torch.tensor([500.0])
        _, _, out_context, out_img = emb(timestep, context)
        # Action context must pass through unchanged (no text projection).
        assert torch.equal(out_context, context)
        assert out_img is None

    def test_action_modulation_changes_temb_per_latent_frame(self):
        emb = self._make_embedder()
        num_latent_frames, spatial_tokens = 3, 4
        seq_len = num_latent_frames * spatial_tokens
        context = torch.randn(1, 10, DIM)
        # Per-token timesteps (expand_timesteps mode).
        timestep = torch.full((seq_len,), 500.0)

        temb_base, proj_base, _, _ = emb(timestep, context, timestep_seq_len=seq_len)

        emb.action_mod_emb = torch.zeros(1, num_latent_frames, DIM)
        emb.action_mod_emb[:, 1] = 1.0  # perturb only the middle latent frame
        temb_mod, proj_mod, _, _ = emb(timestep, context, timestep_seq_len=seq_len)
        emb.action_mod_emb = None

        delta = (temb_mod - temb_base).abs().sum(dim=-1)[0]  # (seq_len,)
        frame0 = delta[:spatial_tokens]
        frame1 = delta[spatial_tokens : 2 * spatial_tokens]
        frame2 = delta[2 * spatial_tokens :]
        assert torch.all(frame0 == 0) and torch.all(frame2 == 0)
        assert torch.all(frame1 > 0)
        assert not torch.allclose(proj_mod, proj_base)
