# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/FunAudioLLM/CosyVoice/tree/main/cosyvoice/flow
"""Conditional Flow Matching (CFM) classes for audio generation."""

import os
from abc import ABC

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask

logger = init_logger(__name__)

# Incremental streaming decode with a per-layer DiT attention KV cache. When
# enabled each streaming chunk denoises only its own new frames against a frozen
# prefix K/V cache, turning the O(N^2) cumulative re-denoise into O(N). Default
# on; export COSYVOICE3_DIT_KV_CACHE=0 for the full-sequence path.
KV_CACHE = os.environ.get("COSYVOICE3_DIT_KV_CACHE", "1") == "1"

# Optional bf16 flow (A/B lever). When set, the DiT flow estimator, the Euler
# solver's CFG input buffers, and the streaming KV cache run in bfloat16, while
# the flow's input/output contract stays fp32 (mu/cond/spks in, mel out) and the
# ODE state accumulates in fp32. RoPE frequency/position math is kept in fp32
# (bf16 rounds absolute positions past ~256, corrupting long-sequence rotary
# embeddings). Default off; export COSYVOICE3_FLOW_BF16=1 to enable.
FLOW_BF16 = os.environ.get("COSYVOICE3_FLOW_BF16", "0") == "1"


class DiTStreamCache:
    """Per-request incremental streaming state for the DiT flow decoder.

    The prefix K/V and causal-conv tails are cached *per Euler step*: entry
    ``att_caches[s]`` is the finalized prefix's per-layer K/V at denoising step
    ``s`` (a list of ``(K, V)``), and ``cnn_caches[s]`` the conv tails at that
    step. Caching per step lets each new chunk attend to the prefix at the
    matching denoising ``t`` (not a frozen t=1), which preserves quality.
    ``finalized_len`` is the number of finalized mel frames.
    """

    __slots__ = ("att_caches", "cnn_caches", "finalized_len")

    def __init__(self):
        self.att_caches = None
        self.cnn_caches = None
        self.finalized_len = 0


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator

    @torch.inference_mode()
    def forward(
        self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=torch.zeros(1, 80, 0, 2)
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond (Optional[Any], optional): Not used but kept for future purposes

        Returns:
            sample (torch.Tensor): generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def _maybe_cast_estimator(self):
        """Cast the DiT estimator to bf16 once when COSYVOICE3_FLOW_BF16 is set.

        RoPE frequency/position math is kept in fp32; the rest of the estimator
        (projections, attention, FFN, norms, timestep MLP) runs in bf16.
        """
        if not FLOW_BF16 or getattr(self, "_flow_bf16_ready", False):
            return
        if isinstance(self.estimator, torch.nn.Module):
            self.estimator.to(torch.bfloat16)
            rotary = getattr(self.estimator, "rotary_embed", None)
            if rotary is not None:
                rotary.to(torch.float32)
        self._flow_bf16_ready = True

    def solve_euler(self, x, t_span, mu, mask, spks, cond, stream_cache=None):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond (Optional[Any], optional): Not used but kept for future purposes
            stream_cache (DiTStreamCache, optional): when given, only the new
                frames in ``x`` are denoised against the frozen prefix KV cache.
        """
        self._maybe_cast_estimator()
        if stream_cache is not None:
            return self._solve_euler_streaming(x, t_span, mu, spks, cond, stream_cache)
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        # NOTE when flow run in amp mode, x.dtype is float32, which cause nan in trt fp16
        # inference, so set dtype=spks.dtype
        compute_dtype = torch.bfloat16 if FLOW_BF16 else spks.dtype
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=compute_dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=compute_dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=compute_dtype)
        t_in = torch.zeros([2], device=x.device, dtype=compute_dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=compute_dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=compute_dtype)
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.forward_estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def _solve_euler_streaming(self, x, t_span, mu, spks, cond, cache):
        """Euler solver over only the new chunk's frames, using the KV cache.

        ``x``/``mu``/``cond`` are ``(1, n_feats, n_new)`` for the new frames;
        ``x`` is the fixed-noise slice for the new absolute positions. At each
        Euler step the new frames attend to the finalized prefix's K/V captured
        at that *same* step (``cache.att_caches[s]``), so prefix and new frames
        share the denoising ``t``. The new frames' per-step K/V and conv tails
        are then appended to the caches. Returns the finalized new frames
        ``(1, n_feats, n_new)``.
        """
        self._maybe_cast_estimator()
        estimator = self.estimator  # DiT
        offset = cache.finalized_len
        n_new = x.size(2)
        device = x.device
        dtype = torch.bfloat16 if FLOW_BF16 else spks.dtype
        n_steps = len(t_span) - 1

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # CFG buffers: row 0 conditioned, row 1 unconditioned.
        x_in = torch.zeros([2, 80, n_new], device=device, dtype=dtype)
        mu_in = torch.zeros([2, 80, n_new], device=device, dtype=dtype)
        t_in = torch.zeros([2], device=device, dtype=dtype)
        spks_in = torch.zeros([2, 80], device=device, dtype=dtype)
        cond_in = torch.zeros([2, 80, n_new], device=device, dtype=dtype)
        mu_in[0] = mu
        spks_in[0] = spks
        cond_in[0] = cond

        new_kv_per_step: list = [None] * n_steps
        new_cnn_per_step: list = [None] * n_steps
        for step in range(1, len(t_span)):
            s = step - 1
            att_s = cache.att_caches[s] if cache.att_caches is not None else None
            cnn_s = cache.cnn_caches[s] if cache.cnn_caches is not None else None
            x_in[:] = x
            t_in[:] = t.unsqueeze(0)
            dphi_dt, kv_s, cnn_new_s = estimator.forward_chunk(
                x_in, mu_in, t_in, spks_in, cond_in, att_s, cnn_s, offset
            )
            new_kv_per_step[s] = kv_s
            new_cnn_per_step[s] = cnn_new_s
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [1, 1], dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        # Append the new frames' per-step K/V; slide the per-step conv tails.
        if cache.att_caches is None:
            cache.att_caches = [
                [(nk.contiguous(), nv.contiguous()) for (nk, nv) in new_kv_per_step[s]] for s in range(n_steps)
            ]
            cache.cnn_caches = [new_cnn_per_step[s] for s in range(n_steps)]
        else:
            for s in range(n_steps):
                prev = cache.att_caches[s]
                cache.att_caches[s] = [
                    (
                        torch.cat([prev[i][0], new_kv_per_step[s][i][0]], dim=1).contiguous(),
                        torch.cat([prev[i][1], new_kv_per_step[s][i][1]], dim=1).contiguous(),
                    )
                    for i in range(len(new_kv_per_step[s]))
                ]
                cache.cnn_caches[s] = new_cnn_per_step[s]
        cache.finalized_len = offset + n_new
        return x.float()

    def forward_estimator(self, x, mask, mu, t, spks, cond):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator(x, mask, mu, t, spks, cond)
        else:
            # TensorRT estimator: bind raw device pointers. The flow runs in
            # fp32 but the engine may have fp16 I/O (strongly-typed fp16 engine),
            # so cast inputs/output to the engine's dtype at the boundary. Keep
            # references to the cast buffers alive until execute completes (a bare
            # ``.contiguous().data_ptr()`` could free the temp -> dangling ptr).
            io_dtype = getattr(self.estimator, "io_dtype", x.dtype)
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            # NOTE need to synchronize when switching stream
            torch.cuda.current_stream().synchronize()
            with stream:
                x_e = x.to(io_dtype).contiguous()
                mask_e = mask.to(io_dtype).contiguous()
                mu_e = mu.to(io_dtype).contiguous()
                t_e = t.to(io_dtype).contiguous()
                spks_e = spks.to(io_dtype).contiguous()
                cond_e = cond.to(io_dtype).contiguous()
                out_e = torch.empty_like(x_e)
                estimator.set_input_shape("x", (2, 80, x_e.size(2)))
                estimator.set_input_shape("mask", (2, 1, x_e.size(2)))
                estimator.set_input_shape("mu", (2, 80, x_e.size(2)))
                estimator.set_input_shape("t", (2,))
                estimator.set_input_shape("spks", (2, 80))
                estimator.set_input_shape("cond", (2, 80, x_e.size(2)))
                data_ptrs = [
                    x_e.data_ptr(),
                    mask_e.data_ptr(),
                    mu_e.data_ptr(),
                    t_e.data_ptr(),
                    spks_e.data_ptr(),
                    cond_e.data_ptr(),
                    out_e.data_ptr(),
                ]
                for i, j in enumerate(data_ptrs):
                    estimator.set_tensor_address(trt_engine.get_tensor_name(i), j)
                # run trt engine
                assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)
            return out_e.to(x.dtype)


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        # Fixed noise indexed by absolute mel position. Streaming chunk denoise
        # must draw the new frames' noise from the same buffer each chunk (a
        # fresh randn per chunk would break prefix continuity and cache
        # validity). Seeded, non-persistent so it stays out of the state dict.
        # Build on CPU explicitly: model init runs under a musa default device,
        # so a CPU generator with an implicit-device randn would mismatch.
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        self.register_buffer("rand_noise", torch.randn([1, 80, 50 * 300], generator=g, device="cpu"), persistent=False)

    @torch.inference_mode()
    def forward(
        self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming: bool = False, stream_cache=None
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond (Optional[Any], optional): Not used but kept for future purposes
            stream_cache (DiTStreamCache, optional): incremental streaming state;
                when given only the frames past ``finalized_len`` are denoised.

        Returns:
            sample (torch.Tensor): generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        if stream_cache is not None:
            offset = stream_cache.finalized_len
            full_len = mu.size(2)
            if full_len <= offset:
                return mu.new_zeros((mu.size(0), mu.size(1), 0)), None
            if full_len > self.rand_noise.size(2):
                g = torch.Generator(device="cpu")
                g.manual_seed(0)
                self.rand_noise = torch.randn([1, 80, full_len + 50 * 50], generator=g, device="cpu").to(
                    self.rand_noise.device
                )
            z = self.rand_noise[:, :, offset:full_len].to(mu.device).to(mu.dtype) * temperature
            mu_new = mu[:, :, offset:full_len]
            cond_new = cond[:, :, offset:full_len]
            t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
            if self.t_scheduler == "cosine":
                t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
            feat = self.solve_euler(
                z, t_span=t_span, mu=mu_new, mask=None, spks=spks, cond=cond_new, stream_cache=stream_cache
            )
            return feat, None

        z = (
            torch.randn(
                (mu.size(0), mu.size(1), mu.size(2)),
                device=mu.device,
                dtype=mu.dtype,
            )
            * temperature
        )

        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)

        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None


class CausalMaskedDiffWithDiT(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        pre_lookahead_layer: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        decoder_conf: dict = {
            "in_channels": 240,
            "out_channel": 80,
            "spk_emb_dim": 80,
            "n_spks": 1,
            "cfm_params": DictConfig(
                {
                    "sigma_min": 1e-06,
                    "solver": "euler",
                    "t_scheduler": "cosine",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                    "reg_loss_type": "l1",
                }
            ),
            "decoder_params": {
                "channels": [256, 256],
                "dropout": 0.0,
                "attention_head_dim": 64,
                "n_blocks": 4,
                "num_mid_blocks": 12,
                "num_heads": 8,
                "act_fn": "gelu",
            },
        },
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logger.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        streaming: bool = True,
        finalize: bool = False,
        n_timesteps: int = 10,
        stream_cache=None,
    ):
        assert token.shape[0] == 1
        # xvec projection

        embedding = F.normalize(embedding, dim=1)

        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask
        # text encode
        if finalize is True:
            h = self.pre_lookahead_layer(token)
        else:
            h = self.pre_lookahead_layer(
                token[:, : -self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len :]
            )
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)

        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)

        conds[:, :mel_len1] = prompt_feat

        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        prev_finalized = stream_cache.finalized_len if stream_cache is not None else 0
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=max(1, int(n_timesteps)),
            streaming=streaming,
            stream_cache=stream_cache,
        )

        if stream_cache is not None:
            # The decoder returned only the newly denoised frames
            # [prev_finalized : mel_len1 + mel_len2]. Drop the prompt portion on
            # the first chunk so the caller still receives generated frames only.
            if prev_finalized < mel_len1:
                feat = feat[:, :, mel_len1 - prev_finalized :]
            return feat.float(), None

        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None
