# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC

import torch

from vllm_omni.model_executor.models.indextts2.s2mel.modules.diffusion_transformer import DiT


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = None

        self.in_channels = args.DiT.in_channels

        if hasattr(args.DiT, "zero_prompt_speech_token"):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
        """Forward diffusion

        Args:
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)
            f0 (None): unused, reserved for f0 conditioning
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.

        Returns:
            sample (torch.Tensor): generated mel-spectrogram
                shape: (batch_size, 80, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device, dtype=mu.dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)
        """
        t = t_span[0]

        # Optional bf16 autocast for the DiT estimator (set by the decoder via
        # `estimator_autocast_dtype`). The Euler solver state stays float32 —
        # only the estimator forward runs in reduced precision, which restores
        # flash-attention eligibility and halves GEMM cost.
        autocast_dtype = getattr(self, "estimator_autocast_dtype", None)

        def _run_estimator(*args, **kw):
            if autocast_dtype is None:
                return self.estimator(*args, **kw)
            with torch.autocast(x.device.type, dtype=autocast_dtype):
                out = self.estimator(*args, **kw)
            return out.float()

        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0

        # Precompute mask once — T and x_lens are constant across all ODE steps.
        # x_mask [B, 1, T] broadcasts to [2B, 1, T] when CFG doubles batch.
        pre_mask = self._precompute_mask(x, x_lens)

        if inference_cfg_rate > 0:
            batch = x.shape[0]
            cfg_prompt_x = prompt_x.new_zeros((2 * batch, *prompt_x.shape[1:]))
            cfg_prompt_x[:batch].copy_(prompt_x)
            cfg_style = style.new_zeros((2 * batch, *style.shape[1:]))
            cfg_style[:batch].copy_(style)
            cfg_mu = mu.new_zeros((2 * batch, *mu.shape[1:]))
            cfg_mu[:batch].copy_(mu)
            cfg_x = x.new_empty((2 * batch, *x.shape[1:]))
            cfg_t = x.new_empty((2 * batch,))
            cfg_pre_mask = self._repeat_pre_mask_for_cfg(pre_mask)

        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                cfg_x[:batch].copy_(x)
                cfg_x[batch:].copy_(x)
                cfg_t.fill_(t)

                stacked_dphi_dt = _run_estimator(
                    cfg_x,
                    cfg_prompt_x,
                    x_lens,
                    cfg_t,
                    cfg_style,
                    cfg_mu,
                    pre_mask=cfg_pre_mask,
                )

                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = _run_estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu, pre_mask=pre_mask)

            x = x + dt * dphi_dt
            t = t + dt
            x[:, :, :prompt_len] = 0

        return x

    @staticmethod
    def _repeat_pre_mask_for_cfg(pre_mask):
        """Duplicate precomputed masks when CFG doubles the batch.

        The original IndexTTS2 inference path used B=1, where a two-element
        timestep tensor happened to match CFG. Stage-1 batching makes the
        estimator input batch 2B, so all batch-indexed masks must be repeated
        consistently with ``torch.cat([cond, uncond], dim=0)``.
        """
        if not isinstance(pre_mask, tuple):
            return pre_mask
        repeated = []
        for mask in pre_mask:
            if isinstance(mask, torch.Tensor):
                repeated.append(torch.cat([mask, mask], dim=0))
            else:
                repeated.append(mask)
        return tuple(repeated)

    def _precompute_mask(self, x: torch.Tensor, x_lens: torch.Tensor):
        """Precompute padding mask for DiT — reused across all ODE steps."""
        from .commons import sequence_mask

        estimator = self._eager_estimator if hasattr(self, "_eager_estimator") else self.estimator
        T = x.size(2)
        style_offset = getattr(estimator, "style_as_token", False)
        time_offset = getattr(estimator, "time_as_token", False)
        T_in = T + int(style_offset) + int(time_offset)
        is_causal = getattr(estimator, "is_causal", False)

        x_mask = sequence_mask(x_lens + int(style_offset) + int(time_offset), max_length=T_in).to(x.device).unsqueeze(1)
        if is_causal:
            return (x_mask, None)
        x_mask_expanded = x_mask[:, None, :].expand(-1, -1, T_in, -1)
        return (x_mask, x_mask_expanded)


class CFM(BASECFM):
    def __init__(self, args):
        super().__init__(args)
        if args.dit_type == "DiT":
            self.estimator = DiT(args)
            object.__setattr__(self, "_eager_estimator", self.estimator)
        else:
            raise NotImplementedError(f"Unknown diffusion type {args.dit_type}")
        self._compiled = False

    def enable_torch_compile(self):
        """Enable torch.compile optimization for the estimator model.

        This method applies torch.compile to the estimator (DiT model) for significant
        performance improvements during inference. It also configures distributed
        training optimizations if applicable.
        """
        if torch.distributed.is_initialized():
            torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.estimator = torch.compile(
            self._eager_estimator,
            fullgraph=True,
            dynamic=True,
        )
        self._compiled = True

    def disable_torch_compile(self):
        self.estimator = self._eager_estimator
        self._compiled = False
