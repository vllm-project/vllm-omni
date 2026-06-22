from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import CausalConditionalCFM

from .flow import DiT


class StepCausalConditionalCFM(CausalConditionalCFM):
    def __init__(
        self,
        estimator: DiT,
        cfm_params,
        in_channels: int = 80,
        n_spks: int = 1,
        spk_emb_dim: int = 192,
    ):
        if isinstance(cfm_params, dict):
            cfm_params = SimpleNamespace(**cfm_params)

        super().__init__(
            in_channels=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
            estimator=estimator,
        )

        self.out_channels = estimator.out_channels
        self.register_buffer(
            "rand_noise",
            torch.randn([1, self.out_channels, 50 * 600]),
            persistent=False,
        )
        self.register_buffer(
            "cnn_cache_buffer",
            torch.zeros(16, 16, 2, 1024, 2),
            persistent=False,
        )
        self.register_buffer(
            "att_cache_buffer",
            torch.zeros(16, 16, 2, 8, 1000, 128),
            persistent=False,
        )

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        streaming: bool = False,
    ):
        z = self.rand_noise[:, :, : mu.size(2)].to(device=mu.device, dtype=mu.dtype) * temperature
        if z.size(0) != mu.size(0):
            z = z.expand(mu.size(0), -1, -1).contiguous()

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None

    def solve_euler_chunk(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
    ):
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
            cond: Not used but kept for future purposes
            cnn_cache: shape (n_time, depth, b, c1+c2, 2)
            att_cache: shape (n_time, depth, b, nh, t, c * 2)
        """
        assert self.inference_cfg_rate > 0, "cfg rate should be > 0"

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)  # (b,)

        # setup initial cache
        if cnn_cache is None:
            cnn_cache = [None for _ in range(len(t_span) - 1)]
        if att_cache is None:
            att_cache = [None for _ in range(len(t_span) - 1)]
        # next chunk's cache at each timestep

        if att_cache[0] is not None:
            last_att_len = att_cache.shape[4]
        else:
            last_att_len = 0

        # constant during denoising
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        for step in range(1, len(t_span)):
            this_att_cache = att_cache[step - 1]
            this_cnn_cache = cnn_cache[step - 1]

            dphi_dt, this_new_cnn_cache, this_new_att_cache = self.estimator.forward_chunk(
                x=x.repeat(2, 1, 1),
                mu=mu_in,
                t=t.repeat(2),
                spks=spks_in,
                cond=cond_in,
                cnn_cache=this_cnn_cache,
                att_cache=this_att_cache,
            )
            dphi_dt, cfg_dphi_dt = dphi_dt.chunk(2, dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

            self.cnn_cache_buffer[step - 1] = this_new_cnn_cache
            self.att_cache_buffer[step - 1][:, :, :, : x.shape[2] + last_att_len, :] = this_new_att_cache

        cnn_cache = self.cnn_cache_buffer
        att_cache = self.att_cache_buffer[:, :, :, :, : x.shape[2] + last_att_len, :]
        return x, cnn_cache, att_cache

    @torch.inference_mode()
    def forward_chunk(
        self,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
    ):
        """
        Args:
            mu(torch.Tensor): shape (b, c, t)
            spks(torch.Tensor): shape (b, 192)
            cond(torch.Tensor): shape (b, c, t)
            cnn_cache: shape (n_time, depth, b, c1+c2, 2)
            att_cache: shape (n_time, depth, b, nh, t, c * 2)
        """
        # get offset from att_cache
        offset = att_cache.shape[4] if att_cache is not None else 0
        z = self.rand_noise[:, :, offset : offset + mu.size(2)] * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        # cosine scheduling
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        x, new_cnn_cache, new_att_cache = self.solve_euler_chunk(
            x=z,
            t_span=t_span,
            mu=mu,
            spks=spks,
            cond=cond,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
        )
        return x, new_cnn_cache, new_att_cache
