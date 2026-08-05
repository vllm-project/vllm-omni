# https://github.com/open-mmlab/Amphion/blob/main/models/svc/flow_matching_transformer/fmt_model.py

# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig

from vllm_omni.diffusion.models.soulx_singer.modules.llama import DiffLlama


class FlowMatchingTransformer(nn.Module):
    def __init__(
        self,
        mel_dim=100,
        hidden_size=1024,
        num_layers=12,
        num_heads=16,
        cfg_drop_prob=0.2,
        use_embedding=True,
        cond_codebook_size=1024,
        cond_scale_factor=1,
        sigma=1e-5,
        time_scheduler="linear",
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg

        if cfg is not None:
            mel_dim = getattr(cfg, "mel_dim", mel_dim)
            hidden_size = getattr(cfg, "hidden_size", hidden_size)
            num_layers = getattr(cfg, "num_layers", num_layers)
            num_heads = getattr(cfg, "num_heads", num_heads)
            cfg_drop_prob = getattr(cfg, "cfg_drop_prob", cfg_drop_prob)
            cond_codebook_size = getattr(cfg, "cond_codebook_size", cond_codebook_size)
            time_scheduler = getattr(cfg, "time_scheduler", time_scheduler)
            sigma = getattr(cfg, "sigma", sigma)
            cond_scale_factor = getattr(cfg, "cond_scale_factor", cond_scale_factor)

        self.mel_dim = mel_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg_drop_prob = cfg_drop_prob
        self.cond_codebook_size = cond_codebook_size
        self.time_scheduler = time_scheduler
        self.sigma = sigma
        self.cond_scale_factor = cond_scale_factor

        if use_embedding:
            self.cond_emb = nn.Embedding(cond_codebook_size, self.hidden_size)
        else:
            self.cond_emb = nn.Linear(cond_codebook_size, self.hidden_size)

        if cond_scale_factor != 1:
            self.do_resampling = True
            assert np.log2(cond_scale_factor).is_integer()

            up_layers = []
            for _ in range(int(np.log2(cond_scale_factor))):
                up_layers.extend(
                    [
                        nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1),
                        nn.GELU(),
                    ]
                )
            self.resampling_layers = nn.Sequential(*up_layers)
        else:
            self.do_resampling = False

        ### REPA: Use the Wav2Vec2Bert features to align. ###
        self.use_repa = "repa" in cfg
        self.repa_layer_index = None
        if self.use_repa:
            self.repa_layer_index = cfg.repa.layer_index

            self.repa_mlp_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, cfg.repa.output_dim),
            )

        ### CTC: Use the ASR loss ###
        self.use_ctc = "ctc" in cfg
        self.ctc_layer_index = None
        if self.use_ctc:
            self.ctc_layer_index = cfg.ctc.layer_index

            self.ctc_mlp_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, cfg.ctc.output_dim),
            )

        self.reset_parameters()

        # DiffLlama config must be overridden manually to make sure the
        # internal RoPE head dimension is correct
        llama_config = LlamaConfig(
            vocab_size=0,
            bos_token_id=None,
            eos_token_id=None,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            rms_norm_eps=1e-6,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        self.diff_estimator = DiffLlama(
            mel_dim=mel_dim, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers, config=llama_config
        )

        self.sigma = sigma

    @torch.no_grad()
    def forward_diffusion(self, x, t, is_prompt=None):
        """
        x: (B, T, mel_dim)
        t: (B,)
        """
        new_t = t
        t = t.unsqueeze(-1).unsqueeze(-1)
        z = torch.randn(x.shape, dtype=x.dtype, device=x.device, requires_grad=False)  # (B, T, mel_dim)

        # get prompt len
        if torch.rand(1) <= self.cfg_drop_prob:
            prompt_len = torch.zeros(x.shape[0]).to(x)
            is_prompt = torch.zeros_like(x[:, :, 0])
        else:
            if is_prompt is None:
                prompt_len = torch.randint(min(x.shape[1] // 4, 5), int(x.shape[1] * 0.4), (x.shape[0],)).to(
                    x.device
                )  # (B,)

                # get is_prompt
                is_prompt = torch.zeros_like(x[:, :, 0])  # (B, T)
                col_indices = torch.arange(is_prompt.shape[1]).repeat(is_prompt.shape[0], 1).to(prompt_len)  # (B, T)
                is_prompt[col_indices < prompt_len.unsqueeze(1)] = 1  # (B, T) 1 if prompt
            else:
                prompt_len = is_prompt.sum(dim=1)  # (B,)

        mask = torch.ones_like(x[:, :, 0])  # mask if 1, not mask if 0
        mask[is_prompt.bool()] = 0
        mask = mask[:, :, None]

        # flow matching: xt = (1 - (1 - sigma) * t) * x0 + t * x; where x0 ~ N(0, 1), x is a sample
        # flow gt: x - (1 - sigma) * x0 = x - (1 - sigma) * noise
        xt = ((1 - (1 - self.sigma) * t) * z + t * x) * mask + x * (1 - mask)

        return xt, z, new_t, prompt_len, mask

    def loss_t(self, x, x_mask, t, cond=None, is_prompt=None):
        xt, z, new_t, prompt_len, mask = self.forward_diffusion(x, t, is_prompt)

        noise = z

        # drop all condition for cfg, so if prompt_len is 0, we also drop cond
        if cond is not None:
            cond = cond * torch.where(
                prompt_len > 0,
                torch.ones_like(prompt_len),
                torch.zeros_like(prompt_len),
            ).to(cond.device).unsqueeze(-1).unsqueeze(-1)

        dit_output = self.diff_estimator(xt, new_t, cond, x_mask, return_dict=True)
        flow_pred = dit_output["output"]  # (B, T, mel_dim)

        # final mask used for loss calculation
        final_mask = mask * x_mask[..., None]  # (B, T, 1)

        results = {"output": (noise, x, flow_pred, final_mask, prompt_len)}

        if self.use_repa:
            repa_hidden_states = dit_output["hidden_states"][self.repa_layer_index]  # (B, T, hidden_size)

            repa_pred = self.repa_mlp_layer(repa_hidden_states)  # (B, T, repa_dim)
            results["repa"] = repa_pred

        if self.use_ctc:
            ctc_hidden_states = dit_output["hidden_states"][self.ctc_layer_index]  # (B, T, hidden_size)
            ctc_pred = self.ctc_mlp_layer(ctc_hidden_states)  # (B, T, ctc_dim)
            results["ctc"] = ctc_pred

        return results

    def compute_loss(self, x, x_mask, cond=None, is_prompt=None):
        # x0: (B, T, num_quantizer)
        # x_mask: (B, T) mask is 0 for padding
        t = torch.rand(x.shape[0], device=x.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)
        # generation is harder at the start than later; use a cosine
        # scheduler for timestep t when time_scheduler == "cos".
        if self.time_scheduler == "cos":
            t = 1 - torch.cos(t * math.pi * 0.5)
        else:
            pass
        return self.loss_t(x, x_mask, t, cond, is_prompt)

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)

    @torch.no_grad()
    def reverse_diffusion(
        self,
        cond,
        prompt,
        x_mask=None,
        prompt_mask=None,
        n_timesteps=10,
        cfg=1.0,
        rescale_cfg=0.75,
    ):
        h = 1.0 / n_timesteps
        prompt_len = prompt.shape[1]
        target_len = cond.shape[1] - prompt_len
        trunk_dtype = cond.dtype

        if x_mask is None:
            x_mask = torch.ones(cond.shape[0], target_len, dtype=trunk_dtype, device=cond.device)
        if prompt_mask is None:
            prompt_mask = torch.ones(cond.shape[0], prompt_len, dtype=trunk_dtype, device=cond.device)
        xt_mask = torch.cat([prompt_mask, x_mask], dim=1)
        z = torch.randn(
            (cond.shape[0], target_len, self.mel_dim),
            dtype=trunk_dtype,
            device=cond.device,
            requires_grad=False,
        )
        xt = z
        h_tensor = torch.tensor(h, dtype=trunk_dtype, device=cond.device)

        # t from 0 to 1: x0 = z ~ N(0, 1)
        for i in range(n_timesteps):
            xt_input = torch.cat([prompt, xt], dim=1)
            t = torch.full(
                (z.shape[0],),
                fill_value=(i + 0.5) * h,
                dtype=trunk_dtype,
                device=z.device,
            )
            flow_pred = self.diff_estimator(xt_input, t, cond, xt_mask)
            flow_pred = flow_pred[:, prompt_len:, :]

            # cfg
            if cfg > 0:
                uncond_flow_pred = self.diff_estimator(xt, t, torch.zeros_like(cond)[:, : xt.shape[1], :], x_mask)
                pos_flow_pred_std = flow_pred.std()
                flow_pred_cfg = flow_pred + cfg * (flow_pred - uncond_flow_pred)
                rescale_flow_pred = flow_pred_cfg * pos_flow_pred_std / flow_pred_cfg.std()
                flow_pred = rescale_cfg * rescale_flow_pred + (1 - rescale_cfg) * flow_pred_cfg

            xt = xt + flow_pred * h_tensor

        return xt

    @torch.no_grad()
    def reverse_diffusion_v2(
        self,
        cond,
        prompt,
        x_mask=None,
        prompt_mask=None,
        n_timesteps=10,
        cfg=1.0,
        rescale_cfg=0.75,
    ):
        h = 1.0 / n_timesteps
        prompt_len = prompt.shape[1]
        target_len = cond.shape[1] - prompt_len * 2

        if x_mask is None:
            x_mask = torch.ones(cond.shape[0], target_len).to(cond.device)  # (B, T)
        if prompt_mask is None:
            prompt_mask = torch.ones(cond.shape[0], prompt_len).to(cond.device)  # (B, prompt_len)
        xt_mask = torch.cat([prompt_mask, x_mask, prompt_mask], dim=1)
        z = torch.randn(
            (cond.shape[0], target_len, self.mel_dim),
            dtype=cond.dtype,
            device=cond.device,
            requires_grad=False,
        )
        xt = z

        # t from 0 to 1: x0 = z ~ N(0, 1)
        for i in range(n_timesteps):
            xt_input = torch.cat([prompt, xt, prompt], dim=1)
            t = (0 + (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            flow_pred = self.diff_estimator(xt_input, t, cond, xt_mask)
            flow_pred = flow_pred[:, prompt_len:-prompt_len, :]

            # cfg
            if cfg > 0:
                uncond_flow_pred = self.diff_estimator(xt, t, torch.zeros_like(cond)[:, : xt.shape[1], :], x_mask)
                pos_flow_pred_std = flow_pred.std()
                flow_pred_cfg = flow_pred + cfg * (flow_pred - uncond_flow_pred)
                rescale_flow_pred = flow_pred_cfg * pos_flow_pred_std / flow_pred_cfg.std()
                flow_pred = rescale_cfg * rescale_flow_pred + (1 - rescale_cfg) * flow_pred_cfg

            dxt = flow_pred * h
            xt = xt + dxt

        return xt

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        cond_code: torch.Tensor,
        is_prompt: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (B, T, mel_dim)
            x_mask: (B, T)
            cond_code: (B, T), Note that cond_code might be not at 50Hz!
        """
        T = x.shape[1]

        cond = self.cond_emb(cond_code)  # (B, T, hidden_size)
        if self.do_resampling:
            # Align to the frame rate of Mels
            cond = self.resampling_layers(cond.transpose(1, 2)).transpose(1, 2)

        # print("cond_code: {}, after resampling: {}".format(cond_code.shape, cond.shape))

        if cond.shape[1] >= T:  # Check time dimension
            cond = cond[:, :T, :]
        else:
            padding_frames = T - cond.shape[1]
            last_frame = cond[:, -1:, :]
            padding = last_frame.repeat(1, padding_frames, 1)
            cond = torch.cat([cond, padding], dim=1)

        return self.compute_loss(x, x_mask, cond, is_prompt)


if __name__ == "__main__":
    model_cfg = {
        "mel_dim": 128,
        "hidden_size": 256,
        "num_layers": 8,
        "num_heads": 8,
        "cfg_drop_prob": 0.2,
        "use_embedding": False,
        "cond_codebook_size": 256,
        "cond_scale_factor": 1,
        "sigma": 1e-5,
        "time_scheduler": "cos",
    }

    device = "cuda"
    x = torch.randn(2, 100, 128).to(device)
    x_mask = torch.ones(2, 100).to(device)
    # cond_code = torch.randint(0, 16384, (2, 25)).to(device)
    cond_code = torch.randn(2, 100, 256).to(device)

    model = FlowMatchingTransformer(cfg=model_cfg, **model_cfg).to(device)
    outputs = model(x, x_mask, cond_code)
    print(outputs)

    noise, x, flow_pred, final_mask, prompt_len = outputs["output"]
    final_mask = final_mask.squeeze(-1)

    flow_gt = x - (1 - 1e-5) * noise

    # [B, n_frames, D]
    diff_loss = F.l1_loss(flow_pred, flow_gt, reduction="none").float() * final_mask.unsqueeze(-1)
    diff_loss = torch.mean(diff_loss, dim=2).sum() / final_mask.sum()

    print("diff_loss:", diff_loss.item())

    diffusion_cond = torch.randn(2, 150, 256).to(device)
    diffusion_cond_emb = model.cond_emb(diffusion_cond)
    diffusion_prompt = torch.randn(2, 50, 128).to(device)
    n_timesteps = 32

    generated = model.reverse_diffusion(diffusion_cond_emb, diffusion_prompt, n_timesteps=n_timesteps)
    print("generated:", generated.shape)
