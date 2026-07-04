import torch
import torch.nn as nn

from vllm_omni.diffusion.models.soulx_singer.modules.flow_matching import FlowMatchingTransformer


class CFMDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = FlowMatchingTransformer(cfg=config, **config)

    def forward(self, mel, x_mask, decoder_inp, is_prompt):
        outputs = self.model(mel, x_mask, decoder_inp, is_prompt)

        noise, x, flow_pred, final_mask, prompt_len = outputs["output"]
        return noise, x, flow_pred, final_mask, prompt_len

    def reverse_diffusion(self, pt_mel, pt_decoder_inp, gt_decoder_inp, n_timesteps=32, cfg=1):
        diffusion_cond = torch.cat([pt_decoder_inp, gt_decoder_inp], dim=1)
        diffusion_cond_emb = self.model.cond_emb(diffusion_cond)
        diffusion_prompt = pt_mel

        generated = self.model.reverse_diffusion(diffusion_cond_emb, diffusion_prompt, n_timesteps=n_timesteps, cfg=cfg)
        return generated
