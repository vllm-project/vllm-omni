import logging
import os
from collections.abc import Iterable

import torch
from diffusers import AutoencoderKLWan
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .causvid import WanModel
from .flow_match import FlowMatchScheduler
from .state_causvid import CausVidState

logger = logging.getLogger(__name__)

CONFIG = {
    "autoregressive_checkpoint": "causvid/autoregressive_checkpoint/model.pt",
    "t5_checkpoint": "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    "t5_tokenizer": "wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl",
    "vae_checkpoint": "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    "denoising_step_list": [1000, 757, 522, 0],
    "num_frame_per_block": 3,
}

T5_CONFIG = UMT5Config(
    vocab_size=256384,
    d_model=4096,
    d_ff=10240,
    num_heads=64,
    num_layers=24,
    dropout_rate=0.1,
)


def get_causvid_post_process_func(
    od_config: OmniDiffusionConfig,
):
    def post_process_func(
        video: torch.Tensor,
    ):
        video = video.permute(1, 0, 2, 3)
        return video

    return post_process_func


class CausVidPipeline(torch.nn.Module):
    def __init__(self, *, od_config: OmniDiffusionConfig):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.target_dtype = od_config.dtype

        self.model_path = self.od_config.model

        self.state = CausVidState()

        # Step 1: Initialize all models
        self.model = WanModel(
            in_dim=od_config.tf_model_config.get("in_dim"),
            dim=od_config.tf_model_config.get("dim"),
            ffn_dim=od_config.tf_model_config.get("ffn_dim"),
            num_layers=od_config.tf_model_config.get("num_layers"),
            num_heads=od_config.tf_model_config.get("num_heads"),
        )

        tokenizer_path = os.path.join(self.model_path, CONFIG["t5_tokenizer"])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        t5_checkpoint = os.path.join(self.model_path, CONFIG["t5_checkpoint"])
        wan_sd = torch.load(t5_checkpoint, map_location="cpu", weights_only=True)
        with torch.device("meta"):
            self.text_encoder = UMT5EncoderModel(T5_CONFIG)
        # The checkpoint is in Wan's naming, not HF's — remap before loading.
        self.text_encoder.load_state_dict(_wan_t5_to_hf_state_dict(wan_sd), assign=True)
        self.text_encoder = self.text_encoder.to(device=self.device, dtype=self.target_dtype).eval()

        vae_path = os.path.join(self.model_path, CONFIG["vae_checkpoint"])
        vae_sd = torch.load(vae_path, map_location="cpu", weights_only=True)
        self.vae = AutoencoderKLWan()
        # The checkpoint is in Wan's naming, not diffusers' — remap before loading.
        self.vae.load_state_dict(_wan_vae_to_diffusers_state_dict(vae_sd), assign=True)
        self.vae = self.vae.to(self.device)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.vae.device, self.vae.dtype
        )
        self.latents_scale = [latents_mean, latents_std]

        # Step 2: Initialize all causal hyperparmeters
        self.denoising_step_list = torch.tensor(CONFIG["denoising_step_list"], dtype=torch.long, device=self.device)
        assert self.denoising_step_list[-1] == 0
        # remove the last timestep (which equals zero)
        self.denoising_step_list = self.denoising_step_list[:-1]

        self.scheduler = FlowMatchScheduler(shift=8.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.num_frame_per_block = CONFIG["num_frame_per_block"]

        logger.info(f"KV inference with {self.num_frame_per_block} frames per block")

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        if len(req.prompts) > 1:
            raise ValueError(
                """This model only supports a single prompt, not a batched request.""",
                """Please pass in a single prompt object or string, or a single-item list.""",
            )

        self.state.reset()

        prompt = req.prompts[0].get("prompt")
        num_frames = req.sampling_params.num_frames

        # The formula for the VAE is pixel_frames = (num_latent_frames - 1) * 4 + 1
        num_latent_frames = (num_frames + 3 + 3) // 4

        # Round up the number of frames to match a multiple of the block size
        num_blocks = (num_latent_frames + self.num_frame_per_block - 1) // self.num_frame_per_block
        num_latent_frames = num_blocks * self.num_frame_per_block

        batch_size = 1

        num_layers = self.model.num_layers
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads

        # Step 1: initialize kv cache
        self.state.create_kv_caches(
            batch_size,
            self.target_dtype,
            self.device,
            self.frame_seq_length * num_latent_frames,
            num_layers,
            num_heads,
            head_dim,
        )

        noise = torch.randn([16, num_latent_frames, 60, 104], device="cuda", dtype=torch.bfloat16)
        num_channels, _, height, width = noise.shape
        prompt_emb = self.encode_prompt([prompt], self.device)

        output = torch.zeros([num_channels, num_latent_frames, height, width], device=noise.device, dtype=noise.dtype)

        # Step 2: Temporal denoising loop
        for block_index in tqdm(range(num_blocks)):
            noisy_input = noise[
                :, block_index * self.num_frame_per_block : (block_index + 1) * self.num_frame_per_block
            ]

            # Step 2.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = (
                    torch.ones([batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64)
                    * current_timestep
                )

                flow_pred = self.model(
                    x=[noisy_input],
                    context=prompt_emb,
                    t=timestep,
                    kv_cache=self.state.get_kv_cache(),
                    crossattn_cache=self.state.get_crossattn_cache(),
                    current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                    current_end=(block_index + 1) * self.num_frame_per_block * self.frame_seq_length,
                )

                flow_pred = torch.cat(flow_pred)

                x0 = self._convert_flow_pred_to_x0(
                    flow_pred,
                    noisy_input,
                    current_timestep,
                )

                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        x0,
                        torch.randn_like(x0),
                        next_timestep * torch.ones([batch_size], device=noise.device, dtype=torch.long),
                    )

            # Step 2.2: rerun with timestep zero to update the cache
            output[:, block_index * self.num_frame_per_block : (block_index + 1) * self.num_frame_per_block] = x0

            self.model(
                x=[x0],
                context=prompt_emb,
                t=timestep * 0,
                crossattn_cache=self.state.get_crossattn_cache(),
                kv_cache=self.state.get_kv_cache(),
                current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                current_end=(block_index + 1) * self.num_frame_per_block * self.frame_seq_length,
            )

        # Step 3: Decode the output
        video = self.decode_video([output])

        return DiffusionOutput(output=video[0])

    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
        sigmas = self.scheduler.sigmas.to(flow_pred.device)
        timesteps = self.scheduler.timesteps.to(flow_pred.device)
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return xt - sigma_t * flow_pred

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        state_dict = torch.load(
            os.path.join(self.model_path, CONFIG["autoregressive_checkpoint"]), map_location=self.device
        )["generator"]

        state_dict = {key.removeprefix("model."): value for (key, value) in state_dict.items()}
        self.model.load_state_dict(state_dict)

    def encode_prompt(self, texts: list[str], device: torch.device):
        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        ids = inputs.input_ids.to(device)
        mask = inputs.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return [u[:v] for u, v in zip(context, seq_lens)]

    def encode_video(self, videos: list[torch.Tensor]):
        mean, inv_std = self.latents_scale  # [latents_mean, 1/std], shapes (1, z, 1, 1, 1)
        out = []
        for u in videos:
            # AutoencoderKLWan.encode -> AutoencoderKLOutput; the custom Wan2_1_VAE.encode
            # returns the deterministic posterior mean (mu, no sampling), so use .mode().
            mu = self.vae.encode(u.unsqueeze(0)).latent_dist.mode()
            mu = (mu - mean) * inv_std
            out.append(mu.float().squeeze(0))
        return out

    def decode_video(self, zs: list[torch.Tensor]):
        mean, inv_std = self.latents_scale
        out = []
        for u in zs:
            z = u.unsqueeze(0) / inv_std + mean
            sample = self.vae.decode(z, return_dict=False)[0]
            out.append(sample.float().clamp_(-1, 1).squeeze(0))
        return out


def _wan_t5_to_hf_state_dict(sd: dict) -> dict:
    """Remap the Wan ``models_t5_umt5-xxl-enc-bf16.pth`` checkpoint (saved from the
    custom ``T5Encoder`` in ``t5.py``) to the key naming expected by HF's
    ``UMT5EncoderModel``. Same weights, different module names."""
    block_map = {
        "norm1.weight": "layer.0.layer_norm.weight",
        "attn.q.weight": "layer.0.SelfAttention.q.weight",
        "attn.k.weight": "layer.0.SelfAttention.k.weight",
        "attn.v.weight": "layer.0.SelfAttention.v.weight",
        "attn.o.weight": "layer.0.SelfAttention.o.weight",
        "pos_embedding.embedding.weight": "layer.0.SelfAttention.relative_attention_bias.weight",
        "norm2.weight": "layer.1.layer_norm.weight",
        "ffn.gate.0.weight": "layer.1.DenseReluDense.wi_0.weight",
        "ffn.fc1.weight": "layer.1.DenseReluDense.wi_1.weight",
        "ffn.fc2.weight": "layer.1.DenseReluDense.wo.weight",
    }
    out = {}
    for k, v in sd.items():
        if k == "token_embedding.weight":
            out["shared.weight"] = v
            out["encoder.embed_tokens.weight"] = v
        elif k == "norm.weight":
            out["encoder.final_layer_norm.weight"] = v
        elif k.startswith("blocks."):
            _, idx, rest = k.split(".", 2)
            mapped = block_map.get(rest)
            if mapped is None:
                raise KeyError(f"Unmapped Wan T5 key: {k}")
            out[f"encoder.block.{idx}.{mapped}"] = v
        else:
            raise KeyError(f"Unmapped Wan T5 key: {k}")
    return out


def _wan_vae_to_diffusers_state_dict(sd: dict) -> dict:
    """Remap the original ``Wan2.1_VAE.pth`` checkpoint (custom ``Wan2_1_VAE`` naming)
    to the key naming expected by diffusers' ``AutoencoderKLWan``. Same weights,
    different module names. Ported from diffusers'
    ``scripts/convert_wan_to_diffusers.py::convert_vae``."""
    middle_key_mapping = {}
    for enc_dec in ("encoder", "decoder"):
        for src_idx, dst_idx in ((0, 0), (2, 1)):
            base = f"{enc_dec}.mid_block.resnets.{dst_idx}"
            middle_key_mapping.update(
                {
                    f"{enc_dec}.middle.{src_idx}.residual.0.gamma": f"{base}.norm1.gamma",
                    f"{enc_dec}.middle.{src_idx}.residual.2.bias": f"{base}.conv1.bias",
                    f"{enc_dec}.middle.{src_idx}.residual.2.weight": f"{base}.conv1.weight",
                    f"{enc_dec}.middle.{src_idx}.residual.3.gamma": f"{base}.norm2.gamma",
                    f"{enc_dec}.middle.{src_idx}.residual.6.bias": f"{base}.conv2.bias",
                    f"{enc_dec}.middle.{src_idx}.residual.6.weight": f"{base}.conv2.weight",
                }
            )

    attention_mapping = {}
    head_mapping = {}
    for enc_dec in ("encoder", "decoder"):
        attention_mapping.update(
            {
                f"{enc_dec}.middle.1.norm.gamma": f"{enc_dec}.mid_block.attentions.0.norm.gamma",
                f"{enc_dec}.middle.1.to_qkv.weight": f"{enc_dec}.mid_block.attentions.0.to_qkv.weight",
                f"{enc_dec}.middle.1.to_qkv.bias": f"{enc_dec}.mid_block.attentions.0.to_qkv.bias",
                f"{enc_dec}.middle.1.proj.weight": f"{enc_dec}.mid_block.attentions.0.proj.weight",
                f"{enc_dec}.middle.1.proj.bias": f"{enc_dec}.mid_block.attentions.0.proj.bias",
            }
        )
        head_mapping.update(
            {
                f"{enc_dec}.head.0.gamma": f"{enc_dec}.norm_out.gamma",
                f"{enc_dec}.head.2.bias": f"{enc_dec}.conv_out.bias",
                f"{enc_dec}.head.2.weight": f"{enc_dec}.conv_out.weight",
            }
        )

    quant_mapping = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }

    residual_renames = {
        ".residual.0.gamma": ".norm1.gamma",
        ".residual.2.bias": ".conv1.bias",
        ".residual.2.weight": ".conv1.weight",
        ".residual.3.gamma": ".norm2.gamma",
        ".residual.6.bias": ".conv2.bias",
        ".residual.6.weight": ".conv2.weight",
    }

    out = {}
    for key, value in sd.items():
        if key in middle_key_mapping:
            out[middle_key_mapping[key]] = value
        elif key in attention_mapping:
            out[attention_mapping[key]] = value
        elif key in head_mapping:
            out[head_mapping[key]] = value
        elif key in quant_mapping:
            out[quant_mapping[key]] = value
        elif key in ("encoder.conv1.weight", "encoder.conv1.bias", "decoder.conv1.weight", "decoder.conv1.bias"):
            out[key.replace(".conv1.", ".conv_in.")] = value
        elif key.startswith("encoder.downsamples."):
            new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.")
            for src, dst in residual_renames.items():
                if src in new_key:
                    new_key = new_key.replace(src, dst)
                    break
            else:
                new_key = new_key.replace(".shortcut.", ".conv_shortcut.")
            out[new_key] = value
        elif key.startswith("decoder.upsamples."):
            block_idx = int(key.split(".")[2])
            if "residual" in key:
                grouping = {
                    **dict.fromkeys((0, 1, 2), 0),
                    **dict.fromkeys((4, 5, 6), 1),
                    **dict.fromkeys((8, 9, 10), 2),
                    **dict.fromkeys((12, 13, 14), 3),
                }
                if block_idx not in grouping:
                    out[key] = value
                    continue
                new_block_idx = grouping[block_idx]
                resnet_idx = block_idx - [0, 4, 8, 12][new_block_idx]
                new_key = key
                for src, dst in residual_renames.items():
                    if src in key:
                        new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}{dst}"
                        break
                out[new_key] = value
            elif ".shortcut." in key:
                if block_idx == 4:
                    new_key = key.replace(".shortcut.", ".resnets.0.conv_shortcut.")
                    new_key = new_key.replace("decoder.upsamples.4", "decoder.up_blocks.1")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                    new_key = new_key.replace(".shortcut.", ".conv_shortcut.")
                out[new_key] = value
            elif ".resample." in key or ".time_conv." in key:
                upsampler = {3: 0, 7: 1, 11: 2}.get(block_idx)
                if upsampler is not None:
                    new_key = key.replace(
                        f"decoder.upsamples.{block_idx}", f"decoder.up_blocks.{upsampler}.upsamplers.0"
                    )
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                out[new_key] = value
            else:
                out[key.replace("decoder.upsamples.", "decoder.up_blocks.")] = value
        else:
            out[key] = value
    return out
