import logging
import math
import os
import random
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from einops import rearrange
from torch import nn
from tqdm import tqdm

# Load dependencies from Lingbot World source code
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.utils.cam_utils import (
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportCameraPosInput, SupportImageInput
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .state_lingbot_world_fast import LingbotWorldFastState
from .wan_fast import WanModelFast

logger = logging.getLogger(__name__)

CONFIG = {
    "text_len": 512,
    "num_train_timesteps": 1000,
    "vae_stride": (4, 8, 8),
    "patch_size": (1, 2, 2),
    "timesteps_index": [0, 179, 358, 679],
    "sample_shift": 10.0,
    "max_area": 480 * 832,
    "max_sequence_length": 512,
    "chunk_size": 3,
    "t5_checkpoint": "models_t5_umt5-xxl-enc-bf16.pth",
    "t5_tokenizer": "google/umt5-xxl",
    "vae_checkpoint": "Wan2.1_VAE.pth",
    "fast_noise_checkpoint": "Lingbot-World-Fast",
    "negative_prompt_sample": (
        "画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
        "三条腿，背景人很多，倒着走，镜头晃动，画面闪烁，模糊，噪点，水印，签名，文字，变形，"
        "扭曲，液化，不合逻辑的结构，卡顿，PPT幻灯片感，过暗，欠曝，低对比度，霓虹灯光感，"
        "过度锐化，3D渲染感，人物，行人，游客，身体，皮肤，肢体，面部特征，汽车，电线"
    ),
}


def get_lingbot_world_fast_post_process_func(
    od_config: OmniDiffusionConfig,
):
    def post_process_func(
        video: torch.Tensor,
    ):
        outputs = video.permute(1, 2, 3, 0)
        return outputs

    return post_process_func


class LingbotWorldFastPipeline(nn.Module, SupportImageInput, SupportCameraPosInput, CFGParallelMixin):
    def __init__(self, *, od_config: OmniDiffusionConfig):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config

        self.device = get_local_device()

        self.target_dtype = od_config.dtype

        self.control_type = "cam"
        self.num_train_timesteps = CONFIG["num_train_timesteps"]

        self.sp_size = od_config.parallel_config.world_size

        self.state = LingbotWorldFastState()

        checkpoint_path = os.path.dirname(self.od_config.model)
        assert checkpoint_path is not None, "lingbot_dir is None"

        self.text_encoder = T5EncoderModel(
            text_len=CONFIG["text_len"],
            dtype=self.target_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_path, CONFIG["t5_checkpoint"]),
            tokenizer_path=os.path.join(checkpoint_path, CONFIG["t5_tokenizer"]),
        )

        self.vae_stride = CONFIG["vae_stride"]
        self.patch_size = CONFIG["patch_size"]
        self.vae = Wan2_1_VAE(vae_pth=os.path.join(checkpoint_path, CONFIG["vae_checkpoint"]), device=self.device)

        logger.info(f"Creating WanModelFast from {checkpoint_path}")
        self.model = WanModelFast.from_pretrained(
            checkpoint_path,
            subfolder=CONFIG["fast_noise_checkpoint"],
            torch_dtype=torch.bfloat16,
            control_type=self.control_type,
        ).to(self.device)

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )

        self.sample_neg_prompt = CONFIG["negative_prompt_sample"]

    def _configure_model(self, model):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, F, H, W]
        xt: the input noisy data with shape [B, C, F, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred

        return x0_pred.to(original_dtype)

    def forward(
        self,
        req: OmniDiffusionRequest,
    ) -> DiffusionOutput:
        if len(req.prompts) > 1:
            raise ValueError(
                """This model only supports a single prompt, not a batched request.""",
                """Please pass in a single prompt object or string, or a single-item list.""",
            )
        prompt = req.prompts[0].get("prompt")
        multi_modal_data = req.prompts[0].get("multi_modal_data", {})

        # Always reset: Lingbot Fast does not support video continuation
        self.state.reset()

        camera = multi_modal_data.get("camera", None)
        if camera is None:
            self.od_config.model
            raise ValueError("A path to camera positions must be passed to this model through action_path.")

        batch_size = 1
        num_frames = req.sampling_params.num_frames
        # In order to generate something num_frames must be at least 5 since it expects 4*n + 1 as input
        # 25 is the smallest length supported by the model. Smaller values generate tensors with dimension zero/negative
        num_frames = max(25, num_frames)

        c2ws = camera.get("poses")

        len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        num_frames = min(num_frames, len_c2ws)
        c2ws = c2ws[:num_frames]

        # preprocess
        img = multi_modal_data.get("image")
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        max_area = CONFIG["max_area"]
        chunk_size = CONFIG["chunk_size"]

        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (num_frames - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        lat_f = max(lat_f, 1)
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        seed = random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, shift=CONFIG["sample_shift"])
        timesteps = self.scheduler.timesteps[CONFIG["timesteps_index"]]

        context = self.text_encoder([prompt], torch.device("cpu"))
        context = [t.to(self.device) for t in context]

        dit_cond_dict = None
        Ks = torch.from_numpy(camera.get("intrinsics"))

        # Transform the provided intrinsics from the original 480p according to the new image size (h, w).
        Ks = get_Ks_transformed(
            Ks, height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w
        )
        Ks = Ks[0]

        len_c2ws = len(c2ws)
        len_c2ws_ = int((len_c2ws - 1) // 4) + 1
        len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1)

        c2ws_infer = c2ws_infer.to(self.device).to(torch.float32)
        Ks = Ks.to(self.device).to(torch.float32)
        only_rays_d = False
        c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w, only_rays_d=only_rays_d)
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(h // lat_h),
            c2=int(w // lat_w),
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...]  # [b, f*h*w, c]
        c2ws_plucker_emb = rearrange(c2ws_plucker_emb, "b (f h w) c -> b c f h w", f=lat_f, h=lat_h, w=lat_w).to(
            self.target_dtype
        )

        y = self.vae.encode(
            [
                torch.concat(
                    [
                        torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                        torch.zeros(3, F - 1, h, w),
                    ],
                    dim=1,
                ).to(self.device)
            ]
        )[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, "no_sync", noop_no_sync)

        # Initialize KV cache to all zeros
        model_args = self.model.config
        transformer_dtype = self.target_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1] // 4)
        kv_size = frame_seqlen * lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size

        self.state.create_kv_caches(
            batch_size, transformer_dtype, self.device, kv_size, model_args.num_layers, local_num_heads, head_dim
        )

        # evaluation mode
        with (
            torch.amp.autocast("cuda", dtype=self.target_dtype),
            torch.no_grad(),
            no_sync_model(),
        ):
            # sample videos
            latent = noise
            latents_chunk = latent.split(chunk_size, dim=1)  # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []
            for chunk_id in tqdm(range(num_inference_chunk)):
                current_latent = latents_chunk[chunk_id]
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]

                dit_cond_dict = {
                    "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
                }

                kwargs = {
                    "context": [context[0]],
                    "seq_len": max_seq_len,
                    "y": [current_condition],
                    "dit_cond_dict": dit_cond_dict,
                    "kv_cache": self.state.get_kv_caches(),
                    "local_end_index": self.state.local_end_index,
                    "global_end_index": self.state.global_end_index,
                    "crossattn_cache": self.state.get_crossattn_caches(),
                    "current_start": chunk_id * chunk_size * frame_seqlen,
                    "max_attention_size": kv_size,
                }

                for timestep_idx in range(len(timesteps)):
                    latent_model_input = [current_latent.to(self.device)]
                    current_timestep = [timesteps[timestep_idx]]

                    timestep = torch.stack(current_timestep).to(self.device)

                    noise_pred = self.model(x=latent_model_input, t=timestep, **kwargs)[0]

                    x0 = self._convert_flow_pred_to_x0(
                        flow_pred=noise_pred,
                        xt=current_latent,
                        timestep=current_timestep[0],
                        scheduler=self.scheduler,
                    )

                    if timestep_idx < len(timesteps) - 1:
                        next_timestep = timesteps[timestep_idx + 1]
                        current_latent = self.scheduler.add_noise(
                            x0, torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype), next_timestep
                        )
                    else:
                        # note return x0
                        break

                pred_latent_chunks.append(x0)

                # Update kv cache
                context_timestep = [timesteps[-1] * 0.0]
                timestep = torch.stack(context_timestep).to(self.device)
                self.model(x=[x0], t=timestep, **kwargs)

            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)

            if self.device.index == 0:
                videos = self.vae.decode([pred_latent_chunks])

        if dist.is_initialized():
            dist.barrier()

        return DiffusionOutput(output=videos[0])

    def load_weights(self, weights):
        pass
