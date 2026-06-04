import logging
import math
import os
import random
import sys
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from einops import rearrange
from torch import nn
from tqdm import tqdm
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_pipeline_parallel_world_size,
    get_pp_group,
    is_pipeline_first_stage,
)
from vllm_omni.diffusion.distributed.pipeline_parallel import (
    AsyncLatents,
    PipelineParallelMixin,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportCameraPosInput, SupportImageInput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

from .cam_utils import (
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from .flow_scheduler import LingbotFlowScheduler
from .fm_solvers_unipc import FlowUniPCMultistepScheduler
from .state_lingbot_world_fast import LingbotWorldFastState
from .stream_vae import StreamVAE
from .t5 import T5EncoderModel
from .vae2_1 import Wan2_1_VAE
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


class LingbotWorldFastPipeline(
    nn.Module,
    SupportImageInput,
    SupportCameraPosInput,
    PipelineParallelMixin,
    CFGParallelMixin,
):
    supports_step_execution: ClassVar[bool] = True
    supports_micro_step_execution: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config

        self.device = get_local_device()

        self.target_dtype = od_config.dtype

        self.control_type = "cam"
        self.num_train_timesteps = CONFIG["num_train_timesteps"]

        self.sp_size = od_config.parallel_config.sequence_parallel_size

        self.state = LingbotWorldFastState()

        checkpoint_path = os.path.dirname(self.od_config.model)
        assert checkpoint_path is not None, "lingbot_dir is None"

        self.text_encoder = T5EncoderModel(
            text_len=CONFIG["text_len"],
            dtype=self.target_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_path, CONFIG["t5_checkpoint"]),
            tokenizer_path=os.path.join(checkpoint_path, CONFIG["t5_tokenizer"]),
        )

        self.vae_stride = CONFIG["vae_stride"]
        self.patch_size = CONFIG["patch_size"]
        base_vae = Wan2_1_VAE(vae_pth=os.path.join(checkpoint_path, CONFIG["vae_checkpoint"]), device=self.device)
        self.vae = StreamVAE(base_vae) if od_config.stream_batch else base_vae

        logger.info(f"Creating WanModelFast from {checkpoint_path}")
        self.model = WanModelFast.from_pretrained(
            checkpoint_path,
            subfolder=CONFIG["fast_noise_checkpoint"],
            torch_dtype=torch.bfloat16,
            control_type=self.control_type,
        ).to(self.device)
        # Partition transformer across PP ranks (no-op at PP=1).
        self.model.apply_pp_split()

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

        session_id = str(req.sampling_params.extra_args.get("session_id") or None)

        force_reset = req.sampling_params.extra_args.get("force_reset") or False

        extension = True

        if force_reset or self.state.session_id is None or self.state.session_id != session_id:
            self.state.reset()
            self.state.session_id = session_id
            extension = False
        else:
            extension = True

        camera = multi_modal_data.get("camera", None)
        if camera is None:
            self.od_config.model
            raise ValueError("A path to camera positions must be passed to this model through action_path.")

        if extension:
            assert multi_modal_data.get("image") is None, (
                "image must not be provided on extension calls; it is only used on the first call of a session"
            )
            assert self.model.config.local_attn_size == -1, (
                "video extension requires the model to be configured with local_attn_size == -1"
            )

        batch_size = 1
        num_frames = req.sampling_params.num_frames
        # In order to generate something num_frames must be at least 5 since it expects 4*n + 1 as input
        # 25 is the smallest length supported by the model. Smaller values generate tensors with dimension zero/negative
        num_frames = max(25, num_frames)

        c2ws = camera.get("poses")
        chunk_size = CONFIG["chunk_size"]
        max_area = CONFIG["max_area"]

        # Fresh:     4N+1 pixel frames → N+1 latents, the first slot is the anchor.
        # Extension: 4N   pixel frames → N regular latents, no anchor.
        if extension:
            len_c2ws = (len(c2ws) // 4) * 4
            num_frames = (num_frames // 4) * 4
            num_frames = min(num_frames, len_c2ws)
            new_lat_f = num_frames // 4
        else:
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            num_frames = ((num_frames - 1) // 4) * 4 + 1
            num_frames = min(num_frames, len_c2ws)
            new_lat_f = (num_frames - 1) // 4 + 1
        c2ws = c2ws[:num_frames]

        # 1. Derive spatial shape: from the input image on fresh start, from cache on extension.
        if not extension:
            img = multi_modal_data.get("image")
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
            h, w = img.shape[1:]
            aspect_ratio = h / w
            lat_h = round(
                np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
            )
            lat_w = round(
                np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
            )
            h = lat_h * self.vae_stride[1]
            w = lat_w * self.vae_stride[2]
        else:
            img = None
            h, w, lat_h, lat_w = self.state.h, self.state.w, self.state.lat_h, self.state.lat_w

        new_lat_f = int(new_lat_f - (new_lat_f % chunk_size))
        new_lat_f = max(new_lat_f, 1)
        max_seq_len = chunk_size * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        seed_g = req.sampling_params.generator
        if seed_g is None:
            seed = req.sampling_params.seed
            if seed is None:
                seed = random.randint(0, sys.maxsize)
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)
        noise = torch.randn(16, new_lat_f, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=self.device)

        # Fresh: msk[0] = 1 (anchor) and the rest = 0, replicated into 4 channels grouped
        # by latent frame to give shape [4, new_lat_f, lat_h, lat_w].
        # Extension: no anchor, all zeros, already in the [4, new_lat_f, ...] layout.
        if not extension:
            F = (new_lat_f - 1) * 4 + 1
            msk = torch.zeros(1, F, lat_h, lat_w, device=self.device)
            msk[:, 0] = 1
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
            msk = msk.transpose(1, 2)[0]
        else:
            msk = torch.zeros(4, new_lat_f, lat_h, lat_w, device=self.device)

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, shift=CONFIG["sample_shift"])
        timesteps = self.scheduler.timesteps[CONFIG["timesteps_index"]]

        context = self.text_encoder([prompt], self.device)

        dit_cond_dict = None
        Ks = torch.from_numpy(camera.get("intrinsics"))

        # Transform the provided intrinsics from the original 480p according to the new image size (h, w).
        Ks = get_Ks_transformed(
            Ks, height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w
        )
        Ks = Ks[0]

        # One target pose per output latent — must match the f= in the rearrange below.
        len_c2ws = len(c2ws)
        len_c2ws_ = new_lat_f
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
        c2ws_plucker_emb = rearrange(c2ws_plucker_emb, "b (f h w) c -> b c f h w", f=new_lat_f, h=lat_h, w=lat_w).to(
            self.target_dtype
        )

        # Fresh:     pixels = [anchor_image, zeros...] of shape [3, 4N+1, h, w].
        #            VAE produces N+1 latents; latent[0] is the anchor encoding.
        # Extension: pixels = zeros [3, 4N+1, h, w]. VAE produces N+1 latents,
        #            of which latent[0] is the special "1-frame init" encoding
        #            (biased differently than the regular 4-frame-group latents).
        #            Slice it off so the N conditioning slots are all regular —
        #            this drops a CONDITIONING slot, not an output latent.
        if not extension:
            F = (new_lat_f - 1) * 4 + 1
            pixels = torch.concat(
                [
                    torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                    torch.zeros(3, F - 1, h, w),
                ],
                dim=1,
            ).to(self.device)
            y = self.vae.encode([pixels])[0]
        else:
            pixels = torch.zeros(3, 4 * new_lat_f + 1, h, w, device=self.device)
            y = self.vae.encode([pixels])[0][:, 1:]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, "no_sync", noop_no_sync)

        # Initialize (fresh) or grow (extension) the KV cache. Cross-attn cache is
        # left untouched on extension so text-context k/v computed on the first call
        # are reused via crossattn_cache[i]["is_init"] == True.
        model_args = self.model.config
        transformer_dtype = self.target_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1] // 4)
        extra_kv_size = frame_seqlen * new_lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size

        if not extension:
            self.state.create_kv_caches(
                batch_size,
                transformer_dtype,
                self.device,
                extra_kv_size,
                model_args.num_layers,
                local_num_heads,
                head_dim,
            )
        else:
            self.state.extend_kv_caches(extra_kv_size)

        # Total cache size after this call, used both as the per-query attention
        # window and as the absolute-token offset base for the chunk loop.
        prev_lat_f = self.state.current_lat_f
        total_kv_size = frame_seqlen * (prev_lat_f + new_lat_f)
        start_token_offset = prev_lat_f * frame_seqlen

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
                    "current_starts": start_token_offset + chunk_id * chunk_size * frame_seqlen,
                    "max_attention_size": total_kv_size,
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
                # Wan VAE decode() calls clear_cache() internally, so the very
                # first latent always runs the i==0 path (no temporal upsample,
                # single-frame output) and leaves feat_map polluted with that
                # bias. The decoder's stacked temporal-causal layers also need
                # ~2 latents of streaming context before deeper feat_map slots
                # match a true mid-stream decode. On extension, prepend the
                # prior chunk's last 2 latents so warmup_0 absorbs the i==0
                # bias and warmup_1 fully primes the cache. Then discard the
                # 4*K - 3 leading pixels (re-decodes of already-shown frames).
                if extension and self.state.last_decoded_latent is not None:
                    warmup = self.state.last_decoded_latent.to(pred_latent_chunks.device, pred_latent_chunks.dtype)
                    k = warmup.shape[1]
                    drop = 4 * k - 3
                    to_decode = torch.cat([warmup, pred_latent_chunks], dim=1)
                    videos = self.vae.decode([to_decode])
                    videos = [v[:, drop:] for v in videos]
                else:
                    videos = self.vae.decode([pred_latent_chunks])

                self.state.last_decoded_latent = pred_latent_chunks[:, -2:].detach().clone()

        if dist.is_initialized():
            dist.barrier()

        if not extension:
            self.state.h = h
            self.state.w = w
            self.state.lat_h = lat_h
            self.state.lat_w = lat_w
            self.state.frame_seqlen = frame_seqlen
        self.state.advance(new_lat_f)

        return DiffusionOutput(output=videos[0])

    # ------------------------------------------------------------------
    # micro-step execution
    # ------------------------------------------------------------------

    def predict_noise(
        self,
        intermediate_tensors: IntermediateTensors | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Single transformer forward; returns IntermediateTensors on non-last PP stages."""
        with torch.amp.autocast("cuda", dtype=self.target_dtype):
            result = self.model(**kwargs, intermediate_tensors=intermediate_tensors)
        if isinstance(result, IntermediateTensors):
            return result
        # Last stage returns List[Tensor] (one per row); stack along dim 0.
        return torch.stack(result, dim=0)

    def prepare_encode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        """One-time request setup mirroring forward()'s prep up to the chunk loop.

        Stashes per-chunk noise / conditioning / Plucker tensors in state.extra,
        initializes (or extends) the model's persistent KV caches sized to this
        rank's owned layer slice, and exposes state.timesteps as length 5
        (4 denoise + 1 t=0 KV-update).
        """
        if not state.prompts or len(state.prompts) > 1:
            raise ValueError("LingbotWorldFastPipeline only supports a single prompt.")

        sampling = state.sampling

        prompt = state.prompts[0].get("prompt")
        multi_modal_data = state.prompts[0].get("multi_modal_data", {}) or {}

        extra_args = state.sampling.extra_args or {}
        session_id = str(extra_args.get("session_id") or None)
        force_reset = bool(extra_args.get("force_reset") or False)

        if force_reset or self.state.session_id is None or self.state.session_id != session_id:
            self.state.reset()
            self.state.session_id = session_id
            extension = False
        else:
            extension = True

        camera = multi_modal_data.get("camera", None)
        if camera is None:
            raise ValueError("LingbotWorldFastPipeline requires camera poses in multi_modal_data['camera'].")

        # chunk_latent_frames: VAE temporal stride is 4, so sampling.chunk_frames
        # (pixel frames per chunk) must be a positive multiple of 4.
        if sampling.chunk_frames is None or sampling.chunk_frames <= 0:
            raise ValueError(
                f"sampling.chunk_frames must be a positive int; got {sampling.chunk_frames}."
            )
        if sampling.chunk_frames % 4 != 0:
            raise ValueError(
                f"sampling.chunk_frames={sampling.chunk_frames} must be divisible by "
                f"the VAE temporal stride 4."
            )
        chunk_latent_frames = sampling.chunk_frames // 4

        # Lingbot ships a 4-step distilled timestep schedule.
        max_steps = len(CONFIG["timesteps_index"])
        if sampling.num_inference_steps is None:
            sampling.num_inference_steps = max_steps
        elif sampling.num_inference_steps <= 0 or sampling.num_inference_steps > max_steps:
            raise ValueError(
                f"sampling.num_inference_steps must be in [1, {max_steps}]; "
                f"got {sampling.num_inference_steps}."
            )

        if extension:
            assert multi_modal_data.get("image") is None, (
                "image must not be provided on extension calls; it is only used on the first call of a session"
            )

        batch_size = 1
        c2ws = camera.get("poses")
        max_area = CONFIG["max_area"]

        new_lat_f = max(sampling.num_chunks * chunk_latent_frames, 1)
        if extension:
            num_frames = new_lat_f * 4
        else:
            num_frames = (new_lat_f - 1) * 4 + 1
        if len(c2ws) < num_frames:
            raise ValueError(
                f"camera trajectory has {len(c2ws)} poses; need >= {num_frames} "
                f"for {sampling.num_chunks} chunks (chunk_latent_frames={chunk_latent_frames})."
            )
        c2ws = c2ws[:num_frames]

        if not extension:
            img = multi_modal_data.get("image")
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
            h, w = img.shape[1:]
            aspect_ratio = h / w
            lat_h = round(
                np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
            )
            lat_w = round(
                np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
            )
            h = lat_h * self.vae_stride[1]
            w = lat_w * self.vae_stride[2]
        else:
            img = None
            h, w, lat_h, lat_w = self.state.h, self.state.w, self.state.lat_h, self.state.lat_w

        max_seq_len = chunk_latent_frames * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = state.sampling.seed
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        # Two separate generators to keep noise consistent across PP ranks:
        # - seed_g: chunk-initial noise.
        # - seed_g_addnoise: scheduler.add_noise consumed on last rank only.
        seed_g = torch.Generator(device=self.device).manual_seed(seed)
        seed_g_addnoise = torch.Generator(device=self.device).manual_seed(seed + 1)

        # Sampler timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, shift=CONFIG["sample_shift"])
        ts_idx = CONFIG["timesteps_index"][: sampling.num_inference_steps]
        denoise_timesteps = self.scheduler.timesteps[ts_idx].to(self.device)

        # Text + camera Plucker
        context_list = self.text_encoder([prompt], self.text_encoder_device)
        context_list = [c.to(self.device) for c in context_list]

        Ks_raw = torch.from_numpy(camera.get("intrinsics"))
        Ks_t = get_Ks_transformed(
            Ks_raw, height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w
        )[0]
        len_c2ws_orig = len(c2ws)
        tgt_indices_full = np.linspace(0, len_c2ws_orig - 1, new_lat_f)
        c2ws_infer_full = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws_orig - 1, len_c2ws_orig),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=tgt_indices_full,
        )
        c2ws_infer_full = compute_relative_poses(c2ws_infer_full, framewise=True)
        c2ws_infer_full = c2ws_infer_full.to(self.device).to(torch.float32)
        Ks_t = Ks_t.to(self.device).to(torch.float32)

        anchor_latent: torch.Tensor | None = None
        if is_pipeline_first_stage():
            self.vae.reset()
            if not extension:
                anchor_pixels = (
                    torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode="bicubic")
                    .transpose(0, 1)
                    .to(self.device)
                )
                anchor_latent = self.vae.init(anchor_pixels)  # [16, 1, lat_h, lat_w]
            else:
                zero_frame = torch.zeros(3, 1, h, w, device=self.device)
                self.vae.init(zero_frame)

        # Per-block self-attn config from sampling.extra_args. sink_size and
        # local_attn_size are in chunks (each chunk has chunk_latent_frames).
        # Defaults match the unbounded behaviour: no sinks, no rolling, no
        # adaptive refresh.
        sink_size = int(extra_args.get("sink_size", 3))
        local_attn_size = int(extra_args.get("local_attn_size", 3))
        sink_threshold = float(extra_args.get("sink_threshold", 0.2))

        if extension:
            prev = (
                self.state.session_chunk_latent_frames,
                self.state.session_sink_size,
                self.state.session_local_attn_size,
            )
            curr = (chunk_latent_frames, sink_size, local_attn_size)
            if prev != curr:
                raise ValueError(
                    f"Extension call config mismatch: "
                    f"chunk_latent_frames/sink_size/local_attn_size went from {prev} to {curr}. "
                    f"These must stay constant within a session."
                )
        else:
            self.state.session_chunk_latent_frames = chunk_latent_frames
            self.state.session_sink_size = sink_size
            self.state.session_local_attn_size = local_attn_size

        for block in self.model.blocks:
            sa = getattr(block, "self_attn", None)
            if sa is None:
                continue  # PPMissingLayer on non-owning ranks
            sa.sink_size = sink_size
            sa.local_attn_size = local_attn_size
            sa.sink_threshold = sink_threshold

        # KV cache sizing — per this rank's owned layer slice and per slot.
        # Rolling cache holds (sink_size + local_attn_size) chunks worth of K/V,
        # each chunk occupying ``chunk_latent_frames * frame_seqlen`` tokens.
        model_args = self.model.config
        transformer_dtype = self.target_dtype
        frame_seqlen = int(lat_h * lat_w // 4)
        if local_attn_size > 0:
            extra_kv_size = (sink_size + local_attn_size) * chunk_latent_frames * frame_seqlen
        else:
            extra_kv_size = frame_seqlen * new_lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size
        owned_num_layers = self.model.end_layer - self.model.start_layer

        if not extension:
            self.state.create_kv_caches(
                batch_size,
                transformer_dtype,
                self.device,
                extra_kv_size,
                owned_num_layers,
                local_num_heads,
                head_dim,
                num_slots=sampling.num_inference_steps,
            )
        elif local_attn_size <= 0:
            # Unbounded slots: grow on extension to fit the new frames.
            self.state.extend_kv_caches(extra_kv_size)
        # If local_attn_size > 0, slot capacity is fixed; rolling region absorbs
        # new chunks without needing to grow the buffer.

        prev_lat_f = self.state.current_lat_f
        total_kv_size = frame_seqlen * (prev_lat_f + new_lat_f)
        start_token_offset = prev_lat_f * frame_seqlen

        # State population.
        state.prompt_embeds = None  # unused; lingbot keeps text as raw list[Tensor]
        state.latents = None  # per-chunk latents are stacked by encode_chunk_inputs
        state.timesteps = denoise_timesteps
        state.step_index = 0
        state.scheduler = LingbotFlowScheduler(self.scheduler, denoise_timesteps)
        state.do_true_cfg = False

        state.extra["context"] = context_list
        state.extra["anchor_latent"] = anchor_latent
        state.extra["start_token_offset"] = start_token_offset
        state.extra["max_attention_size"] = total_kv_size
        state.extra["frame_seqlen"] = frame_seqlen
        state.extra["max_seq_len"] = max_seq_len
        state.extra["chunk_latent_frames"] = chunk_latent_frames
        state.extra["lat_h"] = lat_h
        state.extra["lat_w"] = lat_w
        state.extra["h"] = h
        state.extra["w"] = w
        state.extra["new_lat_f"] = new_lat_f
        state.extra["extension"] = extension
        state.extra["rolling_enabled"] = local_attn_size > 0
        state.extra["seed_g"] = seed_g
        state.extra["seed_g_addnoise"] = seed_g_addnoise

        state.extra["c2ws_infer_full"] = c2ws_infer_full
        state.extra["Ks_transformed"] = Ks_t

        return state

    def encode_chunk_inputs(
        self,
        state: DiffusionRequestState,
        new_idxs: list[int],
    ) -> torch.Tensor:
        """Build per-chunk noise, plus VAE-encoded y and Plucker on first stage."""
        seed_g = state.extra["seed_g"]
        chunk_latent_frames = state.extra["chunk_latent_frames"]
        lat_h = state.extra["lat_h"]
        lat_w = state.extra["lat_w"]
        h = state.extra["h"]
        w = state.extra["w"]
        chunks = state.extra["chunks"]
        B = len(new_idxs)

        # noise
        noise = torch.randn(
            B,
            16,
            chunk_latent_frames,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )

        if not is_pipeline_first_stage():
            return noise

        c2ws_infer_full = state.extra["c2ws_infer_full"]
        Ks_t = state.extra["Ks_transformed"]
        anchor_latent: torch.Tensor | None = state.extra["anchor_latent"]
        extension: bool = state.extra["extension"]

        # per-chunk stream-encode y + per-chunk msk
        for idx in new_idxs:
            is_anchor_chunk = (not extension) and idx == 0
            if is_anchor_chunk:
                tail_frames = 4 * (chunk_latent_frames - 1)
                if tail_frames > 0:
                    zeros = torch.zeros(3, tail_frames, h, w, device=self.device)
                    tail_lat = self.vae.encode(zeros)
                    assert anchor_latent is not None
                    vae_lat = torch.cat([anchor_latent, tail_lat], dim=1)
                else:
                    assert anchor_latent is not None
                    vae_lat = anchor_latent
            else:
                zeros = torch.zeros(3, 4 * chunk_latent_frames, h, w, device=self.device)
                vae_lat = self.vae.encode(zeros)

            msk_chunk = torch.zeros(4, chunk_latent_frames, lat_h, lat_w, device=self.device)
            if is_anchor_chunk:
                msk_chunk[:, 0] = 1
            chunks[idx].extra["y"] = torch.cat([msk_chunk, vae_lat], dim=0)

        # plucker
        frame_indices = torch.tensor(
            [ci * chunk_latent_frames + f for ci in new_idxs for f in range(chunk_latent_frames)],
            device=c2ws_infer_full.device,
            dtype=torch.long,
        )
        batched_c2ws = c2ws_infer_full[frame_indices]  # [B*chunk_latent_frames, 3, 4]
        batched_Ks = Ks_t.repeat(B * chunk_latent_frames, 1)  # [B*chunk_latent_frames, 4]
        batched_plucker = get_plucker_embeddings(batched_c2ws, batched_Ks, h, w, only_rays_d=False)
        batched_plucker = rearrange(
            batched_plucker,
            "f (h c1) (w c2) c -> f h w (c c1 c2)",
            c1=int(h // lat_h),
            c2=int(w // lat_w),
        )
        batched_plucker = batched_plucker.view(B, chunk_latent_frames, lat_h, lat_w, -1)
        batched_plucker = batched_plucker.permute(0, 4, 1, 2, 3).contiguous().to(self.target_dtype)

        for i, idx in enumerate(new_idxs):
            chunks[idx].extra["plucker"] = batched_plucker[i : i + 1]

        return noise

    def set_pp_recv_dict_buffers(self, state: DiffusionRequestState) -> None:
        if get_pipeline_parallel_world_size() == 1:
            return

        pp_group = get_pp_group()
        slo_fps = getattr(state.sampling, "slo_fps", None)
        slo_max_batch = getattr(state.sampling, "slo_max_batch", 1)
        slo_max_batch = max(1, slo_max_batch if slo_fps else 1)

        chunk_latent_frames = state.extra["chunk_latent_frames"]
        lat_h = state.extra["lat_h"]
        lat_w = state.extra["lat_w"]
        max_seq_len = state.extra["max_seq_len"]
        n_steps = int(state.timesteps.shape[0])

        latents_dtype = torch.float32
        it_dtype = self.target_dtype

        for batch_size in range(1, slo_max_batch * n_steps + 1):
            latents_template = {
                "latents": torch.empty(
                    batch_size, 16, chunk_latent_frames, lat_h, lat_w, dtype=latents_dtype, device="meta"
                )
            }
            it_template = {
                "hidden_states": torch.empty(batch_size, max_seq_len, self.model.dim, dtype=it_dtype, device="meta"),
                "grid_sizes": torch.empty(batch_size, 3, dtype=torch.long, device="meta"),
                "seq_lens": torch.empty(batch_size, dtype=torch.long, device="meta"),
                "c2ws_plucker_emb": torch.empty(batch_size, max_seq_len, self.model.dim, dtype=it_dtype, device="meta"),
            }
            pp_group.set_recv_dict_buffer("latents", -1, latents_template, batch_size=batch_size)
            pp_group.set_recv_dict_buffer("intermediate", 0, it_template, batch_size=batch_size)

    def denoise_step(
        self,
        state: DiffusionRequestState,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """Fused transformer forward for the batch of chunks.

        Each row's per-chunk metadata (``current_starts``, ``y``, ``c2ws_plucker_emb``)
        is read from state.extra keyed by chunk index. Rows whose timestep is 0
        carry the KV-update payload (the chunk's saved x0) — their output is
        ignored by ``step_scheduler``.
        """
        chunk_idxs: list[int] = state.extra["current_chunk_idxs"]
        assert len(chunk_idxs) == batch_size

        chunk_latent_frames = state.extra["chunk_latent_frames"]
        frame_seqlen = state.extra["frame_seqlen"]
        start_token_offset = state.extra["start_token_offset"]
        chunks = state.extra["chunks"]
        context_list = state.extra["context"]

        x_list, y_list, plucker_list = None, None, None
        if is_pipeline_first_stage():
            x_list = [state.latents[i] for i in range(batch_size)]
            y_list = [chunks[ci].extra["y"] for ci in chunk_idxs]
            plucker_list = [chunks[ci].extra["plucker"] for ci in chunk_idxs]

        current_starts = [start_token_offset + ci * chunk_latent_frames * frame_seqlen for ci in chunk_idxs]
        slot_idxs = state.extra["chunk_step_idxs"]

        positive_kwargs = {
            "x": x_list,
            "t": state.batched_timesteps,
            "context": [context_list[0]] * batch_size,
            "seq_len": state.extra["max_seq_len"],
            "y": y_list,
            "dit_cond_dict": {"c2ws_plucker_emb": plucker_list},
            "kv_cache": self.state.get_kv_caches(),
            "local_end_index": self.state.local_end_index,
            "global_end_index": self.state.global_end_index,
            "crossattn_cache": self.state.get_crossattn_caches(),
            "current_starts": current_starts,
            "slot_idxs": slot_idxs,
            "evict_queues": self.state.evict_queues,
            "max_attention_size": state.extra["max_attention_size"],
        }

        preposted_its = state.extra.pop("preposted_its", None)
        return self.predict_noise_maybe_with_cfg(
            do_true_cfg=False,
            true_cfg_scale=1.0,
            positive_kwargs=positive_kwargs,
            negative_kwargs=None,
            buf_idx=state.step_index % 2,
            batch_size=batch_size,
            preposted_its=preposted_its,
        )

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
        *,
        per_request_scheduler: Any | list[Any] | None = None,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        if per_request_scheduler is None:
            per_request_scheduler = state.scheduler

        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            state.batched_timesteps,
            state.latents,
            do_true_cfg=False,
            per_request_scheduler=per_request_scheduler,
            generator=state.extra["seed_g_addnoise"],
            batch_size=batch_size,
            receive_latents=False,
        )
        state.step_index += 1

    def prefetch_tensors(
        self,
        state: DiffusionRequestState,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        if get_pipeline_parallel_world_size() == 1:
            return
        buf_idx = state.step_index % 2
        preposted = self.prefetch_tensors_maybe_with_cfg(
            do_true_cfg=False,
            buf_idx=buf_idx,
            batch_size=batch_size,
        )
        if isinstance(preposted, AsyncLatents):
            state.latents = preposted
        elif preposted is not None:
            state.extra["preposted_its"] = preposted

    def post_decode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionOutput:
        """VAE-decode the finished chunks with prior-tail warmup.

        Mirrors forward()'s decode block: on extension calls prepend the prior
        chunk's last 2 latents to prime the temporal-causal feat_map, then drop
        ``4*k - 3`` leading pixels. After decoding, refresh ``last_decoded_latent``
        with the tail of the new latents so the next call's decode is warm.
        """
        self._sync_pp_send()
        pred_latent_chunks = state.latents.transpose(0, 1).reshape(
            state.latents.shape[1],
            state.latents.shape[0] * state.latents.shape[2],
            state.latents.shape[3],
            state.latents.shape[4],
        )
        # pred_latent_chunks: [16, B*chunk_latent_frames, lat_h, lat_w]

        extension = state.extra["extension"]
        if self.state.last_decoded_latent is not None:
            warmup = self.state.last_decoded_latent.to(pred_latent_chunks.device, pred_latent_chunks.dtype)
            k = warmup.shape[1]
            drop = 4 * k - 3
            to_decode = torch.cat([warmup, pred_latent_chunks], dim=1)
            videos = self.vae.decode([to_decode])
            videos = [v[:, drop:] for v in videos]
        else:
            videos = self.vae.decode([pred_latent_chunks])

        self.state.last_decoded_latent = pred_latent_chunks[:, -2:].detach().clone()

        sampling = state.sampling
        chunks_so_far = state.extra.get("chunks_decoded", 0)
        chunks_this_call = state.latents.shape[0]
        is_final = chunks_so_far + chunks_this_call >= sampling.num_chunks
        if is_final:
            if not extension:
                self.state.h = state.extra["h"]
                self.state.w = state.extra["w"]
                self.state.lat_h = state.extra["lat_h"]
                self.state.lat_w = state.extra["lat_w"]
                self.state.frame_seqlen = state.extra["frame_seqlen"]
            self.state.advance(state.extra["new_lat_f"])

        return DiffusionOutput(output=videos[0])

    def load_weights(self, weights):
        pass
