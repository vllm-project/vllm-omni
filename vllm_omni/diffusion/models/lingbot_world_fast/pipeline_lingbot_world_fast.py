import logging
import math
import os
import random
import sys
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan
from einops import rearrange
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader
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
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportCameraPosInput, SupportImageInput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.platforms import current_omni_platform

from .cam_utils import (
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from .flow_scheduler import LingbotFlowScheduler
from .fm_solvers_unipc import FlowUniPCMultistepScheduler
from .state_lingbot_world_fast import LingbotWorldFastState
from .stream_autoencoder_kl_wan import StreamAutoencoderKLWan
from .wan_fast import WanModelFast

logger = logging.getLogger(__name__)


CONFIG = {
    "num_train_timesteps": 1000,
    "timesteps_index": [0, 179, 358, 679],
    "sample_shift": 10.0,
    "t5_checkpoint": "models_t5_umt5-xxl-enc-bf16.pth",
    "t5_tokenizer": "google/umt5-xxl",
    "vae_checkpoint": "Wan2.1_VAE.pth",
    "fast_noise_checkpoint": "Lingbot-World-Fast",
}

T5_CONFIG = UMT5Config(
    vocab_size=256384,
    d_model=4096,
    d_kv=64,
    d_ff=10240,
    num_heads=64,
    num_layers=24,
    relative_attention_num_buckets=32,
    shared_pos=False,
    dropout_rate=0.1,
    is_encoder_decoder=False,
)


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

        model_path = os.path.dirname(self.od_config.model)
        assert model_path is not None, "lingbot_dir is None"

        tokenizer_path = os.path.join(model_path, CONFIG["t5_tokenizer"])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        t5_checkpoint = os.path.join(model_path, CONFIG["t5_checkpoint"])
        wan_sd = torch.load(t5_checkpoint, map_location="cpu", weights_only=True)
        self.text_encoder = UMT5EncoderModel(T5_CONFIG)
        # The checkpoint is in Wan's naming, not HF's — remap before loading.
        self.text_encoder.load_state_dict(_wan_t5_to_hf_state_dict(wan_sd), assign=True)
        self.text_encoder = self.text_encoder.to(device=self.device, dtype=self.target_dtype).eval()

        self.vae_stride = od_config.model_config.get("vae_stride", (4, 8, 8))
        self.patch_size = od_config.model_config.get("patch_size", (1, 2, 2))

        needs_vae = od_config.stream_batch is False or is_pipeline_first_stage()
        if needs_vae:
            vae_path = os.path.join(model_path, CONFIG["vae_checkpoint"])
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
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(self.vae.device, self.vae.dtype)
            self.latents_scale = [latents_mean, latents_std]

            if od_config.stream_batch:
                self.vae = StreamAutoencoderKLWan(self.vae, self.latents_scale)
        else:
            self.vae = None
            self.latents_scale = None

        logger.info(f"Creating WanModelFast from {model_path}")

        self.model = WanModelFast(
            in_dim=od_config.tf_model_config.get("in_dim"),
            dim=od_config.tf_model_config.get("dim"),
            ffn_dim=od_config.tf_model_config.get("ffn_dim"),
            num_layers=od_config.tf_model_config.get("num_layers"),
            num_heads=od_config.tf_model_config.get("num_heads"),
        )

        # Tell the loader where the transformer weights live. Without this,
        # get_all_weights() yields nothing and load_weights() runs empty.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=CONFIG["fast_noise_checkpoint"],
                revision=None,
                prefix="model.",
                fall_back_to_pt=True,
            ),
        ]

        # Partition transformer across PP ranks (no-op at PP=1).
        self.model.apply_pp_split()

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )

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

        session_id = req.sampling_params.extra_args.get("session_id")

        if session_id is None:
            # Create a unique id if none is specified without messing with RNG state
            session_id = time.time()

        session_id = str(session_id)

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
            raise ValueError("Camera positions are required by this model.")

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
        num_frames = max(13, num_frames)

        c2ws = camera.get("poses")
        latent_frames_per_chunk = self.od_config.model_config["latent_frames_per_chunk"]
        max_area = self.od_config.model_config["max_area"]

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

        new_lat_f = int(new_lat_f - (new_lat_f % latent_frames_per_chunk))
        new_lat_f = max(new_lat_f, 1)
        max_seq_len = latent_frames_per_chunk * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
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

        context = self.encode_prompt([prompt], self.device)
        context = [t.to(self.device) for t in context]

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
            y = self.encode_video([pixels])[0]
        else:
            pixels = torch.zeros(3, 4 * new_lat_f + 1, h, w, device=self.device)
            y = self.encode_video([pixels])[0][:, 1:]
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
            latents_chunk = latent.split(latent_frames_per_chunk, dim=1)  # [c, f, h, w]
            condition_chunk = y.split(latent_frames_per_chunk, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(latent_frames_per_chunk, dim=2)
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
                    "kv_cache": self.state.get_kv_cache(),
                    "local_end_index": self.state.local_end_index,
                    "global_end_index": self.state.global_end_index,
                    "crossattn_cache": self.state.get_crossattn_caches(),
                    "current_starts": start_token_offset + chunk_id * latent_frames_per_chunk * frame_seqlen,
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

            videos = None
            if self.device.index == 0:
                # Wan VAE decode() calls clear_cache() internally, so the very first latent always runs the i==0 path
                # (no temporal upsample, single-frame output) and leaves feat_map polluted with thatbias.
                # The decoder's stacked temporal-causal layers also need ~2 latents of streaming context before deeper
                # feat_map slots match a true mid-stream decode. On extension, prepend the prior chunk's last 2 latents
                # so warmup_0 absorbs the i==0 bias and warmup_1 fully primes the cache. Then discard the
                # 4*K - 3 leading pixels (re-decodes of already-shown frames).
                if extension and self.state.last_decoded_latent is not None:
                    warmup = self.state.last_decoded_latent.to(pred_latent_chunks.device, pred_latent_chunks.dtype)
                    k = warmup.shape[1]
                    drop = 4 * k - 3
                    to_decode = torch.cat([warmup, pred_latent_chunks], dim=1)
                    videos = self.decode_video([to_decode])
                    videos = [v[:, drop:] for v in videos]
                else:
                    videos = self.decode_video([pred_latent_chunks])

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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

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

    # ------------------------------------------------------------------
    # micro-step execution
    # ------------------------------------------------------------------

    def predict_noise(
        self,
        intermediate_tensors: IntermediateTensors | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Single transformer forward; returns IntermediateTensors on non-last PP stages."""
        with torch.amp.autocast(current_omni_platform.device_type, dtype=self.target_dtype):
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
        max_area = self.od_config.model_config.get("max_area", 480 * 832)

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
        context_list = self.encode_prompt([prompt], self.device)
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
            "kv_cache": self.state.get_kv_cache(),
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
