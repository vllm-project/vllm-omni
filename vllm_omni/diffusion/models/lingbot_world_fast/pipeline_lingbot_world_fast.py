import itertools
import logging
import math
import os
import random
import sys
import time
from collections.abc import Iterable
from contextlib import contextmanager

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

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportCameraPosInput, SupportImageInput
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .cam_utils import (
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)
from .flow_scheduler import LingbotFlowScheduler
from .fm_solvers_unipc import FlowUniPCMultistepScheduler
from .state_lingbot_world_fast import LingbotWorldFastState
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


class LingbotWorldFastPipeline(nn.Module, SupportImageInput, SupportCameraPosInput, CFGParallelMixin):
    _ar_diffusion_kv_state = None  # set by the runner before each forward

    def _kv_get(self, seq_len=None, update_kv_cache=False):
        return self._ar_diffusion_kv_state.get_kv_caches(
            is_negative=False,
            seq_len=seq_len,
            commit_current=update_kv_cache,
        )

    def _kv_commit(self):
        self._ar_diffusion_kv_state.commit_paged_context(is_negative=False)

    def _kv_get_cross(self):
        """Cross-attn cache from the engine pool (text k/v)."""
        return self._ar_diffusion_kv_state.get_cross_kv_caches(is_negative=False)

    def _kv_populate_cross(self, context: torch.Tensor) -> None:
        s = self._ar_diffusion_kv_state
        is_negative = False

        projected = self.transformer.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )
        for i, block in enumerate(self.transformer.blocks):
            ca = block.cross_attn
            n, d = ca.num_heads, ca.head_dim
            k = v = None
            if projected is not None:
                k = ca.norm_k(ca.k(projected)).unflatten(-1, (n, d))
                v = ca.v(projected).unflatten(-1, (n, d))
            s.kv_cache.write_cross_kv(i, is_negative, k, v)
        s._cross_text_populated[is_negative] = True
        logger.info(
            "AR-Diffusion CROSS POPULATE [%s]: %d layers",
            "neg" if is_negative else "pos",
            len(self.transformer.blocks),
        )

    def _kv_reset(self, clear_video_latents: bool = True):
        """Reset the engine's pooled session window plus the model's non-KV state.

        DreamZero resets at the attention-window boundary; the engine pool drops the
        same window so the next forward starts fresh. ``clear_video_latents=False``
        keeps the accumulated video latents for export.

        ``clear_video_latents=False`` also marks a window ("inference") reset: the
        prompt is unchanged, so the pool keeps the text cross-attn K/V and only the
        image half repopulates on the restart forward.
        """
        self.state.reset()
        self._ar_diffusion_kv_state.reset(keep_cross_text=False)

    def __init__(self, *, od_config: OmniDiffusionConfig):
        super().__init__()

        engine_backend = str(getattr(od_config, "engine_backend", "") or "")
        if "ar_diffusion" not in engine_backend.lower().replace("-", "_"):
            raise ValueError(
                "LingbotWorldFastPipeline requires the AR-Diffusion engine; set "
                "engine_backend: vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine "
                f"in the deploy config (got engine_backend={engine_backend!r})."
            )

        self.od_config = od_config
        self.parallel_config = od_config.parallel_config

        self.device = get_local_device()
        self.num_layers = od_config.tf_model_config.get("num_layers")
        self.text_len = od_config.tf_model_config.get("text_len")

        self.target_dtype = od_config.dtype

        self.control_type = "cam"
        self.num_train_timesteps = CONFIG["num_train_timesteps"]

        self.sp_size = od_config.parallel_config.world_size

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

        self.vae_stride = od_config.model_config["vae_stride"]
        self.patch_size = od_config.model_config["patch_size"]

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
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.vae.device, self.vae.dtype
        )
        self.latents_scale = [latents_mean, latents_std]

        logger.info(f"Creating WanModelFast from {model_path}")

        self.transformer = WanModelFast(
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
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        self.base_scheduler = FlowUniPCMultistepScheduler(
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

        session_id = req.sampling_params.extra_args.get("session_id", str(time.time()))

        session_id = str(session_id)

        force_reset = req.sampling_params.extra_args.get("force_reset") or False

        extension = True

        if force_reset or self.state.session_id is None or self.state.session_id != session_id:
            self._kv_reset()
            self.state.init_state(self.device, self.num_layers)
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
            assert self.transformer.config.local_attn_size == -1, (
                "video extension requires the model to be configured with local_attn_size == -1"
            )
            # The engine KV pool and this pipeline state are torn down through
            # different paths (a failed request or an LRU eviction frees the
            # engine session, while self.state survives), so an extension call
            # can otherwise run against an empty KV history and silently produce
            # a context-free continuation. Fail loudly instead.
            adapter = self._ar_diffusion_kv_state.pos
            expected_tokens = self.state.current_lat_f * (self.state.frame_seqlen or 0)
            if adapter.num_computed_tokens != expected_tokens:
                raise RuntimeError(
                    "Lingbot session out of sync with the engine KV pool: the pipeline has "
                    f"{self.state.current_lat_f} committed latent frames ({expected_tokens} tokens) but the "
                    f"engine session holds {adapter.num_computed_tokens} tokens. An earlier request on this "
                    "session probably failed or the session was evicted, which frees the engine KV while the "
                    "pipeline state survives. Start a new session (force_reset with a fresh image)."
                )

        num_frames = req.sampling_params.num_frames
        # In order to generate something num_frames must be at least 5 since it expects 4*n + 1 as input
        # 25 is the smallest length supported by the model. Smaller values generate tensors with dimension zero/negative
        num_frames = max(13, num_frames)

        c2ws = camera.get("poses")
        latent_frames_per_chunk = self.od_config.model_config["latent_frames_per_chunk"]

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

            # We must have lat_h * lat_f = (72*72) = (2**6)*(3**4) due to limitations of the AR engine kernel
            # The value 1296 because it must be a multiple of 16 and such that h*w computed below is close to max_area
            # Also we want lat_h / lat_f to be as close to aspect_ratio as possible. Additionally, both must be even
            divisors = [(2**x) * (3**y) for (x, y) in itertools.product(range(1, 6), range(5))]
            target = np.sqrt(1296 * 4 * aspect_ratio)
            _, lat_h = min([(np.abs(target - div), div) for div in divisors])

            lat_w = 1296 * 4 // lat_h

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
        self.base_scheduler.set_timesteps(self.num_train_timesteps, shift=CONFIG["sample_shift"])
        timesteps = self.base_scheduler.timesteps[CONFIG["timesteps_index"]]
        self.scheduler = LingbotFlowScheduler(self.base_scheduler, timesteps.to(self.device))

        context = self.encode_prompt([prompt], self.device)
        context = [t.to(self.device) for t in context]
        self._kv_populate_cross(context)

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

        no_sync_model = getattr(self.transformer, "no_sync", noop_no_sync)

        # Initialize (fresh) or grow (extension) the KV cache. Cross-attn cache is
        # left untouched on extension so text-context k/v computed on the first call
        # are reused via crossattn_cache[i]["is_init"] == True.
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1] // 4)

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
                    "current_latent": current_latent,
                    "timesteps": timesteps,
                    "context": [context[0]],
                    "seq_len": max_seq_len,
                    "y": [current_condition],
                    "dit_cond_dict": dit_cond_dict,
                    "current_start": start_token_offset + chunk_id * latent_frames_per_chunk * frame_seqlen,
                    "max_attention_size": total_kv_size,
                    "pred_latent_chunks": pred_latent_chunks,
                    "seed_g": seed_g,
                    "chunk_id": chunk_id,
                }

                self.diffuse(**kwargs)

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

    def diffuse(
        self,
        timesteps,
        current_latent,
        context,
        seq_len,
        y,
        dit_cond_dict,
        current_start,
        max_attention_size,
        pred_latent_chunks,
        seed_g,
        chunk_id,
    ):
        kwargs = {
            "context": context,
            "seq_len": seq_len,
            "y": y,
            "dit_cond_dict": dit_cond_dict,
            "kv_cache": self._kv_get(seq_len, False),
            "local_end_index": self.state.local_end_index,
            "global_end_index": self.state.global_end_index,
            "crossattn_cache": self._kv_get_cross(),
            "current_start": current_start,
            "max_attention_size": max_attention_size,
        }

        for timestep_idx in range(len(timesteps)):
            current_timestep = [timesteps[timestep_idx]]

            timestep = torch.stack(current_timestep).to(self.device)

            kwargs.update({"x": [current_latent], "t": timestep})

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=False,
                positive_kwargs=kwargs,
                negative_kwargs=None,
                cfg_normalize=False,
                true_cfg_scale=0.0,
            )

            current_latent = self.scheduler_step_maybe_with_cfg(
                noise_pred, timestep, current_latent, do_true_cfg=False, generator=seed_g
            )

        pred_latent_chunks.append(current_latent)

        # Update kv cache
        context_timestep = [timesteps[-1] * 0.0]
        timestep = torch.stack(context_timestep).to(self.device)

        kwargs.update(
            {
                "x": [current_latent],
                "t": timestep,
                "update_kv_cache": True,
                "kv_cache": self._kv_get(seq_len, True),
            }
        )

        noise_pred = self.predict_noise_maybe_with_cfg(
            do_true_cfg=False,
            positive_kwargs=kwargs,
            negative_kwargs=None,
            cfg_normalize=False,
            true_cfg_scale=0.0,
        )

        return current_latent

    def predict_noise(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through transformer to predict noise.

        Args:
            current_model: The transformer model to use (transformer or transformer_2)
            **kwargs: Arguments to pass to the transformer

        Returns:
            Predicted noise tensor or IntermediateTensors on non-last PP stages.
        """
        result, _ = self.transformer(**kwargs)

        if kwargs.get("update_kv_cache", False):
            self._kv_commit()

        return result[0]

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
