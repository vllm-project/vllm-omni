import logging
import os
import random
import sys
import time
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
from diffusers import AutoencoderKLWan
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_pipeline_parallel_world_size,
    get_pp_group,
    is_pipeline_first_stage,
)
from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents, PipelineParallelMixin
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.platforms import current_omni_platform

from .causvid import WanModel, balance_layers_by_cost
from .flow_match import FlowMatchScheduler
from .flow_scheduler import CausVidFlowScheduler
from .state_causvid import CausVidState
from .stream_autoencoder_kl_wan import StreamAutoencoderKLWan

logger = logging.getLogger(__name__)

CONFIG = {
    "autoregressive_checkpoint": "causvid/wan_causal_dmd_v2v/model.pt",
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


def _load_v2v_source_video(path: str, height: int | None, width: int | None) -> torch.Tensor:
    """Load a V2V source video from a local path into ``[3, T, H, W]`` in ``[-1, 1]``.

    Used so requests can carry a small path string instead of serializing a
    multi-hundred-MB tensor through the broadcast queue: every PP worker loads
    the clip from its own local disk. Accepts ``.pt`` (a pre-saved
    ``[3, T, H, W]`` tensor) or a video file (mp4/mov/avi) via imageio or opencv.
    Resizes to ``(height, width)`` when both are given.
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".pt":
        video = torch.load(path, map_location="cpu")
        if video.dim() != 4 or video.shape[0] != 3:
            raise ValueError(f"Expected [3, T, H, W] tensor in {path}; got shape {tuple(video.shape)}.")
        video = video.float()
    else:
        frames_np = None
        last_err: Exception | None = None
        try:
            import imageio.v3 as iio

            frames_np = iio.imread(path)  # [T, H, W, 3] uint8
        except Exception as e:  # noqa: BLE001 - fall back to opencv
            last_err = e
        if frames_np is None:
            try:
                import cv2
                import numpy as np

                cap = cv2.VideoCapture(path)
                buf: list = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                if buf:
                    frames_np = np.stack(buf, axis=0)
            except Exception as e:  # noqa: BLE001
                last_err = e
        if frames_np is None:
            raise ImportError(
                "No video backend available to load source video. Install "
                "`imageio imageio-ffmpeg` or `opencv-python`. "
                f"Last error: {last_err}"
            )
        video = torch.from_numpy(frames_np).float() / 255.0 * 2.0 - 1.0  # [T, H, W, 3] in [-1, 1]
        video = video.permute(0, 3, 1, 2)  # [T, 3, H, W]

    if height is not None and width is not None and video.shape[-2:] != (height, width):
        import torch.nn.functional as F

        if video.shape[0] == 3:  # [3, T, H, W] -> resize per frame
            video = F.interpolate(
                video.permute(1, 0, 2, 3), size=(height, width), mode="bilinear", align_corners=False
            ).permute(1, 0, 2, 3)
        else:  # [T, 3, H, W]
            video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)

    if video.shape[0] != 3:  # ensure channel-first [3, T, H, W]
        video = video.permute(1, 0, 2, 3)
    return video.contiguous()


def get_causvid_post_process_func(
    od_config: OmniDiffusionConfig,
):
    def post_process_func(
        video: torch.Tensor,
    ):
        video = video.permute(1, 0, 2, 3)
        return video

    return post_process_func


class CausVidPipeline(torch.nn.Module, PipelineParallelMixin, CFGParallelMixin):
    supports_micro_step_execution: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.target_dtype = od_config.dtype

        torch.backends.cudnn.benchmark = True

        self.model_path = self.od_config.model

        self.state = CausVidState()

        # Step 1: Initialize all models
        self.transformer = WanModel(
            in_dim=od_config.tf_model_config.get("in_dim"),
            dim=od_config.tf_model_config.get("dim"),
            ffn_dim=od_config.tf_model_config.get("ffn_dim"),
            num_layers=od_config.tf_model_config.get("num_layers"),
            num_heads=od_config.tf_model_config.get("num_heads"),
        )
        # Partition transformer across PP ranks (no-op at PP=1). Dynamic scheduling
        # keeps all blocks resident until the warmup rebalance frees the non-owned.
        self.transformer.apply_pp_split(free_blocks=not od_config.enable_dynamic_block_schedule)

        self.sp_size = od_config.parallel_config.sequence_parallel_size
        if getattr(od_config, "stream_batch", False) and self.sp_size > 1:
            raise ValueError(
                "stream_batch is incompatible with sequence parallelism "
                f"(sequence_parallel_size={self.sp_size}); set sequence_parallel_size=1."
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
        self.vae = self.vae.to(self.device, dtype=self.target_dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.vae.device, self.vae.dtype
        )
        self.latents_scale = [latents_mean, latents_std]

        if getattr(self.od_config, "stream_batch", False):
            self.vae = StreamAutoencoderKLWan(self.vae, self.latents_scale)

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

        num_layers = self.transformer.num_layers
        num_heads = self.transformer.num_heads
        head_dim = self.transformer.dim // num_heads

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

                with torch.amp.autocast(current_omni_platform.device_type, dtype=self.target_dtype):
                    flow_pred = self.transformer(
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

            with torch.amp.autocast(current_omni_platform.device_type, dtype=self.target_dtype):
                self.transformer(
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
        owned = set(self.transformer.state_dict().keys())
        state_dict = {k: v for k, v in state_dict.items() if k in owned}
        self.transformer.load_state_dict(state_dict)

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
    # Stream-batch micro-step execution (V2V)
    # ------------------------------------------------------------------
    def prepare_encode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        if not state.prompts or len(state.prompts) > 1:
            raise ValueError("CausVidPipeline only supports a single prompt.")

        self.denoising_step_list = torch.tensor([700, 500, 400, 200], dtype=torch.long, device=self.device)

        sampling = state.sampling

        prompt = state.prompts[0].get("prompt")
        multi_modal_data = state.prompts[0].get("multi_modal_data", {}) or {}

        extra_args = sampling.extra_args or {}
        session_id = str(extra_args.get("session_id") or None)

        # CausVid V2V doesn't support session extension
        self.state.reset()
        self.vae.reset()  # clear streaming VAE feat-maps before a new session
        self.state.session_id = session_id

        # Chunk geometry: chunk_frames defaults to num_frame_per_block * 4 (one
        # block per chunk). Must be a positive multiple of the VAE temporal
        # stride 4.
        if sampling.chunk_frames is None:
            sampling.chunk_frames = self.num_frame_per_block * 4
        if sampling.chunk_frames <= 0:
            raise ValueError(f"sampling.chunk_frames must be > 0; got {sampling.chunk_frames}.")
        if sampling.chunk_frames % 4 != 0:
            raise ValueError(
                f"sampling.chunk_frames={sampling.chunk_frames} must be divisible by the VAE temporal stride 4."
            )
        chunk_latent_frames = sampling.chunk_frames // 4

        # Distilled CausVid timestep schedule (already strips the final t=0).
        max_steps = int(self.denoising_step_list.shape[0])
        if sampling.num_inference_steps is None:
            sampling.num_inference_steps = max_steps
        elif sampling.num_inference_steps <= 0 or sampling.num_inference_steps > max_steps:
            raise ValueError(
                f"sampling.num_inference_steps must be in [1, {max_steps}]; got {sampling.num_inference_steps}."
            )

        source_video = multi_modal_data.get("video", None)
        if source_video is None:
            source_video = extra_args.get("video_path")
        if source_video is None:
            if state.request_id == DUMMY_DIFFUSION_REQUEST_ID:
                # Warmup has no real input video; synthesize a zero clip so the
                # full V2V micro-step path (incl. PP P2P warmup) still runs.
                h = sampling.height or 512
                w = sampling.width or 512
                source_video = torch.zeros(
                    3,
                    1 + sampling.num_chunks * sampling.chunk_frames,
                    h,
                    w,
                    device=self.device,
                    dtype=self.target_dtype,
                )
            else:
                raise ValueError(
                    "CausVidPipeline V2V requires multi_modal_data['video'] as a [3, T, H, W] tensor "
                    "or a local file path (str / pathlib.Path)."
                )

        if isinstance(source_video, os.PathLike):
            source_video = os.fspath(source_video)
        if isinstance(source_video, str):
            source_video = _load_v2v_source_video(source_video, sampling.height, sampling.width)
        if isinstance(source_video, torch.Tensor):
            source_video = source_video.to(self.device, dtype=self.target_dtype)
        else:
            source_video = torch.as_tensor(source_video, device=self.device, dtype=self.target_dtype)
        if source_video.dim() != 4 or source_video.shape[0] != 3:
            raise ValueError(f"source video must be [3, T, H, W]; got shape {tuple(source_video.shape)}.")
        
        _, T_pix, h, w = source_video.shape
        needed_frames = 1 + sampling.num_chunks * sampling.chunk_frames
        if T_pix < needed_frames:
            raise ValueError(
                f"source video has {T_pix} pixel frames; need >= {needed_frames} "
                f"for {sampling.num_chunks} chunks of {sampling.chunk_frames} frames."
            )

        # Latent spatial shape (VAE stride 8).
        lat_h, lat_w = h // 8, w // 8
        frame_seqlen = (lat_h * lat_w) // 4

        # Noise blend factor (V2V SDEdit-style linear blend).
        noise_scale = float(extra_args.get("noise_scale", 0.8))
        if not 0.0 <= noise_scale <= 1.0:
            raise ValueError(f"noise_scale must be in [0, 1]; got {noise_scale}.")
        fixed_noise_scale = bool(extra_args.get("fixed_noise_scale", False))

        seed = sampling.seed if sampling.seed is not None else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device).manual_seed(seed)

        denoise_timesteps = self.denoising_step_list[: sampling.num_inference_steps]

        # Text encoding (once per request).
        context_list = self.encode_prompt([prompt], self.device)
        context_list = [c.to(self.device) for c in context_list]

        sink_size = int(extra_args.get("sink_size", 3))
        local_attn_size = int(extra_args.get("local_attn_size", 3))
        sink_threshold = float(extra_args.get("sink_threshold", 0.2))
        if local_attn_size <= 0:
            raise ValueError(
                f"local_attn_size must be > 0 (rolling KV cache); got {local_attn_size}. "
                f"Unbounded mode is unsupported for stream-batch."
            )

        for block in self.transformer.blocks:
            sa = getattr(block, "self_attn", None)
            if sa is None:
                continue  # PPMissingLayer on non-owning ranks
            sa.sink_size = sink_size
            sa.local_attn_size = local_attn_size
            sa.sink_threshold = sink_threshold

        # KV cache sizing — per this rank's owned layer slice and per slot.
        # Rolling cache holds ``(sink_size + local_attn_size)`` chunks worth of
        # K/V, each chunk occupying ``chunk_latent_frames * frame_seqlen`` tokens,
        # plus chunk 0's extra priming latent frame. Capacity is fixed at session
        # start.
        new_lat_f = sampling.num_chunks * chunk_latent_frames + 1
        kv_size = ((sink_size + local_attn_size) * chunk_latent_frames + 1) * frame_seqlen
        head_dim = self.transformer.dim // self.transformer.num_heads
        local_num_heads = self.transformer.num_heads // self.sp_size
        owned_num_layers = self.transformer.end_layer - self.transformer.start_layer

        # Dynamic block scheduling keeps an all-blocks cache resident (owned on GPU,
        # non-owned on CPU) so a mid-request rebalance can broadcast-migrate state.
        dynamic = self.od_config.enable_dynamic_block_schedule and get_pipeline_parallel_world_size() > 1
        cache_num_layers = self.transformer.num_layers if dynamic else owned_num_layers
        gpu_layers = set(range(self.transformer.start_layer, self.transformer.end_layer)) if dynamic else None

        self.state.create_kv_caches(
            batch_size=1,
            dtype=self.target_dtype,
            device=self.device,
            kv_size=kv_size,
            num_layers=cache_num_layers,
            num_heads=local_num_heads,
            head_dim=head_dim,
            num_slots=sampling.num_inference_steps,
            gpu_layers=gpu_layers,
        )

        # Pin session-level config (informational; we always reset above).
        self.state.session_chunk_latent_frames = chunk_latent_frames
        self.state.session_sink_size = sink_size
        self.state.session_local_attn_size = local_attn_size
        self.state.session_num_inference_steps = sampling.num_inference_steps

        prev_lat_f = self.state.current_lat_f  # always 0 after reset above

        start_token_offset = prev_lat_f * frame_seqlen

        # ── State population ───────────────────────────────────────────────
        state.prompt_embeds = None
        state.latents = None
        state.timesteps = denoise_timesteps
        state.step_index = 0
        state.scheduler = CausVidFlowScheduler(self.scheduler, denoise_timesteps)
        state.do_true_cfg = False

        state.extra["context"] = context_list
        state.extra["source_video"] = source_video
        state.extra["chunk_latent_frames"] = chunk_latent_frames
        state.extra["chunk_frames"] = sampling.chunk_frames
        state.extra["lat_h"] = lat_h
        state.extra["lat_w"] = lat_w
        state.extra["h"] = h
        state.extra["w"] = w
        state.extra["frame_seqlen"] = frame_seqlen
        state.extra["new_lat_f"] = new_lat_f
        state.extra["start_token_offset"] = start_token_offset
        state.extra["noise_scale"] = noise_scale
        state.extra["init_noise_scale"] = noise_scale
        state.extra["fixed_noise_scale"] = fixed_noise_scale
        state.extra["seed_g"] = seed_g

        return state

    def prepare_first_chunk(self, state: DiffusionRequestState) -> DiffusionOutput | None:
        """Denoise chunk 0 alone (batch=1, clean KV) through every step, seed all
        slots from it, then decode it. Mirrors SDV2's ``prepare``."""
        ns = state.total_steps
        state.extra["slot_chunks"] = [0]
        state.latents = self.encode_chunk(state, 0).unsqueeze(0)  # [1, 16, base+1, h, w]
        lat_f = state.latents.shape[2]

        for step in range(ns):
            t = state.timesteps[step : step + 1].view(1, 1).expand(1, lat_f).contiguous()
            noise_pred = self._denoise_forward(state, slot_chunks=[0], t=t, lat_f=lat_f, use_buffer=False)
            state.latents = self.scheduler_step_maybe_with_cfg(
                noise_pred,
                state.timesteps[step : step + 1],
                state.latents,
                do_true_cfg=False,
                per_request_scheduler=[state.scheduler],
                generator=state.extra["seed_g"],
                batch_size=1,
                receive_latents=True,
                use_buffer=False,
            )
            state.step_index += 1

        self.state.seed_all_slots_from(0)
        out = self.decode_chunks(state) if get_pp_group().is_last_rank else None

        state.latents = None
        return out
        
    def prepare_chunks(self, state: DiffusionRequestState) -> None:
        slot_chunks: list[int | None] = state.extra["slot_chunks"]
        ns = len(slot_chunks)
        lat_f = state.extra["chunk_latent_frames"]
        lat_h = state.extra["lat_h"]
        lat_w = state.extra["lat_w"]

        prev = state.latents
        latents = torch.zeros(ns, 16, lat_f, lat_h, lat_w, dtype=torch.float32, device=self.device)
        if prev is not None: latents[1:] = prev[: ns - 1]

        new_idx = slot_chunks[0]
        if new_idx is not None:
            latents[0] = self.encode_chunk(state, new_idx)
        
        state.latents = latents

    def encode_chunk(self, state: DiffusionRequestState, chunk_idx: int) -> torch.Tensor:
        chunk_frames = state.extra["chunk_frames"]
        init_noise_scale = state.extra["init_noise_scale"]
        source_video: torch.Tensor = state.extra["source_video"]
        is_first = chunk_idx == 0

        if state.extra["fixed_noise_scale"] or is_first:
            noise_scale = state.extra["noise_scale"]
            state.extra["first_timestep"] = None
        else:
            noise_scale, first_timestep = self._compute_noise_scale_and_step(
                source_video, 1 + (chunk_idx + 1) * chunk_frames, chunk_frames, state.extra["noise_scale"], init_noise_scale
            )
            state.extra["noise_scale"] = noise_scale
            state.extra["first_timestep"] = first_timestep

        start = chunk_idx * chunk_frames + (not is_first)
        end = start + chunk_frames + is_first
        pixels = source_video[:, start:end]
        source_lat = self.vae.stream_encode(pixels.to(self.device, dtype=self.vae.dtype))
        noise = torch.randn(source_lat.shape, dtype=torch.float32, generator=state.extra["seed_g"], device=self.device)
        return (1.0 - noise_scale) * source_lat + noise_scale * noise

    def _compute_noise_scale_and_step(
        self,
        source_video: torch.Tensor,
        end_idx: int,
        chunk_frames: int,
        prev_noise_scale: float,
        init_noise_scale: float,
    ) -> tuple[float, int]:
        """SDV2 motion-aware noise controller"""
        
        cur = source_video[:, end_idx - chunk_frames : end_idx]
        prv = source_video[:, end_idx - chunk_frames - 1 : end_idx - 1]
        l2 = ((cur - prv) ** 2).mean(dim=(0, 2, 3))  # per-frame, over (C, H, W)
        l2 = (l2.sqrt().max() / 0.2).clamp(0, 1)
        new_noise_scale = (init_noise_scale - 0.1 * l2.item()) * 0.9 + prev_noise_scale * 0.1
        return new_noise_scale, int(1000 * new_noise_scale) - 100



    def denoise_step(self, state: DiffusionRequestState) -> torch.Tensor | None:
        """One steady ladder step: batch position j sits at denoise level j."""
        slot_chunks: list[int | None] = state.extra["slot_chunks"]
        ns = len(slot_chunks)
        base = state.extra["chunk_latent_frames"]

        t = state.timesteps.clone()
        first_t = state.extra.get("first_timestep")
        if first_t is not None:
            t[0] = first_t
        t = t.view(ns, 1).expand(ns, base).contiguous()

        preposted_its = state.extra.get("preposted_its")
        if preposted_its is None and not is_pipeline_first_stage():
            preposted_its = [self._dummy_intermediate(state, t, ns)]



        return self._denoise_forward(
            state,
            slot_chunks=slot_chunks,
            t=t,
            lat_f=base,
            use_buffer=True,
            preposted_its=preposted_its,
        )

    def _dummy_intermediate(self, state: DiffusionRequestState, t: torch.Tensor, batch_size: int) -> IntermediateTensors:
        base = state.extra["chunk_latent_frames"]
        frame_seqlen = state.extra["frame_seqlen"]
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    batch_size, base * frame_seqlen, self.transformer.dim, dtype=self.target_dtype, device=self.device
                ),
                "t": t,
                "xt": torch.zeros(
                    batch_size, 16, base, state.extra["lat_h"], state.extra["lat_w"], dtype=torch.float32, device=self.device
                ),
            }
        )

    def _denoise_forward(
        self,
        state: DiffusionRequestState,
        *,
        slot_chunks: list[int | None],
        t: torch.Tensor,
        lat_f: int,
        use_buffer: bool,
        preposted_its: dict | None = None,
    ) -> torch.Tensor | None:
        """Transformer forward for a batch of (chunk, denoise-level) positions."""
        batch_size = len(slot_chunks)
        base = state.extra["chunk_latent_frames"]
        frame_seqlen = state.extra["frame_seqlen"]
        start_token_offset = state.extra["start_token_offset"]
        context_list = state.extra["context"]
        per_chunk_tokens = lat_f * frame_seqlen
        
        slot_idxs = [i if ci is not None else -1 for i, ci in enumerate(slot_chunks)]
        
        def chunk_start(ci: int | None) -> int:
            # None (dummy) slots reuse chunk 0's window; their output is discarded.
            if ci is None or ci == 0:
                return start_token_offset
            return start_token_offset + (ci * base + 1) * frame_seqlen

        current_starts = [chunk_start(ci) for ci in slot_chunks]
        current_ends = [cs + per_chunk_tokens for cs in current_starts]

        x_list = None
        if is_pipeline_first_stage():
            x_list = [state.latents[i] for i in range(batch_size)]

        patch_size = self.transformer.patch_size
        grid_row = [
            lat_f // patch_size[0],
            state.extra["lat_h"] // patch_size[1],
            state.extra["lat_w"] // patch_size[2],
        ]
        grid_sizes = torch.tensor([grid_row] * batch_size, dtype=torch.long)

        positive_kwargs = {
            "x": x_list,
            "t": t,
            "context": [context_list[0]] * batch_size,
            "grid_sizes": grid_sizes,
            "kv_cache": self.state.get_kv_cache(),
            "local_end_index": self.state.local_end_index,
            "global_end_index": self.state.global_end_index,
            "crossattn_cache": self.state.get_crossattn_cache(),
            "current_start": current_starts,
            "current_end": current_ends,
            "slot_idxs": slot_idxs,
            "evict_queues": self.state.evict_queues,
        }

        noise_pred = self.predict_noise_maybe_with_cfg(
            do_true_cfg=False,
            true_cfg_scale=1.0,
            positive_kwargs=positive_kwargs,
            negative_kwargs=None,
            buf_idx=state.step_index % 2,
            batch_size=batch_size,
            preposted_its=preposted_its,
            use_buffer=use_buffer,
        )
        
        pp_group = get_pp_group()
        if pp_group.world_size > 1 and pp_group.is_last_rank:
            state.latents = get_forward_context().stream_xt
        return noise_pred

    def predict_noise(
        self,
        intermediate_tensors: IntermediateTensors | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Single transformer forward. Returns ``IntermediateTensors`` on non-last
        PP stages; stacks the list-of-tensors output on the last stage."""
        # No autocast: DiT runs natively in bf16 (inputs/norms pre-cast).
        result = self.transformer(**kwargs, intermediate_tensors=intermediate_tensors)
        if isinstance(result, IntermediateTensors):
            return result
        return torch.stack(result, dim=0)
    
    def step_scheduler(self, state: DiffusionRequestState, noise_pred: torch.Tensor) -> None:
        ns = len(state.extra["slot_chunks"])
        stream_t = get_forward_context().stream_t
        t = stream_t[:, 0] if stream_t is not None else None

        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            t,
            state.latents,
            do_true_cfg=False,
            per_request_scheduler=[state.scheduler] * ns,
            generator=state.extra["seed_g"],
            batch_size=ns,
            receive_latents=False,
            use_buffer=True,
        )
        state.step_index += 1

    def decode_chunks(self, state: DiffusionRequestState) -> DiffusionOutput | None:
        """Decode the chunk at the deepest ladder slot (now x0); merge when done."""
        if state.extra["slot_chunks"][-1] is not None:
            lat = state.latents[-1:]
            # lat [1, 16, F, lat_h, lat_w] -> [16, F, lat_h, lat_w]
            pred = lat.transpose(0, 1).reshape(lat.shape[1], lat.shape[0] * lat.shape[2], lat.shape[3], lat.shape[4])
            video = self.vae.stream_decode(pred)  # [C, T_pix, lat_h*8, lat_w*8]
            state.extra.setdefault("decoded_chunks", []).append(DiffusionOutput(output=video))
            state.extra["num_chunks_decoded"] = state.extra.get("num_chunks_decoded", 0) + 1
    
            if state.extra.get("num_chunks_decoded", 0) >= state.sampling.num_chunks:
                self.state.advance(state.extra["new_lat_f"])
                return self._merge_chunk_outputs(state.extra["decoded_chunks"])
        
        return None

    @staticmethod
    def _merge_chunk_outputs(chunks: list[DiffusionOutput]) -> DiffusionOutput:
        # Concatenate decoded chunks along the temporal axis (5D Wan: dim 2; 4D: dim 1).
        try:
            outputs = [c.output for c in chunks]
            merged = torch.cat(outputs, dim=outputs[0].dim() - 3)
        except Exception as e:
            return DiffusionOutput(error=f"Failed to merge {len(chunks)} chunk outputs: {e}")
        return DiffusionOutput(output=merged)

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
    
    def set_pp_recv_dict_buffers(self, state: DiffusionRequestState) -> None:
        """Register PP recv buffers (latents on first rank, intermediates on others)."""

        if get_pipeline_parallel_world_size() == 1:
            return

        pp_group = get_pp_group()
        chunk_latent_frames = state.extra["chunk_latent_frames"]
        lat_h = state.extra["lat_h"]
        lat_w = state.extra["lat_w"]
        frame_seqlen = state.extra["frame_seqlen"]
        batch_size = state.total_steps

        latents_dtype = torch.float32
        it_dtype = self.target_dtype
        max_seq_len = chunk_latent_frames * frame_seqlen  # tokens per chunk

        latents_template = {
            "latents": torch.empty(batch_size, 16, chunk_latent_frames, lat_h, lat_w, dtype=latents_dtype, device="meta")
        }
        it_template = {
            "hidden_states": torch.empty(batch_size, max_seq_len, self.transformer.dim, dtype=it_dtype, device="meta"),
            "t": torch.empty(batch_size, chunk_latent_frames, dtype=state.timesteps.dtype, device="meta"),
            "xt": torch.empty(batch_size, 16, chunk_latent_frames, lat_h, lat_w, dtype=latents_dtype, device="meta"),
        }
        pp_group.set_recv_dict_buffer("latents", -1, latents_template, batch_size=batch_size)
        pp_group.set_recv_dict_buffer("intermediate", 0, it_template, batch_size=batch_size)
        
    def rebalance_blocks(self, local_dit_ns: float, local_vae_ns: float) -> None:
        pp_group = get_pp_group()
        pp_world = pp_group.world_size
        n_layers = self.transformer.num_layers
        local = torch.tensor(
            [local_dit_ns, local_vae_ns, self.transformer.start_layer, self.transformer.end_layer],
            dtype=torch.float64,
            device=self.device,
        )
        gathered = [torch.zeros_like(local) for _ in range(pp_world)]
        torch.distributed.all_gather(gathered, local, group=pp_group.device_group)
        dits = [float(g[0]) for g in gathered]
        vaes = [float(g[1]) for g in gathered]
        if sum(dits) <= 0:
            return
        old_intervals = [(int(g[2]), int(g[3])) for g in gathered]
        per_block = sum(dits) / n_layers
        counts = balance_layers_by_cost(n_layers, [v / per_block for v in vaes])
        new_starts = [sum(counts[:r]) for r in range(pp_world)]
        new_intervals = [(new_starts[r], new_starts[r] + counts[r]) for r in range(pp_world)]
        if new_intervals == old_intervals:
            return
        self._migrate_kv_cache(old_intervals, new_intervals)
        self.transformer.apply_pp_split(counts, free_blocks=False)
        logger.info("[RANK %d]Dynamic block schedule -> %s (dit_ms=%s vae_ms=%s)", pp_group.rank_in_group, counts, [d / 1e6 for d in dits], [v / 1e6 for v in vaes])

    def _migrate_kv_cache(self, old_intervals: list[tuple[int, int]], new_intervals: list[tuple[int, int]]) -> None:
        # Broadcast each block whose owner changed from its old owner to all ranks, so the new owner gets the live rolling KV.
        pp_group = get_pp_group()
        st = self.state
        my = pp_group.rank_in_group

        def owner(intervals: list[tuple[int, int]], i: int) -> int:
            for r, (s, e) in enumerate(intervals):
                if s <= i < e:
                    return r
            return -1

        for i in range(self.transformer.num_layers):
            src, dst = owner(old_intervals, i), owner(new_intervals, i)
            if src == dst:
                continue
            kv = st.kv_cache[i]
            if kv.device.type == "cpu":
                kv = kv.to(self.device)
                st.kv_cache[i] = kv
            torch.distributed.broadcast(kv, src=pp_group.ranks[src], group=pp_group.device_group)
            objs = (
                [st.local_end_index[i], st.global_end_index[i], st.evict_queues[i]]
                if my == src
                else [None, None, None]
            )
            torch.distributed.broadcast_object_list(objs, src=pp_group.ranks[src], group=pp_group.device_group)
            if my == dst:
                st.local_end_index[i], st.global_end_index[i], st.evict_queues[i] = objs
                st.crossattn_cache[i] = {"is_init": False, "k": None, "v": None}

        my_start, my_end = new_intervals[my]
        for i in range(self.transformer.num_layers):
            owned = my_start <= i < my_end
            kv = st.kv_cache[i]
            if owned and kv.device.type == "cpu":
                st.kv_cache[i] = kv.to(self.device)
            elif not owned and kv.device.type != "cpu":
                st.kv_cache[i] = kv.cpu()


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
