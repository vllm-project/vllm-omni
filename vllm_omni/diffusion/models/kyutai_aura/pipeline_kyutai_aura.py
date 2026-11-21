# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kyutai AURA: one portrait image + speech -> a talking-head video, generated chunk by chunk.

Checkpoints: kyutai/AURA-1.3B (CC BY-NC 4.0, non-commercial). The DiT comes from the AURA repo
(standalone merged weights); the VAE, the umT5 encoder and its tokenizer from the original Wan
base repo; the speech features from Kyutai's Mimi codec.

Each chunk is ``frames_per_chunk`` latent frames (16 video frames, 13 for the first chunk),
sampled in 4 steps without CFG over a request-owned rolling K/V history, then decoded by the
causal VAE, which carries its own cache from chunk to chunk. With ``streaming_output`` the chunks
are emitted as soon as they are decoded.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torchvision.transforms.functional as TF
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioInput, SupportImageInput
from vllm_omni.diffusion.models.kyutai_aura.audio import MIMI_REPO, MimiSemanticFeatures, load_speech, to_video_rate
from vllm_omni.diffusion.models.kyutai_aura.config import (
    DEFAULT_PROMPT,
    closest_bucket,
    resolve_kyutai_aura_config,
    unsupported_kyutai_aura,
)
from vllm_omni.diffusion.models.kyutai_aura.kv_cache import RollingChunkKVCache
from vllm_omni.diffusion.models.kyutai_aura.sampling import StudentSigmas, decode_spans
from vllm_omni.diffusion.models.kyutai_aura.transformer import KyutaiAuraTransformer
from vllm_omni.diffusion.request_streaming import get_emit_chunk
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.transformers_utils.repo_utils import hf_api

logger = init_logger(__name__)

UMT5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
UMT5_TOKENIZER = "google/umt5-xxl"


def _local_dir(repo_or_dir: str, allow_patterns: list[str]) -> Path:
    path = Path(repo_or_dir)
    if path.is_dir():
        return path
    return Path(hf_api().snapshot_download(repo_or_dir, allow_patterns=allow_patterns))


def get_kyutai_aura_post_process_func(od_config: OmniDiffusionConfig) -> Callable[..., Any]:
    """Post-process function: the video as frames plus the speech track, its rate and the fps."""
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(output: Any, output_type: str = "np") -> Any:
        """(video, waveform, sample_rate, fps) -> {video, audio, audio_sample_rate, fps}."""
        # output is (video [1, 3, T, H, W] in [-1, 1], speech waveform, sample rate, fps)
        if not (isinstance(output, tuple) and len(output) == 4):
            return output
        video, waveform, sample_rate, fps = output
        if output_type == "latent":
            return video
        return {
            "video": video_processor.postprocess_video(video, output_type=output_type),
            "audio": waveform,
            "audio_sample_rate": sample_rate,
            "fps": fps,
        }

    return post_process_func


class KyutaiAuraPipeline(nn.Module, SupportImageInput, SupportAudioInput):
    """Kyutai AURA on the default diffusion engine: one portrait and a speech recording per request;
    the video is generated chunk by chunk and returned whole or emitted per chunk when streaming."""

    support_image_input: ClassVar[bool] = True
    support_audio_input: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        reason = unsupported_kyutai_aura(od_config.model)
        if reason is not None:
            raise NotImplementedError(reason)
        cfg = resolve_kyutai_aura_config(od_config.model)
        if cfg is None:
            raise ValueError(f"{od_config.model} is not a released Kyutai AURA checkpoint")
        self.cfg = cfg
        self.device = get_local_device()
        extra: dict[str, Any] = getattr(od_config, "model_config", None) or {}

        aura_dir = _local_dir(od_config.model, [cfg.checkpoint])
        base_dir = _local_dir(extra.get("base_dir", cfg.base_repo), [cfg.vae_file, UMT5_FILE, f"{UMT5_TOKENIZER}/*"])
        mimi_dir = _local_dir(extra.get("mimi_dir", MIMI_REPO), ["*.json", "*.safetensors"])

        self.text_encoder, self.tokenizer = self._load_text_encoder(base_dir)
        # fp32 VAE weights; the VAE computes under autocast in ``vae_compute_dtype``.
        self.vae = DistributedAutoencoderKLWan.from_single_file(str(base_dir / cfg.vae_file), torch_dtype=torch.float32)
        self.vae.init_distributed()
        self.vae = self.vae.to(self.device).eval().requires_grad_(False)
        self.vae_compute_dtype = torch.float32 if cfg.vae_fp32 else torch.bfloat16
        self.audio_features = MimiSemanticFeatures(str(mimi_dir), self.device)

        with set_default_torch_dtype(torch.float32):  # fp32 master weights, bf16 compute
            self.transformer = KyutaiAuraTransformer(cfg)
        self.transformer.load_checkpoint(load_file(str(aura_dir / cfg.checkpoint)).items())
        self.transformer = self.transformer.to(self.device).eval().requires_grad_(False)
        self.sigma_of = StudentSigmas(cfg.schedule_steps, cfg.schedule_shift)

    def _load_text_encoder(self, base_dir: Path) -> tuple[nn.Module, Any]:
        from transformers import AutoTokenizer, UMT5EncoderModel

        from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v import _WAN_UMT5_CONFIG, _load_wan_t5_as_umt5

        tokenizer = AutoTokenizer.from_pretrained(str(base_dir / UMT5_TOKENIZER))
        with torch.device("meta"):
            encoder = UMT5EncoderModel(_WAN_UMT5_CONFIG)
        encoder = _load_wan_t5_as_umt5(encoder, str(base_dir / UMT5_FILE), dtype=torch.bfloat16)
        return encoder.to(self.device).eval().requires_grad_(False), tokenizer

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """All weights are loaded in __init__ (the standalone DiT file strictly, the Wan base files and
        Mimi by their own loaders), so every parameter is reported as loaded for the loader's check."""
        for _ in weights:  # no weights_sources: nothing is streamed here
            pass
        return {name for name, _ in self.named_parameters()}

    # ----------------------------------------------------------------------------------- inputs
    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """umT5 context of the prompt's real tokens: [L, 4096] in bf16."""
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        tokens = self.tokenizer(
            [prompt_clean(prompt)],
            padding="max_length",
            max_length=self.cfg.text_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device)
        hidden = self.text_encoder(ids, mask).last_hidden_state
        return hidden[0, : int(mask.gt(0).sum())]

    def load_portrait(self, image: str | Path | Image.Image) -> torch.Tensor:
        """The portrait at the closest training resolution: [1, 3, 1, H, W] in [-1, 1]."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        height, width = closest_bucket(self.cfg.buckets, image.height, image.width)
        resized = TF.resize(image.convert("RGB"), min(height, width))  # shorter side to the bucket's shorter side
        pixels = TF.to_tensor(TF.center_crop(resized, [height, width]))  # [3, H, W] in [0, 1]
        return pixels.mul(2).sub(1).unsqueeze(0).unsqueeze(2).to(self.device)

    def _latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Latent mean and inverse std in the VAE compute dtype, as the checkpoints were trained with
        (the normalization runs at that precision)."""
        shape, dtype = (1, -1, 1, 1, 1), self.vae_compute_dtype
        mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=dtype).view(shape)
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=self.device, dtype=dtype).view(shape)
        return mean, inv_std

    @torch.no_grad()
    def encode_reference(self, portrait: torch.Tensor) -> torch.Tensor:
        """Normalized posterior mean of the portrait: [1, C, 1, h, w] in fp32."""
        mean, inv_std = self._latent_stats()
        with torch.autocast("cuda", dtype=self.vae_compute_dtype):
            latent = self.vae.encode(portrait.to(self.vae_compute_dtype)).latent_dist.mode()
            return ((latent - mean) * inv_std).float()

    @torch.no_grad()
    def decode_chunk(self, latents: torch.Tensor, feat_cache: list[Any], first: bool) -> torch.Tensor:
        """Decode one chunk [C, F, h, w] with the causal VAE, continuing from the previous chunk
        through ``feat_cache`` (see ``decode_spans``). Returns [3, T, H, W] in [-1, 1]."""
        mean, inv_std = self._latent_stats()
        with torch.autocast("cuda", dtype=self.vae_compute_dtype):
            z = self.vae.post_quant_conv(latents.unsqueeze(0) / inv_std + mean)
            frames = []
            for start, end in decode_spans(z.shape[2], self.cfg.vae_decode_group, first):
                frames.append(self.vae.decoder(z[:, :, start:end], feat_cache=feat_cache, feat_idx=[0]))
            video = torch.cat(frames, dim=2).float().clamp_(-1, 1)
        return video[0]

    # ------------------------------------------------------------------------------- generation
    def chunk_generator(self, seed: int, chunk: int) -> torch.Generator:
        """Each chunk draws its noise from its own generator, seeded ``seed + chunk``."""
        return torch.Generator(self.device).manual_seed(seed + chunk)

    def _sample_chunk(
        self,
        generator: torch.Generator,
        latent_shape: tuple[int, ...],
        context: torch.Tensor,
        ref_latent: torch.Tensor,
        speech: tuple[torch.Tensor, torch.Tensor],
        history: RollingChunkKVCache,
    ) -> torch.Tensor:
        """Few-step sampling of one chunk: at each student timestep, predict the clean latents
        (x0 = x - sigma * v, in float64) and re-noise them to the next timestep's sigma with fresh
        noise from the chunk's generator. Returns x0 of the last step, [C, F, h, w] in fp32."""
        timesteps = self.cfg.timesteps
        x = randn_tensor(latent_shape, generator=generator, device=self.device, dtype=torch.bfloat16)
        for step, t in enumerate(timesteps):
            velocity = self.transformer(
                x, torch.tensor([t], dtype=torch.long, device=self.device), context, ref_latent, speech, history
            )
            x0 = (x.double() - self.sigma_of(t) * velocity.double()).float()
            if step + 1 == len(timesteps):
                break
            next_sigma = self.sigma_of(timesteps[step + 1])
            noise = randn_tensor(x0.shape, generator=generator, device=self.device, dtype=x0.dtype)
            x = (1 - next_sigma) * x0 + next_sigma * noise
        return x0

    @torch.no_grad()
    def generate_chunks(
        self,
        image: str | Path | Image.Image,
        audio: Any,
        prompt: str = DEFAULT_PROMPT,
        seed: int = 42,
        num_chunks: int | None = None,
    ) -> Iterator[tuple[int, int, torch.Tensor]]:
        """Yield (chunk index, number of chunks, video [3, frames, H, W] in [-1, 1]) per chunk.
        The video covers the whole speech, rounded up to a whole chunk, or ``num_chunks`` chunks."""
        cfg = self.cfg
        frames_per_chunk = cfg.video_frames_per_chunk
        speech_features, speech_chunks = to_video_rate(self.audio_features(audio), cfg.fps, frames_per_chunk)
        speech_features = speech_features.to(device=self.device, dtype=torch.bfloat16)
        total = speech_chunks if num_chunks is None else min(int(num_chunks), speech_chunks)

        ref_latent = self.encode_reference(self.load_portrait(image))
        context = self.encode_prompt(prompt)

        latent_h, latent_w = ref_latent.shape[-2:]
        latent_shape = (cfg.latent_channels, cfg.frames_per_chunk, latent_h, latent_w)
        tokens_per_latent_frame = (latent_h // cfg.patch_size[1]) * (latent_w // cfg.patch_size[2])
        history = RollingChunkKVCache.for_config(cfg, cfg.frames_per_chunk * tokens_per_latent_frame, self.device)
        vae_cache: list[Any] = [None] * self.vae._cached_conv_counts["decoder"]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            for chunk in range(total):
                generator = self.chunk_generator(seed, chunk)
                chunk_features = speech_features[..., chunk * frames_per_chunk : (chunk + 1) * frames_per_chunk]
                speech = self.transformer.encode_audio(chunk_features)
                x0 = self._sample_chunk(generator, latent_shape, context, ref_latent, speech, history)
                history.advance()
                yield chunk, total, self.decode_chunk(x0, vae_cache, first=chunk == 0)

    # ---------------------------------------------------------------------------------- serving
    def _request_inputs(self, req: DiffusionRequestBatch) -> tuple[str, Any, Any, int, int | None]:
        """Read and check one request before any chunk is generated; bad inputs are client errors."""
        prompt_entry = req.prompts[0] if req.prompts else {"prompt": ""}
        if isinstance(prompt_entry, str):
            raise OmniClientError("Kyutai AURA needs multi_modal_data with an 'image' and an 'audio'")
        text = prompt_entry.get("prompt") or DEFAULT_PROMPT
        media = prompt_entry.get("multi_modal_data") or {}
        image, audio = media.get("image"), media.get("audio")
        if image is None or audio is None:
            raise OmniClientError("Kyutai AURA needs multi_modal_data['image'] (a portrait) and ['audio'] (speech)")
        params = req.sampling_params
        # The speech is read and returned on the model's own clock, so frames written at another
        # rate would play faster or slower than the speech track.
        fps = params.resolved_frame_rate
        if fps is not None and fps != self.cfg.fps:
            raise OmniClientError(
                f"Kyutai AURA generates video at {self.cfg.fps} fps; omit fps or set it to {self.cfg.fps} (got {fps:g})"
            )
        num_chunks = (params.extra_args or {}).get("num_chunks")
        positive_int = isinstance(num_chunks, int) and not isinstance(num_chunks, bool) and num_chunks >= 1
        if num_chunks is not None and not positive_int:
            raise OmniClientError(f"extra_params.num_chunks must be a positive integer, got {num_chunks!r}")
        seed = params.seed if params.seed is not None else 42
        return text, image, audio, int(seed), num_chunks

    def _log_speed(self, delivered: list[float], sizes: list[int]) -> None:
        """One line per request: time to the first chunk on the CPU, then the steady frame rate
        after the second chunk (the playback rate is ``cfg.fps``)."""
        if not delivered:
            return
        steady = ""
        if len(delivered) >= 3 and delivered[-1] > delivered[1]:
            steady = f"; {sum(sizes[2:]) / (delivered[-1] - delivered[1]):.2f} fps after chunk 2"
        logger.info(
            "Kyutai AURA: %d frames in %d chunks; first chunk %.2f s%s (playback %d fps)",
            sum(sizes),
            len(sizes),
            delivered[0],
            steady,
            self.cfg.fps,
        )

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Generate the whole video of one request, or emit it chunk by chunk with ``streaming_output``."""
        if req.is_dummy_run():
            return [DiffusionOutput(output=torch.zeros(1, 3, 1, 8, 8))]
        text, image, audio, seed, num_chunks = self._request_inputs(req)
        emit = get_emit_chunk() if getattr(self.od_config, "streaming_output", False) is True else None
        chunks: list[torch.Tensor] = []
        delivered: list[float] = []
        total = 0
        start = time.perf_counter()
        for index, total, video in self.generate_chunks(image, audio, text, seed=seed, num_chunks=num_chunks):
            if emit is not None:
                emit(DiffusionOutput(finished=False, chunk_index=index, total_chunks=total, output=video[None].half()))
            else:
                chunks.append(video.cpu())
                delivered.append(time.perf_counter() - start)
        if emit is not None:
            return [DiffusionOutput(finished=True, chunk_index=max(total - 1, 0), total_chunks=total)]
        self._log_speed(delivered, [c.shape[1] for c in chunks])
        video = torch.cat(chunks, dim=1)[None]
        sample_rate = self.audio_features.sample_rate
        waveform = load_speech(audio, sample_rate)
        waveform = waveform[: int(round(video.shape[2] / self.cfg.fps * sample_rate))]
        return [DiffusionOutput(output=(video, np.asarray(waveform), sample_rate, self.cfg.fps))]
