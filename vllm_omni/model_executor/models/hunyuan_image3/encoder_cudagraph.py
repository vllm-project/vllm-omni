# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import defaultdict
from typing import Any

import torch
from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager
from vllm.v1.worker.encoder_cudagraph_defs import (
    ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG,
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from .autoencoder_kl_3d import DiagonalGaussianDistribution


def _pad_cu_seqlens(buffer: torch.Tensor, source: torch.Tensor) -> None:
    capacity = buffer[-1].clone()
    buffer.copy_(source[-1].expand_as(buffer))
    buffer[: source.numel()].copy_(source)
    buffer[-1].copy_(capacity)


class _Encoder(SupportsEncoderCudaGraph):
    def __init__(self, model):
        self.model = model

    def get_input_modality(self, mm_kwargs):
        return "image"

    def get_max_frames_per_video(self):
        return 1

    def get_encoder_cudagraph_budget_range(self, vllm_config):
        return self.min_tokens, max(self.min_tokens, vllm_config.scheduler_config.max_num_batched_tokens)

    def select_encoder_cudagraph_items(self, mm_kwargs, indices):
        return {key: value[indices] for key, value in mm_kwargs.items()}


class _VisionEncoder(_Encoder):
    def __init__(self, model):
        super().__init__(model)
        self.min_tokens = model.config.vit_processor["max_num_patches"]

    def get_encoder_cudagraph_config(self):
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=["pixels", "positions", "cu_seqlens", "indices", "max_seqlen"],
            out_hidden_size=self.model.config.hidden_size,
            padding_logics={"cu_seqlens": _pad_cu_seqlens},
        )

    def get_encoder_cudagraph_item_specs(self, mm_kwargs):
        batch, patches = mm_kwargs["mask"].shape
        return [EncoderItemSpec(patches, patches) for _ in range(batch)]

    def prepare_encoder_cudagraph_capture_inputs(
        self, token_budget, max_batch_size, max_frames_per_batch, device, dtype, path="default", axis_keys=None
    ):
        model = self.model.vision_model
        patches = self.min_tokens
        batch = min(max_batch_size, max(1, token_budget // patches))
        capacity = batch * patches
        return EncoderCudaGraphCaptureInputs(
            values={
                "pixels": torch.zeros(
                    capacity, model.embeddings.patch_embedding.in_features, device=device, dtype=dtype
                ),
                "positions": torch.zeros(capacity, model.embed_dim, device=device, dtype=dtype),
                "cu_seqlens": torch.tensor([0] * (batch + 1) + [capacity], device=device, dtype=torch.int32),
                "indices": torch.zeros(batch, patches, device=device, dtype=torch.long),
                "max_seqlen": torch.tensor(capacity, dtype=torch.int32),
            }
        )

    def prepare_encoder_cudagraph_replay_buffers(self, mm_kwargs, max_batch_size, max_frames_per_batch, path="default"):
        pixels, mask, shapes = (mm_kwargs[key] for key in ("pixels", "mask", "shapes"))
        mask = mask.bool()
        lengths = shapes.prod(-1).to(device=pixels.device, dtype=torch.int32)
        cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)])
        indices = mask.flatten().long().cumsum(0).reshape_as(mask) * mask
        return EncoderCudaGraphReplayBuffers(
            values={
                "pixels": pixels[mask].to(self.model.vision_model.embeddings.patch_embedding.weight.dtype),
                "positions": self.model.vision_model.embeddings.interpolate_pos_encoding(shapes),
                "cu_seqlens": cu_seqlens,
                "indices": indices,
            }
        )

    def encoder_cudagraph_forward(self, inputs, path="default"):
        hidden = self.model.vision_model.forward_packed(
            inputs["pixels"], inputs["positions"], inputs["cu_seqlens"], inputs["max_seqlen"]
        )
        hidden = torch.cat([hidden.new_zeros(1, hidden.shape[-1]), hidden])
        padded = hidden[inputs["indices"]]
        return self.model.vision_aligner(padded).flatten(0, 1)

    def encoder_eager_forward(self, mm_kwargs, path="default"):
        hidden = self.model.vision_model(mm_kwargs["pixels"], mm_kwargs["mask"], mm_kwargs["shapes"])
        return self.model.vision_aligner(hidden).flatten(0, 1)


class _VAEEncoder(_Encoder):
    def __init__(self, model):
        super().__init__(model)
        from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
            HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS,
        )

        from .hunyuan_image3 import HunyuanImage3Processor

        group = HunyuanImage3Processor.ResolutionGroup(
            base_size=model.config.image_base_size,
            extra_resolutions=[HunyuanImage3Processor.Resolution(s) for s in HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS],
        )
        self.resolutions = tuple((r.height, r.width) for r in group.data)
        self.factor = model.vae.ffactor_spatial
        self.min_tokens = min(h * w // self.factor**2 for h, w in self.resolutions)

    def get_encoder_cudagraph_config(self):
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=["pixels"],
            out_hidden_size=2 * self.model.vae.config.latent_channels,
            capture_axes=(self.resolutions,),
        )

    def get_encoder_cudagraph_item_specs(self, mm_kwargs):
        pixels = mm_kwargs["pixels"]
        tokens = pixels.shape[-2] * pixels.shape[-1] // self.factor**2
        return [EncoderItemSpec(tokens, tokens) for _ in pixels]

    def select_encoder_cudagraph_items(self, mm_kwargs, indices):
        selected = super().select_encoder_cudagraph_items(mm_kwargs, indices)
        selected[ENCODER_CUDAGRAPH_AXIS_KEYS_KWARG] = (tuple(mm_kwargs["pixels"].shape[-2:]),)
        return selected

    def prepare_encoder_cudagraph_capture_inputs(
        self, token_budget, max_batch_size, max_frames_per_batch, device, dtype, path="default", axis_keys=None
    ):
        height, width = axis_keys[0]
        tokens = height * width // self.factor**2
        batch = min(max_batch_size, max(1, token_budget // tokens))
        return EncoderCudaGraphCaptureInputs(
            values={"pixels": torch.zeros(batch, 3, height, width, device=device, dtype=self.model.vae.dtype)}
        )

    def prepare_encoder_cudagraph_replay_buffers(self, mm_kwargs, max_batch_size, max_frames_per_batch, path="default"):
        return EncoderCudaGraphReplayBuffers(values={"pixels": mm_kwargs["pixels"].to(self.model.vae.dtype)})

    def encoder_cudagraph_forward(self, inputs, path="default"):
        # Preserve the eager path's per-image convolutions, including VAE tiling.
        parameters = [self.model.vae.encode(image.unsqueeze(0)).latent_dist.parameters for image in inputs["pixels"]]
        return torch.cat([p.movedim(1, -1).reshape(-1, p.shape[1]) for p in parameters])

    def encoder_eager_forward(self, mm_kwargs, path="default"):
        return self.encoder_cudagraph_forward({"pixels": mm_kwargs["pixels"].to(self.model.vae.dtype)})


class HunyuanImage3EncoderCudaGraphManager:
    """Use upstream capture/replay independently for the two encoder layouts."""

    def __init__(self, vllm_config, device, dtype, model):
        self.model = model
        self.vision = EncoderCudaGraphManager(vllm_config, device, dtype, _VisionEncoder(model))
        self.vae = EncoderCudaGraphManager(vllm_config, device, dtype, _VAEEncoder(model))
        self.token_budgets = sorted(set(self.vision.token_budgets + self.vae.token_budgets))

    def supports_modality(self, modality):
        return modality == "image"

    def get_num_graphs_to_capture(self):
        return self.vision.get_num_graphs_to_capture() + self.vae.get_num_graphs_to_capture()

    def capture(self, graph_pool):
        self.vision.capture(graph_pool)
        self.vae.capture(graph_pool)

    def clear(self):
        self.vision.clear()
        self.vae.clear()

    def is_captured(self):
        return self.vision.is_captured() and self.vae.is_captured()

    def get_cumulative_stats(self):
        encoders = {"vision": self.vision.get_cumulative_stats(), "vae": self.vae.get_cumulative_stats()}
        hits = sum(s["graph_hits"] for s in encoders.values())
        misses = sum(s["graph_misses"] for s in encoders.values())
        return {
            "graph_hits": hits,
            "graph_misses": misses,
            "hit_rate": hits / max(1, hits + misses),
            "encoders": encoders,
        }

    @torch.inference_mode()
    def execute(self, mm_kwargs: dict[str, Any]) -> list[torch.Tensor]:
        parsed = self.model._parse_and_validate_image_input(**mm_kwargs)
        if parsed is None:
            return []
        images = parsed["pixel_values"]
        vision_inputs = {
            "pixels": images["vit_pixel_values"],
            "mask": images["vit_pixel_attention_mask"],
            "shapes": images["vit_spatial_shapes"],
        }
        if vision_inputs["pixels"].shape[1] == self.vision.model.min_tokens:
            vision = torch.stack(self.vision.execute(vision_inputs))
        else:
            self.vision.graph_misses += len(vision_inputs["pixels"])
            vision = self.model._vit_encode(*vision_inputs.values())

        vae_images = images["vae_pixel_values"]
        groups = defaultdict(list)
        for i, pixels in enumerate(vae_images):
            groups[tuple(pixels.shape[-2:])].append(i)
        posteriors = [None] * len(vae_images)
        factor = self.vae.model.factor
        for (height, width), indices in groups.items():
            inputs = {"pixels": torch.stack([vae_images[i] for i in indices])}
            if (height, width) in self.vae.model.resolutions:
                outputs = self.vae.execute(inputs)
            else:
                self.vae.graph_misses += len(indices)
                raw = self.vae.model.encoder_eager_forward(inputs)
                outputs = raw.split(height * width // factor**2)
            for i, output in zip(indices, outputs):
                posteriors[i] = output.T.reshape(1, -1, 1, height // factor, width // factor)

        # Sampling stays in request order, after all deterministic graph replays.
        seed = images.get("vae_generator_seed")
        generator = None
        if seed is not None and seed.numel():
            generator = torch.Generator(device=vae_images[0].device).manual_seed(int(seed.reshape(-1)[0].item()))
        vae_tokens = []
        for parameters in posteriors:
            t, latents = self.model._sample_vae_latents(DiagonalGaussianDistribution(parameters), generator=generator)
            tokens, _, _ = self.model.patch_embed(latents, self.model.time_embed(t[0]))
            vae_tokens.append(tokens)
        return self.model._combine_image_embeddings(vision, vae_tokens)
