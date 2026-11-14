# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar, cast

import torch
import torch.distributed as dist
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import patchify, unpatchify
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing_extensions import Self
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders import wan_encode_split, wan_spatial_shard
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_sdpa_alignment import (
    install_wan_attention_sdpa_alignment,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    can_encode_frames,
    decode_frames,
    encode_frames,
    is_encoder_installed,
    is_installed,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath._utils import suggests_channels_last
from vllm_omni.diffusion.models.interface import DecodedChunkConsumer
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_GFX11_AOTRITON_MAX_HEAD_DIM = 256


def _wan_attention_head_dims(model: torch.nn.Module) -> Iterator[int]:
    for module in model.modules():
        if module.__class__.__name__ != "WanAttentionBlock":
            continue
        qkv = getattr(module, "to_qkv", None)
        out_channels = getattr(qkv, "out_channels", None)
        if isinstance(out_channels, int) and out_channels % 3 == 0:
            yield out_channels // 3


def _requires_gfx11_math_sdpa(
    model: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    if dtype not in (torch.float16, torch.bfloat16) or device.type != "cuda":
        return False
    if not current_omni_platform.is_rocm():
        return False

    capability = current_omni_platform.get_device_capability(device.index or 0)
    if capability is None or capability[0] != 11:
        return False

    return any(head_dim > _GFX11_AOTRITON_MAX_HEAD_DIM for head_dim in _wan_attention_head_dims(model))


def _fix_wan22_residual_vae_keys(
    source: dict[str, torch.Tensor], converted: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Restore Wan2.2 residual block nesting lost by Diffusers' converter."""
    converted = {
        key: value
        for key, value in converted.items()
        if not key.startswith(("encoder.down_blocks.", "decoder.up_blocks.", "decoder.upsamples."))
    }
    tail_replacements = {
        "residual.0.": "norm1.",
        "residual.2.": "conv1.",
        "residual.3.": "norm2.",
        "residual.6.": "conv2.",
        "shortcut.": "conv_shortcut.",
    }

    for key, value in source.items():
        parts = key.split(".")
        if len(parts) < 6:
            continue
        if parts[:2] == ["encoder", "downsamples"] and parts[3] == "downsamples":
            block, layer, tail = parts[2], int(parts[4]), ".".join(parts[5:])
            component = f"resnets.{layer}" if layer < 2 else "downsampler"
            prefix = f"encoder.down_blocks.{block}.{component}."
        elif parts[:2] == ["decoder", "upsamples"] and parts[3] == "upsamples":
            block, layer, tail = parts[2], int(parts[4]), ".".join(parts[5:])
            component = f"resnets.{layer}" if layer < 3 else "upsampler"
            prefix = f"decoder.up_blocks.{block}.{component}."
        else:
            continue
        for old, new in tail_replacements.items():
            tail = tail.replace(old, new)
        converted[prefix + tail] = value

    return converted


class OmniAutoencoderKLWan(AutoencoderKLWan):
    @classmethod
    def from_wan22_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> Self:
        """Load an original Wan2.2 TI2V VAE using the shared architecture."""
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

        # Wan2.2 TI2V patchifies RGB into 12 pixel-space channels (3 * 2 * 2),
        # while its actual latent width is z_dim=48. Do not substitute the
        # 16-channel Wan2.1 VAE here.
        vae_config = {
            "_class_name": "AutoencoderKLWan",
            "base_dim": 160,
            "decoder_base_dim": 256,
            "z_dim": 48,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_downsample": [False, True, True],
            "dropout": 0.0,
            "in_channels": 12,
            "out_channels": 12,
            "patch_size": 2,
            "scale_factor_temporal": 4,
            "scale_factor_spatial": 16,
            "is_residual": True,
            "latents_mean": [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
            "latents_std": [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ],
        }
        converted = _fix_wan22_residual_vae_keys(state_dict, convert_wan_vae_to_diffusers(state_dict))
        with torch.device("meta"):
            vae = cast(Self, cls.from_config(vae_config))
        vae.load_state_dict(converted, strict=True, assign=True)
        return vae

    @contextmanager
    def _execution_context(self) -> Iterator[None]:
        try:
            first_param = next(self.parameters())
        except StopIteration:
            yield
            return

        dtype = first_param.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            yield
            return

        autocast_context = current_omni_platform.create_autocast_context(
            device_type=first_param.device.type,
            dtype=dtype,
            enabled=True,
        )
        force_math_sdpa = _requires_gfx11_math_sdpa(
            self,
            device=first_param.device,
            dtype=dtype,
        )
        with autocast_context:
            if not force_math_sdpa:
                yield
                return

            # Remove this guard only after the supported ROCm image contains
            # pytorch/pytorch#194100 and #194280 and W7900 validates issue #7484.
            logger.warning_once(
                "Wan VAE is forcing math SDPA because gfx11 AOTriton does not "
                "support attention head dimensions above %d.",
                _GFX11_AOTRITON_MAX_HEAD_DIM,
            )
            with sdpa_kernel(SDPBackend.MATH):
                yield

    def _install_attention_alignment(self) -> None:
        """Realign 1x1 Wan attention inputs before the first encode or decode."""
        install_wan_attention_sdpa_alignment(self)

    def encode(self, x: torch.Tensor, return_dict: bool = True):
        with self._execution_context():
            return super().encode(x, return_dict=return_dict)

    def _encode(self, x: torch.Tensor):
        # Keep the parent's tiling dispatch and patchified tile coordinates.
        # Only the untiled, supported inference schedule uses preallocation.
        self._install_attention_alignment()
        if not is_encoder_installed(self):
            return super()._encode(x)
        if can_encode_frames(self, x):
            tiled = self.use_tiling and (
                x.shape[-2] > self.tile_sample_min_height or x.shape[-1] > self.tile_sample_min_width
            )
            if not tiled:
                return encode_frames(self, x)
        try:
            return super()._encode(x)
        finally:
            self.clear_cache()

    def decode(self, z: torch.Tensor, return_dict: bool = True):
        """Decode a Wan latent using the Diffusers-compatible full-tensor API."""
        with self._execution_context():
            return super().decode(z, return_dict=return_dict)

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        """Decode one latent batch, preserving the parent Diffusers contract."""
        self._install_attention_alignment()
        _, _, _, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if getattr(self, "use_tiling", False) and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        return self._decode_temporal(z, return_dict=return_dict)

    # Chunks are published straight from the decoder, clamped to the
    # checkpoint's output range.
    chunk_value_range: ClassVar[tuple[float, float]] = (-1.0, 1.0)

    @apply_forward_hook
    def decode_with_chunks(self, z: torch.Tensor, *, on_chunk: DecodedChunkConsumer) -> None:
        """Decode ``z`` while synchronously delivering temporal chunks.

        The ordinary :meth:`decode` contract remains the Diffusers contract.
        Chunk callbacks deliberately reject a sliced batch: Diffusers slices
        each sample before decoding, while a streaming consumer needs a stable
        batch ownership contract. Callers can disable ``use_slicing`` or
        invoke this capability once per sample.

        Every rank participating in distributed VAE execution must invoke this
        method. Spatial-sharded execution invokes ``on_chunk`` only on the
        output-owning rank. Chunks have shape ``(B, C, T, H, W)`` and
        values clamped to ``[-1, 1]``. After a callback failure, remaining
        chunks are decoded and discarded before the exception is re-raised.
        """
        if not callable(on_chunk):
            raise TypeError("on_chunk must be callable")
        if getattr(self, "use_slicing", False) and z.shape[0] > 1:
            raise ValueError(
                "Wan chunked decode does not support use_slicing for batch size > 1; "
                "disable VAE slicing or decode one sample at a time"
            )

        with self._execution_context():
            self._decode_with_chunks(z, on_chunk=on_chunk)

    def _decode_temporal(
        self,
        z: torch.Tensor,
        *,
        on_chunk: DecodedChunkConsumer | None = None,
        return_dict: bool = True,
    ):
        """Decode the non-tiled temporal path, optionally streaming chunks."""
        if on_chunk is None and is_installed(self) and not torch.is_grad_enabled():
            try:
                out = decode_frames(self, z)
            finally:
                self.clear_cache()
            if not return_dict:
                return (out,)
            return DecoderOutput(sample=out)

        self.clear_cache()
        callback_error: BaseException | None = None
        try:
            x = self.post_quant_conv(z)
            decoded_chunks: list[torch.Tensor] = []
            for i in range(z.shape[2]):
                self._conv_idx = [0]
                if i == 0:
                    chunk = self.decoder(
                        x[:, :, i : i + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                        first_chunk=True,
                    )
                else:
                    chunk = self.decoder(
                        x[:, :, i : i + 1, :, :],
                        feat_cache=self._feat_map,
                        feat_idx=self._conv_idx,
                    )

                if on_chunk is None:
                    decoded_chunks.append(chunk)
                    continue

                if callback_error is None:
                    try:
                        if self.config.patch_size is not None:
                            chunk = unpatchify(chunk, patch_size=self.config.patch_size)
                        on_chunk(torch.clamp(chunk, min=-1.0, max=1.0))
                    except BaseException as exc:
                        # A previously installed spatial-shard decoder still
                        # runs collectives, even for latents below the tiling
                        # threshold. Drain before surfacing consumer failures.
                        callback_error = exc
                del chunk

            if callback_error is not None:
                raise callback_error
            if on_chunk is not None:
                return None

            out = torch.cat(decoded_chunks, dim=2)
            if self.config.patch_size is not None:
                out = unpatchify(out, patch_size=self.config.patch_size)
            out = torch.clamp(out, min=-1.0, max=1.0)
            if not return_dict:
                return (out,)
            return DecoderOutput(sample=out)
        finally:
            self.clear_cache()

    @staticmethod
    def _discard_decoded_chunk(_chunk: torch.Tensor) -> None:
        """Consume a non-owner rank's chunk without retaining its tensor."""

    def _decode_with_chunks(self, z: torch.Tensor, *, on_chunk: DecodedChunkConsumer) -> None:
        """Run the temporal streaming primitive without API or hook handling."""
        self._install_attention_alignment()
        distributed_enabled = False
        if hasattr(self, "is_distributed_enabled"):
            distributed_enabled = self.is_distributed_enabled()

        if getattr(self, "use_tiling", False):
            tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
            tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
            exceeds_tile = z.shape[-1] > tile_latent_min_width or z.shape[-2] > tile_latent_min_height
            if exceeds_tile:
                if distributed_enabled:
                    self._tiled_decode_with_chunks(z, on_chunk=on_chunk)
                    return
                raise ValueError("Wan chunk callbacks are unsupported for spatial tiling")

        callback = on_chunk
        if distributed_enabled and dist.get_rank(group=self.distributed_executor.group) != 0:
            callback = self._discard_decoded_chunk
        self._decode_temporal(z, on_chunk=callback)


class DistributedAutoencoderKLWan(OmniAutoencoderKLWan, DistributedVaeMixin):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def retain_stage_components(self, *, encode: bool) -> None:
        """Prune the unused Wan half before device placement/offload discovery.

        Diffusers 0.40 caches encoder/decoder convolution counts separately;
        both encode and decode call clear_cache, but neither executes the other
        half. Keep the public attributes and zero only the removed cache count.
        This reduces resident weights, NOT from_pretrained's peak host memory:
        the ordinary checkpoint loader still loads both halves first.
        """
        removed = "decoder" if encode else "encoder"
        projection = "post_quant_conv" if encode else "quant_conv"
        if not isinstance(getattr(self, "_cached_conv_counts", None), dict):
            raise RuntimeError("Wan stage VAE pruning requires Diffusers' cached encoder/decoder conv counts.")
        setattr(self, removed, None)
        setattr(self, projection, None)
        self._cached_conv_counts[removed] = 0
        self.clear_cache()

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_sample_stride_height = self.tile_sample_stride_height
        tile_sample_stride_width = self.tile_sample_stride_width
        if self.config.patch_size is not None:
            sample_height = sample_height // self.config.patch_size
            sample_width = sample_width // self.config.patch_size
            tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
            tile_sample_stride_width = tile_sample_stride_width // self.config.patch_size
            blend_height = self.tile_sample_min_height // self.config.patch_size - tile_sample_stride_height
            blend_width = self.tile_sample_min_width // self.config.patch_size - tile_sample_stride_width
        else:
            blend_height = self.tile_sample_min_height - tile_sample_stride_height
            blend_width = self.tile_sample_min_width - tile_sample_stride_width

        tiletask_list: list[TileTask] = []
        for i in range(0, height, tile_latent_stride_height):
            for j in range(0, width, tile_latent_stride_width):
                time_list = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                    time_list.append(tile)
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // tile_latent_stride_height, j // tile_latent_stride_width),
                        time_list,
                        workload=time_list[0].shape[3] * time_list[0].shape[4],
                    )
                )
        tile_spec = {
            "sample_height": sample_height,
            "sample_width": sample_width,
            "blend_height": blend_height,
            "blend_width": blend_width,
            "tile_sample_stride_height": tile_sample_stride_height,
            "tile_sample_stride_width": tile_sample_stride_width,
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        """Decode one tile, retaining raw values and patch layout for blending."""
        self.clear_cache()
        output = None
        offset = 0
        frames_per_chunk = 2 ** sum(bool(flag) for flag in self.decoder.temperal_upsample)
        channels_last = False
        try:
            with self._execution_context():
                for k in range(len(task.tensor)):
                    self._conv_idx = [0]
                    tile = self.post_quant_conv(task.tensor[k])
                    decoded = self.decoder(
                        tile, feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=(k == 0)
                    )
                    if output is None:
                        total_frames = decoded.shape[2] + (len(task.tensor) - 1) * frames_per_chunk
                        channels_last = suggests_channels_last(decoded)
                        output = torch.empty(
                            (*decoded.shape[:2], total_frames, *decoded.shape[3:]),
                            dtype=decoded.dtype,
                            device=decoded.device,
                            memory_format=torch.channels_last_3d if channels_last else torch.contiguous_format,
                        )
                    elif channels_last and not suggests_channels_last(decoded):
                        # Match cat's layout when chunks have different formats.
                        output = output.contiguous()
                        channels_last = False
                    if decoded.shape[:2] != output.shape[:2] or decoded.shape[3:] != output.shape[3:]:
                        raise RuntimeError("Wan tile decoder changed its non-temporal output shape")
                    if decoded.shape[2] < 1 or (k > 0 and decoded.shape[2] != frames_per_chunk):
                        raise RuntimeError("Wan tile decoder produced an unexpected temporal chunk length")
                    end = offset + decoded.shape[2]
                    output[:, :, offset:end].copy_(decoded)
                    offset = end
                    del decoded
        finally:
            # Keep causal history between temporal chunks, never between tiles.
            self.clear_cache()
        if output is None or offset != output.shape[2]:
            raise RuntimeError("Wan tile decoder produced no output or an incomplete temporal output")
        return output

    def encode_tile_split(self, x: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, num_frames, height, width = x.shape
        encode_spatial_compression_ratio = self.spatial_compression_ratio
        # Scale tile parameters for patchified coordinate system
        tile_sample_min_height = self.tile_sample_min_height
        tile_sample_min_width = self.tile_sample_min_width
        tile_sample_stride_height = self.tile_sample_stride_height
        tile_sample_stride_width = self.tile_sample_stride_width
        if self.config.patch_size is not None:
            assert encode_spatial_compression_ratio % self.config.patch_size == 0
            encode_spatial_compression_ratio = self.spatial_compression_ratio // self.config.patch_size
            # When input is patchified, scale tile parameters accordingly
            tile_sample_min_height = tile_sample_min_height // self.config.patch_size
            tile_sample_min_width = tile_sample_min_width // self.config.patch_size
            tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
            tile_sample_stride_width = tile_sample_stride_width // self.config.patch_size

        latent_height = height // encode_spatial_compression_ratio
        latent_width = width // encode_spatial_compression_ratio

        tile_latent_min_height = tile_sample_min_height // encode_spatial_compression_ratio
        tile_latent_min_width = tile_sample_min_width // encode_spatial_compression_ratio
        tile_latent_stride_height = tile_sample_stride_height // encode_spatial_compression_ratio
        tile_latent_stride_width = tile_sample_stride_width // encode_spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        tiletask_list: list[TileTask] = []
        temporal_compression = self.config.scale_factor_temporal
        for i in range(0, height, tile_sample_stride_height):
            for j in range(0, width, tile_sample_stride_width):
                time_list = []
                frame_range = 1 + (num_frames - 1) // temporal_compression
                for k in range(frame_range):
                    if k == 0:
                        tile = x[:, :, :1, i : i + tile_sample_min_height, j : j + tile_sample_min_width]
                    else:
                        tile = x[
                            :,
                            :,
                            1 + temporal_compression * (k - 1) : 1 + temporal_compression * k,
                            i : i + tile_sample_min_height,
                            j : j + tile_sample_min_width,
                        ]
                    time_list.append(tile)
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // tile_sample_stride_height, j // tile_sample_stride_width),
                        time_list,
                        workload=time_list[0].shape[3] * time_list[0].shape[4],
                    )
                )

        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(tiletask_list[-1].grid_coord[0] + 1, tiletask_list[-1].grid_coord[1] + 1),
            tile_spec={
                "latent_height": latent_height,
                "latent_width": latent_width,
                "blend_height": blend_height,
                "blend_width": blend_width,
                "tile_latent_stride_height": tile_latent_stride_height,
                "tile_latent_stride_width": tile_latent_stride_width,
            },
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def encode_tile_exec(self, task: TileTask) -> torch.Tensor:
        """Encode a single sample tile into latent space."""
        self.clear_cache()
        try:
            time = []
            for tile in task.tensor:
                self._enc_conv_idx = [0]
                encoded = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                encoded = self.quant_conv(encoded)
                time.append(encoded)
            return torch.cat(time, dim=2)
        finally:
            self.clear_cache()

    def encode_tile_merge(
        self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec
    ) -> torch.Tensor:
        """Merge encoded tiles into a full latent tensor."""
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_height"])
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_width"])
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : grid_spec.tile_spec["tile_latent_stride_height"],
                        : grid_spec.tile_spec["tile_latent_stride_width"],
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[
            :, :, :, : grid_spec.tile_spec["latent_height"], : grid_spec.tile_spec["latent_width"]
        ]
        return enc

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        """Merge decoded tiles into a full image."""
        grid_h, grid_w = grid_spec.grid_shape
        result_rows = []
        self.clear_cache()
        for i in range(grid_h):
            result_row = []
            for j in range(grid_w):
                tile = coord_tensor_map[(i, j)]
                if i > 0:
                    tile = self.blend_v(coord_tensor_map[(i - 1, j)], tile, grid_spec.tile_spec["blend_height"])
                if j > 0:
                    tile = self.blend_h(coord_tensor_map[(i, j - 1)], tile, grid_spec.tile_spec["blend_width"])
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : grid_spec.tile_spec["tile_sample_stride_height"],
                        : grid_spec.tile_spec["tile_sample_stride_width"],
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[
            :, :, :, : grid_spec.tile_spec["sample_height"], : grid_spec.tile_spec["sample_width"]
        ]

        if self.config.patch_size is not None:
            dec = unpatchify(dec, patch_size=self.config.patch_size)

        dec = torch.clamp(dec, min=-1.0, max=1.0)
        return dec

    def _spatial_shard_decode_split_dim(self) -> str | None:
        mode = self.distributed_executor.parallel_mode
        if mode == "spatial_shard_width":
            return "width"
        if mode == "spatial_shard_height":
            return "height"
        return None

    def _spatial_shard_decode_enabled(self, z: torch.Tensor) -> bool:
        split_dim = self._spatial_shard_decode_split_dim()
        if split_dim is None:
            return False
        if z.ndim != 5:
            logger.warning("Wan VAE spatial sharded decode expects 5D latent input; falling back to tiled decode.")
            return False
        if not self.is_distributed_enabled():
            return False
        return self._spatial_shard_group_matches("decode", split_dim)

    def _spatial_shard_group_matches(self, stage: str, split_dim: str) -> bool:
        world_size = dist.get_world_size(group=self.distributed_executor.group)
        requested_size = int(self.distributed_executor.parallel_size)
        if requested_size != world_size:
            logger.warning(
                "Wan VAE spatial sharded %s currently requires vae_patch_parallel_size "
                "to match the DIT group size; falling back to tiled %s. "
                "requested=%s dit_group=%s split_dim=%s",
                stage,
                stage,
                requested_size,
                world_size,
                split_dim,
            )
            return False
        return True

    def _spatial_shard_encode_enabled(self, x: torch.Tensor) -> bool:
        split_dim = self._spatial_shard_decode_split_dim()
        if split_dim is None or x.ndim != 5 or not self.is_distributed_enabled():
            return False
        # The encode copy drops the fast forwards (reference_forward_copy), so this could run with
        # the fast path installed; keep declining until that combination has its own coverage.
        if is_encoder_installed(self):
            return False
        if not self._spatial_shard_group_matches("encode", split_dim):
            return False
        world_size = dist.get_world_size(group=self.distributed_executor.group)
        if not wan_spatial_shard.can_spatial_shard_encode(self, x, split_dim=split_dim, world_size=world_size):
            logger.info_once(
                "Wan VAE spatial sharded encode needs the %s extent to split into equal, stride-aligned "
                "shards across %s ranks; falling back to the unsharded encode path for input shape %s.",
                split_dim,
                world_size,
                tuple(x.shape),
            )
            return False
        return True

    def _exceeds_tile(self, z: torch.Tensor) -> bool:
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        return z.shape[-1] > tile_latent_min_width or z.shape[-2] > tile_latent_min_height

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        """Run the spatially sharded decode on the un-tiled path; it decodes the full frame."""
        # A directly constructed VAE has no distributed state; decode it like the parent does.
        if getattr(self, "distributed_executor", None) is not None and self._spatial_shard_decode_split_dim():
            if self._spatial_shard_decode_enabled(z):
                split_dim = self._spatial_shard_decode_split_dim()
                assert split_dim is not None
                logger.debug("Decode running with Wan VAE spatial_shard_%s mode (un-tiled)", split_dim)
                return wan_spatial_shard.spatial_shard_decode(
                    self,
                    z,
                    group=self.distributed_executor.group,
                    return_dict=return_dict,
                    split_dim=split_dim,
                )
            # A partial VAE group cannot shard; keep the tile-parallel fallback even without native tiling.
            if self.is_distributed_enabled() and self._exceeds_tile(z):
                return self._tile_parallel_decode(z, return_dict=return_dict)
        return super()._decode(z, return_dict=return_dict)

    def _encode(self, x: torch.Tensor):
        """Run the spatially sharded encode before the native-tiling check, like ``_decode``."""
        # ``patchify`` assumes the 5D video layout, so the rank check comes before it; a 4D
        # input takes the regular encode path, as ``_spatial_shard_encode_enabled`` would decide.
        if (
            getattr(self, "distributed_executor", None) is not None
            and x.ndim == 5
            and self._spatial_shard_decode_split_dim()
        ):
            patched = x if self.config.patch_size is None else patchify(x, patch_size=self.config.patch_size)
            if self._spatial_shard_encode_enabled(patched):
                split_dim = self._spatial_shard_decode_split_dim()
                assert split_dim is not None
                logger.debug("Encode running with Wan VAE spatial_shard_%s mode", split_dim)
                return wan_spatial_shard.spatial_shard_encode(
                    self, patched, group=self.distributed_executor.group, split_dim=split_dim
                )
        return super()._encode(x)

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
        if not self.is_distributed_enabled():
            return super().tiled_decode(z, return_dict=return_dict)

        spatial_shard_enabled = self._spatial_shard_decode_enabled(z)
        if spatial_shard_enabled:
            split_dim = self._spatial_shard_decode_split_dim()
            assert split_dim is not None
            logger.debug("Decode running with Wan VAE spatial_shard_%s mode", split_dim)
            return wan_spatial_shard.spatial_shard_decode(
                self,
                z,
                group=self.distributed_executor.group,
                return_dict=return_dict,
                split_dim=split_dim,
            )
        return self._tile_parallel_decode(z, return_dict=return_dict)

    def _tile_parallel_decode(self, z: torch.Tensor, return_dict: bool = True):
        logger.debug("Decode running with distributed executor")
        result = self.distributed_executor.execute(
            z,
            DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
            broadcast_result=False,
        )
        if not return_dict:
            return (result,)

        return DecoderOutput(sample=result)

    def _decode_with_chunks(self, z: torch.Tensor, *, on_chunk: DecodedChunkConsumer) -> None:
        """Stream spatial-shard decode before the native-tiling check; it decodes the full frame."""
        if getattr(self, "distributed_executor", None) is not None and self._spatial_shard_decode_enabled(z):
            self._spatial_shard_decode_with_chunks(z, on_chunk=on_chunk)
            return
        super()._decode_with_chunks(z, on_chunk=on_chunk)

    def _tiled_decode_with_chunks(self, z: torch.Tensor, *, on_chunk: DecodedChunkConsumer) -> None:
        """Decode tiled latents and deliver temporal chunks to ``on_chunk``."""
        if not self.is_distributed_enabled():
            raise ValueError("Wan chunk callbacks are unsupported for spatial tiling")

        spatial_shard_enabled = self._spatial_shard_decode_enabled(z)
        if not spatial_shard_enabled:
            raise ValueError("Wan chunk callbacks require spatial-shard decode when tiling is enabled")
        self._spatial_shard_decode_with_chunks(z, on_chunk=on_chunk)

    def _spatial_shard_decode_with_chunks(self, z: torch.Tensor, *, on_chunk: DecodedChunkConsumer) -> None:
        split_dim = self._spatial_shard_decode_split_dim()
        assert split_dim is not None
        logger.debug("Decode running with Wan VAE spatial_shard_%s mode", split_dim)
        wan_spatial_shard.spatial_shard_decode(
            self,
            z,
            group=self.distributed_executor.group,
            return_dict=True,
            split_dim=split_dim,
            on_chunk=on_chunk,
        )

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode using distributed VAE executor.

        Note: x is already patchified by parent's _encode() before calling this method.
        """
        if not self.is_distributed_enabled():
            return super().tiled_encode(x)

        logger.debug("Encode running with distributed executor")
        self.clear_cache()
        result = self.distributed_executor.execute(
            x,
            DistributedOperator(
                split=lambda x: wan_encode_split.rank_split(self, x),
                exec=self.encode_tile_exec,
                merge=wan_encode_split.rank_merge,
            )
            if wan_encode_split.encode_split_mode() == "rank"
            else DistributedOperator(
                split=self.encode_tile_split,
                exec=self.encode_tile_exec,
                merge=self.encode_tile_merge,
            ),
            broadcast_result=True,
        )
        self.clear_cache()
        return result
