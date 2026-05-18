from typing import Any

import torch
from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck, OobleckDecoderOutput
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)

logger = init_logger(__name__)


class DistributedAutoencoderOobleck(AutoencoderOobleck, DistributedVaeMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # According to official implementation, tile size 128 and overlap 32 works well.
        # For result correctness, it is recommended to set overlap_size to 32 or greater.
        self.tile_size = kwargs.get("tile_size", 128)
        self.overlap_size = kwargs.get("overlap_size", 32)
        self.use_tiling = kwargs.get("use_tiling", True)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        model = super().from_pretrained(*args, **kwargs)
        model.init_distributed()
        return model

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> torch.Tensor:
        self.latent_length = z.shape[-1]
        result = self.tiled_decode(z, return_dict=return_dict)
        if not return_dict:
            return result[0]
        else:
            return result.sample

    def tile_split(self, latents: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, latent_length = latents.shape
        self.latent_length = latent_length
        tile_size = self.tile_size
        overlap_size = self.overlap_size
        tile_stride = tile_size - overlap_size
        tiletask_list = []
        if tile_size >= latent_length:
            return [TileTask(tile_id=0, grid_coord=(0,), tensor=latents, workload=latent_length)], GridSpec(
                split_dims=(2,),
                grid_shape=(1,),
                tile_spec={"tile_size": tile_size, "overlap_size": overlap_size},
                output_dtype=latents.dtype,
            )
        else:
            for i in range(0, latent_length - tile_size + 1, tile_stride):
                tile = latents[:, :, i : i + tile_size]
                tiletask_list.append(
                    TileTask(
                        tile_id=len(tiletask_list),
                        grid_coord=(i // tile_stride,),
                        tensor=tile,
                        workload=tile.shape[2],
                    )
                )
            if i + tile_size != latent_length:
                # Final tile
                tile = latents[:, :, -tile_size:]
                tiletask_list.append(
                    TileTask(
                        tile_id=len(tiletask_list),
                        grid_coord=(len(tiletask_list),),
                        tensor=tile,
                        workload=tile.shape[2],
                    )
                )
            tile_spec = {
                "tile_size": tile_size,
                "overlap_size": overlap_size,
            }

            grid_spec = GridSpec(
                split_dims=(2,),
                grid_shape=(tiletask_list[-1].grid_coord[0] + 1,),
                tile_spec=tile_spec,
                output_dtype=latents.dtype,
            )
            return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        """Decode a single latent tile."""
        tile = task.tensor
        decoded = super().decode(tile).sample
        return decoded

    def tile_merge(self, coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
        """Merge decoded tiles into a full audio."""
        grid_len = grid_spec.grid_shape
        result = self.blend_chunks(
            [coord_tensor_map[(i,)] for i in range(grid_len[0])],
        )
        return result

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
        if not self.is_distributed_enabled():
            logger.debug("Distributed execution not enabled, falling back to regular decode")
            return super().decode(z, return_dict=return_dict)

        logger.debug("Decode running with distributed executor")
        result = self.distributed_executor.execute(
            z,
            DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
            broadcast_result=True,
        )
        if not return_dict:
            return (result,)

        return OobleckDecoderOutput(sample=result)

    def blend_chunks(self, decoded_chunks: list[torch.Tensor]):
        # simple linear crossfade for blending two chunks
        num_chunks = len(decoded_chunks)
        samples_per_latent = int(self.hop_length)
        batch_size = decoded_chunks[0].shape[0]
        out_channels = decoded_chunks[0].shape[1]
        chunk_size = decoded_chunks[0].shape[2] // samples_per_latent
        hop_size = chunk_size - self.overlap_size
        y_size = self.latent_length * samples_per_latent
        y_final = torch.zeros(
            (batch_size, out_channels, y_size), dtype=decoded_chunks[0].dtype, device=decoded_chunks[0].device
        )
        for i in range(num_chunks):
            # figure out where to put the audio along the time domain
            y_chunk = decoded_chunks[i]
            if i == num_chunks - 1:
                # final chunk always goes at the end
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size * samples_per_latent
                t_end = t_start + chunk_size * samples_per_latent
            #  remove the edges of the overlaps
            ol = (self.overlap_size // 2) * samples_per_latent
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if i > 0:
                # no overlap for the start of the first chunk
                t_start += ol
                chunk_start += ol
            if i < num_chunks - 1:
                # no overlap for the end of the last chunk
                t_end -= ol
                chunk_end -= ol
            # paste the chunked audio into our y_final output audio
            y_final[:, :, t_start:t_end] = y_chunk[:, :, chunk_start:chunk_end]
        return y_final
