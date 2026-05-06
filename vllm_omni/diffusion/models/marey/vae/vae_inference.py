from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from omegaconf import MISSING
from torch import nn

from . import utils
from .two_stage_vae import TwoStageVAE


def _ensure_sequence(x: float | Sequence[float]) -> Sequence[float]:
    # ensure that x is a sequence, if it is not, wrap it in a list
    return x if isinstance(x, Sequence) else [x]


class TwoStageVAEInference(nn.Module):
    """
    A wrapper around the TwoStageVAE model optimized for inference.
    This wrapper applies scaling and bias factors to the latents, manages chunking of frames,
    optionally sets a maximum batch size for memory management, and also optionally compiles the model.

    Args:
        cfg (TwoStageVAEInferenceConfig): The configuration for the inference wrapper.
        device (torch.device | str): The device to load the model on.
        dtype (torch.dtype | None): The dtype to load the model on.
    """

    def __init__(
        self,
        cfg: TwoStageVAEInferenceConfig,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        if isinstance(cfg.checkpoint, TwoStageVAE):
            self.model: TwoStageVAE = cfg.checkpoint
        else:
            self.model = TwoStageVAE.load_from_checkpoint(
                cfg.checkpoint, device=device, dtype=dtype
            )
        # add trailing axes so that scaling and bias factors are broadcasted to the channel axis:
        scaling_factor = torch.as_tensor(
            _ensure_sequence(cfg.scaling_factor), device=device, dtype=dtype
        )
        bias_factor = torch.as_tensor(
            _ensure_sequence(cfg.bias_factor), device=device, dtype=dtype
        )
        # add trailing axes so that scaling and bias factors are broadcasted to the channel axis
        scaling_factor = scaling_factor[:, None, None, None]
        bias_factor = bias_factor[:, None, None, None]

        # register as buffers so that they are moved just like submodules
        self.register_buffer("scaling_factor", scaling_factor)
        self.register_buffer("bias_factor", bias_factor)

        # build lookup tables for mapping from compression and expansion factors to skip_n_blocks
        self._skip_n_blocks_for_compression: dict[tuple[int, int, int], int] = {}
        self._skip_n_blocks_for_expansion: dict[tuple[int, int, int], int] = {}
        for skip_n_blocks in self.cfg.valid_skip_n_blocks:
            compression = self.model.get_downsample_factors(
                skip_first_n_down_blocks=skip_n_blocks
            )
            self._skip_n_blocks_for_compression[compression] = skip_n_blocks
            expansion = self.model.get_upsample_factors(
                skip_last_n_up_blocks=skip_n_blocks
            )
            self._skip_n_blocks_for_expansion[expansion] = skip_n_blocks

        default_compression = self.model.get_downsample_factors(
            skip_first_n_down_blocks=0
        )
        self._active_compression: tuple[int, int, int] = default_compression
        default_expansion = self.model.get_upsample_factors(skip_last_n_up_blocks=0)
        self._active_expansion: tuple[int, int, int] = default_expansion

        # save the latent chunk length, which is governed by the temporal stage
        temporal_downsampling_factor = self.model.temporal_stage.time_downsample_factor
        assert self.cfg.frame_chunk_len % temporal_downsampling_factor == 0, (
            "frame_chunk_len must be divisible by temporal_downsampling_factor"
        )
        self.latent_chunk_len = self.cfg.frame_chunk_len // temporal_downsampling_factor

        # apply temporal chunking to _encode method
        self._encode = utils.WithTemporalChunking(  # type: ignore[assignment]
            self._encode,
            chunk_size=self.cfg.frame_chunk_len,
            skip_transform_if_single_frame=True,
        )
        # apply requested decode_chunking_strategy to _decode method
        if self.cfg.decode_chunking_strategy == "overlap-and-drop":
            self._decode = utils.WithInputWindowingDropBoundaryOutputs(  # type: ignore[assignment]
                self._decode,
                win_size=self.latent_chunk_len,
                extra_frames_in=1,  # windows overlap by 1 latent frame on either end
                extra_frames_out=temporal_downsampling_factor,  # overlap of 1 video frame on either end
                skip_transform_if_single_frame=True,
            )
        elif self.cfg.decode_chunking_strategy == "basic":
            self._decode = utils.WithTemporalChunking(  # type: ignore[assignment]
                self._decode,
                chunk_size=self.latent_chunk_len,
                skip_transform_if_single_frame=True,
            )
        else:
            raise ValueError(
                f"invalid decode_chunking_strategy: '{self.cfg.decode_chunking_strategy}'"
            )

        # compile the model if requested
        if self.cfg.torch_compile_kwargs is not None:
            # granular compilation of each stage so compilations are more likely to be reused
            self.model.spatial_encoder.forward = torch.compile(  # type: ignore[assignment]
                self.model.spatial_encoder.forward, **self.cfg.torch_compile_kwargs
            )
            self.model.spatial_decoder.forward = torch.compile(  # type: ignore[assignment]
                self.model.spatial_decoder.forward, **self.cfg.torch_compile_kwargs
            )
            self.model.temporal_stage.encoder.forward = torch.compile(  # type: ignore[assignment]
                self.model.temporal_stage.encoder.forward,
                **self.cfg.torch_compile_kwargs,
            )
            self.model.temporal_stage.decoder.forward = torch.compile(  # type: ignore[assignment]
                self.model.temporal_stage.decoder.forward,
                **self.cfg.torch_compile_kwargs,
            )

        # apply `with_max_batch_size` to model steps of each stage, if requested
        if self.cfg.max_batch_size is not None:
            # the first/spatial stage has frames in the batch dimension, so this wrap will chunk
            # frames into chunks of size `max_batch_size` during encoding and decoding:
            self.model.spatial_encoder.forward = utils.WithMaxBatchSize(  # type: ignore[assignment]
                self.model.spatial_encoder.forward, self.cfg.max_batch_size
            )
            self.model.spatial_decoder.forward = utils.WithMaxBatchSize(  # type: ignore[assignment]
                self.model.spatial_decoder.forward, self.cfg.max_batch_size
            )
            # also apply the max batch size to the spatio-temporal stage:
            self.model.temporal_stage.encoder.forward = utils.WithMaxBatchSize(  # type: ignore[assignment]
                self.model.temporal_stage.encoder.forward, self.cfg.max_batch_size
            )
            self.model.temporal_stage.decoder.forward = utils.WithMaxBatchSize(  # type: ignore[assignment]
                self.model.temporal_stage.decoder.forward, self.cfg.max_batch_size
            )

    @property
    def latent_dim(self) -> int:
        """Number of latent channels (D)."""
        return self.model.latent_embed_dim

    @property
    def temporal_chunk_size(self) -> int:
        """Number of raw frames consumed per temporal chunk."""
        return self.cfg.frame_chunk_len

    @property
    def compression_modes(self) -> Sequence[tuple[int, int, int]]:
        """All supported downsampling compression factors ``(Td, Hd, Wd)``."""
        return tuple(self._skip_n_blocks_for_compression.keys())

    def set_compression_mode(self, mode: tuple[int, int, int]) -> None:
        """Select the active compression mode for subsequent ``encode`` calls.

        Raises ``ValueError`` if *mode* is not among ``compression_modes``.
        """
        if mode not in self._skip_n_blocks_for_compression:
            raise ValueError(
                f"{mode!r} is not a supported compression mode; "
                f"available: {self.compression_modes}"
            )
        self._active_compression = mode

    @property
    def expansion_modes(self) -> Sequence[tuple[int, int, int]]:
        """
        Get the upsampling expansion factors for each mode supported by the model.

        Returns:
            Sequence[tuple[int, int, int]]: The upsampling modes.
        """
        return tuple(self._skip_n_blocks_for_expansion.keys())

    def input_to_latent_size(
        self,
        input_size: tuple[int | None, int | None, int | None],
        compression: tuple[int, int, int] | None = None,
    ) -> tuple[int | None, int | None, int | None]:
        """
        Compute what the encoded latent size would be for the given input size.

        Args:
            input_size (tuple[int | None, int | None, int | None]): The input size (Tv, Hv, Wv). Any None elements
                are ignored and returned as None.
            compression (tuple[int, int, int] | None): The compression factors to use (Td, Hd, Wd).

        Returns:
            tuple[int | None, int | None, int | None]: The encoded latent size (Tl, Hl, Wl).
        """
        effective = compression or self._active_compression
        if effective not in self._skip_n_blocks_for_compression:
            raise ValueError(f"'{effective}' is not a supported compression mode")
        skip_n_blocks = self._skip_n_blocks_for_compression[effective]
        return self.model.input_to_latent_size(
            input_size, skip_first_n_down_blocks=skip_n_blocks
        )

    @torch.inference_mode()
    def encode(
        self, x: torch.Tensor, compression: tuple[int, int, int] | None = None,
        shard_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        gather_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Encode a video into latents.

        Args:
            x (torch.Tensor): The input video tensor (B, C, T, H, W).
            compression (tuple[int, int, int] | None): The compression factors to use (Td, Hd, Wd).

        Returns:
            torch.Tensor: The encoded latent tensor (B, C, Tl, Hl, Wl).
        """
        num_frames = x.shape[2]
        if num_frames > 1 and num_frames % self.cfg.frame_chunk_len != 0:
            raise ValueError(
                "number of frames must be 1 or divisible by frame_chunk_len"
            )
        return self._encode(x, compression=compression, shard_fn=shard_fn, gather_fn=gather_fn)

    @torch.inference_mode()
    def decode(
        self,
        z: torch.Tensor,
        num_frames: int | None = None,
        spatial_size: tuple[int, int] | None = None,
        expansion: tuple[int, int, int] | None = None,
        shard_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        gather_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Decode latents into a video.

        Args:
            z (torch.Tensor): The latent tensor (B, C, Tl, Hl, Wl).
            num_frames (int | None): The target number of frames to decode to.
            spatial_size (tuple[int, int] | None): The target spatial size to decode to (H, W).
            expansion (tuple[int, int, int] | None): The expansion factors to use (Tu, Hu, Wu).

        Returns:
            torch.Tensor: The decoded video tensor (B, C, Tv, Hv, Wv).
        """

        num_latent_frames = z.shape[2]

        if (
            num_frames != 1
        ):  # if not explictly decoding to an image, the number of latent frames must be divisible by latent_chunk_len
            if num_latent_frames % self.latent_chunk_len != 0:
                raise ValueError(
                    "number of latent frames must be divisible by latent_chunk_len (if num_frames is not 1)"
                )

        return self._decode(
            z,
            expansion=expansion,
            num_frames=num_frames,
            spatial_size=spatial_size,
            shard_fn=shard_fn,
            gather_fn=gather_fn,
        )

    def forward(
        self, x: torch.Tensor, compression: tuple[int, int, int] | None = None
    ) -> torch.Tensor:
        """
        Encode a video into latents.

        Args:
            x (torch.Tensor): The input video tensor (B, C, T, H, W).
            compression (tuple[int, int, int] | None): The compression factors to use (Td, Hd, Wd).

        Returns:
            torch.Tensor: The encoded latent tensor (B, C, Tl, Hl, Wl).
        """
        return self.encode(x, compression=compression)

    def _encode(
        self, x: torch.Tensor, compression: tuple[int, int, int] | None = None,
        shard_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        gather_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        effective = compression or self._active_compression
        if effective not in self._skip_n_blocks_for_compression:
            raise ValueError(f"'{effective}' is not a supported compression mode")
        skip_n_blocks = self._skip_n_blocks_for_compression[effective]
        if shard_fn is not None:
            x = shard_fn(x,0)
        z = self.model.encode(x, skip_first_n_down_blocks=skip_n_blocks).mean
        if gather_fn is not None:
            z = z.contiguous()
            z = gather_fn(z,0)
        # apply scaling and bias
        z = (z + self.bias_factor) * self.scaling_factor  # type: ignore[operator]
        return z

    def _decode(
        self,
        z: torch.Tensor,
        expansion: tuple[int, int, int] | None = None,
        num_frames: int | None = None,
        spatial_size: tuple[int, int] | None = None,
        shard_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        gather_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        effective_exp = expansion or self._active_expansion
        if effective_exp not in self._skip_n_blocks_for_expansion:
            raise ValueError(f"'{effective_exp}' is not a supported expansion mode")
        skip_n_blocks = self._skip_n_blocks_for_expansion[effective_exp]
        # undo scaling and bias
        z = z / self.scaling_factor - self.bias_factor  # type: ignore[operator]
        if shard_fn is not None:
            z = shard_fn(z,0)
        z = self.model.decode(
            z,
            num_frames=num_frames,
            spatial_size=spatial_size,
            skip_last_n_up_blocks=skip_n_blocks,
        )
        if gather_fn is not None:
            z = z.contiguous()
            z = gather_fn(z,0)
        return z


@dataclass
class TwoStageVAEInferenceConfig:
    """
    Configuration for the TwoStageVAEInference wrapper.

    Parameters:
        checkpoint (FileLike | TwoStageVAE): The path to the checkpoint file to load the TwoStageVAE from, or a pre-loaded
            TwoStageVAE model.
        frame_chunk_len (int): The number of frames to group together for encoding and decoding.
        decode_chunking_strategy (Literal["basic", "overlap-and-drop"]): The strategy to use for chunking the frames during decoding.
            "basic" means basic chunking of the input frames into size `frame_chunk_len`). This means output frames at the boundaries
            of each chunk are computed without the temporal context of neighbouring frames which can lead to temporal artifacts across
            chunk boundaries.
            The "overlap-and-drop" strategy creates sliding windows of input frames with overlap, and the output frames at the
            boundaries are dropped. i.e. only frames computed with the temporal context of neighbouring frames are retained (except
            for the first and last frames of input videos, for which no context is available).
        scaling_factor (float | Sequence[float]): The scaling factor to apply to the latents.
        bias_factor (float | Sequence[float]): The bias factor to apply to the latents.
        valid_skip_n_blocks (Sequence[int]): Valid options for the number of downsampling or upsampling blocks to skip
            (during encoding or decoding respectively). Multiple options enable different compression or expansion factors.
        max_batch_size (int | None): The maximum batch size to use for encoding and decoding. This can be used to manage
            memory usage.
        torch_compile_kwargs (dict[str, Any] | None): The keyword arguments to pass to torch.compile. If None, no compilation
            is performed.
    """

    checkpoint: Any = MISSING
    frame_chunk_len: int = MISSING  # type: ignore[assignment]
    decode_chunking_strategy: str = "basic"
    scaling_factor: Any = 1.0
    bias_factor: Any = 1.0
    valid_skip_n_blocks: List[int] = field(default_factory=lambda: [0])
    max_batch_size: Optional[int] = None
    torch_compile_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        assert self.decode_chunking_strategy in ("basic", "overlap-and-drop"), (
            f"invalid decode_chunking_strategy: {self.decode_chunking_strategy!r}; "
            "must be 'basic' or 'overlap-and-drop'"
        )
        assert len(self.valid_skip_n_blocks) > 0, (
            "valid_skip_n_blocks must be non-empty"
        )
        assert all(skip_n_blocks >= 0 for skip_n_blocks in self.valid_skip_n_blocks), (
            "valid_skip_n_blocks must be non-negative"
        )
        assert self.frame_chunk_len > 0, "frame_chunk_len must be positive"
        assert self.max_batch_size is None or self.max_batch_size > 0, (
            "max_batch_size must be positive or None"
        )
        # if bias and scaling factor are sequences, they must have the same length
        if isinstance(self.bias_factor, Sequence) and isinstance(
            self.scaling_factor, Sequence
        ):
            assert len(self.bias_factor) == len(self.scaling_factor), (
                "bias_factor and scaling_factor must have the same length if they are both sequences"
            )
        if self.torch_compile_kwargs is not None:
            # verify torch_compile_kwargs args are valid by calling torch.compile on a small module
            torch.compile(nn.Linear(1, 1), **self.torch_compile_kwargs)

    def make(
        self, device: torch.device, dtype: torch.dtype | None = None
    ) -> TwoStageVAEInference:
        """Build a ``TwoStageVAEInference`` on *device*.

        *device* is required (not optional) because TwoStageVAE is heavy and
        should always be placed deliberately.

        Raises ``RuntimeError`` if *device* is a CUDA device but CUDA is not
        available or not yet initialised (e.g. ``torch.distributed`` not set up).
        """
        if not isinstance(device, torch.device):
            raise TypeError(
                f"TwoStageVAEInferenceConfig.make() requires an explicit "
                f"torch.device, got {type(device).__name__}"
            )
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "TwoStageVAE requires CUDA but torch.cuda.is_available() is False"
                )
            if not torch.cuda.is_initialized():
                raise RuntimeError(
                    "TwoStageVAE requires CUDA to be initialised "
                    "(call torch.cuda.init() or set up torch.distributed first)"
                )
        return TwoStageVAEInference(self, device=device, dtype=dtype)
