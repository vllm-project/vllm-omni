from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def cast_tuple(t: Any, length: int = 1) -> tuple[Any, ...]:
    """
    Cast a single value to a tuple of the given length.
    If the value is already a tuple (or a list), it is returned as a tuple.
    If the value is not a tuple (or a list), it is wrapped in a tuple of the given length.

    Args:
        t: The value or iterable to cast.
        length: The length of the tuple to return.

    Returns:
        A tuple of the given length.
    """
    return tuple(t) if isinstance(t, (tuple, list)) else ((t,) * length)


def is_odd(n: int) -> bool:
    """Check if a number is odd."""
    return n % 2 == 1


def ceildiv(a: int, b: int) -> int:
    """
    Calculate the ceiling of the division of a by b.

    Args:
        a: The dividend.
        b: The divisor.

    Returns:
        The ceiling of the division of a by b.
    """
    return -(a // -b)


def pad_at_dim(t: torch.Tensor, pad: tuple[int, int], dim: int = -1) -> torch.Tensor:
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), mode="constant")


class DiagonalGaussianDistribution(object):
    def __init__(
        self,
        parameters: torch.Tensor,
        deterministic: bool = False,
    ):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device, dtype=self.mean.dtype
            )

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        # return a reparametrized sample from the distribution
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.mean.dtype,
        )
        x = self.mean + self.std * noise
        return x

    def kl(
        self,
        other: DiagonalGaussianDistribution | None = None,
        reduce_dims: tuple[int, ...] | None = (1, 2, 3, 4),
    ) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if (
                other is None
            ):  # assumes other is a zero-mean, unit variance Gaussian distribution
                res = 0.5 * (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)
            else:
                res = 0.5 * (
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar
                )
            if reduce_dims:
                res = torch.sum(res, dim=reduce_dims)
            return res

    def nll(
        self, sample: torch.Tensor, reduce_dims: tuple[int, ...] | None = (1, 2, 3, 4)
    ) -> torch.Tensor:

        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        res: torch.Tensor = 0.5 * (
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var
        )
        if reduce_dims:
            res = torch.sum(res, dim=reduce_dims)
        return res

    def mode(self) -> torch.Tensor:
        return self.mean


class WithMaxBatchSize:
    """Wrapper for applying a function to an input tensor with a given max batch size.
    This is performed by splitting the input tensor into chunks of at most `max_batch_size`, applying the function to each
    chunk separately, and then concatenating the results back along the batch axis.

    Args:
        fn: The function to apply to the input tensor.
        max_batch_size: The maximum batch size to use for the function.
    """

    def __init__(self, fn: Callable[..., torch.Tensor], max_batch_size: int):
        self.fn = fn
        self.max_batch_size = max_batch_size

    def __call__(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if x.shape[0] > self.max_batch_size:
            return torch.cat(
                [self.fn(x_slice, **kwargs) for x_slice in x.split(self.max_batch_size)]
            )
        else:
            return self.fn(x, **kwargs)


class WithAxisBatched:
    """Wrapper for applying a function to an input tensor with a given axis batched.
    This is performed by folding the input tensor along the given axis into the batch axis, applying the function,
    and then unfolding the output tensor along the given axis from the batch axis back to the original axis.

    Args:
        fn: The function to apply to the input tensor.
        axis: The axis to batch the input tensor along.
    """

    def __init__(self, fn: Callable[..., torch.Tensor], axis: int):
        self.fn = fn
        self.axis = axis

    def __call__(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.axis > 1:
            x = x.moveaxis(self.axis, 1)
        y: torch.Tensor = self.fn(x.flatten(0, 1), **kwargs).unflatten(
            0, (x.shape[0], -1)
        )
        if self.axis > 1:
            y = y.moveaxis(1, self.axis)
        return y


class WithTemporalChunking:
    """Wrapper for applying a function to fixed-sized chunks along the temporal axis (third axis) of an input tensor.
    This is performed by folding chunks of the temporal axis into the batch axis, applying the function to the chunks,
    and then undoing the folding on the output tensor.

    Args:
        fn: The function to apply to the chunks.
        chunk_size: The size of the chunks along the temporal axis.
        skip_transform_if_single_frame: Whether to skip the input and output transformations if the input is a single frame.
    """

    def __init__(
        self,
        fn: Callable[..., torch.Tensor],
        chunk_size: int,
        skip_transform_if_single_frame: bool = True,
    ):
        self.fn = fn
        self.chunk_size = chunk_size
        self.skip_transform_if_single_frame = skip_transform_if_single_frame

    def __call__(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # x is (B, C, Tv, Hv, Wv)

        # skip input/output transformations if a single frame and skip_transform_if_single_frame is True
        single_frame = x.shape[2] == 1
        if self.skip_transform_if_single_frame and single_frame:
            return self.fn(x, **kwargs)

        batch_size = x.shape[0]
        # fold chunks of the temporal axis into the batch axis:
        x = rearrange(x, "b c (n t) ... -> (b n) c t ...", t=self.chunk_size)
        x = self.fn(x, **kwargs)
        # unfold chunks of the temporal axis from the batch axis back to the temporal axis:
        return rearrange(x, "(b n) c t ... -> b c (n t) ...", b=batch_size)


class WithInputWindowingDropBoundaryOutputs:
    """Wrapper for applying a function to sliding windows of an input tensor, retaining only the middle part of the output windows.
    This is performed along the temporal axis of inputs. This means every retained output frame is computed with the temporal context of
    neighbouring input frames (except for the first and last frames), unlike pure chunking (where boundary frames in every chunk are
    computed without context from adjacent chunks).

    Args:
        fn: The function to apply to the overlapping input windows.
        win_size: The size of the windows.
        extra_frames_in: The number of frames overlapping on either end of the input windows.
        extra_frames_out: The number of frames overlapping on either end of the output windows.
        skip_transform_if_single_frame: Whether to skip the input and output transformations if the input is a single frame.
    """

    def __init__(
        self,
        fn: Callable[..., torch.Tensor],
        win_size: int,
        extra_frames_in: int,
        extra_frames_out: int,
        skip_transform_if_single_frame: bool = True,
    ):
        self.fn = fn
        self.win_size = win_size
        self.extra_frames_in = extra_frames_in
        self.extra_frames_out = extra_frames_out
        self.skip_transform_if_single_frame = skip_transform_if_single_frame
        assert win_size > 2 * extra_frames_in, (
            "win_size must be greater than 2 * extra_frames_in"
        )

    def __call__(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # x is (B, C, Tv, Hv, Wv)

        # skip transformations if a single frame and skip_transform_if_single_frame is True
        single_frame = x.shape[2] == 1
        if self.skip_transform_if_single_frame and single_frame:
            return self.fn(x, **kwargs)

        overlap_size = 2 * self.extra_frames_in
        # unfold to create sliding windows of size `win_size` with `overlap_size` overlap
        x_wins = x.unfold(
            dimension=-3, size=self.win_size, step=self.win_size - overlap_size
        )
        # (B, C, num windows, Hv, Wv, win_size)
        x_wins = rearrange(
            x_wins, "B C num_wins ... win_size -> (B num_wins) C win_size ..."
        )  # windows become the temporal axis (or chunk of frames), and multiple windows are batched together
        y_wins = self.fn(x_wins, **kwargs)  # apply the function to the windows
        y_wins = rearrange(
            y_wins,
            "(B num_wins) C win_size_out ... -> B C num_wins win_size_out ...",
            B=x.shape[0],
        )  # pull windows back from the batch axis
        win_size_out = y_wins.shape[3]
        # extract the middle part of the output windows. this means we keep only the parts which have
        # neighbouring context at the boundaries on either end of the window. this removes
        # the overlapping portions of the output windows.
        y_mids = y_wins[
            :, :, :, self.extra_frames_out : win_size_out - self.extra_frames_out
        ]
        # flatten the windows to restore the continuous temporal axis.
        y_mids_flat = rearrange(
            y_mids, "B C num_windows L ... -> B C (num_windows L) ..."
        )
        # re-add the very beginning and end of the temporal axis, which were lost when we extracted the middle parts.
        # this leaves us with correct number of output frames.
        y_prefix = y_wins[:, :, 0, : self.extra_frames_out]
        y_postfix = y_wins[:, :, -1, win_size_out - self.extra_frames_out :]
        y_chunks_flat = torch.cat([y_prefix, y_mids_flat, y_postfix], dim=2)
        return y_chunks_flat
