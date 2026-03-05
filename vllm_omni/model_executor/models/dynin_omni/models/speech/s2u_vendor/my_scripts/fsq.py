# ---------------------------------------------------------------------------------
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------------

# This script is modified from https://github.com/google-research/google-research/blob/master/fsq/fsq.ipynb
# by converting jax to torch.

import numpy as np
import torch
import torch.nn as nn
from nemo.core import NeuralModule

Codeword = torch.FloatTensor
Indices = torch.FloatTensor


def round_ste(z):
  """Round with straight through gradients."""
  zhat = torch.round(z)
  return z + (zhat - z).detach()


class FSQ(NeuralModule):
  """Quantizer."""

  def __init__(self, levels: list, eps: float = 1e-3, l2_norm: bool = False, batch_norm: bool = False):
    super().__init__()

    self._levels = levels
    self._eps = eps
    self.l2_norm = l2_norm
    self.batch_norm = batch_norm
    # self._levels_np = torch.Tensor(levels)
    # self._basis = torch.cat((torch.Tensor([1]), torch.cumprod(self._levels_np[:-1], dim=0)))
    self.register_buffer("_levels_np", torch.Tensor(levels))
    self.register_buffer("_basis", torch.cat((torch.Tensor([1]), torch.cumprod(self._levels_np[:-1], dim=0))))

    self._implicit_codebook = self.indexes_to_codes(torch.arange(self.codebook_size))

    if self.batch_norm:
      self.bn = nn.BatchNorm1d(self.num_dimensions, momentum=0.01, eps=1e-3)

  @property
  def num_dimensions(self) -> int:
    """Number of dimensions expected from inputs."""
    return len(self._levels)

  @property
  def codebook_size(self) -> int:
    """Size of the codebook."""
    return np.prod(self._levels)

  @property
  def codebook(self):
    """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
    return self._implicit_codebook

  def bound(self, z: torch.FloatTensor) -> torch.FloatTensor:
    """Bound `z`, an array of shape (..., d)."""
    half_l = (self._levels_np - 1) * (1 - self._eps) / 2
    offset = torch.where(self._levels_np % 2 == 1, 0.0, 0.5)
    shift = torch.tan(offset / half_l)
    return torch.tanh(z + shift) * half_l - offset

  def quantize(self, z: torch.FloatTensor) -> Codeword:
    """Quanitzes z, returns quantized zhat, same shape as z."""
    quantized = round_ste(self.bound(z))

    # Renormalize to [-1, 1].
    half_width = torch.div(self._levels_np, 2, rounding_mode='floor')
    return quantized / half_width

  def _scale_and_shift(self, zhat_normalized):
    # Scale and shift to range [0, ..., L-1]
    half_width = torch.div(self._levels_np, 2, rounding_mode='floor')
    return (zhat_normalized * half_width) + half_width

  def _scale_and_shift_inverse(self, zhat):
    # Note that array(x) // 2 != tensor(x) // 2 when x is negative
    half_width = torch.div(self._levels_np, 2, rounding_mode='floor')
    return (zhat - half_width) / half_width

  def codes_to_indexes(self, zhat: Codeword) -> Indices:
    """Converts a `code` to an index in the codebook."""
    assert zhat.shape[-1] == self.num_dimensions
    zhat = self._scale_and_shift(zhat)
    return torch.sum(zhat * self._basis, axis=-1)

  def indexes_to_codes(self, indices: Indices) -> Codeword:
    """Inverse of `indexes_to_codes`."""
    indices = indices.unsqueeze(-1)
    codes_non_centered = torch.remainder(
      torch.div(indices, self._basis, rounding_mode='floor'), self._levels_np
    )
    return self._scale_and_shift_inverse(codes_non_centered)

  def forward(self, z: torch.FloatTensor) -> Codeword:    
    # z.shape: [batch_size, seq_len, feat_size]
    if self.l2_norm:
      z = nn.functional.normalize(z, p=2, dim=-1)

    zhat = self.quantize(z)

    return zhat