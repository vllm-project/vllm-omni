/*
 * Copyright (c) 2024 NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kBufferSize = 32;
constexpr int kFilterSize = 12;
constexpr int kHalfFilterSize = 6;
constexpr int kUpsampleReplicationPad = 5;
constexpr int kDownsampleReplicationPadLeft = 5;
constexpr int kDownsampleReplicationPadRight = 6;

template <typename scalar_t>
__global__ void anti_alias_activation_forward(
    scalar_t *dst,
    const scalar_t *src,
    const scalar_t *up_filter_ptr,
    const scalar_t *down_filter_ptr,
    const scalar_t *alpha,
    const scalar_t *beta,
    int channels,
    int seq_len) {
  scalar_t up_filter[kFilterSize];
  scalar_t down_filter[kFilterSize];
  scalar_t elements[
      2 * kFilterSize + 2 * kBufferSize + 2 * kUpsampleReplicationPad] = {0};
  scalar_t intermediates[
      2 * kFilterSize + 2 * kBufferSize + kDownsampleReplicationPadLeft +
      kDownsampleReplicationPadRight] = {0};
  scalar_t output[kBufferSize];

  const int block_offset =
      blockIdx.x * 128 * kBufferSize +
      seq_len * (blockIdx.y + gridDim.y * blockIdx.z);
  const int local_offset = threadIdx.x * kBufferSize;
  const int seq_offset = blockIdx.x * 128 * kBufferSize + local_offset;
  const int intermediate_seq_offset =
      blockIdx.x * 128 * kBufferSize * 2 + threadIdx.x * kBufferSize * 2;

  const scalar_t *row_start =
      src + seq_len * (blockIdx.y + channels * blockIdx.z);
  const scalar_t left_value = row_start[0];
  const scalar_t right_value = row_start[seq_len - 1];
  src += block_offset + local_offset;
  dst += block_offset + local_offset;

  const float alpha_value = expf(static_cast<float>(alpha[blockIdx.y]));
  const float beta_value = expf(static_cast<float>(beta[blockIdx.y]));

#pragma unroll
  for (int index = 0; index < kFilterSize; ++index) {
    up_filter[index] = up_filter_ptr[index];
    down_filter[index] = down_filter_ptr[index];
  }

#pragma unroll
  for (int index = -kHalfFilterSize;
       index < kBufferSize + kHalfFilterSize;
       ++index) {
    const int element_index = seq_offset + index;
    if (element_index < 0 && element_index >= -kUpsampleReplicationPad) {
      elements[2 * (kHalfFilterSize + index)] = scalar_t(2) * left_value;
    }
    if (element_index >= seq_len &&
        element_index < seq_len + kUpsampleReplicationPad) {
      elements[2 * (kHalfFilterSize + index)] = scalar_t(2) * right_value;
    }
    if (element_index >= 0 && element_index < seq_len) {
      elements[2 * (kHalfFilterSize + index)] = scalar_t(2) * src[index];
    }
  }

#pragma unroll
  for (int index = 0;
       index < 2 * kBufferSize + 2 * kFilterSize;
       ++index) {
    float accumulator = 0.0f;
    const int element_index = intermediate_seq_offset + index;
#pragma unroll
    for (int filter_index = 0; filter_index < kFilterSize; ++filter_index) {
      if (element_index + filter_index >= 0) {
        accumulator += static_cast<float>(up_filter[filter_index]) *
                       static_cast<float>(elements[index + filter_index]);
      }
    }
    intermediates[index + kDownsampleReplicationPadLeft] = scalar_t(accumulator);
  }

#pragma unroll
  for (int index = 0;
       index < 2 * kBufferSize + 2 * kFilterSize;
       ++index) {
    const int offset = index + kDownsampleReplicationPadLeft;
    const float value = static_cast<float>(intermediates[offset]);
    const float periodic = sinf(value * alpha_value);
    intermediates[offset] =
        scalar_t(value + periodic * periodic / (beta_value + 1.0e-9f));
  }

#pragma unroll
  for (int index = 0; index < kDownsampleReplicationPadLeft; ++index) {
    intermediates[index] = intermediates[kDownsampleReplicationPadLeft];
  }
#pragma unroll
  for (int index = kDownsampleReplicationPadLeft + 2 * kBufferSize +
                   2 * kFilterSize;
       index < kDownsampleReplicationPadLeft + 2 * kBufferSize +
                   2 * kFilterSize + kDownsampleReplicationPadRight;
       ++index) {
    intermediates[index] =
        intermediates[kDownsampleReplicationPadLeft + 2 * kBufferSize +
                      2 * kFilterSize - 1];
  }

#pragma unroll
  for (int index = 0; index < kBufferSize; ++index) {
    float accumulator = 0.0f;
#pragma unroll
    for (int filter_index = 0; filter_index < kFilterSize; ++filter_index) {
      accumulator += static_cast<float>(down_filter[filter_index]) *
                     static_cast<float>(intermediates[
                         index * 2 + filter_index +
                         kDownsampleReplicationPadRight]);
    }
    output[index] = scalar_t(accumulator);
  }

#pragma unroll
  for (int index = 0; index < kBufferSize; ++index) {
    if (seq_offset + index < seq_len) {
      dst[index] = output[index];
    }
  }
}

template <typename scalar_t>
void dispatch(
    scalar_t *dst,
    const scalar_t *src,
    const scalar_t *up_filter,
    const scalar_t *down_filter,
    const scalar_t *alpha,
    const scalar_t *beta,
    int batch_size,
    int channels,
    int seq_len) {
  if (seq_len == 0) {
    return;
  }
  constexpr int threads_per_block = 128;
  constexpr int seq_len_per_block = 4096;
  const int blocks_per_seq_len =
      (seq_len + seq_len_per_block - 1) / seq_len_per_block;
  const dim3 blocks(blocks_per_seq_len, channels, batch_size);
  anti_alias_activation_forward<scalar_t>
      <<<blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          dst,
          src,
          up_filter,
          down_filter,
          alpha,
          beta,
          channels,
          seq_len);
}

}  // namespace

extern "C" torch::Tensor fwd_cuda(
    torch::Tensor const &input,
    torch::Tensor const &up_filter,
    torch::Tensor const &down_filter,
    torch::Tensor const &alpha,
    torch::Tensor const &beta) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() == 3, "input must be [B, C, T]");
  TORCH_CHECK(up_filter.numel() == kFilterSize, "up filter must have 12 taps");
  TORCH_CHECK(
      down_filter.numel() == kFilterSize,
      "down filter must have 12 taps");
  TORCH_CHECK(alpha.numel() == input.size(1), "alpha channel mismatch");
  TORCH_CHECK(beta.numel() == input.size(1), "beta channel mismatch");
  TORCH_CHECK(
      input.scalar_type() == up_filter.scalar_type() &&
          input.scalar_type() == down_filter.scalar_type() &&
          input.scalar_type() == alpha.scalar_type() &&
          input.scalar_type() == beta.scalar_type(),
      "fused activation tensors must have the same dtype");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty_like(input, input.options().requires_grad(false));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "indextts_alias_free_forward",
      [&] {
        dispatch<scalar_t>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            up_filter.data_ptr<scalar_t>(),
            down_filter.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            input.size(0),
            input.size(1),
            input.size(2));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
