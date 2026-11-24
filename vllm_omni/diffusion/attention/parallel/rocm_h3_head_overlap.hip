// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

namespace {
constexpr int rows = 9440;
constexpr int ranks = 4;
constexpr int channels = 128;
constexpr int local_heads = 14;
constexpr int first_heads = 4;
constexpr int second_heads = local_heads - first_heads;
// A 16-byte load spans 8 BF16 channels.
constexpr int head_vectors = channels / 8;
constexpr int first_width = first_heads * head_vectors;
constexpr int second_width = second_heads * head_vectors;
constexpr int width = first_width + second_width;
constexpr int vectors = rows * ranks * width;
constexpr int threads = 256;
constexpr int blocks = (vectors + threads - 1) / threads;

__global__ void pack_groups(const uint4* q, const uint4* k, const uint4* v,
                            uint4* first, uint4* second) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vectors) return;
    int packet = i / width, lane = i % width;
    int rank = packet / rows, row = packet % rows;
    int qk_offset = (row * ranks + rank) * width + lane;
    int v_offset = (row * 3 * ranks + rank) * width + lane;
    bool leading = lane < first_width;
    int group_width = leading ? first_width : second_width;
    int group_lane = leading ? lane : lane - first_width;
    uint4* out = (leading ? first : second) + packet * 3 * group_width + group_lane;
    out[0] = q[qk_offset];
    out[group_width] = k[qk_offset];
    out[2 * group_width] = v[v_offset];
}

__global__ void restore_groups(const uint4* first, const uint4* second, uint4* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vectors) return;
    int packet = i / width, lane = i % width;
    int rank = packet / rows, row = packet % rows;
    output[(row * ranks + rank) * width + lane] =
        lane < first_width ? first[packet * first_width + lane]
                           : second[packet * second_width + lane - first_width];
}

bool vector_layout(const at::Tensor& t, int row_stride) {
    return t.stride(0) == row_stride && t.stride(1) == channels && t.stride(2) == 1 &&
           !t.is_neg() && !t.is_conj() && reinterpret_cast<uintptr_t>(t.data_ptr()) % sizeof(uint4) == 0;
}

std::tuple<at::Tensor, at::Tensor> pack(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    TORCH_CHECK(q.dim() == 3 && q.size(0) == rows && q.size(1) == ranks * local_heads &&
                q.size(2) == channels, "Expected Q/K/V shape [9440, 56, 128]");
    for (const auto* plane : {&q, &k, &v}) {
        TORCH_CHECK(plane->is_cuda() && plane->device() == q.device() && plane->sizes() == q.sizes() &&
                    plane->scalar_type() == at::kBFloat16, "Expected three BF16 planes on one GPU");
    }
    c10::DeviceGuard guard(q.device());
    auto first = at::empty({ranks, rows, 3, first_heads, channels}, q.options());
    auto second = at::empty({ranks, rows, 3, second_heads, channels}, q.options());
    constexpr int row_stride = ranks * local_heads * channels;
    if (vector_layout(q, row_stride) && vector_layout(k, row_stride) && vector_layout(v, 3 * row_stride)) {
        pack_groups<<<blocks, threads, 0, c10::hip::getCurrentHIPStream()>>>(
            static_cast<const uint4*>(q.data_ptr()), static_cast<const uint4*>(k.data_ptr()),
            static_cast<const uint4*>(v.data_ptr()), static_cast<uint4*>(first.data_ptr()),
            static_cast<uint4*>(second.data_ptr()));
        auto err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess, hipGetErrorString(err));
    } else {
        int slot = 0;
        for (const auto* plane : {&q, &k, &v}) {
            auto grouped = plane->reshape({rows, ranks, local_heads, channels}).transpose(0, 1);
            first.select(2, slot).copy_(grouped.narrow(2, 0, first_heads));
            second.select(2, slot).copy_(grouped.narrow(2, first_heads, second_heads));
            ++slot;
        }
    }
    return {first, second};
}

at::Tensor restore(const at::Tensor& first, const at::Tensor& second) {
    for (const auto* packet : {&first, &second}) {
        int heads = packet == &first ? first_heads : second_heads;
        TORCH_CHECK(packet->dim() == 4 && packet->size(0) == ranks && packet->size(1) == rows &&
                    packet->size(2) == heads && packet->size(3) == channels && packet->is_contiguous() &&
                    !packet->is_neg() && !packet->is_conj() && packet->is_cuda() &&
                    packet->scalar_type() == at::kBFloat16 && packet->device() == first.device() &&
                    reinterpret_cast<uintptr_t>(packet->data_ptr()) % sizeof(uint4) == 0,
                    "Expected contiguous aligned BF16 return packets for 4 and 10 heads");
    }
    c10::DeviceGuard guard(first.device());
    auto output = at::empty({rows, ranks * local_heads, channels}, first.options());
    restore_groups<<<blocks, threads, 0, c10::hip::getCurrentHIPStream()>>>(
        static_cast<const uint4*>(first.data_ptr()), static_cast<const uint4*>(second.data_ptr()),
        static_cast<uint4*>(output.data_ptr()));
    auto err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, hipGetErrorString(err));
    return output;
}
}  // namespace

TORCH_LIBRARY(vllm_omni_h3_overlap, m) {
    m.def("pack(Tensor q, Tensor k, Tensor v) -> (Tensor, Tensor)");
    m.def("restore(Tensor first, Tensor second) -> Tensor");
}
TORCH_LIBRARY_IMPL(vllm_omni_h3_overlap, CUDA, m) {
    m.impl("pack", &pack);
    m.impl("restore", &restore);
}
