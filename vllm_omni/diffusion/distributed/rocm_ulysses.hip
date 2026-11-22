// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>
#include <c10/core/DeviceGuard.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

namespace {
constexpr int ranks = 4;
constexpr int heads = 56;
constexpr int width = 128;
constexpr int local_heads = heads / ranks;
// One uint4 contains eight BF16 values.
constexpr int vectors = local_heads * width / 8;
constexpr int threads = 256;

__global__ void pack(const uint4* q, const uint4* k, const uint4* v,
                     uint4* wire, int rows, int64_t qs, int64_t ks, int64_t vs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * ranks * vectors) return;
    int packet = i / vectors, lane = i % vectors;
    int row = packet % rows, dst = packet / rows;
    int column = dst * vectors + lane;
    int output = packet * 3 * vectors + lane;
    wire[output] = q[row * qs + column];
    wire[output + vectors] = k[row * ks + column];
    wire[output + 2 * vectors] = v[row * vs + column];
}

__global__ void restore(const uint4* wire, uint4* output, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * ranks * vectors) return;
    int packet = i / vectors, lane = i % vectors;
    int row = packet % rows, source = packet / rows;
    output[(row * ranks + source) * vectors + lane] = wire[i];
}

void check_rows(int64_t rows) {
    TORCH_CHECK(rows > 0 && rows <= INT32_MAX / (3 * ranks * vectors),
                "Unsupported Ulysses row count");
}

at::Tensor pack_qkv(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(2) == heads && q.size(3) == width,
                "Expected Q/K/V [1, rows, 56, 128]");
    check_rows(q.size(1));
    bool aligned = true;
    for (auto t : {&q, &k, &v}) {
        TORCH_CHECK(t->is_cuda() && t->device() == q.device() && t->sizes() == q.sizes() &&
                    t->scalar_type() == at::kBFloat16, "Expected equal BF16 Q/K/V on one GPU");
        aligned = aligned && !t->is_neg() && !t->is_conj() &&
                  t->stride(3) == 1 && t->stride(2) == width && t->stride(1) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(t->data_ptr()) % sizeof(uint4) == 0;
    }
    c10::DeviceGuard guard(q.device());
    int rows = q.size(1), blocks = (rows * ranks * vectors + threads - 1) / threads;
    auto out = at::empty({ranks, rows, 3, local_heads, width}, q.options());
    if (!aligned) {
        // Rank-local packing must preserve the packet order used by its peers.
        int slot = 0;
        for (auto t : {&q, &k, &v}) {
            out.select(2, slot++).copy_(t->reshape({rows, ranks, local_heads, width}).transpose(0, 1));
        }
        return out;
    }
    pack<<<blocks, threads, 0, c10::hip::getCurrentHIPStream()>>>(
        static_cast<const uint4*>(q.data_ptr()), static_cast<const uint4*>(k.data_ptr()),
        static_cast<const uint4*>(v.data_ptr()), static_cast<uint4*>(out.data_ptr()),
        rows, q.stride(1) / 8, k.stride(1) / 8, v.stride(1) / 8);
    auto err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, hipGetErrorString(err));
    return out;
}

at::Tensor restore_heads(const at::Tensor& wire) {
    TORCH_CHECK(wire.is_cuda() && wire.dim() == 4 && wire.size(0) == ranks &&
                wire.size(2) == local_heads && wire.size(3) == width &&
                wire.scalar_type() == at::kBFloat16 && wire.is_contiguous() &&
                !wire.is_neg() && !wire.is_conj() &&
                reinterpret_cast<uintptr_t>(wire.data_ptr()) % sizeof(uint4) == 0,
                "Expected contiguous BF16 receive buffer [4, rows, 14, 128]");
    check_rows(wire.size(1));
    c10::DeviceGuard guard(wire.device());
    int rows = wire.size(1), blocks = (rows * ranks * vectors + threads - 1) / threads;
    auto out = at::empty({1, rows, heads, width}, wire.options());
    restore<<<blocks, threads, 0, c10::hip::getCurrentHIPStream()>>>(
        static_cast<const uint4*>(wire.data_ptr()), static_cast<uint4*>(out.data_ptr()), rows);
    auto err = hipGetLastError();
    TORCH_CHECK(err == hipSuccess, hipGetErrorString(err));
    return out;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(vllm_omni_rocm, m) {
    m.def("ulysses_pack(Tensor q, Tensor k, Tensor v) -> Tensor");
    m.def("ulysses_restore(Tensor wire) -> Tensor");
}
TORCH_LIBRARY_IMPL(vllm_omni_rocm, CUDA, m) {
    m.impl("ulysses_pack", &pack_qkv);
    m.impl("ulysses_restore", &restore_heads);
}
