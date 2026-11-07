// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <torch/library.h>
#include <dlfcn.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" void launch_ulysses_fused_exchange(
    void* stream, uint64_t ffts, void* peers, void* q, void* k, void* v,
    uint32_t rank, uint32_t world, uint32_t batch, uint32_t seq,
    uint32_t qr, uint32_t kr, uint32_t vr, uint32_t reverse, uint32_t count);

namespace {
uint64_t get_ffts_config() {
    static void* library = dlopen("libshmem.so", RTLD_NOW | RTLD_LOCAL);
    TORCH_CHECK(library, "Failed to load symmetric memory backend (libshmem.so): ", dlerror());
    static auto get_ffts = reinterpret_cast<uint64_t (*)()>(dlsym(library, "util_get_ffts_config"));
    TORCH_CHECK(get_ffts, "ffts symbol missing");
    return get_ffts();
}

void pull(const at::Tensor& peers, at::Tensor output, int64_t rank, int64_t world,
          int64_t batch, int64_t seq, int64_t row_bytes, bool reverse) {
    at::DeviceGuard guard(output.device());
    auto stream = c10_npu::getCurrentNPUStream(output.device().index());
    void* p = peers.data_ptr();
    void* o = output.data_ptr();
    at_npu::native::OpCommand::RunOpApiV2("UlyssesFusedPull", [=]() -> int {
        launch_ulysses_fused_exchange(stream.stream(false), get_ffts_config(), p, o, nullptr, nullptr,
                                      rank, world, batch, seq, row_bytes, 0, 0, reverse, 1);
        return 0;
    });
}

void qkv(const at::Tensor& peers, at::Tensor q, at::Tensor k, at::Tensor v,
         int64_t rank, int64_t world, int64_t batch, int64_t seq,
         int64_t qr, int64_t kr, int64_t vr) {
    at::DeviceGuard guard(q.device());
    auto stream = c10_npu::getCurrentNPUStream(q.device().index());
    void* p = peers.data_ptr();
    void* qp = q.data_ptr();
    void* kp = k.data_ptr();
    void* vp = v.data_ptr();
    at_npu::native::OpCommand::RunOpApiV2("UlyssesFusedQKV", [=]() -> int {
        launch_ulysses_fused_exchange(stream.stream(false), get_ffts_config(), p, qp, kp, vp,
                                      rank, world, batch, seq, qr, kr, vr, 0, 3);
        return 0;
    });
}
}  // namespace

TORCH_LIBRARY(omni_ulysses_symmetric_memory_fused, m) {
    m.def("pull(Tensor peers, Tensor(a!) output, int rank, int world, int batch, int seq, int row_bytes, bool reverse) -> ()");
    m.def("qkv(Tensor peers, Tensor(a!) q, Tensor(b!) k, Tensor(c!) v, int rank, int world, int batch, int seq, int qr, int kr, int vr) -> ()");
}
TORCH_LIBRARY_IMPL(omni_ulysses_symmetric_memory_fused, PrivateUse1, m) {
    m.impl("pull", pull);
    m.impl("qkv", qkv);
}
