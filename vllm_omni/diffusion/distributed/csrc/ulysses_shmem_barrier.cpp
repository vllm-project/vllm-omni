// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <torch/library.h>
#include <dlfcn.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace {

using BarrierFn = void (*)(aclrtStream);

BarrierFn get_barrier() {
  // Retain the library for the lifetime of queued callbacks.
  static void* library = dlopen("libshmem.so", RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(library != nullptr, "Unable to load libshmem.so: ", dlerror());
  static auto barrier = reinterpret_cast<BarrierFn>(
      dlsym(library, "aclshmemx_barrier_all_on_stream"));
  TORCH_CHECK(barrier != nullptr, "libshmem.so lacks stream barriers");
  return barrier;
}

void stream_barrier(const at::Tensor& workspace) {
  at::DeviceGuard guard(workspace.device());
  const auto stream = c10_npu::getCurrentNPUStream(workspace.device().index());
  const auto barrier = get_barrier();
  // Submit through the same host queue as copy_ and Triton-Ascend. A direct
  // ctypes call can overtake kernels which have not reached the device yet.
  at_npu::native::OpCommand::RunOpApiV2(
      "OmniUlyssesSHMEMBarrier", [stream, barrier]() -> int {
        barrier(stream.stream(false));
        return 0;
      });
}

}  // namespace

TORCH_LIBRARY(omni_ulysses_shmem, m) {
  m.def("barrier(Tensor workspace) -> ()");
}

TORCH_LIBRARY_IMPL(omni_ulysses_shmem, PrivateUse1, m) {
  m.impl("barrier", stream_barrier);
}
