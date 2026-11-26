// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
// Batch host submission of the original capture-compatible runtime calls.
// No GPU kernel, CUDA batch-copy API or data movement is added.
#include <cstddef>
#include <cstdint>

using Memcpy2D = int (*)(void*, std::size_t, const void*, std::size_t,
                        std::size_t, std::size_t, int, void*);

extern "C" int hcms_submit_copies(const std::uintptr_t* const* operations,
                                std::size_t count, std::uintptr_t stream,
                                Memcpy2D memcpy_2d) {
  for (std::size_t index = 0; index < count; ++index) {
    const auto* op = operations[index];
    const int result = memcpy_2d(
        reinterpret_cast<void*>(op[0]), op[1],
        reinterpret_cast<const void*>(op[2]), op[3], op[4], op[5],
        static_cast<int>(op[6]), reinterpret_cast<void*>(stream));
    if (result != 0) return result;
  }
  return 0;
}
