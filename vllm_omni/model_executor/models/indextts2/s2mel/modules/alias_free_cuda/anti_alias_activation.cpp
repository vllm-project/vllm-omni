/*
 * Copyright (c) 2024 NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include <torch/extension.h>

extern "C" torch::Tensor fwd_cuda(
    torch::Tensor const &input,
    torch::Tensor const &up_filter,
    torch::Tensor const &down_filter,
    torch::Tensor const &alpha,
    torch::Tensor const &beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &fwd_cuda, "Anti-alias activation forward (CUDA)");
}
