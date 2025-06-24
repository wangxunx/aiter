#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {

void vector_add(torch::Tensor &out, torch::Tensor &input_vec_a, torch::Tensor &input_vec_b);

} // namespace aiter
