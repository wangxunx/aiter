/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include "hip_compat.h"
#include "vector_add.h"






__global__ void vector_add_kernel(float *out, const float *a, float *b, unsigned int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
    {
	    out[tid] = a[tid] + b[tid];
    }
}


namespace aiter {

void vector_add(torch::Tensor &out, torch::Tensor &input_vec_a, torch::Tensor &input_vec_b)
{
    TORCH_CHECK(input_vec_a.is_cuda() && input_vec_b.is_cuda() && out.is_cuda(),
                "All tensors must be CUDA tensors.");
    TORCH_CHECK(input_vec_a.dtype() == torch::kFloat32 &&
                input_vec_b.dtype() == torch::kFloat32 &&
                out.dtype() == torch::kFloat32,
                "All tensors must be of type float32.");
    unsigned int num_elements = input_vec_a.numel();
    TORCH_CHECK(num_elements == input_vec_b.numel() && num_elements == out.numel(),
                "All tensors must have the same number of elements.");

    dim3 block_size(256);
    dim3 grid_size((num_elements + block_size.x - 1) / block_size.x);
    vector_add_kernel<<<grid_size, block_size>>>(
        out.data_ptr<float>(),
        input_vec_a.data_ptr<float>(),
        input_vec_b.data_ptr<float>(),
        num_elements);
}

} // namespace aiter