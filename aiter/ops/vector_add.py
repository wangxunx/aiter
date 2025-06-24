# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from ..jit.core import compile_ops


MD_NAME = "module_vectoradd"


@compile_ops("module_vectoradd")
def vector_add_cu(
    out: Tensor,
    input_a: Tensor,
    input_b: Tensor,
):
    """
    Cuda version of vector addition.
    """
    ...