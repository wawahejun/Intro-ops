from __future__ import annotations

import pytest
import torch

from operator_runtime import copy_, reduce_sum, softmax, vector_add
from operator_runtime.testing import require_cuda


def test_copy_rejects_shape_mismatch(backend):
    require_cuda()
    src = torch.randn((16,), device="cuda")
    out = torch.empty((8,), device="cuda")
    with pytest.raises(ValueError):
        copy_(out, src, backend=backend)


def test_vector_add_rejects_shape_mismatch(backend):
    require_cuda()
    a = torch.randn((16,), device="cuda")
    b = torch.randn((8,), device="cuda")
    with pytest.raises(ValueError):
        vector_add(a, b, backend=backend)


def test_reduce_sum_rejects_wrong_dim(backend):
    require_cuda()
    src = torch.randn((16, 16), device="cuda")
    with pytest.raises(ValueError):
        reduce_sum(src, dim=0, backend=backend)


def test_softmax_rejects_wrong_dtype(backend):
    require_cuda()
    src = torch.randn((16, 16), dtype=torch.float16, device="cuda")
    with pytest.raises(TypeError):
        softmax(src, dim=1, backend=backend)

