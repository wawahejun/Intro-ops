from __future__ import annotations

import pytest
import torch

from operator_runtime.ops.vector_add import prepare_vector_add
from operator_runtime.testing import assert_close, require_cuda


def test_prepared_vector_add_reuses_descriptor(backend):
    require_cuda()
    if backend != "nvidia":
        pytest.skip("descriptor lifecycle test targets C ABI backend")
    a = torch.randn((1024,), device="cuda")
    b = torch.randn((1024,), device="cuda")
    out = torch.empty_like(a)
    prepared = prepare_vector_add(out, a, b, backend=backend)
    try:
        prepared.run()
        prepared.run()
        assert_close(out, a + b, atol=1e-6, rtol=1e-6)
    finally:
        prepared.destroy()

