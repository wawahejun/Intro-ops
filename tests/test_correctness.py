from __future__ import annotations

import pytest
import torch

from operator_runtime import copy, reduce_sum, softmax, vector_add
from operator_runtime.testing import assert_close, require_cuda
from tests.cases import copy as copy_cases
from tests.cases import reduce_sum as reduce_sum_cases
from tests.cases import softmax as softmax_cases
from tests.cases import vector_add as vector_add_cases


@pytest.mark.parametrize("case", copy_cases.correctness_cases(), ids=lambda c: c["name"])
def test_copy_correctness(case, backend):
    require_cuda()
    src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
    out = copy(src, backend=backend)
    assert_close(out, src, atol=case["atol"], rtol=case["rtol"])


@pytest.mark.parametrize("case", vector_add_cases.correctness_cases(), ids=lambda c: c["name"])
def test_vector_add_correctness(case, backend):
    require_cuda()
    a = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
    b = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
    out = vector_add(a, b, backend=backend)
    assert_close(out, a + b, atol=case["atol"], rtol=case["rtol"])


@pytest.mark.parametrize("case", reduce_sum_cases.correctness_cases(), ids=lambda c: c["name"])
def test_reduce_sum_correctness(case, backend):
    require_cuda()
    src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
    out = reduce_sum(src, dim=1, backend=backend)
    assert_close(out, torch.sum(src, dim=1), atol=case["atol"], rtol=case["rtol"])


@pytest.mark.parametrize("case", softmax_cases.correctness_cases(), ids=lambda c: c["name"])
def test_softmax_correctness(case, backend):
    require_cuda()
    src = torch.randn(case["shape"], dtype=case["dtype"], device="cuda")
    out = softmax(src, dim=1, backend=backend)
    assert_close(out, torch.softmax(src, dim=1), atol=case["atol"], rtol=case["rtol"])

