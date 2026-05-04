from __future__ import annotations

import importlib.util

import pytest
import torch

from operator_runtime_testing import require_cuda

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tilelang") is None,
    reason="tilelang is not installed",
)


@pytest.mark.parametrize(
    ("copy_fn_name", "copy_out_fn_name", "module_name"),
    [
        ("copy_eager", "copy_eager_", "ops.common.tilelang.eager_copy"),
        ("copy_lazy_out_idx", "copy_lazy_out_idx_", "ops.common.tilelang.lazy_out_idx_copy"),
    ],
)
def test_tilelang_copy_templates_match_torch(copy_fn_name, copy_out_fn_name, module_name) -> None:
    require_cuda()
    module = __import__(module_name, fromlist=[copy_fn_name, copy_out_fn_name])
    copy_fn = getattr(module, copy_fn_name)
    copy_out_fn = getattr(module, copy_out_fn_name)

    src = torch.randn((1024,), dtype=torch.float32, device="cuda")
    out = copy_fn(src)
    torch.testing.assert_close(out, src)

    user_out = torch.empty_like(src)
    returned = copy_out_fn(user_out, src)
    assert returned is user_out
    torch.testing.assert_close(user_out, src)


@pytest.mark.parametrize(
    ("copy_out_fn_name", "module_name", "message"),
    [
        ("copy_eager_", "ops.common.tilelang.eager_copy", "matching shapes"),
        ("copy_lazy_out_idx_", "ops.common.tilelang.lazy_out_idx_copy", "matching shapes"),
    ],
)
def test_tilelang_copy_templates_reject_mismatched_out(
    copy_out_fn_name,
    module_name,
    message,
) -> None:
    require_cuda()
    module = __import__(module_name, fromlist=[copy_out_fn_name])
    copy_out_fn = getattr(module, copy_out_fn_name)

    src = torch.randn((1024,), dtype=torch.float32, device="cuda")
    out = torch.empty((512,), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match=message):
        copy_out_fn(out, src)
