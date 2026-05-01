from __future__ import annotations

import tilelang
import tilelang.language as T


@tilelang.jit
def vector_add_kernel(a, b, BLOCK_N: int, dtype):
    N = T.const("N")
    a: T.Tensor((N,), dtype)
    b: T.Tensor((N,), dtype)
    out = T.empty((N,), dtype)

    with T.Kernel(N // BLOCK_N, threads=256) as pid_n:
        base = pid_n * BLOCK_N
        for i in T.Parallel(BLOCK_N):
            out[base + i] = a[base + i] + b[base + i]

    return out
