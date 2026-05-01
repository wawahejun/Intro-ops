from __future__ import annotations

import tilelang
import tilelang.language as T


@tilelang.jit
def copy_kernel(src, BLOCK_N: int, dtype):
    N = T.const("N")
    src: T.Tensor((N,), dtype)
    out = T.empty((N,), dtype)

    with T.Kernel(N // BLOCK_N, threads=256) as pid_n:
        T.copy(
            src[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N],
            out[pid_n * BLOCK_N : (pid_n + 1) * BLOCK_N],
        )

    return out
