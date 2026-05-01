from __future__ import annotations

import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def reduce_sum_kernel(src, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float32
    src: T.Tensor((N, M), dtype)
    out = T.empty((N,), dtype)

    with T.Kernel(N // BLOCK_N, threads=256) as pid_n:
        src_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        out_local = T.alloc_fragment((BLOCK_N,), dtype)
        T.clear(out_local)

        for m_blk in T.Serial(M // BLOCK_M):
            T.copy(src[pid_n * BLOCK_N, m_blk * BLOCK_M], src_local)
            T.reduce_sum(src_local, out_local, dim=1, clear=False)

        T.copy(out_local, out[pid_n * BLOCK_N])

    return out
