from __future__ import annotations

import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def softmax_kernel(src, BLOCK_N: int, BLOCK_M: int):
    log2_e = 1.44269504
    N, M = T.const("N, M")
    dtype = T.float32
    src: T.Tensor((N, M), dtype)
    out = T.empty((N, M), dtype)

    with T.Kernel(N // BLOCK_N, threads=256) as pid_n:
        src_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        out_local = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        cur_exp = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        cur_max = T.alloc_fragment((BLOCK_N,), dtype)
        cur_sum = T.alloc_fragment((BLOCK_N,), dtype)
        lse = T.alloc_fragment((BLOCK_N,), dtype)

        T.fill(lse, -T.infinity(dtype))

        for m_blk in T.Serial(M // BLOCK_M):
            T.copy(src[pid_n * BLOCK_N, m_blk * BLOCK_M], src_local, disable_tma=True)
            T.reduce_max(src_local, cur_max, dim=1, clear=True)

            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                cur_exp[i, j] = T.exp2(src_local[i, j] * log2_e - cur_max[i] * log2_e)

            T.reduce_sum(cur_exp, cur_sum, dim=1, clear=True)

            for i in T.Parallel(BLOCK_N):
                lse[i] = cur_max[i] * log2_e + T.log2(
                    T.exp2(lse[i] - cur_max[i] * log2_e) + cur_sum[i]
                )

        for m_blk in T.Serial(M // BLOCK_M):
            T.copy(src[pid_n * BLOCK_N, m_blk * BLOCK_M], src_local, disable_tma=True)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                out_local[i, j] = T.exp2(src_local[i, j] * log2_e - lse[i])
            T.copy(out_local, out[pid_n * BLOCK_N, m_blk * BLOCK_M])

    return out
