from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def _softmax_kernel(src, BLOCK_N: int, BLOCK_M: int):
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
            T.copy(src[pid_n * BLOCK_N, m_blk * BLOCK_M], src_local)
            T.reduce_max(src_local, cur_max, dim=1, clear=True)

            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                cur_exp[i, j] = T.exp2(src_local[i, j] * log2_e - cur_max[i] * log2_e)

            T.reduce_sum(cur_exp, cur_sum, dim=1, clear=True)

            for i in T.Parallel(BLOCK_N):
                lse[i] = cur_max[i] * log2_e + T.log2(
                    T.exp2(lse[i] - cur_max[i] * log2_e) + cur_sum[i]
                )

        for m_blk in T.Serial(M // BLOCK_M):
            T.copy(src[pid_n * BLOCK_N, m_blk * BLOCK_M], src_local)
            for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                out_local[i, j] = T.exp2(src_local[i, j] * log2_e - lse[i])
            T.copy(out_local, out[pid_n * BLOCK_N, m_blk * BLOCK_M])

    return out


@lru_cache(maxsize=32)
def _compiled_softmax(n: int, m: int, block_n: int, block_m: int):
    return _softmax_kernel.compile(N=n, M=m, BLOCK_N=block_n, BLOCK_M=block_m)


def _blocks(n: int, m: int) -> tuple[int, int]:
    block_n = 16 if n % 16 == 0 else 1
    block_m = 256 if m % 256 == 0 else m
    return block_n, block_m


@dataclass
class TileLangSoftmaxPrepared:
    out: torch.Tensor
    src: torch.Tensor
    dim: int = 1
    kernel: object | None = None

    def run(self) -> None:
        assert self.kernel is not None
        self.out.copy_(self.kernel(self.src))

    def destroy(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()


def prepare_softmax_tl(out: torch.Tensor, src: torch.Tensor, dim: int = 1) -> TileLangSoftmaxPrepared:
    if dim != 1:
        raise ValueError("TileLang softmax v1 supports dim=1")
    n, m = src.shape
    block_n, block_m = _blocks(n, m)
    if n % block_n != 0 or m % block_m != 0:
        raise ValueError("TileLang softmax v1 requires divisible block sizes")
    kernel = _compiled_softmax(n, m, block_n, block_m)
    return TileLangSoftmaxPrepared(out, src, dim, kernel)


def softmax_tl_(out: torch.Tensor, src: torch.Tensor, dim: int = 1) -> torch.Tensor:
    with prepare_softmax_tl(out, src, dim) as prepared:
        prepared.run()
    return out


def softmax_tl(src: torch.Tensor, dim: int = 1) -> torch.Tensor:
    out = torch.empty_like(src)
    return softmax_tl_(out, src, dim)
