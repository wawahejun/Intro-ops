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
def _reduce_sum_kernel(src, BLOCK_N: int, BLOCK_M: int):
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


@lru_cache(maxsize=32)
def _compiled_reduce_sum(n: int, m: int, block_n: int, block_m: int):
    return _reduce_sum_kernel.compile(N=n, M=m, BLOCK_N=block_n, BLOCK_M=block_m)


def _blocks(n: int, m: int) -> tuple[int, int]:
    block_n = 16 if n % 16 == 0 else 1
    block_m = 128 if m % 128 == 0 else m
    return block_n, block_m


@dataclass
class TileLangReduceSumPrepared:
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


def prepare_reduce_sum_tl(out: torch.Tensor, src: torch.Tensor, dim: int = 1) -> TileLangReduceSumPrepared:
    if dim != 1:
        raise ValueError("TileLang reduce_sum v1 supports dim=1")
    n, m = src.shape
    block_n, block_m = _blocks(n, m)
    if n % block_n != 0 or m % block_m != 0:
        raise ValueError("TileLang reduce_sum v1 requires divisible block sizes")
    kernel = _compiled_reduce_sum(n, m, block_n, block_m)
    return TileLangReduceSumPrepared(out, src, dim, kernel)


def reduce_sum_tl_(out: torch.Tensor, src: torch.Tensor, dim: int = 1) -> torch.Tensor:
    with prepare_reduce_sum_tl(out, src, dim) as prepared:
        prepared.run()
    return out


def reduce_sum_tl(src: torch.Tensor, dim: int = 1) -> torch.Tensor:
    out = torch.empty((src.shape[0],), dtype=src.dtype, device=src.device)
    return reduce_sum_tl_(out, src, dim)
