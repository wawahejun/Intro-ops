from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PerformanceResult:
    operator: str
    backend: str
    shape: str
    dtype: str
    bytes: int
    flops: int
    runtime_ms: float
    torch_ms: float | None = None

    @property
    def gbytes_per_sec(self) -> float:
        if self.runtime_ms <= 0:
            return 0.0
        return self.bytes / (1024**3) / (self.runtime_ms / 1000.0)

    @property
    def gflops_per_sec(self) -> float:
        if self.runtime_ms <= 0:
            return 0.0
        return self.flops / 1e9 / (self.runtime_ms / 1000.0)

    @property
    def speedup(self) -> float | None:
        if self.torch_ms is None or self.runtime_ms <= 0:
            return None
        return self.torch_ms / self.runtime_ms
