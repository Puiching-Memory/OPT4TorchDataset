"""JIT-loadable C++ extension for shared atomic counters.

This module exposes `fetch_add`, `load`, and `store` backed by a CPU shared
``torch.LongTensor``. It is intended for early prototyping of cross-process
metadata (global sequence, hit/miss counters, etc.).
"""

from __future__ import annotations

import platform
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def _load_ext() -> Any:
    base_dir = Path(__file__).parent
    source = base_dir / "cpp" / "atomic_ext.cpp"
    if not source.exists():
        raise FileNotFoundError(f"Extension source missing: {source}")

    cxx_args = ["/std:c++17"] if platform.system() == "Windows" else ["-std=c++17"]

    return load(
        name="opt4_atomic_ext",
        sources=[str(source)],
        extra_cflags=cxx_args,
        verbose=False,
    )


class OPTCore:
    """Wrapper for the C++ OPTCore engine."""

    def __init__(
        self,
        meta: torch.Tensor,
        decision_table: torch.Tensor,
        slot_map: torch.Tensor,
        free_slots: torch.Tensor,
        pool: torch.Tensor,
    ):
        self._core = _load_ext().OPTCore(
            meta, decision_table, slot_map, free_slots, pool
        )

    def execute_step(self, input_index: int) -> int:
        return self._core.execute_step(input_index)

    def update_cache(self, slot: int, data: torch.Tensor) -> None:
        self._core.update_cache(slot, data)

    def start_prefetch(
        self, loader_func, lookahead: int, future_index: torch.Tensor
    ) -> None:
        self._core.start_prefetch(loader_func, lookahead, future_index)

    def stop_prefetch(self) -> None:
        self._core.stop_prefetch()


def build_opt_plan(future_index: torch.Tensor, maxsize: int) -> torch.Tensor:
    """Build OPT eviction plan and decision table using C++."""
    return _load_ext().build_opt_plan(future_index, maxsize)


__all__ = ["OPTCore", "build_opt_plan"]
