from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import RandomSampler
import safetensors.torch

__all__ = [
    "generate_precomputed_file",
    "OPTCacheDecorator",
    "SharedOPTCacheDecorator",
    "CachetoolsDecorator",
]

from .atomic_extension import OPTCore, build_opt_plan


def _extract_index(args: Tuple[Any, ...]) -> Any:
    """从装饰器参数中提取索引。"""
    # 处理两种情况：
    # 1. args = (self, index) - 当装饰器装饰的是未绑定方法时
    # 2. args = (index,) - 当装饰器装饰的是已绑定方法或函数时
    try:
        if len(args) >= 2:
            return args[1]
        elif len(args) == 1:
            return args[0]
        else:
            raise IndexError("empty args")
    except IndexError as exc:
        raise ValueError("被装饰函数需要至少一个参数作为索引") from exc


def generate_precomputed_file(
    dataset_size: int,
    total_iterations: int,
    persist_path: Union[str, Path],
    random_seed: int = 0,
    replacement: bool = True,
    maxsize: int = 1,
    distribution_seq: Optional[torch.Tensor] = None,
) -> None:
    # ... (validation logic same) ...
    if dataset_size <= 0:
        raise ValueError(f"dataset_size 必须为正整数，得到: {dataset_size}")

    if total_iterations <= 0:
        raise ValueError(f"total_iterations 必须为正整数，得到: {total_iterations}")

    if not replacement and total_iterations > dataset_size:
        raise ValueError(
            f"无放回采样时，total_iterations ({total_iterations}) "
            f"不能超过 dataset_size ({dataset_size})"
        )

    if maxsize < 0:
        raise ValueError(f"maxsize 必须为非负整数，得到: {maxsize}")

    # 生成访问序列
    if distribution_seq is not None:
        if len(distribution_seq) < total_iterations:
            raise ValueError(
                f"提供序列长度({len(distribution_seq)})小于 total_iterations({total_iterations})"
            )
        future_index = distribution_seq[:total_iterations].to(torch.int64)
    else:
        # 创建生成器并设置种子
        generator = torch.Generator()
        generator.manual_seed(random_seed)

        # 创建采样器 (默认均匀分布)
        sampler = RandomSampler(
            list(range(dataset_size)),
            replacement=replacement,
            num_samples=total_iterations,
            generator=generator,
        )
        future_index = torch.tensor(list(sampler), dtype=torch.int64)

    # 使用 C++ 预计算决策表
    decision_table = build_opt_plan(future_index, maxsize)

    # 保存文件
    target = Path(persist_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Store scalars as a 1D tensor: [maxsize, dataset_size, total_iterations, seed]
        config_tensor = torch.tensor(
            [maxsize, dataset_size, total_iterations, random_seed], dtype=torch.int64
        )

        tensors = {
            "future_index": future_index,
            "decision_table": decision_table,
            "config": config_tensor,
        }
        safetensors.torch.save_file(tensors, str(target))

    except OSError as e:
        raise OSError(f"无法创建或写入文件 {persist_path}: {e}")


def prepare_shared_tensors(
    precomputed_path: Union[str, Path],
    dataset_size: int,
    maxsize: int,
):
    """把预计算的列表数据转换为共享内存 Tensor。"""
    payload = safetensors.torch.load_file(str(precomputed_path))

    future_index = payload["future_index"]

    # 如果有预计算好的 decision_table 则直接用，否则用 C++ 现场计算
    if "decision_table" in payload:
        decision_table = payload["decision_table"]
    else:
        # Compatibility fallback or rebuild (usually safetensors will include it)
        decision_table = build_opt_plan(future_index, maxsize)

    # 2. 槽位映射 [dataset_size] -> slot_idx (-1 for missing)
    slot_map = torch.full((dataset_size,), -1, dtype=torch.int64)

    # 3. 共享空闲槽位栈
    free_slots = torch.arange(maxsize, dtype=torch.int64)

    # 4. 元数据 [16]
    # idx 0: global_idx, 1: hits, 2: misses, 3: evicts, 4: free_stack_top, 8: prefetch_cursor
    meta = torch.zeros(16, dtype=torch.int64)
    meta[4] = maxsize  # 初期栈顶在末尾

    # 设为共享内存
    future_index = future_index.share_memory_()
    decision_table = decision_table.share_memory_()
    slot_map = slot_map.share_memory_()
    free_slots = free_slots.share_memory_()
    meta = meta.share_memory_()

    return meta, decision_table, slot_map, free_slots, future_index


class SharedOPTCacheDecorator:
    """高性能跨进程 OPT 缓存装饰器。"""

    def __init__(
        self,
        precomputed_path: Union[str, Path],
        maxsize: int,
        dataset_size: int,
        item_shape: Tuple[int, ...],
        item_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> None:
        # ... existing validation ...
        if maxsize <= 0:
            raise ValueError("SharedOPTCacheDecorator requires maxsize > 0")

        self.maxsize = maxsize
        self.item_shape = item_shape
        self.item_dtype = item_dtype

        # 准备共享内存
        (
            self._meta,
            self._decision_table,
            self._slot_map,
            self._free_slots,
            self._future_index,
        ) = prepare_shared_tensors(precomputed_path, dataset_size, maxsize)

        self.total_iter = len(self._future_index)

        # 预分配数据池
        self._pool = torch.zeros(
            (maxsize, *item_shape), dtype=item_dtype
        ).share_memory_()

        self._core_instance = None

    @property
    def core(self):
        # ...
        if self._core_instance is None:
            self._core_instance = OPTCore(
                self._meta,
                self._decision_table,
                self._slot_map,
                self._free_slots,
                self._pool,
            )
        return self._core_instance

    def stats(self) -> Dict[str, Any]:
        # ...
        m = self._meta
        hits = int(m[1])
        miss = int(m[2])
        total = hits + miss
        return {
            "current": int(m[0]),
            "hits": hits,
            "miss": miss,
            "evict": int(m[3]),
            "hit_rate": (hits / total) if total else 0.0,
            "pool_free": int(m[4]),
            "pool_utilized": self.maxsize - int(m[4]),
        }

    def reset(self) -> None:
        """重置共享内存中的统计信息。"""
        self._meta.zero_()
        self._meta[4] = self.maxsize  # 重置空闲栈顶
        self._slot_map.fill_(-1)
        self._free_slots.copy_(torch.arange(self.maxsize, dtype=torch.int64))
        # 注意：_pool 不需要重置，因为逻辑上 index -> slot 已经被清空了

    def __call__(self, func: Callable):
        # 注意：这里不能在外部获取 self.core 或 pool，否则 C++ 对象会被捕获到闭包中导致 pickling 失败。
        # 必须全部在 wrapper 内部延迟获取。
        item_dtype = self.item_dtype

        @wraps(func)
        def wrapper(*args, **kwargs):
            core_obj = self.core  # 延迟加载/获取 C++ 引擎

            input_index = _extract_index(args)

            # 执行缓存逻辑 (C++ 侧处理原子同步)
            val = core_obj.execute_step(input_index)

            if val >= 0:
                return self._pool[val]

            data = func(*args, **kwargs)

            if val < -1:
                slot = -(val + 2)
                if not torch.is_tensor(data):
                    data_t = torch.as_tensor(data, dtype=item_dtype)
                else:
                    data_t = data
                core_obj.update_cache(slot, data_t)

            return data

        return wrapper


class CachetoolsDecorator:
    """A picklable decorator for cachetools caches."""

    def __init__(self, cache: Any) -> None:
        self.cache = cache

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            idx = _extract_index(args)

            if idx in self.cache:
                return self.cache[idx]
            result = func(*args, **kwargs)
            try:
                self.cache[idx] = result
            except ValueError:
                # Some caches might raise ValueError if item is too large
                pass
            return result

        return wrapper


class OPTCacheDecorator:
    """Belady(OPT) 缓存淘汰装饰器 (纯 Python 版，适用于单进程)。"""

    def __init__(
        self,
        precomputed_path: Union[str, Path],
        maxsize: int,
        total_iter: int,
        **kwargs,
    ) -> None:
        """
        初始化OPT缓存装饰器。

        Args:
            precomputed_path: 预计算文件路径（包含所有离线预计算结果）
            maxsize: 缓存容量（必须与预计算时使用的 maxsize 一致）
            total_iter: 总迭代次数（必须与预计算时的 total_iterations 一致）
        """
        if maxsize < 0:
            raise ValueError("maxsize 必须为非负整数")

        self.maxsize = maxsize
        self.total_iter = total_iter

        # 运行期状态 (统计信息)
        self._stats = [0, 0, 0, 0]  # hits, miss, evict, current_idx

        self._cache: Dict[Any, Any] = {}

        # 如果缓存大小为0，不需要加载预计算数据
        if maxsize == 0:
            self.future_index = []
            self._eviction_plan: List[Optional[Tuple[Any, bool]]] = []
            self._cache_decision: List[bool] = []
            self._release_on_exit: List[Optional[Any]] = []
            return

        # 加载离线预计算数据
        payload = safetensors.torch.load_file(str(precomputed_path))
        self.future_index = payload["future_index"].tolist()
        if "decision_table" in payload:
            # Optimized C++ generated table
            dt = payload["decision_table"]
            self._cache_decision = (dt[:, 0] == 1).tolist()

            # Reconstruct eviction_plan and release from decision table
            self._eviction_plan = [None] * len(dt)
            dt_list = dt.tolist()

            self._eviction_plan = [
                (int(row[1]), False) if row[1] != -1 else None for row in dt_list
            ]
            self._release_on_exit = [
                int(row[2]) if row[2] != -1 else None for row in dt_list
            ]
        else:
            # Should not happen with new generator
            raise ValueError("Safetensors file missing 'decision_table'")

        # 验证预计算数据长度
        if len(self.future_index) < total_iter:
            raise ValueError(
                f"预计算结果长度不足。预计算长度: {len(self.future_index)}, "
                f"所需长度: {total_iter}。请重新生成预计算文件。"
            )

    def stats(self) -> Dict[str, Any]:
        """返回缓存统计信息。"""
        hits = int(self._stats[0])
        miss = int(self._stats[1])
        total = hits + miss
        hit_rate = (hits / total) if total else 0.0

        return {
            "current": int(self._stats[3]),
            "size": len(self._cache),
            "capacity": self.maxsize,
            "hits": hits,
            "miss": miss,
            "evict": int(self._stats[2]),
            "hit_rate": hit_rate,
        }

    def reset(self) -> None:
        """重置运行期状态。"""
        self._stats = [0, 0, 0, 0]
        self._cache.clear()

    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.maxsize == 0:
                return func(*args, **kwargs)
            return self._execute(func, args, kwargs)

        return wrapper

    def _handle_release(self, current_pos: int) -> None:
        """处理预计算的释放操作。"""
        release_key = self._release_on_exit[current_pos]
        if release_key is not None:
            removed = self._cache.pop(release_key, None)
            if removed is not None:
                self._stats[2] += 1  # evict

    def _execute(
        self,
        func: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        # 递增全局索引
        current_pos = self._stats[3]
        self._stats[3] += 1

        if current_pos >= self.total_iter:
            raise IndexError(
                f"访问次数超过预生成未来序列长度 total_iter={self.total_iter}"
            )

        input_index = _extract_index(args)

        # 缓存命中
        if input_index in self._cache:
            self._stats[0] += 1  # hits
            result = self._cache[input_index]
            self._handle_release(current_pos)
            return result

        # 缓存未命中
        self._stats[1] += 1  # miss
        result = func(*args, **kwargs)

        # 根据预计算决策判断是否需要缓存
        if self._cache_decision[current_pos]:
            # 如果缓存已满，淘汰受害者
            plan_entry = self._eviction_plan[current_pos]
            if plan_entry is not None:
                victim_key, _ = plan_entry
                evicted = self._cache.pop(victim_key, None)
                if evicted is not None:
                    self._stats[2] += 1  # evict

            # 缓存新项
            self._cache[input_index] = result

        # 处理需要释放的键（预计算）
        self._handle_release(current_pos)
        return result
