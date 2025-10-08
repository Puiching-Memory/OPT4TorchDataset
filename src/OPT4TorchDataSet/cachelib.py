import heapq
import multiprocessing as mp
from multiprocessing.managers import SyncManager
import pickle
import warnings
from collections import defaultdict
from enum import Enum
from functools import wraps
from itertools import count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:  # torch is optional for some utility functions
    import torch
except ImportError:  # pragma: no cover - torch is expected to be installed in runtime env
    torch = None

__all__ = [
    "precompute_opt_indices",
    "generate_precomputed_file",
    "OPTCacheDecorator",
]


class _CacheEntryState(Enum):
    READY = 1
    PENDING = 2


_GLOBAL_MANAGER: Optional[SyncManager] = None


def _get_sync_manager() -> SyncManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = mp.Manager()
    return _GLOBAL_MANAGER


def _maybe_share_tensor(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if not tensor.is_shared():
        tensor.share_memory_()
    return tensor


def _prepare_for_cache(value: Any) -> Any:
    if torch is None:
        return value

    if torch.is_tensor(value):
        return _maybe_share_tensor(value)

    if isinstance(value, (list, tuple)):
        prepared = [_prepare_for_cache(item) for item in value]
        return type(value)(prepared)

    if isinstance(value, dict):
        return {k: _prepare_for_cache(v) for k, v in value.items()}

    return value


class _SharedOPTState:
    __slots__ = (
        "maxsize",
        "cache",
        "condition",
        "current",
        "hits",
        "miss",
        "evict",
    )

    def __init__(self, maxsize: int) -> None:
        manager = _get_sync_manager()
        self.maxsize = maxsize
        self.cache = manager.dict()  # type: ignore[assignment]
        self.condition = manager.Condition()  # type: ignore[assignment]
        self.current = manager.Value("Q", 0)
        self.hits = manager.Value("Q", 0)
        self.miss = manager.Value("Q", 0)
        self.evict = manager.Value("Q", 0)

    def reset(self) -> None:
        with self.condition:
            self.cache.clear()
            self.current.value = 0
            self.hits.value = 0
            self.miss.value = 0
            self.evict.value = 0
            self.condition.notify_all()


def precompute_opt_indices(
    sampler: Callable,
    generator,
    total_iter: int,
    persist_path: Union[str, Path],
    seed: Optional[int] = None,
) -> None:
    """CLI 预计算 OPT 缓存索引。
    
    Args:
        sampler: 采样函数
        generator: 随机数生成器
        total_iter: 总迭代次数
        persist_path: 保存路径
        seed: 随机种子
    """
    future_index = list(
        sampler(
            [None] * total_iter,
            replacement=True,
            num_samples=total_iter,
            generator=generator,
        )
    )
    
    # 构建未来映射
    future_map: Dict[Any, List[int]] = defaultdict(list)
    for pos, key in enumerate(future_index):
        future_map[key].append(pos)
    
    # 保存到文件
    target = Path(persist_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(
            {
                "future_index": future_index,
                "future_map": dict(future_map),
                "seed": seed,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _load_precomputed_data(path: Union[str, Path], expected_seed: Optional[int] = None) -> Tuple[List[Any], Dict[Any, List[int]]]:
    """加载预计算数据。
    
    Args:
        path: 预计算文件路径
        expected_seed: 期望的种子值，如果提供则会进行验证
        
    Raises:
        ValueError: 当种子不匹配时抛出异常
    """
    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    
    # 检查种子是否一致
    if expected_seed is not None:
        saved_seed = payload.get("seed")
        if saved_seed != expected_seed:
            raise ValueError(
                f"预计算文件种子与当前种子不一致。预计算文件种子: {saved_seed}, "
                f"当前种子: {expected_seed}。请确保使用相同的种子或重新生成预计算文件。"
            )
    
    return payload["future_index"], payload["future_map"]


def generate_precomputed_file(
    dataset_size: int,
    total_iterations: int,
    persist_path: Union[str, Path],
    random_seed: int = 0,
    replacement: bool = True,
) -> None:
    """
    生成OPT缓存预计算文件的便捷API。
    
    这个函数创建一个包含访问序列和未来映射的预计算文件，
    用于OPT缓存装饰器。生成的文件与CLI工具的效果完全相同。
    
    Args:
        dataset_size (int): 数据集大小（索引范围：0 到 dataset_size-1）
        total_iterations (int): 总迭代次数
        persist_path (Union[str, Path]): 保存路径
        random_seed (int, optional): 随机种子。默认为0
        replacement (bool, optional): 是否有放回采样。默认为True
    
    Raises:
        ImportError: 当torch库未安装时
        ValueError: 当参数无效时（如无放回采样时total_iterations > dataset_size）
        OSError: 当文件无法创建或写入时
    
    Example:
        基本使用:
        >>> generate_precomputed_file(
        ...     dataset_size=10000,
        ...     total_iterations=100000,
        ...     persist_path="precomputed/my_experiment.pkl"
        ... )
        
        自定义参数:
        >>> generate_precomputed_file(
        ...     dataset_size=5000,
        ...     total_iterations=25000,
        ...     persist_path="data/custom_opt.pkl",
        ...     random_seed=42,
        ...     replacement=True
        ... )
    """
    # 参数验证
    if dataset_size <= 0:
        raise ValueError(f"dataset_size 必须为正整数，得到: {dataset_size}")
    
    if total_iterations <= 0:
        raise ValueError(f"total_iterations 必须为正整数，得到: {total_iterations}")
    
    if not replacement and total_iterations > dataset_size:
        raise ValueError(
            f"无放回采样时，total_iterations ({total_iterations}) "
            f"不能超过 dataset_size ({dataset_size})"
        )
    
    try:
        import torch
        from torch.utils.data import RandomSampler
    except ImportError:
        raise ImportError(
            "需要安装 torch 库才能使用此功能。"
            "请运行: pip install torch"
        )
    
    # 创建生成器并设置种子
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    
    # 创建数据集索引
    dataset_indices = list(range(dataset_size))
    
    # 创建采样器
    sampler = RandomSampler(
        dataset_indices,
        replacement=replacement,
        num_samples=total_iterations,
        generator=generator
    )
    
    # 生成访问序列
    future_index = list(sampler)
    
    # 构建未来映射
    future_map: Dict[Any, List[int]] = defaultdict(list)
    for pos, key in enumerate(future_index):
        future_map[key].append(pos)
    
    # 确保目录存在并保存文件
    target = Path(persist_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(
                {
                    "future_index": future_index,
                    "future_map": dict(future_map),
                    "seed": random_seed,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except OSError as e:
        raise OSError(f"无法创建或写入文件 {persist_path}: {e}")


class OPTCacheDecorator:
    """Belady(OPT) 缓存淘汰装饰器。"""

    def __init__(
        self,
        precomputed_path: Union[str, Path],
        maxsize: int,
        total_iter: int,
        seed: Optional[int] = None,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize 必须为正整数")
        
        # 加载预计算数据
        self.future_index, self.future_map = _load_precomputed_data(precomputed_path, seed)
        
        if len(self.future_index) < total_iter:
            raise ValueError("预计算结果长度不足，无法覆盖 total_iter")
        
        self.future_index = self.future_index[:total_iter]
        self.maxsize = maxsize
        self.total_iter = total_iter
        
        # 计算下次出现位置
        self._next_occurrence = self._compute_next_occurrences()
        
        # 预计算淘汰计划与缓存决策
        (
            self._eviction_plan,
            self._cache_decision,
            self._release_on_exit,
        ) = self._build_eviction_plan()
        
        # 运行期状态
        self._current = 0
        self._cache: Dict[Any, Any] = {}
        self._hits = 0
        self._miss = 0
        self._evict = 0
        self._enable_multi_worker = True
        self._shared_state = _SharedOPTState(maxsize)

    def _compute_next_occurrences(self) -> List[Optional[int]]:
        """计算每个位置的下次出现位置。"""
        next_occurrence: List[Optional[int]] = [None] * self.total_iter
        last_seen: Dict[Any, int] = {}
        for position in range(self.total_iter - 1, -1, -1):
            key = self.future_index[position]
            next_occurrence[position] = last_seen.get(key)
            last_seen[key] = position
        return next_occurrence

    def _build_eviction_plan(self) -> Tuple[List[Optional[Tuple[Any, bool]]], List[bool], List[Optional[Any]]]:
        """构建淘汰计划、缓存决策以及按访问释放计划。"""
        plan: List[Optional[Tuple[Any, bool]]] = [None] * self.total_iter
        cache_decision: List[bool] = [False] * self.total_iter
        release_on_exit: List[Optional[Any]] = [None] * self.total_iter
        if self.maxsize == 0:
            return plan, cache_decision, release_on_exit

        cache_next: Dict[Any, int] = {}
        heap: List[Tuple[int, int, Any]] = []
        ticket = count()

        for position in range(self.total_iter):
            key = self.future_index[position]
            next_pos = self._next_occurrence[position]

            if key in cache_next:
                if next_pos is None:
                    cache_next.pop(key, None)
                    release_on_exit[position] = key
                else:
                    cache_next[key] = next_pos
                    heapq.heappush(heap, (-next_pos, next(ticket), key))
                continue

            if next_pos is None:
                # 不会再访问，跳过缓存
                continue

            if len(cache_next) >= self.maxsize:
                victim_key: Optional[Any] = None
                victim_value: Optional[int] = None
                while heap:
                    neg_value, _, candidate_key = heapq.heappop(heap)
                    candidate_value = -neg_value
                    if candidate_key not in cache_next:
                        continue
                    if cache_next[candidate_key] != candidate_value:
                        continue
                    victim_key = candidate_key
                    victim_value = candidate_value
                    break
                if victim_key is None:
                    raise RuntimeError("未能找到可淘汰的元素")
                cache_next.pop(victim_key, None)
                plan[position] = (victim_key, False)

            cache_next[key] = next_pos
            cache_decision[position] = True
            heapq.heappush(heap, (-next_pos, next(ticket), key))

        return plan, cache_decision, release_on_exit

    def stats(self) -> Dict[str, Any]:
        """返回缓存统计信息。"""
        shared_hits = int(self._shared_state.hits.value)
        shared_miss = int(self._shared_state.miss.value)
        shared_evict = int(self._shared_state.evict.value)
        total_hits = self._hits + shared_hits
        total_miss = self._miss + shared_miss
        total = total_hits + total_miss
        hit_rate = (total_hits / total) if total else 0.0
        shared_cache_size = 0
        for entry in list(self._shared_state.cache.values()):
            if isinstance(entry, tuple) and entry and entry[0] == _CacheEntryState.READY.value:
                shared_cache_size += 1

        return {
            "current": self._current + int(self._shared_state.current.value),
            "size": len(self._cache) + shared_cache_size,
            "capacity": self.maxsize,
            "hits": total_hits,
            "miss": total_miss,
            "evict": self._evict + shared_evict,
            "hit_rate": hit_rate,
        }

    def reset(self) -> None:
        """重置运行期状态。"""
        self._current = 0
        self._cache.clear()
        self._hits = self._miss = self._evict = 0
        self._shared_state.reset()

    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.maxsize == 0:
                return func(*args, **kwargs)

            worker_info = None
            if self._enable_multi_worker:
                try:
                    from torch.utils.data import get_worker_info  # type: ignore
                except (ImportError, RuntimeError):
                    worker_info = None
                else:
                    worker_info = get_worker_info()

            if worker_info is None:
                return self._single_worker_execute(func, args, kwargs)

            if torch is None:
                warnings.warn(
                    "检测到多 Worker DataLoader，但当前环境缺少 torch 库，OPT 缓存将退回单Worker模式。",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._single_worker_execute(func, args, kwargs)

            return self._multi_worker_execute(func, args, kwargs)

        return wrapper

    @staticmethod
    def _extract_index(args: Tuple[Any, ...]) -> Any:
        try:
            return args[1]
        except IndexError as exc:
            raise ValueError("被装饰函数需要至少两个参数: (self_or_obj, index)") from exc

    def _single_worker_execute(
        self,
        func: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if self._current >= self.total_iter:
            raise IndexError(
                f"访问次数超过预生成未来序列长度 total_iter={self.total_iter}"
            )

        input_index = self._extract_index(args)

        if input_index in self._cache:
            self._hits += 1
            result = self._cache[input_index]
            release_key = (
                self._release_on_exit[self._current]
                if self.maxsize > 0
                else None
            )
            if release_key is not None:
                removed = self._cache.pop(release_key, None)
                if removed is not None:
                    self._evict += 1
            self._current += 1
            return result

        self._miss += 1
        result = func(*args, **kwargs)

        should_cache = self._cache_decision[self._current] if self.maxsize > 0 else False
        if should_cache and self.maxsize > 0:
            plan_entry = self._eviction_plan[self._current]
            if plan_entry is not None:
                victim_key, _ = plan_entry
                if victim_key is not None:
                    evicted = self._cache.pop(victim_key, None)
                    if evicted is not None:
                        self._evict += 1
            self._cache[input_index] = result

        release_key = (
            self._release_on_exit[self._current]
            if self.maxsize > 0
            else None
        )
        if release_key is not None:
            removed = self._cache.pop(release_key, None)
            if removed is not None:
                self._evict += 1

        self._current += 1
        return result

    def _multi_worker_execute(
        self,
        func: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        state = self._shared_state
        position: Optional[int] = None
        should_cache = False

        while True:
            with state.condition:
                if position is None:
                    current = state.current.value
                    if current >= self.total_iter:
                        raise IndexError(
                            f"访问次数超过预生成未来序列长度 total_iter={self.total_iter}"
                        )
                    state.current.value = current + 1
                    position = current

                input_index = self._extract_index(args)
                entry = state.cache.get(input_index)
                if entry is not None:
                    status, payload = entry
                    if status == _CacheEntryState.READY.value:
                        release_key = (
                            self._release_on_exit[position]
                            if self.maxsize > 0
                            else None
                        )
                        if release_key is not None:
                            existing = state.cache.pop(release_key, None)
                            if existing is not None:
                                state.evict.value += 1
                                state.condition.notify_all()
                        state.hits.value += 1
                        return payload
                    state.condition.wait()
                    continue

                state.miss.value += 1
                should_cache = (
                    self._cache_decision[position] if self.maxsize > 0 else False
                )
                if should_cache and self.maxsize > 0:
                    plan_entry = self._eviction_plan[position]
                    if plan_entry is not None:
                        victim_key, _ = plan_entry
                        if victim_key is not None:
                            while True:
                                victim_entry = state.cache.get(victim_key)
                                if victim_entry is None:
                                    break
                                victim_status, _ = victim_entry
                                if victim_status == _CacheEntryState.PENDING.value:
                                    state.condition.wait()
                                    continue
                                state.cache.pop(victim_key, None)
                                state.evict.value += 1
                                break

                if should_cache and self.maxsize > 0:
                    state.cache[input_index] = (_CacheEntryState.PENDING.value, None)
                break

        result = func(*args, **kwargs)
        cached_result = _prepare_for_cache(result)

        release_key = (
            self._release_on_exit[position] if self.maxsize > 0 else None
        )

        if should_cache and self.maxsize > 0:
            with state.condition:
                state.cache[input_index] = (
                    _CacheEntryState.READY.value,
                    cached_result,
                )
                if release_key is not None:
                    existing = state.cache.pop(release_key, None)
                    if existing is not None:
                        state.evict.value += 1
                state.condition.notify_all()
        elif release_key is not None:
            with state.condition:
                existing = state.cache.pop(release_key, None)
                if existing is not None:
                    state.evict.value += 1
                state.condition.notify_all()

        return cached_result