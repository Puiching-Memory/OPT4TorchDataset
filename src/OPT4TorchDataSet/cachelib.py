import heapq
import pickle
from collections import defaultdict
from functools import wraps
from itertools import count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import RandomSampler

__all__ = [
    "generate_precomputed_file",
    "OPTCacheDecorator",
]


def generate_precomputed_file(
    dataset_size: int,
    total_iterations: int,
    persist_path: Union[str, Path],
    random_seed: int = 0,
    replacement: bool = True,
    maxsize: int = 1,
) -> None:
    """
    生成OPT缓存完整的离线预计算文件。
    
    Args:
        dataset_size (int): 数据集大小（索引范围：0 到 dataset_size-1）
        total_iterations (int): 总迭代次数
        persist_path (Union[str, Path]): 保存路径
        random_seed (int, optional): 随机种子。默认为0
        replacement (bool, optional): 是否有放回采样。默认为True
        maxsize (int, optional): 缓存容量。默认为1。用于预计算淘汰计划。
    
    Raises:
        ValueError: 当参数无效时
        OSError: 当文件无法创建或写入时
    
    Example:
        >>> generate_precomputed_file(
        ...     dataset_size=10000,
        ...     total_iterations=100000,
        ...     persist_path="precomputed/my_experiment.pkl",
        ...     maxsize=3000
        ... )
    """
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
    
    # 创建生成器并设置种子
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    
    # 创建采样器
    sampler = RandomSampler(
        list(range(dataset_size)),
        replacement=replacement,
        num_samples=total_iterations,
        generator=generator
    )
    
    # 生成访问序列
    future_index = list(sampler)
    
    # 构建未来映射 - 记录每个键的所有访问位置
    future_map: Dict[Any, List[int]] = defaultdict(list)
    for pos, key in enumerate(future_index):
        future_map[key].append(pos)
    
    # 计算下次出现位置 - 预计算每个位置的下次访问时刻
    next_occurrence: List[Optional[int]] = [None] * total_iterations
    last_seen: Dict[Any, int] = {}
    for position in range(total_iterations - 1, -1, -1):
        key = future_index[position]
        next_occurrence[position] = last_seen.get(key)
        last_seen[key] = position
    
    # 离线预计算淘汰计划、缓存决策和释放计划
    eviction_plan, cache_decision, release_on_exit = _build_eviction_plan_offline(
        dataset_size=dataset_size,
        maxsize=maxsize,
        total_iterations=total_iterations,
        future_index=future_index,
        next_occurrence=next_occurrence,
    )
    
    # 保存文件
    target = Path(persist_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(
                {
                    "future_index": future_index,
                    "future_map": dict(future_map),
                    "next_occurrence": next_occurrence,
                    "eviction_plan": eviction_plan,
                    "cache_decision": cache_decision,
                    "release_on_exit": release_on_exit,
                    "maxsize": maxsize,
                    "total_iterations": total_iterations,
                    "seed": random_seed,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except OSError as e:
        raise OSError(f"无法创建或写入文件 {persist_path}: {e}")


def _build_eviction_plan_offline(
    dataset_size: int,
    maxsize: int,
    total_iterations: int,
    future_index: List[Any],
    next_occurrence: List[Optional[int]],
) -> Tuple[List[Optional[Tuple[Any, bool]]], List[bool], List[Optional[Any]]]:
    """
    离线预计算淘汰计划、缓存决策以及释放计划。
    
    Args:
        dataset_size: 数据集大小
        maxsize: 缓存容量
        total_iterations: 总迭代次数
        future_index: 未来访问序列
        next_occurrence: 每个位置的下次出现位置
        
    Returns:
        Tuple of (eviction_plan, cache_decision, release_on_exit)
    """
    plan: List[Optional[Tuple[Any, bool]]] = [None] * total_iterations
    cache_decision: List[bool] = [False] * total_iterations
    release_on_exit: List[Optional[Any]] = [None] * total_iterations

    if maxsize == 0:
        return plan, cache_decision, release_on_exit

    cache_next: Dict[Any, int] = {}
    heap: List[Tuple[int, int, Any]] = []
    ticket = count()

    for position in range(total_iterations):
        key = future_index[position]
        next_pos = next_occurrence[position]

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

        if len(cache_next) >= maxsize:
            victim_key: Optional[Any] = None
            while heap:
                neg_value, _, candidate_key = heapq.heappop(heap)
                candidate_value = -neg_value
                if candidate_key not in cache_next:
                    continue
                if cache_next[candidate_key] != candidate_value:
                    continue
                victim_key = candidate_key
                break
            if victim_key is None:
                raise RuntimeError("未能找到可淘汰的元素")
            cache_next.pop(victim_key, None)
            plan[position] = (victim_key, False)

        cache_next[key] = next_pos
        cache_decision[position] = True
        heapq.heappush(heap, (-next_pos, next(ticket), key))

    return plan, cache_decision, release_on_exit


def _load_precomputed_data(path: Union[str, Path]) -> Tuple[List[Any], List[Optional[int]], List[Optional[Tuple[Any, bool]]], List[bool], List[Optional[Any]]]:
    """
    加载离线预计算数据。
    
    Returns:
        Tuple of (future_index, next_occurrence, eviction_plan, cache_decision, release_on_exit)
    """
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    
    return (
        payload["future_index"],
        payload["next_occurrence"],
        payload["eviction_plan"],
        payload["cache_decision"],
        payload["release_on_exit"],
    )


class OPTCacheDecorator:
    """Belady(OPT) 缓存淘汰装饰器。"""

    def __init__(
        self,
        precomputed_path: Union[str, Path],
        maxsize: int,
        total_iter: int,
    ) -> None:
        """
        初始化OPT缓存装饰器。
        
        Args:
            precomputed_path: 预计算文件路径（包含所有离线预计算结果）
            maxsize: 缓存容量（必须与预计算时使用的 maxsize 一致）
            total_iter: 总迭代次数（必须与预计算时的 total_iterations 一致）
            
        Raises:
            ValueError: 当 maxsize 为负或预计算数据长度不足时
        """
        if maxsize < 0:
            raise ValueError("maxsize 必须为非负整数")
        
        self.maxsize = maxsize
        self.total_iter = total_iter
        
        # 运行期状态
        self._current = 0
        self._cache: Dict[Any, Any] = {}
        self._hits = 0
        self._miss = 0
        self._evict = 0
        
        # 如果缓存大小为0，不需要加载预计算数据
        if maxsize == 0:
            self.future_index = []
            self._next_occurrence = []
            self._eviction_plan: List[Optional[Tuple[Any, bool]]] = []
            self._cache_decision: List[bool] = []
            self._release_on_exit: List[Optional[Any]] = []
            return
        
        # 加载离线预计算数据（无需运行时计算）
        (
            self.future_index,
            self._next_occurrence,
            self._eviction_plan,
            self._cache_decision,
            self._release_on_exit,
        ) = _load_precomputed_data(precomputed_path)
        
        # 验证预计算数据长度
        if len(self.future_index) < total_iter:
            raise ValueError(
                f"预计算结果长度不足。预计算长度: {len(self.future_index)}, "
                f"所需长度: {total_iter}。请重新生成预计算文件。"
            )

    def stats(self) -> Dict[str, Any]:
        """返回缓存统计信息。"""
        total_hits = self._hits
        total_miss = self._miss
        total = total_hits + total_miss
        hit_rate = (total_hits / total) if total else 0.0

        return {
            "current": self._current,
            "size": len(self._cache),
            "capacity": self.maxsize,
            "hits": total_hits,
            "miss": total_miss,
            "evict": self._evict,
            "hit_rate": hit_rate,
        }

    def reset(self) -> None:
        """重置运行期状态。"""
        self._current = 0
        self._cache.clear()
        self._hits = self._miss = self._evict = 0

    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.maxsize == 0:
                return func(*args, **kwargs)
            return self._single_worker_execute(func, args, kwargs)

        return wrapper

    def _handle_release(self, current_pos: int) -> None:
        """处理预计算的释放操作。"""
        release_key = self._release_on_exit[current_pos]
        if release_key is not None:
            removed = self._cache.pop(release_key, None)
            if removed is not None:
                self._evict += 1

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
        current_pos = self._current

        # 缓存命中
        if input_index in self._cache:
            self._hits += 1
            result = self._cache[input_index]
            self._handle_release(current_pos)
            self._current += 1
            return result

        # 缓存未命中
        self._miss += 1
        result = func(*args, **kwargs)

        # 根据预计算决策判断是否需要缓存
        if self._cache_decision[current_pos]:
            # 如果缓存已满，淘汰受害者
            plan_entry = self._eviction_plan[current_pos]
            if plan_entry is not None:
                victim_key, _ = plan_entry
                evicted = self._cache.pop(victim_key, None)
                if evicted is not None:
                    self._evict += 1
            
            # 缓存新项
            self._cache[input_index] = result

        # 处理需要释放的键（预计算）
        self._handle_release(current_pos)
        self._current += 1
        return result