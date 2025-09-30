import heapq
import pickle
from collections import defaultdict
from functools import wraps
from itertools import count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

__all__ = [
    "precompute_opt_indices",
    "generate_precomputed_file",
    "OPTCacheDecorator",
]


def precompute_opt_indices(
    sampler: Callable,
    generator,
    total_iter: int,
    persist_path: Union[str, Path],
) -> None:
    """CLI 预计算 OPT 缓存索引。
    
    Args:
        sampler: 采样函数
        generator: 随机数生成器
        total_iter: 总迭代次数
        persist_path: 保存路径
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
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _load_precomputed_data(path: Union[str, Path]) -> Tuple[List[Any], Dict[Any, List[int]]]:
    """加载预计算数据。"""
    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
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
        prediction_window: int,
        total_iter: int,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize 必须为正整数")
        
        # 加载预计算数据
        self.future_index, self.future_map = _load_precomputed_data(precomputed_path)
        
        if len(self.future_index) < total_iter:
            raise ValueError("预计算结果长度不足，无法覆盖 total_iter")
        
        self.future_index = self.future_index[:total_iter]
        self.maxsize = maxsize
        self.prediction_window = prediction_window
        self.total_iter = total_iter
        
        # 计算下次出现位置
        self._next_occurrence = self._compute_next_occurrences()
        self._inf_marker = self.total_iter + self.prediction_window + 1
        
        # 预计算淘汰计划
        self._eviction_plan = self._build_eviction_plan()
        
        # 运行期状态
        self._current = 0
        self._cache: Dict[Any, Any] = {}
        self._hits = 0
        self._miss = 0
        self._evict = 0

    def _compute_next_occurrences(self) -> List[Optional[int]]:
        """计算每个位置的下次出现位置。"""
        next_occurrence: List[Optional[int]] = [None] * self.total_iter
        last_seen: Dict[Any, int] = {}
        for position in range(self.total_iter - 1, -1, -1):
            key = self.future_index[position]
            next_occurrence[position] = last_seen.get(key)
            last_seen[key] = position
        return next_occurrence

    def _encode_next_value(self, position: int, next_pos: Optional[int]) -> int:
        """编码下次访问位置。"""
        if next_pos is None:
            return self._inf_marker
        distance = next_pos - position
        if distance > self.prediction_window:
            return self._inf_marker
        return next_pos

    def _build_eviction_plan(self) -> List[Optional[Tuple[Any, bool]]]:
        """构建淘汰计划。"""
        plan: List[Optional[Tuple[Any, bool]]] = [None] * self.total_iter
        if self.maxsize == 0:
            return plan

        cache_next: Dict[Any, int] = {}
        heap: List[Tuple[int, int, Any]] = []
        ticket = count()

        for position in range(self.total_iter):
            key = self.future_index[position]
            next_pos = self._next_occurrence[position]
            next_value = self._encode_next_value(position, next_pos)

            if key in cache_next:
                cache_next[key] = next_value
                heapq.heappush(heap, (-next_value, next(ticket), key))
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
                future_exhaust = victim_value is None or victim_value >= self._inf_marker
                plan[position] = (victim_key, future_exhaust)

            cache_next[key] = next_value
            heapq.heappush(heap, (-next_value, next(ticket), key))

        return plan

    def stats(self) -> Dict[str, Any]:
        """返回缓存统计信息。"""
        total = self._hits + self._miss
        hit_rate = (self._hits / total) if total else 0.0
        return {
            "current": self._current,
            "size": len(self._cache),
            "capacity": self.maxsize,
            "hits": self._hits,
            "miss": self._miss,
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
            if self._current >= self.total_iter:
                raise IndexError(
                    f"访问次数超过预生成未来序列长度 total_iter={self.total_iter}"
                )

            # 获取访问的索引（假设是第二个参数）
            try:
                input_index = args[1]
            except IndexError:
                raise ValueError("被装饰函数需要至少两个参数: (self_or_obj, index)")

            # 命中检查
            if input_index in self._cache:
                self._hits += 1
                result = self._cache[input_index]
                self._current += 1
                return result

            # 未命中 - 执行实际计算
            self._miss += 1
            result = func(*args, **kwargs)

            # 根据淘汰计划执行淘汰
            plan_entry = self._eviction_plan[self._current]
            if plan_entry is not None:
                victim_key, _ = plan_entry
                if victim_key is not None:
                    if victim_key not in self._cache:
                        raise RuntimeError(
                            f"预计算淘汰计划与实际缓存不一致: position={self._current}, victim={victim_key}"
                        )
                    self._cache.pop(victim_key, None)
                    self._evict += 1
            elif len(self._cache) >= self.maxsize:
                raise RuntimeError(
                    f"OPT 淘汰计划缺失导致缓存溢出: position={self._current}, key={input_index}"
                )

            self._cache[input_index] = result
            self._current += 1
            return result

        return wrapper