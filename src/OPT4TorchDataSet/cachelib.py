import math
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "OPTContext",
    "OPTCacheDecorator",
    "make_opt_cache",  # 工厂函数
]

class OPTContext:
    """未来访问序列上下文 (无全局状态)。"""

    def __init__(self, sampler, generator, total_iter: int, prediction_window: int):
        future_index = list(
            sampler(
                [None] * total_iter,
                replacement=True,
                num_samples=total_iter,
                generator=generator,
            )
        )
        future_map: Dict[Any, List[int]] = defaultdict(list)
        for pos, k in enumerate(future_index):
            future_map[k].append(pos)

        self.future_index: List[int] = future_index
        self.future_map: Dict[Any, List[int]] = future_map
        # key -> pointer (下一次尚未消费的位置在 positions 列表中的下标)
        self.future_ptr: Dict[Any, int] = defaultdict(int)
        self.total_iter = total_iter
        self.prediction_window = prediction_window

    def next_occurrence_distance(self, key: Any, current: int) -> Optional[int]:
        """返回 key 下一次出现距离; 不再出现则返回 None。"""
        positions = self.future_map.get(key, [])
        ptr = self.future_ptr[key]
        # 跳过已过去的位置
        while ptr < len(positions) and positions[ptr] < current:
            ptr += 1
        self.future_ptr[key] = ptr
        if ptr >= len(positions):
            return None
            
        distance = positions[ptr] - current
        # 如果距离超出窗口大小，则返回None（视为无穷大）
        if distance > self.prediction_window:
            return None
        return distance

class OPTCacheDecorator:
    """Belady(OPT) 缓存淘汰装饰器。"""

    def __init__(
        self,
        context: OPTContext,
        maxsize: int = 1,
        *,
        assert_alignment: bool = False,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize 必须为正整数")
        self.ctx = context
        self.maxsize = maxsize
        self.assert_alignment = assert_alignment

        # 运行期状态
        self._current: int = 0  # 已访问次数 (相当于未来序列当前位置)
        self._cache: Dict[Any, Any] = {}
        # 统计
        self._hits: int = 0
        self._miss: int = 0
        self._evict: int = 0
        self._future_exhaust: int = 0  # 遇到 key 无未来访问时淘汰

    # ---------------- Public API ----------------
    def stats(self) -> Dict[str, Any]:  # 简易统计
        total = self._hits + self._miss
        hit_rate = (self._hits / total) if total else 0.0
        return {
            "current": self._current,
            "size": len(self._cache),
            "capacity": self.maxsize,
            "hits": self._hits,
            "miss": self._miss,
            "evict": self._evict,
            "future_exhaust": self._future_exhaust,
            "hit_rate": hit_rate,
        }

    def reset_runtime(self) -> None:
        """重置运行期统计与缓存。"""
        self._current = 0
        self._cache.clear()
        self._hits = self._miss = self._evict = self._future_exhaust = 0
        # 重置所有指针
        for k in list(self.ctx.future_ptr.keys()):
            self.ctx.future_ptr[k] = 0

    # 作为装饰器使用
    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self._current >= self.ctx.total_iter:
                raise IndexError(
                    f"访问次数超过预生成未来序列长度 total_iter={self.ctx.total_iter}"
                )

            # 默认约定: args[1] 为 key / index
            try:
                input_index = args[1]
            except IndexError:  # 兼容只传 key 的函数
                raise ValueError("被装饰函数需要至少两个参数: (self_or_obj, index)")

            if self.assert_alignment:
                expected = self.ctx.future_index[self._current]
                if expected != input_index:
                    raise AssertionError(
                        f"访问序列不对齐: position={self._current}, expected={expected}, got={input_index}"
                    )

            # 命中
            if input_index in self._cache:
                self._hits += 1
                result = self._cache[input_index]
                self._current += 1
                return result

            # 未命中 -> 真实计算
            self._miss += 1
            result = func(*args, **kwargs)

            # 缓存未满
            if len(self._cache) < self.maxsize:
                self._cache[input_index] = result
                self._current += 1
                return result

            # 缓存已满 -> Belady 选择
            victim_key = None
            victim_distance = -1  # 最大距离
            for k in self._cache.keys():
                dist = self.ctx.next_occurrence_distance(k, self._current)
                if dist is None:  # 不再出现或超出预测窗口, 立即选择
                    victim_key = k
                    victim_distance = math.inf
                    self._future_exhaust += 1
                    break
                if dist > victim_distance:
                    victim_key = k
                    victim_distance = dist

            # 淘汰与插入
            if victim_key is not None:
                self._cache.pop(victim_key, None)
                self._evict += 1
            self._cache[input_index] = result
            self._current += 1
            return result

        return wrapper

def make_opt_cache(sampler, generator, total_iter: int, *, maxsize: int, prediction_window: int, assert_alignment: bool = False) -> OPTCacheDecorator:
    """创建 OPT 上下文并返回装饰器。"""
    ctx = OPTContext(sampler, generator, total_iter, prediction_window)
    return OPTCacheDecorator(ctx, maxsize=maxsize, assert_alignment=assert_alignment)