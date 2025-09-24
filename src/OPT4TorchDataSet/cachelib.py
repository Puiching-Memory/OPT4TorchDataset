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
    """管理 *未来访问序列* 的上下文, 消除原实现中的全局变量。

    典型使用:
        ctx = OPTContext(sampler, generator, total_iter)
        @OPTCacheDecorator(ctx, maxsize=256)
        def get_item(ds, idx): ...

    线程/进程: 当前为单实例非线程安全, 多 worker 请为每个 worker 各建一份。
    """

    def __init__(self, sampler, generator, total_iter: int):
        # 不再 deepcopy, 假设外部生成器种子已固定；如需隔离可在外部 clone。
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

    def next_occurrence_distance(self, key: Any, current: int) -> Optional[int]:
        """返回 key 距离 current 的下一次出现距离; 若不再出现返回 None.
        通过推进 self.future_ptr[key] 指针实现摊销 O(1)。"""
        positions = self.future_map.get(key, [])
        ptr = self.future_ptr[key]
        # 跳过已过去的位置
        while ptr < len(positions) and positions[ptr] < current:
            ptr += 1
        self.future_ptr[key] = ptr
        if ptr >= len(positions):
            return None
        return positions[ptr] - current

class OPTCacheDecorator:
    """为函数提供 Belady (OPT) 最优缓存淘汰的装饰器(实例化形式)。

    与原 @OPTCache 不同之处:
        1. 不使用模块级全局状态; 由 OPTContext 提供未来序列。
        2. 支持 stats() / reset()。
        3. 可选 assert_alignment 检查访问序列是否与预生成一致。
    """

    def __init__(
        self,
        context: OPTContext,
        maxsize: int = 1,
        *,
        assert_alignment: bool = False,
        name: Optional[str] = None,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize 必须为正整数")
        self.ctx = context
        self.maxsize = maxsize
        self.assert_alignment = assert_alignment
        self.name = name or f"OPTCache[{id(self)}]"

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
            "name": self.name,
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
        """不重建未来序列, 仅清理缓存与指针 (实验/测试用途)。"""
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
                if dist is None:  # 不再出现, 立即选择
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

def make_opt_cache(sampler, generator, total_iter: int, *, maxsize: int, assert_alignment: bool = False, name: Optional[str] = None) -> OPTCacheDecorator:
    """工厂: 一步创建上下文 + 装饰器实例。

    示例:
        opt_cache = make_opt_cache(sampler, gen, N, maxsize=256)
        @opt_cache
        def get_item(ds, idx): ...
    """
    ctx = OPTContext(sampler, generator, total_iter)
    return OPTCacheDecorator(ctx, maxsize=maxsize, assert_alignment=assert_alignment, name=name)