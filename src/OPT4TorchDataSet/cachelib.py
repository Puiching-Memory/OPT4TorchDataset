import heapq
import copy
from functools import wraps


class OptimalCache:
    """
    OPT缓存实现
    复杂度为O(log n)
    """
    
    def __init__(self, capacity, future_access_map):
        self.capacity = capacity
        self.cache = {}
        self.future_access = future_access_map  # key -> [future_positions...]
        self.current_time = 0
        # 优先队列: (-next_access_time, insertion_order, key)
        # 使用负时间实现最大堆行为（最远访问优先）
        self.replacement_heap = []
        self._insertion_order = 0
        
    def get(self, key):
        """如果存在，从缓存中获取项目"""
        if key in self.cache:
            self.current_time += 1
            return self.cache[key]
        return None
    
    def put(self, key, value):
        """将项目放入缓存，必要时进行驱逐"""
        if key in self.cache:
            # 更新现有条目
            self.cache[key] = value
            self.current_time += 1
            return
            
        if len(self.cache) >= self.capacity:
            self._evict_optimal()
            
        self.cache[key] = value
        self._add_to_heap(key)
        self.current_time += 1
    
    def _add_to_heap(self, key):
        """将键和下一次访问时间添加到替换堆中"""
        next_access = self._get_next_access_time(key)
        self._insertion_order += 1
        heapq.heappush(self.replacement_heap, 
                      (-next_access, self._insertion_order, key))
    
    def _get_next_access_time(self, key):
        """获取键的下一次访问时间，如果不再访问则返回无穷大"""
        if key not in self.future_access:
            return float('inf')
        
        accesses = self.future_access[key]
        # 使用二分查找替代线性搜索
        left, right = 0, len(accesses)
        while left < right:
            mid = (left + right) // 2
            if accesses[mid] > self.current_time:
                right = mid
            else:
                left = mid + 1
        
        # 如果找到了下一个访问时间
        if left < len(accesses):
            return accesses[left]
        return float('inf')
    
    def _evict_optimal(self):
        """
        驱逐未来最远访问的键 - O(log n)
        这是OPT算法的核心
        """
        while self.replacement_heap:
            neg_next_access, _, key = heapq.heappop(self.replacement_heap)
            
            # 如果键已不在缓存中则跳过（延迟删除）
            if key not in self.cache:
                continue
                
            # 验证这仍然是正确的下一次访问时间
            actual_next = self._get_next_access_time(key)
            if actual_next == -neg_next_access:
                del self.cache[key]
                return
            else:
                # 仅当键仍在缓存中且访问时间有效时才重新添加
                if key in self.cache and actual_next != float('inf'):
                    self._insertion_order += 1
                    heapq.heappush(self.replacement_heap, 
                                  (-actual_next, self._insertion_order, key))
        
        # 后备方案：如果堆损坏则删除任意键
        if self.cache:
            key = next(iter(self.cache))
            del self.cache[key]


def OPTInit(sampler, generator, iter):
    """
    使用未来的访问模式初始化OPT缓存
    
    返回预处理的未来访问映射供新API使用
    """
    generator = copy.deepcopy(generator)
    future_index = list(
        sampler([None] * iter,
                replacement=True,
                num_samples=iter,
                generator=generator)
    )
    
    # 预处理future_index为key -> 访问时间列表
    future_map = {}
    for idx, key in enumerate(future_index):
        if key not in future_map:
            future_map[key] = []
        future_map[key].append(idx)
    
    return future_map


def OPTCache(cache_max=1):
    """
    OPT缓存装饰器
    """
    # 这将在第一次调用时设置
    cache_instance = None
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal cache_instance
            
            # 第一次调用时延迟初始化
            if cache_instance is None:
                # 尝试从kwargs或args获取future_access_map
                future_access_map = kwargs.get('future_access_map')
                if future_access_map is None and len(args) > 2 and isinstance(args[2], dict):
                    future_access_map = args[2]
                elif future_access_map is None:
                    future_access_map = {}
                    
                cache_instance = OptimalCache(cache_max, future_access_map)
            
            # 确保参数足够
            if len(args) < 2:
                return func(*args, **kwargs)
                
            input_index = args[1]
            
            # 先尝试缓存
            result = cache_instance.get(input_index)
            if result is not None:
                return result
            
            # 缓存未命中：计算并存储
            result = func(*args, **kwargs)
            cache_instance.put(input_index, result)
            return result
        
        return wrapper
    return decorator
