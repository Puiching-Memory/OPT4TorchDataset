import copy
import time
from collections import defaultdict, deque
from functools import wraps

def OPTInit(sampler,generator,iter):
    """
    初始化OPT缓存
    ---
    seed: 种子, 必须与DataGenerator一致  
    sampler: 采样器  
    generator: 生成器  
    iter: 迭代次数  
    """
    global future_index, future_map
    generator = copy.deepcopy(generator)
    future_index = list(
                    sampler([None]*iter,
                    replacement=True,
                    num_samples=iter,
                    generator=generator
                    )
                    )
    # 预处理future_index为 key->队列
    future_map = defaultdict(deque)
    for idx, k in enumerate(future_index):
        future_map[k].append(idx)
    
def OPTCache(cache_max=1,future_rate=0.01):
    """
    OPT缓存装饰器
    ---
    cache_max: 缓存最大值  
    """
    current = 0
    cache = dict()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal current,cache
            global future_map
            # print(f"[Arguments] {args} [K Arguments] {kwargs}")
            # datasetOBJ = args[0]
            input_index = args[1]
            
            # start_time = time.perf_counter()
            # 分支1:缓存命中,直接返回缓存中的内容
            if input_index in cache:
                # print(f"[find in cache] time usage: {time.perf_counter() - start_time}")
                current += 1
                return cache[input_index]

            # print(future_index[current])

            # 缓存未命中,执行函数
            result = func(*args, **kwargs)
            
            # 分支2:缓存未满,直接添加到缓存中
            if len(cache) < cache_max:
                cache[input_index] = result
                # print(f"[NOT find in cache] cache usage: {len(cache)} {round(len(cache)/cache_max,2)}%")
                current += 1
                return result
            
            # 分支3:缓存已满,使用OPT算法替换（future_map队列弹出优化）
            # start_time = time.perf_counter()
            max_distance = [None, -1]
            for k in cache.keys():
                # future_map[k] 队首小于 current 的都弹出
                while future_map[k] and future_map[k][0] < current:
                    future_map[k].popleft()
                if not future_map[k]:
                    # 未来不会再访问，直接替换
                    max_distance = [k, float('inf')]
                    break
                else:
                    distance = future_map[k][0] - current
                    if distance > max_distance[1]:
                        max_distance = [k, distance]
            # 删除最久未使用的键，并将新结果添加到缓存中
            cache.pop(max_distance[0])
            cache[input_index] = result
            # print(f"[find in future] time usage: {time.perf_counter() - start_time}")

            # print(f"Result: {result}")
            current += 1
            return result
        
        return wrapper
    return decorator