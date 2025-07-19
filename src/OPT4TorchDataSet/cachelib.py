from functools import wraps
import copy
import time
import sys

def OPTInit(sampler,generator,iter):
    """
    初始化OPT缓存
    ---
    seed: 种子, 必须与DataGenerator一致  
    sampler: 采样器  
    generator: 生成器  
    iter: 迭代次数  
    """
    global future_index
    generator = copy.deepcopy(generator)
    future_index = list(
                    sampler([None]*iter,
                    replacement=True,
                    num_samples=iter,
                    generator=generator
                    )
                    )
    
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
            # print(f"[Arguments] {args} [K Arguments] {kwargs}")
            # datasetOBJ = args[0]
            input_index = args[1]
            # start_time = time.perf_counter()

            # 分支1:缓存命中,直接返回缓存中的内容
            if input_index in cache:
                current += 1
                # print(f"[find in cache] time usage: {time.perf_counter() - start_time}")
                return cache[input_index]

            # print(future_index[current])

            # 缓存未命中,执行函数
            result = func(*args, **kwargs)
            
            # 分支2:缓存未满,直接添加到缓存中
            if len(cache) < cache_max:
                cache[input_index] = result
                current += 1
                # print(f"[NOT find in cache] cache usage: {len(cache)} {round(len(cache)/cache_max,2)}%")
                return result
            
            # 分支3:缓存已满,使用OPT算法替换
            # start_time = time.perf_counter()
            # TODO: 滑动窗口优化
            max_distance = ["key",0]
            for k in cache.keys(): # 遍历缓存中的所有键,计算距离
                # 尝试在窗口内查找,如果找不到则认为距离为inf,如果找到保留最大距离
                try: 
                    distance = future_index.index(k, current, current + int(1281)) - current
                except ValueError:
                    max_distance = [k,float("inf")]
                    break
                else:  
                    if distance > max_distance[1]:
                        max_distance = [k,distance]
            
            # 删除最久未使用的键，并将新结果添加到缓存中
            cache.pop(max_distance[0])
            cache[input_index] = result
            # print(f"[future find] time usage: {time.perf_counter() - start_time}")

            current += 1
            # print(f"Result: {result}")
            return result
        
        return wrapper
    return decorator