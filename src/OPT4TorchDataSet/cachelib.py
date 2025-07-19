from functools import wraps
import copy
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
    
def OPTCache(cache_max=1):
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
            # print(f"Arguments: {args}, Keyword Arguments: {kwargs}")
            # datasetOBJ = args[0]
            input_index = args[1]
            if input_index in cache:
                print("Cache hit")
                return cache[input_index]

            # print(future_index[current])

            result = func(*args, **kwargs)

            if len(cache) < cache_max:
                cache[input_index] = result
            else:
                max_distance = ["key",0]
                for k in cache.keys():
                    try:
                        distance = future_index.index(k, current, current + int(1000)) - current
                    except ValueError:
                        max_distance = [k,float("inf")]
                        break
                    else:  
                        if distance > max_distance[1]:
                            max_distance = [k,distance]
                    
                cache.pop(max_distance[0])
                cache[input_index] = result

            current += 1
            # print(f"Result: {result}")
            return result
        return wrapper
    return decorator