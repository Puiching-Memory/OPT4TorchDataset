from functools import wraps
import copy
def OPTInit(seed,sampler,generator,iter):
    global OPTSEED,future_index
    OPTSEED = seed
    generator = copy.deepcopy(generator)
    future_index = list(
                    sampler([None]*iter,
                    replacement=True,
                    num_samples=iter,
                    generator=generator
                    )
                    )
def OPTCache():
    location = 0

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal location
            print(f"Arguments: {args}, Keyword Arguments: {kwargs}")
            datasetOBJ = args[0]
            input_index = args[1]

            print(future_index[location])

            result = func(*args, **kwargs)

            location += 1
            print(f"Result: {result}")
            return result
        return wrapper
    return decorator