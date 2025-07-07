
# 定义装饰器
def optA(func):
    def wrapper(self, *args, **kwargs):
        print("Before original method")
        print(f"Arguments: {args}, Keyword Arguments: {kwargs}")
        result = func(self, *args, **kwargs)
        print("After original method")
        print(f"Result: {result}")
        return result

    return wrapper