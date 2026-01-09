from .cachelib import (
    OPTCacheDecorator,
    SharedOPTCacheDecorator,
    generate_precomputed_file,
)

__version__ = "1.1.0"


def get_opt_cache(mode="cpp", **kwargs):
    """
    获取 OPT 缓存装饰器的工厂方法。

    Args:
        mode (str): "cpp" (共享内存高性能版) 或 "python" (单进程纯 Python 版)
        **kwargs: 传给装饰器构造函数的参数
    """
    if mode == "cpp":
        return SharedOPTCacheDecorator(**kwargs)
    elif mode == "python":
        return OPTCacheDecorator(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


__all__ = [
    "get_opt_cache",
    "OPTCacheDecorator",
    "SharedOPTCacheDecorator",
    "generate_precomputed_file",
]
