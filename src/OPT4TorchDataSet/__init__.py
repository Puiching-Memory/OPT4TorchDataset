from .cachelib import (
    OPTCacheDecorator,
    SharedOPTCacheDecorator,
    generate_precomputed_file,
)

__version__ = "1.1.0"


def get_opt_cache(mode=None, num_workers=0, **kwargs):
    """
    获取 OPT 缓存装饰器的工厂方法。

    Args:
        mode (str, optional):
            "cpp" - 共享内存高性能版，使用 C++ 引擎和 Tensor Pool，支持多进程共享同步。
            "python" - 单进程纯 Python 版，使用原生 dict，性能极高，不支持跨进程。
            如果为 None（默认），将根据 num_workers 自动选择。
        num_workers (int): 数据加载器的工作进程数。
        **kwargs: 传给装饰器构造函数的参数
    """
    # 智能选择模式
    if mode is None:
        if num_workers > 0:
            mode = "cpp"
        else:
            mode = "python"

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
