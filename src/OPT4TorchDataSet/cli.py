import argparse
import importlib
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

if __package__ in {None, ""}:  # pragma: no cover - 直接运行文件时
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from OPT4TorchDataSet.cachelib import precompute_opt_indices
else:
    from .cachelib import precompute_opt_indices

DEFAULT_SAMPLER = "torch.utils.data:RandomSampler"


def _resolve_callable(target: str) -> Callable:
    module_name, sep, attr = target.partition(":")
    if not sep:
        raise ValueError(
            "sampler 必须以 'module:callable' 格式提供，例如 'torch.utils.data:RandomSampler'"
        )
    module = importlib.import_module(module_name)
    try:
        candidate = getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"在模块 {module_name!r} 中找不到 {attr!r}") from exc
    if not callable(candidate):
        raise TypeError(f"指定对象 {target!r} 不是可调用对象")
    return candidate


def _build_generator(seed: Optional[int]):
    try:
        import torch
    except ModuleNotFoundError:
        if seed is not None:
            print("警告: 未安装 torch，忽略 --seed。", file=sys.stderr)
        return None

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线预生成 OPT 未来访问索引并持久化",
    )
    parser.add_argument(
        "--total-iter",
        type=int,
        required=True,
        help="未来访问序列总长度 (通常等于采样总次数)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="保存 precomputed pickle 的路径",
    )
    parser.add_argument(
        "--sampler",
        default=DEFAULT_SAMPLER,
        help=(
            "采样器的导入路径，格式 module:callable。"
            "默认使用 torch.utils.data.RandomSampler"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子 (若安装 torch 则用于 torch.Generator)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的输出文件",
    )
    return parser


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    sampler_class = _resolve_callable(args.sampler)
    generator = _build_generator(args.seed)

    # 检查输出文件是否存在
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"目标文件 {args.output} 已存在，如需覆盖请添加 --overwrite"
        )

    print(
        f"开始预计算: total_iter={args.total_iter}, sampler={args.sampler}",
    )
    
    # 创建一个包装函数来适配不同的采样器接口
    def sampler_wrapper(data_source, replacement=True, num_samples=None, generator=None):
        """适配器：将torch采样器接口适配到我们的预计算函数"""
        if hasattr(sampler_class, '__name__') and 'RandomSampler' in sampler_class.__name__:
            # 对于RandomSampler，创建实例并获取索引
            sampler_instance = sampler_class(
                data_source=list(range(args.total_iter)),
                replacement=replacement,
                num_samples=num_samples,
                generator=generator
            )
            return list(sampler_instance)
        else:
            # 对于其他采样器，尝试直接调用
            return sampler_class(
                data_source,
                replacement=replacement,
                num_samples=num_samples,
                generator=generator
            )
    
    # 调用precompute_opt_indices，它会直接保存文件
    try:
        precompute_opt_indices(
            sampler_wrapper,
            generator,
            args.total_iter,
            args.output,
        )
    except Exception as e:
        print(f"错误: {e}")
        print("提示: 请确保已安装torch，或使用简单的采样器")
        return 1

    print(
        f"完成: 预计算结果已保存到 {args.output}"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run(argv)
    except Exception as exc:  # pragma: no cover - 统一异常出口
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
