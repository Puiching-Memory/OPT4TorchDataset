import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence
import time

if __package__ in {None, ""}:  # pragma: no cover - 直接运行文件时
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from OPT4TorchDataSet.cachelib import generate_precomputed_file
else:
    from .cachelib import generate_precomputed_file


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成 OPT 缓存离线预计算文件（包含完整的淘汰计划）",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        required=True,
        help="数据集大小 (索引范围: 0 到 dataset_size-1)",
    )
    parser.add_argument(
        "--total-iter",
        type=int,
        required=True,
        help="总迭代次数 (采样总次数)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="保存预计算文件的路径 (.pkl 文件)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子 (默认: 0)",
    )
    parser.add_argument(
        "--maxsize",
        type=int,
        default=1,
        help="缓存容量 - 用于离线预计算淘汰计划 (默认: 1)",
    )
    parser.add_argument(
        "--no-replacement",
        action="store_false",
        dest="replacement",
        default=True,
        help="禁用有放回采样（默认启用）",
    )
    return parser


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    print(
        f"开始离线预计算: dataset_size={args.dataset_size}, "
        f"total_iter={args.total_iter}, maxsize={args.maxsize}, seed={args.seed}",
    )
    
    start_time = time.perf_counter()
    try:
        generate_precomputed_file(
            dataset_size=args.dataset_size,
            total_iterations=args.total_iter,
            persist_path=args.output,
            random_seed=args.seed,
            replacement=args.replacement,
            maxsize=args.maxsize,
        )
    except Exception as e:
        print(f"错误: {e}")
        return 1
    end_time = time.perf_counter()

    print(f"✓ 完成: 预计算结果已保存到 {args.output}")
    print(f"✓ 耗时: {end_time - start_time:.2f} 秒")
    print(f"✓ 缓存容量: {args.maxsize}")
    print(f"✓ 提示: 运行时使用 maxsize={args.maxsize} 时需要使用此文件")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run(argv)
    except Exception as exc:  # pragma: no cover - 统一异常出口
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())