import time
from pathlib import Path

import typer
from rich.console import Console

try:
    from OPT4TorchDataSet.cachelib import generate_precomputed_file
except ImportError:
    from .cachelib import generate_precomputed_file

console = Console()
app = typer.Typer(
    help="生成 OPT 缓存离线预计算文件（包含完整的淘汰计划）",
    add_completion=False,
)


@app.command()
def generate(
    dataset_size: int = typer.Option(
        ..., "--dataset-size", "-d", help="数据集大小 (索引范围: 0 到 dataset_size-1)"
    ),
    total_iter: int = typer.Option(
        ..., "--total-iter", "-t", help="总迭代次数 (采样总次数)"
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="保存预计算文件的路径 (.safetensors 文件)"
    ),
    seed: int = typer.Option(0, "--seed", "-s", help="随机种子 (默认: 0)"),
    maxsize: int = typer.Option(
        1, "--maxsize", "-m", help="缓存容量 - 用于离线预计算淘汰计划 (默认: 1)"
    ),
    replacement: bool = typer.Option(
        True, "--replacement/--no-replacement", help="启用/禁用有放回采样"
    ),
):
    """
    生成离线预计算文件，包含完整的淘汰计划。
    """
    console.print(
        f"[bold blue]开始离线预计算:[/bold blue] dataset_size={dataset_size}, "
        f"total_iter={total_iter}, maxsize={maxsize}, seed={seed}"
    )

    start_time = time.perf_counter()
    try:
        generate_precomputed_file(
            dataset_size=dataset_size,
            total_iterations=total_iter,
            persist_path=output,
            random_seed=seed,
            replacement=replacement,
            maxsize=maxsize,
        )
    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {e}")
        raise typer.Exit(code=1)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    console.print(
        f"\n[bold green]✓ 完成:[/bold green] 预计算结果已保存到 [cyan]{output}[/cyan]"
    )
    console.print(f"✓ 耗时: {elapsed:.2f} 秒")
    console.print(f"✓ 缓存容量: {maxsize}")
    console.print(f"✓ 提示: 运行时使用 maxsize={maxsize} 时需要使用此文件")


def main():
    app()


if __name__ == "__main__":
    main()
