import numpy as np

# 导入HitRateDataset
from lib.hit_rate_dataset import HitRateDataset


class ZipfDataset(HitRateDataset):
    """
    基于Zipf分布的数据集实现

    Zipf分布是一种幂律分布，常用于模拟自然语言中词频分布等现象。
    在缓存研究中，Zipf分布常用于模拟数据访问模式。
    """

    def __init__(self, N=20000, alpha=1.0, seed=None):
        """
        初始化Zipf数据集

        Args:
            N: 数据集大小，即不同元素的总数
            alpha: Zipf分布的指数参数，控制分布的偏斜程度
                   alpha=0时为均匀分布，alpha>1时分布更加偏斜
            seed: 随机种子，用于生成可重现的访问序列
        """
        # 初始化父类HitRateDataset，size为N
        super().__init__(size=N)

        self.N = N
        self.alpha = alpha
        self.seed = seed
        self.access_sequence = None

        if seed is not None:
            self._generator.manual_seed(seed)

        # 生成Zipf分布的概率分布
        self._generate_probability_distribution()

    def _generate_probability_distribution(self):
        """
        生成Zipf分布的概率分布
        """
        ranks = np.arange(1, self.N + 1, dtype=np.float64)
        self.probs = 1.0 / (ranks**self.alpha)
        self.probs /= self.probs.sum()

    def __len__(self):
        """
        返回数据集大小
        """
        return self.N

    def _raw_getitem(self, idx):
        """
        获取指定索引的数据项，重写父类方法以适配ZipfDataset

        Args:
            idx: 数据项索引

        Returns:
            数据项（在此实现中直接返回索引）
        """
        # 调用父类方法增加miss计数
        self.miss += 1

        # 检查索引是否有效
        if idx < 0 or idx >= self.N:
            raise IndexError(
                f"Index {idx} is out of range for dataset of size {self.N}"
            )

        # 在Zipf数据集中，我们直接返回索引作为数据项
        return idx

    def generate_access_sequence(self, T, seed=None):
        """
        生成长度为T的访问序列

        Args:
            T: 序列长度
            seed: 随机种子

        Returns:
            访问序列列表
        """
        rng = np.random.default_rng(seed)
        sequence = rng.choice(
            np.arange(self.N, dtype=np.int64), size=T, p=self.probs
        ).tolist()
        return sequence


# 示例用法
if __name__ == "__main__":
    # 创建Zipf数据集实例
    dataset = ZipfDataset(N=1000, alpha=1.2, seed=42)

    # 生成访问序列
    sequence = dataset.generate_access_sequence(T=10000, seed=123)

    # 打印一些统计信息
    print(f"Dataset size: {len(dataset)}")
    print(f"Alpha parameter: {dataset.alpha}")
    print(f"First 20 items in sequence: {sequence[:20]}")

    # 统计访问频率
    unique, counts = np.unique(sequence, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    print(f"Top 10 most accessed items: {unique[sorted_indices[:10]]}")
    print(f"Their counts: {counts[sorted_indices[:10]]}")

    # 测试命中率统计功能
    from cachetools import cached, LRUCache

    dataset.setCache(cached(LRUCache(maxsize=int(1000 * 0.5))))

    print(f"Cache miss count before access: {dataset.getMissCount()}")

    # 访问前10个元素
    for i in range(10):
        _ = dataset[i]

    print(f"Cache miss count after access: {dataset.getMissCount()}")
    print(f"Cache miss rate: {dataset.getMissCount() / 10}")
