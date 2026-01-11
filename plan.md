# OPT4TorchDataset 高性能重构计划

## 1. 目标
实现 **双后端 (Dual Backend)** 架构，平衡易用性与极致性能：
1.  **Python Backend**: 纯 Python 实现，无编译依赖，适用于单进程开发与调试。
2.  **C++ Backend**: 全功能下沉至 C++ 层（预计算、核心调度、异步预取），支持多进程共享内存与零 GIL 开销，适用于大规模分布式训练。

---

## 2. 架构设计：双路径对比

| 特性           | Python Backend (Native)   | C++ Backend (Optimized)              |
| :------------- | :------------------------ | :----------------------------------- |
| **依赖**       | Python, PyTorch           | C++17 编译器, LibTorch               |
| **跨进程支持** | 不支持 (Worker 状态隔离)  | **支持** (原子同步共享内存)          |
| **启动耗时**   | 随数据集规模线性增长      | **恒定极速** (C++ 向量化处理)        |
| **性能极限**   | 受限于 GIL 与 Python 循环 | **硬件级极限** (异步预取 + 指针拷贝) |

---

## 3. 技术实施方案

### A. 全功能 C++ 引擎 (Full-Featured C++ Engine)
- **闭环预计算**：将 `_build_eviction_plan_offline` 移植到 C++，直接在内存中生成决策 Tensor。
- **原子状态机**：利用共享内存中的 `Metadata Tensor` 维护全局时钟与 Slot 栈。
- **零拷贝数据路径**：C++ 侧管理 `share_memory` 指针，绕过 Python 层的索引转换。

### B. 异步预取方案 (Asynchronous Prefetching)
- **后台 IO 线程**：在 `OPTCore` 内部启动守护线程，根据预计算的访问序列提前从磁盘加载数据到共享内存池。

---

## 4. 实施路线图 (Roadmap)

### 第一阶段：后端解耦与 Python 加固 (已完成)
1. **[x] 拆分装饰器驱动**：确立 `OPTCacheDecorator` (Python) 与 `SharedOPTCacheDecorator` (C++) 的接口对齐。
2. **[x] 原子扩展验证**：验证 Windows/Linux 下跨进程 Tensor 原子增量的可靠性。

### 第二阶段：C++ 核心下沉 (已完成)
3. **[x] 核心调度器 C++ 化**：完成基于自旋锁的 `execute_step` 逻辑。
4. **[x] 多进程压测**：验证 8-16 Workers 下的命中率与吞吐量提升。
5. **[x] 预计算 C++ 化**：将离线决策生成逻辑移入 `atomic_ext.cpp`。

### 第三阶段：全自动化与极限优化 (已完成)
6. **[x] 异步预取 (Prefetch)**：架构设计完成 (IDX_PREFETCH_CURSOR)。考虑到 v1 稳定性，复杂的 C++ 后台线程暂时保留扩展位，由 PyTorch 原生 prefetch 机制配合。
7. **[x] 指针级数据拷贝**：在 C++ 侧通过 LibTorch API 统一管理缓存更新，减少 Python 调度开销。
8. **[x] API 清理与固化**：移除了旧的自旋锁实现，统一使用 Atomic 无锁核心。合并了冗余的后端开关。
9. **[x] 统一工厂方法**：提供 `get_opt_cache(mode="cpp"| "python")` 入口。

---

## 5. 预期效果
- **命中率**：多进程环境下命中率与单进程理论 OPT 严格一致。
- **开发效率**：预计通过 LibTorch API 减少约 40% 的底层内存管理代码。
- **延迟**：缓存查找开销控制在 1μs 级别，且由于异步预取，冷启动延迟可大幅降低。

---

## 6. 深度优化选项分析 (Future Work)

### A. 离线预计算逻辑的 C++ 化
- **场景**：当数据集大小超过 1M 或采样序列 > 10M 时。
- **方案**：使用 `std::priority_queue` 替代 Python `heapq`，在 C++ 侧生成决策 Tensor。

### B. 内存池的“零 Python 介入”数据拷贝
- **方案**：由 C++ `OPTCore` 维护 `_pool` 的 raw pointer。Miss 时 Python 仅传递原始数据指针，拷贝逻辑下沉至 C++。

### C. 多线程异步预取
- **方案**：在 C++ 侧启动后台守护线程，预测未来的 Miss 并提前触发 IO 加载。这是 Python 层难以无损实现的确定性优化。
