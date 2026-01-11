#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <atomic>
#include <cstdint>
#include <vector>
#include <queue>

namespace py = pybind11;

namespace {

// Metadata indices
constexpr int64_t IDX_GLOBAL_SEQ = 0;
constexpr int64_t IDX_HITS = 1;
constexpr int64_t IDX_MISSES = 2;
constexpr int64_t IDX_EVICTS = 3;
constexpr int64_t IDX_FREE_STACK_TOP = 4;
constexpr int64_t IDX_PREFETCH_CURSOR = 8; // Pointer in future_index to prefetch

// Node struct for build_opt_plan
struct Node {
    int64_t next_pos;
    int64_t key;
    bool operator<(const Node& other) const {
        return next_pos < other.next_pos;
    }
};

// Optimized implementation using vector (Fast)
torch::Tensor build_opt_plan(torch::Tensor future_index_tensor, int64_t maxsize) {
    py::gil_scoped_release release;
    
    int64_t total_iterations = future_index_tensor.size(0);
    auto future_index_ptr = future_index_tensor.data_ptr<int64_t>();

    // Find max_key for vector sizing
    int64_t max_key = 0;
    for (int64_t i = 0; i < total_iterations; ++i) {
        if (future_index_ptr[i] > max_key) {
            max_key = future_index_ptr[i];
        }
    }
    
    // 1. Precompute next_occurrence
    // Use vector for O(1) access instead of map
    std::vector<int64_t> next_occurrence(total_iterations, -1);
    std::vector<int64_t> last_seen(max_key + 1, -1);

    for (int64_t i = total_iterations - 1; i >= 0; --i) {
        int64_t key = future_index_ptr[i];
        if (last_seen[key] != -1) {
            next_occurrence[i] = last_seen[key];
        }
        last_seen[key] = i;
    }
    
    // 2. Decision Table [total_iter, 3] initialized to -1
    // Column 0: should_cache (0/1)
    // Column 1: victim_key (-1 or key)
    // Column 2: release_key (-1 or key)
    auto decision_table = torch::full({total_iterations, 3}, -1, torch::kInt64);
    auto decision_ptr = decision_table.data_ptr<int64_t>();
    
    // cache_next: key -> next_pos
    // Use vector for O(1) state tracking. -2 indicates not in cache.
    std::vector<int64_t> cache_next(max_key + 1, -2); // -2: not in cache, -1: used but no next (infinity)
    int64_t cache_size = 0;
    std::priority_queue<Node> heap;
    
    for (int64_t i = 0; i < total_iterations; ++i) {
        int64_t key = future_index_ptr[i];
        int64_t next_pos = next_occurrence[i];
        int64_t base = i * 3;
        
        // Check if in cache (O(1) lookup)
        bool is_cached = (cache_next[key] != -2);

        if (is_cached) {
            // HIT
            if (next_pos == -1) {
                // Last access, will release
                cache_next[key] = -2;
                cache_size--;
                decision_ptr[base + 2] = key; // release_key
            } else {
                cache_next[key] = next_pos;
                heap.push({next_pos, key});
            }
            continue;
        }
        
        // MISS
        if (next_pos == -1) continue; // No use caching if never used again
        
        if (cache_size >= maxsize && maxsize > 0) {
            // EVICT
            int64_t victim_key = -1;
            int64_t v_next = -1;
            while (!heap.empty()) {
                auto top = heap.top();
                heap.pop();
                
                // Lazy deletion check: is this heap entry still valid?
                if (cache_next[top.key] == top.next_pos) {
                    victim_key = top.key;
                    v_next = top.next_pos;
                    break;
                }
            }
            
            if (victim_key != -1) {
                // BUG FIX: Only evict if the new key is "better" (next use is sooner) 
                // than the furthest key in cache.
                if (next_pos < v_next || v_next == -1) {
                    cache_next[victim_key] = -2;
                    cache_size--;
                    decision_ptr[base + 1] = victim_key; // victim_key
                } else {
                    // The current key is even more useless than what we have.
                    // Mark as "do not cache"
                    decision_ptr[base + 0] = 0; 
                }
            }
        }
        
        if (cache_size < maxsize && decision_ptr[base + 0] != 0) {
            // Add to cache
            cache_next[key] = next_pos;
            cache_size++;
            decision_ptr[base + 0] = 1; // should_cache
            heap.push({next_pos, key});
        }
    }
    
    return decision_table;
}

class OPTCore {
public:
    OPTCore(torch::Tensor meta, torch::Tensor decision_table, torch::Tensor slot_map, torch::Tensor free_slots, torch::Tensor pool)
        : meta_(meta), decision_table_(decision_table), slot_map_(slot_map), free_slots_(free_slots), pool_(pool) {
        
        meta_ptr_ = meta.data_ptr<int64_t>();
        decision_ptr_ = decision_table.data_ptr<int64_t>();
        slot_map_ptr_ = slot_map.data_ptr<int64_t>();
        free_slots_ptr_ = free_slots.data_ptr<int64_t>();
        
        total_iter_ = decision_table.size(0);
    }

    int64_t execute_step_atomic(int64_t input_index) {
        // 使用原子操作直接获取并递增全局序列号
        auto* global_seq = reinterpret_cast<std::atomic<int64_t>*>(meta_ptr_ + IDX_GLOBAL_SEQ);
        int64_t step_idx = global_seq->fetch_add(1, std::memory_order_relaxed);

        if (step_idx >= total_iter_) {
            return -1;
        }

        // 提取该步骤的预计算指令
        int64_t base = step_idx * 3;
        int64_t instr_should_cache = decision_ptr_[base + 0];
        int64_t instr_victim_key = decision_ptr_[base + 1];
        int64_t instr_release_key = decision_ptr_[base + 2];

        auto* slot_map_atomic = reinterpret_cast<std::atomic<int64_t>*>(slot_map_ptr_);
        auto* hits = reinterpret_cast<std::atomic<int64_t>*>(meta_ptr_ + IDX_HITS);
        auto* misses = reinterpret_cast<std::atomic<int64_t>*>(meta_ptr_ + IDX_MISSES);
        auto* evicts = reinterpret_cast<std::atomic<int64_t>*>(meta_ptr_ + IDX_EVICTS);
        auto* free_top = reinterpret_cast<std::atomic<int64_t>*>(meta_ptr_ + IDX_FREE_STACK_TOP);

        // 尝试检查命中 (Lock-free Load)
        int64_t slot = slot_map_atomic[input_index].load(std::memory_order_acquire);
        
        if (slot != -1) {
            hits->fetch_add(1, std::memory_order_relaxed);
            // 即便命中，如果预计算指令要求释放该 key，则执行释放
            if (instr_release_key != -1) {
                int64_t r_slot = slot_map_atomic[instr_release_key].exchange(-1, std::memory_order_acq_rel);
                if (r_slot != -1) {
                    int64_t pos = free_top->fetch_add(1, std::memory_order_relaxed);
                    free_slots_ptr_[pos] = r_slot;
                }
            }
            return slot;
        }

        // 未命中
        misses->fetch_add(1, std::memory_order_relaxed);
        int64_t out_slot = -1;

        if (instr_should_cache != 0) {
            // 1. 如果有受害者，原子性地驱逐受害者
            if (instr_victim_key != -1) {
                int64_t v_slot = slot_map_atomic[instr_victim_key].exchange(-1, std::memory_order_acq_rel);
                if (v_slot != -1) {
                    out_slot = v_slot;
                    evicts->fetch_add(1, std::memory_order_relaxed);
                }
            }
            
            // 2. 如果没有（或没抢到）受害者槽位，从空闲栈弹出一个
            if (out_slot == -1) {
                int64_t current_top = free_top->load(std::memory_order_acquire);
                while (current_top > 0) {
                    if (free_top->compare_exchange_weak(current_top, current_top - 1, std::memory_order_acq_rel)) {
                        out_slot = free_slots_ptr_[current_top - 1];
                        break;
                    }
                }
            }

            // 3. 记录新项的槽位
            if (out_slot != -1) {
                slot_map_atomic[input_index].store(out_slot, std::memory_order_release);
            }
        }

        // 处理其他需要在本步释放的 key
        if (instr_release_key != -1 && instr_release_key != input_index) {
            int64_t r_slot = slot_map_atomic[instr_release_key].exchange(-1, std::memory_order_acq_rel);
            if (r_slot != -1) {
                int64_t pos = free_top->fetch_add(1, std::memory_order_relaxed);
                free_slots_ptr_[pos] = r_slot;
            }
        }

        return (out_slot == -1) ? -1 : -(out_slot + 2);
    }

    void update_cache(int64_t slot, torch::Tensor data) {
        py::gil_scoped_release release;
        pool_[slot].copy_(data);
    }

private:
    torch::Tensor meta_;
    torch::Tensor decision_table_;
    torch::Tensor slot_map_;
    torch::Tensor free_slots_;
    torch::Tensor pool_;
    
    int64_t* meta_ptr_;
    int64_t* decision_ptr_;
    int64_t* slot_map_ptr_;
    int64_t* free_slots_ptr_;
    int64_t total_iter_;
};

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_opt_plan", &build_opt_plan, "Build OPT eviction plan and decision table");

    py::class_<OPTCore>(m, "OPTCore")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>())
        .def("execute_step", &OPTCore::execute_step_atomic)
        .def("update_cache", &OPTCore::update_cache);
}
