#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <atomic>
#include <cstdint>
#include <vector>
#include <queue>
#include <unordered_map>
#include <thread>
#include <atomic>

namespace py = pybind11;

namespace {

// Metadata indices
constexpr int64_t IDX_GLOBAL_SEQ = 0;
constexpr int64_t IDX_HITS = 1;
constexpr int64_t IDX_MISSES = 2;
constexpr int64_t IDX_EVICTS = 3;
constexpr int64_t IDX_FREE_STACK_TOP = 4;
constexpr int64_t IDX_PREFETCH_CURSOR = 8; // Pointer in future_index to prefetch
constexpr int64_t IDX_LOCK = 9; // Use meta[9] as a spinlock

// ... Node struct for build_opt_plan ...
struct Node {
    int64_t next_pos;
    int64_t key;
    bool operator<(const Node& other) const {
        return next_pos < other.next_pos;
    }
};

torch::Tensor build_opt_plan(torch::Tensor future_index_tensor, int64_t maxsize) {
    py::gil_scoped_release release;
    
    int64_t total_iterations = future_index_tensor.size(0);
    auto future_index_ptr = future_index_tensor.data_ptr<int64_t>();
    
    // 1. Precompute next_occurrence
    std::vector<int64_t> next_occurrence(total_iterations, -1);
    std::unordered_map<int64_t, int64_t> last_seen;
    for (int64_t i = total_iterations - 1; i >= 0; --i) {
        int64_t key = future_index_ptr[i];
        auto it = last_seen.find(key);
        if (it != last_seen.end()) {
            next_occurrence[i] = it->second;
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
    std::unordered_map<int64_t, int64_t> cache_next;
    std::priority_queue<Node> heap;
    
    for (int64_t i = 0; i < total_iterations; ++i) {
        int64_t key = future_index_ptr[i];
        int64_t next_pos = next_occurrence[i];
        int64_t base = i * 3;
        
        auto it = cache_next.find(key);
        if (it != cache_next.end()) {
            // HIT
            if (next_pos == -1) {
                // Last access, will release
                cache_next.erase(it);
                decision_ptr[base + 2] = key; // release_key
            } else {
                it->second = next_pos;
                heap.push({next_pos, key});
            }
            continue;
        }
        
        // MISS
        if (next_pos == -1) continue; // No use caching if never used again
        
        if (cache_next.size() >= (size_t)maxsize && maxsize > 0) {
            // EVICT
            int64_t victim_key = -1;
            while (!heap.empty()) {
                auto top = heap.top();
                heap.pop();
                
                auto v_it = cache_next.find(top.key);
                if (v_it != cache_next.end() && v_it->second == top.next_pos) {
                    victim_key = top.key;
                    break;
                }
            }
            
            if (victim_key != -1) {
                cache_next.erase(victim_key);
                decision_ptr[base + 1] = victim_key; // victim_key
            }
        }
        
        if (cache_next.size() < (size_t)maxsize) {
            // Add to cache
            cache_next[key] = next_pos;
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
        stop_prefetch_ = false;
    }

    ~OPTCore() {
        stop_prefetch();
    }

    void start_prefetch(py::function loader_func, int64_t lookahead, torch::Tensor future_index_tensor) {
        // 禁用：C++ 后台线程中的 Python 回调涉及复杂的 GIL 管理。
        // 改为由 Python 层的 DataLoader prefetch_factor 负责或异步任务。
        return;
    }

    void stop_prefetch() {
        stop_prefetch_ = true;
        if (prefetch_thread_.joinable()) {
            prefetch_thread_.join();
        }
    }

    void lock() {
        // ... same lock logic ...
        auto* lock_ptr = reinterpret_cast<std::atomic<int64_t>*>(meta_ptr_ + IDX_LOCK);
        int64_t expected = 0;
        while (!lock_ptr->compare_exchange_weak(expected, 1, std::memory_order_acquire)) {
            expected = 0;
        }
    }

    void unlock() {
        // ... same unlock logic ...
        auto* lock_ptr = reinterpret_cast<std::atomic<int64_t>*>(meta_ptr_ + IDX_LOCK);
        lock_ptr->store(0, std::memory_order_release);
    }

    int64_t execute_step(int64_t input_index) {
        py::gil_scoped_release release;
        
        lock();

        int64_t step_idx = meta_ptr_[IDX_GLOBAL_SEQ]++;

        if (step_idx >= total_iter_) {
            unlock();
            return -1;
        }

        int64_t base = step_idx * 3;
        bool should_cache = decision_ptr_[base + 0] != 0;
        int64_t victim_key = decision_ptr_[base + 1];
        int64_t release_key = decision_ptr_[base + 2];

        int64_t slot = slot_map_ptr_[input_index];
        bool is_hit = (slot != -1);
        
        if (is_hit) {
            meta_ptr_[IDX_HITS]++;
            unlock();
            return slot;
        }

        meta_ptr_[IDX_MISSES]++;
        int64_t out_slot = -1;

        if (should_cache) {
            if (victim_key != -1) {
                int64_t v_slot = slot_map_ptr_[victim_key];
                if (v_slot != -1) {
                    slot_map_ptr_[victim_key] = -1;
                    out_slot = v_slot;
                    meta_ptr_[IDX_EVICTS]++;
                }
            }
            
            if (out_slot == -1) {
                if (meta_ptr_[IDX_FREE_STACK_TOP] > 0) {
                    int64_t pos = --meta_ptr_[IDX_FREE_STACK_TOP];
                    out_slot = free_slots_ptr_[pos];
                }
            }

            if (out_slot != -1) {
                slot_map_ptr_[input_index] = out_slot;
            }
        }

        if (release_key != -1) {
            int64_t r_slot = slot_map_ptr_[release_key];
            if (r_slot != -1) {
                slot_map_ptr_[release_key] = -1;
                int64_t pos = meta_ptr_[IDX_FREE_STACK_TOP]++;
                free_slots_ptr_[pos] = r_slot;
            }
        }

        unlock();
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

    std::thread prefetch_thread_;
    std::atomic<bool> stop_prefetch_;
};

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_opt_plan", &build_opt_plan, "Build OPT eviction plan and decision table");

    py::class_<OPTCore>(m, "OPTCore")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>())
        .def("execute_step", &OPTCore::execute_step)
        .def("update_cache", &OPTCore::update_cache)
        .def("start_prefetch", &OPTCore::start_prefetch)
        .def("stop_prefetch", &OPTCore::stop_prefetch);
}
