#include "leonova_a_radix_merge_sort/tbb/include/ops_tbb.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "leonova_a_radix_merge_sort/common/include/common.hpp"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"
#include "util/include/util.hpp"

namespace leonova_a_radix_merge_sort {

tbb::task_arena &LeonovaARadixMergeSortTBB::GetTbbArena() {
  static int num_threads = std::max(1, ppc::util::GetNumThreads());
  static tbb::task_arena arena(num_threads);
  return arena;
}

inline uint64_t LeonovaARadixMergeSortTBB::ToUnsignedValue(int64_t value) {
  return static_cast<uint64_t>(value) ^ kSignBitMask;
}

void LeonovaARadixMergeSortTBB::FillUnsignedKeys(const std::vector<int64_t> &arr, size_t left, size_t size,
                                                 std::vector<uint64_t> &keys) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t> &r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      keys[i] = ToUnsignedValue(arr[left + i]);
    }
  });
}

void LeonovaARadixMergeSortTBB::CountBytesParallel(const std::vector<uint64_t> &keys, size_t size, int shift,
                                                   tbb::enumerable_thread_specific<CounterRow> &local_counts) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t> &r) {
    auto &counts = local_counts.local();

    for (size_t i = r.begin(); i < r.end(); ++i) {
      ++counts[(keys[i] >> shift) & 0xFFU];
    }
  });
}

void LeonovaARadixMergeSortTBB::ReduceCounts(const tbb::enumerable_thread_specific<CounterRow> &local_counts,
                                             CounterRow &global_counts) {
  std::ranges::fill(global_counts, 0);

  for (const auto &row : local_counts) {
    for (size_t i = 0; i < kNumCounters; ++i) {
      global_counts[i] += row[i];
    }
  }
}

void LeonovaARadixMergeSortTBB::BuildOffsets(const CounterRow &global_counts,
                                             const tbb::enumerable_thread_specific<CounterRow> &local_counts,
                                             std::vector<CounterRow> &thread_offsets) {
  // 1. Собираем local_counts в стабильный массив
  std::vector<CounterRow> counts;
  counts.reserve(local_counts.size());

  for (const auto &row : local_counts) {
    counts.push_back(row);
  }

  const size_t num_threads = counts.size();

  thread_offsets.assign(num_threads, CounterRow(kNumCounters, 0));

  // 2. Глобальные offsets (prefix sum)
  CounterRow global_offsets(kNumCounters);
  size_t sum = 0;

  for (size_t i = 0; i < kNumCounters; ++i) {
    global_offsets[i] = sum;
    sum += global_counts[i];
  }

  // 3. Для каждого "потока" считаем смещения
  CounterRow running = global_offsets;

  for (size_t tindex = 0; tindex < num_threads; ++tindex) {
    for (size_t index = 0; index < kNumCounters; ++index) {
      thread_offsets[tindex][index] = running[index];
      running[index] += counts[tindex][index];
    }
  }
}

void LeonovaARadixMergeSortTBB::ScatterParallel(const std::vector<uint64_t> &keys, const std::vector<int64_t> &arr,
                                                size_t left, size_t size, int shift,
                                                std::vector<CounterRow> &thread_offsets, std::vector<int64_t> &temp_arr,
                                                std::vector<uint64_t> &temp_keys) {
  const size_t num_threads = thread_offsets.size();

  tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](const tbb::blocked_range<size_t> &r) {
    // ⚠️ стабильный thread id
    int tid = tbb::this_task_arena::current_thread_index();
    if (tid < 0 || std::cmp_greater_equal(tid, num_threads)) {
      tid = 0;  // fallback (редкий, но безопасный)
    }

    auto &offsets = thread_offsets[tid];

    for (size_t i = r.begin(); i < r.end(); ++i) {
      const size_t byte = (keys[i] >> shift) & 0xFFU;
      const size_t pos = offsets[byte]++;

      temp_arr[pos] = arr[left + i];
      temp_keys[pos] = keys[i];
    }
  });
}

LeonovaARadixMergeSortTBB::LeonovaARadixMergeSortTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int64_t>(GetInput().size());
}

bool LeonovaARadixMergeSortTBB::ValidationImpl() {
  return !GetInput().empty();
}

bool LeonovaARadixMergeSortTBB::PreProcessingImpl() {
  return true;
}

bool LeonovaARadixMergeSortTBB::RunImpl() {
  if (!ValidationImpl()) {
    return false;
  }

  GetOutput() = GetInput();

  if (GetOutput().size() > 1) {
    RadixMergeSort(GetOutput(), 0, GetOutput().size());
  }

  return true;
}

bool LeonovaARadixMergeSortTBB::PostProcessingImpl() {
  return true;
}

void LeonovaARadixMergeSortTBB::SequentialRadixSort(std::vector<int64_t> &arr, size_t left, size_t right) {
  const size_t size = right - left;
  if (size <= 1) {
    return;
  }

  std::vector<uint64_t> keys(size);
  std::vector<int64_t> temp_arr(size);
  std::vector<uint64_t> temp_keys(size);

  for (size_t i = 0; i < size; ++i) {
    keys[i] = ToUnsignedValue(arr[left + i]);
  }

  for (int byte_pos = 0; byte_pos < kNumBytes; ++byte_pos) {
    const int shift = byte_pos * kByteSize;

    std::vector<size_t> counts(kNumCounters, 0);
    std::vector<size_t> offsets(kNumCounters, 0);

    for (size_t i = 0; i < size; ++i) {
      const size_t byte_val = (keys[i] >> shift) & 0xFFU;
      ++counts[byte_val];
    }

    size_t sum = 0;
    for (size_t i = 0; i < kNumCounters; ++i) {
      offsets[i] = sum;
      sum += counts[i];
    }

    for (size_t i = 0; i < size; ++i) {
      const size_t byte_val = (keys[i] >> shift) & 0xFFU;
      const size_t pos = offsets[byte_val]++;
      temp_arr[pos] = arr[left + i];
      temp_keys[pos] = keys[i];
    }

    std::ranges::copy(temp_arr, arr.begin() + static_cast<std::ptrdiff_t>(left));
    keys.swap(temp_keys);
  }
}

void LeonovaARadixMergeSortTBB::RadixSort(std::vector<int64_t> &arr, size_t left, size_t right) {
  const size_t size = right - left;
  if (size <= 1) {
    return;
  }

  if (size < kMinParallelSize) {
    SequentialRadixSort(arr, left, right);
    return;
  }

  std::vector<uint64_t> keys(size);
  std::vector<uint64_t> temp_keys(size);
  std::vector<int64_t> temp_arr(size);

  CounterRow global_counts(kNumCounters);

  tbb::enumerable_thread_specific<CounterRow> local_counts(CounterRow(kNumCounters, 0));
  std::vector<CounterRow> thread_offsets;

  auto &arena = GetTbbArena();

  arena.execute([&] {
    FillUnsignedKeys(arr, left, size, keys);

    for (int byte_pos = 0; byte_pos < kNumBytes; ++byte_pos) {
      int shift = byte_pos * kByteSize;

      local_counts.clear();
      thread_offsets.clear();

      CountBytesParallel(keys, size, shift, local_counts);
      ReduceCounts(local_counts, global_counts);
      BuildOffsets(global_counts, local_counts, thread_offsets);

      ScatterParallel(keys, arr, left, size, shift, thread_offsets, temp_arr, temp_keys);

      std::ranges::copy(temp_arr, arr.begin() + static_cast<std::ptrdiff_t>(left));
      keys.swap(temp_keys);
    }
  });
}

void LeonovaARadixMergeSortTBB::SimpleMerge(std::vector<int64_t> &arr, size_t left, size_t mid, size_t right) {
  const size_t left_size = mid - left;
  const size_t right_size = right - mid;

  std::vector<int64_t> merged(left_size + right_size);

  size_t i = 0;
  size_t j = 0;
  size_t k = 0;

  while (i < left_size && j < right_size) {
    if (arr[left + i] <= arr[mid + j]) {
      merged[k++] = arr[left + i++];
    } else {
      merged[k++] = arr[mid + j++];
    }
  }

  while (i < left_size) {
    merged[k++] = arr[left + i++];
  }

  while (j < right_size) {
    merged[k++] = arr[mid + j++];
  }

  std::ranges::copy(merged, arr.begin() + static_cast<std::ptrdiff_t>(left));
}

void LeonovaARadixMergeSortTBB::RadixMergeSort(std::vector<int64_t> &arr, size_t left, size_t right) {
  struct SortTask {
    size_t left;
    size_t right;
    bool sorted;
  };

  std::vector<SortTask> stack;
  stack.reserve(128);
  stack.push_back({left, right, false});

  while (!stack.empty()) {
    SortTask current = stack.back();
    stack.pop_back();

    const size_t size = current.right - current.left;

    if (size <= 1) {
      continue;
    }

    if (size <= kRadixThreshold) {
      RadixSort(arr, current.left, current.right);
      continue;
    }

    if (!current.sorted) {
      const size_t mid = current.left + (size / 2);

      stack.push_back({current.left, current.right, true});
      stack.push_back({mid, current.right, false});
      stack.push_back({current.left, mid, false});
    } else {
      const size_t mid = current.left + (size / 2);
      SimpleMerge(arr, current.left, mid, current.right);
    }
  }
}

}  // namespace leonova_a_radix_merge_sort
