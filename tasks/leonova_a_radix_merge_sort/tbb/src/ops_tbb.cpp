#include "leonova_a_radix_merge_sort/tbb/include/ops_tbb.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "leonova_a_radix_merge_sort/common/include/common.hpp"
#include "tbb/blocked_range.h"
#include "tbb/global_control.h"
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

inline void LeonovaARadixMergeSortTBB::ResetLocalCounts(CounterTable &local_counts) {
  for (auto &counter : local_counts) {
    std::ranges::fill(counter, 0);
  }
}

inline void LeonovaARadixMergeSortTBB::BuildThreadOffsets(const CounterTable &local_counts, size_t thread_count,
                                                          CounterTable &local_offsets) {
  std::vector<size_t> bucket_totals(kNumCounters, 0);

  for (size_t thread = 0; thread < thread_count; ++thread) {
    const auto &row = local_counts[thread];
    for (size_t i = 0; i < kNumCounters; ++i) {
      bucket_totals[i] += row[i];
    }
  }

  size_t prefix = 0;
  for (auto &bucket_total : bucket_totals) {
    const size_t count = bucket_total;
    bucket_total = prefix;
    prefix += count;
  }

  for (size_t thread = 0; thread < thread_count; ++thread) {
    auto &offset_row = local_offsets[thread];
    const auto &count_row = local_counts[thread];
    size_t bucket_index = 0;
    for (auto &offset : offset_row) {
      offset = bucket_totals[bucket_index];
      bucket_totals[bucket_index] += count_row[bucket_index];
      ++bucket_index;
    }
  }
}

void LeonovaARadixMergeSortTBB::FillUnsignedKeys(const std::vector<int64_t> &arr, size_t left, size_t size,
                                                 std::vector<uint64_t> &keys) {
  int num_threads = GetTbbArena().max_concurrency();
  size_t grain_size = std::max(static_cast<size_t>(1), size / (static_cast<size_t>(4 * num_threads)));

  tbb::parallel_for(tbb::blocked_range<size_t>(0, size, grain_size), [&](const tbb::blocked_range<size_t> &range) {
    for (size_t index = range.begin(); index < range.end(); ++index) {
      keys[index] = ToUnsignedValue(arr[left + index]);
    }
  });
}

void LeonovaARadixMergeSortTBB::CountByteValues(const std::vector<uint64_t> &keys, size_t size, int shift,
                                                CounterTable &local_counts) {
  int num_threads = GetTbbArena().max_concurrency();
  size_t grain_size = std::max(static_cast<size_t>(1), size / (static_cast<size_t>(4 * num_threads)));

  tbb::parallel_for(tbb::blocked_range<size_t>(0, size, grain_size), [&](const tbb::blocked_range<size_t> &range) {
    int thread_idx = tbb::this_task_arena::current_thread_index();
    if (thread_idx < 0 || static_cast<size_t>(thread_idx) >= local_counts.size()) {
      thread_idx = 0;
    }

    auto &row = local_counts[thread_idx];

    for (size_t index = range.begin(); index < range.end(); ++index) {
      const auto byte_val = static_cast<size_t>((keys[index] >> shift) & 0xFFU);
      ++row[byte_val];
    }
  });
}

void LeonovaARadixMergeSortTBB::ScatterByte(const std::vector<uint64_t> &keys, const std::vector<int64_t> &arr,
                                            size_t left, size_t size, int shift, CounterTable &local_offsets,
                                            std::vector<int64_t> &temp_arr, std::vector<uint64_t> &temp_keys) {
  int num_threads = GetTbbArena().max_concurrency();
  size_t grain_size = std::max(static_cast<size_t>(1), size / (static_cast<size_t>(4 * num_threads)));

  tbb::parallel_for(tbb::blocked_range<size_t>(0, size, grain_size), [&](const tbb::blocked_range<size_t> &range) {
    int thread_idx = tbb::this_task_arena::current_thread_index();
    if (thread_idx < 0 || static_cast<size_t>(thread_idx) >= local_offsets.size()) {
      thread_idx = 0;
    }

    auto &thread_offsets = local_offsets[thread_idx];

    for (size_t index = range.begin(); index < range.end(); ++index) {
      const auto byte_val = static_cast<size_t>((keys[index] >> shift) & 0xFFU);
      const size_t pos = thread_offsets[byte_val]++;
      temp_arr[pos] = arr[left + index];
      temp_keys[pos] = keys[index];
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
      const auto byte_val = static_cast<size_t>((keys[i] >> shift) & 0xFFU);
      ++counts[byte_val];
    }

    size_t prefix = 0;
    for (size_t i = 0; i < kNumCounters; ++i) {
      offsets[i] = prefix;
      prefix += counts[i];
    }

    for (size_t i = 0; i < size; ++i) {
      const auto byte_val = static_cast<size_t>((keys[i] >> shift) & 0xFFU);
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

  const int requested_threads = std::max(1, ppc::util::GetNumThreads());
  const int tbb_threads = std::max(1, std::min(requested_threads, static_cast<int>(size / 1000)));
  const auto thread_count = static_cast<size_t>(tbb_threads);

  static thread_local std::vector<uint64_t> tls_keys;
  static thread_local std::vector<int64_t> tls_temp_arr;
  static thread_local std::vector<uint64_t> tls_temp_keys;

  tls_keys.resize(size);
  tls_temp_arr.resize(size);
  tls_temp_keys.resize(size);

  auto &keys = tls_keys;
  auto &temp_arr = tls_temp_arr;
  auto &temp_keys = tls_temp_keys;

  CounterTable local_counts(thread_count, CounterRow(kNumCounters, 0));
  CounterTable local_offsets(thread_count, CounterRow(kNumCounters, 0));

  auto &arena = GetTbbArena();
  arena.execute([&] {
    FillUnsignedKeys(arr, left, size, keys);

    for (int byte_pos = 0; byte_pos < kNumBytes; ++byte_pos) {
      const int shift = byte_pos * kByteSize;

      ResetLocalCounts(local_counts);

      CountByteValues(keys, size, shift, local_counts);

      BuildThreadOffsets(local_counts, thread_count, local_offsets);

      ScatterByte(keys, arr, left, size, shift, local_offsets, temp_arr, temp_keys);

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
