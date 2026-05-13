#include "leonova_a_radix_merge_sort/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

#include "leonova_a_radix_merge_sort/common/include/common.hpp"
#include "util/include/util.hpp"

namespace leonova_a_radix_merge_sort {

LeonovaARadixMergeSortSTL::LeonovaARadixMergeSortSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int64_t>(GetInput().size());
}

bool LeonovaARadixMergeSortSTL::ValidationImpl() {
  return !GetInput().empty();
}

bool LeonovaARadixMergeSortSTL::PreProcessingImpl() {
  return true;
}

bool LeonovaARadixMergeSortSTL::RunImpl() {
  if (!ValidationImpl()) {
    return false;
  }

  GetOutput() = GetInput();

  if (GetOutput().size() > 1) {
    RadixMergeSort(GetOutput(), 0, GetOutput().size());
  }

  return true;
}

bool LeonovaARadixMergeSortSTL::PostProcessingImpl() {
  return true;
}

inline uint64_t LeonovaARadixMergeSortSTL::ToUnsignedValue(int64_t value) {
  return static_cast<uint64_t>(value) ^ kSignBitMask;
}

size_t LeonovaARadixMergeSortSTL::GetThreadCount(size_t size) {
  const size_t requested_threads = static_cast<size_t>(std::max(1, ppc::util::GetNumThreads()));

  return std::max<size_t>(1, std::min(requested_threads, size));
}

std::pair<size_t, size_t> LeonovaARadixMergeSortSTL::GetChunk(size_t tid, size_t thread_count, size_t size) {
  const size_t chunk_size = (size + thread_count - 1) / thread_count;

  const size_t begin = tid * chunk_size;
  const size_t end = std::min(begin + chunk_size, size);

  return {begin, end};
}

void LeonovaARadixMergeSortSTL::RunThreads(size_t thread_count, const std::function<void(size_t)> &func) {
  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  for (size_t tid = 0; tid < thread_count; ++tid) {
    threads.emplace_back([&, tid] { func(tid); });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

void LeonovaARadixMergeSortSTL::FillUnsignedKeys(const std::vector<int64_t> &arr, size_t left, size_t size,
                                                 std::vector<uint64_t> &keys, size_t thread_count) {
  RunThreads(thread_count, [&](size_t tid) {
    const auto [begin, end] = GetChunk(tid, thread_count, size);

    for (size_t index = begin; index < end; ++index) {
      keys[index] = ToUnsignedValue(arr[left + index]);
    }
  });
}

void LeonovaARadixMergeSortSTL::CountBytesParallel(const std::vector<uint64_t> &keys, size_t size, int shift,
                                                   CounterTable &local_counts, size_t thread_count) {
  RunThreads(thread_count, [&](size_t tid) {
    const auto [begin, end] = GetChunk(tid, thread_count, size);

    auto &row = local_counts[tid];

    std::ranges::fill(row, 0);

    for (size_t index = begin; index < end; ++index) {
      const size_t byte = (keys[index] >> shift) & 0xFFU;
      ++row[byte];
    }
  });
}

void LeonovaARadixMergeSortSTL::BuildOffsets(const CounterTable &local_counts, CounterTable &local_offsets,
                                             size_t thread_count) {
  CounterRow bucket_totals(kNumCounters, 0);

  for (size_t tid = 0; tid < thread_count; ++tid) {
    const auto &row = local_counts[tid];

    for (size_t index = 0; index < kNumCounters; ++index) {
      bucket_totals[index] += row[index];
    }
  }

  size_t prefix = 0;

  for (size_t index = 0; index < kNumCounters; ++index) {
    const size_t count = bucket_totals[index];
    bucket_totals[index] = prefix;
    prefix += count;
  }

  for (size_t tid = 0; tid < thread_count; ++tid) {
    auto &offset_row = local_offsets[tid];
    const auto &count_row = local_counts[tid];

    for (size_t index = 0; index < kNumCounters; ++index) {
      offset_row[index] = bucket_totals[index];
      bucket_totals[index] += count_row[index];
    }
  }
}

void LeonovaARadixMergeSortSTL::ScatterParallel(const std::vector<uint64_t> &keys, const std::vector<int64_t> &arr,
                                                size_t left, size_t size, int shift, CounterTable &local_offsets,
                                                std::vector<int64_t> &temp_arr, std::vector<uint64_t> &temp_keys,
                                                size_t thread_count) {
  RunThreads(thread_count, [&](size_t tid) {
    const auto [begin, end] = GetChunk(tid, thread_count, size);

    auto offsets = local_offsets[tid];

    for (size_t index = begin; index < end; ++index) {
      const size_t byte = (keys[index] >> shift) & 0xFFU;
      const size_t position = offsets[byte]++;

      temp_arr[position] = arr[left + index];
      temp_keys[position] = keys[index];
    }
  });
}

void LeonovaARadixMergeSortSTL::ParallelCopy(std::vector<int64_t> &arr, size_t left,
                                             const std::vector<int64_t> &temp_arr, size_t thread_count) {
  const size_t size = temp_arr.size();

  RunThreads(thread_count, [&](size_t tid) {
    const auto [begin, end] = GetChunk(tid, thread_count, size);

    std::copy(temp_arr.begin() + static_cast<std::ptrdiff_t>(begin),
              temp_arr.begin() + static_cast<std::ptrdiff_t>(end),
              arr.begin() + static_cast<std::ptrdiff_t>(left + begin));
  });
}

void LeonovaARadixMergeSortSTL::SequentialRadixSort(std::vector<int64_t> &arr, size_t left, size_t right) {
  const size_t size = right - left;

  if (size <= 1) {
    return;
  }

  std::vector<uint64_t> keys(size);
  std::vector<uint64_t> temp_keys(size);
  std::vector<int64_t> temp_arr(size);

  for (size_t index = 0; index < size; ++index) {
    keys[index] = ToUnsignedValue(arr[left + index]);
  }

  for (int byte_pos = 0; byte_pos < kNumBytes; ++byte_pos) {
    const int shift = byte_pos * kByteSize;

    CounterRow counts(kNumCounters, 0);
    CounterRow offsets(kNumCounters, 0);

    for (size_t index = 0; index < size; ++index) {
      const size_t byte = (keys[index] >> shift) & 0xFFU;
      ++counts[byte];
    }

    size_t prefix = 0;

    for (size_t index = 0; index < kNumCounters; ++index) {
      offsets[index] = prefix;
      prefix += counts[index];
    }

    for (size_t index = 0; index < size; ++index) {
      const size_t byte = (keys[index] >> shift) & 0xFFU;
      const size_t position = offsets[byte]++;

      temp_arr[position] = arr[left + index];
      temp_keys[position] = keys[index];
    }

    std::ranges::copy(temp_arr, arr.begin() + static_cast<std::ptrdiff_t>(left));

    keys.swap(temp_keys);
  }
}

void LeonovaARadixMergeSortSTL::RadixSort(std::vector<int64_t> &arr, size_t left, size_t right) {
  const size_t size = right - left;

  if (size <= 1) {
    return;
  }

  if (size < kMinParallelSize) {
    SequentialRadixSort(arr, left, right);
    return;
  }

  const size_t thread_count = GetThreadCount(size);

  std::vector<uint64_t> keys(size);
  std::vector<uint64_t> temp_keys(size);
  std::vector<int64_t> temp_arr(size);

  CounterTable local_counts(thread_count, CounterRow(kNumCounters, 0));

  CounterTable local_offsets(thread_count, CounterRow(kNumCounters, 0));

  FillUnsignedKeys(arr, left, size, keys, thread_count);

  for (int byte_pos = 0; byte_pos < kNumBytes; ++byte_pos) {
    const int shift = byte_pos * kByteSize;

    CountBytesParallel(keys, size, shift, local_counts, thread_count);

    BuildOffsets(local_counts, local_offsets, thread_count);

    ScatterParallel(keys, arr, left, size, shift, local_offsets, temp_arr, temp_keys, thread_count);

    ParallelCopy(arr, left, temp_arr, thread_count);

    keys.swap(temp_keys);
  }
}

void LeonovaARadixMergeSortSTL::SimpleMerge(std::vector<int64_t> &arr, size_t left, size_t mid, size_t right) {
  const size_t left_size = mid - left;
  const size_t right_size = right - mid;

  std::vector<int64_t> merged(left_size + right_size);

  size_t left_index = left;
  size_t right_index = mid;
  size_t merged_index = 0;

  while (left_index < mid && right_index < right) {
    if (arr[left_index] <= arr[right_index]) {
      merged[merged_index++] = arr[left_index++];
    } else {
      merged[merged_index++] = arr[right_index++];
    }
  }

  while (left_index < mid) {
    merged[merged_index++] = arr[left_index++];
  }

  while (right_index < right) {
    merged[merged_index++] = arr[right_index++];
  }

  std::ranges::copy(merged,

                    arr.begin() + static_cast<std::ptrdiff_t>(left));
}

void LeonovaARadixMergeSortSTL::RadixMergeSort(std::vector<int64_t> &arr, size_t left, size_t right) {
  struct Task {
    size_t left;
    size_t right;
    bool sorted;
  };

  std::vector<Task> stack;
  stack.reserve(128);

  stack.push_back({left, right, false});

  while (!stack.empty()) {
    const Task current = stack.back();
    stack.pop_back();

    const size_t size = current.right - current.left;

    if (size <= 1) {
      continue;
    }

    if (size <= kRadixThreshold) {
      RadixSort(arr, current.left, current.right);
      continue;
    }

    const size_t mid = current.left + (size / 2);

    if (!current.sorted) {
      stack.push_back({current.left, current.right, true});
      stack.push_back({mid, current.right, false});
      stack.push_back({current.left, mid, false});
    } else {
      SimpleMerge(arr, current.left, mid, current.right);
    }
  }
}

}  // namespace leonova_a_radix_merge_sort
