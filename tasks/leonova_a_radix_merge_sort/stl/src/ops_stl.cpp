#include "leonova_a_radix_merge_sort/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

std::pair<size_t, size_t> LeonovaARadixMergeSortSTL::GetChunk(size_t tid, size_t num_threads, size_t size) {
  const size_t chunk = (size + num_threads - 1) / num_threads;

  const size_t begin = tid * chunk;
  const size_t end = std::min(begin + chunk, size);

  return {begin, end};
}

void LeonovaARadixMergeSortSTL::FillUnsignedKeys(const std::vector<int64_t> &arr, size_t left, size_t size,
                                                 std::vector<uint64_t> &keys, size_t num_threads) {
  ParallelFor(num_threads, [&](size_t tid) {
    auto [begin, end] = GetChunk(tid, num_threads, size);

    for (size_t index = begin; index < end; ++index) {
      keys[index] = ToUnsignedValue(arr[left + index]);
    }
  });
}

void LeonovaARadixMergeSortSTL::CountBytesParallel(const std::vector<uint64_t> &keys, size_t size, int shift,
                                                   CounterTable &local_counts, size_t num_threads) {
  ParallelFor(num_threads, [&](size_t tid) {
    auto [begin, end] = GetChunk(tid, num_threads, size);

    auto &row = local_counts[tid];

    for (size_t index = begin; index < end; ++index) {
      const size_t byte = (keys[index] >> shift) & 0xFFU;

      ++row.At(byte);
    }
  });
}

void LeonovaARadixMergeSortSTL::ReduceCounts(const CounterTable &local_counts, CounterRow &global_counts) {
  global_counts.Fill(0);

  for (const auto &row : local_counts) {
    for (size_t index = 0; index < kNumCounters; ++index) {
      global_counts.At(index) += row.At(index);
    }
  }
}

void LeonovaARadixMergeSortSTL::BuildOffsets(const CounterTable &local_counts, CounterTable &local_offsets,
                                             CounterRow &global_counts, size_t num_threads) {
  CounterRow bucket_totals = global_counts;

  size_t prefix = 0;

  for (size_t index = 0; index < kNumCounters; ++index) {
    const size_t count = bucket_totals.At(index);

    bucket_totals.At(index) = prefix;
    prefix += count;
  }

  for (size_t tid = 0; tid < num_threads; ++tid) {
    auto &offset_row = local_offsets[tid];
    const auto &count_row = local_counts[tid];

    for (size_t index = 0; index < kNumCounters; ++index) {
      offset_row.At(index) = bucket_totals.At(index);

      bucket_totals.At(index) += count_row.At(index);
    }
  }
}

void LeonovaARadixMergeSortSTL::ScatterParallel(const std::vector<uint64_t> &keys, const std::vector<int64_t> &arr,
                                                size_t left, size_t size, int shift, CounterTable &local_offsets,
                                                std::vector<int64_t> &temp_arr, std::vector<uint64_t> &temp_keys,
                                                size_t num_threads) {
  ParallelFor(num_threads, [&](size_t tid) {
    auto [begin, end] = GetChunk(tid, num_threads, size);

    auto offsets = local_offsets[tid];

    for (size_t index = begin; index < end; ++index) {
      const size_t byte = (keys[index] >> shift) & 0xFFU;

      const size_t pos = offsets.At(byte)++;

      temp_arr[pos] = arr[left + index];
      temp_keys[pos] = keys[index];
    }
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

    CounterRow counts{};
    CounterRow offsets{};

    for (size_t index = 0; index < size; ++index) {
      const size_t byte = (keys[index] >> shift) & 0xFFU;

      ++counts.At(byte);
    }

    size_t prefix = 0;

    for (size_t index = 0; index < kNumCounters; ++index) {
      offsets.At(index) = prefix;
      prefix += counts.At(index);
    }

    for (size_t index = 0; index < size; ++index) {
      const size_t byte = (keys[index] >> shift) & 0xFFU;

      const size_t pos = offsets.At(byte)++;

      temp_arr[pos] = arr[left + index];
      temp_keys[pos] = keys[index];
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

  const size_t num_threads =
      std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(ppc::util::GetNumThreads()), size));

  std::vector<uint64_t> keys(size);
  std::vector<uint64_t> temp_keys(size);
  std::vector<int64_t> temp_arr(size);

  CounterTable local_counts(num_threads);
  CounterTable local_offsets(num_threads);

  CounterRow global_counts{};

  FillUnsignedKeys(arr, left, size, keys, num_threads);

  for (int byte_pos = 0; byte_pos < kNumBytes; ++byte_pos) {
    const int shift = byte_pos * kByteSize;

    for (auto &row : local_counts) {
      row.Fill(0);
    }

    CountBytesParallel(keys, size, shift, local_counts, num_threads);

    ReduceCounts(local_counts, global_counts);

    BuildOffsets(local_counts, local_offsets, global_counts, num_threads);

    ScatterParallel(keys, arr, left, size, shift, local_offsets, temp_arr, temp_keys, num_threads);

    std::ranges::copy(temp_arr, arr.begin() + static_cast<std::ptrdiff_t>(left));

    keys.swap(temp_keys);
  }
}

void LeonovaARadixMergeSortSTL::SimpleMerge(std::vector<int64_t> &arr, size_t left, size_t mid, size_t right) {
  std::vector<int64_t> merged(right - left);

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

  std::ranges::copy(merged, arr.begin() + static_cast<std::ptrdiff_t>(left));
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
