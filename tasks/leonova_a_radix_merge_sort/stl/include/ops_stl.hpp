#pragma once

#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "leonova_a_radix_merge_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace leonova_a_radix_merge_sort {

class LeonovaARadixMergeSortSTL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSTL;
  }

  explicit LeonovaARadixMergeSortSTL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  using CounterRow = std::vector<size_t>;
  using CounterTable = std::vector<CounterRow>;

  static void RadixMergeSort(std::vector<int64_t> &arr, size_t left, size_t right);
  static void RadixSort(std::vector<int64_t> &arr, size_t left, size_t right);
  static void SequentialRadixSort(std::vector<int64_t> &arr, size_t left, size_t right);
  static void SimpleMerge(std::vector<int64_t> &arr, size_t left, size_t mid, size_t right);

  static uint64_t ToUnsignedValue(int64_t value);

  static size_t GetThreadCount(size_t size);

  static std::pair<size_t, size_t> GetChunk(size_t tid, size_t thread_count, size_t size);

  static void FillUnsignedKeys(const std::vector<int64_t> &arr, size_t left, size_t size, std::vector<uint64_t> &keys,
                               size_t thread_count);

  static void CountBytesParallel(const std::vector<uint64_t> &keys, size_t size, int shift, CounterTable &local_counts,
                                 size_t thread_count);

  static void BuildOffsets(const CounterTable &local_counts, CounterTable &local_offsets, size_t thread_count);

  static void ScatterParallel(const std::vector<uint64_t> &keys, const std::vector<int64_t> &arr, size_t left,
                              size_t size, int shift, CounterTable &local_offsets, std::vector<int64_t> &temp_arr,
                              std::vector<uint64_t> &temp_keys, size_t thread_count);

  static void ParallelCopy(std::vector<int64_t> &arr, size_t left, const std::vector<int64_t> &temp_arr,
                           size_t thread_count);

  static void RunThreads(size_t thread_count, const std::function<void(size_t)> &func);

  static constexpr size_t kRadixThreshold = 131072;
  static constexpr size_t kMinParallelSize = 10000;
  static constexpr int kByteSize = 8;
  static constexpr int kNumBytes = 8;
  static constexpr int kNumCounters = 256;
  static constexpr uint64_t kSignBitMask = 0x8000000000000000ULL;
};

}  // namespace leonova_a_radix_merge_sort
