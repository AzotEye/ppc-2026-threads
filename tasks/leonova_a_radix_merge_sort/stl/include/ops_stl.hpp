#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <thread>
#include <utility>
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

  struct alignas(64) CounterRow {
    std::array<size_t, 256> values{};

    size_t &At(size_t index) {
      return values.at(index);
    }

    [[nodiscard]] const size_t &At(size_t index) const {
      return values.at(index);
    }

    void Fill(size_t value) {
      values.fill(value);
    }
  };

  using CounterTable = std::vector<CounterRow>;

  static void RadixMergeSort(std::vector<int64_t> &arr, size_t left, size_t right);

  static void RadixSort(std::vector<int64_t> &arr, size_t left, size_t right);

  static void SequentialRadixSort(std::vector<int64_t> &arr, size_t left, size_t right);

  static void SimpleMerge(std::vector<int64_t> &arr, size_t left, size_t mid, size_t right);

  static uint64_t ToUnsignedValue(int64_t value);

  static std::pair<size_t, size_t> GetChunk(size_t tid, size_t num_threads, size_t size);

  template <class Func>
  static void ParallelFor(size_t num_threads, Func func);

  static void FillUnsignedKeys(const std::vector<int64_t> &arr, size_t left, size_t size, std::vector<uint64_t> &keys,
                               size_t num_threads);

  static void CountBytesParallel(const std::vector<uint64_t> &keys, size_t size, int shift, CounterTable &local_counts,
                                 size_t num_threads);

  static void ReduceCounts(const CounterTable &local_counts, CounterRow &global_counts);

  static void BuildOffsets(const CounterTable &local_counts, CounterTable &local_offsets, CounterRow &global_counts,
                           size_t num_threads);

  static void ScatterParallel(const std::vector<uint64_t> &keys, const std::vector<int64_t> &arr, size_t left,
                              size_t size, int shift, CounterTable &local_offsets, std::vector<int64_t> &temp_arr,
                              std::vector<uint64_t> &temp_keys, size_t num_threads);

  static constexpr size_t kRadixThreshold = 131072;
  static constexpr size_t kMinParallelSize = 10000;

  static constexpr int kByteSize = 8;
  static constexpr int kNumBytes = 8;
  static constexpr int kNumCounters = 256;

  static constexpr uint64_t kSignBitMask = 0x8000000000000000ULL;
};

template <class Func>
void LeonovaARadixMergeSortSTL::ParallelFor(size_t num_threads, Func func) {
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (size_t tid = 0; tid < num_threads; ++tid) {
    threads.emplace_back([&, tid] { func(tid); });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

}  // namespace leonova_a_radix_merge_sort
