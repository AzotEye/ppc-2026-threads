// #include <gtest/gtest.h>

// #include "nikolaev_d_block_linear_image_filtering_seq/all/include/ops_all.hpp"
// #include "nikolaev_d_block_linear_image_filtering_seq/common/include/common.hpp"
// #include "nikolaev_d_block_linear_image_filtering_seq/omp/include/ops_omp.hpp"
// #include "nikolaev_d_block_linear_image_filtering_seq/seq/include/ops_seq.hpp"
// #include "nikolaev_d_block_linear_image_filtering_seq/stl/include/ops_stl.hpp"
// #include "nikolaev_d_block_linear_image_filtering_seq/tbb/include/ops_tbb.hpp"
// #include "util/include/perf_test_util.hpp"

// namespace nikolaev_d_block_linear_image_filtering {

// class ExampleRunPerfTestThreads : public ppc::util::BaseRunPerfTests<InType, OutType> {
//   const int kCount_ = 200;
//   InType input_data_{};

//   void SetUp() override {
//     input_data_ = kCount_;
//   }

//   bool CheckTestOutputData(OutType &output_data) final {
//     return input_data_ == output_data;
//   }

//   InType GetTestInputData() final {
//     return input_data_;
//   }
// };

// TEST_P(ExampleRunPerfTestThreads, RunPerfModes) {
//   ExecuteTest(GetParam());
// }

// namespace {

// const auto kAllPerfTasks =
//     ppc::util::MakeAllPerfTasks<InType, NesterovATestTaskALL, NesterovATestTaskOMP, NikolaevDBlockLinearImageFilteringSEQ,
//                                 NesterovATestTaskSTL, NesterovATestTaskTBB>(PPC_SETTINGS_nikolaev_d_block_linear_image_filtering_seq);

// const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

// const auto kPerfTestName = ExampleRunPerfTestThreads::CustomPerfTestName;

// INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestThreads, kGtestValues, kPerfTestName);

// }  // namespace

// }  // namespace nikolaev_d_block_linear_image_filtering
