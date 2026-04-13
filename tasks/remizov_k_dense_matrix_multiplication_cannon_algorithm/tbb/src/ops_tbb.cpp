#include "remizov_k_dense_matrix_multiplication_cannon_algorithm/tbb/include/ops_tbb.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include "remizov_k_dense_matrix_multiplication_cannon_algorithm/common/include/common.hpp"

namespace remizov_k_dense_matrix_multiplication_cannon_algorithm {

RemizovKDenseMatrixMultiplicationCannonAlgorithmTbb::RemizovKDenseMatrixMultiplicationCannonAlgorithmTbb(
    const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RemizovKDenseMatrixMultiplicationCannonAlgorithmTbb::ValidationImpl() {
  const auto &input_data = GetInput();

  int block_dim = std::get<0>(input_data);
  const auto &mat_a = std::get<1>(input_data);
  const auto &mat_b = std::get<2>(input_data);

  if (block_dim <= 0) {
    return false;
  }
  if (mat_a.empty() || mat_b.empty()) {
    return false;
  }

  size_t n = mat_a.size();
  if (n != mat_a[0].size()) {
    return false;
  }
  if (n != mat_b.size() || n != mat_b[0].size()) {
    return false;
  }

  return (n % static_cast<size_t>(block_dim) == 0);
}

bool RemizovKDenseMatrixMultiplicationCannonAlgorithmTbb::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

void RemizovKDenseMatrixMultiplicationCannonAlgorithmTbb::MultiplyBlock(
    const std::vector<std::vector<double>> &a,
    const std::vector<std::vector<double>> &b,
    std::vector<std::vector<double>> &c,
    int block_size) {
  for (int i = 0; i < block_size; ++i) {
    for (int j = 0; j < block_size; ++j) {
      double accumulator = 0.0;
      for (int k = 0; k < block_size; ++k) {
        accumulator += a[i][k] * b[k][j];
      }
      c[i][j] += accumulator;
    }
  }
}

void RemizovKDenseMatrixMultiplicationCannonAlgorithmTbb::ShiftBlocksLeft(
    std::vector<std::vector<std::vector<std::vector<double>>>> &matrix_blocks,
    int block_count) {
  // Parallelize over rows; each row shift is independent
  tbb::parallel_for(0, block_count, [&](int i) {
    auto first_element = std::move(matrix_blocks[i][0]);
    for (int j = 1; j < block_count; ++j) {
      matrix_blocks[i][j - 1] = std::move(matrix_blocks[i][j]);
    }
    matrix_blocks[i][block_count - 1] = std::move(first_element);
  });
}

}  // namespace remizov_k_dense_matrix_multiplication_cannon_algorithm
