#include "zyazeva_s_matrix_mult_cannon_alg/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "zyazeva_s_matrix_mult_cannon_alg/common/include/common.hpp"

namespace zyazeva_s_matrix_mult_cannon_alg {

std::vector<double> CannonMatrixMultiplication(const std::vector<double> &a, const std::vector<double> &b, int n) {
  int size_block = std::min(n, 64);

  std::vector<double> mtrx_c(n * n, 0.0);//6

  if (n == 0) {
    return {};
  }

#pragma omp parallel for schedule(static) collapse(2)
  for (int i = 0; i < n; i += size_block) {
    for (int j = 0; j < n; j += size_block) {
      for (int k = 0; k < n; k += size_block) {
        int i_end = std::min(i + size_block, n);
        int j_end = std::min(j + size_block, n);
        int k_end = std::min(k + size_block, n);

        for (int ii = i; ii < i_end; ++ii) {
          for (int kk = k; kk < k_end; ++kk) {
            double a_ik = a[(ii * n) + kk];
            for (int jj = j; jj < j_end; ++jj) {
              mtrx_c[(ii * n) + jj] += a_ik * b[(kk * n) + jj];
            }
          }
        }
      }
    }
  }

  return mtrx_c;
}

ZyazevaSMatrixMultCannonAlgOMP::ZyazevaSMatrixMultCannonAlgOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool ZyazevaSMatrixMultCannonAlgOMP::ValidationImpl() {
  const size_t sz = std::get<0>(GetInput());
  const auto &m1 = std::get<1>(GetInput());
  const auto &m2 = std::get<2>(GetInput());

  return sz > 0 && m1.size() == sz * sz && m2.size() == sz * sz;
}

bool ZyazevaSMatrixMultCannonAlgOMP::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool ZyazevaSMatrixMultCannonAlgOMP::RunImpl() {
  const auto sz = std::get<0>(GetInput());
  const auto &m1 = std::get<1>(GetInput());
  const auto &m2 = std::get<2>(GetInput());

  std::vector<double> res_m = CannonMatrixMultiplication(m1, m2, static_cast<int>(sz));

  GetOutput() = res_m;
  return true;
}

bool ZyazevaSMatrixMultCannonAlgOMP::PostProcessingImpl() {
  return true;
}

}  // namespace zyazeva_s_matrix_mult_cannon_alg
