#include "vinyaikina_e_multidimensional_integrals_simpson_method/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stack>
#include <vector>

#include "util/include/util.hpp"
#include "vinyaikina_e_multidimensional_integrals_simpson_method/common/include/common.hpp"

namespace vinyaikina_e_multidimensional_integrals_simpson_method {
namespace {

double customRound(double value, double h) {
  h *= 2;
  int tmp = static_cast<int>(1 / h);
  int decimalPlaces = 0;
  while (tmp > 0 && tmp % 10 == 0) {
    decimalPlaces++;
    tmp /= 10;
  }

  double factor = std::pow(10.0, decimalPlaces);
  return std::round(value * factor) / factor;
}

double count(double left_border, double right_border, double simpson_factor,
             std::vector<std::pair<double, double>> &limits, std::vector<double> &actual_step,

             std::function<double(const std::vector<double> &)> function) {
  std::stack<std::pair<std::vector<double>, double>> stack;
  double i_res = 0.0;

  int steps_count_0 = static_cast<int>((right_border - left_border) / actual_step[0] + 0.5);

  for (int i0 = 0; i0 <= steps_count_0; ++i0) {
    double x0 = left_border + i0 * actual_step[0];

    double weight_0 = 2.0;
    if (i0 == 0 || i0 == steps_count_0) {
      weight_0 = 1.0;
    } else if (i0 % 2 != 0) {
      weight_0 = 4.0;
    }

    stack.emplace(std::vector<double>{x0}, weight_0);

    while (!stack.empty()) {
      std::vector<double> point = stack.top().first;
      double weight = stack.top().second;
      stack.pop();

      if (point.size() == limits.size()) {
        i_res += function(point) * weight * simpson_factor;
        continue;
      }

      int dim = point.size();
      double step = actual_step[dim];

      int steps_count = static_cast<int>((limits[dim].second - limits[dim].first) / step + 0.5);

      for (int i = 0; i <= steps_count; ++i) {
        double x = limits[dim].first + i * step;

        double dim_weight = 2.0;
        if (i == 0 || i == steps_count) {
          dim_weight = 1.0;
        } else if (i % 2 != 0) {
          dim_weight = 4.0;
        }

        point.push_back(x);
        stack.emplace(point, weight * dim_weight);
        point.pop_back();
      }
    }
  }

  return i_res;
}
};  // namespace

VinyaikinaEMultidimIntegrSimpsonOMP::VinyaikinaEMultidimIntegrSimpsonOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VinyaikinaEMultidimIntegrSimpsonOMP::PreProcessingImpl() {
  I_res = 0.0;

  return true;
}

bool VinyaikinaEMultidimIntegrSimpsonOMP::ValidationImpl() {
  const auto &[h, limits, function] = GetInput();
  if (limits.empty() || !function || h > 0.01) {
    return false;
  }
  return true;
}

bool VinyaikinaEMultidimIntegrSimpsonOMP::RunImpl() {
  auto &[h, limits, function] = GetInput();

  const int num_threads = ppc::util::GetNumThreads();

  double delta = limits[0].second / num_threads - limits[0].first / num_threads;
  double res = 0.0;

  std::vector<double> actual_step(limits.size());
  double simpson_factor = 1.0;

  for (int i = 0; i < limits.size(); i++) {
    int quan_steps = ((limits[i].second - limits[i].first) / (h) + 0.5);
    if (quan_steps % 2 != 0) {
      quan_steps++;
    }
    actual_step[i] = (limits[i].second - limits[i].first) / quan_steps;
    simpson_factor *= actual_step[i] / 3.0;
  }

#pragma omp parallel num_threads(num_threads) default(none) \
    shared(limits, simpson_factor, actual_step, delta, function, std::cout) reduction(+ : res)
  {
    double left_border, right_border;
    if (omp_get_thread_num() != 0) {
      left_border = customRound(limits[0].first + delta * omp_get_thread_num(), actual_step[0]);
    } else {
      left_border = limits[0].first;
    }
    if (omp_get_thread_num() != num_threads - 1) {
      right_border = customRound(limits[0].second - delta * (num_threads - omp_get_thread_num() - 1), actual_step[0]);
    } else {
      right_border = limits[0].second;
    }

    res += count(left_border, right_border, simpson_factor, limits, actual_step, function);
  }

  I_res = res;

  return true;
}

bool VinyaikinaEMultidimIntegrSimpsonOMP::PostProcessingImpl() {
  GetOutput() = I_res;
  return true;
}
}  // namespace vinyaikina_e_multidimensional_integrals_simpson_method
