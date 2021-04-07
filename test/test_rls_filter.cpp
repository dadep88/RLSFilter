//
// Created by Davide Pilastro on 2/27/21.
//

#include <random>

#include "RLSFilter.h"
#include "gtest/gtest.h"

using namespace rls_filter;

constexpr long long int ITERATIONS = 1000000;

TEST(RLSFilter_StaticCtor, StadyStateEstimation_LongDouble) {
  long double lower_bound = 0;
  long double upper_bound = 10000;
  std::uniform_real_distribution<long double> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  RLSFilter<long double, 4> rls_filter(0.99999, 1.0);
  Matrix<long double, 4, 1> w_real(4);
  Matrix<long double, 4, 1> w0(4);
  w_real << 4.0, 0.5, 3.0, 1.0;
  w0 << 1.0, 1.0, 1.0, 1.0;

  rls_filter.setEstimatedCoefficients(w0);

  for (auto i = 0; i < ITERATIONS; i++) {
    Matrix<long double, 4, 1> x(4);
    x << unif(re), unif(re), unif(re), unif(re);
    double y = x.transpose() * w_real + unif(re) * 0.01;
    rls_filter.update(x, y);
  }
  ASSERT_TRUE(rls_filter.estimatedCoefficients().isApprox(w_real, 1e-3));
}

TEST(RLSFilter_DynamicCtor, StadyStateEstimation_LongDouble) {
  long double lower_bound = 0;
  long double upper_bound = 10000;
  std::uniform_real_distribution<long double> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  RLSFilter<long double, -1> rls_filter(4, 0.99999, 1.0);
  Matrix<long double, 4, 1> w_real(4);
  Matrix<long double, 4, 1> w0(4);
  w_real << 4.0, 0.5, 3.0, 1.0;
  w0 << 1.0, 1.0, 1.0, 1.0;

  rls_filter.setEstimatedCoefficients(w0);

  for (auto i = 0; i < ITERATIONS; i++) {
    Matrix<long double, 4, 1> x(4);
    x << unif(re), unif(re), unif(re), unif(re);
    double y = x.transpose() * w_real + unif(re) * 0.01;
    rls_filter.update(x, y);
  }
  ASSERT_TRUE(rls_filter.estimatedCoefficients().isApprox(w_real, 1e-3));
}

TEST(RLSFilter_StaticCtor, StadyStateEstimation_Double) {
  double lower_bound = 0;
  double upper_bound = 10000;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  RLSFilter<double, 4> rls_filter(0.99999, 1.0);
  Matrix<double, 4, 1> w_real(4);
  Matrix<double, 4, 1> w0(4);
  w_real << 4.0, 0.5, 3.0, 1.0;
  w0 << 1.0, 1.0, 1.0, 1.0;

  rls_filter.setEstimatedCoefficients(w0);

  for (auto i = 0; i < ITERATIONS; i++) {
    Matrix<double, 4, 1> x(4);
    x << unif(re), unif(re), unif(re), unif(re);
    double y = x.transpose() * w_real + unif(re) * 0.01;
    rls_filter.update(x, y);
  }
  ASSERT_TRUE(rls_filter.estimatedCoefficients().isApprox(w_real, 1e-3));
}

TEST(RLSFilter_DynamicCtor, StadyStateEstimation_Double) {
  double lower_bound = 0;
  double upper_bound = 10000;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  RLSFilter<double, -1> rls_filter(4, 0.99999, 1.0);
  Matrix<double, 4, 1> w_real(4);
  Matrix<double, 4, 1> w0(4);
  w_real << 4.0, 0.5, 3.0, 1.0;
  w0 << 1.0, 1.0, 1.0, 1.0;

  rls_filter.setEstimatedCoefficients(w0);

  for (auto i = 0; i < ITERATIONS; i++) {
    Matrix<double, 4, 1> x(4);
    x << unif(re), unif(re), unif(re), unif(re);
    double y = x.transpose() * w_real + unif(re) * 0.01;
    rls_filter.update(x, y);
  }
  ASSERT_TRUE(rls_filter.estimatedCoefficients().isApprox(w_real, 1e-3));
}

TEST(RLSFilter_StaticCtor, StadyStateEstimation_Float) {
  float lower_bound = 0;
  float upper_bound = 10000;
  std::uniform_real_distribution<float> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  RLSFilter<float, 4> rls_filter(0.99999, 1.0);
  Matrix<float, 4, 1> w_real(4);
  Matrix<float, 4, 1> w0(4);
  w_real << 4.0, 0.5, 3.0, 1.0;
  w0 << 1.0, 1.0, 1.0, 1.0;

  rls_filter.setEstimatedCoefficients(w0);

  for (auto i = 0; i < ITERATIONS; i++) {
    Matrix<float, 4, 1> x(4);
    x << unif(re), unif(re), unif(re), unif(re);
    float y = x.transpose() * w_real + unif(re) * 0.01;
    rls_filter.update(x, y);
  }
  ASSERT_TRUE(rls_filter.estimatedCoefficients().isApprox(w_real, 1e-3));
}

TEST(RLSFilter_DynamicCtor, StadyStateEstimation_Float) {
  float lower_bound = 0;
  float upper_bound = 10000;
  std::uniform_real_distribution<float> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  RLSFilter<float, -1> rls_filter(4, 0.99999, 1.0);
  Matrix<float, 4, 1> w_real(4);
  Matrix<float, 4, 1> w0(4);
  w_real << 4.0, 0.5, 3.0, 1.0;
  w0 << 1.0, 1.0, 1.0, 1.0;

  rls_filter.setEstimatedCoefficients(w0);

  for (auto i = 0; i < ITERATIONS; i++) {
    Matrix<float, 4, 1> x(4);
    x << unif(re), unif(re), unif(re), unif(re);
    float y = x.transpose() * w_real + unif(re) * 0.01;
    rls_filter.update(x, y);
  }
  ASSERT_TRUE(rls_filter.estimatedCoefficients().isApprox(w_real, 1e-3));
}

TEST(RLSFilter_DynamicCtor, CreateRandomOrderFilter) {
  int order_lower_bound = 1;
  int order_upper_bound = 100;
  std::uniform_int_distribution<int> unif(order_lower_bound, order_upper_bound);
  std::default_random_engine re;

  int random_order = unif(re);
  RLSFilter<float, -1> rls_filter(random_order, 0.99999, 1.0);
  VectorXf w0 = VectorXf::Zero(random_order);
  ASSERT_TRUE(rls_filter.estimatedCoefficients().isApprox(w0, 1e-3));
}
