//
// Created by Davide Pilastro on 2/27/21.
//

#include "RLSFilter.h"
#include "gtest/gtest.h"
#include <random>

using namespace rls_filter;

TEST(RLSFilterEstimation, StadyStateEstimation) {

  double lower_bound = 0;
  double upper_bound = 10000;
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  RLSFilter rls_filter(4, 0.99999, 1.0);
  VectorXd w_real(4);
  VectorXd w0(4);
  w_real << 4.0, 0.5, 3.0, 1.0;
  w0 << 1.0, 1.0, 1.0, 1.0;

  rls_filter.set_estimated_coefficients(w0);

  for (auto i = 0; i < 100000; i++) {
    VectorXd x(4);
    x << unif(re), unif(re), unif(re), unif(re);
    double y = x.transpose() * w_real + unif(re) * 0.01;
    rls_filter.update(x, y);
  }
  ASSERT_TRUE(rls_filter.estimated_coefficients().isApprox(w_real, 1e-3));
}
