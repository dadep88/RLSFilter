//
// Created by Davide Pilastro on 2/27/21.
//


#include <random>
#include "RLSFilter.h"
#include "gtest/gtest.h"

using namespace rls_filter;

TEST(RLSFilterEstimation, StadyStateEstimation) {

    double lower_bound = 0;
    double upper_bound = 10000;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

  RLSFilter rls_filter(4, 0.99999, 1.0);
  VectorXd real_coeffs(4);
  real_coeffs << 4, 0.5, 3.0,  1.0;

  for (auto i=0; i < 10000; i++){
      VectorXd x(4);
      x << unif(re), unif(re), unif(re), unif(re);
      double y = x.transpose() * real_coeffs;
      rls_filter.update(x, y);
  }
    ASSERT_TRUE(rls_filter.estimated_coefficients().isApprox(real_coeffs, 1e-5));
}
