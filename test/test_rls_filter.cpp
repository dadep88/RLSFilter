//
// Created by Davide Pilastro on 2/27/21.
//
#include "RLSFilter.h"
#include "gtest/gtest.h"

using namespace rls_filter;

TEST(RLSFilterEstimation, sample_test) {
  RLSFilter rls_filter(4, 0.99999, 1.0);

  EXPECT_EQ(1, 1);
}
