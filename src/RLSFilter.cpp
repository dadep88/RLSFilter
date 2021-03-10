//
// Created by Davide Pilastro on 2/20/21.
//
#include "RLSFilter.h"

#include <stdexcept>

using namespace rls_filter;
using namespace Eigen;

RLSFilter::RLSFilter(unsigned int n, double lam, double delta)
    : n_(n), delta_(delta), w_(VectorXd::Zero(n_)),
      P_(MatrixXd::Identity(n_, n_) * delta_), g_(VectorXd::Zero(n_)),
      count_(0) {
  set_forgetting_factor(lam);
};

void RLSFilter::set_estimated_coeffs(const VectorXd &w0) {
  if (w0.rows() == n_) {
    w_ = w0;
  } else {
    throw std::invalid_argument("Wrong initial state dimension.");
  }
}

void RLSFilter::set_forgetting_factor(const double lam) {
  if ((lam > 0) && (lam <= 1.0)) {
    lam_ = lam;
    lam_inv_ = 1.0 / lam_;
  } else {
    throw std::invalid_argument("Invalid forgetting factor (0 < lambda <= 1).");
    ;
  }
}

void RLSFilter::update(const VectorXd x, const double y) {
  err_ = y - predict(x);
  MatrixXd alpha = P_ * lam_inv_;
  g_ = (P_ * x) / (lam_ + x.transpose() * P_ * x);
  P_ = (MatrixXd::Identity(n_, n_) - g_ * x.transpose()) * alpha;
  w_ += g_ * err_;
  count_++;
};
