//
// Created by Davide Pilastro on 2/19/21.
//

#ifndef RLS_DYN_MODEL_IDENT_RLS_H
#define RLS_DYN_MODEL_IDENT_RLS_H

#include <Eigen/Dense>
using namespace Eigen;

namespace rls_filter {

class RLSFilter {
private:
  unsigned int n_;           /**< Filter order */
  double lam_;               /**< Forgetting factor */
  double lam_inv_;           /**< Inverse forgetting factor */
  double delta_;             /**< Initial gain value of matrix P */
  VectorXd w_;               /**< Filter coefficients vector */
  MatrixXd P_;               /**< Covariance error matrix */
  VectorXd g_;               /**< Filter gains */
  double err_;               /**< A priori error */
  unsigned long long count_; /**< Count of filter updates */

public:
  /// Recursive least square ctor
  /// \param n - Filter order
  /// \param lam - Forgetting factor
  /// \param delta - Initial gain value of matrix P
  RLSFilter(unsigned int n, double lam, double delta)
      : n_(n), lam_(1.0), lam_inv_(1.0), delta_(delta), w_(VectorXd::Zero(n_)),
        P_(MatrixXd::Identity(n_, n_) * delta_), g_(VectorXd::Zero(n_)),
        err_(0.0), count_(0) {
    set_forgetting_factor(lam);
  }

  /// Update filter with new data
  /// \param x - Input vector
  /// \param y - Output value
  void update(const VectorXd &x, const double y) {
    err_ = y - predict(x);
    MatrixXd alpha = P_ * lam_inv_;
    g_ = (P_ * x) / (lam_ + x.transpose() * P_ * x);
    P_ = (MatrixXd::Identity(n_, n_) - g_ * x.transpose()) * alpha;
    w_ += g_ * err_;
    count_++;
  };

  /// Estimate filter output
  /// \param x
  /// \return a priori output estimate
  [[nodiscard]] double predict(const VectorXd &x) const noexcept {
    return w_.transpose() * x;
  };

  /// Set filter coefficient values
  /// \param w0 - Coefficient values
  void set_estimated_coefficients(const VectorXd &w0) {
    if (w0.rows() == n_) {
      w_ = w0;
    } else {
      throw std::invalid_argument("Wrong initial state dimension.");
    }
  }

  ///  Set forgetting factor value
  /// \param lam - Forgetting factor value
  void set_forgetting_factor(const double lam) {
    if ((lam > 0) && (lam <= 1.0)) {
      lam_ = lam;
      lam_inv_ = 1.0 / lam_;
    } else {
      throw std::invalid_argument(
          "Invalid forgetting factor (0 < lambda <= 1).");
    }
  }

  /// Get estimated filter coefficients
  /// \return vector od estimated filter coefficients
  [[nodiscard]] const VectorXd &estimated_coefficients() const noexcept {
    return w_;
  };

  /// Get a priori estimate error
  /// \return a priori error
  [[nodiscard]] const double a_priori_err() const noexcept { return err_; };

  /// Get filter gains vector
  /// \return filter gains vector
  [[nodiscard]] const VectorXd &gains() const noexcept { return g_; };

  /// Get filter covariance matrix
  /// \return filter covariance matrix
  [[nodiscard]] const MatrixXd &P() const noexcept { return P_; };
};

} // namespace rls_filter

#endif // RLS_DYN_MODEL_IDENT_RLS_H