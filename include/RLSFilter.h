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
  unsigned int n_; /// Filter order
  double lam_;     /// Forgetting factor
  double lam_inv_; /// Inverse forgetting facotr
  double delta_;   /// Initial gain value of matrix P
  VectorXd w_;     /// Filter coefficients vector
  MatrixXd P_;
  VectorXd g_; /// Filter gains
  double err_; /// A priori error
  unsigned long long count_;

public:
  /// Recursive least square ctor
  /// \param n - Filter order
  /// \param lam - Forgetting factor
  /// \param delta - Initial gain value of matrix P
  RLSFilter(unsigned int n, double lam, double delta);

  /// Update filter with new data
  /// \param x - Input vector
  /// \param y - Output value
  void update(const VectorXd x, const double y);

  /// Estimate filter output
  /// \param x
  /// \return
  [[nodiscard]] double predict(const VectorXd x) const noexcept {
    return w_.transpose() * x;
  };

  /// Set filter coefficient values
  /// \param w0 - Coefficient values
  void set_estimated_coeffs(const VectorXd &w0);

  ///  Set forgetting factor value
  /// \param lam - Forgetting factor value
  void set_forgetting_factor(const double lam);

  /// Get estimated filter coefficients
  [[nodiscard]] const VectorXd &estimated_coeffs() const noexcept {
    return w_;
  };

  [[nodiscard]] const double a_priori_err() const noexcept { return err_; };
  [[nodiscard]] const VectorXd &gains() const noexcept { return g_; };
  [[nodiscard]] const MatrixXd &P() const noexcept { return P_; };
};

} // namespace rls_filter

#endif // RLS_DYN_MODEL_IDENT_RLS_H