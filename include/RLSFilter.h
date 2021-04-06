//
// Created by Davide Pilastro on 2/19/21.
//

#ifndef RLS_DYN_MODEL_IDENT_RLS_H
#define RLS_DYN_MODEL_IDENT_RLS_H

#include <Eigen/Dense>
#include <type_traits>
using namespace Eigen;

namespace rls_filter {

template <bool B>
using EnableIfB = typename std::enable_if<B, int>::type;

/// Template class implementing a Recursive Least Square (RLS) filter, managing
/// both static and dynamic implementation.
/// \tparam T filter data values type
/// \tparam N filter order(Static) or -1 (Dynamic)
template <typename T, int N>
class RLSFilter {
  static_assert((std::is_same<long double, T>::value ||
                 std::is_same<double, T>::value ||
                 std::is_same<float, T>::value),
                "T must be: long double, double or float");

 public:
  typedef Matrix<T, N, 1> VectorXt;
  typedef Matrix<T, N, N> MatrixXt;

 private:
  unsigned int n_;           /**< Filter order */
  T lam_;                    /**< Forgetting factor */
  T lam_inv_;                /**< Inverse forgetting factor */
  T delta_;                  /**< Initial gain value of matrix P */
  VectorXt w_;               /**< Filter coefficients vector */
  MatrixXt P_;               /**< Inverse covariance error matrix */
  VectorXt g_;               /**< Filter gains */
  T err_;                    /**< A priori error */
  unsigned long long count_; /**< Count of filter updates */

 public:
  /// Recursive least square filter static ctor
  /// \param lam - Forgetting factor
  /// \param delta - Initial gain value of matrix P
  template <int N1 = N, EnableIfB<(N1 > 0)> = 0>
  RLSFilter(T lam, T delta)
      : n_(N),
        lam_(1.0),
        lam_inv_(1.0),
        delta_(delta),
        w_(VectorXt::Zero()),
        P_(MatrixXt::Identity()),
        g_(VectorXt::Zero()),
        err_(0.0),
        count_(0) {
    set_forgetting_factor(lam);
    set_initial_covariance_matrix_gain(delta);
    P_ *= delta_;
  }

  /// Recursive least square filter dynamic ctor
  /// \param n - Filter order
  /// \param lam - Forgetting factor
  /// \param delta - Initial gain value of matrix P
  template <int N1 = N, EnableIfB<(N1 == -1)> = 0>
  RLSFilter(unsigned int n, T lam, T delta)
      : n_(n),
        lam_(1.0),
        lam_inv_(1.0),
        delta_(delta),
        w_(VectorXt::Zero(n_)),
        P_(MatrixXt::Identity(n_, n_)),
        g_(VectorXt::Zero(n)),
        err_(0.0),
        count_(0) {
    set_forgetting_factor(lam);
    set_initial_covariance_matrix_gain(delta);
    P_ *= delta_;
  }

  /// Update filter with new data
  /// \param x - Input vector
  /// \param y - Output value
  void update(const VectorXt &x, T y) {
    err_ = y - predict(x);
    MatrixXt alpha = P_ * lam_inv_;
    g_ = (P_ * x) / (lam_ + x.transpose() * P_ * x);
    P_ = (MatrixXt::Identity(n_, n_) - g_ * x.transpose()) * alpha;
    w_ += g_ * err_;
    count_++;
  };

  /// Estimate filter output
  /// \param x
  /// \return a priori output estimate
  T predict(const VectorXt &x) const noexcept { return w_.transpose() * x; };

  /// Set filter coefficient values
  /// \param w0 - Coefficient values
  void set_estimated_coefficients(const VectorXt &w0) {
    if (w0.rows() == n_) {
      w_ = w0;
    } else {
      throw std::invalid_argument("Wrong initial state dimension.");
    }
  }

  ///  Set forgetting factor value
  /// \param lam - Forgetting factor value
  void set_forgetting_factor(double lam) {
    if ((lam > 0) && (lam <= 1.0)) {
      lam_ = lam;
      lam_inv_ = 1.0 / lam_;
    } else {
      throw std::invalid_argument(
          "Invalid forgetting factor (0 < lambda <= 1).");
    }
  }

  /// Set initial covariance matrix gain
  /// \param delta
  void set_initial_covariance_matrix_gain(double delta) {
    if (delta > 0.0) {
      delta_ = delta;
    } else {
      throw std::invalid_argument(
          "Invalid covariance matrix gain factor (delta > 0).");
    }
  }

  /// Get estimated filter coefficients
  /// \return vector of estimated filter coefficients
  const VectorXt &estimated_coefficients() const noexcept { return w_; };

  /// Get a priori estimate error
  /// \return a priori error
  T a_priori_err() const noexcept { return err_; };

  /// Get filter gains vector
  /// \return filter gains vector
  const VectorXt &gains() const noexcept { return g_; };

  /// Get filter covariance matrix
  /// \return filter covariance matrix
  const MatrixXt &P() const noexcept { return P_; };
};

}  // namespace rls_filter

#endif  // RLS_DYN_MODEL_IDENT_RLS_H