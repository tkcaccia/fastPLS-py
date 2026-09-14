// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_OPERATOR_RSVD_HPP
#define FASTPLS_CORE_OPERATOR_RSVD_HPP

#include <fastpls/core/matrix.hpp>
#include <fastpls/core/rsvd.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace fastpls {
namespace core {

template<class T>
struct OperatorRsvdWorkspace {
  Matrix<T> random;
  Matrix<T> sample;
  Matrix<T> forward_basis;
  Matrix<T> reverse;
  Matrix<T> reverse_basis;
  Matrix<T> projected;
  Matrix<T> gram;
  Matrix<T> reduced_left;
  Matrix<T> reduced_right;
};

namespace detail {

template<class Operator, class T>
auto stabilize_operator_direction(Operator& input, Matrix<T>& direction, int)
    -> decltype(input.stabilize(direction), void()) {
  input.stabilize(direction);
}

template<class Operator, class T>
void stabilize_operator_direction(Operator&, Matrix<T>&, long) {}

template<class T>
bool normalize_column(Matrix<T>& values) {
  if (values.columns() != 1 || values.rows() == 0) return false;
  long double squared_norm = 0.0L;
  for (std::size_t row = 0; row < values.rows(); ++row) {
    const long double value = static_cast<long double>(values(row, 0));
    squared_norm += value * value;
  }
  const T norm = static_cast<T>(std::sqrt(squared_norm));
  if (!std::isfinite(norm) ||
      norm <= std::numeric_limits<T>::epsilon()) {
    return false;
  }
  for (std::size_t row = 0; row < values.rows(); ++row) {
    values(row, 0) /= norm;
  }
  return true;
}

template<class T, class Backend>
SingularTriplets<T> finalize_operator_sample(
    ConstMatrixView<T> basis,
    ConstMatrixView<T> projected,
    std::size_t retained,
    bool left_only,
    Backend& backend,
    OperatorRsvdWorkspace<T>& workspace) {
  Matrix<T> small_left;
  std::vector<T> singular_values;
  Matrix<T> small_vt;
  bool completed = false;

  if (projected.rows() >= 4 && projected.rows() <= projected.columns()) {
    workspace.gram.resize(projected.rows(), projected.rows());
    backend.gemm(
      projected, projected, false, true, workspace.gram.view()
    );
    std::vector<T> eigenvalues;
    if (backend.symmetric_eigen(workspace.gram, eigenvalues)) {
      const T largest = eigenvalues.empty() ? T(0) : eigenvalues.back();
      const T dimension = static_cast<T>(
        std::max(projected.rows(), projected.columns())
      );
      const T relative_tolerance =
        std::numeric_limits<T>::epsilon() * dimension;
      // Eigenvalues of B B' are squared singular values, so the usual
      // singular-value rank tolerance must also be squared. Do not impose a
      // unit-scale floor: cross-covariance operators can be well-conditioned
      // while all singular values are much smaller than one.
      const T tolerance = largest > T(0) ?
        relative_tolerance * relative_tolerance * largest : T(0);
      std::size_t usable = 0;
      for (std::size_t index = eigenvalues.size(); index > 0; --index) {
        if (eigenvalues[index - 1] <= tolerance || usable == retained) break;
        ++usable;
      }
      if (usable == retained) {
        small_left.resize(projected.rows(), usable);
        singular_values.resize(usable);
        for (std::size_t column = 0; column < usable; ++column) {
          const std::size_t source = eigenvalues.size() - 1 - column;
          singular_values[column] = std::sqrt(
            std::max(eigenvalues[source], T(0))
          );
          for (std::size_t row = 0; row < projected.rows(); ++row) {
            small_left(row, column) = workspace.gram(row, source);
          }
        }
        if (!left_only) {
          small_vt.resize(usable, projected.columns());
          backend.gemm(
            small_left.view(), projected, true, false, small_vt.view()
          );
          for (std::size_t row = 0; row < usable; ++row) {
            const T inverse = T(1) / singular_values[row];
            for (std::size_t column = 0;
                 column < small_vt.columns(); ++column) {
              small_vt(row, column) *= inverse;
            }
          }
        }
        completed = true;
      }
    }
  }

  if (!completed && !backend.svd_economy(
        projected, left_only, small_left, singular_values, small_vt)) {
    throw std::runtime_error(
      "fastPLS operator rSVD reduced decomposition failed"
    );
  }
  retained = std::min({
    retained, small_left.columns(), singular_values.size()
  });
  if (retained == 0) {
    throw std::runtime_error(
      "fastPLS operator rSVD returned no usable directions"
    );
  }

  SingularTriplets<T> output;
  Matrix<T> retained_left(small_left.rows(), retained);
  for (std::size_t column = 0; column < retained; ++column) {
    for (std::size_t row = 0; row < small_left.rows(); ++row) {
      retained_left(row, column) = small_left(row, column);
    }
  }
  output.U.resize(basis.rows(), retained);
  backend.gemm(
    basis, retained_left.view(), false, false, output.U.view()
  );
  output.singular_values.assign(
    singular_values.begin(), singular_values.begin() + retained
  );
  if (!left_only) {
    output.Vt.resize(retained, projected.columns());
    for (std::size_t column = 0; column < projected.columns(); ++column) {
      for (std::size_t row = 0; row < retained; ++row) {
        output.Vt(row, column) = small_vt(row, column);
      }
    }
  }
  return output;
}

}  // namespace detail

// Fresh rank-one subspace iteration avoids the oversampled block workspace
// when only the next leading direction is required. This is important for
// SIMPLS operators whose response side is too large to materialize.
template<class T, class Operator, class Backend>
bool randomized_dominant_operator_direction(
    Operator& input,
    const RsvdControls& controls,
    Backend& backend,
    OperatorRsvdWorkspace<T>& workspace,
    Matrix<T>& direction) {
  if (input.rows() == 0 || input.columns() == 0) return false;
  std::mt19937 generator(controls.seed);
  std::normal_distribution<T> normal(T(0), T(1));
  direction.resize(input.rows(), 1);
  for (std::size_t row = 0; row < direction.rows(); ++row) {
    direction(row, 0) = normal(generator);
  }
  detail::stabilize_operator_direction(input, direction, 0);
  if (!detail::normalize_column(direction)) return false;

  const int iterations = std::max(controls.power, 1);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    input.multiply(direction.view(), true, workspace.reverse);
    if (!detail::normalize_column(workspace.reverse)) return false;
    input.multiply(workspace.reverse.view(), false, workspace.sample);
    detail::stabilize_operator_direction(input, workspace.sample, 0);
    if (!detail::normalize_column(workspace.sample)) return false;
    direction = std::move(workspace.sample);
  }
  return true;
}

template<class T, class Operator, class Backend>
SingularTriplets<T> randomized_operator_svd(
    Operator& input,
    int retained,
    const RsvdControls& controls,
    Backend& backend,
    OperatorRsvdWorkspace<T>& workspace) {
  const std::size_t maximum = std::min(input.rows(), input.columns());
  const std::size_t target = std::min(
    maximum, static_cast<std::size_t>(std::max(retained, 1))
  );
  if (target == 0) return {};
  const std::size_t width = std::min(
    maximum,
    target + static_cast<std::size_t>(std::max(controls.oversample, 0))
  );

  std::mt19937 generator(controls.seed);
  std::normal_distribution<T> normal(T(0), T(1));
  workspace.random.resize(input.columns(), width);
  for (std::size_t index = 0; index < workspace.random.size(); ++index) {
    workspace.random.data()[index] = normal(generator);
  }
  input.multiply(workspace.random.view(), false, workspace.sample);

  for (int iteration = 0; iteration < std::max(controls.power, 0);
       ++iteration) {
    if (!backend.qr_economy(
          workspace.sample.view(), workspace.forward_basis)) {
      throw std::runtime_error(
        "fastPLS operator rSVD forward orthogonalization failed"
      );
    }
    input.multiply(
      workspace.forward_basis.view(), true, workspace.reverse
    );
    if (!backend.qr_economy(
          workspace.reverse.view(), workspace.reverse_basis)) {
      throw std::runtime_error(
        "fastPLS operator rSVD reverse orthogonalization failed"
      );
    }
    input.multiply(
      workspace.reverse_basis.view(), false, workspace.sample
    );
  }

  if (!backend.qr_economy(
        workspace.sample.view(), workspace.forward_basis)) {
    throw std::runtime_error(
      "fastPLS operator rSVD sketch orthogonalization failed"
    );
  }
  input.multiply(workspace.forward_basis.view(), true, workspace.reverse);
  workspace.projected.resize(
    workspace.reverse.columns(), workspace.reverse.rows()
  );
  for (std::size_t column = 0;
       column < workspace.projected.columns(); ++column) {
    for (std::size_t row = 0; row < workspace.projected.rows(); ++row) {
      workspace.projected(row, column) = workspace.reverse(column, row);
    }
  }
  return detail::finalize_operator_sample(
    ConstMatrixView<T>(workspace.forward_basis.view()),
    ConstMatrixView<T>(workspace.projected.view()), target,
    controls.left_only, backend, workspace
  );
}

}  // namespace core
}  // namespace fastpls

#endif
