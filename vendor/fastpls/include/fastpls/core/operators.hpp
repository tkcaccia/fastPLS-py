// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_OPERATORS_HPP
#define FASTPLS_CORE_OPERATORS_HPP

#include <fastpls/core/matrix.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace fastpls {
namespace core {

namespace detail {

template<class Backend, class T>
auto prepare_centered_crosscov(
    Backend& backend, ConstMatrixView<T> predictors,
    ConstMatrixView<T> responses, int)
    -> decltype(backend.prepare_centered_crosscov(predictors, responses),
                void()) {
  backend.prepare_centered_crosscov(predictors, responses);
}

template<class Backend, class T>
void prepare_centered_crosscov(
    Backend&, ConstMatrixView<T>, ConstMatrixView<T>, long) {}

template<class Backend, class T>
auto centered_crosscov_transpose(
    Backend& backend, ConstMatrixView<T> predictors,
    ConstMatrixView<T> responses, ConstMatrixView<T> right,
    MatrixView<T> intermediate, MatrixView<T> output, int)
    -> decltype(backend.centered_crosscov_transpose(
                  predictors, responses, right, intermediate, output),
                bool()) {
  return backend.centered_crosscov_transpose(
    predictors, responses, right, intermediate, output
  );
}

template<class Backend, class T>
bool centered_crosscov_transpose(
    Backend&, ConstMatrixView<T>, ConstMatrixView<T>, ConstMatrixView<T>,
    MatrixView<T>, MatrixView<T>, long) {
  return false;
}

}  // namespace detail

template<class T, class Backend>
class ExplicitOperator {
 public:
  ExplicitOperator(ConstMatrixView<T> matrix, Backend& backend)
      : matrix_(matrix), backend_(backend) {
    if (matrix_.data() == nullptr || matrix_.empty()) {
      throw std::invalid_argument(
        "fastPLS explicit operator requires a nonempty matrix"
      );
    }
  }

  std::size_t rows() const noexcept { return matrix_.rows(); }
  std::size_t columns() const noexcept { return matrix_.columns(); }
  std::size_t workspace_rows() const noexcept { return 0; }

  void multiply(ConstMatrixView<T> right, bool transpose,
                Matrix<T>& output) {
    const std::size_t output_rows = transpose ? columns() : rows();
    output.resize(output_rows, right.columns());
    multiply(right, transpose, output.view());
  }

  void multiply(ConstMatrixView<T> right, bool transpose,
                MatrixView<T> output) {
    backend_.gemm(matrix_, right, transpose, false, output);
  }

  void materialize(Matrix<T>& output) const {
    output.resize(rows(), columns());
    for (std::size_t column = 0; column < columns(); ++column) {
      std::copy_n(
        matrix_.data() + column * matrix_.leading_dimension(), rows(),
        output.data() + column * rows()
      );
    }
  }

 private:
  ConstMatrixView<T> matrix_;
  Backend& backend_;
};

// Applies X' (Y - 1 mean(Y)) without storing a centered response matrix or
// materializing the predictor-by-response cross-covariance.
template<class T, class Backend>
class CenteredCrosscovOperator {
 public:
  CenteredCrosscovOperator(ConstMatrixView<T> predictors,
                           ConstMatrixView<T> responses,
                           const T* response_mean,
                           std::size_t response_count,
                           Backend& backend)
      : predictors_(predictors), responses_(responses),
        response_mean_(response_mean),
        backend_(backend) {
    if (predictors_.empty() || responses_.empty() ||
        predictors_.rows() != responses_.rows() ||
        response_mean == nullptr || response_count != responses_.columns()) {
      throw std::invalid_argument(
        "fastPLS centered cross-covariance operator dimensions are invalid"
      );
    }
    detail::prepare_centered_crosscov(
      backend_, predictors_, responses_, 0
    );
  }

  std::size_t rows() const noexcept { return predictors_.columns(); }
  std::size_t columns() const noexcept { return responses_.columns(); }
  std::size_t workspace_rows() const noexcept { return predictors_.rows(); }

  void multiply(ConstMatrixView<T> right, bool transpose,
                Matrix<T>& output) {
    output.resize(transpose ? columns() : rows(), right.columns());
    multiply(right, transpose, output.view());
  }

  void multiply(ConstMatrixView<T> right, bool transpose,
                MatrixView<T> output) {
    const std::size_t expected_rows = transpose ? rows() : columns();
    const std::size_t output_rows = transpose ? columns() : rows();
    if (right.rows() != expected_rows || output.rows() != output_rows ||
        output.columns() != right.columns()) {
      throw std::invalid_argument(
        "fastPLS centered cross-covariance product dimensions are inconsistent"
      );
    }
    intermediate_.resize(predictors_.rows(), right.columns());
    if (transpose) {
      if (!detail::centered_crosscov_transpose(
            backend_, predictors_, responses_, right,
            intermediate_.view(), output, 0
          )) {
        backend_.gemm(
          predictors_, right, false, false, intermediate_.view()
        );
        backend_.gemm(
          responses_, intermediate_.view(), true, false, output
        );
      }
      for (std::size_t column = 0; column < output.columns(); ++column) {
        T sum = T(0);
        for (std::size_t row = 0; row < intermediate_.rows(); ++row) {
          sum += intermediate_(row, column);
        }
        for (std::size_t response = 0; response < output.rows(); ++response) {
          output(response, column) -= response_mean_[response] * sum;
        }
      }
      return;
    }

    backend_.gemm(
      responses_, right, false, false, intermediate_.view()
    );
    for (std::size_t column = 0; column < right.columns(); ++column) {
      T offset = T(0);
      for (std::size_t response = 0; response < right.rows(); ++response) {
        offset += response_mean_[response] * right(response, column);
      }
      for (std::size_t row = 0; row < intermediate_.rows(); ++row) {
        intermediate_(row, column) -= offset;
      }
    }
    backend_.gemm(
      predictors_, intermediate_.view(), true, false, output
    );
  }

  void materialize(Matrix<T>& output) {
    Matrix<T> identity(columns(), columns());
    for (std::size_t index = 0; index < columns(); ++index) {
      identity(index, index) = T(1);
    }
    multiply(identity.view(), false, output);
  }

 private:
  ConstMatrixView<T> predictors_;
  ConstMatrixView<T> responses_;
  const T* response_mean_;
  Backend& backend_;
  Matrix<T> intermediate_;
};

// Applies an orthogonal predictor-side projection to another matrix operator.
// SIMPLS appends one orthonormal deflation direction at a time, so the current
// cross-covariance is represented as (I - V V') S without storing S.
template<class T, class Operator, class Backend>
class ProjectedOperator {
 public:
  ProjectedOperator(Operator& input, std::size_t component_capacity,
                    Backend& backend)
      : input_(input), basis_(input.rows(), component_capacity),
        backend_(backend) {
    if (input.rows() == 0 || input.columns() == 0 ||
        component_capacity == 0) {
      throw std::invalid_argument(
        "fastPLS projected operator dimensions are invalid"
      );
    }
  }

  std::size_t rows() const noexcept { return input_.rows(); }
  std::size_t columns() const noexcept { return input_.columns(); }
  std::size_t workspace_rows() const noexcept {
    return input_.workspace_rows();
  }
  std::size_t deflations() const noexcept { return active_; }

  void multiply(ConstMatrixView<T> right, bool transpose,
                Matrix<T>& output) {
    output.resize(transpose ? columns() : rows(), right.columns());
    multiply(right, transpose, output.view());
  }

  void multiply(ConstMatrixView<T> right, bool transpose,
                MatrixView<T> output) {
    const std::size_t expected_rows = transpose ? rows() : columns();
    const std::size_t output_rows = transpose ? columns() : rows();
    if (right.rows() != expected_rows || output.rows() != output_rows ||
        output.columns() != right.columns()) {
      throw std::invalid_argument(
        "fastPLS projected operator product dimensions are inconsistent"
      );
    }
    if (active_ == 0) {
      input_.multiply(right, transpose, output);
      return;
    }

    const ConstMatrixView<T> active_basis(
      basis_.data(), basis_.rows(), active_, basis_.rows()
    );
    if (transpose) {
      projected_.resize(right.rows(), right.columns());
      for (std::size_t column = 0; column < right.columns(); ++column) {
        std::copy_n(
          right.data() + column * right.leading_dimension(), right.rows(),
          projected_.data() + column * projected_.rows()
        );
      }
      coefficients_.resize(active_, right.columns());
      backend_.gemm(
        active_basis, right, true, false, coefficients_.view()
      );
      correction_.resize(rows(), right.columns());
      backend_.gemm(
        active_basis, coefficients_.view(), false, false, correction_.view()
      );
      for (std::size_t index = 0; index < projected_.size(); ++index) {
        projected_.data()[index] -= correction_.data()[index];
      }
      input_.multiply(projected_.view(), true, output);
      return;
    }

    input_.multiply(right, false, output);
    coefficients_.resize(active_, output.columns());
    backend_.gemm(
      active_basis, output, true, false, coefficients_.view()
    );
    correction_.resize(rows(), output.columns());
    backend_.gemm(
      active_basis, coefficients_.view(), false, false, correction_.view()
    );
    for (std::size_t column = 0; column < output.columns(); ++column) {
      for (std::size_t row = 0; row < output.rows(); ++row) {
        output(row, column) -= correction_(row, column);
      }
    }
  }

  void deflate(ConstMatrixView<T> direction) {
    if (direction.rows() != rows() || direction.columns() != 1 ||
        active_ >= basis_.columns()) {
      throw std::invalid_argument(
        "fastPLS projected operator deflation dimensions are inconsistent"
      );
    }
    for (std::size_t row = 0; row < rows(); ++row) {
      basis_(row, active_) = direction(row, 0);
    }
    ++active_;
  }

  void stabilize(Matrix<T>& direction) {
    if (active_ == 0) return;
    if (direction.rows() != rows() || direction.columns() != 1) {
      throw std::invalid_argument(
        "fastPLS projected-operator direction dimensions are inconsistent"
      );
    }
    const ConstMatrixView<T> active_basis(
      basis_.data(), basis_.rows(), active_, basis_.rows()
    );
    for (int pass = 0; pass < 2; ++pass) {
      coefficients_.resize(active_, 1);
      backend_.gemm(
        active_basis, direction.view(), true, false, coefficients_.view()
      );
      correction_.resize(rows(), 1);
      backend_.gemm(
        active_basis, coefficients_.view(), false, false, correction_.view()
      );
      for (std::size_t row = 0; row < rows(); ++row) {
        direction(row, 0) -= correction_(row, 0);
      }
    }
  }

  void materialize(Matrix<T>& output) {
    Matrix<T> identity(columns(), columns());
    for (std::size_t index = 0; index < columns(); ++index) {
      identity(index, index) = T(1);
    }
    multiply(identity.view(), false, output);
  }

  ConstMatrixView<T> basis() const noexcept {
    return ConstMatrixView<T>(
      basis_.data(), basis_.rows(), active_, basis_.rows()
    );
  }

 private:
  Operator& input_;
  Matrix<T> basis_;
  Backend& backend_;
  Matrix<T> projected_;
  Matrix<T> coefficients_;
  Matrix<T> correction_;
  std::size_t active_ = 0;
};

}  // namespace core
}  // namespace fastpls

#endif
