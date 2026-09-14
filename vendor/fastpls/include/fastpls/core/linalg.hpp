// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_LINALG_HPP
#define FASTPLS_CORE_LINALG_HPP

#include <fastpls/core/matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace fastpls {
namespace core {

template<class T>
void reference_gemm(ConstMatrixView<T> left,
                    ConstMatrixView<T> right,
                    bool transpose_left,
                    bool transpose_right,
                    MatrixView<T> output) {
  const std::size_t rows = transpose_left ? left.columns() : left.rows();
  const std::size_t inner_left = transpose_left ? left.rows() : left.columns();
  const std::size_t inner_right = transpose_right ? right.columns() : right.rows();
  const std::size_t columns = transpose_right ? right.rows() : right.columns();
  if (inner_left != inner_right || output.rows() != rows ||
      output.columns() != columns) {
    throw std::invalid_argument("fastPLS matrix-product dimensions are inconsistent");
  }

  for (std::size_t column = 0; column < columns; ++column) {
    for (std::size_t row = 0; row < rows; ++row) {
      T value = T(0);
      for (std::size_t inner = 0; inner < inner_left; ++inner) {
        const T lhs = transpose_left ? left(inner, row) : left(row, inner);
        const T rhs = transpose_right ? right(column, inner) : right(inner, column);
        value += lhs * rhs;
      }
      output(row, column) = value;
    }
  }
}

template<class T>
void reference_gemm(MatrixView<T> left,
                    MatrixView<T> right,
                    bool transpose_left,
                    bool transpose_right,
                    MatrixView<T> output) {
  reference_gemm(
    ConstMatrixView<T>(left), ConstMatrixView<T>(right), transpose_left,
    transpose_right, output
  );
}

template<class T>
bool cholesky_solve(ConstMatrixView<T> matrix,
                    ConstMatrixView<T> right,
                    Matrix<T>& solution) {
  if (matrix.rows() != matrix.columns() || right.rows() != matrix.rows()) {
    throw std::invalid_argument(
      "fastPLS symmetric-solve dimensions are inconsistent"
    );
  }
  const std::size_t size = matrix.rows();
  Matrix<T> lower(size, size);
  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      T value = matrix(row, column);
      for (std::size_t inner = 0; inner < column; ++inner) {
        value -= lower(row, inner) * lower(column, inner);
      }
      if (row == column) {
        if (!std::isfinite(value) || value <= T(0)) return false;
        lower(row, column) = std::sqrt(value);
      } else {
        lower(row, column) = value / lower(column, column);
      }
    }
  }
  solution.resize(size, right.columns());
  for (std::size_t column = 0; column < right.columns(); ++column) {
    for (std::size_t row = 0; row < size; ++row) {
      T value = right(row, column);
      for (std::size_t inner = 0; inner < row; ++inner) {
        value -= lower(row, inner) * solution(inner, column);
      }
      solution(row, column) = value / lower(row, row);
    }
    for (std::size_t reverse = size; reverse > 0; --reverse) {
      const std::size_t row = reverse - 1;
      T value = solution(row, column);
      for (std::size_t inner = row + 1; inner < size; ++inner) {
        value -= lower(inner, row) * solution(inner, column);
      }
      solution(row, column) = value / lower(row, row);
      if (!std::isfinite(solution(row, column))) return false;
    }
  }
  return true;
}

template<class T>
bool pivoted_solve(ConstMatrixView<T> matrix,
                   ConstMatrixView<T> right,
                   Matrix<T>& solution) {
  if (matrix.rows() != matrix.columns() || right.rows() != matrix.rows()) {
    throw std::invalid_argument(
      "fastPLS linear-solve dimensions are inconsistent"
    );
  }
  const std::size_t size = matrix.rows();
  Matrix<T> factor(size, size);
  solution.resize(size, right.columns());
  T scale = T(0);
  for (std::size_t column = 0; column < size; ++column) {
    for (std::size_t row = 0; row < size; ++row) {
      factor(row, column) = matrix(row, column);
      scale = std::max(scale, std::abs(factor(row, column)));
    }
  }
  for (std::size_t column = 0; column < right.columns(); ++column) {
    for (std::size_t row = 0; row < size; ++row) {
      solution(row, column) = right(row, column);
    }
  }
  const T tolerance = std::numeric_limits<T>::epsilon() *
    static_cast<T>(std::max<std::size_t>(size, 1)) * std::max(scale, T(1));
  for (std::size_t pivot = 0; pivot < size; ++pivot) {
    std::size_t selected = pivot;
    for (std::size_t row = pivot + 1; row < size; ++row) {
      if (std::abs(factor(row, pivot)) >
          std::abs(factor(selected, pivot))) selected = row;
    }
    if (!std::isfinite(factor(selected, pivot)) ||
        std::abs(factor(selected, pivot)) <= tolerance) return false;
    if (selected != pivot) {
      for (std::size_t column = 0; column < size; ++column) {
        std::swap(factor(pivot, column), factor(selected, column));
      }
      for (std::size_t column = 0; column < right.columns(); ++column) {
        std::swap(solution(pivot, column), solution(selected, column));
      }
    }
    for (std::size_t row = pivot + 1; row < size; ++row) {
      const T multiplier = factor(row, pivot) / factor(pivot, pivot);
      factor(row, pivot) = T(0);
      for (std::size_t column = pivot + 1; column < size; ++column) {
        factor(row, column) -= multiplier * factor(pivot, column);
      }
      for (std::size_t column = 0; column < right.columns(); ++column) {
        solution(row, column) -= multiplier * solution(pivot, column);
      }
    }
  }
  for (std::size_t column = 0; column < right.columns(); ++column) {
    for (std::size_t reverse = size; reverse > 0; --reverse) {
      const std::size_t row = reverse - 1;
      T value = solution(row, column);
      for (std::size_t inner = row + 1; inner < size; ++inner) {
        value -= factor(row, inner) * solution(inner, column);
      }
      solution(row, column) = value / factor(row, row);
      if (!std::isfinite(solution(row, column))) return false;
    }
  }
  return true;
}

template<class T>
bool solve_symmetric_system(ConstMatrixView<T> matrix,
                            ConstMatrixView<T> right,
                            Matrix<T>& solution) {
  return cholesky_solve(matrix, right, solution) ||
    pivoted_solve(matrix, right, solution);
}

template<class T>
bool solve_symmetric_system(MatrixView<T> matrix,
                            MatrixView<T> right,
                            Matrix<T>& solution) {
  return solve_symmetric_system(
    ConstMatrixView<T>(matrix), ConstMatrixView<T>(right), solution
  );
}

}  // namespace core
}  // namespace fastpls

#endif
