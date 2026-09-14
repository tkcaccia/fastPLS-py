// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_RSVD_HPP
#define FASTPLS_CORE_RSVD_HPP

#include <fastpls/core/matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace fastpls {
namespace core {

struct RsvdControls {
  int oversample = 32;
  int power = 5;
  unsigned int seed = 1;
  bool left_only = false;
};

template<class T>
struct SingularTriplets {
  Matrix<T> U;
  std::vector<T> singular_values;
  Matrix<T> Vt;
};

namespace detail {

template<class T>
Matrix<T> first_columns(ConstMatrixView<T> input, std::size_t columns) {
  columns = std::min(columns, input.columns());
  Matrix<T> output(input.rows(), columns);
  for (std::size_t column = 0; column < columns; ++column) {
    for (std::size_t row = 0; row < input.rows(); ++row) {
      output(row, column) = input(row, column);
    }
  }
  return output;
}

template<class T>
Matrix<T> first_rows(ConstMatrixView<T> input, std::size_t rows) {
  rows = std::min(rows, input.rows());
  Matrix<T> output(rows, input.columns());
  for (std::size_t column = 0; column < input.columns(); ++column) {
    for (std::size_t row = 0; row < rows; ++row) {
      output(row, column) = input(row, column);
    }
  }
  return output;
}

template<class T, class Backend>
SingularTriplets<T> dense_triplets(ConstMatrixView<T> input,
                                   std::size_t retained,
                                   bool left_only,
                                   Backend& backend) {
  SingularTriplets<T> complete;
  if (!backend.svd_economy(input, left_only, complete.U,
                           complete.singular_values, complete.Vt)) {
    throw std::runtime_error("fastPLS rSVD dense decomposition failed");
  }
  const std::size_t available = std::min(
    complete.singular_values.size(), complete.U.columns()
  );
  retained = std::min(retained, available);
  SingularTriplets<T> output;
  output.U = first_columns(ConstMatrixView<T>(complete.U.view()), retained);
  output.singular_values.assign(
    complete.singular_values.begin(),
    complete.singular_values.begin() + retained
  );
  if (!left_only) {
    output.Vt = first_rows(ConstMatrixView<T>(complete.Vt.view()), retained);
  }
  return output;
}

template<class T, class Backend>
SingularTriplets<T> finalize_sample(ConstMatrixView<T> input,
                                    ConstMatrixView<T> sample,
                                    std::size_t retained,
                                    bool left_only,
                                    Backend& backend) {
  Matrix<T> basis;
  if (!backend.qr_economy(sample, basis)) {
    throw std::runtime_error("fastPLS rSVD sketch orthogonalization failed");
  }

  Matrix<T> projected(basis.columns(), input.columns());
  backend.gemm(basis.view(), input, true, false, projected.view());

  if (projected.rows() <= projected.columns()) {
    Matrix<T> gram(projected.rows(), projected.rows());
    backend.gemm(
      projected.view(), projected.view(), false, true, gram.view()
    );
    std::vector<T> eigenvalues;
    if (backend.symmetric_eigen(gram, eigenvalues)) {
      const T largest = eigenvalues.empty() ? T(1) :
        std::max(eigenvalues.back(), T(1));
      const T tolerance = std::numeric_limits<T>::epsilon() *
        static_cast<T>(std::max(projected.rows(), projected.columns())) *
        largest;
      std::size_t usable = 0;
      for (std::size_t index = eigenvalues.size(); index > 0; --index) {
        if (eigenvalues[index - 1] <= tolerance || usable == retained) break;
        ++usable;
      }
      if (usable > 0) {
        Matrix<T> reduced_left(gram.rows(), usable);
        std::vector<T> singular_values(usable);
        for (std::size_t column = 0; column < usable; ++column) {
          const std::size_t source = eigenvalues.size() - 1 - column;
          singular_values[column] = std::sqrt(
            std::max(eigenvalues[source], T(0))
          );
          for (std::size_t row = 0; row < gram.rows(); ++row) {
            reduced_left(row, column) = gram(row, source);
          }
        }

        SingularTriplets<T> output;
        output.U.resize(basis.rows(), usable);
        backend.gemm(
          basis.view(), reduced_left.view(), false, false, output.U.view()
        );
        output.singular_values = std::move(singular_values);
        if (!left_only) {
          output.Vt.resize(usable, projected.columns());
          backend.gemm(
            reduced_left.view(), projected.view(), true, false,
            output.Vt.view()
          );
          for (std::size_t row = 0; row < usable; ++row) {
            const T inverse = T(1) / output.singular_values[row];
            for (std::size_t column = 0;
                 column < output.Vt.columns(); ++column) {
              output.Vt(row, column) *= inverse;
            }
          }
        }
        return output;
      }
    }
  }

  SingularTriplets<T> reduced = dense_triplets(
    ConstMatrixView<T>(projected.view()), retained, left_only, backend
  );
  SingularTriplets<T> output;
  output.U.resize(basis.rows(), reduced.U.columns());
  backend.gemm(
    basis.view(), reduced.U.view(), false, false, output.U.view()
  );
  output.singular_values = std::move(reduced.singular_values);
  output.Vt = std::move(reduced.Vt);
  return output;
}

}  // namespace detail

template<class T, class Backend>
SingularTriplets<T> finalize_rsvd_sample(ConstMatrixView<T> input,
                                         ConstMatrixView<T> sample,
                                         int retained,
                                         bool left_only,
                                         Backend& backend) {
  const std::size_t maximum = std::min(input.rows(), input.columns());
  const std::size_t target = std::min(
    maximum, static_cast<std::size_t>(std::max(retained, 1))
  );
  if (target == 0) return {};
  return detail::finalize_sample(
    input, sample, target, left_only, backend
  );
}

template<class T, class Backend>
SingularTriplets<T> randomized_svd(ConstMatrixView<T> input,
                                   int retained,
                                   const RsvdControls& controls,
                                   Backend& backend) {
  const std::size_t maximum = std::min(input.rows(), input.columns());
  const std::size_t target = std::min(
    maximum, static_cast<std::size_t>(std::max(retained, 1))
  );
  if (target == 0) return {};
  const std::size_t width = std::min(
    maximum,
    target + static_cast<std::size_t>(std::max(controls.oversample, 0))
  );
  if (width >= maximum) {
    return detail::dense_triplets(
      input, target, controls.left_only, backend
    );
  }

  std::mt19937 generator(controls.seed);
  std::normal_distribution<T> normal(T(0), T(1));
  Matrix<T> random(input.columns(), width);
  for (std::size_t index = 0; index < random.size(); ++index) {
    random.data()[index] = normal(generator);
  }

  Matrix<T> sample(input.rows(), width);
  backend.gemm(input, random.view(), false, false, sample.view());
  for (int iteration = 0; iteration < std::max(controls.power, 0);
       ++iteration) {
    Matrix<T> forward_basis;
    if (!backend.qr_economy(sample.view(), forward_basis)) {
      throw std::runtime_error("fastPLS rSVD forward orthogonalization failed");
    }
    Matrix<T> reverse(input.columns(), forward_basis.columns());
    backend.gemm(
      input, forward_basis.view(), true, false, reverse.view()
    );
    Matrix<T> reverse_basis;
    if (!backend.qr_economy(reverse.view(), reverse_basis)) {
      throw std::runtime_error("fastPLS rSVD reverse orthogonalization failed");
    }
    sample.resize(input.rows(), reverse_basis.columns());
    backend.gemm(
      input, reverse_basis.view(), false, false, sample.view()
    );
  }
  return detail::finalize_sample(
    input, ConstMatrixView<T>(sample.view()), target, controls.left_only,
    backend
  );
}

}  // namespace core
}  // namespace fastpls

#endif
