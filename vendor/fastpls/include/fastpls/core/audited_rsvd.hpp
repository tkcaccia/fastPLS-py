// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_AUDITED_RSVD_HPP
#define FASTPLS_CORE_AUDITED_RSVD_HPP

#include <fastpls/core/operator_rsvd.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace fastpls {
namespace core {

struct RsvdAudit {
  bool certified = false;
  bool deterministic_fallback = false;
  int attempts = 0;
  int effective_oversample = 0;
  int effective_power = 0;
  unsigned int effective_seed = 0;
  double subspace_error = 0.0;
  double singular_value_error = 0.0;
  double triplet_residual = 0.0;
  double omitted_direction_ratio = 0.0;
};

template<class T>
struct AuditedSingularTriplets {
  SingularTriplets<T> decomposition;
  RsvdAudit audit;
};

namespace detail {

template<class T>
bool finite_triplets(const SingularTriplets<T>& value) {
  if (value.U.columns() == 0 || value.singular_values.empty() ||
      value.Vt.rows() == 0) {
    return false;
  }
  for (std::size_t index = 0; index < value.U.size(); ++index) {
    if (!std::isfinite(value.U.data()[index])) return false;
  }
  for (const T singular : value.singular_values) {
    if (!std::isfinite(singular)) return false;
  }
  for (std::size_t index = 0; index < value.Vt.size(); ++index) {
    if (!std::isfinite(value.Vt.data()[index])) return false;
  }
  return true;
}

template<class T>
T vector_norm(ConstMatrixView<T> value) {
  long double squared = 0.0L;
  for (std::size_t index = 0; index < value.rows(); ++index) {
    const long double current = static_cast<long double>(value(index, 0));
    squared += current * current;
  }
  return static_cast<T>(std::sqrt(squared));
}

template<class T>
bool normalize_vector(Matrix<T>& value) {
  if (value.columns() != 1) return false;
  const T norm = vector_norm(ConstMatrixView<T>(value.view()));
  if (!std::isfinite(norm) || norm <= std::numeric_limits<T>::epsilon()) {
    return false;
  }
  for (std::size_t row = 0; row < value.rows(); ++row) {
    value(row, 0) /= norm;
  }
  return true;
}

template<class T>
void project_away(Matrix<T>& value, ConstMatrixView<T> basis,
                  std::size_t columns) {
  columns = std::min(columns, basis.columns());
  for (std::size_t component = 0; component < columns; ++component) {
    long double coefficient = 0.0L;
    for (std::size_t row = 0; row < value.rows(); ++row) {
      coefficient += static_cast<long double>(basis(row, component)) *
        static_cast<long double>(value(row, 0));
    }
    const T converted = static_cast<T>(coefficient);
    for (std::size_t row = 0; row < value.rows(); ++row) {
      value(row, 0) -= converted * basis(row, component);
    }
  }
}

template<class T, class Operator>
double triplet_residual(Operator& input,
                        const SingularTriplets<T>& value) {
  const std::size_t rank = std::min({
    value.U.columns(), value.Vt.rows(), value.singular_values.size()
  });
  if (rank == 0 || value.Vt.columns() != input.columns() ||
      value.U.rows() != input.rows()) {
    return std::numeric_limits<double>::infinity();
  }
  Matrix<T> right(input.columns(), rank);
  for (std::size_t column = 0; column < rank; ++column) {
    for (std::size_t row = 0; row < input.columns(); ++row) {
      right(row, column) = value.Vt(column, row);
    }
  }
  Matrix<T> forward;
  Matrix<T> reverse;
  input.multiply(right.view(), false, forward);
  input.multiply(value.U.view(), true, reverse);
  const long double scale = std::max(
    std::abs(static_cast<long double>(value.singular_values.front())),
    static_cast<long double>(std::numeric_limits<T>::epsilon())
  );
  long double worst = 0.0L;
  for (std::size_t component = 0; component < rank; ++component) {
    long double forward_squared = 0.0L;
    long double reverse_squared = 0.0L;
    const long double singular =
      static_cast<long double>(value.singular_values[component]);
    for (std::size_t row = 0; row < input.rows(); ++row) {
      const long double residual =
        static_cast<long double>(forward(row, component)) -
        singular * static_cast<long double>(value.U(row, component));
      forward_squared += residual * residual;
    }
    for (std::size_t row = 0; row < input.columns(); ++row) {
      const long double residual =
        static_cast<long double>(reverse(row, component)) -
        singular * static_cast<long double>(right(row, component));
      reverse_squared += residual * residual;
    }
    worst = std::max(worst, std::max(
      std::sqrt(forward_squared), std::sqrt(reverse_squared)
    ) / scale);
  }
  return static_cast<double>(worst);
}

template<class T, class Operator>
double omitted_ratio(Operator& input, const SingularTriplets<T>& value,
                     std::size_t retained, unsigned int seed) {
  retained = std::min(retained, value.U.columns());
  if (retained == 0 || value.singular_values.size() < retained) {
    return std::numeric_limits<double>::infinity();
  }
  const T boundary = std::max(
    std::abs(value.singular_values[retained - 1]),
    std::numeric_limits<T>::epsilon()
  );
  std::mt19937 generator(seed);
  std::normal_distribution<T> normal(T(0), T(1));
  T largest = T(0);
  for (int probe = 0; probe < 3; ++probe) {
    Matrix<T> right(input.columns(), 1);
    for (std::size_t row = 0; row < right.rows(); ++row) {
      right(row, 0) = normal(generator);
    }
    if (!normalize_vector(right)) continue;
    Matrix<T> left;
    for (int iteration = 0; iteration < 2; ++iteration) {
      input.multiply(right.view(), false, left);
      project_away(left, value.U.view(), retained);
      if (!normalize_vector(left)) break;
      input.multiply(left.view(), true, right);
      if (!normalize_vector(right)) break;
    }
    input.multiply(right.view(), false, left);
    project_away(left, value.U.view(), retained);
    largest = std::max(
      largest, vector_norm(ConstMatrixView<T>(left.view()))
    );
  }
  return static_cast<double>(largest / boundary);
}

template<class T, class Operator>
bool audit_triplets(Operator& input, const SingularTriplets<T>& value,
                    std::size_t retained, unsigned int seed,
                    double& residual, double& omitted) {
  if (!finite_triplets(value) || value.U.columns() < retained ||
      value.singular_values.size() < retained ||
      value.Vt.rows() < retained) {
    return false;
  }
  residual = triplet_residual(input, value);
  omitted = omitted_ratio(input, value, retained, seed + 32452843U);
  if (value.singular_values.size() > retained) {
    const T boundary = std::max(
      std::abs(value.singular_values[retained - 1]),
      std::numeric_limits<T>::epsilon()
    );
    omitted = std::max(omitted, static_cast<double>(
      std::abs(value.singular_values[retained] / boundary)
    ));
  }
  return std::isfinite(residual) && residual <= 1e-2 &&
    std::isfinite(omitted) && omitted <= 1.01;
}

template<class T, class Operator>
bool consensus(Operator& input, const SingularTriplets<T>& lhs,
               const SingularTriplets<T>& rhs, double& subspace_error,
               double& singular_value_error, double& residual) {
  if (!finite_triplets(lhs) || !finite_triplets(rhs) ||
      lhs.U.columns() != rhs.U.columns() ||
      lhs.singular_values.size() != rhs.singular_values.size()) {
    return false;
  }
  const std::size_t rank = lhs.U.columns();
  if (rank == 0) return false;
  long double overlap_squared = 0.0L;
  for (std::size_t left = 0; left < rank; ++left) {
    for (std::size_t right = 0; right < rank; ++right) {
      long double inner = 0.0L;
      for (std::size_t row = 0; row < lhs.U.rows(); ++row) {
        inner += static_cast<long double>(lhs.U(row, left)) *
          static_cast<long double>(rhs.U(row, right));
      }
      overlap_squared += inner * inner;
    }
  }
  overlap_squared = std::min(
    static_cast<long double>(rank), overlap_squared
  );
  subspace_error = static_cast<double>(std::sqrt(std::max(
    0.0L, (static_cast<long double>(rank) - overlap_squared) /
      static_cast<long double>(rank)
  )));
  const long double scale = std::max(
    std::abs(static_cast<long double>(rhs.singular_values.front())),
    static_cast<long double>(std::numeric_limits<T>::epsilon())
  );
  long double largest_difference = 0.0L;
  for (std::size_t index = 0; index < lhs.singular_values.size(); ++index) {
    largest_difference = std::max(largest_difference, std::abs(
      static_cast<long double>(lhs.singular_values[index]) -
      static_cast<long double>(rhs.singular_values[index])
    ));
  }
  singular_value_error = static_cast<double>(largest_difference / scale);
  residual = triplet_residual(input, rhs);
  return subspace_error <= 1e-3 && singular_value_error <= 1e-5 &&
    residual <= 1e-6;
}

template<class T>
void trim_triplets(SingularTriplets<T>& value, std::size_t retained,
                   bool left_only) {
  retained = std::min({
    retained, value.U.columns(), value.singular_values.size()
  });
  Matrix<T> left(value.U.rows(), retained);
  for (std::size_t column = 0; column < retained; ++column) {
    for (std::size_t row = 0; row < value.U.rows(); ++row) {
      left(row, column) = value.U(row, column);
    }
  }
  value.U = std::move(left);
  value.singular_values.resize(retained);
  if (left_only) {
    value.Vt = Matrix<T>();
    return;
  }
  Matrix<T> right(retained, value.Vt.columns());
  for (std::size_t column = 0; column < value.Vt.columns(); ++column) {
    for (std::size_t row = 0; row < retained; ++row) {
      right(row, column) = value.Vt(row, column);
    }
  }
  value.Vt = std::move(right);
}

template<class T, class Operator, class Backend>
SingularTriplets<T> exact_operator_svd(Operator& input, std::size_t retained,
                                       Backend& backend) {
  Matrix<T> materialized;
  input.materialize(materialized);
  SingularTriplets<T> output;
  if (!backend.svd_economy(
        materialized.view(), false, output.U, output.singular_values,
        output.Vt)) {
    throw std::runtime_error("fastPLS dense SVD recovery failed");
  }
  trim_triplets(output, retained, false);
  return output;
}

}  // namespace detail

template<class T, class Operator, class Backend>
AuditedSingularTriplets<T> audited_operator_rsvd(
    Operator& input, int retained_components, const RsvdControls& requested,
    Backend& backend) {
  const std::size_t maximum = std::min(input.rows(), input.columns());
  const std::size_t retained = std::min(
    maximum,
    static_cast<std::size_t>(std::max(retained_components, 1))
  );
  if (retained == 0) {
    throw std::invalid_argument("fastPLS rSVD requires a nonempty matrix");
  }

  AuditedSingularTriplets<T> output;
  if (maximum < 6) {
    output.decomposition = detail::exact_operator_svd<T>(
      input, retained, backend
    );
    output.audit.certified = true;
    output.audit.deterministic_fallback = true;
    output.audit.effective_oversample = std::max(requested.oversample, 0);
    output.audit.effective_power = std::max(requested.power, 0);
    output.audit.effective_seed = requested.seed;
    return output;
  }

  const int audit_rank = static_cast<int>(std::min(maximum, retained + 1));
  OperatorRsvdWorkspace<T> workspace;
  RsvdControls controls = requested;
  controls.left_only = false;
  SingularTriplets<T> previous;
  SingularTriplets<T> accepted;
  double residual = std::numeric_limits<double>::infinity();
  double omitted = std::numeric_limits<double>::infinity();
  double subspace = 0.0;
  double singular = 0.0;
  bool certified = false;
  int attempts = 0;

  const int oversamples[] = {
    std::max(requested.oversample, 0),
    std::max(requested.oversample, 32),
    std::max(requested.oversample, 48)
  };
  const int powers[] = {
    std::max(requested.power, 0),
    std::max(requested.power, 3),
    std::max(requested.power, 4)
  };
  const unsigned int seeds[] = {
    requested.seed, requested.seed + 104729U, requested.seed + 209759U
  };

  for (int attempt = 0; attempt < 3; ++attempt) {
    controls.oversample = oversamples[attempt];
    controls.power = powers[attempt];
    controls.seed = seeds[attempt];
    accepted = randomized_operator_svd<T>(
      input, audit_rank, controls, backend, workspace
    );
    ++attempts;
    const bool checked = detail::audit_triplets(
      input, accepted, retained, controls.seed, residual, omitted
    );
    bool agreed = false;
    if (attempt > 0) {
      agreed = detail::consensus(
        input, previous, accepted, subspace, singular, residual
      );
    }
    certified = checked || agreed;
    if (certified) break;
    previous = accepted;
  }

  for (int recovery = 0; !certified && recovery < 4; ++recovery) {
    const int current_width = std::min<int>(
      static_cast<int>(maximum),
      audit_rank + std::max(controls.oversample, 0)
    );
    const int remaining = static_cast<int>(maximum) - current_width;
    const int enlarged_width = std::min<int>(
      static_cast<int>(maximum),
      std::max(current_width + std::min(32, remaining),
               current_width + std::min(current_width, remaining))
    );
    controls.oversample = enlarged_width - audit_rank;
    controls.power = std::max(controls.power, 6) + 2 * recovery;
    controls.seed += 104729U * static_cast<unsigned int>(recovery + 1);
    accepted = randomized_operator_svd<T>(
      input, audit_rank, controls, backend, workspace
    );
    ++attempts;
    certified = detail::audit_triplets(
      input, accepted, retained, controls.seed, residual, omitted
    );
  }

  if (!certified) {
    throw std::runtime_error(
      "rSVD recovery did not meet numerical tolerances within its "
      "workspace/iteration budget; no unchecked result was returned"
    );
  }
  detail::trim_triplets(accepted, retained, requested.left_only);
  output.decomposition = std::move(accepted);
  output.audit.certified = true;
  output.audit.attempts = attempts;
  output.audit.effective_oversample = controls.oversample;
  output.audit.effective_power = controls.power;
  output.audit.effective_seed = controls.seed;
  output.audit.subspace_error = subspace;
  output.audit.singular_value_error = singular;
  output.audit.triplet_residual = residual;
  output.audit.omitted_direction_ratio = omitted;
  return output;
}

}  // namespace core
}  // namespace fastpls

#endif
