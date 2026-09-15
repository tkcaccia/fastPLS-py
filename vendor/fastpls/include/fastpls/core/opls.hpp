// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_OPLS_HPP
#define FASTPLS_CORE_OPLS_HPP

#include <fastpls/core/classification.hpp>
#include <fastpls/core/matrix.hpp>
#include <fastpls/core/operator_rsvd.hpp>
#include <fastpls/core/operators.hpp>
#include <fastpls/core/rsvd.hpp>
#include <fastpls/core/supervised.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fastpls {
namespace core {

template<class T>
struct OplsFilter {
  Matrix<T> predictors;
  Matrix<T> weights;
  Matrix<T> loadings;
  std::vector<T> predictor_center;
  std::vector<T> predictor_scale;
  std::size_t completed_components = 0;
};

template<class T>
struct OplsMomentFilter {
  OplsFilter<T> filter;
  Matrix<T> predictor_gram;
  Matrix<T> crosscov;
};

namespace opls_detail {

template<class T>
void copy_matrix(ConstMatrixView<T> source, Matrix<T>& destination) {
  destination.resize(source.rows(), source.columns());
  for (std::size_t column = 0; column < source.columns(); ++column) {
    std::copy_n(
      source.data() + column * source.leading_dimension(), source.rows(),
      destination.data() + column * destination.view().leading_dimension()
    );
  }
}

template<class T, class Backend>
auto rank1_subtract(MatrixView<T> target, ConstMatrixView<T> column,
                    ConstMatrixView<T> row, Backend& backend, int)
    -> decltype(backend.rank1_subtract(target, column, row), bool()) {
  return backend.rank1_subtract(target, column, row);
}

template<class T, class Backend>
bool rank1_subtract(MatrixView<T> target, ConstMatrixView<T> column,
                    ConstMatrixView<T> row, Backend&, long) {
  if (column.columns() != 1 || row.rows() != 1 ||
      column.rows() != target.rows() || row.columns() != target.columns()) {
    throw std::invalid_argument("fastPLS OPLS rank-one dimensions differ");
  }
  for (std::size_t output_column = 0;
       output_column < target.columns(); ++output_column) {
    const T multiplier = row(0, output_column);
    T* output = target.data() +
      output_column * target.leading_dimension();
    for (std::size_t output_row = 0;
         output_row < target.rows(); ++output_row) {
      output[output_row] -= column(output_row, 0) * multiplier;
    }
  }
  return true;
}

template<class T, class Backend>
void subtract_rank1(MatrixView<T> target, ConstMatrixView<T> column,
                    ConstMatrixView<T> row, Backend& backend) {
  if (!rank1_subtract(target, column, row, backend, 0)) {
    rank1_subtract(target, column, row, backend, 0L);
  }
}

template<class T>
T vector_norm(ConstMatrixView<T> vector) {
  T sum = T(0);
  for (std::size_t row = 0; row < vector.rows(); ++row) {
    sum += vector(row, 0) * vector(row, 0);
  }
  return std::sqrt(std::max(sum, T(0)));
}

template<class T>
T vector_dot(ConstMatrixView<T> left, ConstMatrixView<T> right) {
  T sum = T(0);
  for (std::size_t row = 0; row < left.rows(); ++row) {
    sum += left(row, 0) * right(row, 0);
  }
  return sum;
}

template<class T>
void normalize(Matrix<T>& vector) {
  const T norm = vector_norm(ConstMatrixView<T>(vector.view()));
  if (!std::isfinite(norm) || norm <= T(0)) {
    vector.resize(0, 0);
    return;
  }
  for (std::size_t row = 0; row < vector.rows(); ++row) {
    vector(row, 0) /= norm;
  }
}

template<class T, class Backend>
bool leading_left_direction(ConstMatrixView<T> crosscov,
                            Backend& backend, Matrix<T>& direction) {
  if (crosscov.empty()) return false;
  const bool right_gram = crosscov.columns() <= crosscov.rows();
  const std::size_t order = right_gram ?
    crosscov.columns() : crosscov.rows();
  Matrix<T> gram(order, order);
  backend.gemm(
    crosscov, crosscov, right_gram, !right_gram, gram.view()
  );
  std::vector<T> eigenvalues;
  if (backend.symmetric_eigen(gram, eigenvalues) && !eigenvalues.empty()) {
    const T largest = eigenvalues.back();
    if (std::isfinite(largest) && largest > T(0)) {
      if (right_gram) {
        const auto eigenvector = make_const_view(
          gram.data() + (order - 1) * gram.rows(), order, 1, gram.rows()
        );
        direction.resize(crosscov.rows(), 1);
        backend.gemm(
          crosscov, eigenvector, false, false, direction.view()
        );
      } else {
        direction.resize(crosscov.rows(), 1);
        std::copy_n(
          gram.data() + (order - 1) * gram.rows(), order, direction.data()
        );
      }
      normalize(direction);
      if (direction.size() != 0) return true;
    }
  }

  Matrix<T> left;
  Matrix<T> right_transpose;
  std::vector<T> singular_values;
  if (!backend.svd_economy(
        crosscov, true, left, singular_values, right_transpose
      ) || left.columns() == 0) {
    return false;
  }
  direction.resize(left.rows(), 1);
  std::copy_n(left.data(), left.rows(), direction.data());
  normalize(direction);
  return direction.size() != 0;
}

template<class T>
Matrix<T> retained_columns(ConstMatrixView<T> source, std::size_t columns) {
  Matrix<T> result(source.rows(), columns);
  for (std::size_t column = 0; column < columns; ++column) {
    std::copy_n(
      source.data() + column * source.leading_dimension(), source.rows(),
      result.data() + column * result.rows()
    );
  }
  return result;
}

template<class T, class Backend, class RefreshCrosscov, class SolveDirection>
OplsFilter<T> fit_preprocessed_filter_inplace(
    MatrixView<T> predictors, Matrix<T> crosscov, std::size_t components,
    std::vector<T> predictor_center, std::vector<T> predictor_scale,
    Backend& backend, RefreshCrosscov&& refresh_crosscov,
    SolveDirection&& solve_direction) {
  OplsFilter<T> model;
  model.predictor_center = std::move(predictor_center);
  model.predictor_scale = std::move(predictor_scale);
  model.weights.resize(predictors.columns(), components);
  model.loadings.resize(predictors.columns(), components);
  Matrix<T> weight;
  Matrix<T> score;
  Matrix<T> loading;
  Matrix<T> orthogonal_weight;
  Matrix<T> orthogonal_score;
  Matrix<T> orthogonal_loading;

  for (std::size_t component = 0; component < components; ++component) {
    if (component > 0) {
      refresh_crosscov(ConstMatrixView<T>(predictors), crosscov);
    }
    if (!solve_direction(
          ConstMatrixView<T>(crosscov.view()), component, weight
        )) break;

    score.resize(predictors.rows(), 1);
    backend.gemm(
      ConstMatrixView<T>(predictors), weight.view(), false, false, score.view()
    );
    const T score_sum_squares = vector_dot<T>(score.view(), score.view());
    if (!std::isfinite(score_sum_squares) || score_sum_squares <= T(0)) break;
    loading.resize(predictors.columns(), 1);
    backend.gemm(
      ConstMatrixView<T>(predictors), score.view(), true, false, loading.view()
    );
    for (std::size_t row = 0; row < loading.rows(); ++row) {
      loading(row, 0) /= score_sum_squares;
    }

    const T weight_sum_squares = vector_dot<T>(weight.view(), weight.view());
    if (!std::isfinite(weight_sum_squares) || weight_sum_squares <= T(0)) break;
    const T projection = vector_dot<T>(weight.view(), loading.view()) /
      weight_sum_squares;
    orthogonal_weight.resize(predictors.columns(), 1);
    for (std::size_t row = 0; row < orthogonal_weight.rows(); ++row) {
      orthogonal_weight(row, 0) = loading(row, 0) -
        weight(row, 0) * projection;
    }
    normalize(orthogonal_weight);
    if (orthogonal_weight.size() == 0) break;

    orthogonal_score.resize(predictors.rows(), 1);
    backend.gemm(
      ConstMatrixView<T>(predictors), orthogonal_weight.view(), false, false,
      orthogonal_score.view()
    );
    const T orthogonal_sum_squares = vector_dot<T>(
      orthogonal_score.view(), orthogonal_score.view()
    );
    if (!std::isfinite(orthogonal_sum_squares) ||
        orthogonal_sum_squares <= T(0)) break;
    orthogonal_loading.resize(predictors.columns(), 1);
    backend.gemm(
      ConstMatrixView<T>(predictors), orthogonal_score.view(), true, false,
      orthogonal_loading.view()
    );
    for (std::size_t row = 0; row < orthogonal_loading.rows(); ++row) {
      orthogonal_loading(row, 0) /= orthogonal_sum_squares;
      model.weights(row, model.completed_components) =
        orthogonal_weight(row, 0);
      model.loadings(row, model.completed_components) =
        orthogonal_loading(row, 0);
    }

    // Apply the rank-one deflation directly in column-major order. Building
    // an n-by-p correction matrix doubles memory traffic and can add several
    // gigabytes of avoidable workspace for large sample matrices.
    for (std::size_t column = 0; column < predictors.columns(); ++column) {
      const T loading_value = orthogonal_loading(column, 0);
      T* values = predictors.data() + column * predictors.leading_dimension();
      for (std::size_t row = 0; row < predictors.rows(); ++row) {
        values[row] -= orthogonal_score(row, 0) * loading_value;
      }
    }
    ++model.completed_components;
  }

  if (model.completed_components < components) {
    model.weights = retained_columns<T>(
      model.weights.view(), model.completed_components
    );
    model.loadings = retained_columns<T>(
      model.loadings.view(), model.completed_components
    );
  }
  return model;
}

template<class T, class Backend, class RefreshCrosscov>
OplsFilter<T> fit_preprocessed_filter_inplace(
    MatrixView<T> predictors, Matrix<T> crosscov, std::size_t components,
    std::vector<T> predictor_center, std::vector<T> predictor_scale,
    Backend& backend, RefreshCrosscov&& refresh_crosscov) {
  auto solve = [&backend](ConstMatrixView<T> crosscov, std::size_t,
                          Matrix<T>& direction) {
    return leading_left_direction(crosscov, backend, direction);
  };
  return fit_preprocessed_filter_inplace<T>(
    predictors, std::move(crosscov), components,
    std::move(predictor_center), std::move(predictor_scale), backend,
    std::forward<RefreshCrosscov>(refresh_crosscov), solve
  );
}

template<class T, class Backend, class RefreshCrosscov, class SolveDirection>
OplsFilter<T> fit_preprocessed_filter(
    Matrix<T> predictors, Matrix<T> crosscov, std::size_t components,
    std::vector<T> predictor_center, std::vector<T> predictor_scale,
    Backend& backend, RefreshCrosscov&& refresh_crosscov,
    SolveDirection&& solve_direction) {
  OplsFilter<T> model = fit_preprocessed_filter_inplace<T>(
    predictors.view(), std::move(crosscov), components,
    std::move(predictor_center), std::move(predictor_scale), backend,
    std::forward<RefreshCrosscov>(refresh_crosscov),
    std::forward<SolveDirection>(solve_direction)
  );
  model.predictors = std::move(predictors);
  return model;
}

template<class T, class Backend, class RefreshCrosscov>
OplsFilter<T> fit_preprocessed_filter(
    Matrix<T> predictors, Matrix<T> crosscov, std::size_t components,
    std::vector<T> predictor_center, std::vector<T> predictor_scale,
    Backend& backend, RefreshCrosscov&& refresh_crosscov) {
  auto solve = [&backend](ConstMatrixView<T> crosscov, std::size_t,
                          Matrix<T>& direction) {
    return leading_left_direction(crosscov, backend, direction);
  };
  return fit_preprocessed_filter<T>(
    std::move(predictors), std::move(crosscov), components,
    std::move(predictor_center), std::move(predictor_scale), backend,
    std::forward<RefreshCrosscov>(refresh_crosscov), solve
  );
}

}  // namespace opls_detail

template<class T, class Backend>
void apply_opls_filter_to_moments(
    MatrixView<T> values, ConstMatrixView<T> weights,
    ConstMatrixView<T> loadings, Backend& backend) {
  if (values.rows() != weights.rows() ||
      values.rows() != loadings.rows() ||
      weights.columns() != loadings.columns()) {
    throw std::invalid_argument(
      "fastPLS OPLS moment-filter dimensions are invalid"
    );
  }
  Matrix<T> projection(1, values.columns());
  for (std::size_t component = 0;
       component < weights.columns(); ++component) {
    const auto weight = make_const_view(
      weights.data() + component * weights.leading_dimension(),
      weights.rows(), 1, weights.leading_dimension()
    );
    const auto loading = make_const_view(
      loadings.data() + component * loadings.leading_dimension(),
      loadings.rows(), 1, loadings.leading_dimension()
    );
    backend.gemm(weight, ConstMatrixView<T>(values), true, false,
                 projection.view());
    opls_detail::subtract_rank1<T>(
      values, loading, projection.view(), backend
    );
  }
}

template<class T, class Backend, class SolveDirection>
OplsMomentFilter<T> fit_opls_filter_from_moments_impl(
    ConstMatrixView<T> predictor_gram, ConstMatrixView<T> crosscov,
    std::size_t components, std::vector<T> predictor_center,
    std::vector<T> predictor_scale, Backend& backend,
    SolveDirection&& solve_direction) {
  if (predictor_gram.empty() ||
      predictor_gram.rows() != predictor_gram.columns() ||
      crosscov.empty() || crosscov.rows() != predictor_gram.rows() ||
      components == 0 || predictor_center.size() != predictor_gram.rows() ||
      predictor_scale.size() != predictor_gram.rows()) {
    throw std::invalid_argument(
      "fastPLS OPLS moment dimensions are invalid"
    );
  }

  OplsMomentFilter<T> result;
  opls_detail::copy_matrix(predictor_gram, result.predictor_gram);
  opls_detail::copy_matrix(crosscov, result.crosscov);
  result.filter.predictor_center = std::move(predictor_center);
  result.filter.predictor_scale = std::move(predictor_scale);
  result.filter.weights.resize(predictor_gram.rows(), components);
  result.filter.loadings.resize(predictor_gram.rows(), components);

  Matrix<T> weight;
  Matrix<T> gram_weight;
  Matrix<T> loading;
  Matrix<T> orthogonal_weight;
  Matrix<T> gram_orthogonal_weight;
  Matrix<T> orthogonal_loading;
  Matrix<T> crosscov_projection(1, crosscov.columns());
  Matrix<T> gram_projection(1, predictor_gram.rows());

  for (std::size_t component = 0; component < components; ++component) {
    if (!solve_direction(
          ConstMatrixView<T>(result.crosscov.view()), component, weight)) {
      break;
    }
    gram_weight.resize(predictor_gram.rows(), 1);
    backend.gemm(
      result.predictor_gram.view(), weight.view(), false, false,
      gram_weight.view()
    );
    const T score_sum_squares = opls_detail::vector_dot<T>(
      weight.view(), gram_weight.view()
    );
    if (!std::isfinite(score_sum_squares) || score_sum_squares <= T(0)) {
      break;
    }
    loading.resize(predictor_gram.rows(), 1);
    for (std::size_t row = 0; row < loading.rows(); ++row) {
      loading(row, 0) = gram_weight(row, 0) / score_sum_squares;
    }

    const T weight_sum_squares = opls_detail::vector_dot<T>(
      weight.view(), weight.view()
    );
    if (!std::isfinite(weight_sum_squares) ||
        weight_sum_squares <= T(0)) {
      break;
    }
    const T projection = opls_detail::vector_dot<T>(
      weight.view(), loading.view()
    ) / weight_sum_squares;
    orthogonal_weight.resize(predictor_gram.rows(), 1);
    for (std::size_t row = 0; row < orthogonal_weight.rows(); ++row) {
      orthogonal_weight(row, 0) = loading(row, 0) -
        weight(row, 0) * projection;
    }
    opls_detail::normalize(orthogonal_weight);
    if (orthogonal_weight.size() == 0) break;

    gram_orthogonal_weight.resize(predictor_gram.rows(), 1);
    backend.gemm(
      result.predictor_gram.view(), orthogonal_weight.view(), false, false,
      gram_orthogonal_weight.view()
    );
    const T orthogonal_sum_squares = opls_detail::vector_dot<T>(
      orthogonal_weight.view(), gram_orthogonal_weight.view()
    );
    if (!std::isfinite(orthogonal_sum_squares) ||
        orthogonal_sum_squares <= T(0)) {
      break;
    }
    orthogonal_loading.resize(predictor_gram.rows(), 1);
    for (std::size_t row = 0; row < orthogonal_loading.rows(); ++row) {
      orthogonal_loading(row, 0) =
        gram_orthogonal_weight(row, 0) / orthogonal_sum_squares;
      result.filter.weights(row, result.filter.completed_components) =
        orthogonal_weight(row, 0);
      result.filter.loadings(row, result.filter.completed_components) =
        orthogonal_loading(row, 0);
    }

    backend.gemm(
      orthogonal_weight.view(), result.crosscov.view(), true, false,
      crosscov_projection.view()
    );
    opls_detail::subtract_rank1<T>(
      result.crosscov.view(), orthogonal_loading.view(),
      crosscov_projection.view(), backend
    );
    for (std::size_t row = 0;
         row < gram_projection.columns(); ++row) {
      gram_projection(0, row) =
        gram_orthogonal_weight(row, 0) / orthogonal_sum_squares;
    }
    opls_detail::subtract_rank1<T>(
      result.predictor_gram.view(), gram_orthogonal_weight.view(),
      gram_projection.view(), backend
    );
    ++result.filter.completed_components;
  }

  if (result.filter.completed_components < components) {
    result.filter.weights = opls_detail::retained_columns<T>(
      result.filter.weights.view(), result.filter.completed_components
    );
    result.filter.loadings = opls_detail::retained_columns<T>(
      result.filter.loadings.view(), result.filter.completed_components
    );
  }
  return result;
}

template<class T, class Backend>
OplsMomentFilter<T> fit_opls_filter_from_moments(
    ConstMatrixView<T> predictor_gram, ConstMatrixView<T> crosscov,
    std::size_t components, std::vector<T> predictor_center,
    std::vector<T> predictor_scale, Backend& backend) {
  auto solve = [&backend](ConstMatrixView<T> current, std::size_t,
                          Matrix<T>& direction) {
    return opls_detail::leading_left_direction<T>(
      current, backend, direction
    );
  };
  return fit_opls_filter_from_moments_impl<T>(
    predictor_gram, crosscov, components, std::move(predictor_center),
    std::move(predictor_scale), backend, solve
  );
}

template<class T, class Backend>
OplsMomentFilter<T> fit_opls_filter_from_moments_rsvd(
    ConstMatrixView<T> predictor_gram, ConstMatrixView<T> crosscov,
    std::size_t components, std::vector<T> predictor_center,
    std::vector<T> predictor_scale, RsvdControls controls, Backend& backend) {
  auto solve = [controls, &backend](
      ConstMatrixView<T> current, std::size_t component,
      Matrix<T>& direction) mutable {
    RsvdControls component_controls = controls;
    component_controls.seed += static_cast<unsigned int>(component);
    component_controls.left_only = true;
    auto decomposition = randomized_svd(
      current, 1, component_controls, backend
    );
    if (decomposition.U.columns() == 0) return false;
    direction.resize(decomposition.U.rows(), 1);
    std::copy_n(
      decomposition.U.data(), decomposition.U.rows(), direction.data()
    );
    opls_detail::normalize(direction);
    return direction.size() != 0;
  };
  return fit_opls_filter_from_moments_impl<T>(
    predictor_gram, crosscov, components, std::move(predictor_center),
    std::move(predictor_scale), backend, solve
  );
}

template<class T, class Backend>
OplsFilter<T> fit_opls_filter_inplace(MatrixView<T> predictors,
                                      ConstMatrixView<T> responses,
                                      std::size_t components,
                                      PredictorScaling scaling,
                                      Backend& backend) {
  if (predictors.empty() || responses.empty() ||
      predictors.rows() != responses.rows()) {
    throw std::invalid_argument("fastPLS OPLS dimensions are invalid");
  }
  auto prepared = prepare_scaled_dense_crossprod(
    predictors, responses, scaling, backend
  );
  Matrix<T> centered_response(responses.rows(), responses.columns());
  for (std::size_t column = 0; column < responses.columns(); ++column) {
    for (std::size_t row = 0; row < responses.rows(); ++row) {
      centered_response(row, column) = responses(row, column) -
        prepared.response_mean[column];
    }
  }
  auto refresh = [&centered_response, &backend](
      ConstMatrixView<T> current, Matrix<T>& crosscov) {
    backend.gemm(
      current, centered_response.view(), true, false, crosscov.view()
    );
  };
  return opls_detail::fit_preprocessed_filter_inplace<T>(
    predictors, std::move(prepared.crossprod), components,
    std::move(prepared.predictor_center),
    std::move(prepared.predictor_scale), backend, refresh
  );
}

template<class T, class Backend>
OplsFilter<T> fit_opls_filter(Matrix<T> predictors,
                              ConstMatrixView<T> responses,
                              std::size_t components,
                              PredictorScaling scaling,
                              Backend& backend) {
  OplsFilter<T> model = fit_opls_filter_inplace(
    predictors.view(), responses, components, scaling, backend
  );
  model.predictors = std::move(predictors);
  return model;
}

template<class T, class Backend>
OplsFilter<T> fit_opls_filter_rsvd(Matrix<T> predictors,
                                   ConstMatrixView<T> responses,
                                   std::size_t components,
                                   PredictorScaling scaling,
                                   RsvdControls controls,
                                   Backend& backend) {
  if (predictors.size() == 0 || responses.empty() ||
      predictors.rows() != responses.rows()) {
    throw std::invalid_argument("fastPLS OPLS dimensions are invalid");
  }
  auto prepared = prepare_scaled_dense_crossprod(
    predictors.view(), responses, scaling, backend
  );
  Matrix<T> centered_response(responses.rows(), responses.columns());
  for (std::size_t column = 0; column < responses.columns(); ++column) {
    for (std::size_t row = 0; row < responses.rows(); ++row) {
      centered_response(row, column) = responses(row, column) -
        prepared.response_mean[column];
    }
  }
  auto refresh = [&centered_response, &backend](
      ConstMatrixView<T> current, Matrix<T>& crosscov) {
    backend.gemm(
      current, centered_response.view(), true, false, crosscov.view()
    );
  };
  auto solve = [controls, &backend](
      ConstMatrixView<T> crosscov, std::size_t component,
      Matrix<T>& direction) mutable {
    RsvdControls component_controls = controls;
    component_controls.seed += static_cast<unsigned int>(component);
    component_controls.left_only = true;
    auto decomposition = randomized_svd(
      crosscov, 1, component_controls, backend
    );
    if (decomposition.U.columns() == 0) return false;
    direction.resize(decomposition.U.rows(), 1);
    std::copy_n(
      decomposition.U.data(), decomposition.U.rows(), direction.data()
    );
    opls_detail::normalize(direction);
    return direction.size() != 0;
  };
  return opls_detail::fit_preprocessed_filter<T>(
    std::move(predictors), std::move(prepared.crossprod), components,
    std::move(prepared.predictor_center),
    std::move(prepared.predictor_scale), backend, refresh, solve
  );
}

// Fit the OPLS orthogonal filter without materializing X'Y. The centered
// response sample Gram is shared with the subsequent PLS fit during CV.
template<class T, class Backend>
OplsFilter<T> fit_opls_filter_operator(
    Matrix<T> predictors, ConstMatrixView<T> responses,
    const T* response_mean, std::size_t response_count,
    std::size_t components, RsvdControls controls, Backend& backend,
    ConstMatrixView<T> sample_response_gram,
    OperatorRsvdWorkspace<T>& workspace) {
  if (predictors.size() == 0 || responses.empty() ||
      predictors.rows() != responses.rows()) {
    throw std::invalid_argument(
      "fastPLS matrix-free OPLS predictor/response dimensions are invalid"
    );
  }
  if (response_mean == nullptr || response_count != responses.columns()) {
    throw std::invalid_argument(
      "fastPLS matrix-free OPLS response centering is invalid"
    );
  }
  if (sample_response_gram.rows() != predictors.rows() ||
      sample_response_gram.columns() != predictors.rows()) {
    throw std::invalid_argument(
      "fastPLS matrix-free OPLS sample Gram dimensions are invalid"
    );
  }

  OplsFilter<T> result;
  result.predictors = std::move(predictors);
  result.weights.resize(result.predictors.columns(), components);
  result.loadings.resize(result.predictors.columns(), components);
  result.predictor_center.assign(result.predictors.columns(), T(0));
  result.predictor_scale.assign(result.predictors.columns(), T(1));

  Matrix<T> score;
  Matrix<T> loading;
  Matrix<T> orthogonal_weight;
  Matrix<T> orthogonal_score;
  Matrix<T> orthogonal_loading;
  for (std::size_t component = 0; component < components; ++component) {
    CenteredCrosscovOperator<T, Backend> crosscovariance(
      result.predictors.view(), responses, response_mean, response_count,
      backend
    );
    RsvdControls component_controls = controls;
    component_controls.seed += static_cast<unsigned int>(component);
    component_controls.left_only = true;
    auto decomposition = randomized_operator_svd_from_sample_gram<T>(
      crosscovariance, result.predictors.view(), sample_response_gram, 1,
      component_controls, backend, workspace
    );
    if (decomposition.U.columns() == 0) break;
    Matrix<T> weight(result.predictors.columns(), 1);
    std::copy_n(decomposition.U.data(), weight.rows(), weight.data());
    opls_detail::normalize(weight);
    if (weight.size() == 0) break;

    score.resize(result.predictors.rows(), 1);
    backend.gemm(
      result.predictors.view(), weight.view(), false, false, score.view()
    );
    const T score_sum_squares = opls_detail::vector_dot<T>(
      score.view(), score.view()
    );
    if (!std::isfinite(score_sum_squares) || score_sum_squares <= T(0)) {
      break;
    }
    loading.resize(result.predictors.columns(), 1);
    backend.gemm(
      result.predictors.view(), score.view(), true, false, loading.view()
    );
    for (std::size_t row = 0; row < loading.rows(); ++row) {
      loading(row, 0) /= score_sum_squares;
    }

    const T weight_sum_squares = opls_detail::vector_dot<T>(
      weight.view(), weight.view()
    );
    if (!std::isfinite(weight_sum_squares) || weight_sum_squares <= T(0)) {
      break;
    }
    const T projection = opls_detail::vector_dot<T>(
      weight.view(), loading.view()
    ) / weight_sum_squares;
    orthogonal_weight.resize(result.predictors.columns(), 1);
    for (std::size_t row = 0; row < orthogonal_weight.rows(); ++row) {
      orthogonal_weight(row, 0) = loading(row, 0) -
        weight(row, 0) * projection;
    }
    opls_detail::normalize(orthogonal_weight);
    if (orthogonal_weight.size() == 0) break;

    orthogonal_score.resize(result.predictors.rows(), 1);
    backend.gemm(
      result.predictors.view(), orthogonal_weight.view(), false, false,
      orthogonal_score.view()
    );
    const T orthogonal_sum_squares = opls_detail::vector_dot<T>(
      orthogonal_score.view(), orthogonal_score.view()
    );
    if (!std::isfinite(orthogonal_sum_squares) ||
        orthogonal_sum_squares <= T(0)) {
      break;
    }
    orthogonal_loading.resize(result.predictors.columns(), 1);
    backend.gemm(
      result.predictors.view(), orthogonal_score.view(), true, false,
      orthogonal_loading.view()
    );
    for (std::size_t row = 0; row < orthogonal_loading.rows(); ++row) {
      orthogonal_loading(row, 0) /= orthogonal_sum_squares;
      result.weights(row, result.completed_components) =
        orthogonal_weight(row, 0);
      result.loadings(row, result.completed_components) =
        orthogonal_loading(row, 0);
    }
    const auto orthogonal_loading_row = make_const_view(
      orthogonal_loading.data(), 1, orthogonal_loading.rows(), 1
    );
    opls_detail::subtract_rank1<T>(
      result.predictors.view(), orthogonal_score.view(),
      orthogonal_loading_row, backend
    );
    ++result.completed_components;
  }

  if (result.completed_components < components) {
    result.weights = opls_detail::retained_columns<T>(
      result.weights.view(), result.completed_components
    );
    result.loadings = opls_detail::retained_columns<T>(
      result.loadings.view(), result.completed_components
    );
  }
  return result;
}

template<class T, class Backend>
OplsFilter<T> fit_opls_filter(Matrix<T> predictors,
                              MatrixView<T> responses,
                              std::size_t components,
                              PredictorScaling scaling,
                              Backend& backend) {
  return fit_opls_filter(
    std::move(predictors), ConstMatrixView<T>(responses), components, scaling,
    backend
  );
}

template<class T, class Label, class Backend>
OplsFilter<T> fit_opls_filter_labels(
    Matrix<T> predictors, const Label* labels, std::size_t label_count,
    std::size_t class_count, std::size_t components,
    PredictorScaling scaling, Backend& backend) {
  if (predictors.size() == 0 || labels == nullptr ||
      predictors.rows() != label_count || class_count < 2) {
    throw std::invalid_argument("fastPLS label-aware OPLS dimensions are invalid");
  }
  auto prepared = prepare_scaled_label_crossprod(
    predictors.view(), labels, label_count, class_count, scaling, backend
  );
  std::vector<T> center_offset(prepared.response_mean.size());
  for (std::size_t index = 0; index < center_offset.size(); ++index) {
    center_offset[index] = -prepared.response_mean[index];
  }
  auto refresh = [labels, label_count, class_count, &center_offset](
      ConstMatrixView<T> current, Matrix<T>& crosscov) {
    centered_label_crossprod(
      current, labels, label_count, center_offset.data(), class_count,
      crosscov.view()
    );
  };
  return opls_detail::fit_preprocessed_filter<T>(
    std::move(predictors), std::move(prepared.crossprod), components,
    std::move(prepared.predictor_center),
    std::move(prepared.predictor_scale), backend, refresh
  );
}

template<class T, class Label, class Backend>
OplsFilter<T> fit_opls_filter_labels_rsvd(
    Matrix<T> predictors, const Label* labels, std::size_t label_count,
    std::size_t class_count, std::size_t components,
    PredictorScaling scaling, RsvdControls controls, Backend& backend) {
  if (predictors.size() == 0 || labels == nullptr ||
      predictors.rows() != label_count || class_count < 2) {
    throw std::invalid_argument(
      "fastPLS label-aware OPLS dimensions are invalid"
    );
  }
  auto prepared = prepare_scaled_label_crossprod(
    predictors.view(), labels, label_count, class_count, scaling, backend
  );
  std::vector<T> center_offset(prepared.response_mean.size());
  for (std::size_t index = 0; index < center_offset.size(); ++index) {
    center_offset[index] = -prepared.response_mean[index];
  }
  auto refresh = [labels, label_count, class_count, &center_offset](
      ConstMatrixView<T> current, Matrix<T>& crosscov) {
    centered_label_crossprod(
      current, labels, label_count, center_offset.data(), class_count,
      crosscov.view()
    );
  };
  auto solve = [controls, &backend](
      ConstMatrixView<T> crosscov, std::size_t component,
      Matrix<T>& direction) mutable {
    RsvdControls component_controls = controls;
    component_controls.seed += static_cast<unsigned int>(component);
    component_controls.left_only = true;
    auto decomposition = randomized_svd(
      crosscov, 1, component_controls, backend
    );
    if (decomposition.U.columns() == 0) return false;
    direction.resize(decomposition.U.rows(), 1);
    std::copy_n(
      decomposition.U.data(), decomposition.U.rows(), direction.data()
    );
    opls_detail::normalize(direction);
    return direction.size() != 0;
  };
  return opls_detail::fit_preprocessed_filter<T>(
    std::move(predictors), std::move(prepared.crossprod), components,
    std::move(prepared.predictor_center),
    std::move(prepared.predictor_scale), backend, refresh, solve
  );
}

template<class T, class Backend>
void apply_opls_deflation_inplace(
    MatrixView<T> predictors, ConstMatrixView<T> weights,
    ConstMatrixView<T> loadings, Backend& backend) {
  if (weights.rows() != predictors.columns() ||
      loadings.rows() != predictors.columns() ||
      weights.columns() != loadings.columns()) {
    throw std::invalid_argument(
      "fastPLS OPLS deflation dimensions are invalid"
    );
  }
  Matrix<T> score(predictors.rows(), 1);
  for (std::size_t component = 0; component < weights.columns(); ++component) {
    const auto weight = make_const_view(
      weights.data() + component * weights.leading_dimension(),
      weights.rows(), 1, weights.leading_dimension()
    );
    const auto loading = make_const_view(
      loadings.data() + component * loadings.leading_dimension(),
      loadings.rows(), 1, loadings.leading_dimension()
    );
    backend.gemm(
      ConstMatrixView<T>(predictors), weight, false, false, score.view()
    );
    const auto loading_row = make_const_view(
      loading.data(), 1, loading.rows(), 1
    );
    opls_detail::subtract_rank1<T>(
      predictors, score.view(), loading_row, backend
    );
  }
}

template<class T, class Backend>
void apply_opls_filter_inplace(
    MatrixView<T> predictors, const T* center, const T* scale,
    std::size_t statistic_count, ConstMatrixView<T> weights,
    ConstMatrixView<T> loadings, Backend& backend) {
  if (predictors.columns() != statistic_count || center == nullptr ||
      scale == nullptr) {
    throw std::invalid_argument("fastPLS OPLS prediction dimensions are invalid");
  }
  for (std::size_t column = 0; column < predictors.columns(); ++column) {
    for (std::size_t row = 0; row < predictors.rows(); ++row) {
      predictors(row, column) =
        (predictors(row, column) - center[column]) / scale[column];
    }
  }
  apply_opls_deflation_inplace(
    predictors, weights, loadings, backend
  );
}

template<class T, class Backend>
Matrix<T> apply_opls_filter(
    Matrix<T> predictors, const T* center, const T* scale,
    std::size_t statistic_count, ConstMatrixView<T> weights,
    ConstMatrixView<T> loadings, Backend& backend) {
  apply_opls_filter_inplace(
    predictors.view(), center, scale, statistic_count, weights, loadings,
    backend
  );
  return predictors;
}

}  // namespace core
}  // namespace fastpls

#endif
