// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_OPLS_HPP
#define FASTPLS_CORE_OPLS_HPP

#include <fastpls/core/classification.hpp>
#include <fastpls/core/matrix.hpp>
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

namespace opls_detail {

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
void apply_opls_filter_inplace(
    MatrixView<T> predictors, const T* center, const T* scale,
    std::size_t statistic_count, ConstMatrixView<T> weights,
    ConstMatrixView<T> loadings, Backend& backend) {
  if (predictors.columns() != statistic_count || center == nullptr ||
      scale == nullptr || weights.rows() != predictors.columns() ||
      loadings.rows() != predictors.columns() ||
      weights.columns() != loadings.columns()) {
    throw std::invalid_argument("fastPLS OPLS prediction dimensions are invalid");
  }
  for (std::size_t column = 0; column < predictors.columns(); ++column) {
    for (std::size_t row = 0; row < predictors.rows(); ++row) {
      predictors(row, column) =
        (predictors(row, column) - center[column]) / scale[column];
    }
  }
  Matrix<T> score(predictors.rows(), 1);
  Matrix<T> correction(predictors.rows(), predictors.columns());
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
    backend.gemm(score.view(), loading, false, true, correction.view());
    const std::size_t predictor_size =
      predictors.rows() * predictors.columns();
    for (std::size_t index = 0; index < predictor_size; ++index) {
      predictors.data()[index] -= correction.data()[index];
    }
  }
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
