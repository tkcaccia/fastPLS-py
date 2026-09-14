// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_PLSSVD_HPP
#define FASTPLS_CORE_PLSSVD_HPP

#include <fastpls/core/linalg.hpp>
#include <fastpls/core/matrix.hpp>
#include <fastpls/core/operator_rsvd.hpp>
#include <fastpls/core/rsvd.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fastpls {
namespace core {

struct PlssvdControls {
  RsvdControls rsvd;
};

inline bool plssvd_prefer_predictor_gram(
    std::size_t samples, std::size_t predictors, std::size_t components) {
  if (samples == 0 || predictors == 0 || components == 0) return false;
  const long double n = static_cast<long double>(samples);
  const long double p = static_cast<long double>(predictors);
  const long double k = static_cast<long double>(components);
  const long double predictor_gram_storage = p * p;
  const long double score_storage = n * k;
  const long double predictor_gram_work = n * p * p + p * p * k;
  const long double score_gram_work = n * p * k + n * k * k;
  return predictor_gram_storage <= score_storage &&
    predictor_gram_work <= score_gram_work;
}

inline std::size_t plssvd_retained_components(
    const int* components, std::size_t count, std::size_t rank_bound) {
  if (components == nullptr || count == 0 || rank_bound == 0) {
    throw std::invalid_argument(
      "fastPLS PLS-SVD dimensions or component counts are invalid"
    );
  }
  std::size_t retained = 0;
  for (std::size_t index = 0; index < count; ++index) {
    if (components[index] < 1 ||
        static_cast<std::size_t>(components[index]) > rank_bound) {
      throw std::invalid_argument(
        "fastPLS PLS-SVD component count exceeds cross-covariance rank"
      );
    }
    retained = std::max(
      retained, static_cast<std::size_t>(components[index])
    );
  }
  return retained;
}

template<class T>
struct PlssvdModel {
  Matrix<T> weights;
  Matrix<T> response_loadings;
  Matrix<T> scores;
  Matrix<T> score_gram;
  std::vector<Matrix<T>> latent_coefficients;
  std::vector<Matrix<T>> prediction_weights;
  std::vector<T> singular_values;
  std::vector<int> components;
  std::size_t completed_components = 0;
};

template<class T, class Backend>
PlssvdModel<T> assemble_plssvd_model(
    ConstMatrixView<T> predictors, std::size_t response_count,
    const int* components, std::size_t component_count,
    SingularTriplets<T>& decomposition, Backend& backend,
    ConstMatrixView<T> predictor_gram = ConstMatrixView<T>(),
    bool retain_scores = true) {
  const std::size_t predictor_count = predictor_gram.empty() ?
    predictors.columns() : predictor_gram.rows();
  if (predictor_count == 0 || response_count == 0 || components == nullptr ||
      component_count == 0 ||
      (!predictor_gram.empty() &&
       predictor_gram.columns() != predictor_count) ||
      (!predictors.empty() && predictors.columns() != predictor_count)) {
    throw std::invalid_argument(
      "fastPLS PLS-SVD dimensions or component counts are invalid"
    );
  }
  const std::size_t rank_bound = std::min(
    predictor_count, response_count
  );
  const std::size_t retained = plssvd_retained_components(
    components, component_count, rank_bound
  );
  PlssvdModel<T> model;
  model.components.assign(components, components + component_count);

  model.completed_components = std::min({
    decomposition.U.columns(), decomposition.Vt.rows(),
    decomposition.singular_values.size()
  });
  if (model.completed_components < retained) {
    throw std::runtime_error(
      "fastPLS PLS-SVD returned fewer components than requested"
    );
  }

  model.weights.resize(predictor_count, retained);
  model.response_loadings.resize(response_count, retained);
  model.singular_values.assign(
    decomposition.singular_values.begin(),
    decomposition.singular_values.begin() + retained
  );
  for (std::size_t component = 0; component < retained; ++component) {
    for (std::size_t predictor = 0;
         predictor < predictor_count; ++predictor) {
      model.weights(predictor, component) =
        decomposition.U(predictor, component);
    }
    for (std::size_t response = 0;
         response < response_count; ++response) {
      model.response_loadings(response, component) =
        decomposition.Vt(component, response);
    }
  }
  Matrix<T> full_gram(retained, retained);
  if (predictor_gram.empty()) {
    if (predictors.empty()) {
      throw std::invalid_argument(
        "fastPLS PLS-SVD requires predictors or their Gram matrix"
      );
    }
    if (!retain_scores && plssvd_prefer_predictor_gram(
          predictors.rows(), predictors.columns(), retained)) {
      Matrix<T> computed_predictor_gram(
        predictors.columns(), predictors.columns()
      );
      backend.self_gram(
        predictors, true, computed_predictor_gram.view(), true
      );
      Matrix<T> gram_weights(predictors.columns(), retained);
      backend.gemm(
        computed_predictor_gram.view(), model.weights.view(), false, false,
        gram_weights.view()
      );
      backend.gemm(
        model.weights.view(), gram_weights.view(), true, false,
        full_gram.view()
      );
    } else {
      model.scores.resize(predictors.rows(), retained);
      backend.gemm(
        predictors, model.weights.view(), false, false, model.scores.view()
      );
      backend.gemm(
        model.scores.view(), model.scores.view(), true, false,
        full_gram.view()
      );
    }
  } else {
    Matrix<T> gram_weights(predictor_count, retained);
    backend.gemm(
      predictor_gram, model.weights.view(), false, false,
      gram_weights.view()
    );
    backend.gemm(
      model.weights.view(), gram_weights.view(), true, false,
      full_gram.view()
    );
  }
  model.score_gram = full_gram;

  model.latent_coefficients.reserve(component_count);
  model.prediction_weights.reserve(component_count);
  for (const int requested : model.components) {
    const std::size_t count = static_cast<std::size_t>(requested);
    Matrix<T> gram(count, count);
    Matrix<T> diagonal(count, count);
    for (std::size_t column = 0; column < count; ++column) {
      diagonal(column, column) = model.singular_values[column];
      for (std::size_t row = 0; row < count; ++row) {
        gram(row, column) = full_gram(row, column);
      }
    }
    Matrix<T> latent;
    if (!backend.cholesky_solve(
          gram.view(), diagonal.view(), latent) &&
        !backend.general_solve(gram.view(), diagonal.view(), latent)) {
      throw std::runtime_error("fastPLS PLS-SVD latent solve failed");
    }
    Matrix<T> weights(count, response_count);
    ConstMatrixView<T> loadings(
      model.response_loadings.data(), model.response_loadings.rows(), count,
      model.response_loadings.rows()
    );
    backend.gemm(
      latent.view(), loadings, false, true, weights.view()
    );
    model.latent_coefficients.push_back(std::move(latent));
    model.prediction_weights.push_back(std::move(weights));
  }
  return model;
}

template<class T, class Backend>
PlssvdModel<T> fit_plssvd_from_moments(
    ConstMatrixView<T> predictor_gram, ConstMatrixView<T> crosscov,
    const int* components, std::size_t component_count,
    const PlssvdControls& controls, Backend& backend) {
  if (predictor_gram.empty() || crosscov.empty() || components == nullptr ||
      component_count == 0 ||
      predictor_gram.rows() != predictor_gram.columns() ||
      crosscov.rows() != predictor_gram.rows()) {
    throw std::invalid_argument(
      "fastPLS PLS-SVD moment dimensions or component counts are invalid"
    );
  }
  const std::size_t retained = plssvd_retained_components(
    components, component_count,
    std::min(crosscov.rows(), crosscov.columns())
  );
  RsvdControls rsvd = controls.rsvd;
  rsvd.left_only = false;
  auto decomposition = randomized_svd(
    crosscov, static_cast<int>(retained), rsvd, backend
  );
  return assemble_plssvd_model<T>(
    ConstMatrixView<T>(), crosscov.columns(), components, component_count,
    decomposition, backend, predictor_gram, false
  );
}

template<class T, class Backend>
PlssvdModel<T> fit_plssvd_preprocessed(
    ConstMatrixView<T> predictors, ConstMatrixView<T> crosscov,
    const int* components, std::size_t component_count,
    const PlssvdControls& controls, Backend& backend,
    bool retain_scores = true) {
  if (predictors.empty() || crosscov.empty() || components == nullptr ||
      component_count == 0 || crosscov.rows() != predictors.columns()) {
    throw std::invalid_argument(
      "fastPLS PLS-SVD dimensions or component counts are invalid"
    );
  }
  const std::size_t retained = plssvd_retained_components(
    components, component_count,
    std::min(crosscov.rows(), crosscov.columns())
  );
  RsvdControls rsvd = controls.rsvd;
  rsvd.left_only = false;
  auto decomposition = randomized_svd(
    crosscov, static_cast<int>(retained), rsvd, backend
  );
  return assemble_plssvd_model(
    predictors, crosscov.columns(), components, component_count,
    decomposition, backend, ConstMatrixView<T>(), retain_scores
  );
}

template<class T, class Operator, class Backend>
PlssvdModel<T> fit_plssvd_operator(
    ConstMatrixView<T> predictors, Operator& crosscov,
    const int* components, std::size_t component_count,
    const PlssvdControls& controls, Backend& backend,
    OperatorRsvdWorkspace<T>& workspace) {
  if (predictors.empty() || crosscov.rows() != predictors.columns() ||
      crosscov.columns() == 0 || components == nullptr ||
      component_count == 0) {
    throw std::invalid_argument(
      "fastPLS implicit PLS-SVD dimensions or component counts are invalid"
    );
  }
  const std::size_t retained = plssvd_retained_components(
    components, component_count,
    std::min(crosscov.rows(), crosscov.columns())
  );
  RsvdControls rsvd = controls.rsvd;
  rsvd.left_only = false;
  auto decomposition = randomized_operator_svd<T>(
    crosscov, static_cast<int>(retained), rsvd, backend, workspace
  );
  return assemble_plssvd_model(
    predictors, crosscov.columns(), components, component_count,
    decomposition, backend
  );
}

template<class T, class Backend>
PlssvdModel<T> fit_plssvd_preprocessed(
    MatrixView<T> predictors, MatrixView<T> crosscov,
    const int* components, std::size_t component_count,
    const PlssvdControls& controls, Backend& backend,
    bool retain_scores = true) {
  return fit_plssvd_preprocessed(
    ConstMatrixView<T>(predictors), ConstMatrixView<T>(crosscov), components,
    component_count, controls, backend, retain_scores
  );
}

}  // namespace core
}  // namespace fastpls

#endif
