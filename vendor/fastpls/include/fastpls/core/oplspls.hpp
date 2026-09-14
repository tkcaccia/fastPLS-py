// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_OPLSPLS_HPP
#define FASTPLS_CORE_OPLSPLS_HPP

#include <fastpls/core/opls.hpp>
#include <fastpls/core/simpls.hpp>
#include <fastpls/core/supervised.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fastpls {
namespace core {

struct OplsControls {
  std::size_t orthogonal_components = 1;
  PredictorScaling scaling = PredictorScaling::centering;
  bool randomized_filter = true;
  RsvdControls filter_rsvd;
  SimplsControls simpls;
};

template<class T>
struct OplsModel {
  OplsFilter<T> filter;
  SimplsModel<T> inner;
  std::vector<T> response_mean;
};

template<class T, class Backend>
OplsModel<T> fit_opls(
    Matrix<T> predictors,
    ConstMatrixView<T> responses,
    const OplsControls& controls,
    Backend& backend) {
  if (predictors.rows() < 2 || predictors.columns() == 0 ||
      responses.empty() || predictors.rows() != responses.rows() ||
      controls.simpls.components == 0 ||
      controls.orthogonal_components >=
        std::min(predictors.rows(), predictors.columns())) {
    throw std::invalid_argument(
      "OPLS requires conformable inputs and valid component counts"
    );
  }

  OplsModel<T> model;
  if (controls.randomized_filter) {
    model.filter = fit_opls_filter_rsvd(
      std::move(predictors), responses, controls.orthogonal_components,
      controls.scaling, controls.filter_rsvd, backend
    );
  } else {
    model.filter = fit_opls_filter(
      std::move(predictors), responses, controls.orthogonal_components,
      controls.scaling, backend
    );
  }
  if (model.filter.completed_components < controls.orthogonal_components) {
    throw std::runtime_error(
      "OPLS returned fewer orthogonal components than requested"
    );
  }

  const auto prepared = unscaled_dense_crossprod(
    ConstMatrixView<T>(model.filter.predictors.view()), responses, backend
  );
  model.response_mean = prepared.response_mean;
  SimplsWorkspace<T> workspace;
  model.inner = fit_simpls_preprocessed<T>(
    model.filter.predictors.view(), prepared.crossprod.view(),
    controls.simpls, backend, workspace
  );
  if (model.inner.completed_components < controls.simpls.components) {
    throw std::runtime_error(
      "OPLS predictive fit returned fewer components than requested"
    );
  }
  model.filter.predictors = Matrix<T>();
  return model;
}

template<class T, class Backend>
OplsModel<T> fit_opls(
    Matrix<T> predictors,
    MatrixView<T> responses,
    const OplsControls& controls,
    Backend& backend) {
  return fit_opls(
    std::move(predictors), ConstMatrixView<T>(responses), controls, backend
  );
}

template<class T, class Backend>
Matrix<T> predict_opls(
    const OplsModel<T>& model,
    Matrix<T> predictors,
    std::size_t components,
    Backend& backend) {
  Matrix<T> filtered = apply_opls_filter(
    std::move(predictors), model.filter.predictor_center.data(),
    model.filter.predictor_scale.data(),
    model.filter.predictor_center.size(), model.filter.weights.view(),
    model.filter.loadings.view(), backend
  );
  Matrix<T> prediction = predict_simpls_preprocessed<T>(
    filtered.view(), model.inner, components, backend
  );
  if (prediction.columns() != model.response_mean.size()) {
    throw std::runtime_error("OPLS response dimensions changed after fitting");
  }
  for (std::size_t column = 0; column < prediction.columns(); ++column) {
    for (std::size_t row = 0; row < prediction.rows(); ++row) {
      prediction(row, column) += model.response_mean[column];
    }
  }
  return prediction;
}

}  // namespace core
}  // namespace fastpls

#endif
