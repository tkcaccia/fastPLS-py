// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_KERNELPLS_HPP
#define FASTPLS_CORE_KERNELPLS_HPP

#include <fastpls/core/classification.hpp>
#include <fastpls/core/kernels.hpp>
#include <fastpls/core/simpls.hpp>
#include <fastpls/core/supervised.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fastpls {
namespace core {

struct KernelPlsControls {
  KernelType kernel = KernelType::linear;
  double gamma = 1.0;
  int degree = 2;
  double offset = 1.0;
  PredictorScaling scaling = PredictorScaling::centering;
  SimplsControls simpls;
};

template<class T>
struct KernelPlsModel {
  SimplsModel<T> inner;
  Matrix<T> reference;
  std::vector<T> predictor_center;
  std::vector<T> predictor_scale;
  std::vector<T> kernel_column_means;
  std::vector<T> response_mean;
  T kernel_grand_mean = T(0);
  T gamma = T(1);
  T offset = T(1);
  int degree = 2;
  KernelType kernel = KernelType::linear;
};

namespace kernelpls_detail {

template<class T>
void standardize(MatrixView<T> values,
                 const std::vector<T>& center,
                 const std::vector<T>& scale) {
  if (values.columns() != center.size() || center.size() != scale.size()) {
    throw std::invalid_argument(
      "Kernel PLS prediction columns do not match training"
    );
  }
  for (std::size_t column = 0; column < values.columns(); ++column) {
    if (!std::isfinite(scale[column]) || scale[column] <= T(0)) {
      throw std::invalid_argument("Kernel PLS contains an invalid scale");
    }
    for (std::size_t row = 0; row < values.rows(); ++row) {
      values(row, column) =
        (values(row, column) - center[column]) / scale[column];
    }
  }
}

template<class T>
void add_response_mean(MatrixView<T> prediction,
                       const std::vector<T>& response_mean) {
  if (prediction.columns() != response_mean.size()) {
    throw std::invalid_argument(
      "Kernel PLS response dimensions do not match training"
    );
  }
  for (std::size_t column = 0; column < prediction.columns(); ++column) {
    for (std::size_t row = 0; row < prediction.rows(); ++row) {
      prediction(row, column) += response_mean[column];
    }
  }
}

}  // namespace kernelpls_detail

template<class T, class Backend>
KernelPlsModel<T> fit_kernelpls(
    Matrix<T> predictors,
    ConstMatrixView<T> responses,
    const KernelPlsControls& controls,
    Backend& backend) {
  if (predictors.rows() < 2 || predictors.columns() == 0 ||
      responses.empty() || predictors.rows() != responses.rows() ||
      controls.simpls.components == 0) {
    throw std::invalid_argument(
      "Kernel PLS requires nonempty conformable training inputs"
    );
  }
  if (controls.kernel != KernelType::linear &&
      controls.kernel != KernelType::radial_basis &&
      controls.kernel != KernelType::polynomial) {
    throw std::invalid_argument("Kernel PLS received an unknown kernel type");
  }

  KernelPlsModel<T> model;
  model.kernel = controls.kernel;
  model.gamma = static_cast<T>(controls.gamma);
  model.offset = static_cast<T>(controls.offset);
  model.degree = controls.degree;
  if (model.kernel != KernelType::linear &&
      (!std::isfinite(model.gamma) || model.gamma <= T(0) ||
       !std::isfinite(model.offset) || model.degree < 1)) {
    throw std::invalid_argument("Kernel PLS received invalid kernel controls");
  }

  const auto preprocessing = prepare_scaled_dense_operator(
    predictors.view(), responses, controls.scaling, backend
  );
  model.predictor_center = preprocessing.predictor_center;
  model.predictor_scale = preprocessing.predictor_scale;
  model.response_mean = preprocessing.response_mean;

  Matrix<T> design;
  if (model.kernel == KernelType::linear) {
    design = std::move(predictors);
  } else {
    design = kernel_matrix(
      predictors.view(), predictors.view(), model.kernel, model.gamma,
      model.degree, model.offset, backend
    );
    const auto centered = center_kernel_train(design.view());
    model.kernel_column_means = centered.column_means;
    model.kernel_grand_mean = centered.grand_mean;
    model.reference = std::move(predictors);
  }

  const auto prepared = unscaled_dense_crossprod(
    ConstMatrixView<T>(design.view()), responses, backend
  );
  SimplsWorkspace<T> workspace;
  model.inner = fit_simpls_preprocessed<T>(
    design.view(), prepared.crossprod.view(), controls.simpls,
    backend, workspace
  );
  if (model.inner.completed_components < controls.simpls.components) {
    throw std::runtime_error(
      "Kernel PLS returned fewer components than requested"
    );
  }
  return model;
}

template<class T, class Backend>
KernelPlsModel<T> fit_kernelpls(
    Matrix<T> predictors,
    MatrixView<T> responses,
    const KernelPlsControls& controls,
    Backend& backend) {
  return fit_kernelpls(
    std::move(predictors), ConstMatrixView<T>(responses), controls, backend
  );
}

template<class T, class Backend>
Matrix<T> predict_kernelpls(
    const KernelPlsModel<T>& model,
    Matrix<T> predictors,
    std::size_t components,
    Backend& backend) {
  kernelpls_detail::standardize(
    predictors.view(), model.predictor_center, model.predictor_scale
  );
  Matrix<T> design;
  if (model.kernel == KernelType::linear) {
    design = std::move(predictors);
  } else {
    design = kernel_matrix(
      ConstMatrixView<T>(predictors.view()), model.reference.view(),
      model.kernel, model.gamma, model.degree, model.offset, backend
    );
    center_kernel_test(
      design.view(), model.kernel_column_means.data(),
      model.kernel_column_means.size(), model.kernel_grand_mean
    );
  }
  Matrix<T> prediction = predict_simpls_preprocessed<T>(
    design.view(), model.inner, components, backend
  );
  kernelpls_detail::add_response_mean(
    prediction.view(), model.response_mean
  );
  return prediction;
}

}  // namespace core
}  // namespace fastpls

#endif
