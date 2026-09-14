// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_SUPERVISED_HPP
#define FASTPLS_CORE_SUPERVISED_HPP

#include <fastpls/core/classification.hpp>
#include <fastpls/core/matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fastpls {
namespace core {

template<class T>
struct DensePreprocessingResult {
  Matrix<T> crossprod;
  std::vector<T> predictor_center;
  std::vector<T> predictor_scale;
  std::vector<T> response_mean;
};

template<class T, class Backend>
DensePreprocessingResult<T> scaled_dense_crossprod_impl(
    ConstMatrixView<T> predictors, T* scaled_predictors,
    std::size_t scaled_leading_dimension,
    ConstMatrixView<T> responses, PredictorScaling scaling,
    Backend& backend, bool form_crossprod) {
  if (predictors.empty() || responses.empty() ||
      predictors.rows() != responses.rows()) {
    throw std::invalid_argument(
      "fastPLS dense cross-product dimensions are invalid"
    );
  }
  if (scaling != PredictorScaling::none && scaled_predictors == nullptr) {
    throw std::invalid_argument(
      "fastPLS scaled dense cross-product requires mutable predictors"
    );
  }

  DensePreprocessingResult<T> result;
  if (form_crossprod) {
    result.crossprod.resize(predictors.columns(), responses.columns());
  }
  result.predictor_center.assign(predictors.columns(), T(0));
  result.predictor_scale.assign(predictors.columns(), T(1));
  result.response_mean.assign(responses.columns(), T(0));
  std::vector<T> predictor_sums(predictors.columns(), T(0));

  for (std::size_t response = 0; response < responses.columns(); ++response) {
    T sum = T(0);
    for (std::size_t sample = 0; sample < responses.rows(); ++sample) {
      sum += responses(sample, response);
    }
    result.response_mean[response] =
      sum / static_cast<T>(responses.rows());
  }

  for (std::size_t predictor = 0;
       predictor < predictors.columns(); ++predictor) {
    T sum = T(0);
    T sum_squares = T(0);
    for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
      const T value = predictors(sample, predictor);
      sum += value;
      sum_squares += value * value;
    }
    if (scaling != PredictorScaling::none) {
      result.predictor_center[predictor] =
        sum / static_cast<T>(predictors.rows());
      if (scaling == PredictorScaling::autoscaling) {
        const T centered_sum_squares = std::max(
          T(0), sum_squares - static_cast<T>(predictors.rows()) *
            result.predictor_center[predictor] *
            result.predictor_center[predictor]
        );
        T standard_deviation = std::sqrt(
          centered_sum_squares /
          static_cast<T>(std::max<std::size_t>(predictors.rows() - 1, 1))
        );
        if (!std::isfinite(standard_deviation) || standard_deviation <= T(0)) {
          standard_deviation = T(1);
        }
        result.predictor_scale[predictor] = standard_deviation;
      }
    }
    T scaled_sum = T(0);
    for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
      const T value = (predictors(sample, predictor) -
        result.predictor_center[predictor]) /
        result.predictor_scale[predictor];
      if (scaled_predictors != nullptr) {
        scaled_predictors[sample +
          predictor * scaled_leading_dimension] = value;
      }
      scaled_sum += value;
    }
    predictor_sums[predictor] = scaled_sum;
  }

  const auto scaled = scaled_predictors == nullptr ? predictors :
    make_const_view(
      scaled_predictors, predictors.rows(), predictors.columns(),
      scaled_leading_dimension
    );
  if (form_crossprod) {
    backend.gemm(
      scaled, responses, true, false, result.crossprod.view()
    );
    for (std::size_t response = 0;
         response < responses.columns(); ++response) {
      for (std::size_t predictor = 0;
           predictor < predictors.columns(); ++predictor) {
        result.crossprod(predictor, response) -=
          predictor_sums[predictor] * result.response_mean[response];
      }
    }
  }
  return result;
}

template<class T, class Backend>
DensePreprocessingResult<T> prepare_scaled_dense_crossprod(
    MatrixView<T> predictors, ConstMatrixView<T> responses,
    PredictorScaling scaling, Backend& backend) {
  return scaled_dense_crossprod_impl(
    ConstMatrixView<T>(predictors), predictors.data(),
    predictors.leading_dimension(), responses, scaling, backend, true
  );
}

template<class T, class Backend>
DensePreprocessingResult<T> prepare_scaled_dense_crossprod(
    MatrixView<T> predictors, MatrixView<T> responses,
    PredictorScaling scaling, Backend& backend) {
  return prepare_scaled_dense_crossprod(
    predictors, ConstMatrixView<T>(responses), scaling, backend
  );
}

template<class T, class Backend>
DensePreprocessingResult<T> prepare_scaled_dense_operator(
    MatrixView<T> predictors, ConstMatrixView<T> responses,
    PredictorScaling scaling, Backend& backend) {
  return scaled_dense_crossprod_impl(
    ConstMatrixView<T>(predictors), predictors.data(),
    predictors.leading_dimension(), responses, scaling, backend, false
  );
}

template<class T, class Backend>
DensePreprocessingResult<T> unscaled_dense_crossprod(
    ConstMatrixView<T> predictors, ConstMatrixView<T> responses,
    Backend& backend) {
  return scaled_dense_crossprod_impl(
    predictors, static_cast<T*>(nullptr), 0, responses,
    PredictorScaling::none, backend, true
  );
}

template<class T, class Backend>
DensePreprocessingResult<T> unscaled_dense_crossprod(
    MatrixView<T> predictors, MatrixView<T> responses, Backend& backend) {
  return unscaled_dense_crossprod(
    ConstMatrixView<T>(predictors), ConstMatrixView<T>(responses), backend
  );
}

template<class T>
double dense_response_r2(
    ConstMatrixView<T> observed, const T* response_mean,
    std::size_t response_count, ConstMatrixView<T> predicted_centered) {
  if (observed.empty() || response_mean == nullptr ||
      response_count != observed.columns() ||
      predicted_centered.rows() != observed.rows() ||
      predicted_centered.columns() != observed.columns()) {
    throw std::invalid_argument(
      "fastPLS dense-response R2 dimensions are invalid"
    );
  }
  long double residual = 0.0L;
  long double total = 0.0L;
  for (std::size_t response = 0; response < observed.columns(); ++response) {
    for (std::size_t sample = 0; sample < observed.rows(); ++sample) {
      const long double centered =
        static_cast<long double>(observed(sample, response)) -
        static_cast<long double>(response_mean[response]);
      const long double difference = centered -
        static_cast<long double>(predicted_centered(sample, response));
      total += centered * centered;
      residual += difference * difference;
    }
  }
  if (!std::isfinite(static_cast<double>(total)) || total <= 0.0L) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return static_cast<double>(1.0L - residual / total);
}

template<class T>
double dense_response_r2(
    ConstMatrixView<T> observed, const T* response_mean,
    std::size_t response_count, MatrixView<T> predicted_centered) {
  return dense_response_r2(
    observed, response_mean, response_count,
    ConstMatrixView<T>(predicted_centered)
  );
}

template<class T>
double dense_response_r2(
    MatrixView<T> observed, const T* response_mean,
    std::size_t response_count, MatrixView<T> predicted_centered) {
  return dense_response_r2(
    ConstMatrixView<T>(observed), response_mean, response_count,
    ConstMatrixView<T>(predicted_centered)
  );
}

}  // namespace core
}  // namespace fastpls

#endif
