// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_CLASSIFICATION_HPP
#define FASTPLS_CORE_CLASSIFICATION_HPP

#include <fastpls/core/matrix.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fastpls {
namespace core {

enum class PredictorScaling {
  centering = 1,
  autoscaling = 2,
  none = 3
};

template<class T, class Label>
double dummy_response_r2(
    const Label* labels, std::size_t label_count, const T* response_mean,
    std::size_t class_count, ConstMatrixView<T> predicted_centered) {
  if (labels == nullptr || response_mean == nullptr || label_count == 0 ||
      predicted_centered.rows() != label_count ||
      predicted_centered.columns() != class_count) {
    throw std::invalid_argument(
      "fastPLS dummy-response R2 dimensions are invalid"
    );
  }
  long double mean_square = 0.0L;
  for (std::size_t response = 0; response < class_count; ++response) {
    mean_square += static_cast<long double>(response_mean[response]) *
      static_cast<long double>(response_mean[response]);
  }
  const long double total = static_cast<long double>(label_count) *
    (1.0L - mean_square);
  if (!std::isfinite(static_cast<double>(total)) || total <= 0.0L) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  long double prediction_square = 0.0L;
  long double cross = 0.0L;
  for (std::size_t sample = 0; sample < label_count; ++sample) {
    const std::size_t observed = static_cast<std::size_t>(labels[sample]);
    if (observed >= class_count) {
      throw std::invalid_argument(
        "fastPLS dummy-response R2 contains an invalid class index"
      );
    }
    long double mean_projection = 0.0L;
    for (std::size_t response = 0; response < class_count; ++response) {
      const long double prediction = predicted_centered(sample, response);
      prediction_square += prediction * prediction;
      mean_projection += prediction * response_mean[response];
    }
    cross += predicted_centered(sample, observed) - mean_projection;
  }
  const long double residual = total + prediction_square - 2.0L * cross;
  return static_cast<double>(1.0L - residual / total);
}

template<class T, class Label>
double dummy_response_r2(
    const Label* labels, std::size_t label_count, const T* response_mean,
    std::size_t class_count, MatrixView<T> predicted_centered) {
  return dummy_response_r2(
    labels, label_count, response_mean, class_count,
    ConstMatrixView<T>(predicted_centered)
  );
}

template<class T>
struct LabelCrossprodResult {
  Matrix<T> crossprod;
  Matrix<T> class_predictor_sums;
  std::vector<T> predictor_center;
  std::vector<T> predictor_scale;
  std::vector<T> response_mean;
  std::vector<T> class_counts;
};

template<class T, class Label>
LabelCrossprodResult<T> scaled_label_crossprod_impl(
    ConstMatrixView<T> predictors, T* scaled_predictors,
    std::size_t scaled_leading_dimension, const Label* labels,
    std::size_t label_count, std::size_t class_count,
    PredictorScaling scaling, bool form_crossprod) {
  if (predictors.data() == nullptr || labels == nullptr ||
      predictors.empty() || predictors.rows() != label_count ||
      class_count < 2) {
    throw std::invalid_argument(
      "fastPLS scaled label cross-product dimensions are invalid"
    );
  }
  LabelCrossprodResult<T> result;
  result.crossprod.resize(predictors.columns(), class_count);
  result.class_predictor_sums.resize(predictors.columns(), class_count);
  result.predictor_center.assign(predictors.columns(), T(0));
  result.predictor_scale.assign(predictors.columns(), T(1));
  result.response_mean.assign(class_count, T(0));
  result.class_counts.assign(class_count, T(0));

  for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
    const std::size_t label = static_cast<std::size_t>(labels[sample]);
    if (label >= class_count) {
      throw std::invalid_argument(
        "fastPLS scaled label cross-product contains an invalid class index"
      );
    }
    result.class_counts[label] += T(1);
  }
  for (std::size_t response = 0; response < class_count; ++response) {
    if (result.class_counts[response] == T(0)) {
      throw std::invalid_argument(
        "fastPLS scaled label cross-product contains an empty class"
      );
    }
    result.response_mean[response] = result.class_counts[response] /
      static_cast<T>(predictors.rows());
  }

  std::vector<T> class_sums(class_count);
  for (std::size_t predictor = 0;
       predictor < predictors.columns(); ++predictor) {
    std::fill(class_sums.begin(), class_sums.end(), T(0));
    T total = T(0);
    if (scaling != PredictorScaling::none) {
      T sum = T(0);
      T sum_squares = T(0);
      for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
        const T value = predictors(sample, predictor);
        sum += value;
        if (scaling == PredictorScaling::autoscaling) {
          sum_squares += value * value;
        }
        if (form_crossprod) {
          class_sums[static_cast<std::size_t>(labels[sample])] += value;
        }
      }
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
      if (scaled_predictors != nullptr) {
        const T center = result.predictor_center[predictor];
        const T scale = result.predictor_scale[predictor];
        for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
          scaled_predictors[sample + predictor * scaled_leading_dimension] =
            (predictors(sample, predictor) - center) / scale;
        }
      }
      if (form_crossprod) {
        const T center = result.predictor_center[predictor];
        const T scale = result.predictor_scale[predictor];
        total = (sum - static_cast<T>(predictors.rows()) * center) / scale;
        for (std::size_t response = 0; response < class_count; ++response) {
          class_sums[response] =
            (class_sums[response] - result.class_counts[response] * center) /
            scale;
        }
      }
    } else if (form_crossprod) {
      for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
        const std::size_t label = static_cast<std::size_t>(labels[sample]);
        const T value = predictors(sample, predictor);
        class_sums[label] += value;
        total += value;
      }
    } else {
      continue;
    }
    if (form_crossprod) {
      for (std::size_t response = 0; response < class_count; ++response) {
        result.class_predictor_sums(predictor, response) =
          class_sums[response];
        result.crossprod(predictor, response) = class_sums[response] -
          total * result.response_mean[response];
      }
    }
  }
  return result;
}

template<class T, class Label>
LabelCrossprodResult<T> scaled_label_crossprod(
    ConstMatrixView<T> predictors, const Label* labels,
    std::size_t label_count, std::size_t class_count,
    PredictorScaling scaling) {
  return scaled_label_crossprod_impl(
    predictors, static_cast<T*>(nullptr), 0, labels, label_count,
    class_count, scaling, true
  );
}

template<class T, class Label>
LabelCrossprodResult<T> prepare_scaled_label_crossprod(
    MatrixView<T> predictors, const Label* labels,
    std::size_t label_count, std::size_t class_count,
    PredictorScaling scaling) {
  return scaled_label_crossprod_impl(
    ConstMatrixView<T>(predictors), predictors.data(),
    predictors.leading_dimension(), labels, label_count, class_count, scaling,
    true
  );
}

template<class T, class Label, class Backend>
bool label_crossprod_from_runs(
    ConstMatrixView<T> predictors, const Label* labels,
    std::size_t label_count, std::size_t class_count,
    LabelCrossprodResult<T>& result, Backend& backend) {
  struct LabelRun {
    std::size_t begin;
    std::size_t length;
    std::size_t label;
  };
  std::vector<LabelRun> runs;
  if (label_count > 0) {
    std::size_t begin = 0;
    for (std::size_t row = 1; row <= label_count; ++row) {
      if (row == label_count || labels[row] != labels[begin]) {
        runs.push_back({
          begin, row - begin, static_cast<std::size_t>(labels[begin])
        });
        begin = row;
      }
    }
  }
  if (runs.empty() ||
      runs.size() > std::max<std::size_t>(1024, 4 * class_count)) {
    return false;
  }
  std::fill(
    result.crossprod.data(), result.crossprod.data() + result.crossprod.size(),
    T(0)
  );
  std::size_t maximum_run = 0;
  for (const auto& run : runs) maximum_run = std::max(maximum_run, run.length);
  Matrix<T> ones(maximum_run, 1);
  std::fill(ones.data(), ones.data() + ones.size(), T(1));
  Matrix<T> reduction(predictors.columns(), 1);
  std::vector<unsigned char> initialized(class_count, 0);
  for (const auto& run : runs) {
    if (run.label >= class_count) {
      throw std::invalid_argument(
        "fastPLS scaled label cross-product contains an invalid class index"
      );
    }
    backend.gemm(
      ConstMatrixView<T>(
        predictors.data() + run.begin, run.length, predictors.columns(),
        predictors.leading_dimension()
      ),
      ConstMatrixView<T>(ones.data(), run.length, 1, maximum_run),
      true, false, reduction.view()
    );
    for (std::size_t predictor = 0;
         predictor < predictors.columns(); ++predictor) {
      if (initialized[run.label]) {
        result.crossprod(predictor, run.label) += reduction(predictor, 0);
      } else {
        result.crossprod(predictor, run.label) = reduction(predictor, 0);
      }
    }
    initialized[run.label] = 1;
  }
  for (std::size_t predictor = 0;
       predictor < predictors.columns(); ++predictor) {
    T total = T(0);
    for (std::size_t response = 0; response < class_count; ++response) {
      result.class_predictor_sums(predictor, response) =
        result.crossprod(predictor, response);
      total += result.crossprod(predictor, response);
    }
    for (std::size_t response = 0; response < class_count; ++response) {
      result.crossprod(predictor, response) -=
        total * result.response_mean[response];
    }
  }
  return true;
}

template<class T, class Label, class Backend>
LabelCrossprodResult<T> scaled_label_crossprod(
    ConstMatrixView<T> predictors, const Label* labels,
    std::size_t label_count, std::size_t class_count,
    PredictorScaling scaling, Backend& backend) {
  if (scaling != PredictorScaling::none) {
    return scaled_label_crossprod(
      predictors, labels, label_count, class_count, scaling
    );
  }
  auto result = scaled_label_crossprod_impl(
    predictors, static_cast<T*>(nullptr), 0, labels, label_count,
    class_count, scaling, false
  );
  if (!label_crossprod_from_runs(
      predictors, labels, label_count, class_count, result, backend)) {
    return scaled_label_crossprod(
      predictors, labels, label_count, class_count, scaling
    );
  }
  return result;
}

template<class T, class Label>
void centered_label_crossprod(ConstMatrixView<T> predictors,
                              const Label* labels,
                              std::size_t label_count,
                              const T* center_offset,
                              std::size_t class_count,
                              MatrixView<T> output);

template<class T, class Label, class Backend>
LabelCrossprodResult<T> prepare_scaled_label_crossprod(
    MatrixView<T> predictors, const Label* labels,
    std::size_t label_count, std::size_t class_count,
    PredictorScaling scaling, Backend& backend) {
  if (scaling != PredictorScaling::none) {
    return scaled_label_crossprod_impl(
      ConstMatrixView<T>(predictors), predictors.data(),
      predictors.leading_dimension(), labels, label_count, class_count,
      scaling, true
    );
  }
  auto result = scaled_label_crossprod_impl(
    ConstMatrixView<T>(predictors), predictors.data(),
    predictors.leading_dimension(), labels, label_count, class_count, scaling,
    false
  );
  if (!label_crossprod_from_runs(
      ConstMatrixView<T>(predictors), labels, label_count, class_count,
      result, backend)) {
    return scaled_label_crossprod(
      ConstMatrixView<T>(predictors), labels, label_count, class_count,
      scaling
    );
  }
  return result;
}

template<class T, class Label>
void centered_label_crossprod(ConstMatrixView<T> predictors,
                              const Label* labels,
                              std::size_t label_count,
                              const T* center_offset,
                              std::size_t class_count,
                              MatrixView<T> output) {
  if (predictors.data() == nullptr || labels == nullptr ||
      center_offset == nullptr || output.data() == nullptr) {
    throw std::invalid_argument(
      "fastPLS label cross-product received a null buffer"
    );
  }
  if (predictors.rows() != label_count ||
      output.rows() != predictors.columns() ||
      output.columns() != class_count) {
    throw std::invalid_argument(
      "fastPLS label cross-product dimensions are inconsistent"
    );
  }

  std::vector<T> class_sums(class_count);
  for (std::size_t predictor = 0;
       predictor < predictors.columns(); ++predictor) {
    std::fill(class_sums.begin(), class_sums.end(), T(0));
    T total = T(0);
    for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
      const std::size_t label = static_cast<std::size_t>(labels[sample]);
      if (label >= class_count) {
        throw std::invalid_argument(
          "fastPLS label cross-product contains an invalid class index"
        );
      }
      const T value = predictors.data()[
        sample + predictor * predictors.leading_dimension()
      ];
      total += value;
      class_sums[label] += value;
    }
    for (std::size_t response = 0; response < class_count; ++response) {
      output.data()[predictor + response * output.leading_dimension()] =
        class_sums[response] + total * center_offset[response];
    }
  }
}

template<class T, class Label>
void centered_label_crossprod(MatrixView<T> predictors,
                              const Label* labels,
                              std::size_t label_count,
                              const T* center_offset,
                              std::size_t class_count,
                              MatrixView<T> output) {
  centered_label_crossprod(
    ConstMatrixView<T>(predictors), labels, label_count, center_offset,
    class_count, output
  );
}

template<class T>
std::size_t row_argmax(ConstMatrixView<T> values, std::size_t row) {
  if (values.empty() || row >= values.rows()) {
    throw std::invalid_argument("fastPLS argmax dimensions are invalid");
  }
  std::size_t best = 0;
  T best_value = values.data()[row];
  for (std::size_t column = 1; column < values.columns(); ++column) {
    const T value = values.data()[
      row + column * values.leading_dimension()
    ];
    if (value > best_value) {
      best = column;
      best_value = value;
    }
  }
  return best;
}

template<class T>
std::size_t row_argmax(MatrixView<T> values, std::size_t row) {
  return row_argmax(ConstMatrixView<T>(values), row);
}

template<class T>
void row_top_k(ConstMatrixView<T> values, std::size_t row,
               std::size_t keep, std::vector<std::size_t>& workspace,
               std::size_t* indices, T* scores) {
  if (values.empty() || row >= values.rows() || keep == 0 ||
      keep > values.columns() || indices == nullptr || scores == nullptr) {
    throw std::invalid_argument("fastPLS top-rank dimensions are invalid");
  }
  workspace.resize(values.columns());
  for (std::size_t column = 0; column < values.columns(); ++column) {
    workspace[column] = column;
  }
  const auto before = [&values, row](std::size_t left, std::size_t right) {
    const T lhs = values(row, left);
    const T rhs = values(row, right);
    if (std::isnan(lhs)) return false;
    if (std::isnan(rhs)) return true;
    return lhs == rhs ? left < right : lhs > rhs;
  };
  std::partial_sort(
    workspace.begin(), workspace.begin() + keep, workspace.end(), before
  );
  for (std::size_t rank = 0; rank < keep; ++rank) {
    indices[rank] = workspace[rank];
    scores[rank] = values(row, workspace[rank]);
  }
}

template<class T>
void row_top_k(MatrixView<T> values, std::size_t row,
               std::size_t keep, std::vector<std::size_t>& workspace,
               std::size_t* indices, T* scores) {
  row_top_k(
    ConstMatrixView<T>(values), row, keep, workspace, indices, scores
  );
}

}  // namespace core
}  // namespace fastpls

#endif
