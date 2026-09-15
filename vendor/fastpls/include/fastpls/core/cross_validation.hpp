// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_CROSS_VALIDATION_HPP
#define FASTPLS_CORE_CROSS_VALIDATION_HPP

#include <fastpls/core/classification.hpp>
#include <fastpls/core/kernels.hpp>
#include <fastpls/core/lda.hpp>
#include <fastpls/core/operators.hpp>
#include <fastpls/core/opls.hpp>
#include <fastpls/core/plssvd.hpp>
#include <fastpls/core/simpls.hpp>
#include <fastpls/core/statistics.hpp>
#include <fastpls/core/supervised.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fastpls {
namespace core {

enum class LinearPlsFamily {
  plssvd = 1,
  simpls = 3,
  opls = 4,
  kernelpls = 5
};

struct KernelCvControls {
  KernelType kernel = KernelType::radial_basis;
  double gamma = 1.0;
  int degree = 3;
  double offset = 1.0;
};

enum class ClassificationHead {
  argmax = 0,
  lda = 1
};

enum class RegressionMetric {
  r2 = 2,
  q2 = 3,
  rmsd = 4
};

template<class T>
struct ClassificationCvResult {
  std::vector<int> folds;
  std::vector<int> status;
  std::vector<double> metrics;
  std::vector<double> q2;
  Matrix<double> fold_training_r2;
  Matrix<int> predictions;
  std::vector<Matrix<T>> scores;
  std::size_t best_index = 0;
  int best_component = 0;
};

template<class T>
struct RegressionCvResult {
  std::vector<int> folds;
  std::vector<int> status;
  std::vector<double> metrics;
  std::vector<double> q2;
  std::vector<double> rmsd;
  std::vector<double> observed_r2;
  Matrix<double> fold_training_r2;
  std::vector<RegressionMetrics> evaluation;
  std::vector<Matrix<T>> predictions;
  std::size_t best_index = 0;
  int best_component = 0;
};

namespace cv_detail {

struct FoldPartition {
  std::vector<std::size_t> train;
  std::vector<std::size_t> test;
  std::size_t training_size = 0;
};

inline std::size_t best_metric_index(
    const std::vector<double>& values, bool minimize) {
  std::size_t selected = 0;
  bool found = false;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) continue;
    if (!found || (minimize ? values[index] < values[selected] :
                              values[index] > values[selected])) {
      selected = index;
      found = true;
    }
  }
  return selected;
}

template<class Backend>
auto configure_backend_problem(Backend& backend, std::size_t rows,
                               std::size_t predictors,
                               std::size_t responses, int)
    -> decltype(
      backend.configure_problem(rows, predictors, responses), void()
    ) {
  backend.configure_problem(rows, predictors, responses);
}

template<class Backend>
void configure_backend_problem(Backend&, std::size_t, std::size_t,
                               std::size_t, long) {}

template<class T>
void standardize(MatrixView<T> values, const std::vector<T>& center,
                 const std::vector<T>& scale);

template<class T>
struct LabelSufficientStatistics {
#if defined(__APPLE__)
  using Accumulator = long double;
#else
  using Accumulator = double;
#endif
  std::vector<Accumulator> predictor_sum;
  std::vector<Accumulator> predictor_sum_squares;
  std::vector<Accumulator> class_count;
  std::vector<Accumulator> class_predictor_sum;
  std::size_t predictors = 0;
  std::size_t classes = 0;
};

template<class T>
struct DenseSufficientStatistics {
  std::vector<double> predictor_sum;
  std::vector<double> predictor_sum_squares;
  std::vector<double> response_sum;
  Matrix<T> predictor_response;
};

inline bool fold_label_statistics_enabled() {
  const char* value = std::getenv("FASTPLS_CV_CLASS_SUM_CACHE");
  return value == nullptr || value[0] != '0' || value[1] != '\0';
}

template<class T>
bool fold_dense_statistics_enabled(std::size_t predictors,
                                   std::size_t responses) {
  const char* value = std::getenv("FASTPLS_CV_FOLD_CROSSCOV_CACHE");
  if (value != nullptr && value[0] == '0' && value[1] == '\0') {
    return false;
  }
  constexpr std::size_t maximum_bytes = 256ULL * 1024ULL * 1024ULL;
  if (predictors == 0 || responses == 0 ||
      predictors > std::numeric_limits<std::size_t>::max() / responses) {
    return false;
  }
  const std::size_t elements = predictors * responses;
  return elements <= maximum_bytes / sizeof(T);
}

template<class T>
bool fold_predictor_gram_enabled(std::size_t predictors) {
  const char* value = std::getenv("FASTPLS_CV_FOLD_GRAM_CACHE");
  if (value != nullptr && value[0] == '0' && value[1] == '\0') {
    return false;
  }
  constexpr std::size_t maximum_bytes = 256ULL * 1024ULL * 1024ULL;
  if (predictors == 0 ||
      predictors > std::numeric_limits<std::size_t>::max() / predictors) {
    return false;
  }
  return predictors * predictors <= maximum_bytes / sizeof(T);
}

template<class T>
bool fold_sample_response_gram_storage_enabled(
    std::size_t samples, std::size_t responses) {
  const char* value = std::getenv("FASTPLS_CV_SAMPLE_RESPONSE_GRAM");
  if (value != nullptr && value[0] == '0' && value[1] == '\0') {
    return false;
  }
  constexpr std::size_t maximum_bytes = 256ULL * 1024ULL * 1024ULL;
  return samples >= 2 && responses > samples &&
    samples <= std::numeric_limits<std::size_t>::max() / samples &&
    samples * samples <= maximum_bytes / sizeof(T);
}

template<class T>
bool fold_sample_response_gram_enabled(
    std::size_t samples, std::size_t responses, std::size_t components,
    int power, std::size_t folds) {
  if (!fold_sample_response_gram_storage_enabled<T>(samples, responses) ||
      components == 0 || folds < 2 ||
      samples > std::numeric_limits<std::size_t>::max() / samples ||
      responses > std::numeric_limits<std::size_t>::max() / samples) {
    return false;
  }

  const long double n = static_cast<long double>(samples);
  const long double q = static_cast<long double>(responses);
  const long double k = static_cast<long double>(folds);
  const long double a = static_cast<long double>(components);
  const long double iterations = static_cast<long double>(std::max(power, 1));
  const long double training_rows = n * (k - 1.0L) / k;
  const long double repeated_operator =
    2.0L * k * a * iterations * training_rows * q;
  const long double cached_sample_gram =
    n * n * q + k * a * iterations * training_rows * training_rows;
  return cached_sample_gram < 0.60L * repeated_operator;
}

inline bool fold_simpls_moments_enabled(
    std::size_t samples, std::size_t predictors, std::size_t components) {
#if defined(FASTPLS_USE_OPENBLAS)
  // For a small retained path, direct fold projections can beat a predictor
  // Gram with single-threaded OpenBLAS. Once the retained path approaches the
  // predictor dimension, the Gram route avoids substantially more repeated
  // fold work and is also the bounded-memory route for very large n.
  const char* raw_threads = std::getenv("OPENBLAS_NUM_THREADS");
  const long openblas_threads = raw_threads == nullptr ? 1L :
    std::strtol(raw_threads, nullptr, 10);
  if (openblas_threads <= 1L && components * 5 < predictors) return false;
#endif
  return components >= 20 && predictors <= 2048 && predictors <= samples &&
    samples >= predictors * 8;
}

template<class Backend>
auto accelerator_backend_code(const Backend& backend, int)
    -> decltype(backend.accelerator_backend_code()) {
  return backend.accelerator_backend_code();
}

template<class Backend>
int accelerator_backend_code(const Backend&, long) {
  return 0;
}

template<class Backend>
inline bool fold_opls_moments_enabled(
    std::size_t samples, std::size_t predictors, std::size_t folds,
    bool classification, const Backend& backend) {
  if (samples == 0 || predictors == 0 || folds < 2) return false;
#if defined(FASTPLS_USE_OPENBLAS)
  // Compact class products make direct OPLS folds cheaper than building a
  // predictor Gram on OpenBLAS. Regression retains moments because they avoid
  // repeating the generally much larger predictor-response product.
  if (classification && accelerator_backend_code(backend, 0) == 0) {
    return false;
  }
#endif
  // One full predictor Gram plus fold-heldout Grams replaces repeated
  // training-matrix deflation. Keep the route only where its leading work is
  // below the aggregate fold work; the common 256 MiB guard is applied by the
  // caller before this comparison.
  return static_cast<long double>(predictors) <
    static_cast<long double>(samples) *
    static_cast<long double>(folds - 1);
}

template<class T, class Backend>
void predictor_gram(ConstMatrixView<T> predictors, Backend& backend,
                    Matrix<T>& result) {
  result.resize(predictors.columns(), predictors.columns());
  backend.gemm(predictors, predictors, true, false, result.view());
}

template<class T, class Backend>
Matrix<T> predictor_gram(ConstMatrixView<T> predictors, Backend& backend) {
  Matrix<T> result;
  predictor_gram(predictors, backend, result);
  return result;
}

template<class T, class Backend>
auto response_gram_product(ConstMatrixView<T> responses, Backend& backend,
                           MatrixView<T> output, int)
    -> decltype(
      backend.self_gram(responses, false, output, false), void()
    ) {
  backend.self_gram(responses, false, output, false);
}

template<class T, class Backend>
void response_gram_product(ConstMatrixView<T> responses, Backend& backend,
                           MatrixView<T> output, long) {
  // Generic accelerator backends return a complete product. The fold
  // extractor consumes only the lower triangle in either representation.
  backend.gemm(responses, responses, false, true, output);
}

template<class T, class Backend>
Matrix<T> response_gram(ConstMatrixView<T> responses, Backend& backend) {
  Matrix<T> result(responses.rows(), responses.rows());
  response_gram_product<T>(responses, backend, result.view(), 0);
  return result;
}

template<class T>
std::vector<long double> symmetric_row_sums(ConstMatrixView<T> gram) {
  if (gram.empty() || gram.rows() != gram.columns()) {
    throw std::invalid_argument("response Gram matrix must be square");
  }
  std::vector<long double> sums(gram.rows(), 0.0L);
  for (std::size_t column = 0; column < gram.columns(); ++column) {
    for (std::size_t row = column; row < gram.rows(); ++row) {
      const long double value = static_cast<long double>(gram(row, column));
      sums[row] += value;
      if (row != column) sums[column] += value;
    }
  }
  return sums;
}

template<class T>
void prepare_centered_training_response_gram(
    ConstMatrixView<T> full_gram,
    const std::vector<std::size_t>& training_rows,
    const std::vector<std::size_t>& heldout_rows,
    const std::vector<long double>& full_row_sums,
    Matrix<T>& centered_gram,
    std::vector<long double>& row_means) {
  const std::size_t n = training_rows.size();
  if (full_gram.rows() != full_gram.columns() || n == 0) {
    throw std::invalid_argument(
      "fold response-Gram dimensions are invalid"
    );
  }
  if (full_row_sums.size() != full_gram.rows()) {
    throw std::invalid_argument("response Gram row sums are invalid");
  }

  centered_gram.resize(n, n);
  row_means.assign(n, 0.0L);
  for (std::size_t row = 0; row < n; ++row) {
    const std::size_t source_row = training_rows[row];
    long double sum = full_row_sums[source_row];
    for (const std::size_t source_column : heldout_rows) {
      sum -= source_row >= source_column ?
        static_cast<long double>(full_gram(source_row, source_column)) :
        static_cast<long double>(full_gram(source_column, source_row));
    }
    row_means[row] = sum / static_cast<long double>(n);
  }
  long double grand_mean = 0.0L;
  for (const long double value : row_means) grand_mean += value;
  grand_mean /= static_cast<long double>(n);
  for (std::size_t column = 0; column < n; ++column) {
    const std::size_t source_column = training_rows[column];
    for (std::size_t row = 0; row < n; ++row) {
      const std::size_t source_row = training_rows[row];
      const T value = source_row >= source_column ?
        full_gram(source_row, source_column) :
        full_gram(source_column, source_row);
      centered_gram(row, column) = static_cast<T>(
        static_cast<long double>(value) -
        row_means[row] - row_means[column] + grand_mean
      );
    }
  }
}

template<class T>
void preload_standardized_predictor_gram(
    ConstMatrixView<T> full_gram, ConstMatrixView<T> heldout_gram,
    const std::vector<T>& center, const std::vector<T>& scale,
    std::size_t training_rows, SimplsWorkspace<T>& workspace) {
  const std::size_t p = full_gram.rows();
  if (p == 0 || full_gram.columns() != p || heldout_gram.rows() != p ||
      heldout_gram.columns() != p || center.size() != p ||
      scale.size() != p || training_rows == 0) {
    throw std::invalid_argument(
      "fold predictor-Gram dimensions are invalid"
    );
  }
  workspace.predictor_crossprod.resize(p, p);
  for (std::size_t column = 0; column < p; ++column) {
    if (!std::isfinite(scale[column]) || scale[column] <= T(0)) {
      throw std::runtime_error("fold predictor scale is invalid");
    }
    for (std::size_t row = 0; row < p; ++row) {
      if (!std::isfinite(scale[row]) || scale[row] <= T(0)) {
        throw std::runtime_error("fold predictor scale is invalid");
      }
      const long double raw =
        static_cast<long double>(full_gram(row, column)) -
        static_cast<long double>(heldout_gram(row, column));
      const long double centered = raw -
        static_cast<long double>(training_rows) *
        static_cast<long double>(center[row]) *
        static_cast<long double>(center[column]);
      workspace.predictor_crossprod(row, column) = static_cast<T>(
        centered /
        (static_cast<long double>(scale[row]) *
         static_cast<long double>(scale[column]))
      );
    }
  }
  workspace.predictor_crossprod_preloaded = true;
}

template<class T, class Backend>
DenseSufficientStatistics<T> dense_sufficient_statistics(
    ConstMatrixView<T> predictors, ConstMatrixView<T> responses,
    Backend& backend) {
  if (predictors.empty() || responses.empty() ||
      predictors.rows() != responses.rows()) {
    throw std::invalid_argument(
      "dense cross-validation sufficient-statistic dimensions are invalid"
    );
  }
  DenseSufficientStatistics<T> result;
  result.predictor_sum.assign(predictors.columns(), 0.0L);
  result.predictor_sum_squares.assign(predictors.columns(), 0.0L);
  result.response_sum.assign(responses.columns(), 0.0L);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t predictor = 0;
       predictor < predictors.columns(); ++predictor) {
    for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
      const double value = static_cast<double>(predictors(sample, predictor));
      result.predictor_sum[predictor] += value;
      result.predictor_sum_squares[predictor] += value * value;
    }
  }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t response = 0;
       response < responses.columns(); ++response) {
    for (std::size_t sample = 0; sample < responses.rows(); ++sample) {
      result.response_sum[response] += responses(sample, response);
    }
  }
  result.predictor_response.resize(
    predictors.columns(), responses.columns()
  );
  backend.gemm(
    predictors, responses, true, false, result.predictor_response.view()
  );
  return result;
}

// Marginal moments are inexpensive to retain even when the predictor-response
// cross-covariance is too large to materialize.  Computing them once and
// subtracting each held-out fold avoids rescanning every 90% training response
// matrix merely to recover fold-specific centering constants.
template<class T>
DenseSufficientStatistics<T> dense_marginal_statistics(
    ConstMatrixView<T> predictors, ConstMatrixView<T> responses) {
  if (predictors.empty() || responses.empty() ||
      predictors.rows() != responses.rows()) {
    throw std::invalid_argument(
      "dense cross-validation marginal-statistic dimensions are invalid"
    );
  }
  DenseSufficientStatistics<T> result;
  result.predictor_sum.assign(predictors.columns(), 0.0L);
  result.predictor_sum_squares.assign(predictors.columns(), 0.0L);
  result.response_sum.assign(responses.columns(), 0.0L);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t predictor = 0;
       predictor < predictors.columns(); ++predictor) {
    for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
      const double value = static_cast<double>(predictors(sample, predictor));
      result.predictor_sum[predictor] += value;
      result.predictor_sum_squares[predictor] += value * value;
    }
  }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t response = 0;
       response < responses.columns(); ++response) {
    for (std::size_t sample = 0; sample < responses.rows(); ++sample) {
      result.response_sum[response] += responses(sample, response);
    }
  }
  return result;
}

template<class T>
LabelSufficientStatistics<T> label_sufficient_statistics(
    ConstMatrixView<T> predictors, const int* labels,
    std::size_t class_count) {
  LabelSufficientStatistics<T> result;
  result.predictors = predictors.columns();
  result.classes = class_count;
  result.predictor_sum.assign(result.predictors, 0.0L);
  result.predictor_sum_squares.assign(result.predictors, 0.0L);
  result.class_count.assign(class_count, 0.0L);
  result.class_predictor_sum.assign(
    result.predictors * class_count, 0.0L
  );
  for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
    const int encoded = labels[sample];
    if (encoded < 1 || static_cast<std::size_t>(encoded) > class_count) {
      throw std::invalid_argument(
        "cross-validation labels must be encoded as 1..n_classes"
      );
    }
    const std::size_t label = static_cast<std::size_t>(encoded - 1);
    result.class_count[label] += 1.0L;
  }
#if defined(__APPLE__)
  for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
    const std::size_t label = static_cast<std::size_t>(labels[sample] - 1);
    for (std::size_t predictor = 0;
         predictor < result.predictors; ++predictor) {
      const typename LabelSufficientStatistics<T>::Accumulator value =
        predictors(sample, predictor);
      result.predictor_sum[predictor] += value;
      result.predictor_sum_squares[predictor] += value * value;
      result.class_predictor_sum[predictor + label * result.predictors] +=
        value;
    }
  }
#else
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t predictor = 0;
       predictor < result.predictors; ++predictor) {
    for (std::size_t sample = 0; sample < predictors.rows(); ++sample) {
      const typename LabelSufficientStatistics<T>::Accumulator value =
        predictors(sample, predictor);
      const std::size_t label = static_cast<std::size_t>(labels[sample] - 1);
      result.predictor_sum[predictor] += value;
      result.predictor_sum_squares[predictor] += value * value;
      result.class_predictor_sum[predictor + label * result.predictors] +=
        value;
    }
  }
#endif
  return result;
}

template<class T>
LabelCrossprodResult<T> prepare_label_fold_from_statistics(
    MatrixView<T> train, MatrixView<T> test,
    ConstMatrixView<T> all_predictors, const int* labels,
    const std::vector<std::size_t>& test_rows,
    const std::vector<int>& active, PredictorScaling scaling,
    const LabelSufficientStatistics<T>& full,
    std::size_t training_rows_override = 0) {
  const std::size_t p = all_predictors.columns();
  const std::size_t classes = active.size();
  const std::size_t ntrain = train.empty() ?
    training_rows_override : train.rows();
  if (p != full.predictors ||
      (!train.empty() && train.columns() != p) || test.columns() != p ||
      classes < 2 || ntrain < 1) {
    throw std::invalid_argument(
      "fold label sufficient-statistic dimensions are invalid"
    );
  }

  using Accumulator = typename LabelSufficientStatistics<T>::Accumulator;
  std::vector<Accumulator> predictor_sum = full.predictor_sum;
  std::vector<Accumulator> predictor_sum_squares =
    full.predictor_sum_squares;
  std::vector<Accumulator> class_count(classes, Accumulator(0));
  std::vector<Accumulator> class_predictor_sum(
    p * classes, Accumulator(0)
  );
  std::vector<int> active_map(full.classes, -1);
  for (std::size_t index = 0; index < classes; ++index) {
    const std::size_t label = static_cast<std::size_t>(active[index] - 1);
    active_map[label] = static_cast<int>(index);
    class_count[index] = full.class_count[label];
    for (std::size_t predictor = 0; predictor < p; ++predictor) {
      class_predictor_sum[predictor + index * p] =
        full.class_predictor_sum[predictor + label * p];
    }
  }
#if defined(__APPLE__)
  for (const std::size_t sample : test_rows) {
    const std::size_t label = static_cast<std::size_t>(labels[sample] - 1);
    const int compact = label < active_map.size() ? active_map[label] : -1;
    for (std::size_t predictor = 0; predictor < p; ++predictor) {
      const Accumulator value = all_predictors(sample, predictor);
      predictor_sum[predictor] -= value;
      predictor_sum_squares[predictor] -= value * value;
      if (compact >= 0) {
        class_predictor_sum[
          predictor + static_cast<std::size_t>(compact) * p
        ] -= value;
      }
    }
    if (compact >= 0) {
      class_count[static_cast<std::size_t>(compact)] -= Accumulator(1);
    }
  }
#else
  for (const std::size_t sample : test_rows) {
    const std::size_t label = static_cast<std::size_t>(labels[sample] - 1);
    const int compact = label < active_map.size() ? active_map[label] : -1;
    if (compact >= 0) class_count[static_cast<std::size_t>(compact)] -= 1.0;
  }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t predictor = 0; predictor < p; ++predictor) {
    for (const std::size_t sample : test_rows) {
      const std::size_t label = static_cast<std::size_t>(labels[sample] - 1);
      const int compact = label < active_map.size() ? active_map[label] : -1;
      const Accumulator value = all_predictors(sample, predictor);
      predictor_sum[predictor] -= value;
      predictor_sum_squares[predictor] -= value * value;
      if (compact >= 0) {
        class_predictor_sum[
          predictor + static_cast<std::size_t>(compact) * p
        ] -= value;
      }
    }
  }
#endif

  LabelCrossprodResult<T> result;
  result.crossprod.resize(p, classes);
  result.class_predictor_sums.resize(p, classes);
  result.predictor_center.assign(p, T(0));
  result.predictor_scale.assign(p, T(1));
  result.response_mean.resize(classes);
  result.class_counts.resize(classes);
  for (std::size_t response = 0; response < classes; ++response) {
    if (class_count[response] <= 0.0L) {
      throw std::invalid_argument(
        "fold label sufficient statistics contain an empty class"
      );
    }
    result.class_counts[response] = static_cast<T>(class_count[response]);
    result.response_mean[response] = static_cast<T>(
      class_count[response] / static_cast<long double>(ntrain)
    );
  }

  for (std::size_t predictor = 0; predictor < p; ++predictor) {
    long double center = 0.0L;
    long double scale = 1.0L;
    if (scaling != PredictorScaling::none) {
      center = predictor_sum[predictor] /
        static_cast<long double>(ntrain);
      if (scaling == PredictorScaling::autoscaling) {
        const long double centered_ss = std::max(
          0.0L, predictor_sum_squares[predictor] -
            static_cast<long double>(ntrain) * center * center
        );
        scale = std::sqrt(
          centered_ss /
          static_cast<long double>(std::max<std::size_t>(ntrain - 1, 1))
        );
        if (!std::isfinite(static_cast<double>(scale)) || scale <= 0.0L) {
          scale = 1.0L;
        }
      }
    }
    result.predictor_center[predictor] = static_cast<T>(center);
    result.predictor_scale[predictor] = static_cast<T>(scale);
    const long double standardized_total =
      (predictor_sum[predictor] -
       static_cast<long double>(ntrain) * center) / scale;
    for (std::size_t response = 0; response < classes; ++response) {
      const long double standardized_class = (
        class_predictor_sum[predictor + response * p] -
        class_count[response] * center
      ) / scale;
      result.class_predictor_sums(predictor, response) =
        static_cast<T>(standardized_class);
      result.crossprod(predictor, response) = static_cast<T>(
        standardized_class - standardized_total *
          static_cast<long double>(result.response_mean[response])
      );
    }
  }
  if (!train.empty()) {
    standardize(train, result.predictor_center, result.predictor_scale);
  }
  standardize(test, result.predictor_center, result.predictor_scale);
  return result;
}

template<class T, class Model, class Backend>
std::vector<LdaModel<T>> train_lda_from_predictor_moments(
    const Model& model, ConstMatrixView<T> predictor_gram,
    ConstMatrixView<T> class_predictor_sums,
    const std::vector<T>& class_counts, std::size_t sample_count,
    const int* components, std::size_t prefix_count, Backend& backend) {
  const std::size_t retained = model.completed_components;
  if (retained == 0 || predictor_gram.rows() != predictor_gram.columns() ||
      predictor_gram.rows() != model.weights.rows() ||
      class_predictor_sums.rows() != model.weights.rows() ||
      class_predictor_sums.columns() != class_counts.size()) {
    throw std::invalid_argument(
      "cross-validation LDA predictor moments are inconsistent"
    );
  }
  ConstMatrixView<T> projection(
    model.weights.data(), model.weights.rows(), retained,
    model.weights.view().leading_dimension()
  );
  Matrix<T> gram_projection(predictor_gram.rows(), retained);
  backend.gemm(
    predictor_gram, projection, false, false, gram_projection.view()
  );
  Matrix<T> score_gram(retained, retained);
  backend.gemm(
    projection, gram_projection.view(), true, false, score_gram.view()
  );
  Matrix<T> class_score_sums(class_counts.size(), retained);
  backend.gemm(
    class_predictor_sums, projection, true, false,
    class_score_sums.view()
  );
  return train_lda_prefixes_from_moments<T>(
    score_gram.view(), class_score_sums.view(), class_counts.data(),
    class_counts.size(), sample_count, components, prefix_count, backend
  );
}

template<class T, class Backend>
DensePreprocessingResult<T> prepare_dense_fold_from_statistics(
    MatrixView<T> train, MatrixView<T> test,
    ConstMatrixView<T> test_responses,
    const DenseSufficientStatistics<T>& full, PredictorScaling scaling,
    Backend& backend, std::size_t training_rows_override = 0) {
  const std::size_t p = train.empty() ?
    full.predictor_sum.size() : train.columns();
  const std::size_t q = test_responses.columns();
  const std::size_t ntrain = train.empty() ?
    training_rows_override : train.rows();
  if (p == 0 || q == 0 || ntrain == 0 || test.rows() == 0 ||
      test.rows() != test_responses.rows() || test.columns() != p ||
      (!train.empty() && train.columns() != p) ||
      full.predictor_sum.size() != p ||
      full.predictor_sum_squares.size() != p ||
      full.response_sum.size() != q ||
      full.predictor_response.rows() != p ||
      full.predictor_response.columns() != q) {
    throw std::invalid_argument(
      "fold dense sufficient-statistic dimensions are invalid"
    );
  }

  std::vector<double> predictor_sum = full.predictor_sum;
  std::vector<double> predictor_sum_squares =
    full.predictor_sum_squares;
  std::vector<double> response_sum = full.response_sum;
  for (std::size_t predictor = 0; predictor < p; ++predictor) {
    for (std::size_t sample = 0; sample < test.rows(); ++sample) {
      const double value = static_cast<double>(test(sample, predictor));
      predictor_sum[predictor] -= value;
      predictor_sum_squares[predictor] -= value * value;
    }
  }
  for (std::size_t response = 0; response < q; ++response) {
    for (std::size_t sample = 0; sample < test_responses.rows(); ++sample) {
      response_sum[response] -= test_responses(sample, response);
    }
  }

  Matrix<T> heldout_product(p, q);
  backend.gemm(
    ConstMatrixView<T>(test), test_responses, true, false,
    heldout_product.view()
  );
  DensePreprocessingResult<T> result;
  result.crossprod.resize(p, q);
  result.predictor_center.assign(p, T(0));
  result.predictor_scale.assign(p, T(1));
  result.response_mean.resize(q);
  for (std::size_t response = 0; response < q; ++response) {
    result.response_mean[response] = static_cast<T>(
      response_sum[response] / static_cast<long double>(ntrain)
    );
  }

  for (std::size_t predictor = 0; predictor < p; ++predictor) {
    long double center = 0.0L;
    long double scale = 1.0L;
    if (scaling != PredictorScaling::none) {
      center = predictor_sum[predictor] /
        static_cast<long double>(ntrain);
      if (scaling == PredictorScaling::autoscaling) {
        const long double centered_ss = std::max(
          0.0L, predictor_sum_squares[predictor] -
            static_cast<long double>(ntrain) * center * center
        );
        scale = std::sqrt(
          centered_ss /
          static_cast<long double>(std::max<std::size_t>(ntrain - 1, 1))
        );
        if (!std::isfinite(static_cast<double>(scale)) || scale <= 0.0L) {
          scale = 1.0L;
        }
      }
    }
    result.predictor_center[predictor] = static_cast<T>(center);
    result.predictor_scale[predictor] = static_cast<T>(scale);
    const long double standardized_sum =
      (predictor_sum[predictor] -
       static_cast<long double>(ntrain) * center) / scale;
    for (std::size_t response = 0; response < q; ++response) {
      const long double raw_product =
        static_cast<long double>(full.predictor_response(
          predictor, response
        )) - static_cast<long double>(heldout_product(predictor, response));
      const long double standardized_product =
        (raw_product - center * response_sum[response]) / scale;
      result.crossprod(predictor, response) = static_cast<T>(
        standardized_product - standardized_sum *
          static_cast<long double>(result.response_mean[response])
      );
    }
  }
  if (!train.empty()) {
    standardize(train, result.predictor_center, result.predictor_scale);
  }
  standardize(test, result.predictor_center, result.predictor_scale);
  return result;
}

template<class T>
DensePreprocessingResult<T> prepare_dense_fold_from_marginals(
    MatrixView<T> train, MatrixView<T> test,
    ConstMatrixView<T> all_predictors,
    ConstMatrixView<T> all_responses,
    const std::vector<std::size_t>& test_rows,
    const DenseSufficientStatistics<T>& full, PredictorScaling scaling) {
  const std::size_t p = all_predictors.columns();
  const std::size_t q = all_responses.columns();
  const std::size_t ntrain = train.rows();
  if (p == 0 || q == 0 || ntrain == 0 || test.rows() == 0 ||
      all_predictors.rows() != all_responses.rows() ||
      train.columns() != p || test.columns() != p ||
      test.rows() != test_rows.size() ||
      full.predictor_sum.size() != p ||
      full.predictor_sum_squares.size() != p ||
      full.response_sum.size() != q) {
    throw std::invalid_argument(
      "fold dense marginal-statistic dimensions are invalid"
    );
  }

  std::vector<double> predictor_sum = full.predictor_sum;
  std::vector<double> predictor_sum_squares =
    full.predictor_sum_squares;
  std::vector<double> response_sum = full.response_sum;
  for (std::size_t predictor = 0; predictor < p; ++predictor) {
    for (const std::size_t sample : test_rows) {
      const double value = static_cast<double>(
        all_predictors(sample, predictor)
      );
      predictor_sum[predictor] -= value;
      predictor_sum_squares[predictor] -= value * value;
    }
  }
  for (std::size_t response = 0; response < q; ++response) {
    for (const std::size_t sample : test_rows) {
      response_sum[response] -= all_responses(sample, response);
    }
  }

  DensePreprocessingResult<T> result;
  result.predictor_center.assign(p, T(0));
  result.predictor_scale.assign(p, T(1));
  result.response_mean.resize(q);
  for (std::size_t response = 0; response < q; ++response) {
    result.response_mean[response] = static_cast<T>(
      response_sum[response] / static_cast<long double>(ntrain)
    );
  }
  for (std::size_t predictor = 0; predictor < p; ++predictor) {
    long double center = 0.0L;
    long double scale = 1.0L;
    if (scaling != PredictorScaling::none) {
      center = predictor_sum[predictor] /
        static_cast<long double>(ntrain);
      if (scaling == PredictorScaling::autoscaling) {
        const long double centered_ss = std::max(
          0.0L, predictor_sum_squares[predictor] -
            static_cast<long double>(ntrain) * center * center
        );
        scale = std::sqrt(
          centered_ss /
          static_cast<long double>(std::max<std::size_t>(ntrain - 1, 1))
        );
        if (!std::isfinite(static_cast<double>(scale)) || scale <= 0.0L) {
          scale = 1.0L;
        }
      }
    }
    result.predictor_center[predictor] = static_cast<T>(center);
    result.predictor_scale[predictor] = static_cast<T>(scale);
  }
  standardize(train, result.predictor_center, result.predictor_scale);
  standardize(test, result.predictor_center, result.predictor_scale);
  return result;
}

inline std::vector<FoldPartition> fold_partitions(
    const int* folds, std::size_t sample_count,
    bool retain_training_rows = true) {
  if (folds == nullptr || sample_count < 2) {
    throw std::invalid_argument("cross-validation folds are invalid");
  }
  int maximum = 0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    maximum = std::max(maximum, folds[sample]);
  }
  if (maximum < 1) {
    throw std::invalid_argument(
      "cross-validation requires at least one held-out fold"
    );
  }
  std::vector<FoldPartition> output(static_cast<std::size_t>(maximum));
  std::size_t assigned = 0;
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const int fold = folds[sample];
    if (fold > 0) {
      output[static_cast<std::size_t>(fold - 1)].test.push_back(sample);
    }
    if (fold != 0) {
      ++assigned;
    }
  }
  for (auto& partition : output) {
    partition.training_size = assigned - partition.test.size();
    if (retain_training_rows) {
      partition.train.reserve(partition.training_size);
    }
  }
  if (retain_training_rows) {
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
      if (folds[sample] == 0) continue;
      for (int fold = 1; fold <= maximum; ++fold) {
        if (folds[sample] != fold) {
          output[static_cast<std::size_t>(fold - 1)].train.push_back(sample);
        }
      }
    }
  }
  return output;
}

inline void retain_training_rows(
    std::vector<FoldPartition>& partitions, const int* folds,
    std::size_t sample_count) {
  for (auto& partition : partitions) {
    partition.train.clear();
    partition.train.reserve(partition.training_size);
  }
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    if (folds[sample] == 0) continue;
    for (std::size_t fold = 0; fold < partitions.size(); ++fold) {
      if (folds[sample] != static_cast<int>(fold + 1)) {
        partitions[fold].train.push_back(sample);
      }
    }
  }
}

template<class T>
std::vector<int> active_classes_from_statistics(
    const int* labels, const std::vector<std::size_t>& heldout,
    const LabelSufficientStatistics<T>& full) {
  using Accumulator = typename LabelSufficientStatistics<T>::Accumulator;
  std::vector<Accumulator> counts = full.class_count;
  for (const std::size_t row : heldout) {
    const int label = labels[row] - 1;
    if (label < 0 || static_cast<std::size_t>(label) >= counts.size()) {
      throw std::invalid_argument(
        "cross-validation labels must be encoded as 1..n_classes"
      );
    }
    counts[static_cast<std::size_t>(label)] -= Accumulator(1);
  }
  std::vector<int> active;
  active.reserve(counts.size());
  for (std::size_t label = 0; label < counts.size(); ++label) {
    if (counts[label] > Accumulator(0)) {
      active.push_back(static_cast<int>(label) + 1);
    }
  }
  return active;
}

template<class T>
std::vector<double> classification_q2_path(
    const int* labels, std::size_t class_count,
    const std::vector<FoldPartition>& partitions,
    const std::vector<Matrix<T>>& scores) {
  if (scores.empty()) return {};
  std::vector<long double> tss_by_row(scores.front().rows(), 0.0L);
  for (const auto& partition : partitions) {
    if (partition.train.empty() || partition.test.empty()) continue;
    std::vector<std::size_t> counts(class_count, 0);
    for (const std::size_t row : partition.train) {
      const int label = labels[row] - 1;
      if (label >= 0 && static_cast<std::size_t>(label) < class_count) {
        ++counts[static_cast<std::size_t>(label)];
      }
    }
    const long double train_count = static_cast<long double>(
      partition.train.size()
    );
    for (const std::size_t row : partition.test) {
      const int observed = labels[row] - 1;
      long double row_tss = 0.0L;
      for (std::size_t class_index = 0;
           class_index < class_count; ++class_index) {
        const long double mean = counts[class_index] / train_count;
        const long double value =
          static_cast<int>(class_index) == observed ? 1.0L : 0.0L;
        const long double centered = value - mean;
        row_tss += centered * centered;
      }
      tss_by_row[row] = row_tss;
    }
  }
  const long double tss = std::accumulate(
    tss_by_row.begin(), tss_by_row.end(), 0.0L
  );
  std::vector<double> result(scores.size(),
    std::numeric_limits<double>::quiet_NaN());
  if (!(tss > 0.0L)) return result;
  for (std::size_t prefix = 0; prefix < scores.size(); ++prefix) {
    long double press = 0.0L;
    for (std::size_t class_index = 0;
         class_index < class_count; ++class_index) {
      for (std::size_t row = 0; row < scores[prefix].rows(); ++row) {
        if (!(tss_by_row[row] > 0.0L)) continue;
        const long double observed =
          labels[row] - 1 == static_cast<int>(class_index) ? 1.0L : 0.0L;
        const long double residual = observed -
          static_cast<long double>(scores[prefix](row, class_index));
        press += residual * residual;
      }
    }
    result[prefix] = 1.0 - static_cast<double>(press / tss);
  }
  return result;
}

template<class T>
void gather_rows(ConstMatrixView<T> source,
                 const std::vector<std::size_t>& rows,
                 Matrix<T>& output) {
  output.resize(rows.size(), source.columns());
  for (std::size_t column = 0; column < source.columns(); ++column) {
    for (std::size_t row = 0; row < rows.size(); ++row) {
      output(row, column) = source(rows[row], column);
    }
  }
}

template<class T>
void gather_rows_padded(ConstMatrixView<T> source,
                        const std::vector<std::size_t>& rows,
                        std::size_t row_multiple,
                        PaddedMatrix<T>& output) {
  output.resize(rows.size(), source.columns(), row_multiple);
  for (std::size_t column = 0; column < source.columns(); ++column) {
    for (std::size_t row = 0; row < rows.size(); ++row) {
      output(row, column) = source(rows[row], column);
    }
  }
}

template<class T>
std::size_t response_row_alignment(std::size_t responses) {
  if (responses < 2048) return 1;
  constexpr std::size_t cache_line_bytes = 64;
  return cache_line_bytes / sizeof(T);
}

template<class T>
Matrix<T> gather_rows(ConstMatrixView<T> source,
                      const std::vector<std::size_t>& rows) {
  Matrix<T> output;
  gather_rows(source, rows, output);
  return output;
}

template<class T>
void standardize(MatrixView<T> values, const std::vector<T>& center,
                 const std::vector<T>& scale) {
  if (values.columns() != center.size() || center.size() != scale.size()) {
    throw std::invalid_argument(
      "cross-validation predictor preprocessing is inconsistent"
    );
  }
  for (std::size_t column = 0; column < values.columns(); ++column) {
    if (!std::isfinite(scale[column]) || scale[column] <= T(0)) {
      throw std::runtime_error("cross-validation predictor scale is invalid");
    }
    for (std::size_t row = 0; row < values.rows(); ++row) {
      values(row, column) =
        (values(row, column) - center[column]) / scale[column];
    }
  }
}

template<class T, class Backend>
void make_kernel_fold(Matrix<T>& train, Matrix<T>& test,
                      const KernelCvControls& controls, Backend& backend) {
  if (controls.kernel == KernelType::linear) {
    throw std::invalid_argument(
      "linear kernel PLS must use the direct SIMPLS route"
    );
  }
  const T gamma = static_cast<T>(controls.gamma);
  const T offset = static_cast<T>(controls.offset);
  if (!std::isfinite(gamma) || gamma <= T(0) ||
      !std::isfinite(offset) || controls.degree < 1) {
    throw std::invalid_argument("kernel PLS controls are invalid");
  }
  Matrix<T> train_kernel = kernel_matrix(
    train.view(), train.view(), controls.kernel, gamma,
    controls.degree, offset, backend
  );
  const auto centering = center_kernel_train(train_kernel.view());
  Matrix<T> test_kernel = kernel_matrix(
    test.view(), train.view(), controls.kernel, gamma,
    controls.degree, offset, backend
  );
  center_kernel_test(
    test_kernel.view(), centering.column_means.data(),
    centering.column_means.size(), centering.grand_mean
  );
  train = std::move(train_kernel);
  test = std::move(test_kernel);
}

template<class T, class Backend>
Matrix<T> project_scores(ConstMatrixView<T> predictors,
                         ConstMatrixView<T> weights,
                         std::size_t components, Backend& backend) {
  if (components < 1 || components > weights.columns() ||
      predictors.columns() != weights.rows()) {
    throw std::invalid_argument(
      "cross-validation score projection dimensions are invalid"
    );
  }
  ConstMatrixView<T> prefix(
    weights.data(), weights.rows(), components, weights.leading_dimension()
  );
  Matrix<T> scores(predictors.rows(), components);
  backend.gemm(predictors, prefix, false, false, scores.view());
  return scores;
}

template<class T, class Backend>
Matrix<T> predict_plssvd_from_scores(
    const PlssvdModel<T>& model, ConstMatrixView<T> scores,
    std::size_t prefix_index, const std::vector<T>& response_mean,
    Backend& backend) {
  if (prefix_index >= model.components.size() ||
      prefix_index >= model.prediction_weights.size()) {
    throw std::invalid_argument("PLS-SVD prediction prefix is invalid");
  }
  const std::size_t components = static_cast<std::size_t>(
    model.components[prefix_index]
  );
  if (scores.columns() < components) {
    throw std::invalid_argument("PLS-SVD projected score path is incomplete");
  }
  ConstMatrixView<T> score_prefix(
    scores.data(), scores.rows(), components, scores.leading_dimension()
  );
  Matrix<T> prediction(scores.rows(), response_mean.size());
  backend.gemm(
    score_prefix, model.prediction_weights[prefix_index].view(),
    false, false, prediction.view()
  );
  for (std::size_t column = 0; column < prediction.columns(); ++column) {
    for (std::size_t row = 0; row < prediction.rows(); ++row) {
      prediction(row, column) += response_mean[column];
    }
  }
  return prediction;
}

template<class T, class Backend>
Matrix<T> predict_plssvd(const PlssvdModel<T>& model,
                         ConstMatrixView<T> predictors,
                         std::size_t prefix_index,
                         const std::vector<T>& response_mean,
                         Backend& backend) {
  if (prefix_index >= model.components.size()) {
    throw std::invalid_argument("PLS-SVD prediction prefix is invalid");
  }
  const std::size_t components = static_cast<std::size_t>(
    model.components[prefix_index]
  );
  Matrix<T> scores = project_scores(
    predictors, model.weights.view(), components, backend
  );
  return predict_plssvd_from_scores(
    model, scores.view(), prefix_index, response_mean, backend
  );
}

template<class T>
Matrix<T> prediction_with_response_mean(
    ConstMatrixView<T> centered_prediction,
    const std::vector<T>& response_mean) {
  if (centered_prediction.columns() != response_mean.size()) {
    throw std::invalid_argument(
      "prediction response mean dimensions are invalid"
    );
  }
  Matrix<T> prediction(
    centered_prediction.rows(), centered_prediction.columns()
  );
  for (std::size_t column = 0; column < prediction.columns(); ++column) {
    for (std::size_t row = 0; row < prediction.rows(); ++row) {
      prediction(row, column) =
        centered_prediction(row, column) + response_mean[column];
    }
  }
  return prediction;
}

template<class T, class Backend>
Matrix<T> predict_simpls_from_scores(
    const SimplsModel<T>& model, ConstMatrixView<T> scores,
    std::size_t components, const std::vector<T>& response_mean,
    Backend& backend) {
  if (components == 0 || components > model.completed_components ||
      components > scores.columns()) {
    throw std::invalid_argument(
      "SIMPLS projected prediction dimensions are invalid"
    );
  }
  ConstMatrixView<T> score_prefix(
    scores.data(), scores.rows(), components, scores.leading_dimension()
  );
  ConstMatrixView<T> loading_prefix(
    model.response_loadings.data(), model.response_loadings.rows(),
    components, model.response_loadings.rows()
  );
  Matrix<T> centered_prediction(scores.rows(), response_mean.size());
  backend.gemm(
    score_prefix, loading_prefix, false, true,
    centered_prediction.view()
  );
  return prediction_with_response_mean<T>(
    centered_prediction.view(), response_mean
  );
}

template<class T>
Matrix<T> initialize_simpls_prediction(
    std::size_t rows, const std::vector<T>& response_mean) {
  Matrix<T> prediction(rows, response_mean.size());
  for (std::size_t column = 0; column < prediction.columns(); ++column) {
    std::fill_n(
      prediction.data() + column * prediction.rows(), prediction.rows(),
      response_mean[column]
    );
  }
  return prediction;
}

template<class T, class Backend>
auto accumulate_matrix_product(
    Backend& backend, ConstMatrixView<T> left, ConstMatrixView<T> right,
    bool transpose_left, bool transpose_right, MatrixView<T> output, int)
    -> decltype(
      backend.gemm_accumulate(
        left, right, transpose_left, transpose_right, output
      ),
      bool()) {
  backend.gemm_accumulate(
    left, right, transpose_left, transpose_right, output
  );
  return true;
}

template<class T, class Backend>
bool accumulate_matrix_product(
    Backend&, ConstMatrixView<T>, ConstMatrixView<T>, bool, bool,
    MatrixView<T>, long) {
  return false;
}

template<class T, class Backend>
void update_simpls_prediction_from_scores(
    const SimplsModel<T>& model, ConstMatrixView<T> scores,
    std::size_t first_component, std::size_t component_count,
    MatrixView<T> prediction, Matrix<T>& contribution, Backend& backend) {
  if (component_count <= first_component ||
      component_count > model.completed_components ||
      component_count > scores.columns() ||
      prediction.rows() != scores.rows() ||
      prediction.columns() != model.response_loadings.rows()) {
    throw std::invalid_argument(
      "SIMPLS incremental prediction dimensions are invalid"
    );
  }
  const std::size_t added = component_count - first_component;
  ConstMatrixView<T> score_delta(
    scores.data() + first_component * scores.leading_dimension(),
    scores.rows(), added, scores.leading_dimension()
  );
  ConstMatrixView<T> loading_delta(
    model.response_loadings.data() +
      first_component * model.response_loadings.rows(),
    model.response_loadings.rows(), added,
    model.response_loadings.view().leading_dimension()
  );
  if (accumulate_matrix_product<T>(
        backend, score_delta, loading_delta, false, true, prediction, 0
      )) {
    return;
  }
  contribution.resize(prediction.rows(), prediction.columns());
  backend.gemm(
    score_delta, loading_delta, false, true, contribution.view()
  );
  for (std::size_t column = 0; column < prediction.columns(); ++column) {
    for (std::size_t row = 0; row < prediction.rows(); ++row) {
      prediction(row, column) += contribution(row, column);
    }
  }
}

template<class T, class Backend>
Matrix<T> predict_simpls(const SimplsModel<T>& model,
                         ConstMatrixView<T> predictors,
                         std::size_t components,
                         const std::vector<T>& response_mean,
                         Backend& backend) {
  Matrix<T> prediction = predict_simpls_preprocessed(
    predictors, model, components, backend
  );
  return prediction_with_response_mean<T>(
    prediction.view(), response_mean
  );
}

template<class T>
std::vector<int> active_classes(const int* labels,
                                const std::vector<std::size_t>& rows,
                                std::size_t class_count) {
  std::vector<int> counts(class_count, 0);
  for (const std::size_t row : rows) {
    const int label = labels[row];
    if (label < 1 || static_cast<std::size_t>(label) > class_count) {
      throw std::invalid_argument(
        "cross-validation labels must be encoded as 1..n_classes"
      );
    }
    ++counts[static_cast<std::size_t>(label - 1)];
  }
  std::vector<int> active;
  for (std::size_t label = 0; label < class_count; ++label) {
    if (counts[label] > 0) active.push_back(static_cast<int>(label) + 1);
  }
  return active;
}

inline std::vector<std::size_t> compact_labels(
    const int* labels, const std::vector<std::size_t>& rows,
    const std::vector<int>& active) {
  std::vector<int> map(active.empty() ? 0 : active.back() + 1, -1);
  for (std::size_t index = 0; index < active.size(); ++index) {
    if (active[index] >= static_cast<int>(map.size())) {
      map.resize(static_cast<std::size_t>(active[index]) + 1, -1);
    }
    map[static_cast<std::size_t>(active[index])] = static_cast<int>(index);
  }
  std::vector<std::size_t> output(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    const int label = labels[rows[index]];
    if (label < 0 || static_cast<std::size_t>(label) >= map.size() ||
        map[static_cast<std::size_t>(label)] < 0) {
      throw std::invalid_argument("cross-validation class mapping failed");
    }
    output[index] = static_cast<std::size_t>(
      map[static_cast<std::size_t>(label)]
    );
  }
  return output;
}

template<class T>
std::vector<int> predicted_classes(ConstMatrixView<T> scores,
                                   const std::vector<int>& active) {
  if (scores.columns() != active.size()) {
    throw std::invalid_argument(
      "cross-validation class-score dimensions are invalid"
    );
  }
  std::vector<int> output(scores.rows());
  for (std::size_t row = 0; row < scores.rows(); ++row) {
    output[row] = active[row_argmax(scores, row)];
  }
  return output;
}

template<class T>
std::vector<int> lda_predictions(ConstMatrixView<T> scores,
                                 const LdaModel<T>& model,
                                 const std::vector<int>& active) {
  Matrix<T> discriminants = lda_scores(scores, model);
  return predicted_classes<T>(
    ConstMatrixView<T>(discriminants.view()), active
  );
}

template<class T, class Backend>
std::vector<int> lda_predictions(ConstMatrixView<T> scores,
                                 const LdaModel<T>& model,
                                 const std::vector<int>& active,
                                 Backend& backend) {
  if (scores.empty() || scores.columns() != model.linear.columns() ||
      model.linear.rows() != model.constants.size() ||
      model.linear.rows() != active.size()) {
    throw std::invalid_argument(
      "cross-validation LDA prediction dimensions are invalid"
    );
  }
  const long double work = static_cast<long double>(scores.rows()) *
    scores.columns() * model.linear.rows();
  if (work < 1.0e6L) {
    return lda_predictions<T>(scores, model, active);
  }
  Matrix<T> discriminants(scores.rows(), model.linear.rows());
  backend.gemm(
    scores, model.linear.view(), false, true, discriminants.view()
  );
  for (std::size_t class_index = 0;
       class_index < discriminants.columns(); ++class_index) {
    const T constant = model.constants[class_index];
    for (std::size_t row = 0; row < discriminants.rows(); ++row) {
      discriminants(row, class_index) += constant;
    }
  }
  return predicted_classes<T>(
    ConstMatrixView<T>(discriminants.view()), active
  );
}

template<class T>
void accumulate_classification_fold_tss(
    const std::vector<T>& class_counts, std::size_t training_rows,
    const std::vector<int>& active, const int* labels,
    const std::vector<std::size_t>& heldout, long double& tss) {
  if (class_counts.size() != active.size() || training_rows == 0) {
    throw std::invalid_argument(
      "cross-validation Q2 class-count dimensions are invalid"
    );
  }
  std::vector<long double> probabilities(class_counts.size(), 0.0L);
  std::vector<long double> probability_by_label(
    static_cast<std::size_t>(active.back()) + 1, 0.0L
  );
  long double squared_sum = 0.0L;
  for (std::size_t index = 0; index < class_counts.size(); ++index) {
    probabilities[index] = static_cast<long double>(class_counts[index]) /
      static_cast<long double>(training_rows);
    probability_by_label[static_cast<std::size_t>(active[index])] =
      probabilities[index];
    squared_sum += probabilities[index] * probabilities[index];
  }
  for (const std::size_t row : heldout) {
    const int observed = labels[row];
    const long double observed_probability = observed >= 0 &&
      static_cast<std::size_t>(observed) < probability_by_label.size() ?
      probability_by_label[static_cast<std::size_t>(observed)] : 0.0L;
    tss += 1.0L - 2.0L * observed_probability + squared_sum;
  }
}

template<class T>
void accumulate_classification_press(
    ConstMatrixView<T> values, const std::vector<int>& active,
    const int* labels, const std::vector<std::size_t>& rows,
    long double& press) {
  for (std::size_t column = 0; column < values.columns(); ++column) {
    for (std::size_t index = 0; index < rows.size(); ++index) {
      const long double observed = active[column] == labels[rows[index]] ?
        1.0L : 0.0L;
      const long double residual = observed -
        static_cast<long double>(values(index, column));
      press += residual * residual;
    }
  }
  std::vector<unsigned char> active_label(
    static_cast<std::size_t>(active.back()) + 1, 0
  );
  for (const int label : active) {
    active_label[static_cast<std::size_t>(label)] = 1;
  }
  for (const std::size_t row : rows) {
    const int label = labels[row];
    if (label < 0 || static_cast<std::size_t>(label) >= active_label.size() ||
        active_label[static_cast<std::size_t>(label)] == 0) {
      press += 1.0L;
    }
  }
}

template<class T>
std::vector<int> store_scores_and_predict_classes(
    Matrix<T>* destination, const std::vector<std::size_t>& rows,
    ConstMatrixView<T> values, const std::vector<int>& active,
    const int* labels = nullptr, long double* press = nullptr) {
  if (rows.size() != values.rows() || values.columns() != active.size() ||
      (destination != nullptr &&
       destination->columns() < static_cast<std::size_t>(active.back()))) {
    throw std::invalid_argument(
      "cross-validation response-score dimensions are invalid"
    );
  }
  if (values.rows() * values.columns() < 1000000ULL) {
    if (destination != nullptr) {
      for (std::size_t column = 0; column < values.columns(); ++column) {
        const std::size_t destination_column = static_cast<std::size_t>(
          active[column] - 1
        );
        for (std::size_t index = 0; index < rows.size(); ++index) {
          (*destination)(rows[index], destination_column) =
            values(index, column);
        }
      }
    }
    if (press != nullptr) {
      accumulate_classification_press<T>(
        values, active, labels, rows, *press
      );
    }
    return predicted_classes<T>(values, active);
  }
  std::vector<T> maxima(values.rows(), std::numeric_limits<T>::lowest());
  std::vector<int> predicted(values.rows(), active.front());
  for (std::size_t column = 0; column < values.columns(); ++column) {
    const std::size_t destination_column = static_cast<std::size_t>(
      active[column] - 1
    );
    for (std::size_t index = 0; index < rows.size(); ++index) {
      const T value = values(index, column);
      if (destination != nullptr) {
        (*destination)(rows[index], destination_column) = value;
      }
      if (value > maxima[index]) {
        maxima[index] = value;
        predicted[index] = active[column];
      }
    }
  }
  if (press != nullptr) {
    accumulate_classification_press<T>(values, active, labels, rows, *press);
  }
  return predicted;
}

template<class T>
void store_classes(Matrix<int>& destination,
                   const std::vector<std::size_t>& rows,
                   std::size_t prefix,
                   const std::vector<int>& values) {
  if (rows.size() != values.size()) {
    throw std::invalid_argument(
      "cross-validation class prediction size is invalid"
    );
  }
  for (std::size_t index = 0; index < rows.size(); ++index) {
    destination(rows[index], prefix) = values[index];
  }
}

template<class T>
double accumulate_regression_error(
    Matrix<T>* destination, const std::vector<std::size_t>& rows,
    ConstMatrixView<T> values, ConstMatrixView<T> observed) {
  if (rows.size() != values.rows() ||
      observed.columns() != values.columns() ||
      (destination != nullptr &&
       destination->columns() != values.columns())) {
    throw std::invalid_argument(
      "cross-validation response prediction size is invalid"
    );
  }
  long double sum_squares = 0.0L;
  for (std::size_t column = 0; column < values.columns(); ++column) {
    for (std::size_t index = 0; index < rows.size(); ++index) {
      const std::size_t row = rows[index];
      if (row >= observed.rows() ||
          (destination != nullptr && row >= destination->rows())) {
        throw std::invalid_argument(
          "cross-validation response prediction row is invalid"
        );
      }
      const long double error = values(index, column) - observed(row, column);
      sum_squares += error * error;
      if (destination != nullptr) {
        (*destination)(row, column) = values(index, column);
      }
    }
  }
  return static_cast<double>(sum_squares);
}

template<class T>
double regression_r2(ConstMatrixView<T> observed,
                     ConstMatrixView<T> predicted) {
  if (observed.rows() != predicted.rows() ||
      observed.columns() != predicted.columns() || observed.empty()) {
    throw std::invalid_argument(
      "cross-validation training R2 dimensions are invalid"
    );
  }
  long double residual = 0.0L;
  long double total = 0.0L;
  for (std::size_t column = 0; column < observed.columns(); ++column) {
    long double mean = 0.0L;
    for (std::size_t row = 0; row < observed.rows(); ++row) {
      mean += observed(row, column);
    }
    mean /= static_cast<long double>(observed.rows());
    for (std::size_t row = 0; row < observed.rows(); ++row) {
      const long double error = predicted(row, column) - observed(row, column);
      const long double centered = observed(row, column) - mean;
      residual += error * error;
      total += centered * centered;
    }
  }
  return total > 0.0L ?
    static_cast<double>(1.0L - residual / total) :
    std::numeric_limits<double>::quiet_NaN();
}

inline double balanced_accuracy(const int* labels,
                                const int* predictions,
                                const int* folds,
                                std::size_t sample_count,
                                std::size_t class_count) {
  std::vector<long double> correct(class_count, 0.0L);
  std::vector<long double> total(class_count, 0.0L);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    if (folds != nullptr && folds[sample] == 0) continue;
    const int observed = labels[sample] - 1;
    if (observed < 0 || static_cast<std::size_t>(observed) >= class_count) {
      throw std::invalid_argument(
        "balanced accuracy contains an invalid class label"
      );
    }
    total[static_cast<std::size_t>(observed)] += 1.0L;
    if (predictions[sample] == labels[sample]) {
      correct[static_cast<std::size_t>(observed)] += 1.0L;
    }
  }
  long double sum = 0.0L;
  std::size_t represented = 0;
  for (std::size_t class_index = 0; class_index < class_count; ++class_index) {
    if (total[class_index] > 0.0L) {
      sum += correct[class_index] / total[class_index];
      ++represented;
    }
  }
  return represented > 0 ? static_cast<double>(sum / represented) :
    std::numeric_limits<double>::quiet_NaN();
}

template<class T>
void store_active_scores(Matrix<T>& destination,
                         const std::vector<std::size_t>& rows,
                         ConstMatrixView<T> values,
                         const std::vector<int>& active) {
  if (rows.size() != values.rows() || values.columns() != active.size()) {
    throw std::invalid_argument(
      "cross-validation response-score dimensions are invalid"
    );
  }
  for (std::size_t column = 0; column < values.columns(); ++column) {
    const std::size_t destination_column = static_cast<std::size_t>(
      active[column] - 1
    );
    if (destination_column >= destination.columns()) {
      throw std::invalid_argument(
        "cross-validation response-score class mapping is invalid"
      );
    }
    for (std::size_t index = 0; index < rows.size(); ++index) {
      destination(rows[index], destination_column) = values(index, column);
    }
  }
}

}  // namespace cv_detail

template<class T, class Backend>
ClassificationCvResult<T> cross_validate_classification(
    ConstMatrixView<T> predictors, const int* labels,
    std::size_t class_count, const int* folds,
    const int* components, std::size_t prefix_count,
    PredictorScaling scaling, LinearPlsFamily family,
    ClassificationHead head, const PlssvdControls& plssvd_controls,
    const SimplsControls& simpls_controls, Backend& backend,
    bool store_predictions, bool store_scores,
    std::size_t orthogonal_components = 0,
    const KernelCvControls& kernel_controls = KernelCvControls(),
    bool calculate_training_r2 = false) {
  if (predictors.empty() || labels == nullptr || class_count < 2 ||
      components == nullptr || prefix_count < 1) {
    throw std::invalid_argument(
      "classification cross-validation inputs are invalid"
    );
  }
  auto partitions = cv_detail::fold_partitions(
    folds, predictors.rows(), false
  );
  ClassificationCvResult<T> result;
  result.folds.assign(folds, folds + predictors.rows());
  result.status.assign(partitions.size(), 0);
  result.metrics.assign(prefix_count, 0.0);
  if (store_predictions) {
    result.predictions.resize(predictors.rows(), prefix_count);
  }
  if (store_scores) {
    result.scores.reserve(prefix_count);
    for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
      result.scores.emplace_back(predictors.rows(), class_count);
    }
  }
  if (calculate_training_r2) {
    result.fold_training_r2.resize(partitions.size(), prefix_count);
    std::fill_n(
      result.fold_training_r2.data(), result.fold_training_r2.size(),
      std::numeric_limits<double>::quiet_NaN()
    );
  }
  std::vector<double> totals(prefix_count, 0.0);
  std::vector<long double> q2_press(prefix_count, 0.0L);
  long double q2_tss = 0.0L;
  const bool complete_fold_cover = std::none_of(
    folds, folds + predictors.rows(), [](int value) { return value == 0; }
  );
  const bool reuse_label_statistics =
    complete_fold_cover &&
    cv_detail::fold_label_statistics_enabled() &&
    (family == LinearPlsFamily::plssvd ||
     family == LinearPlsFamily::simpls ||
     family == LinearPlsFamily::opls);
  const auto full_label_statistics = reuse_label_statistics ?
    cv_detail::label_sufficient_statistics(
      predictors, labels, class_count
    ) : cv_detail::LabelSufficientStatistics<T>();
  const std::size_t maximum_component = static_cast<std::size_t>(
    *std::max_element(components, components + prefix_count)
  );
  const bool reuse_predictor_gram =
    complete_fold_cover &&
    (family == LinearPlsFamily::plssvd ||
     family == LinearPlsFamily::simpls ||
     family == LinearPlsFamily::opls) &&
    cv_detail::fold_predictor_gram_enabled<T>(predictors.columns()) &&
    ((family == LinearPlsFamily::opls &&
      cv_detail::fold_opls_moments_enabled(
        predictors.rows(), predictors.columns(), partitions.size(), true,
        backend
      )) || simpls_controls.cache_predictor_crossprod ||
     cv_detail::fold_simpls_moments_enabled(
       predictors.rows(), predictors.columns(), maximum_component
     ));
  Matrix<T> full_predictor_gram;
  if (reuse_predictor_gram) {
    full_predictor_gram = cv_detail::predictor_gram(predictors, backend);
  }
  const bool moments_only_cv = !calculate_training_r2 &&
    reuse_label_statistics && reuse_predictor_gram &&
    (family == LinearPlsFamily::plssvd ||
     family == LinearPlsFamily::simpls ||
     family == LinearPlsFamily::opls);
  if (!moments_only_cv) {
    cv_detail::retain_training_rows(partitions, folds, predictors.rows());
  }
  SimplsWorkspace<T> shared_simpls_workspace;
  Matrix<T> heldout_predictor_gram;
  Matrix<T> train;
  Matrix<T> test;

  for (std::size_t fold = 0; fold < partitions.size(); ++fold) {
    const auto& partition = partitions[fold];
    if (partition.test.empty()) {
      result.status[fold] = 2;
      continue;
    }
    cv_detail::configure_backend_problem(
      backend, partition.training_size, predictors.columns(), class_count, 0
    );
    const auto active = reuse_label_statistics ?
      cv_detail::active_classes_from_statistics(
        labels, partition.test, full_label_statistics
      ) : cv_detail::active_classes<T>(
        labels, partition.train, class_count
      );
    if (active.size() <= 1) {
      const int fallback = active.empty() ? 1 : active.front();
      if (store_scores && !active.empty()) {
        const std::vector<T> counts{
          static_cast<T>(partition.training_size)
        };
        cv_detail::accumulate_classification_fold_tss<T>(
          counts, partition.training_size, active, labels,
          partition.test, q2_tss
        );
        for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
          q2_press[prefix] += static_cast<long double>(partition.test.size());
        }
      }
      for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
        std::vector<int> predicted(partition.test.size(), fallback);
        if (store_predictions) {
          cv_detail::store_classes<T>(
            result.predictions, partition.test, prefix, predicted
          );
        }
        if (store_scores) {
          for (const std::size_t row : partition.test) {
            result.scores[prefix](
              row, static_cast<std::size_t>(fallback - 1)
            ) = T(1);
          }
        }
        for (const std::size_t row : partition.test) {
          result.metrics[prefix] += labels[row] == fallback ? 1.0 : 0.0;
          totals[prefix] += 1.0;
        }
      }
      result.status[fold] = 4;
      continue;
    }

    const bool moments_only = moments_only_cv;
    const bool moments_only_simpls = moments_only &&
      family == LinearPlsFamily::simpls;
    const bool moments_only_plssvd = moments_only &&
      family == LinearPlsFamily::plssvd;
    const bool moments_only_opls = moments_only &&
      family == LinearPlsFamily::opls;
    if (moments_only) {
      train.resize(0, 0);
    } else {
      cv_detail::gather_rows(predictors, partition.train, train);
    }
    cv_detail::gather_rows(predictors, partition.test, test);
    const bool needs_fold_predictor_gram = reuse_predictor_gram &&
      (family == LinearPlsFamily::simpls || moments_only);
    if (needs_fold_predictor_gram) {
      cv_detail::predictor_gram(
        ConstMatrixView<T>(test.view()), backend, heldout_predictor_gram
      );
    }
    const bool needs_compact_labels = !moments_only;
    const auto compact = needs_compact_labels ?
      cv_detail::compact_labels(labels, partition.train, active) :
      std::vector<std::size_t>();
    LabelCrossprodResult<T> prepared;
    if (moments_only_opls) {
      prepared = cv_detail::prepare_label_fold_from_statistics(
        train.view(), test.view(), predictors, labels, partition.test,
        active, scaling, full_label_statistics, partition.training_size
      );
    } else if (family == LinearPlsFamily::opls) {
      if (orthogonal_components < 1) {
        throw std::invalid_argument(
          "OPLS cross-validation requires an orthogonal component"
        );
      }
      auto filter = fit_opls_filter_labels<T>(
        std::move(train), compact.data(), compact.size(), active.size(),
        orthogonal_components, scaling, backend
      );
      test = apply_opls_filter<T>(
        std::move(test), filter.predictor_center.data(),
        filter.predictor_scale.data(), filter.predictor_center.size(),
        filter.weights.view(), filter.loadings.view(), backend
      );
      train = std::move(filter.predictors);
      prepared = prepare_scaled_label_crossprod(
        train.view(), compact.data(), compact.size(), active.size(),
        PredictorScaling::none, backend
      );
    } else if (family == LinearPlsFamily::kernelpls) {
      auto input_preprocessing = scaled_label_crossprod_impl(
        ConstMatrixView<T>(train.view()), train.data(),
        train.view().leading_dimension(), compact.data(), compact.size(),
        active.size(), scaling, false
      );
      cv_detail::standardize(
        test.view(), input_preprocessing.predictor_center,
        input_preprocessing.predictor_scale
      );
      cv_detail::make_kernel_fold(
        train, test, kernel_controls, backend
      );
      prepared = prepare_scaled_label_crossprod(
        train.view(), compact.data(), compact.size(), active.size(),
        PredictorScaling::none, backend
      );
    } else if (reuse_label_statistics) {
      prepared = cv_detail::prepare_label_fold_from_statistics(
        train.view(), test.view(), predictors, labels, partition.test,
        active, scaling, full_label_statistics,
        moments_only ? partition.training_size : 0
      );
    } else {
      prepared = prepare_scaled_label_crossprod(
        train.view(), compact.data(), compact.size(), active.size(),
        scaling, backend
      );
      cv_detail::standardize(
        test.view(), prepared.predictor_center, prepared.predictor_scale
      );
    }

    if (needs_fold_predictor_gram) {
      cv_detail::preload_standardized_predictor_gram<T>(
        full_predictor_gram.view(), heldout_predictor_gram.view(),
        prepared.predictor_center, prepared.predictor_scale,
        partition.training_size, shared_simpls_workspace
      );
    }

    if (store_scores) {
      cv_detail::accumulate_classification_fold_tss<T>(
        prepared.class_counts, partition.training_size, active, labels,
        partition.test, q2_tss
      );
    }

    if (moments_only_opls) {
      if (orthogonal_components < 1) {
        throw std::invalid_argument(
          "OPLS cross-validation requires an orthogonal component"
        );
      }
      auto filter = fit_opls_filter_from_moments<T>(
        shared_simpls_workspace.predictor_crossprod.view(),
        prepared.crossprod.view(), orthogonal_components,
        prepared.predictor_center, prepared.predictor_scale, backend
      );
      if (filter.filter.completed_components != orthogonal_components) {
        throw std::runtime_error(
          "OPLS cross-validation could not complete orthogonal filtering"
        );
      }
      apply_opls_deflation_inplace<T>(
        test.view(), filter.filter.weights.view(),
        filter.filter.loadings.view(), backend
      );
      apply_opls_filter_to_moments<T>(
        prepared.class_predictor_sums.view(),
        filter.filter.weights.view(), filter.filter.loadings.view(), backend
      );
      prepared.crossprod = std::move(filter.crosscov);
      shared_simpls_workspace.predictor_crossprod =
        std::move(filter.predictor_gram);
      shared_simpls_workspace.predictor_crossprod_preloaded = true;
    }

    if (family == LinearPlsFamily::plssvd) {
      PlssvdControls controls = plssvd_controls;
      controls.rsvd.seed += static_cast<unsigned int>(fold);
      auto model = moments_only_plssvd ?
        fit_plssvd_from_moments<T>(
          shared_simpls_workspace.predictor_crossprod.view(),
          prepared.crossprod.view(), components, prefix_count,
          controls, backend
        ) : fit_plssvd_preprocessed<T>(
          train.view(), prepared.crossprod.view(), components, prefix_count,
          controls, backend
        );
      std::vector<LdaModel<T>> lda_models;
      Matrix<T> test_scores = cv_detail::project_scores<T>(
        ConstMatrixView<T>(test.view()),
        ConstMatrixView<T>(model.weights.view()),
        model.completed_components, backend
      );
      if (head == ClassificationHead::lda) {
        if (moments_only_plssvd) {
          lda_models = cv_detail::train_lda_from_predictor_moments<T>(
            model, shared_simpls_workspace.predictor_crossprod.view(),
            prepared.class_predictor_sums.view(), prepared.class_counts,
            partition.training_size, components, prefix_count, backend
          );
        } else {
          std::vector<int> lda_labels(compact.size());
          for (std::size_t i = 0; i < compact.size(); ++i) {
            lda_labels[i] = static_cast<int>(compact[i]) + 1;
          }
          lda_models = train_lda_prefixes(
            model.scores.view(), lda_labels.data(), lda_labels.size(),
            active.size(), components, prefix_count
          );
        }
      }
      const std::vector<T> centered_response_mean(active.size(), T(0));
      for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
        if (calculate_training_r2) {
          const auto centered_fit = cv_detail::predict_plssvd_from_scores<T>(
            model, model.scores.view(), prefix,
            centered_response_mean, backend
          );
          result.fold_training_r2(fold, prefix) = dummy_response_r2(
            compact.data(), compact.size(), prepared.response_mean.data(),
            active.size(), centered_fit.view()
          );
        }
        std::vector<int> predicted;
        if (head == ClassificationHead::lda) {
          ConstMatrixView<T> score_prefix(
            test_scores.data(), test_scores.rows(),
            static_cast<std::size_t>(components[prefix]), test_scores.rows()
          );
          predicted = cv_detail::lda_predictions<T>(
            score_prefix, lda_models[prefix], active, backend
          );
        } else {
          const auto scores = cv_detail::predict_plssvd_from_scores<T>(
            model, test_scores.view(), prefix,
            prepared.response_mean, backend
          );
          predicted = cv_detail::store_scores_and_predict_classes<T>(
            store_scores ? &result.scores[prefix] : nullptr,
            partition.test, ConstMatrixView<T>(scores.view()), active,
            store_scores ? labels : nullptr,
            store_scores ? &q2_press[prefix] : nullptr
          );
        }
        if (head == ClassificationHead::lda && store_scores) {
          const auto scores = cv_detail::predict_plssvd_from_scores<T>(
            model, test_scores.view(), prefix,
            prepared.response_mean, backend
          );
          cv_detail::store_active_scores<T>(
            result.scores[prefix], partition.test,
            ConstMatrixView<T>(scores.view()), active
          );
          cv_detail::accumulate_classification_press<T>(
            ConstMatrixView<T>(scores.view()), active, labels,
            partition.test, q2_press[prefix]
          );
        }
        if (store_predictions) {
          cv_detail::store_classes<T>(
            result.predictions, partition.test, prefix, predicted
          );
        }
        for (std::size_t i = 0; i < partition.test.size(); ++i) {
          result.metrics[prefix] +=
            predicted[i] == labels[partition.test[i]] ? 1.0 : 0.0;
          totals[prefix] += 1.0;
        }
      }
    } else if (family == LinearPlsFamily::simpls ||
               family == LinearPlsFamily::opls ||
               family == LinearPlsFamily::kernelpls) {
      SimplsControls controls = simpls_controls;
      controls.rsvd.seed += static_cast<unsigned int>(fold);
      controls.cache_predictor_crossprod = reuse_predictor_gram;
      const bool moments_only_predictive =
        moments_only_simpls || moments_only_opls;
      controls.store_scores =
        (head == ClassificationHead::lda && !moments_only_predictive) ||
        calculate_training_r2;
      auto model = fit_simpls_preprocessed<T>(
        moments_only_predictive ? ConstMatrixView<T>() :
          ConstMatrixView<T>(train.view()),
        prepared.crossprod.view(), controls, backend,
        shared_simpls_workspace,
        moments_only_predictive ? partition.training_size : 0
      );
      std::vector<LdaModel<T>> lda_models;
      Matrix<T> test_scores = cv_detail::project_scores<T>(
        ConstMatrixView<T>(test.view()),
        ConstMatrixView<T>(model.weights.view()),
        model.completed_components, backend
      );
      if (head == ClassificationHead::lda && moments_only_predictive) {
        lda_models = cv_detail::train_lda_from_predictor_moments<T>(
          model, shared_simpls_workspace.predictor_crossprod.view(),
          prepared.class_predictor_sums.view(), prepared.class_counts,
          partition.training_size, components, prefix_count, backend
        );
      } else if (head == ClassificationHead::lda) {
        std::vector<int> lda_labels(compact.size());
        for (std::size_t i = 0; i < compact.size(); ++i) {
          lda_labels[i] = static_cast<int>(compact[i]) + 1;
        }
        lda_models = train_lda_prefixes(
          model.scores.view(), lda_labels.data(), lda_labels.size(),
          active.size(), components, prefix_count
        );
      }
      const bool need_response_scores =
        head == ClassificationHead::argmax || store_scores;
      Matrix<T> response_scores = need_response_scores ?
        cv_detail::initialize_simpls_prediction<T>(
          test_scores.rows(), prepared.response_mean
        ) : Matrix<T>();
      Matrix<T> prediction_contribution;
      std::size_t previous_component = 0;
      Matrix<T> training_response_scores = calculate_training_r2 ?
        cv_detail::initialize_simpls_prediction<T>(
          model.scores.rows(), prepared.response_mean
        ) : Matrix<T>();
      Matrix<T> training_prediction_contribution;
      std::size_t previous_training_component = 0;
      for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
        const std::size_t requested = static_cast<std::size_t>(
          components[prefix]
        );
        if (need_response_scores) {
          cv_detail::update_simpls_prediction_from_scores<T>(
            model, test_scores.view(), previous_component, requested,
            response_scores.view(), prediction_contribution, backend
          );
          previous_component = requested;
        }
        if (calculate_training_r2) {
          cv_detail::update_simpls_prediction_from_scores<T>(
            model, model.scores.view(), previous_training_component,
            requested, training_response_scores.view(),
            training_prediction_contribution, backend
          );
          previous_training_component = requested;
          Matrix<T> centered_fit(
            training_response_scores.rows(), training_response_scores.columns()
          );
          for (std::size_t column = 0;
               column < centered_fit.columns(); ++column) {
            for (std::size_t row = 0; row < centered_fit.rows(); ++row) {
              centered_fit(row, column) = training_response_scores(row, column) -
                prepared.response_mean[column];
            }
          }
          result.fold_training_r2(fold, prefix) = dummy_response_r2(
            compact.data(), compact.size(), prepared.response_mean.data(),
            active.size(), centered_fit.view()
          );
        }
        std::vector<int> predicted;
        if (head == ClassificationHead::lda) {
          ConstMatrixView<T> score_prefix(
            test_scores.data(), test_scores.rows(),
            requested, test_scores.rows()
          );
          predicted = cv_detail::lda_predictions<T>(
            score_prefix, lda_models[prefix], active, backend
          );
        } else {
          predicted = cv_detail::store_scores_and_predict_classes<T>(
            store_scores ? &result.scores[prefix] : nullptr,
            partition.test,
            ConstMatrixView<T>(response_scores.view()), active,
            store_scores ? labels : nullptr,
            store_scores ? &q2_press[prefix] : nullptr
          );
        }
        if (head == ClassificationHead::lda && store_scores) {
          cv_detail::store_active_scores<T>(
            result.scores[prefix], partition.test,
            ConstMatrixView<T>(response_scores.view()), active
          );
          cv_detail::accumulate_classification_press<T>(
            ConstMatrixView<T>(response_scores.view()), active, labels,
            partition.test, q2_press[prefix]
          );
        }
        if (store_predictions) {
          cv_detail::store_classes<T>(
            result.predictions, partition.test, prefix, predicted
          );
        }
        for (std::size_t i = 0; i < partition.test.size(); ++i) {
          result.metrics[prefix] +=
            predicted[i] == labels[partition.test[i]] ? 1.0 : 0.0;
          totals[prefix] += 1.0;
        }
      }
    } else {
      throw std::invalid_argument(
        "classification CV received an unsupported PLS family"
      );
    }
    result.status[fold] = 1;
  }
  for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
    result.metrics[prefix] = totals[prefix] > 0.0 ?
      result.metrics[prefix] / totals[prefix] :
      std::numeric_limits<double>::quiet_NaN();
  }
  if (store_scores) {
    result.q2.resize(prefix_count,
      std::numeric_limits<double>::quiet_NaN());
    if (q2_tss > 0.0L) {
      for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
        result.q2[prefix] = 1.0 -
          static_cast<double>(q2_press[prefix] / q2_tss);
      }
    }
  }
  result.best_index = cv_detail::best_metric_index(result.metrics, false);
  result.best_component = components[result.best_index];
  return result;
}

template<class T, class Backend>
RegressionCvResult<T> cross_validate_regression(
    ConstMatrixView<T> predictors, ConstMatrixView<T> responses,
    const int* folds, const int* components, std::size_t prefix_count,
    PredictorScaling scaling, LinearPlsFamily family,
    RegressionMetric metric, const PlssvdControls& plssvd_controls,
    const SimplsControls& simpls_controls, Backend& backend,
    bool store_predictions, std::size_t orthogonal_components = 0,
    const KernelCvControls& kernel_controls = KernelCvControls(),
    bool calculate_training_r2 = false) {
  if (predictors.empty() || responses.empty() ||
      predictors.rows() != responses.rows() || components == nullptr ||
      prefix_count < 1) {
    throw std::invalid_argument("regression cross-validation inputs are invalid");
  }
  auto partitions = cv_detail::fold_partitions(
    folds, predictors.rows(), false
  );
  const std::size_t maximum_component = static_cast<std::size_t>(
    *std::max_element(components, components + prefix_count)
  );
  RegressionCvResult<T> result;
  result.folds.assign(folds, folds + predictors.rows());
  result.status.assign(partitions.size(), 0);
  result.metrics.assign(prefix_count, 0.0);
  result.q2.assign(prefix_count, 0.0);
  result.rmsd.assign(prefix_count, 0.0);
  result.observed_r2.assign(prefix_count, 0.0);
  std::vector<double> counts(prefix_count, 0.0);
  if (store_predictions) {
    result.predictions.reserve(prefix_count);
    for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
      result.predictions.emplace_back(predictors.rows(), responses.columns());
    }
  }
  if (calculate_training_r2) {
    result.fold_training_r2.resize(partitions.size(), prefix_count);
    std::fill_n(
      result.fold_training_r2.data(), result.fold_training_r2.size(),
      std::numeric_limits<double>::quiet_NaN()
    );
  }
  long double observed_total_ss = 0.0L;
  for (std::size_t column = 0; column < responses.columns(); ++column) {
    long double total_sum = 0.0L;
    long double total_square = 0.0L;
    for (std::size_t row = 0; row < responses.rows(); ++row) {
      const long double value = responses(row, column);
      total_sum += value;
      total_square += value * value;
    }
    observed_total_ss += total_square -
      total_sum * total_sum / static_cast<long double>(responses.rows());
  }
  long double fold_training_total_ss = 0.0L;
  const bool complete_fold_cover = std::none_of(
    folds, folds + predictors.rows(), [](int value) { return value == 0; }
  );
  const bool reuse_dense_statistics =
    complete_fold_cover &&
    (family == LinearPlsFamily::plssvd ||
     family == LinearPlsFamily::simpls ||
     family == LinearPlsFamily::opls) &&
    cv_detail::fold_dense_statistics_enabled<T>(
      predictors.columns(), responses.columns()
    );
  const long double crosscovariance_bytes =
    static_cast<long double>(predictors.columns()) *
    static_cast<long double>(responses.columns()) * sizeof(T);
  const bool use_implicit_crosscovariance =
    (family == LinearPlsFamily::plssvd ||
     family == LinearPlsFamily::simpls ||
     family == LinearPlsFamily::opls) &&
    crosscovariance_bytes > 512.0L * 1024.0L * 1024.0L;
  const bool reuse_sample_response_gram =
    complete_fold_cover &&
    (family == LinearPlsFamily::plssvd ||
     family == LinearPlsFamily::simpls ||
     family == LinearPlsFamily::opls) &&
    use_implicit_crosscovariance &&
    (cv_detail::fold_sample_response_gram_enabled<T>(
       predictors.rows(), responses.columns(), maximum_component,
       simpls_controls.rsvd.power, partitions.size()
     ) ||
     (family == LinearPlsFamily::opls &&
      cv_detail::fold_sample_response_gram_storage_enabled<T>(
        predictors.rows(), responses.columns()
      )));
  const auto full_dense_statistics = reuse_dense_statistics ?
    cv_detail::dense_sufficient_statistics(
      predictors, responses, backend
    ) : cv_detail::DenseSufficientStatistics<T>();
  const auto full_dense_marginals =
    complete_fold_cover && use_implicit_crosscovariance &&
      !reuse_dense_statistics ?
      cv_detail::dense_marginal_statistics(predictors, responses) :
      cv_detail::DenseSufficientStatistics<T>();
  const bool reuse_predictor_gram =
    complete_fold_cover &&
    (family == LinearPlsFamily::plssvd ||
     family == LinearPlsFamily::simpls ||
     family == LinearPlsFamily::opls) &&
    cv_detail::fold_predictor_gram_enabled<T>(predictors.columns()) &&
    ((family == LinearPlsFamily::opls &&
      cv_detail::fold_opls_moments_enabled(
        predictors.rows(), predictors.columns(), partitions.size(), false,
        backend
      )) || simpls_controls.cache_predictor_crossprod ||
     cv_detail::fold_simpls_moments_enabled(
       predictors.rows(), predictors.columns(), maximum_component
     ));
  Matrix<T> full_predictor_gram;
  if (reuse_predictor_gram) {
    full_predictor_gram = cv_detail::predictor_gram(predictors, backend);
  }
  const auto full_response_gram = reuse_sample_response_gram ?
    cv_detail::response_gram(responses, backend) : Matrix<T>();
  const auto full_response_gram_row_sums = reuse_sample_response_gram ?
    cv_detail::symmetric_row_sums<T>(full_response_gram.view()) :
    std::vector<long double>();
  const bool moments_only_cv = !calculate_training_r2 &&
    reuse_dense_statistics && reuse_predictor_gram &&
    (family == LinearPlsFamily::plssvd ||
     family == LinearPlsFamily::simpls ||
     family == LinearPlsFamily::opls);
  const bool needs_training_rows = !moments_only_cv ||
    ((family == LinearPlsFamily::simpls ||
      family == LinearPlsFamily::opls) && use_implicit_crosscovariance);
  if (needs_training_rows) {
    cv_detail::retain_training_rows(partitions, folds, predictors.rows());
  }
  SimplsWorkspace<T> shared_simpls_workspace;
  OperatorRsvdWorkspace<T> shared_operator_workspace;
  Matrix<T> fold_response_gram;
  std::vector<long double> fold_response_row_means;
  Matrix<T> heldout_predictor_gram;
  Matrix<T> train;
  Matrix<T> test;
  PaddedMatrix<T> train_response;
  Matrix<T> test_response;

  for (std::size_t fold = 0; fold < partitions.size(); ++fold) {
    const auto& partition = partitions[fold];
    if (partition.test.empty()) {
      result.status[fold] = 2;
      continue;
    }
    cv_detail::configure_backend_problem(
      backend, partition.training_size, predictors.columns(),
      responses.columns(), 0
    );
    const bool moments_only = moments_only_cv;
    const bool moments_only_plssvd = moments_only &&
      family == LinearPlsFamily::plssvd;
    const bool moments_only_simpls = moments_only &&
      family == LinearPlsFamily::simpls;
    const bool moments_only_opls = moments_only &&
      family == LinearPlsFamily::opls;
    if (moments_only) {
      train.resize(0, 0);
    } else {
      cv_detail::gather_rows(predictors, partition.train, train);
    }
    cv_detail::gather_rows(predictors, partition.test, test);
    const bool needs_fold_predictor_gram = reuse_predictor_gram &&
      (family == LinearPlsFamily::simpls || moments_only);
    if (needs_fold_predictor_gram) {
      cv_detail::predictor_gram(
        ConstMatrixView<T>(test.view()), backend, heldout_predictor_gram
      );
    }
    if (!moments_only) {
      cv_detail::gather_rows_padded(
        responses, partition.train,
        cv_detail::response_row_alignment<T>(responses.columns()),
        train_response
      );
    }
    if (reuse_dense_statistics) {
      cv_detail::gather_rows(responses, partition.test, test_response);
    }
    DensePreprocessingResult<T> prepared;
    if (moments_only_opls) {
      prepared = cv_detail::prepare_dense_fold_from_statistics<T>(
        train.view(), test.view(), test_response.view(),
        full_dense_statistics, scaling, backend, partition.training_size
      );
    } else if (family == LinearPlsFamily::opls &&
               use_implicit_crosscovariance) {
      if (orthogonal_components < 1) {
        throw std::invalid_argument(
          "OPLS cross-validation requires an orthogonal component"
        );
      }
      if (complete_fold_cover) {
        prepared = cv_detail::prepare_dense_fold_from_marginals<T>(
          train.view(), test.view(), predictors, responses, partition.test,
          full_dense_marginals, scaling
        );
      } else {
        prepared = prepare_scaled_dense_operator(
          train.view(), ConstMatrixView<T>(train_response.view()), scaling,
          backend
        );
        cv_detail::standardize(
          test.view(), prepared.predictor_center, prepared.predictor_scale
        );
      }
      if (reuse_sample_response_gram) {
        cv_detail::prepare_centered_training_response_gram<T>(
          full_response_gram.view(), partition.train, partition.test,
          full_response_gram_row_sums, fold_response_gram,
          fold_response_row_means
        );
      }
      RsvdControls opls_controls = simpls_controls.rsvd;
      opls_controls.seed += static_cast<unsigned int>(fold);
      auto filter = fit_opls_filter_operator<T>(
        std::move(train), train_response.view(),
        prepared.response_mean.data(), prepared.response_mean.size(),
        orthogonal_components, opls_controls, backend,
        fold_response_gram.view(), shared_operator_workspace
      );
      if (filter.completed_components != orthogonal_components) {
        throw std::runtime_error(
          "OPLS cross-validation could not complete orthogonal filtering"
        );
      }
      apply_opls_deflation_inplace<T>(
        test.view(), filter.weights.view(), filter.loadings.view(), backend
      );
      train = std::move(filter.predictors);
    } else if (family == LinearPlsFamily::opls) {
      if (orthogonal_components < 1) {
        throw std::invalid_argument(
          "OPLS cross-validation requires an orthogonal component"
        );
      }
      RsvdControls opls_controls = simpls_controls.rsvd;
      opls_controls.seed += static_cast<unsigned int>(fold);
      auto filter = fit_opls_filter_rsvd<T>(
        std::move(train), train_response.view(), orthogonal_components,
        scaling, opls_controls, backend
      );
      test = apply_opls_filter<T>(
        std::move(test), filter.predictor_center.data(),
        filter.predictor_scale.data(), filter.predictor_center.size(),
        filter.weights.view(), filter.loadings.view(), backend
      );
      train = std::move(filter.predictors);
      prepared = prepare_scaled_dense_crossprod(
        train.view(), train_response.view(), PredictorScaling::none, backend
      );
    } else if (family == LinearPlsFamily::kernelpls) {
      auto input_preprocessing = prepare_scaled_dense_operator(
        train.view(), ConstMatrixView<T>(train_response.view()), scaling,
        backend
      );
      cv_detail::standardize(
        test.view(), input_preprocessing.predictor_center,
        input_preprocessing.predictor_scale
      );
      cv_detail::make_kernel_fold(
        train, test, kernel_controls, backend
      );
      prepared = prepare_scaled_dense_crossprod(
        train.view(), train_response.view(), PredictorScaling::none, backend
      );
    } else if (use_implicit_crosscovariance && complete_fold_cover) {
      prepared = cv_detail::prepare_dense_fold_from_marginals<T>(
        train.view(), test.view(), predictors, responses, partition.test,
        full_dense_marginals, scaling
      );
    } else if (use_implicit_crosscovariance) {
      prepared = prepare_scaled_dense_operator(
        train.view(), ConstMatrixView<T>(train_response.view()), scaling,
        backend
      );
      cv_detail::standardize(
        test.view(), prepared.predictor_center, prepared.predictor_scale
      );
    } else if (reuse_dense_statistics) {
      prepared = cv_detail::prepare_dense_fold_from_statistics<T>(
        train.view(), test.view(), ConstMatrixView<T>(test_response.view()),
        full_dense_statistics, scaling, backend,
        moments_only ? partition.training_size : 0
      );
    } else {
      prepared = prepare_scaled_dense_crossprod(
        train.view(), train_response.view(), scaling, backend
      );
      cv_detail::standardize(
        test.view(), prepared.predictor_center, prepared.predictor_scale
      );
    }
    for (std::size_t column = 0; column < responses.columns(); ++column) {
      const long double training_mean = prepared.response_mean[column];
      for (const std::size_t row : partition.test) {
        const long double centered = responses(row, column) - training_mean;
        fold_training_total_ss += centered * centered;
      }
    }

    if (needs_fold_predictor_gram) {
      cv_detail::preload_standardized_predictor_gram<T>(
        full_predictor_gram.view(), heldout_predictor_gram.view(),
        prepared.predictor_center, prepared.predictor_scale,
        partition.training_size, shared_simpls_workspace
      );
    }
    if (moments_only_opls) {
      RsvdControls opls_controls = simpls_controls.rsvd;
      opls_controls.seed += static_cast<unsigned int>(fold);
      auto filter = fit_opls_filter_from_moments_rsvd<T>(
        shared_simpls_workspace.predictor_crossprod.view(),
        prepared.crossprod.view(), orthogonal_components,
        prepared.predictor_center, prepared.predictor_scale, opls_controls,
        backend
      );
      if (filter.filter.completed_components != orthogonal_components) {
        throw std::runtime_error(
          "OPLS cross-validation could not complete orthogonal filtering"
        );
      }
      apply_opls_deflation_inplace<T>(
        test.view(), filter.filter.weights.view(),
        filter.filter.loadings.view(), backend
      );
      prepared.crossprod = std::move(filter.crosscov);
      shared_simpls_workspace.predictor_crossprod =
        std::move(filter.predictor_gram);
      shared_simpls_workspace.predictor_crossprod_preloaded = true;
    }

    if (family == LinearPlsFamily::plssvd) {
      PlssvdControls controls = plssvd_controls;
      controls.rsvd.seed += static_cast<unsigned int>(fold);
      PlssvdModel<T> model;
      if (moments_only_plssvd) {
        model = fit_plssvd_from_moments<T>(
          shared_simpls_workspace.predictor_crossprod.view(),
          prepared.crossprod.view(), components, prefix_count,
          controls, backend
        );
      } else if (use_implicit_crosscovariance) {
        if (reuse_sample_response_gram &&
            family != LinearPlsFamily::opls) {
          cv_detail::prepare_centered_training_response_gram<T>(
            full_response_gram.view(), partition.train, partition.test,
            full_response_gram_row_sums, fold_response_gram,
            fold_response_row_means
          );
        }
        CenteredCrosscovOperator<T, Backend> crosscovariance(
          train.view(), train_response.view(), prepared.response_mean.data(),
          prepared.response_mean.size(), backend
        );
        model = fit_plssvd_operator<T>(
          train.view(), crosscovariance, components, prefix_count, controls,
          backend, shared_operator_workspace,
          reuse_sample_response_gram ?
            ConstMatrixView<T>(fold_response_gram.view()) :
            ConstMatrixView<T>()
        );
      } else {
        model = fit_plssvd_preprocessed<T>(
          train.view(), prepared.crossprod.view(), components, prefix_count,
          controls, backend
        );
      }
      Matrix<T> test_scores = cv_detail::project_scores<T>(
        ConstMatrixView<T>(test.view()),
        ConstMatrixView<T>(model.weights.view()),
        model.completed_components, backend
      );
      Matrix<T> training_scores;
      if (calculate_training_r2) {
        training_scores = cv_detail::project_scores<T>(
          ConstMatrixView<T>(train.view()),
          ConstMatrixView<T>(model.weights.view()),
          model.completed_components, backend
        );
      }
      for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
        const auto prediction = cv_detail::predict_plssvd_from_scores<T>(
          model, test_scores.view(), prefix,
          prepared.response_mean, backend
        );
        Matrix<T>* stored = store_predictions ?
          &result.predictions[prefix] : nullptr;
        result.metrics[prefix] += cv_detail::accumulate_regression_error<T>(
          stored, partition.test, ConstMatrixView<T>(prediction.view()),
          responses
        );
        counts[prefix] += static_cast<double>(prediction.size());
        if (calculate_training_r2) {
          const auto fitted = cv_detail::predict_plssvd_from_scores<T>(
            model, training_scores.view(), prefix,
            prepared.response_mean, backend
          );
          result.fold_training_r2(fold, prefix) = cv_detail::regression_r2<T>(
            train_response.view(), fitted.view()
          );
        }
      }
    } else if (family == LinearPlsFamily::simpls ||
               family == LinearPlsFamily::opls ||
               family == LinearPlsFamily::kernelpls) {
      SimplsControls controls = simpls_controls;
      controls.rsvd.seed += static_cast<unsigned int>(fold);
      controls.cache_predictor_crossprod = reuse_predictor_gram;
      controls.store_scores = calculate_training_r2;
      SimplsModel<T> model;
      if (use_implicit_crosscovariance) {
        controls.rank_one_operator_direction = controls.maximum_block == 1;
        controls.reorthogonalize = true;
        CenteredCrosscovOperator<T, Backend> initial_crosscovariance(
          train.view(), train_response.view(), prepared.response_mean.data(),
          prepared.response_mean.size(), backend
        );
        ProjectedOperator<
          T, CenteredCrosscovOperator<T, Backend>, Backend
        > projected_crosscovariance(
          initial_crosscovariance, controls.components, backend
        );
        if (reuse_sample_response_gram &&
            family != LinearPlsFamily::opls) {
          cv_detail::prepare_centered_training_response_gram<T>(
            full_response_gram.view(), partition.train, partition.test,
            full_response_gram_row_sums, fold_response_gram,
            fold_response_row_means
          );
        }
        model = fit_simpls_operator<T>(
          train.view(), initial_crosscovariance, projected_crosscovariance,
          controls, backend, shared_simpls_workspace,
          shared_operator_workspace,
          reuse_sample_response_gram ?
            ConstMatrixView<T>(fold_response_gram.view()) :
            ConstMatrixView<T>()
        );
      } else {
        model = fit_simpls_preprocessed<T>(
          (moments_only_simpls || moments_only_opls) ?
            ConstMatrixView<T>() : ConstMatrixView<T>(train.view()),
          prepared.crossprod.view(), controls, backend,
          shared_simpls_workspace,
          (moments_only_simpls || moments_only_opls) ?
            partition.training_size : 0
        );
      }
      Matrix<T> test_scores = cv_detail::project_scores<T>(
        ConstMatrixView<T>(test.view()),
        ConstMatrixView<T>(model.weights.view()),
        model.completed_components, backend
      );
      Matrix<T> prediction = cv_detail::initialize_simpls_prediction<T>(
        test_scores.rows(), prepared.response_mean
      );
      Matrix<T> prediction_contribution;
      std::size_t previous_component = 0;
      Matrix<T> training_prediction = calculate_training_r2 ?
        cv_detail::initialize_simpls_prediction<T>(
          model.scores.rows(), prepared.response_mean
        ) : Matrix<T>();
      Matrix<T> training_prediction_contribution;
      std::size_t previous_training_component = 0;
      for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
        const std::size_t requested = static_cast<std::size_t>(
          components[prefix]
        );
        cv_detail::update_simpls_prediction_from_scores<T>(
          model, test_scores.view(), previous_component, requested,
          prediction.view(), prediction_contribution, backend
        );
        previous_component = requested;
        Matrix<T>* stored = store_predictions ?
          &result.predictions[prefix] : nullptr;
        result.metrics[prefix] += cv_detail::accumulate_regression_error<T>(
          stored, partition.test, ConstMatrixView<T>(prediction.view()),
          responses
        );
        counts[prefix] += static_cast<double>(prediction.size());
        if (calculate_training_r2) {
          cv_detail::update_simpls_prediction_from_scores<T>(
            model, model.scores.view(), previous_training_component,
            requested, training_prediction.view(),
            training_prediction_contribution, backend
          );
          previous_training_component = requested;
          result.fold_training_r2(fold, prefix) = cv_detail::regression_r2<T>(
            train_response.view(), training_prediction.view()
          );
        }
      }
    } else {
      throw std::invalid_argument(
        "regression CV received an unsupported PLS family"
      );
    }
    result.status[fold] = 1;
  }

  for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
    const double sse = result.metrics[prefix];
    result.rmsd[prefix] = counts[prefix] > 0.0 ?
      std::sqrt(sse / counts[prefix]) :
      std::numeric_limits<double>::quiet_NaN();
    result.q2[prefix] = fold_training_total_ss > 0.0L ?
      1.0 - sse / static_cast<double>(fold_training_total_ss) :
      std::numeric_limits<double>::quiet_NaN();
    result.observed_r2[prefix] = observed_total_ss > 0.0L ?
      1.0 - sse / static_cast<double>(observed_total_ss) :
      std::numeric_limits<double>::quiet_NaN();
    result.metrics[prefix] = metric == RegressionMetric::rmsd ?
      result.rmsd[prefix] : metric == RegressionMetric::q2 ?
      result.q2[prefix] : result.observed_r2[prefix];
  }
  if (store_predictions) {
    const auto metric_reference = regression_metrics_reference<T>(responses);
    result.evaluation.reserve(prefix_count);
    for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
      result.evaluation.push_back(regression_metrics<T>(
        responses, ConstMatrixView<T>(result.predictions[prefix].view()),
        result.q2[prefix], metric_reference
      ));
    }
  }
  result.best_index = cv_detail::best_metric_index(
    result.metrics, metric == RegressionMetric::rmsd
  );
  result.best_component = components[result.best_index];
  return result;
}

}  // namespace core
}  // namespace fastpls

#endif
