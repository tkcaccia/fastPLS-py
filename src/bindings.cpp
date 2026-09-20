// SPDX-License-Identifier: MIT
#include "native_backend.hpp"

#include <fastpls/core.hpp>
#include <fastpls/core/cross_validation.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;
namespace core = fastpls::core;

namespace fastpls_py {
namespace {

enum class Family { simpls, plssvd, opls, kernelpls };
enum class Head { regression, argmax, lda };

Family parse_family(const std::string& value) {
  if (value == "simpls") return Family::simpls;
  if (value == "plssvd" || value == "pls-svd") return Family::plssvd;
  if (value == "opls") return Family::opls;
  if (value == "kernelpls" || value == "kernel-pls") return Family::kernelpls;
  throw std::invalid_argument(
      "method must be one of: simpls, plssvd, opls, kernelpls");
}

Head parse_head(const std::string& value) {
  if (value == "regression") return Head::regression;
  if (value == "argmax") return Head::argmax;
  if (value == "lda") return Head::lda;
  throw std::invalid_argument("classifier must be None, 'argmax', or 'lda'");
}

core::LinearPlsFamily cv_family(Family family) {
  if (family == Family::plssvd) return core::LinearPlsFamily::plssvd;
  if (family == Family::simpls) return core::LinearPlsFamily::simpls;
  if (family == Family::opls) return core::LinearPlsFamily::opls;
  return core::LinearPlsFamily::kernelpls;
}

core::PredictorScaling parse_scaling(const std::string& value) {
  if (value == "center" || value == "centering") {
    return core::PredictorScaling::centering;
  }
  if (value == "scale" || value == "autoscaling") {
    return core::PredictorScaling::autoscaling;
  }
  if (value == "none") return core::PredictorScaling::none;
  throw std::invalid_argument(
      "scaling must be one of: autoscaling, centering, none");
}

core::KernelType parse_kernel(const std::string& value) {
  if (value == "linear") return core::KernelType::linear;
  if (value == "rbf" || value == "radial_basis") {
    return core::KernelType::radial_basis;
  }
  if (value == "polynomial" || value == "poly") {
    return core::KernelType::polynomial;
  }
  throw std::invalid_argument("kernel must be one of: linear, rbf, polynomial");
}

template<class T>
core::Matrix<T> matrix_from_numpy(
    const py::array_t<T, py::array::forcecast>& array) {
  if (array.ndim() != 2) throw std::invalid_argument("matrix must be two-dimensional");
  const auto view = array.template unchecked<2>();
  core::Matrix<T> output(
      static_cast<std::size_t>(view.shape(0)),
      static_cast<std::size_t>(view.shape(1)));
  const auto information = array.request();
  if (information.strides[0] == static_cast<py::ssize_t>(sizeof(T)) &&
      information.strides[1] ==
          static_cast<py::ssize_t>(sizeof(T) * output.rows())) {
    std::memcpy(output.data(), information.ptr, output.size() * sizeof(T));
    return output;
  }
  for (py::ssize_t column = 0; column < view.shape(1); ++column) {
    for (py::ssize_t row = 0; row < view.shape(0); ++row) {
      output(static_cast<std::size_t>(row), static_cast<std::size_t>(column)) =
          view(row, column);
    }
  }
  return output;
}

template<class T>
py::array_t<T> matrix_to_numpy(const core::Matrix<T>& matrix) {
  py::array_t<T> output(
      {static_cast<py::ssize_t>(matrix.rows()),
       static_cast<py::ssize_t>(matrix.columns())},
      {static_cast<py::ssize_t>(sizeof(T)),
       static_cast<py::ssize_t>(sizeof(T) * matrix.rows())});
  std::memcpy(output.mutable_data(), matrix.data(), matrix.size() * sizeof(T));
  return output;
}

template<class T>
void standardize(core::Matrix<T>& matrix, const std::vector<T>& center,
                 const std::vector<T>& scale) {
  if (matrix.columns() != center.size() || center.size() != scale.size()) {
    throw std::invalid_argument("prediction columns do not match training");
  }
  for (std::size_t column = 0; column < matrix.columns(); ++column) {
    if (!(scale[column] > T(0))) throw std::runtime_error("invalid predictor scale");
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
      matrix(row, column) = (matrix(row, column) - center[column]) / scale[column];
    }
  }
}

template<class T>
void standardize_predictor_gram(
    core::MatrixView<T> gram, std::size_t sample_count,
    const std::vector<T>& center, const std::vector<T>& scale,
    core::PredictorScaling scaling) {
  if (scaling == core::PredictorScaling::none) return;
  if (gram.rows() != gram.columns() || gram.rows() != center.size() ||
      center.size() != scale.size()) {
    throw std::invalid_argument("predictor Gram preprocessing dimensions are invalid");
  }
  const T samples = static_cast<T>(sample_count);
  for (std::size_t column = 0; column < gram.columns(); ++column) {
    for (std::size_t row = 0; row < gram.rows(); ++row) {
      gram(row, column) =
          (gram(row, column) - samples * center[row] * center[column]) /
          (scale[row] * scale[column]);
    }
  }
}

core::SimplsControls simpls_controls(
    std::size_t n, std::size_t p, std::size_t q, std::size_t components,
    bool classification, int oversample, int power, unsigned int seed) {
  core::SimplsControls controls;
  controls.components = components;
  controls.maximum_block = core::simpls_candidate_block_size(
      components, p, q, classification, n, 64);
#if defined(FASTPLS_PY_ACCELERATE)
  constexpr std::size_t maximum_predictors = 2048;
#else
  constexpr std::size_t maximum_predictors = 512;
#endif
  const bool component_work_justifies_cache =
      components >= 20 || p <= 5 * std::max<std::size_t>(components, 1);
  controls.cache_predictor_crossprod = component_work_justifies_cache &&
      p <= n && n >= p * 8 && p <= maximum_predictors;
  controls.batch_candidate_geometry =
      controls.maximum_block > 1 && !controls.cache_predictor_crossprod;
  controls.reorthogonalize = false;
  controls.store_scores = false;
  controls.store_score_moments = classification;
  controls.use_right_gram = true;
  controls.rsvd.oversample = oversample;
  controls.rsvd.power = power;
  controls.rsvd.seed = seed;
  return controls;
}

template<class T>
using ModelVariant = std::variant<
    core::SimplsModel<T>, core::PlssvdModel<T>,
    core::OplsModel<T>, core::KernelPlsModel<T>>;

template<class T>
std::size_t fitted_component_count(
    Family family, const ModelVariant<T>& model) {
  if (family == Family::simpls) {
    return std::get<core::SimplsModel<T>>(model).completed_components;
  }
  if (family == Family::plssvd) {
    return std::get<core::PlssvdModel<T>>(model).completed_components;
  }
  if (family == Family::opls) {
    return std::get<core::OplsModel<T>>(model).inner.completed_components;
  }
  return std::get<core::KernelPlsModel<T>>(model).inner.completed_components;
}

class ModelBase {
 public:
  virtual ~ModelBase() = default;
  virtual py::array predict(py::array input) const = 0;
  virtual py::array predict_scores(py::array input) const = 0;
  virtual py::array_t<std::int64_t> predict_classes(
      py::array input, std::size_t top) const = 0;
  virtual py::array scores() const = 0;
  virtual py::object vip() const = 0;
  virtual std::size_t components() const = 0;
  virtual std::string dtype() const = 0;
  virtual std::string method() const = 0;
  virtual std::string head() const = 0;
};

template<class T>
class Model final : public ModelBase {
 public:
  Model(Family family, Head head, std::size_t components,
        std::size_t classes, ModelVariant<T> model,
        std::vector<T> predictor_center,
        std::vector<T> predictor_scale,
        std::vector<T> response_mean,
        core::LdaModel<T> lda,
        core::Matrix<T> direct_class_weights,
        std::vector<T> direct_class_offsets)
      : family_(family), head_(head), components_(components), classes_(classes),
        model_(std::move(model)), predictor_center_(std::move(predictor_center)),
        predictor_scale_(std::move(predictor_scale)),
        response_mean_(std::move(response_mean)), lda_(std::move(lda)),
        direct_class_weights_(std::move(direct_class_weights)),
        direct_class_offsets_(std::move(direct_class_offsets)) {}

  py::array predict(py::array input) const override {
    auto x = matrix_from_numpy<T>(
        py::array_t<T, py::array::forcecast>::ensure(input));
    core::Matrix<T> values;
    {
      py::gil_scoped_release release;
      values = predict_response(std::move(x));
    }
    return matrix_to_numpy(values);
  }

  py::array predict_scores(py::array input) const override {
    auto x = matrix_from_numpy<T>(
        py::array_t<T, py::array::forcecast>::ensure(input));
    core::Matrix<T> values;
    {
      py::gil_scoped_release release;
      values = latent_scores(std::move(x));
    }
    return matrix_to_numpy(values);
  }

  py::array_t<std::int64_t> predict_classes(
      py::array input, std::size_t top) const override {
    if (head_ == Head::regression) {
      throw std::invalid_argument("class prediction is unavailable for regression");
    }
    if (top < 1 || top > classes_) {
      throw std::invalid_argument("top exceeds the number of classes");
    }
    auto x = matrix_from_numpy<T>(
        py::array_t<T, py::array::forcecast>::ensure(input));
    core::Matrix<T> class_scores;
    {
      py::gil_scoped_release release;
      if (direct_class_weights_.size() != 0) {
        class_scores.resize(x.rows(), classes_);
        NativeBackend<T> backend;
        backend.gemm(x.view(), direct_class_weights_.view(), false, false,
                     class_scores.view());
        for (std::size_t column = 0; column < classes_; ++column) {
          for (std::size_t row = 0; row < class_scores.rows(); ++row) {
            class_scores(row, column) += direct_class_offsets_[column];
          }
        }
      } else if (head_ == Head::lda) {
        class_scores = core::lda_scores(latent_scores(std::move(x)).view(), lda_);
      } else {
        class_scores = predict_response(std::move(x));
      }
    }
    py::array_t<std::int64_t> output({
        static_cast<py::ssize_t>(class_scores.rows()),
        static_cast<py::ssize_t>(top)});
    auto view = output.template mutable_unchecked<2>();
    std::vector<std::size_t> order(classes_);
    for (std::size_t row = 0; row < class_scores.rows(); ++row) {
      for (std::size_t index = 0; index < classes_; ++index) order[index] = index;
      std::partial_sort(order.begin(), order.begin() + top, order.end(),
                        [&](std::size_t left, std::size_t right) {
        return class_scores(row, left) > class_scores(row, right);
      });
      for (std::size_t rank = 0; rank < top; ++rank) {
        view(static_cast<py::ssize_t>(row), static_cast<py::ssize_t>(rank)) =
            static_cast<std::int64_t>(order[rank]);
      }
    }
    return output;
  }

  py::array scores() const override {
    return matrix_to_numpy(training_scores());
  }

  py::object vip() const override {
    const auto& scores = training_scores();
    const core::Matrix<T>* weights = nullptr;
    const core::Matrix<T>* loadings = nullptr;
    if (family_ == Family::simpls) {
      const auto& model = std::get<core::SimplsModel<T>>(model_);
      weights = &model.weights;
      loadings = &model.response_loadings;
    } else if (family_ == Family::plssvd) {
      const auto& model = std::get<core::PlssvdModel<T>>(model_);
      weights = &model.weights;
      loadings = &model.response_loadings;
    } else if (family_ == Family::opls) {
      const auto& model = std::get<core::OplsModel<T>>(model_).inner;
      weights = &model.weights;
      loadings = &model.response_loadings;
    } else {
      const auto& model = std::get<core::KernelPlsModel<T>>(model_).inner;
      weights = &model.weights;
      loadings = &model.response_loadings;
    }
    if (scores.size() == 0) {
      throw std::invalid_argument(
          "VIP requires a model fitted with store_scores=True");
    }
    if (scores.columns() != weights->columns() ||
        loadings->columns() != weights->columns()) {
      throw std::runtime_error("VIP model component dimensions do not match");
    }
    const std::size_t components = weights->columns();
    const std::size_t predictors = weights->rows();
    py::list output;
    for (std::size_t response = 0; response < loadings->rows(); ++response) {
      py::array_t<double> value({
          static_cast<py::ssize_t>(components),
          static_cast<py::ssize_t>(predictors)});
      auto destination = value.template mutable_unchecked<2>();
      std::vector<long double> numerator(predictors, 0.0L);
      long double denominator = 0.0L;
      for (std::size_t component = 0; component < components; ++component) {
        long double score_square = 0.0L;
        long double weight_square = 0.0L;
        for (std::size_t row = 0; row < scores.rows(); ++row) {
          const long double current = scores(row, component);
          score_square += current * current;
        }
        for (std::size_t row = 0; row < predictors; ++row) {
          const long double current = (*weights)(row, component);
          weight_square += current * current;
        }
        const long double loading = (*loadings)(response, component);
        const long double explained = loading * loading * score_square;
        denominator += explained;
        for (std::size_t predictor = 0; predictor < predictors; ++predictor) {
          const long double weight = (*weights)(predictor, component);
          if (weight_square > 0.0L) {
            numerator[predictor] +=
                weight * weight * explained / weight_square;
          }
          destination(component, predictor) = denominator > 0.0L ?
              std::sqrt(static_cast<double>(
                  predictors * numerator[predictor] / denominator)) :
              std::numeric_limits<double>::quiet_NaN();
        }
      }
      output.append(std::move(value));
    }
    return loadings->rows() == 1 ? output[0] : py::object(output);
  }

  std::size_t components() const override { return components_; }
  std::string dtype() const override {
    return std::is_same<T, float>::value ? "float32" : "float64";
  }
  std::string method() const override {
    switch (family_) {
      case Family::simpls: return "simpls";
      case Family::plssvd: return "plssvd";
      case Family::opls: return "opls";
      case Family::kernelpls: return "kernelpls";
    }
    return "unknown";
  }
  std::string head() const override {
    if (head_ == Head::argmax) return "argmax";
    if (head_ == Head::lda) return "lda";
    return "regression";
  }

 private:
  core::Matrix<T> latent_scores(core::Matrix<T> predictors) const {
    if (components_ == 0) {
      return core::Matrix<T>(predictors.rows(), 0);
    }
    NativeBackend<T> backend;
    if (family_ == Family::simpls || family_ == Family::plssvd) {
      standardize(predictors, predictor_center_, predictor_scale_);
      const auto& weights = family_ == Family::simpls ?
          std::get<core::SimplsModel<T>>(model_).weights :
          std::get<core::PlssvdModel<T>>(model_).weights;
      core::ConstMatrixView<T> prefix(
          weights.data(), weights.rows(), components_, weights.rows());
      core::Matrix<T> output(predictors.rows(), components_);
      backend.gemm(predictors.view(), prefix, false, false, output.view());
      return output;
    }
    if (family_ == Family::opls) {
      const auto& model = std::get<core::OplsModel<T>>(model_);
      auto filtered = core::apply_opls_filter(
          std::move(predictors), model.filter.predictor_center.data(),
          model.filter.predictor_scale.data(),
          model.filter.predictor_center.size(), model.filter.weights.view(),
          model.filter.loadings.view(), backend);
      core::Matrix<T> output(filtered.rows(), components_);
      core::ConstMatrixView<T> weights(
          model.inner.weights.data(), model.inner.weights.rows(), components_,
          model.inner.weights.rows());
      backend.gemm(filtered.view(), weights, false, false, output.view());
      return output;
    }
    const auto& model = std::get<core::KernelPlsModel<T>>(model_);
    core::kernelpls_detail::standardize(
        predictors.view(), model.predictor_center, model.predictor_scale);
    core::Matrix<T> design;
    if (model.kernel == core::KernelType::linear) {
      design = std::move(predictors);
    } else {
      design = core::kernel_matrix<T>(
          core::ConstMatrixView<T>(predictors.view()), model.reference.view(),
          model.kernel, model.gamma,
          model.degree, model.offset, backend);
      core::center_kernel_test(
          design.view(), model.kernel_column_means.data(),
          model.kernel_column_means.size(), model.kernel_grand_mean);
    }
    core::Matrix<T> output(design.rows(), components_);
    core::ConstMatrixView<T> weights(
        model.inner.weights.data(), model.inner.weights.rows(), components_,
        model.inner.weights.rows());
    backend.gemm(design.view(), weights, false, false, output.view());
    return output;
  }

  core::Matrix<T> predict_response(core::Matrix<T> predictors) const {
    NativeBackend<T> backend;
    if (family_ == Family::simpls) {
      if (components_ == 0) {
        core::Matrix<T> output(predictors.rows(), response_mean_.size());
        for (std::size_t column = 0; column < output.columns(); ++column) {
          for (std::size_t row = 0; row < output.rows(); ++row) {
            output(row, column) = response_mean_[column];
          }
        }
        return output;
      }
      standardize(predictors, predictor_center_, predictor_scale_);
      auto centered = core::predict_simpls_preprocessed<T>(
          core::ConstMatrixView<T>(predictors.view()),
          std::get<core::SimplsModel<T>>(model_),
          components_, backend);
      for (std::size_t column = 0; column < centered.columns(); ++column) {
        for (std::size_t row = 0; row < centered.rows(); ++row) {
          centered(row, column) += response_mean_[column];
        }
      }
      return centered;
    }
    if (family_ == Family::plssvd) {
      standardize(predictors, predictor_center_, predictor_scale_);
      const auto& model = std::get<core::PlssvdModel<T>>(model_);
      core::Matrix<T> scores(predictors.rows(), components_);
      core::ConstMatrixView<T> weights(
          model.weights.data(), model.weights.rows(), components_,
          model.weights.rows());
      backend.gemm(predictors.view(), weights, false, false, scores.view());
      core::Matrix<T> prediction(predictors.rows(), response_mean_.size());
      backend.gemm(scores.view(), model.prediction_weights.front().view(),
                   false, false, prediction.view());
      for (std::size_t column = 0; column < prediction.columns(); ++column) {
        for (std::size_t row = 0; row < prediction.rows(); ++row) {
          prediction(row, column) += response_mean_[column];
        }
      }
      return prediction;
    }
    if (family_ == Family::opls) {
      return core::predict_opls(
          std::get<core::OplsModel<T>>(model_), std::move(predictors),
          components_, backend);
    }
    return core::predict_kernelpls(
        std::get<core::KernelPlsModel<T>>(model_), std::move(predictors),
        components_, backend);
  }

  const core::Matrix<T>& training_scores() const {
    if (family_ == Family::simpls) return std::get<core::SimplsModel<T>>(model_).scores;
    if (family_ == Family::plssvd) return std::get<core::PlssvdModel<T>>(model_).scores;
    if (family_ == Family::opls) return std::get<core::OplsModel<T>>(model_).inner.scores;
    return std::get<core::KernelPlsModel<T>>(model_).inner.scores;
  }

  Family family_;
  Head head_;
  std::size_t components_;
  std::size_t classes_;
  ModelVariant<T> model_;
  std::vector<T> predictor_center_;
  std::vector<T> predictor_scale_;
  std::vector<T> response_mean_;
  core::LdaModel<T> lda_;
  core::Matrix<T> direct_class_weights_;
  std::vector<T> direct_class_offsets_;
};

template<class T>
std::shared_ptr<ModelBase> fit_typed(
    const py::array_t<T, py::array::forcecast>& x_array, py::handle y,
    int requested_components, const std::string& method,
    const std::string& classifier, const std::string& scaling,
    int oversample, int power, int seed, int orthogonal_components,
    const std::string& kernel, double gamma, int degree, double offset,
    bool store_scores) {
  if (requested_components < 1 || oversample < 0 || power < 0 || seed < 0) {
    throw std::invalid_argument("components and rSVD controls must be non-negative");
  }
  const Family family = parse_family(method);
  const Head head = parse_head(classifier);
  const auto scale = parse_scaling(scaling);
  NativeBackend<T> backend;
  auto predictors = matrix_from_numpy<T>(x_array);
  const std::size_t n = predictors.rows();
  const std::size_t p = predictors.columns();
  if (n < 2 || p == 0) throw std::invalid_argument("X must contain at least two rows");

  std::size_t classes = 0;
  std::vector<int> labels;
  core::Matrix<T> responses;
  if (head == Head::regression) {
    auto y_array = py::array_t<T, py::array::forcecast>::ensure(y);
    responses = matrix_from_numpy<T>(y_array);
    if (responses.rows() != n) throw std::invalid_argument("X and y row counts differ");
  } else {
    auto y_array = py::array_t<std::int64_t, py::array::forcecast>::ensure(y);
    if (!y_array || y_array.ndim() != 1 ||
        static_cast<std::size_t>(y_array.shape(0)) != n) {
      throw std::invalid_argument("classification y must be a one-dimensional label array");
    }
    auto values = y_array.template unchecked<1>();
    std::int64_t maximum = -1;
    labels.resize(n);
    for (std::size_t row = 0; row < n; ++row) {
      const auto value = values(static_cast<py::ssize_t>(row));
      if (value < 0) throw std::invalid_argument("class indices must be non-negative");
      maximum = std::max(maximum, value);
      labels[row] = static_cast<int>(value + 1);
    }
    classes = static_cast<std::size_t>(maximum + 1);
    if (classes < 2) throw std::invalid_argument("classification requires at least two classes");
    if (family == Family::opls || family == Family::kernelpls) {
      responses.resize(n, classes);
      for (std::size_t row = 0; row < n; ++row) {
        responses(row, static_cast<std::size_t>(labels[row] - 1)) = T(1);
      }
    }
  }

  std::size_t cap = std::min(p, std::max<std::size_t>(n - 1, 1));
  if (family == Family::plssvd) {
    cap = std::min(cap, head == Head::regression ? responses.columns() :
                   std::max<std::size_t>(classes - 1, 1));
  }
  const std::size_t components = std::min<std::size_t>(
      static_cast<std::size_t>(requested_components), cap);
  if (components == 0) throw std::invalid_argument("no component can be fitted");

  std::vector<T> center;
  std::vector<T> predictor_scale;
  std::vector<T> response_mean;
  core::Matrix<T> class_predictor_sums;
  std::vector<T> class_counts;
  ModelVariant<T> fitted;
  {
    py::gil_scoped_release release;
    auto simpls = simpls_controls(
        n, p, head == Head::regression ? responses.columns() : classes,
        components, head != Head::regression, oversample, power,
        static_cast<unsigned int>(seed));
    simpls.store_scores = store_scores ||
        (head == Head::lda &&
         (family == Family::opls || family == Family::kernelpls));
    if (family == Family::simpls || family == Family::plssvd) {
      if (head == Head::regression) {
        const auto prepared = core::prepare_scaled_dense_crossprod(
            predictors.view(), responses.view(), scale, backend);
        center = prepared.predictor_center;
        predictor_scale = prepared.predictor_scale;
        response_mean = prepared.response_mean;
        if (family == Family::simpls) {
          core::SimplsWorkspace<T> workspace;
          fitted = core::fit_simpls_preprocessed<T>(
              predictors.view(), prepared.crossprod.view(), simpls, backend,
              workspace);
        } else {
          const int component = static_cast<int>(components);
          core::PlssvdControls controls;
          controls.rsvd = simpls.rsvd;
          fitted = core::fit_plssvd_preprocessed<T>(
              predictors.view(), prepared.crossprod.view(), &component, 1,
              controls, backend);
        }
      } else {
        std::vector<std::size_t> zero_based(n);
        for (std::size_t row = 0; row < n; ++row) {
          zero_based[row] = static_cast<std::size_t>(labels[row] - 1);
        }
        const auto prepared = core::scaled_label_crossprod<T>(
            core::ConstMatrixView<T>(predictors.view()), zero_based.data(),
            zero_based.size(), classes, scale, backend);
        center = prepared.predictor_center;
        predictor_scale = prepared.predictor_scale;
        response_mean = prepared.response_mean;
        class_predictor_sums = prepared.class_predictor_sums;
        class_counts = prepared.class_counts;
        core::Matrix<T> predictor_gram(p, p);
        backend.self_gram(predictors.view(), true, predictor_gram.view(), true);
        standardize_predictor_gram(
            predictor_gram.view(), n, center, predictor_scale, scale);
        if (family == Family::simpls) {
          core::SimplsWorkspace<T> workspace;
          workspace.predictor_crossprod = std::move(predictor_gram);
          workspace.predictor_crossprod_preloaded = true;
          simpls.cache_predictor_crossprod = true;
          simpls.reorthogonalize = false;
          simpls.store_scores = false;
          simpls.store_score_moments = true;
          fitted = core::fit_simpls_preprocessed<T>(
              core::ConstMatrixView<T>(), prepared.crossprod.view(), simpls,
              backend, workspace, n);
        } else {
          const int component = static_cast<int>(components);
          core::PlssvdControls controls;
          controls.rsvd = simpls.rsvd;
          fitted = core::fit_plssvd_from_moments<T>(
              predictor_gram.view(), prepared.crossprod.view(), &component, 1,
              controls, backend);
        }
        if (store_scores) {
          standardize(predictors, center, predictor_scale);
          if (family == Family::simpls) {
            auto& model = std::get<core::SimplsModel<T>>(fitted);
            const std::size_t retained = model.completed_components;
            model.scores.resize(n, retained);
            if (retained > 0) {
              const core::ConstMatrixView<T> weights(
                  model.weights.data(), model.weights.rows(), retained,
                  model.weights.rows());
              backend.gemm(predictors.view(), weights, false, false,
                           model.scores.view());
            }
          } else {
            auto& model = std::get<core::PlssvdModel<T>>(fitted);
            const std::size_t retained = model.completed_components;
            model.scores.resize(n, retained);
            if (retained > 0) {
              const core::ConstMatrixView<T> weights(
                  model.weights.data(), model.weights.rows(), retained,
                  model.weights.rows());
              backend.gemm(predictors.view(), weights, false, false,
                           model.scores.view());
            }
          }
        }
      }
    } else if (family == Family::opls) {
      core::OplsControls controls;
      controls.orthogonal_components = static_cast<std::size_t>(orthogonal_components);
      controls.scaling = scale;
      controls.randomized_filter = true;
      controls.filter_rsvd = simpls.rsvd;
      controls.simpls = simpls;
      fitted = core::fit_opls(std::move(predictors), responses.view(), controls, backend);
    } else {
      core::KernelPlsControls controls;
      controls.kernel = parse_kernel(kernel);
      controls.gamma = gamma;
      controls.degree = degree;
      controls.offset = offset;
      controls.scaling = scale;
      controls.simpls = simpls;
      fitted = core::fit_kernelpls(
          std::move(predictors), responses.view(), controls, backend);
    }
  }

  const std::size_t fitted_components =
      fitted_component_count(family, fitted);

  core::LdaModel<T> lda;
  if (head == Head::lda && fitted_components > 0) {
    const core::Matrix<T>* score_matrix = nullptr;
    const core::Matrix<T>* score_gram = nullptr;
    if (family == Family::simpls) score_matrix = &std::get<core::SimplsModel<T>>(fitted).scores;
    if (family == Family::plssvd) score_matrix = &std::get<core::PlssvdModel<T>>(fitted).scores;
    if (family == Family::opls) score_matrix = &std::get<core::OplsModel<T>>(fitted).inner.scores;
    if (family == Family::kernelpls) score_matrix = &std::get<core::KernelPlsModel<T>>(fitted).inner.scores;
    if (family == Family::simpls) score_gram = &std::get<core::SimplsModel<T>>(fitted).score_gram;
    if (family == Family::plssvd) score_gram = &std::get<core::PlssvdModel<T>>(fitted).score_gram;
    if (family == Family::opls) score_gram = &std::get<core::OplsModel<T>>(fitted).inner.score_gram;
    if (family == Family::kernelpls) score_gram = &std::get<core::KernelPlsModel<T>>(fitted).inner.score_gram;
    const int component = static_cast<int>(fitted_components);
    core::Matrix<T> class_sums(classes, fitted_components);
    std::vector<T> counts(classes, T(0));
    if (class_predictor_sums.size() != 0 &&
        (family == Family::simpls || family == Family::plssvd)) {
      const auto& projection = family == Family::simpls ?
          std::get<core::SimplsModel<T>>(fitted).weights :
          std::get<core::PlssvdModel<T>>(fitted).weights;
      const core::ConstMatrixView<T> retained_projection(
          projection.data(), projection.rows(), fitted_components,
          projection.rows());
      backend.gemm(class_predictor_sums.view(), retained_projection,
                   true, false, class_sums.view());
      counts = class_counts;
    } else {
      for (std::size_t row = 0; row < n; ++row) {
        const std::size_t label = static_cast<std::size_t>(labels[row] - 1);
        counts[label] += T(1);
        for (std::size_t column = 0;
             column < fitted_components; ++column) {
          class_sums(label, column) += (*score_matrix)(row, column);
        }
      }
    }
    auto models = core::train_lda_prefixes_from_moments<T>(
        core::ConstMatrixView<T>(score_gram->view()),
        core::ConstMatrixView<T>(class_sums.view()),
        counts.data(), counts.size(),
        n, &component, 1, backend);
    lda = std::move(models.front());
  }

  core::Matrix<T> direct_class_weights;
  std::vector<T> direct_class_offsets;
  if (head != Head::regression &&
      (family == Family::simpls || family == Family::plssvd)) {
    const auto& projection = family == Family::simpls ?
        std::get<core::SimplsModel<T>>(fitted).weights :
        std::get<core::PlssvdModel<T>>(fitted).weights;
    core::Matrix<T> latent_weights(fitted_components, classes);
    if (fitted_components == 0) {
      direct_class_offsets.resize(classes);
      for (std::size_t class_index = 0;
           class_index < classes; ++class_index) {
        const T prior = response_mean[class_index];
        direct_class_offsets[class_index] = head == Head::lda ?
            std::log(std::max(prior, std::numeric_limits<T>::min())) : prior;
      }
    } else if (head == Head::lda) {
      for (std::size_t class_index = 0; class_index < classes; ++class_index) {
        for (std::size_t component_index = 0;
             component_index < fitted_components; ++component_index) {
          latent_weights(component_index, class_index) =
              lda.linear(class_index, component_index);
        }
      }
      direct_class_offsets = lda.constants;
    } else if (family == Family::simpls) {
      const auto& loadings =
          std::get<core::SimplsModel<T>>(fitted).response_loadings;
      for (std::size_t class_index = 0; class_index < classes; ++class_index) {
        for (std::size_t component_index = 0;
             component_index < fitted_components; ++component_index) {
          latent_weights(component_index, class_index) =
              loadings(class_index, component_index);
        }
      }
      direct_class_offsets = response_mean;
    } else {
      latent_weights =
          std::get<core::PlssvdModel<T>>(fitted).prediction_weights.front();
      direct_class_offsets = response_mean;
    }
    direct_class_weights.resize(p, classes);
    if (fitted_components > 0) {
      const core::ConstMatrixView<T> retained_projection(
          projection.data(), projection.rows(), fitted_components,
          projection.rows());
      backend.gemm(retained_projection, latent_weights.view(), false, false,
                   direct_class_weights.view());
    }
    for (std::size_t predictor = 0; predictor < p; ++predictor) {
      const T inverse_scale = T(1) / predictor_scale[predictor];
      const T centered = center[predictor] * inverse_scale;
      for (std::size_t class_index = 0; class_index < classes; ++class_index) {
        const T original = direct_class_weights(predictor, class_index);
        direct_class_offsets[class_index] -= centered * original;
        direct_class_weights(predictor, class_index) = original * inverse_scale;
      }
    }
  }

  if (family == Family::opls) {
    const auto& model = std::get<core::OplsModel<T>>(fitted);
    response_mean = model.response_mean;
  } else if (family == Family::kernelpls) {
    const auto& model = std::get<core::KernelPlsModel<T>>(fitted);
    response_mean = model.response_mean;
  }
  return std::make_shared<Model<T>>(
      family, head, fitted_components, classes, std::move(fitted),
      std::move(center),
      std::move(predictor_scale), std::move(response_mean), std::move(lda),
      std::move(direct_class_weights), std::move(direct_class_offsets));
}

std::shared_ptr<ModelBase> fit(
    py::array x, py::handle y, int components, const std::string& method,
    const std::string& classifier, const std::string& scaling,
    int oversample, int power, int seed, int orthogonal_components,
    const std::string& kernel, double gamma, int degree, double offset,
    bool store_scores) {
  if (py::dtype::from_args(x.attr("dtype")).is(py::dtype::of<float>())) {
    return fit_typed<float>(
        py::array_t<float, py::array::forcecast>::ensure(x), y, components,
        method, classifier, scaling, oversample, power, seed,
        orthogonal_components, kernel, gamma, degree, offset, store_scores);
  }
  return fit_typed<double>(
      py::array_t<double, py::array::forcecast>::ensure(x), y, components,
      method, classifier, scaling, oversample, power, seed,
      orthogonal_components, kernel, gamma, degree, offset, store_scores);
}

template<class T>
py::dict fastsvd_typed(
    const py::array_t<T, py::array::forcecast>& input, int components,
    int oversample, int power, int seed) {
  if (components < 1 || oversample < 0 || power < 0 || seed < 0) {
    throw std::invalid_argument("invalid rSVD controls");
  }
  auto matrix = matrix_from_numpy<T>(input);
  const int maximum = static_cast<int>(std::min(matrix.rows(), matrix.columns()));
  if (components > maximum) throw std::invalid_argument("n_components exceeds matrix rank bound");
  core::RsvdControls controls;
  controls.oversample = oversample;
  controls.power = power;
  controls.seed = static_cast<unsigned int>(seed);
  controls.left_only = false;
  NativeBackend<T> backend;
  core::SingularTriplets<T> result;
  {
    py::gil_scoped_release release;
    result = core::randomized_svd(
        core::ConstMatrixView<T>(matrix.view()), components, controls, backend);
  }
  py::array_t<T> values(static_cast<py::ssize_t>(result.singular_values.size()));
  std::copy(result.singular_values.begin(), result.singular_values.end(),
            values.mutable_data());
  py::dict output;
  output["u"] = matrix_to_numpy(result.U);
  output["d"] = std::move(values);
  output["vt"] = matrix_to_numpy(result.Vt);
  return output;
}

py::dict fastsvd(py::array input, int components, int oversample,
                 int power, int seed) {
  if (input.dtype().is(py::dtype::of<float>())) {
    return fastsvd_typed<float>(
        py::array_t<float, py::array::forcecast>::ensure(input), components,
        oversample, power, seed);
  }
  return fastsvd_typed<double>(
      py::array_t<double, py::array::forcecast>::ensure(input), components,
      oversample, power, seed);
}

template<class T>
py::dict cross_validate_typed(
    const py::array_t<T, py::array::forcecast>& input, py::handle response,
    const py::array_t<int, py::array::forcecast>& component_array,
    const py::array_t<int, py::array::forcecast>& fold_array,
    const std::string& method, const std::string& classifier,
    const std::string& scaling, const std::string& selection,
    int oversample, int power, int seed, int orthogonal_components,
    const std::string& kernel, double gamma, int degree, double offset,
    bool store_predictions, bool store_scores) {
  auto predictors = matrix_from_numpy<T>(input);
  if (component_array.ndim() != 1 || component_array.size() < 1) {
    throw std::invalid_argument("components must be a non-empty vector");
  }
  if (fold_array.ndim() != 1 ||
      static_cast<std::size_t>(fold_array.size()) != predictors.rows()) {
    throw std::invalid_argument("folds must contain one value per sample");
  }
  std::vector<int> components(
      component_array.data(), component_array.data() + component_array.size());
  std::vector<int> folds(fold_array.data(), fold_array.data() + fold_array.size());
  if (std::any_of(components.begin(), components.end(),
                  [](int value) { return value < 1; })) {
    throw std::invalid_argument("components must contain positive integers");
  }
  if (std::any_of(folds.begin(), folds.end(),
                  [](int value) { return value < 1; })) {
    throw std::invalid_argument("fold identifiers must be positive integers");
  }

  const Family family = parse_family(method);
  const Head head = parse_head(classifier);
  const auto predictor_scaling = parse_scaling(scaling);
  const std::size_t maximum_component = static_cast<std::size_t>(
      *std::max_element(components.begin(), components.end()));
  core::PlssvdControls plssvd;
  plssvd.rsvd.oversample = oversample;
  plssvd.rsvd.power = power;
  plssvd.rsvd.seed = static_cast<unsigned int>(seed);
  core::KernelCvControls kernel_cv;
  kernel_cv.kernel = parse_kernel(kernel);
  kernel_cv.gamma = gamma;
  kernel_cv.degree = degree;
  kernel_cv.offset = offset;
  NativeBackend<T> backend;
  const core::LinearPlsFamily effective_family =
      family == Family::kernelpls && kernel_cv.kernel == core::KernelType::linear ?
      core::LinearPlsFamily::simpls : cv_family(family);

  py::dict output;
  output["ncomp"] = components;
  output["fold"] = folds;
  if (head != Head::regression) {
    auto label_array = py::array_t<std::int64_t, py::array::forcecast>::ensure(response);
    if (!label_array || label_array.ndim() != 1 ||
        static_cast<std::size_t>(label_array.size()) != predictors.rows()) {
      throw std::invalid_argument("classification labels must have one value per sample");
    }
    std::vector<int> labels(predictors.rows());
    std::int64_t maximum_label = -1;
    for (std::size_t row = 0; row < predictors.rows(); ++row) {
      const auto value = *label_array.data(static_cast<py::ssize_t>(row));
      if (value < 0) throw std::invalid_argument("class indices must be non-negative");
      maximum_label = std::max(maximum_label, value);
      labels[row] = static_cast<int>(value + 1);
    }
    const std::size_t classes = static_cast<std::size_t>(maximum_label + 1);
    auto simpls = simpls_controls(
        predictors.rows(), predictors.columns(), classes, maximum_component,
        true, oversample, power, static_cast<unsigned int>(seed));
    const bool need_scores = store_scores || selection == "q2y";
    core::ClassificationCvResult<T> result;
    {
      py::gil_scoped_release release;
      result = core::cross_validate_classification<T>(
          predictors.view(), labels.data(), classes, folds.data(),
          components.data(), components.size(), predictor_scaling,
          effective_family,
          head == Head::lda ? core::ClassificationHead::lda :
                              core::ClassificationHead::argmax,
          plssvd, simpls, backend, true, need_scores,
          static_cast<std::size_t>(orthogonal_components), kernel_cv, false);
    }
    output["status"] = result.status;
    output["accuracy"] = result.metrics;
    output["Q2Y"] = result.q2;
    output["prediction_index"] = matrix_to_numpy(result.predictions);
    if (need_scores) {
      py::list scores;
      for (const auto& value : result.scores) scores.append(matrix_to_numpy(value));
      output["scores"] = std::move(scores);
    }
  } else {
    auto response_array = py::array_t<T, py::array::forcecast>::ensure(response);
    auto responses = matrix_from_numpy<T>(response_array);
    if (responses.rows() != predictors.rows()) {
      throw std::invalid_argument("X and y row counts differ");
    }
    auto simpls = simpls_controls(
        predictors.rows(), predictors.columns(), responses.columns(),
        maximum_component, false, oversample, power,
        static_cast<unsigned int>(seed));
    const core::RegressionMetric metric = selection == "q2y" ?
        core::RegressionMetric::q2 : selection == "r2y" ?
        core::RegressionMetric::r2 : core::RegressionMetric::rmsd;
    core::RegressionCvResult<T> result;
    {
      py::gil_scoped_release release;
      result = core::cross_validate_regression<T>(
          predictors.view(), responses.view(), folds.data(), components.data(),
          components.size(), predictor_scaling, effective_family, metric,
          plssvd, simpls, backend, true,
          static_cast<std::size_t>(orthogonal_components), kernel_cv, false);
    }
    output["status"] = result.status;
    output["Q2Y"] = result.q2;
    output["R2Y"] = result.observed_r2;
    output["RMSD"] = result.rmsd;
    py::list predictions;
    for (const auto& value : result.predictions) {
      predictions.append(matrix_to_numpy(value));
    }
    output["predictions"] = std::move(predictions);
  }
  return output;
}

py::dict cross_validate(
    py::array input, py::handle response, py::array components, py::array folds,
    const std::string& method, const std::string& classifier,
    const std::string& scaling, const std::string& selection,
    int oversample, int power, int seed, int orthogonal_components,
    const std::string& kernel, double gamma, int degree, double offset,
    bool store_predictions, bool store_scores) {
  auto component_array =
      py::array_t<int, py::array::forcecast>::ensure(components);
  auto fold_array = py::array_t<int, py::array::forcecast>::ensure(folds);
  if (input.dtype().is(py::dtype::of<float>())) {
    return cross_validate_typed<float>(
        py::array_t<float, py::array::forcecast>::ensure(input), response,
        component_array, fold_array, method, classifier, scaling, selection,
        oversample, power, seed, orthogonal_components, kernel, gamma, degree,
        offset, store_predictions, store_scores);
  }
  return cross_validate_typed<double>(
      py::array_t<double, py::array::forcecast>::ensure(input), response,
      component_array, fold_array, method, classifier, scaling, selection,
      oversample, power, seed, orthogonal_components, kernel, gamma, degree,
      offset, store_predictions, store_scores);
}

}  // namespace
}  // namespace fastpls_py

PYBIND11_MODULE(_core, module) {
  using fastpls_py::ModelBase;
  module.doc() = "Python bindings to the fastPLS C++ core";
  module.def("backend_info", &fastpls_py::backend_info);
  module.def("fastsvd", &fastpls_py::fastsvd,
             py::arg("x"), py::arg("n_components"),
             py::arg("oversample") = 32, py::arg("power") = 5,
             py::arg("seed") = 1);
  module.def("cross_validate", &fastpls_py::cross_validate,
             py::arg("X"), py::arg("y"), py::arg("components"),
             py::arg("folds"), py::arg("method"), py::arg("classifier"),
             py::arg("scaling"), py::arg("selection"),
             py::arg("oversample") = 32, py::arg("power") = 5,
             py::arg("seed") = 1, py::arg("orthogonal_components") = 1,
             py::arg("kernel") = "linear", py::arg("gamma") = 1.0,
             py::arg("degree") = 3, py::arg("offset") = 1.0,
             py::arg("store_predictions") = true,
             py::arg("store_scores") = false);
  module.def("fit", &fastpls_py::fit,
             py::arg("X"), py::arg("y"), py::arg("n_components"),
             py::arg("method"), py::arg("classifier"), py::arg("scaling"),
             py::arg("oversample"), py::arg("power"), py::arg("seed"),
             py::arg("orthogonal_components"), py::arg("kernel"),
             py::arg("gamma"), py::arg("degree"), py::arg("offset"),
             py::arg("store_scores") = false);
  py::class_<ModelBase, std::shared_ptr<ModelBase>>(module, "NativeModel")
      .def("predict", &ModelBase::predict)
      .def("predict_scores", &ModelBase::predict_scores)
      .def("predict_classes", &ModelBase::predict_classes)
      .def("vip", &ModelBase::vip)
      .def_property_readonly("scores", &ModelBase::scores)
      .def_property_readonly("n_components", &ModelBase::components)
      .def_property_readonly("dtype", &ModelBase::dtype)
      .def_property_readonly("method", &ModelBase::method)
      .def_property_readonly("classifier", &ModelBase::head);
}
