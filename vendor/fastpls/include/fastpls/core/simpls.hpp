// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_SIMPLS_HPP
#define FASTPLS_CORE_SIMPLS_HPP

#include <fastpls/core/matrix.hpp>
#include <fastpls/core/operator_rsvd.hpp>
#include <fastpls/core/rsvd.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace fastpls {
namespace core {

struct SimplsControls {
  std::size_t components = 2;
  std::size_t maximum_block = 1;
  bool cache_predictor_crossprod = false;
  bool reorthogonalize = true;
  bool store_scores = true;
  bool store_score_moments = false;
  bool use_right_gram = true;
  bool rank_one_operator_direction = false;
  bool batch_candidate_geometry = false;
  bool phase_timing = false;
  RsvdControls rsvd;
};

struct SimplsTiming {
  double setup = 0;
  double direction = 0;
  double candidate_geometry = 0;
  double component_updates = 0;
  double total = 0;
};

inline std::size_t simpls_candidate_block_size(
    std::size_t remaining, std::size_t predictors, std::size_t responses,
    bool classification, std::size_t samples, std::size_t maximum = 8) {
  const double work = static_cast<double>(samples) *
    static_cast<double>(predictors) * static_cast<double>(responses);
  if (responses <= 1 || remaining < 4 || work < 5.0e8) return 1;
  if (classification && responses <= 2048) {
    return std::max<std::size_t>(
      1, std::min({maximum, remaining, predictors, responses})
    );
  }
  const double crosscov_elements = static_cast<double>(predictors) *
    static_cast<double>(responses);
  if (!classification && crosscov_elements > 64.0 * 1024.0 * 1024.0) {
    return std::max<std::size_t>(
      1, std::min({std::size_t(8), maximum, remaining, predictors, responses})
    );
  }
  return 1;
}

template<class T>
struct SimplsModel {
  Matrix<T> weights;
  Matrix<T> response_loadings;
  Matrix<T> deflation_basis;
  Matrix<T> scores;
  Matrix<T> score_gram;
  std::size_t completed_components = 0;
  SimplsTiming timing;
};

template<class T>
struct SimplsWorkspace {
  Matrix<T> crosscov;
  Matrix<T> predictor_crossprod;
  Matrix<T> score;
  Matrix<T> predictor_loading;
  Matrix<T> response_loading;
  Matrix<T> deflation_direction;
  Matrix<T> projection;
  Matrix<T> correction;
  Matrix<T> deflation_row;
  Matrix<T> right_gram;
  Matrix<T> random;
  Matrix<T> direction_sample;
  Matrix<T> direction_basis;
  Matrix<T> reverse_sample;
  Matrix<T> reverse_basis;
  Matrix<T> reduced;
  Matrix<T> reduced_gram;
  Matrix<T> reduced_left;
  Matrix<T> candidate_scores;
  Matrix<T> candidate_loadings;
  bool predictor_crossprod_preloaded = false;
};

namespace simpls_detail {

template<class T>
T dot(ConstMatrixView<T> left, ConstMatrixView<T> right) {
  if (left.columns() != 1 || right.columns() != 1 ||
      left.rows() != right.rows()) {
    throw std::invalid_argument("fastPLS SIMPLS vector dimensions differ");
  }
  T result = T(0);
  for (std::size_t row = 0; row < left.rows(); ++row) {
    result += left(row, 0) * right(row, 0);
  }
  return result;
}

template<class T>
T norm(ConstMatrixView<T> value) {
  return std::sqrt(std::max(dot(value, value), T(0)));
}

template<class T>
void scale(MatrixView<T> value, T divisor) {
  for (std::size_t column = 0; column < value.columns(); ++column) {
    for (std::size_t row = 0; row < value.rows(); ++row) {
      value(row, column) /= divisor;
    }
  }
}

template<class T>
void copy_column(ConstMatrixView<T> source, std::size_t source_column,
                 MatrixView<T> destination, std::size_t destination_column) {
  for (std::size_t row = 0; row < destination.rows(); ++row) {
    destination(row, destination_column) = source(row, source_column);
  }
}

template<class T, class Backend>
void remove_basis(ConstMatrixView<T> basis, std::size_t used,
                  Matrix<T>& vector, bool twice,
                  Backend& backend, Matrix<T>& projection,
                  Matrix<T>& correction) {
  if (used == 0) return;
  ConstMatrixView<T> previous(
    basis.data(), basis.rows(), used, basis.leading_dimension()
  );
  const int passes = twice ? 2 : 1;
  for (int pass = 0; pass < passes; ++pass) {
    projection.resize(used, 1);
    backend.gemm(
      previous, vector.view(), true, false, projection.view()
    );
    correction.resize(vector.rows(), 1);
    backend.gemm(
      previous, projection.view(), false, false, correction.view()
    );
    for (std::size_t row = 0; row < vector.rows(); ++row) {
      vector(row, 0) -= correction(row, 0);
    }
  }
}

template<class T, class Backend>
void remove_score_basis(ConstMatrixView<T> score_basis,
                        ConstMatrixView<T> weight_basis,
                        std::size_t used,
                        Matrix<T>& score,
                        Matrix<T>& weight,
                        Backend& backend, Matrix<T>& projection,
                        Matrix<T>& correction) {
  if (used == 0) return;
  ConstMatrixView<T> previous_scores(
    score_basis.data(), score_basis.rows(), used,
    score_basis.leading_dimension()
  );
  ConstMatrixView<T> previous_weights(
    weight_basis.data(), weight_basis.rows(), used,
    weight_basis.leading_dimension()
  );
  for (int pass = 0; pass < 2; ++pass) {
    projection.resize(used, 1);
    backend.gemm(
      previous_scores, score.view(), true, false, projection.view()
    );
    correction.resize(score.rows(), 1);
    backend.gemm(
      previous_scores, projection.view(), false, false,
      correction.view()
    );
    for (std::size_t row = 0; row < score.rows(); ++row) {
      score(row, 0) -= correction(row, 0);
    }
    correction.resize(weight.rows(), 1);
    backend.gemm(
      previous_weights, projection.view(), false, false,
      correction.view()
    );
    for (std::size_t row = 0; row < weight.rows(); ++row) {
      weight(row, 0) -= correction(row, 0);
    }
  }
}

template<class T>
void copy_matrix(ConstMatrixView<T> source, Matrix<T>& destination) {
  destination.resize(source.rows(), source.columns());
  for (std::size_t column = 0; column < source.columns(); ++column) {
    for (std::size_t row = 0; row < source.rows(); ++row) {
      destination(row, column) = source(row, column);
    }
  }
}

template<class T, class Backend>
auto backend_rank1_subtract(
    MatrixView<T> target, ConstMatrixView<T> column,
    ConstMatrixView<T> row, Backend& backend, int)
    -> decltype(backend.rank1_subtract(target, column, row), bool()) {
  return backend.rank1_subtract(target, column, row);
}

template<class T, class Backend>
bool backend_rank1_subtract(
    MatrixView<T>, ConstMatrixView<T>, ConstMatrixView<T>, Backend&, long) {
  return false;
}

template<class T, class Backend>
bool refresh_directions(ConstMatrixView<T> crosscov,
                        ConstMatrixView<T> right_gram,
                        bool use_right_gram,
                        std::size_t retained,
                        const RsvdControls& controls,
                        Backend& backend,
                        SimplsWorkspace<T>& workspace,
                        Matrix<T>& directions) {
  const std::size_t maximum = std::min(
    crosscov.rows(), crosscov.columns()
  );
  const std::size_t width = std::min(
    maximum,
    retained + static_cast<std::size_t>(std::max(controls.oversample, 0))
  );
  if (retained == 0 || width == 0) return false;

  if (use_right_gram && right_gram.rows() <= 64) {
    copy_matrix(right_gram, workspace.reduced_gram);
    std::vector<T> eigenvalues;
    if (!backend.symmetric_eigen(workspace.reduced_gram, eigenvalues)) {
      return false;
    }
    const std::size_t available = std::min(retained, eigenvalues.size());
    workspace.reverse_basis.resize(right_gram.rows(), available);
    for (std::size_t column = 0; column < available; ++column) {
      const std::size_t source = eigenvalues.size() - 1 - column;
      for (std::size_t row = 0; row < right_gram.rows(); ++row) {
        workspace.reverse_basis(row, column) =
          workspace.reduced_gram(row, source);
      }
    }
    directions.resize(crosscov.rows(), available);
    backend.gemm(
      crosscov, workspace.reverse_basis.view(), false, false,
      directions.view()
    );
    return true;
  }

  std::mt19937 generator(controls.seed);
  std::normal_distribution<T> normal(T(0), T(1));
  workspace.random.resize(crosscov.columns(), width);
  for (std::size_t index = 0; index < workspace.random.size(); ++index) {
    workspace.random.data()[index] = normal(generator);
  }

  if (use_right_gram) {
    copy_matrix(
      ConstMatrixView<T>(workspace.random.view()), workspace.reverse_basis
    );
    for (int iteration = 0; iteration < std::max(controls.power, 0);
         ++iteration) {
      workspace.reverse_sample.resize(crosscov.columns(), width);
      backend.gemm(
        right_gram, workspace.reverse_basis.view(), false, false,
        workspace.reverse_sample.view()
      );
      if (!backend.qr_economy(
            workspace.reverse_sample.view(), workspace.reverse_basis)) {
        return false;
      }
    }
    if (controls.power == 0 && !backend.qr_economy(
          workspace.reverse_basis.view(), workspace.reverse_sample)) {
      return false;
    } else if (controls.power == 0) {
      workspace.reverse_basis = std::move(workspace.reverse_sample);
    }
    workspace.direction_sample.resize(crosscov.rows(), width);
    backend.gemm(
      crosscov, workspace.reverse_basis.view(), false, false,
      workspace.direction_sample.view()
    );
  } else {
    workspace.direction_sample.resize(crosscov.rows(), width);
    backend.gemm(
      crosscov, workspace.random.view(), false, false,
      workspace.direction_sample.view()
    );
    for (int iteration = 0; iteration < std::max(controls.power, 0);
         ++iteration) {
      if (!backend.qr_economy(
            workspace.direction_sample.view(), workspace.direction_basis)) {
        return false;
      }
      workspace.reverse_sample.resize(
        crosscov.columns(), workspace.direction_basis.columns()
      );
      backend.gemm(
        crosscov, workspace.direction_basis.view(), true, false,
        workspace.reverse_sample.view()
      );
      if (!backend.qr_economy(
            workspace.reverse_sample.view(), workspace.reverse_basis)) {
        return false;
      }
      workspace.direction_sample.resize(
        crosscov.rows(), workspace.reverse_basis.columns()
      );
      backend.gemm(
        crosscov, workspace.reverse_basis.view(), false, false,
        workspace.direction_sample.view()
      );
    }
  }

  if (!backend.qr_economy(
        workspace.direction_sample.view(), workspace.direction_basis)) {
    return false;
  }
  workspace.reduced.resize(
    workspace.direction_basis.columns(), crosscov.columns()
  );
  backend.gemm(
    workspace.direction_basis.view(), crosscov, true, false,
    workspace.reduced.view()
  );
  workspace.reduced_gram.resize(
    workspace.reduced.rows(), workspace.reduced.rows()
  );
  backend.gemm(
    workspace.reduced.view(), workspace.reduced.view(), false, true,
    workspace.reduced_gram.view()
  );
  std::vector<T> eigenvalues;
  if (!backend.symmetric_eigen(workspace.reduced_gram, eigenvalues)) {
    std::vector<T> singular_values;
    Matrix<T> unused;
    if (!backend.svd_economy(
          workspace.reduced.view(), true, workspace.reduced_left,
          singular_values, unused)) {
      return false;
    }
  } else {
    const std::size_t available = std::min(
      retained, eigenvalues.size()
    );
    workspace.reduced_left.resize(
      workspace.reduced_gram.rows(), available
    );
    for (std::size_t column = 0; column < available; ++column) {
      const std::size_t source = eigenvalues.size() - 1 - column;
      for (std::size_t row = 0;
           row < workspace.reduced_gram.rows(); ++row) {
        workspace.reduced_left(row, column) =
          workspace.reduced_gram(row, source);
      }
    }
  }
  const std::size_t available = std::min(
    retained, workspace.reduced_left.columns()
  );
  if (available == 0) return false;
  ConstMatrixView<T> reduced_left(
    workspace.reduced_left.data(), workspace.reduced_left.rows(), available,
    workspace.reduced_left.rows()
  );
  directions.resize(crosscov.rows(), available);
  backend.gemm(
    workspace.direction_basis.view(), reduced_left, false, false,
    directions.view()
  );
  return true;
}

template<class T, class Backend>
auto backend_sample_gram_apply(
    Backend& backend, ConstMatrixView<T> predictors,
    ConstMatrixView<T> sample_gram, ConstMatrixView<T> direction,
    MatrixView<T> output, int) -> decltype(
      backend.sample_gram_apply(
        predictors, sample_gram, direction, output
      ), bool()) {
  return backend.sample_gram_apply(
    predictors, sample_gram, direction, output
  );
}

template<class T, class Backend>
bool backend_sample_gram_apply(
    Backend&, ConstMatrixView<T>, ConstMatrixView<T>,
    ConstMatrixView<T>, MatrixView<T>, long) {
  return false;
}

template<class T, class Backend>
auto backend_sample_geometry(
    Backend& backend, ConstMatrixView<T> predictors,
    ConstMatrixView<T> direction, MatrixView<T> score,
    MatrixView<T> loading, int) -> decltype(
      backend.sample_geometry(
        predictors, direction, score, loading
      ), bool()) {
  return backend.sample_geometry(
    predictors, direction, score, loading
  );
}

template<class T, class Backend>
bool backend_sample_geometry(
    Backend&, ConstMatrixView<T>, ConstMatrixView<T>, MatrixView<T>,
    MatrixView<T>, long) {
  return false;
}

template<class T, class Backend>
bool sample_gram_direction(
    ConstMatrixView<T> predictors, ConstMatrixView<T> sample_gram,
    ConstMatrixView<T> deflation_basis, std::size_t used,
    const RsvdControls& controls, Backend& backend,
    SimplsWorkspace<T>& workspace, Matrix<T>& direction) {
  if (sample_gram.rows() != predictors.rows() ||
      sample_gram.columns() != predictors.rows()) {
    return false;
  }
  std::mt19937 generator(controls.seed);
  std::normal_distribution<T> normal(T(0), T(1));
  direction.resize(predictors.columns(), 1);
  for (std::size_t row = 0; row < direction.rows(); ++row) {
    direction(row, 0) = normal(generator);
  }
  remove_basis(
    deflation_basis, used, direction, true, backend,
    workspace.projection, workspace.correction
  );
  T direction_norm = norm(ConstMatrixView<T>(direction.view()));
  if (!std::isfinite(direction_norm) || direction_norm <= T(0)) return false;
  scale(direction.view(), direction_norm);

  const int iterations = std::max(controls.power, 1);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    workspace.direction_basis.resize(predictors.columns(), 1);
    const bool fused = backend_sample_gram_apply(
      backend, predictors, sample_gram,
      ConstMatrixView<T>(direction.view()),
      workspace.direction_basis.view(), 0
    );
    if (!fused) {
      workspace.direction_sample.resize(predictors.rows(), 1);
      backend.gemm(
        predictors, direction.view(), false, false,
        workspace.direction_sample.view()
      );
      workspace.reverse_sample.resize(predictors.rows(), 1);
      backend.gemm(
        sample_gram, workspace.direction_sample.view(), false, false,
        workspace.reverse_sample.view()
      );
      backend.gemm(
        predictors, workspace.reverse_sample.view(), true, false,
        workspace.direction_basis.view()
      );
    }
    remove_basis(
      deflation_basis, used, workspace.direction_basis, true, backend,
      workspace.projection, workspace.correction
    );
    direction_norm = norm(
      ConstMatrixView<T>(workspace.direction_basis.view())
    );
    if (!std::isfinite(direction_norm) || direction_norm <= T(0)) {
      return false;
    }
    scale(workspace.direction_basis.view(), direction_norm);
    direction = std::move(workspace.direction_basis);
  }
  return true;
}

}  // namespace simpls_detail

// Fits the sequential SIMPLS estimator from an already preprocessed predictor
// matrix and its predictor-response cross-covariance. Centering, scaling and
// response encoding remain boundary-layer responsibilities.
template<class T, class Backend>
SimplsModel<T> fit_simpls_preprocessed(
    ConstMatrixView<T> predictors,
    ConstMatrixView<T> initial_crosscov,
    const SimplsControls& controls,
    Backend& backend,
    SimplsWorkspace<T>& workspace,
    std::size_t sample_count_override = 0) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  const bool moments_only = predictors.data() == nullptr;
  const bool valid_moments_only = moments_only && sample_count_override > 0 &&
    controls.cache_predictor_crossprod && !controls.store_scores &&
    !controls.reorthogonalize &&
    workspace.predictor_crossprod_preloaded &&
    workspace.predictor_crossprod.rows() == initial_crosscov.rows() &&
    workspace.predictor_crossprod.columns() == initial_crosscov.rows();
  if (initial_crosscov.data() == nullptr || initial_crosscov.empty() ||
      controls.components == 0 ||
      (!moments_only && (predictors.empty() ||
        predictors.columns() != initial_crosscov.rows())) ||
      (moments_only && !valid_moments_only)) {
    throw std::invalid_argument(
      "fastPLS SIMPLS requires nonempty conformable inputs and components"
    );
  }
  const std::size_t n = moments_only ? sample_count_override : predictors.rows();
  const std::size_t p = initial_crosscov.rows();
  const std::size_t q = initial_crosscov.columns();
  const std::size_t maximum = std::min({
    controls.components, p, std::max<std::size_t>(n - 1, 1)
  });

  SimplsModel<T> model;
  model.weights.resize(p, maximum);
  model.response_loadings.resize(q, maximum);
  model.deflation_basis.resize(p, maximum);
  const bool retain_scores = controls.store_scores ||
    controls.reorthogonalize;
  const bool retain_score_moments = controls.store_score_moments;
  if (retain_score_moments) {
    model.score_gram.resize(maximum, maximum);
  }
  const bool defer_score_materialization =
    controls.cache_predictor_crossprod && controls.store_scores &&
    !controls.reorthogonalize && !moments_only;
  if (retain_scores) model.scores.resize(n, maximum);
  simpls_detail::copy_matrix(initial_crosscov, workspace.crosscov);

  if (controls.cache_predictor_crossprod) {
    const bool use_preloaded = workspace.predictor_crossprod_preloaded &&
      workspace.predictor_crossprod.rows() == p &&
      workspace.predictor_crossprod.columns() == p;
    if (!use_preloaded) {
      workspace.predictor_crossprod.resize(p, p);
      backend.gemm(
        predictors, predictors, true, false,
        workspace.predictor_crossprod.view()
      );
    }
    workspace.predictor_crossprod_preloaded = false;
  }
  const bool use_right_gram = controls.use_right_gram &&
    q <= p && q <= 512;
  if (use_right_gram) {
    workspace.right_gram.resize(q, q);
    backend.gemm(
      workspace.crosscov.view(), workspace.crosscov.view(), true, false,
      workspace.right_gram.view()
    );
  }
  model.timing.setup = std::chrono::duration<double>(
    Clock::now() - started
  ).count();

  std::size_t component = 0;
  while (component < maximum) {
    const std::size_t block_start = component;
    const auto direction_started = Clock::now();
    const std::size_t block = std::min({
      std::max<std::size_t>(controls.maximum_block, 1),
      maximum - component,
      std::min(p, q)
    });
    RsvdControls rsvd = controls.rsvd;
    rsvd.seed += static_cast<unsigned int>(component);
    rsvd.left_only = true;
    Matrix<T> candidates;
    if (!simpls_detail::refresh_directions(
      ConstMatrixView<T>(workspace.crosscov.view()),
      ConstMatrixView<T>(workspace.right_gram.view()), use_right_gram,
      block, rsvd, backend, workspace, candidates)) break;
    model.timing.direction += std::chrono::duration<double>(
      Clock::now() - direction_started
    ).count();
    const std::size_t available = std::min(
      block, candidates.columns()
    );
    if (controls.batch_candidate_geometry &&
        !controls.cache_predictor_crossprod) {
      const auto geometry_started = Clock::now();
      workspace.candidate_scores.resize(n, available);
      backend.gemm(
        predictors, candidates.view(), false, false,
        workspace.candidate_scores.view()
      );
      workspace.candidate_loadings.resize(p, available);
      backend.gemm(
        predictors, workspace.candidate_scores.view(), true, false,
        workspace.candidate_loadings.view()
      );
      model.timing.candidate_geometry += std::chrono::duration<double>(
        Clock::now() - geometry_started
      ).count();
    }

    bool stopped = false;
    for (std::size_t candidate = 0;
         candidate < available && component < maximum;
         ++candidate, ++component) {
      const auto component_started = Clock::now();
      Matrix<T> direction(p, 1);
      simpls_detail::copy_column(
        ConstMatrixView<T>(candidates.view()), candidate,
        direction.view(), 0
      );

      workspace.score.resize(n, 1);
      workspace.predictor_loading.resize(p, 1);
      T score_norm = T(0);
      if (controls.cache_predictor_crossprod) {
        backend.gemm(
          workspace.predictor_crossprod.view(), direction.view(),
          false, false, workspace.predictor_loading.view()
        );
        const T norm_squared = simpls_detail::dot(
          ConstMatrixView<T>(direction.view()),
          ConstMatrixView<T>(workspace.predictor_loading.view())
        );
        if (!std::isfinite(norm_squared) || norm_squared <= T(0)) {
          stopped = true;
          break;
        }
        score_norm = std::sqrt(norm_squared);
        simpls_detail::scale(direction.view(), score_norm);
        simpls_detail::scale(
          workspace.predictor_loading.view(), score_norm
        );
        if (retain_scores && !defer_score_materialization) {
          backend.gemm(
            predictors, direction.view(), false, false,
            workspace.score.view()
          );
        }
      } else {
        if (controls.batch_candidate_geometry) {
          simpls_detail::copy_column(
            ConstMatrixView<T>(workspace.candidate_scores.view()),
            candidate, workspace.score.view(), 0
          );
        } else {
          backend.gemm(
            predictors, direction.view(), false, false,
            workspace.score.view()
          );
        }
        const bool score_reorthogonalized =
          controls.reorthogonalize && component > 0;
        if (controls.reorthogonalize && component > 0) {
          simpls_detail::remove_score_basis(
            ConstMatrixView<T>(model.scores.view()),
            ConstMatrixView<T>(model.weights.view()), component,
            workspace.score, direction, backend, workspace.projection,
            workspace.correction
          );
        }
        score_norm = simpls_detail::norm(
          ConstMatrixView<T>(workspace.score.view())
        );
        if (!std::isfinite(score_norm) || score_norm <= T(0)) {
          stopped = true;
          break;
        }
        simpls_detail::scale(workspace.score.view(), score_norm);
        simpls_detail::scale(direction.view(), score_norm);
        if (controls.batch_candidate_geometry &&
            !score_reorthogonalized) {
          simpls_detail::copy_column(
            ConstMatrixView<T>(workspace.candidate_loadings.view()),
            candidate, workspace.predictor_loading.view(), 0
          );
          simpls_detail::scale(
            workspace.predictor_loading.view(), score_norm
          );
        } else {
          backend.gemm(
            predictors, workspace.score.view(), true, false,
            workspace.predictor_loading.view()
          );
        }
      }

      if (retain_score_moments) {
        model.score_gram(component, component) = T(1);
      }
      if (retain_score_moments && !controls.reorthogonalize) {
        for (std::size_t previous = block_start;
             previous < component; ++previous) {
          T value = T(0);
          for (std::size_t predictor = 0; predictor < p; ++predictor) {
            value += model.weights(predictor, previous) *
              workspace.predictor_loading(predictor, 0);
          }
          model.score_gram(previous, component) = value;
          model.score_gram(component, previous) = value;
        }
      }

      workspace.response_loading.resize(q, 1);
      backend.gemm(
        initial_crosscov, direction.view(), true, false,
        workspace.response_loading.view()
      );
      simpls_detail::copy_matrix(
        ConstMatrixView<T>(workspace.predictor_loading.view()),
        workspace.deflation_direction
      );
      simpls_detail::remove_basis(
        ConstMatrixView<T>(model.deflation_basis.view()), component,
        workspace.deflation_direction, controls.reorthogonalize,
        backend, workspace.projection, workspace.correction
      );
      const T deflation_norm = simpls_detail::norm(
        ConstMatrixView<T>(workspace.deflation_direction.view())
      );
      if (!std::isfinite(deflation_norm) || deflation_norm <= T(0)) {
        stopped = true;
        break;
      }
      simpls_detail::scale(
        workspace.deflation_direction.view(), deflation_norm
      );

      workspace.deflation_row.resize(1, q);
      backend.gemm(
        workspace.deflation_direction.view(), workspace.crosscov.view(),
        true, false, workspace.deflation_row.view()
      );
      if (!simpls_detail::backend_rank1_subtract(
            workspace.crosscov.view(),
            ConstMatrixView<T>(workspace.deflation_direction.view()),
            ConstMatrixView<T>(workspace.deflation_row.view()), backend, 0)) {
        for (std::size_t column = 0; column < q; ++column) {
          const T coefficient = workspace.deflation_row(0, column);
          for (std::size_t row = 0; row < p; ++row) {
            workspace.crosscov(row, column) -=
              workspace.deflation_direction(row, 0) * coefficient;
          }
        }
      }
      if (use_right_gram) {
        for (std::size_t column = 0; column < q; ++column) {
          for (std::size_t row = 0; row < q; ++row) {
            workspace.right_gram(row, column) -=
              workspace.deflation_row(0, row) *
              workspace.deflation_row(0, column);
          }
        }
        for (std::size_t column = 0; column < q; ++column) {
          for (std::size_t row = 0; row < column; ++row) {
            const T value = T(0.5) * (
              workspace.right_gram(row, column) +
              workspace.right_gram(column, row)
            );
            workspace.right_gram(row, column) = value;
            workspace.right_gram(column, row) = value;
          }
        }
      }

      simpls_detail::copy_column(
        ConstMatrixView<T>(direction.view()), 0,
        model.weights.view(), component
      );
      simpls_detail::copy_column(
        ConstMatrixView<T>(workspace.response_loading.view()), 0,
        model.response_loadings.view(), component
      );
      simpls_detail::copy_column(
        ConstMatrixView<T>(workspace.deflation_direction.view()), 0,
        model.deflation_basis.view(), component
      );
      if (retain_scores && !defer_score_materialization) {
        simpls_detail::copy_column(
          ConstMatrixView<T>(workspace.score.view()), 0,
          model.scores.view(), component
        );
      }
      model.completed_components = component + 1;
      model.timing.component_updates += std::chrono::duration<double>(
        Clock::now() - component_started
      ).count();
    }
    if (stopped) break;
  }
  if (defer_score_materialization && model.completed_components > 0) {
    ConstMatrixView<T> weights(
      model.weights.data(), model.weights.rows(), model.completed_components,
      model.weights.rows()
    );
    model.scores.resize(n, model.completed_components);
    backend.gemm(
      predictors, weights, false, false, model.scores.view()
    );
  }
  model.timing.total = std::chrono::duration<double>(
    Clock::now() - started
  ).count();
  return model;
}

// Fits SIMPLS through a predictor-response operator. The initial operator is
// retained for response loadings, while the projected operator accumulates the
// orthogonal SIMPLS deflations without materializing the cross-covariance.
template<class T, class InitialOperator, class ProjectedOperator,
         class Backend>
SimplsModel<T> fit_simpls_operator(
    ConstMatrixView<T> predictors,
    InitialOperator& initial_crosscov,
    ProjectedOperator& projected_crosscov,
    const SimplsControls& controls,
    Backend& backend,
    SimplsWorkspace<T>& workspace,
    OperatorRsvdWorkspace<T>& rsvd_workspace,
    ConstMatrixView<T> sample_response_gram = ConstMatrixView<T>()) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  if (predictors.empty() || initial_crosscov.rows() != predictors.columns() ||
      projected_crosscov.rows() != predictors.columns() ||
      initial_crosscov.columns() == 0 ||
      projected_crosscov.columns() != initial_crosscov.columns() ||
      controls.components == 0) {
    throw std::invalid_argument(
      "fastPLS implicit SIMPLS dimensions or controls are invalid"
    );
  }
  const std::size_t n = predictors.rows();
  const std::size_t p = predictors.columns();
  const std::size_t q = initial_crosscov.columns();
  const std::size_t maximum = std::min({
    controls.components, p, std::max<std::size_t>(n - 1, 1)
  });

  SimplsModel<T> model;
  model.weights.resize(p, maximum);
  model.response_loadings.resize(q, maximum);
  model.deflation_basis.resize(p, maximum);
  const bool retain_scores = controls.store_scores ||
    controls.reorthogonalize;
  if (retain_scores) model.scores.resize(n, maximum);
  if (controls.cache_predictor_crossprod) {
    const bool use_preloaded = workspace.predictor_crossprod_preloaded &&
      workspace.predictor_crossprod.rows() == p &&
      workspace.predictor_crossprod.columns() == p;
    if (!use_preloaded) {
      workspace.predictor_crossprod.resize(p, p);
      backend.gemm(
        predictors, predictors, true, false,
        workspace.predictor_crossprod.view()
      );
    }
    workspace.predictor_crossprod_preloaded = false;
  }
  model.timing.setup = std::chrono::duration<double>(
    Clock::now() - started
  ).count();

  std::size_t component = 0;
  while (component < maximum) {
    const auto direction_started = Clock::now();
    const bool use_sample_gram = !sample_response_gram.empty();
    const std::size_t block = use_sample_gram ? 1 : std::min({
      std::max<std::size_t>(controls.maximum_block, 1),
      maximum - component,
      std::min(p, q)
    });
    RsvdControls rsvd = controls.rsvd;
    rsvd.seed += static_cast<unsigned int>(component);
    rsvd.left_only = true;
    Matrix<T> candidates;
    if (use_sample_gram) {
      if (!simpls_detail::sample_gram_direction(
            predictors, sample_response_gram,
            projected_crosscov.basis(), component, rsvd, backend,
            workspace, candidates)) {
        break;
      }
    } else if (block == 1 && controls.rank_one_operator_direction) {
      if (!randomized_dominant_operator_direction<T>(
            projected_crosscov, rsvd, backend, rsvd_workspace,
            candidates)) {
        break;
      }
    } else {
      auto decomposition = randomized_operator_svd<T>(
        projected_crosscov, static_cast<int>(block), rsvd, backend,
        rsvd_workspace
      );
      candidates = std::move(decomposition.U);
    }
    model.timing.direction += std::chrono::duration<double>(
      Clock::now() - direction_started
    ).count();
    const std::size_t available = std::min(
      block, candidates.columns()
    );
    if (available == 0) break;

    if (controls.batch_candidate_geometry &&
        !controls.cache_predictor_crossprod) {
      const auto geometry_started = Clock::now();
      workspace.candidate_scores.resize(n, available);
      backend.gemm(
        predictors, candidates.view(), false, false,
        workspace.candidate_scores.view()
      );
      workspace.candidate_loadings.resize(p, available);
      backend.gemm(
        predictors, workspace.candidate_scores.view(), true, false,
        workspace.candidate_loadings.view()
      );
      model.timing.candidate_geometry += std::chrono::duration<double>(
        Clock::now() - geometry_started
      ).count();
    }

    bool stopped = false;
    for (std::size_t candidate = 0;
         candidate < available && component < maximum;
         ++candidate, ++component) {
      const auto component_started = Clock::now();
      Matrix<T> direction(p, 1);
      simpls_detail::copy_column(
        ConstMatrixView<T>(candidates.view()), candidate,
        direction.view(), 0
      );

      workspace.score.resize(n, 1);
      workspace.predictor_loading.resize(p, 1);
      if (controls.cache_predictor_crossprod) {
        backend.gemm(
          workspace.predictor_crossprod.view(), direction.view(),
          false, false, workspace.predictor_loading.view()
        );
        const T norm_squared = simpls_detail::dot(
          ConstMatrixView<T>(direction.view()),
          ConstMatrixView<T>(workspace.predictor_loading.view())
        );
        if (!std::isfinite(norm_squared) || norm_squared <= T(0)) {
          stopped = true;
          break;
        }
        const T score_norm = std::sqrt(norm_squared);
        simpls_detail::scale(direction.view(), score_norm);
        simpls_detail::scale(
          workspace.predictor_loading.view(), score_norm
        );
        if (retain_scores) {
          backend.gemm(
            predictors, direction.view(), false, false,
            workspace.score.view()
          );
        }
      } else {
        bool fused_geometry = false;
        if (controls.batch_candidate_geometry) {
          simpls_detail::copy_column(
            ConstMatrixView<T>(workspace.candidate_scores.view()),
            candidate, workspace.score.view(), 0
          );
        } else {
          fused_geometry = use_sample_gram &&
            simpls_detail::backend_sample_geometry(
              backend, predictors, ConstMatrixView<T>(direction.view()),
              workspace.score.view(), workspace.predictor_loading.view(), 0
            );
          if (!fused_geometry) {
            backend.gemm(
              predictors, direction.view(), false, false,
              workspace.score.view()
            );
          }
        }
        const bool score_reorthogonalized =
          controls.reorthogonalize && component > 0;
        if (score_reorthogonalized) {
          simpls_detail::remove_score_basis(
            ConstMatrixView<T>(model.scores.view()),
            ConstMatrixView<T>(model.weights.view()), component,
            workspace.score, direction, backend, workspace.projection,
            workspace.correction
          );
        }
        const T score_norm = simpls_detail::norm(
          ConstMatrixView<T>(workspace.score.view())
        );
        if (!std::isfinite(score_norm) || score_norm <= T(0)) {
          stopped = true;
          break;
        }
        simpls_detail::scale(workspace.score.view(), score_norm);
        simpls_detail::scale(direction.view(), score_norm);
        if ((controls.batch_candidate_geometry || fused_geometry) &&
            !score_reorthogonalized) {
          if (controls.batch_candidate_geometry) {
            simpls_detail::copy_column(
              ConstMatrixView<T>(workspace.candidate_loadings.view()),
              candidate, workspace.predictor_loading.view(), 0
            );
          }
          simpls_detail::scale(
            workspace.predictor_loading.view(), score_norm
          );
        } else {
          backend.gemm(
            predictors, workspace.score.view(), true, false,
            workspace.predictor_loading.view()
          );
        }
      }

      if (!use_sample_gram) {
        initial_crosscov.multiply(
          direction.view(), true, workspace.response_loading
        );
      }
      simpls_detail::copy_matrix(
        ConstMatrixView<T>(workspace.predictor_loading.view()),
        workspace.deflation_direction
      );
      simpls_detail::remove_basis(
        ConstMatrixView<T>(model.deflation_basis.view()), component,
        workspace.deflation_direction, controls.reorthogonalize,
        backend, workspace.projection, workspace.correction
      );
      const T deflation_norm = simpls_detail::norm(
        ConstMatrixView<T>(workspace.deflation_direction.view())
      );
      if (!std::isfinite(deflation_norm) || deflation_norm <= T(0)) {
        stopped = true;
        break;
      }
      simpls_detail::scale(
        workspace.deflation_direction.view(), deflation_norm
      );
      projected_crosscov.deflate(workspace.deflation_direction.view());

      simpls_detail::copy_column(
        ConstMatrixView<T>(direction.view()), 0,
        model.weights.view(), component
      );
      if (!use_sample_gram) {
        simpls_detail::copy_column(
          ConstMatrixView<T>(workspace.response_loading.view()), 0,
          model.response_loadings.view(), component
        );
      }
      simpls_detail::copy_column(
        ConstMatrixView<T>(workspace.deflation_direction.view()), 0,
        model.deflation_basis.view(), component
      );
      if (retain_scores) {
        simpls_detail::copy_column(
          ConstMatrixView<T>(workspace.score.view()), 0,
          model.scores.view(), component
        );
      }
      model.completed_components = component + 1;
      model.timing.component_updates += std::chrono::duration<double>(
        Clock::now() - component_started
      ).count();
    }
    if (stopped) break;
  }
  if (!sample_response_gram.empty() && model.completed_components > 0) {
    ConstMatrixView<T> active_weights(
      model.weights.data(), model.weights.rows(), model.completed_components,
      model.weights.rows()
    );
    Matrix<T> response_loadings;
    initial_crosscov.multiply(active_weights, true, response_loadings);
    for (std::size_t column = 0;
         column < model.completed_components; ++column) {
      for (std::size_t row = 0; row < q; ++row) {
        model.response_loadings(row, column) =
          response_loadings(row, column);
      }
    }
  }
  model.timing.total = std::chrono::duration<double>(
    Clock::now() - started
  ).count();
  return model;
}

template<class T, class Backend>
Matrix<T> predict_simpls_preprocessed(
    ConstMatrixView<T> predictors,
    const SimplsModel<T>& model,
    std::size_t components,
    Backend& backend) {
  if (predictors.columns() != model.weights.rows() || components == 0 ||
      components > model.completed_components) {
    throw std::invalid_argument(
      "fastPLS SIMPLS prediction dimensions or components are invalid"
    );
  }
  ConstMatrixView<T> weights(
    model.weights.data(), model.weights.rows(), components,
    model.weights.rows()
  );
  ConstMatrixView<T> loadings(
    model.response_loadings.data(), model.response_loadings.rows(),
    components, model.response_loadings.rows()
  );
  Matrix<T> scores(predictors.rows(), components);
  backend.gemm(predictors, weights, false, false, scores.view());
  Matrix<T> prediction(predictors.rows(), loadings.rows());
  backend.gemm(
    scores.view(), loadings, false, true, prediction.view()
  );
  return prediction;
}

}  // namespace core
}  // namespace fastpls

#endif
