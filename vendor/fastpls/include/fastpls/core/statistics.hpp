// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_STATISTICS_HPP
#define FASTPLS_CORE_STATISTICS_HPP

#include <fastpls/core/matrix.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace fastpls::core {

template<typename Scalar>
double observed_mean_r2(BasicMatrixView<const Scalar> observed,
                        BasicMatrixView<const Scalar> predicted) {
  if (observed.rows() != predicted.rows() ||
      observed.columns() != predicted.columns()) {
    throw std::invalid_argument("R2 matrices must have matching dimensions");
  }
  double total_sum_squares = 0.0;
  double prediction_sum_squares = 0.0;
  for (std::size_t column = 0; column < observed.columns(); ++column) {
    double mean = 0.0;
    for (std::size_t row = 0; row < observed.rows(); ++row) {
      mean += static_cast<double>(observed(row, column));
    }
    mean /= static_cast<double>(observed.rows());
    for (std::size_t row = 0; row < observed.rows(); ++row) {
      const double value = static_cast<double>(observed(row, column));
      const double error = value - static_cast<double>(predicted(row, column));
      const double centered = value - mean;
      prediction_sum_squares += error * error;
      total_sum_squares += centered * centered;
    }
  }
  return 1.0 - prediction_sum_squares / total_sum_squares;
}

enum class CorrelationStatus {
  success,
  insufficient_pairs,
  no_complete_pairs,
  constant_input
};

struct CorrelationResult {
  double value;
  std::size_t complete_pairs;
  CorrelationStatus status;
};

namespace detail {

struct RankEntry {
  double value;
  std::size_t pair;
};

struct FloatRankEntry {
  float value;
  std::uint32_t pair;
};

inline std::uint64_t sortable_double_key(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr std::uint64_t sign = std::uint64_t{1} << 63;
  return (bits & sign) ? ~bits : (bits ^ sign);
}

inline std::uint32_t sortable_float_key(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr std::uint32_t sign = std::uint32_t{1} << 31;
  return (bits & sign) ? ~bits : (bits ^ sign);
}

inline void radix_sort_rank_entries(std::vector<RankEntry>& entries) {
  constexpr unsigned int radix_bits = 8;
  constexpr std::size_t bucket_count = 1U << radix_bits;
  constexpr std::uint64_t bucket_mask = bucket_count - 1U;
  std::vector<RankEntry> scratch(entries.size());
  std::array<std::size_t, bucket_count> offsets{};
  std::vector<RankEntry>* source = &entries;
  std::vector<RankEntry>* destination = &scratch;
  for (unsigned int pass = 0; pass < 8; ++pass) {
    std::fill(offsets.begin(), offsets.end(), 0);
    const unsigned int shift = pass * radix_bits;
    for (const auto& entry : *source) {
      const auto bucket = static_cast<std::size_t>(
        (sortable_double_key(entry.value) >> shift) & bucket_mask
      );
      ++offsets[bucket];
    }
    std::size_t position = 0;
    for (std::size_t bucket = 0; bucket < offsets.size(); ++bucket) {
      const std::size_t count = offsets[bucket];
      offsets[bucket] = position;
      position += count;
    }
    for (const auto& entry : *source) {
      const auto bucket = static_cast<std::size_t>(
        (sortable_double_key(entry.value) >> shift) & bucket_mask
      );
      (*destination)[offsets[bucket]++] = entry;
    }
    std::swap(source, destination);
  }
  if (source != &entries) entries.swap(*source);
}

inline bool rank_values(std::vector<RankEntry>& entries,
                        std::vector<double>& ranks) {
  constexpr std::size_t radix_threshold = 65536;
  if (entries.size() >= radix_threshold) {
    radix_sort_rank_entries(entries);
  } else {
    std::sort(entries.begin(), entries.end(),
              [](const RankEntry& left, const RankEntry& right) {
                return left.value < right.value;
              });
  }
  if (entries.front().value == entries.back().value) return false;
  for (std::size_t begin = 0; begin < entries.size();) {
    std::size_t end = begin + 1;
    while (end < entries.size() &&
           entries[end].value == entries[begin].value) {
      ++end;
    }
    const double rank =
      (static_cast<double>(begin) + static_cast<double>(end) + 1.0) / 2.0;
    for (std::size_t index = begin; index < end; ++index) {
      ranks[entries[index].pair] = rank;
    }
    begin = end;
  }
  return true;
}

inline void radix_sort_float_rank_entries(
    std::vector<FloatRankEntry>& entries) {
  constexpr unsigned int radix_bits = 8;
  constexpr std::size_t bucket_count = 1U << radix_bits;
  constexpr std::uint32_t bucket_mask = bucket_count - 1U;
  std::vector<FloatRankEntry> scratch(entries.size());
  std::array<std::size_t, bucket_count> offsets{};
  std::vector<FloatRankEntry>* source = &entries;
  std::vector<FloatRankEntry>* destination = &scratch;
  for (unsigned int pass = 0; pass < 4; ++pass) {
    std::fill(offsets.begin(), offsets.end(), 0);
    const unsigned int shift = pass * radix_bits;
    for (const auto& entry : *source) {
      const auto bucket = static_cast<std::size_t>(
        (sortable_float_key(entry.value) >> shift) & bucket_mask
      );
      ++offsets[bucket];
    }
    std::size_t position = 0;
    for (std::size_t bucket = 0; bucket < offsets.size(); ++bucket) {
      const std::size_t count = offsets[bucket];
      offsets[bucket] = position;
      position += count;
    }
    for (const auto& entry : *source) {
      const auto bucket = static_cast<std::size_t>(
        (sortable_float_key(entry.value) >> shift) & bucket_mask
      );
      (*destination)[offsets[bucket]++] = entry;
    }
    std::swap(source, destination);
  }
  if (source != &entries) entries.swap(*source);
}

inline bool rank_values(std::vector<FloatRankEntry>& entries,
                        std::vector<double>& ranks) {
  constexpr std::size_t radix_threshold = 65536;
  if (entries.size() >= radix_threshold) {
    radix_sort_float_rank_entries(entries);
  } else {
    std::sort(entries.begin(), entries.end(),
              [](const FloatRankEntry& left, const FloatRankEntry& right) {
                return left.value < right.value;
              });
  }
  if (entries.front().value == entries.back().value) return false;
  for (std::size_t begin = 0; begin < entries.size();) {
    std::size_t end = begin + 1;
    while (end < entries.size() &&
           entries[end].value == entries[begin].value) {
      ++end;
    }
    const double rank =
      (static_cast<double>(begin) + static_cast<double>(end) + 1.0) / 2.0;
    for (std::size_t index = begin; index < end; ++index) {
      ranks[entries[index].pair] = rank;
    }
    begin = end;
  }
  return true;
}

}  // namespace detail

inline CorrelationResult spearman_correlation(const double* observed,
                                              const double* predicted,
                                              std::size_t size) {
  std::size_t pairs = 0;
  bool all_complete = true;
  for (std::size_t index = 0; index < size; ++index) {
    if (!std::isnan(observed[index]) && !std::isnan(predicted[index])) {
      ++pairs;
    } else {
      all_complete = false;
    }
  }
  if (pairs == 0) {
    return {NAN, 0, CorrelationStatus::no_complete_pairs};
  }
  if (pairs < 2) {
    return {NAN, pairs, CorrelationStatus::insufficient_pairs};
  }

  std::vector<detail::RankEntry> observed_entries;
  std::vector<std::size_t> complete_indices;
  observed_entries.reserve(pairs);
  if (!all_complete) complete_indices.reserve(pairs);
  for (std::size_t index = 0; index < size; ++index) {
    if (!std::isnan(observed[index]) && !std::isnan(predicted[index])) {
      observed_entries.push_back({observed[index], observed_entries.size()});
      if (!all_complete) complete_indices.push_back(index);
    }
  }

  std::vector<double> observed_ranks(pairs);
  if (!detail::rank_values(observed_entries, observed_ranks)) {
    return {NAN, pairs, CorrelationStatus::constant_input};
  }
  for (std::size_t index = 0; index < pairs; ++index) {
    const std::size_t source = all_complete ? index : complete_indices[index];
    observed_entries[index] = {predicted[source], index};
  }
  std::vector<double> predicted_ranks(pairs);
  if (!detail::rank_values(observed_entries, predicted_ranks)) {
    return {NAN, pairs, CorrelationStatus::constant_input};
  }

  const long double mean = (static_cast<long double>(pairs) + 1.0L) / 2.0L;
  long double cross = 0.0L;
  long double observed_sum = 0.0L;
  long double predicted_sum = 0.0L;
  for (std::size_t index = 0; index < pairs; ++index) {
    const long double left = observed_ranks[index] - mean;
    const long double right = predicted_ranks[index] - mean;
    cross += left * right;
    observed_sum += left * left;
    predicted_sum += right * right;
  }
  const long double denominator = std::sqrt(observed_sum * predicted_sum);
  if (!(denominator > 0.0L)) {
    return {NAN, pairs, CorrelationStatus::constant_input};
  }
  return {static_cast<double>(cross / denominator), pairs,
          CorrelationStatus::success};
}

inline CorrelationResult spearman_correlation(const float* observed,
                                              const float* predicted,
                                              std::size_t size) {
  if (size > static_cast<std::size_t>(
        std::numeric_limits<std::uint32_t>::max())) {
    throw std::length_error(
      "float32 Spearman correlation exceeds the supported vector length"
    );
  }
  std::size_t pairs = 0;
  bool all_complete = true;
  for (std::size_t index = 0; index < size; ++index) {
    if (std::isfinite(observed[index]) && std::isfinite(predicted[index])) {
      ++pairs;
    } else {
      all_complete = false;
    }
  }
  if (pairs == 0) {
    return {NAN, 0, CorrelationStatus::no_complete_pairs};
  }
  if (pairs < 2) {
    return {NAN, pairs, CorrelationStatus::insufficient_pairs};
  }

  std::vector<detail::FloatRankEntry> entries;
  std::vector<std::uint32_t> complete_indices;
  entries.reserve(pairs);
  if (!all_complete) complete_indices.reserve(pairs);
  for (std::size_t index = 0; index < size; ++index) {
    if (std::isfinite(observed[index]) && std::isfinite(predicted[index])) {
      entries.push_back({
        observed[index], static_cast<std::uint32_t>(entries.size())
      });
      if (!all_complete) {
        complete_indices.push_back(static_cast<std::uint32_t>(index));
      }
    }
  }
  std::vector<double> observed_ranks(pairs);
  if (!detail::rank_values(entries, observed_ranks)) {
    return {NAN, pairs, CorrelationStatus::constant_input};
  }
  for (std::size_t index = 0; index < pairs; ++index) {
    const std::size_t source = all_complete ? index : complete_indices[index];
    entries[index] = {
      predicted[source], static_cast<std::uint32_t>(index)
    };
  }
  std::vector<double> predicted_ranks(pairs);
  if (!detail::rank_values(entries, predicted_ranks)) {
    return {NAN, pairs, CorrelationStatus::constant_input};
  }

  const long double mean = (static_cast<long double>(pairs) + 1.0L) / 2.0L;
  long double cross = 0.0L;
  long double observed_sum = 0.0L;
  long double predicted_sum = 0.0L;
  for (std::size_t index = 0; index < pairs; ++index) {
    const long double left = observed_ranks[index] - mean;
    const long double right = predicted_ranks[index] - mean;
    cross += left * right;
    observed_sum += left * left;
    predicted_sum += right * right;
  }
  const long double denominator = std::sqrt(observed_sum * predicted_sum);
  if (!(denominator > 0.0L)) {
    return {NAN, pairs, CorrelationStatus::constant_input};
  }
  return {static_cast<double>(cross / denominator), pairs,
          CorrelationStatus::success};
}

struct RegressionMetrics {
  std::array<double, 12> values{};
};

template<class T>
struct RegressionMetricsReference {
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::size_t relative_pairs = 0;
  long double observed_tss = 0.0L;
  long double observed_sum = 0.0L;
  long double observed_square = 0.0L;
  long double rank_square = 0.0L;
  double observed_sd = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> ranks;
  bool all_finite = false;
  bool rank_valid = false;
};

template<class T>
RegressionMetricsReference<T> regression_metrics_reference(
    ConstMatrixView<T> observed,
    double relative_epsilon = std::numeric_limits<double>::epsilon()) {
  if (observed.empty() || !observed.contiguous()) {
    throw std::invalid_argument(
      "regression metric reference must be a non-empty contiguous matrix"
    );
  }
  RegressionMetricsReference<T> result;
  result.rows = observed.rows();
  result.columns = observed.columns();
  const std::size_t size = result.rows * result.columns;
  std::vector<long double> means(result.columns, 0.0L);
  for (std::size_t column = 0; column < result.columns; ++column) {
    for (std::size_t row = 0; row < result.rows; ++row) {
      const T value = observed(row, column);
      if (!std::isfinite(value)) return result;
      means[column] += static_cast<long double>(value);
      result.observed_sum += static_cast<long double>(value);
      result.observed_square += static_cast<long double>(value) * value;
      if (std::abs(static_cast<double>(value)) > relative_epsilon) {
        ++result.relative_pairs;
      }
    }
    means[column] /= static_cast<long double>(result.rows);
  }
  for (std::size_t column = 0; column < result.columns; ++column) {
    for (std::size_t row = 0; row < result.rows; ++row) {
      const long double centered =
        static_cast<long double>(observed(row, column)) - means[column];
      result.observed_tss += centered * centered;
    }
  }
  const long double count = static_cast<long double>(size);
  const long double flattened_tss = result.observed_square -
    result.observed_sum * result.observed_sum / count;
  result.observed_sd = size > 1 && flattened_tss >= 0.0L ?
    std::sqrt(static_cast<double>(flattened_tss / (count - 1.0L))) :
    std::numeric_limits<double>::quiet_NaN();

  result.ranks.resize(size);
  if constexpr (std::is_same<T, float>::value) {
    if (size > static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max())) {
      throw std::length_error(
        "float32 Spearman correlation exceeds the supported vector length"
      );
    }
    std::vector<detail::FloatRankEntry> entries(size);
    for (std::size_t index = 0; index < size; ++index) {
      entries[index] = {
        observed.data()[index], static_cast<std::uint32_t>(index)
      };
    }
    result.rank_valid = detail::rank_values(entries, result.ranks);
  } else {
    std::vector<detail::RankEntry> entries(size);
    for (std::size_t index = 0; index < size; ++index) {
      entries[index] = {
        static_cast<double>(observed.data()[index]), index
      };
    }
    result.rank_valid = detail::rank_values(entries, result.ranks);
  }
  if (result.rank_valid) {
    const long double mean = (count + 1.0L) / 2.0L;
    for (const double rank : result.ranks) {
      const long double centered = rank - mean;
      result.rank_square += centered * centered;
    }
  }
  result.all_finite = true;
  return result;
}

template<class T>
RegressionMetrics regression_metrics(
    ConstMatrixView<T> observed, ConstMatrixView<T> predicted,
    double cross_validated_q2,
    double relative_epsilon = std::numeric_limits<double>::epsilon()) {
  if (observed.rows() != predicted.rows() ||
      observed.columns() != predicted.columns() || observed.empty() ||
      !observed.contiguous() || !predicted.contiguous()) {
    throw std::invalid_argument(
      "regression metric matrices must be contiguous with matching dimensions"
    );
  }
  const std::size_t columns = observed.columns();
  const std::size_t size = observed.rows() * columns;
  std::vector<long double> observed_sums(columns, 0.0L);
  std::vector<std::size_t> observed_counts(columns, 0);
  std::size_t relative_pairs = 0;
  for (std::size_t column = 0; column < columns; ++column) {
    for (std::size_t row = 0; row < observed.rows(); ++row) {
      const T value = observed(row, column);
      const T estimate = predicted(row, column);
      if (!std::isfinite(value) || !std::isfinite(estimate)) continue;
      observed_sums[column] += static_cast<long double>(value);
      ++observed_counts[column];
      if (std::abs(static_cast<double>(value)) > relative_epsilon) {
        ++relative_pairs;
      }
    }
  }

  long double sse = 0.0L;
  long double absolute_error = 0.0L;
  long double error_sum = 0.0L;
  long double observed_tss = 0.0L;
  long double observed_sum = 0.0L;
  long double observed_square = 0.0L;
  long double predicted_sum = 0.0L;
  long double predicted_square = 0.0L;
  long double cross_sum = 0.0L;
  long double relative_sum = 0.0L;
  std::size_t complete = 0;
  std::size_t relative_count = 0;
  double median_relative = std::numeric_limits<double>::quiet_NaN();
  {
    std::vector<double> relative_values;
    relative_values.reserve(relative_pairs);
    for (std::size_t column = 0; column < columns; ++column) {
      const long double observed_mean = observed_counts[column] ?
        observed_sums[column] / observed_counts[column] :
        std::numeric_limits<long double>::quiet_NaN();
      for (std::size_t row = 0; row < observed.rows(); ++row) {
        const T value = observed(row, column);
        const T estimate = predicted(row, column);
        if (!std::isfinite(value) || !std::isfinite(estimate)) continue;
        const long double error =
          static_cast<long double>(estimate) - value;
        sse += error * error;
        absolute_error += std::abs(error);
        error_sum += error;
        observed_tss +=
          (static_cast<long double>(value) - observed_mean) *
          (static_cast<long double>(value) - observed_mean);
        observed_sum += value;
        observed_square += static_cast<long double>(value) * value;
        predicted_sum += estimate;
        predicted_square += static_cast<long double>(estimate) * estimate;
        cross_sum += static_cast<long double>(value) * estimate;
        if (std::abs(static_cast<double>(value)) > relative_epsilon) {
          const double relative = std::abs(
            static_cast<double>(error) / static_cast<double>(value)
          ) * 100.0;
          relative_sum += relative;
          relative_values.push_back(relative);
          ++relative_count;
        }
        ++complete;
      }
    }
    if (!relative_values.empty()) {
      const std::size_t middle = relative_values.size() / 2;
      std::nth_element(
        relative_values.begin(), relative_values.begin() + middle,
        relative_values.end()
      );
      median_relative = relative_values[middle];
      if (relative_values.size() % 2 == 0) {
        const double lower = *std::max_element(
          relative_values.begin(), relative_values.begin() + middle
        );
        median_relative = (lower + median_relative) / 2.0;
      }
    }
  }
  if (complete == 0) {
    throw std::invalid_argument("no complete regression pairs");
  }

  const long double count = static_cast<long double>(complete);
  const double rmsd = std::sqrt(static_cast<double>(sse / count));
  const long double pearson_left = observed_square -
    observed_sum * observed_sum / count;
  const long double pearson_right = predicted_square -
    predicted_sum * predicted_sum / count;
  const long double pearson_cross = cross_sum -
    observed_sum * predicted_sum / count;
  const long double pearson_denominator = std::sqrt(
    pearson_left * pearson_right
  );
  const double pearson = pearson_denominator > 0.0L ?
    static_cast<double>(pearson_cross / pearson_denominator) :
    std::numeric_limits<double>::quiet_NaN();
  const auto rank = spearman_correlation(
    observed.data(), predicted.data(), size
  );
  const double spearman = rank.status == CorrelationStatus::success ?
    rank.value : std::numeric_limits<double>::quiet_NaN();
  const long double flattened_tss = observed_square -
    observed_sum * observed_sum / count;
  const double observed_sd = complete > 1 && flattened_tss >= 0.0L ?
    std::sqrt(static_cast<double>(flattened_tss / (count - 1.0L))) :
    std::numeric_limits<double>::quiet_NaN();

  RegressionMetrics result;
  result.values = {
    static_cast<double>(complete),
    observed_tss > 0.0L ?
      1.0 - static_cast<double>(sse / observed_tss) :
      std::numeric_limits<double>::quiet_NaN(),
    cross_validated_q2,
    rmsd,
    rmsd,
    static_cast<double>(absolute_error / count),
    static_cast<double>(error_sum / count),
    median_relative,
    relative_count ? static_cast<double>(relative_sum / relative_count) :
      std::numeric_limits<double>::quiet_NaN(),
    std::isfinite(observed_sd) && rmsd > 0.0 ? observed_sd / rmsd :
      std::numeric_limits<double>::quiet_NaN(),
    pearson,
    spearman
  };
  return result;
}

template<class T>
RegressionMetrics regression_metrics(
    ConstMatrixView<T> observed, ConstMatrixView<T> predicted,
    double cross_validated_q2, const RegressionMetricsReference<T>& reference,
    double relative_epsilon = std::numeric_limits<double>::epsilon()) {
  if (observed.rows() != predicted.rows() ||
      observed.columns() != predicted.columns() || observed.empty() ||
      !observed.contiguous() || !predicted.contiguous() ||
      reference.rows != observed.rows() ||
      reference.columns != observed.columns()) {
    throw std::invalid_argument(
      "regression metric matrices must be contiguous with matching dimensions"
    );
  }
  if (!reference.all_finite) {
    return regression_metrics(
      observed, predicted, cross_validated_q2, relative_epsilon
    );
  }

  const std::size_t size = observed.rows() * observed.columns();
  long double sse = 0.0L;
  long double absolute_error = 0.0L;
  long double error_sum = 0.0L;
  long double predicted_sum = 0.0L;
  long double predicted_square = 0.0L;
  long double cross_sum = 0.0L;
  long double relative_sum = 0.0L;
  std::vector<double> relative_values;
  relative_values.reserve(reference.relative_pairs);
  for (std::size_t index = 0; index < size; ++index) {
    const T value = observed.data()[index];
    const T estimate = predicted.data()[index];
    if (!std::isfinite(estimate)) {
      return regression_metrics(
        observed, predicted, cross_validated_q2, relative_epsilon
      );
    }
    const long double error = static_cast<long double>(estimate) - value;
    sse += error * error;
    absolute_error += std::abs(error);
    error_sum += error;
    predicted_sum += estimate;
    predicted_square += static_cast<long double>(estimate) * estimate;
    cross_sum += static_cast<long double>(value) * estimate;
    if (std::abs(static_cast<double>(value)) > relative_epsilon) {
      const double relative = std::abs(
        static_cast<double>(error) / static_cast<double>(value)
      ) * 100.0;
      relative_sum += relative;
      relative_values.push_back(relative);
    }
  }

  double median_relative = std::numeric_limits<double>::quiet_NaN();
  if (!relative_values.empty()) {
    const std::size_t middle = relative_values.size() / 2;
    std::nth_element(
      relative_values.begin(), relative_values.begin() + middle,
      relative_values.end()
    );
    median_relative = relative_values[middle];
    if (relative_values.size() % 2 == 0) {
      const double lower = *std::max_element(
        relative_values.begin(), relative_values.begin() + middle
      );
      median_relative = (lower + median_relative) / 2.0;
    }
  }

  const long double count = static_cast<long double>(size);
  const double rmsd = std::sqrt(static_cast<double>(sse / count));
  const long double pearson_left = reference.observed_square -
    reference.observed_sum * reference.observed_sum / count;
  const long double pearson_right = predicted_square -
    predicted_sum * predicted_sum / count;
  const long double pearson_cross = cross_sum -
    reference.observed_sum * predicted_sum / count;
  const long double pearson_denominator = std::sqrt(
    pearson_left * pearson_right
  );
  const double pearson = pearson_denominator > 0.0L ?
    static_cast<double>(pearson_cross / pearson_denominator) :
    std::numeric_limits<double>::quiet_NaN();

  std::vector<double> predicted_ranks(size);
  bool predicted_rank_valid = false;
  if constexpr (std::is_same<T, float>::value) {
    std::vector<detail::FloatRankEntry> entries(size);
    for (std::size_t index = 0; index < size; ++index) {
      entries[index] = {
        predicted.data()[index], static_cast<std::uint32_t>(index)
      };
    }
    predicted_rank_valid = detail::rank_values(entries, predicted_ranks);
  } else {
    std::vector<detail::RankEntry> entries(size);
    for (std::size_t index = 0; index < size; ++index) {
      entries[index] = {
        static_cast<double>(predicted.data()[index]), index
      };
    }
    predicted_rank_valid = detail::rank_values(entries, predicted_ranks);
  }
  double spearman = std::numeric_limits<double>::quiet_NaN();
  if (reference.rank_valid && predicted_rank_valid &&
      reference.rank_square > 0.0L) {
    const long double mean = (count + 1.0L) / 2.0L;
    long double cross = 0.0L;
    long double predicted_rank_square = 0.0L;
    for (std::size_t index = 0; index < size; ++index) {
      const long double left = reference.ranks[index] - mean;
      const long double right = predicted_ranks[index] - mean;
      cross += left * right;
      predicted_rank_square += right * right;
    }
    const long double denominator = std::sqrt(
      reference.rank_square * predicted_rank_square
    );
    if (denominator > 0.0L) {
      spearman = static_cast<double>(cross / denominator);
    }
  }

  RegressionMetrics result;
  result.values = {
    static_cast<double>(size),
    reference.observed_tss > 0.0L ?
      1.0 - static_cast<double>(sse / reference.observed_tss) :
      std::numeric_limits<double>::quiet_NaN(),
    cross_validated_q2,
    rmsd,
    rmsd,
    static_cast<double>(absolute_error / count),
    static_cast<double>(error_sum / count),
    median_relative,
    relative_values.empty() ? std::numeric_limits<double>::quiet_NaN() :
      static_cast<double>(relative_sum / relative_values.size()),
    std::isfinite(reference.observed_sd) && rmsd > 0.0 ?
      reference.observed_sd / rmsd :
      std::numeric_limits<double>::quiet_NaN(),
    pearson,
    spearman
  };
  return result;
}

}  // namespace fastpls::core

#endif
