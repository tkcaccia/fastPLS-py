// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_FOLDS_HPP
#define FASTPLS_CORE_FOLDS_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace fastpls {
namespace core {

namespace detail {

template<class Draw>
std::vector<std::size_t> ordered_sample(std::size_t count, Draw&& draw) {
  std::vector<std::size_t> pool(count);
  for (std::size_t index = 0; index < count; ++index) pool[index] = index;
  std::vector<std::size_t> output(count);
  for (std::size_t selected = 0; selected < count; ++selected) {
    const std::size_t remaining = count - selected;
    std::size_t index = draw(remaining);
    if (index >= remaining) index = remaining - 1;
    output[selected] = pool[index];
    pool[index] = pool[remaining - 1];
  }
  return output;
}

}  // namespace detail

template<class Group, class Draw>
std::vector<int> grouped_folds(const Group* groups, std::size_t sample_count,
                               const int* labels, std::size_t class_count,
                               int fold_count, Draw&& draw) {
  if (groups == nullptr || sample_count < 2) {
    throw std::invalid_argument(
      "fastPLS grouped folds require at least two samples"
    );
  }
  std::vector<Group> unique(groups, groups + sample_count);
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  if (unique.empty()) {
    throw std::invalid_argument("fastPLS grouped folds contain no groups");
  }
  std::unordered_map<Group, std::size_t> group_index;
  group_index.reserve(unique.size());
  for (std::size_t index = 0; index < unique.size(); ++index) {
    group_index.emplace(unique[index], index);
  }
  std::vector<std::size_t> mapped(sample_count);
  std::vector<std::size_t> first(unique.size(), sample_count);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const std::size_t index = group_index.at(groups[sample]);
    mapped[sample] = index;
    if (first[index] == sample_count) first[index] = sample;
  }

  const bool leave_one_group_out = fold_count < 0 ||
    fold_count >= static_cast<int>(unique.size());
  if (leave_one_group_out) fold_count = static_cast<int>(unique.size());
  fold_count = std::max(fold_count, 1);
  std::vector<int> group_fold(unique.size(), 0);
  if (leave_one_group_out) {
    for (std::size_t index = 0; index < unique.size(); ++index) {
      group_fold[index] = static_cast<int>(index);
    }
  } else if (labels != nullptr && class_count > 1) {
    std::vector<std::vector<std::size_t>> by_class(class_count);
    std::vector<std::size_t> class_order;
    std::vector<unsigned char> class_seen(class_count, 0);
    for (std::size_t index = 0; index < unique.size(); ++index) {
      const int label = labels[first[index]];
      const std::size_t encoded = static_cast<std::size_t>(
        std::max(1, std::min(label, static_cast<int>(class_count))) - 1
      );
      if (!class_seen[encoded]) {
        class_seen[encoded] = 1;
        class_order.push_back(encoded);
      }
      by_class[encoded].push_back(index);
    }
    for (const std::size_t encoded : class_order) {
      const auto& members = by_class[encoded];
      if (members.empty()) continue;
      const auto order = detail::ordered_sample(members.size(), draw);
      for (std::size_t position = 0; position < order.size(); ++position) {
        group_fold[members[order[position]]] =
          static_cast<int>(position % static_cast<std::size_t>(fold_count));
      }
    }
  } else {
    const auto order = detail::ordered_sample(unique.size(), draw);
    for (std::size_t position = 0; position < order.size(); ++position) {
      group_fold[order[position]] = static_cast<int>(
        position % static_cast<std::size_t>(fold_count)
      );
    }
  }
  std::vector<int> result(sample_count);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    result[sample] = group_fold[mapped[sample]] + 1;
  }
  return result;
}

}  // namespace core
}  // namespace fastpls

#endif
