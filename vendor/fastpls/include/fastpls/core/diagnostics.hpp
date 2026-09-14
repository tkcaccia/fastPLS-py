// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_DIAGNOSTICS_HPP
#define FASTPLS_CORE_DIAGNOSTICS_HPP

namespace fastpls {
namespace core {

struct RSVDAuditSummary {
  int solves = 0;
  int certified = 0;
  int deterministic_fallbacks = 0;
  int failures = 0;
  int max_attempts = 0;
  int max_effective_oversample = 0;
  int max_effective_power_iters = 0;
  double max_triplet_residual = 0.0;
  double max_omitted_direction_ratio = 0.0;
};

}  // namespace core
}  // namespace fastpls

#endif
