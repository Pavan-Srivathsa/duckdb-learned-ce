#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/optimizer/learned_ce/ce_features.hpp"
#include "duckdb/optimizer/learned_ce/learned_ce_config.hpp"

namespace duckdb {

//! Thompson Sampling adaptive estimator selector (stub until Milestone 10).
class AdaptiveSelector {
public:
	LearnedCEMode SelectMode(const CEFeatures &features, LearnedCEMode configured_mode) const;
};

} // namespace duckdb
