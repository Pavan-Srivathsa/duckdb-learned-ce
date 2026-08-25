#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/optimizer/learned_ce/ce_features.hpp"
#include "duckdb/optimizer/learned_ce/learned_ce_config.hpp"

namespace duckdb {

struct CardinalityPrediction {
	double estimate = 0;
	double learned_estimate = 0;
	double confidence = 0;
	bool used_learned_model = false;
	string fallback_reason;
};

class LearnedCardinalityEstimator {
public:
	explicit LearnedCardinalityEstimator(LearnedCEConfig config);

	CardinalityPrediction Estimate(const CEFeatures &features, double native_estimate) const;

private:
	LearnedCEConfig config;
};

} // namespace duckdb
