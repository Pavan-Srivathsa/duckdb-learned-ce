#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/optimizer/learned_ce/ce_features.hpp"
#include "duckdb/optimizer/learned_ce/learned_ce_config.hpp"
#include "duckdb/optimizer/learned_ce/onnx_model.hpp"

namespace duckdb {

class EstimatorGate {
public:
	EstimatorGate(LearnedCEConfig config, optional_ptr<ONNXCardinalityModel> model);

	bool IsEligible(const CEFeatures &features, string &reason) const;
	bool ShouldUseLearnedEstimate(LearnedCEMode mode) const;
	bool ShouldRunShadowInference(LearnedCEMode mode) const;

private:
	LearnedCEConfig config;
	optional_ptr<ONNXCardinalityModel> model;
};

} // namespace duckdb
