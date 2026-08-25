#include "duckdb/optimizer/learned_ce/estimator_gate.hpp"

namespace duckdb {

EstimatorGate::EstimatorGate(LearnedCEConfig config_p, optional_ptr<ONNXCardinalityModel> model_p)
    : config(std::move(config_p)), model(model_p) {
}

bool EstimatorGate::ShouldUseLearnedEstimate(LearnedCEMode mode) const {
	return mode == LearnedCEMode::LEARNED || mode == LearnedCEMode::HYBRID;
}

bool EstimatorGate::ShouldRunShadowInference(LearnedCEMode mode) const {
	return mode == LearnedCEMode::SHADOW || ShouldUseLearnedEstimate(mode);
}

bool EstimatorGate::IsEligible(const CEFeatures &features, string &reason) const {
	if (!config.enabled) {
		reason = "learned_ce_disabled";
		return false;
	}
	if (!model || !model->IsLoaded()) {
		reason = "model_unavailable";
		return false;
	}
	if (features.relation_count < 2) {
		reason = "unsupported_relation_count";
		return false;
	}
	reason.clear();
	return true;
}

} // namespace duckdb
