#include "duckdb/optimizer/learned_ce/learned_cardinality_estimator.hpp"

#include "duckdb/optimizer/learned_ce/adaptive_selector.hpp"
#include "duckdb/optimizer/learned_ce/estimator_gate.hpp"
#include "duckdb/optimizer/learned_ce/onnx_model.hpp"

#include <cmath>

namespace duckdb {

static double ClampPrediction(double value) {
	if (!std::isfinite(value) || value < 1) {
		return 1;
	}
	constexpr double MAX_CARDINALITY = 1e18;
	return MinValue(value, MAX_CARDINALITY);
}

LearnedCardinalityEstimator::LearnedCardinalityEstimator(LearnedCEConfig config_p) : config(std::move(config_p)) {
}

CardinalityPrediction LearnedCardinalityEstimator::Estimate(const CEFeatures &features,
                                                            double native_estimate) const {
	CardinalityPrediction result;
	result.estimate = native_estimate;

	if (!config.enabled || config.mode == LearnedCEMode::NATIVE) {
		result.fallback_reason = "native_mode";
		return result;
	}

	AdaptiveSelector selector;
	auto effective_mode = selector.SelectMode(features, config.mode);
	ONNXCardinalityModel model(config.model_path);
	EstimatorGate gate(config, model);

	string eligibility_reason;
	const bool eligible = gate.IsEligible(features, eligibility_reason);
	const bool run_shadow = gate.ShouldRunShadowInference(effective_mode);
	const bool use_learned = gate.ShouldUseLearnedEstimate(effective_mode) && eligible;

	if (run_shadow || use_learned) {
		auto model_output = model.Predict(features);
		result.learned_estimate = ClampPrediction(model_output);
		result.confidence = eligible ? 1.0 : 0.0;
	}

	if (use_learned && result.confidence >= config.confidence_threshold) {
		result.estimate = result.learned_estimate;
		result.used_learned_model = true;
		return result;
	}

	if (!eligible) {
		result.fallback_reason = eligibility_reason;
	} else if (effective_mode == LearnedCEMode::SHADOW) {
		result.fallback_reason = "shadow_mode";
	} else {
		result.fallback_reason = "hybrid_gate";
	}
	return result;
}

} // namespace duckdb
