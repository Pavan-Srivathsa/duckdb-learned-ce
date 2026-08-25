#include "duckdb/optimizer/learned_ce/learned_ce_config.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"

namespace duckdb {

LearnedCEMode LearnedCEConfig::ParseMode(const string &mode) {
	if (mode == "native") {
		return LearnedCEMode::NATIVE;
	}
	if (mode == "shadow") {
		return LearnedCEMode::SHADOW;
	}
	if (mode == "learned") {
		return LearnedCEMode::LEARNED;
	}
	if (mode == "hybrid") {
		return LearnedCEMode::HYBRID;
	}
	if (mode == "adaptive") {
		return LearnedCEMode::ADAPTIVE;
	}
	return LearnedCEMode::INVALID;
}

string LearnedCEConfig::ModeToString(LearnedCEMode mode) {
	switch (mode) {
	case LearnedCEMode::NATIVE:
		return "native";
	case LearnedCEMode::SHADOW:
		return "shadow";
	case LearnedCEMode::LEARNED:
		return "learned";
	case LearnedCEMode::HYBRID:
		return "hybrid";
	case LearnedCEMode::ADAPTIVE:
		return "adaptive";
	default:
		return "invalid";
	}
}

LearnedCEConfig LearnedCEConfig::FromContext(const ClientContext &context) {
	LearnedCEConfig result;
	result.enabled = Settings::Get<LearnedCeEnabledSetting>(context);
	if (!result.enabled) {
		result.mode = LearnedCEMode::NATIVE;
		return result;
	}
	result.mode = ParseMode(Settings::Get<LearnedCeModeSetting>(context));
	if (result.mode == LearnedCEMode::INVALID) {
		result.mode = LearnedCEMode::NATIVE;
	}
	result.model_path = Settings::Get<LearnedCeModelPathSetting>(context);
	result.confidence_threshold = Settings::Get<LearnedCeConfidenceThresholdSetting>(context);
	result.collect_telemetry = Settings::Get<LearnedCeCollectTelemetrySetting>(context);
	return result;
}

} // namespace duckdb
