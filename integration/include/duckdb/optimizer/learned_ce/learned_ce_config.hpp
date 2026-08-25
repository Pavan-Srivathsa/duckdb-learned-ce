#pragma once

#include "duckdb/common/common.hpp"

namespace duckdb {

class ClientContext;

enum class LearnedCEMode : uint8_t { NATIVE, SHADOW, LEARNED, HYBRID, ADAPTIVE, INVALID };

struct LearnedCEConfig {
	bool enabled = false;
	LearnedCEMode mode = LearnedCEMode::NATIVE;
	string model_path;
	double confidence_threshold = 0.5;
	bool collect_telemetry = false;

	static LearnedCEConfig FromContext(const ClientContext &context);
	static LearnedCEMode ParseMode(const string &mode);
	static string ModeToString(LearnedCEMode mode);
};

} // namespace duckdb
