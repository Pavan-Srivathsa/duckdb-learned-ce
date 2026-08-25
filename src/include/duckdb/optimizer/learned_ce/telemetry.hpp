#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/optimizer/learned_ce/ce_features.hpp"

namespace duckdb {

struct LearnedCETelemetryRecord {
	string query_id;
	string candidate;
	double native_estimate = 0;
	double learned_estimate = 0;
	optional_idx actual_cardinality;
	idx_t inference_us = 0;
};

class LearnedCETelemetry {
public:
	void Record(LearnedCETelemetryRecord record);
	void FlushToFile(const string &path) const;
	void Clear();

private:
	vector<LearnedCETelemetryRecord> records;
};

} // namespace duckdb
