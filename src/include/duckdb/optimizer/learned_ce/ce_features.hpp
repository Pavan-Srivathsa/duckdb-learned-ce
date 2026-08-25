#pragma once

#include "duckdb/common/common.hpp"

namespace duckdb {

//! Version 1 feature schema for learned cardinality estimation.
struct CEFeatures {
	float log_native_estimate = 0;
	float log_left_cardinality = 0;
	float log_right_cardinality = 0;
	float log_left_ndv = 0;
	float log_right_ndv = 0;
	float ndv_ratio = 0;
	float relation_count = 0;
	float join_edge_count = 0;
	float equality_predicate_count = 0;
	float range_predicate_count = 0;
	float filter_count = 0;
	float join_graph_density = 0;
	float has_left_stats = 0;
	float has_right_stats = 0;

	static constexpr idx_t FEATURE_COUNT = 14;
	static constexpr idx_t FEATURE_SCHEMA_VERSION = 1;

	void ToFloatArray(float out[FEATURE_COUNT]) const;
};

} // namespace duckdb
