#include "duckdb/optimizer/learned_ce/ce_features.hpp"

#include <cmath>

namespace duckdb {

void CEFeatures::ToFloatArray(float out[FEATURE_COUNT]) const {
	out[0] = log_native_estimate;
	out[1] = log_left_cardinality;
	out[2] = log_right_cardinality;
	out[3] = log_left_ndv;
	out[4] = log_right_ndv;
	out[5] = ndv_ratio;
	out[6] = relation_count;
	out[7] = join_edge_count;
	out[8] = equality_predicate_count;
	out[9] = range_predicate_count;
	out[10] = filter_count;
	out[11] = join_graph_density;
	out[12] = has_left_stats;
	out[13] = has_right_stats;
}

} // namespace duckdb
