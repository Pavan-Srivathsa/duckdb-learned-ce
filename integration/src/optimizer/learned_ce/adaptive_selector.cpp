#include "duckdb/optimizer/learned_ce/adaptive_selector.hpp"

#include "duckdb/optimizer/learned_ce/ce_features.hpp"

namespace duckdb {

LearnedCEMode AdaptiveSelector::SelectMode(const CEFeatures &features, LearnedCEMode configured_mode) const {
	(void)features;
	return configured_mode;
}

} // namespace duckdb
