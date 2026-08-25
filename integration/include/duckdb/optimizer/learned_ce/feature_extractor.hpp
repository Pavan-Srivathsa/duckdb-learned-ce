#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/optimizer/join_order/join_relation_set.hpp"
#include "duckdb/optimizer/learned_ce/ce_features.hpp"
#include "duckdb/optimizer/relation_statistics/relation_statistics_helper.hpp"

namespace duckdb {

class JoinPredicateModel;

struct EstimationContext {
	JoinRelationSet &relation_set;
	double native_estimate = 0;
	optional_ptr<JoinPredicateModel> predicate_model;
	optional_ptr<vector<RelationStats>> relation_stats;
};

class CEFeatureExtractor {
public:
	static CEFeatures Extract(const EstimationContext &context);
};

} // namespace duckdb
