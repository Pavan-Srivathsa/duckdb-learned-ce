#include "duckdb/optimizer/learned_ce/feature_extractor.hpp"

#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/optimizer/join_order/join_predicate.hpp"
#include "duckdb/optimizer/relation_statistics/relation_statistics.hpp"

#include <cmath>

namespace duckdb {

static float SafeLog1p(double value) {
	if (value <= 0) {
		return 0;
	}
	return static_cast<float>(std::log1p(value));
}

static idx_t CountEqualityPredicates(const JoinPredicateModel &model) {
	idx_t count = 0;
	for (auto predicate : model.GetPredicates()) {
		if (predicate.get().IsEquivalencePredicate()) {
			count++;
		}
	}
	return count;
}

static idx_t CountRangePredicates(const JoinPredicateModel &model) {
	idx_t count = 0;
	for (auto predicate : model.GetPredicates()) {
		auto comparison = predicate.get().GetComparisonType();
		if (comparison == ExpressionType::COMPARE_LESSTHAN || comparison == ExpressionType::COMPARE_GREATERTHAN ||
		    comparison == ExpressionType::COMPARE_LESSTHANOREQUALTO ||
		    comparison == ExpressionType::COMPARE_GREATERTHANOREQUALTO || comparison == ExpressionType::COMPARE_BETWEEN) {
			count++;
		}
	}
	return count;
}

CEFeatures CEFeatureExtractor::Extract(const EstimationContext &context) {
	CEFeatures features;
	features.log_native_estimate = SafeLog1p(context.native_estimate);
	features.relation_count = static_cast<float>(context.relation_set.count);

	if (context.predicate_model) {
		auto &model = *context.predicate_model;
		features.join_edge_count = static_cast<float>(model.GetGraphPredicates().size());
		features.equality_predicate_count = static_cast<float>(CountEqualityPredicates(model));
		features.range_predicate_count = static_cast<float>(CountRangePredicates(model));
		features.filter_count = static_cast<float>(model.GetSelectivityPredicates().size());
		if (features.relation_count > 1) {
			features.join_graph_density =
			    features.join_edge_count / (features.relation_count * (features.relation_count - 1));
		}
	}

	if (context.relation_stats) {
		double left_card = 0;
		double right_card = 0;
		for (auto &stats : *context.relation_stats) {
			if (!stats.stats_initialized) {
				continue;
			}
			left_card = MaxValue(left_card, static_cast<double>(stats.cardinality));
			right_card = MinValue(right_card == 0 ? static_cast<double>(stats.cardinality) : right_card,
			                      static_cast<double>(stats.cardinality));
			if (right_card == 0) {
				right_card = static_cast<double>(stats.cardinality);
			}
		}
		features.log_left_cardinality = SafeLog1p(left_card);
		features.log_right_cardinality = SafeLog1p(right_card);
		features.has_left_stats = left_card > 0 ? 1.0f : 0.0f;
		features.has_right_stats = right_card > 0 ? 1.0f : 0.0f;
	}

	return features;
}

} // namespace duckdb
