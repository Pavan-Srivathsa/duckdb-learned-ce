#include "duckdb/optimizer/learned_ce/learned_ce_bridge.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/optimizer/join_order/join_predicate.hpp"
#include "duckdb/optimizer/learned_ce/feature_extractor.hpp"
#include "duckdb/optimizer/learned_ce/learned_cardinality_estimator.hpp"
#include "duckdb/optimizer/learned_ce/learned_ce_config.hpp"
#include "duckdb/optimizer/learned_ce/prediction_cache.hpp"
#include "duckdb/optimizer/learned_ce/telemetry.hpp"

namespace duckdb {

static constexpr const char *LEARNED_CE_STATE_KEY = "learned_ce_state";

struct LearnedCEState : public ClientContextState {
	PredictionCache cache;
	LearnedCETelemetry telemetry;

	void QueryEnd() override {
		cache.Clear();
		telemetry.Clear();
	}
};

void LearnedCEQueryState::QueryEnd() {
}

double LearnedCEBridge::Apply(optional_ptr<ClientContext> context, JoinRelationSet &relation_set, double native_estimate,
                              const JoinPredicateModel &predicate_model, vector<RelationStats> &relation_stats) {
	if (!context) {
		return native_estimate;
	}

	auto config = LearnedCEConfig::FromContext(*context);
	if (!config.enabled) {
		return native_estimate;
	}

	auto state = context->registered_state->GetOrCreate<LearnedCEState>(LEARNED_CE_STATE_KEY);
	auto cache_key = relation_set.ToString();

	if (auto cached = state->cache.Get(cache_key)) {
		if (config.mode == LearnedCEMode::NATIVE) {
			return native_estimate;
		}
		if (config.mode == LearnedCEMode::SHADOW) {
			return native_estimate;
		}
		return *cached;
	}

	EstimationContext estimation_context;
	estimation_context.relation_set = relation_set;
	estimation_context.native_estimate = native_estimate;
	estimation_context.predicate_model = optional_ptr<JoinPredicateModel>(
	    const_cast<JoinPredicateModel *>(&predicate_model));
	estimation_context.relation_stats = &relation_stats;

	auto features = CEFeatureExtractor::Extract(estimation_context);
	LearnedCardinalityEstimator estimator(config);
	auto prediction = estimator.Estimate(features, native_estimate);

	state->cache.Put(cache_key, prediction.estimate);

	if (config.collect_telemetry || config.mode == LearnedCEMode::SHADOW) {
		LearnedCETelemetryRecord record;
		record.candidate = cache_key;
		record.native_estimate = native_estimate;
		record.learned_estimate = prediction.learned_estimate > 0 ? prediction.learned_estimate : prediction.estimate;
		state->telemetry.Record(std::move(record));
	}

	return prediction.estimate;
}

} // namespace duckdb
