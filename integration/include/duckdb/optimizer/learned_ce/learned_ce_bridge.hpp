#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/optimizer/join_order/join_relation_set.hpp"
#include "duckdb/optimizer/learned_ce/ce_features.hpp"

namespace duckdb {

class ClientContext;
class JoinPredicateModel;
struct RelationStats;

//! Per-query runtime state for learned CE (cache, model, telemetry).
class LearnedCEQueryState : public ClientContextState {
public:
	void QueryEnd() override;
};

class LearnedCEBridge {
public:
	static double Apply(optional_ptr<ClientContext> context, JoinRelationSet &relation_set, double native_estimate,
	                    const JoinPredicateModel &predicate_model, vector<RelationStats> &relation_stats);
};

} // namespace duckdb
