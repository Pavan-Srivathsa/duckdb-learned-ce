#include "catch.hpp"
#include "duckdb/optimizer/learned_ce/ce_features.hpp"
#include "duckdb/optimizer/learned_ce/estimator_gate.hpp"
#include "duckdb/optimizer/learned_ce/learned_cardinality_estimator.hpp"
#include "duckdb/optimizer/learned_ce/learned_ce_config.hpp"
#include "duckdb/optimizer/learned_ce/onnx_model.hpp"
#include "duckdb/optimizer/learned_ce/prediction_cache.hpp"

using namespace duckdb;

TEST_CASE("Learned CE native mode preserves native estimate", "[optimizer][learned_ce]") {
	LearnedCEConfig config;
	config.enabled = true;
	config.mode = LearnedCEMode::NATIVE;

	CEFeatures features;
	features.relation_count = 2;
	features.log_native_estimate = 10;

	LearnedCardinalityEstimator estimator(config);
	auto prediction = estimator.Estimate(features, 12345);
	REQUIRE(prediction.estimate == 12345);
	REQUIRE(!prediction.used_learned_model);
}

TEST_CASE("Learned CE shadow mode keeps native estimate", "[optimizer][learned_ce]") {
	LearnedCEConfig config;
	config.enabled = true;
	config.mode = LearnedCEMode::SHADOW;
	config.model_path = "artifacts/model.onnx";

	CEFeatures features;
	features.relation_count = 2;
	features.log_native_estimate = 8;

	LearnedCardinalityEstimator estimator(config);
	auto prediction = estimator.Estimate(features, 5000);
	REQUIRE(prediction.estimate == 5000);
	REQUIRE(!prediction.used_learned_model);
	REQUIRE(prediction.fallback_reason == "shadow_mode");
}

TEST_CASE("Learned CE prediction cache", "[optimizer][learned_ce]") {
	PredictionCache cache;
	REQUIRE(!cache.Get("a").IsValid());
	cache.Put("a", 42);
	REQUIRE(cache.Get("a").IsValid());
	REQUIRE(*cache.Get("a") == 42);
}

TEST_CASE("Learned CE gate rejects unsupported relation count", "[optimizer][learned_ce]") {
	LearnedCEConfig config;
	config.enabled = true;
	ONNXCardinalityModel model("artifacts/model.onnx");
	EstimatorGate gate(config, model);

	CEFeatures features;
	features.relation_count = 1;
	string reason;
	REQUIRE(!gate.IsEligible(features, reason));
	REQUIRE(reason == "unsupported_relation_count");
}
