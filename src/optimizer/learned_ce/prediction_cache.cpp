#include "duckdb/optimizer/learned_ce/prediction_cache.hpp"

namespace duckdb {

optional<double> PredictionCache::Get(const string &key) const {
	auto entry = cache.find(key);
	if (entry == cache.end()) {
		return optional<double>();
	}
	return entry->second;
}

void PredictionCache::Put(const string &key, double prediction) {
	cache[key] = prediction;
}

void PredictionCache::Clear() {
	cache.clear();
}

} // namespace duckdb
