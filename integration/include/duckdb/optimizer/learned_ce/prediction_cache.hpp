#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/unordered_map.hpp"

namespace duckdb {

class PredictionCache {
public:
	optional<double> Get(const string &key) const;
	void Put(const string &key, double prediction);
	void Clear();

private:
	unordered_map<string, double> cache;
};

} // namespace duckdb
