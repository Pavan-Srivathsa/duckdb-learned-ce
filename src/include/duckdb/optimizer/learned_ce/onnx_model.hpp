#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/optimizer/learned_ce/ce_features.hpp"

namespace duckdb {

//! ONNX Runtime wrapper (stub until Milestone 5).
class ONNXCardinalityModel {
public:
	explicit ONNXCardinalityModel(const string &model_path);

	double Predict(const CEFeatures &features) const;
	bool IsLoaded() const;

private:
	string model_path;
	bool loaded = false;
};

} // namespace duckdb
