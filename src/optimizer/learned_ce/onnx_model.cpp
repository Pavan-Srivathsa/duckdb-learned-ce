#include "duckdb/optimizer/learned_ce/onnx_model.hpp"

#include <cmath>

namespace duckdb {

ONNXCardinalityModel::ONNXCardinalityModel(const string &model_path_p) : model_path(model_path_p) {
	loaded = !model_path.empty();
}

bool ONNXCardinalityModel::IsLoaded() const {
	return loaded;
}

double ONNXCardinalityModel::Predict(const CEFeatures &features) const {
	if (!loaded) {
		return 0;
	}
	// Stub: identity on log-native until ONNX Runtime is wired (Milestone 5).
	return std::expm1(static_cast<double>(features.log_native_estimate));
}

} // namespace duckdb
