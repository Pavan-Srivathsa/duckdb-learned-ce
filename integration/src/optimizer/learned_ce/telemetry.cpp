#include "duckdb/optimizer/learned_ce/telemetry.hpp"

#include "duckdb/common/file_system.hpp"
#include "duckdb/common/local_file_system.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

void LearnedCETelemetry::Record(LearnedCETelemetryRecord record) {
	records.push_back(std::move(record));
}

void LearnedCETelemetry::Clear() {
	records.clear();
}

void LearnedCETelemetry::FlushToFile(const string &path) const {
	if (records.empty() || path.empty()) {
		return;
	}
	string payload;
	for (auto &record : records) {
		payload += StringUtil::Format(
		    "{\"candidate\":\"%s\",\"native_estimate\":%.6f,\"learned_estimate\":%.6f,\"inference_us\":%llu}\n",
		    record.candidate, record.native_estimate, record.learned_estimate, record.inference_us);
	}
	LocalFileSystem fs;
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE |
	                                      FileFlags::FILE_FLAGS_APPEND);
	fs.Write(*handle, (void *)payload.data(), payload.size());
	handle->Close();
}

} // namespace duckdb
