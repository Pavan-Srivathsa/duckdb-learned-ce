add_library_unity(
  duckdb_optimizer_learned_ce
  OBJECT
  ce_features.cpp
  learned_ce_config.cpp
  feature_extractor.cpp
  learned_cardinality_estimator.cpp
  onnx_model.cpp
  estimator_gate.cpp
  prediction_cache.cpp
  telemetry.cpp
  adaptive_selector.cpp
  learned_ce_bridge.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_optimizer_learned_ce>
    PARENT_SCOPE)
