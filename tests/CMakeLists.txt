# ABOUTME: Builds ACE-Step unit and integration test executables.
# ABOUTME: Links tests against the same GGML backends used by production binaries.

# test-philox: Philox RNG reference dumper.
add_executable(test-philox test-philox.cpp)
target_include_directories(test-philox PRIVATE ${CMAKE_SOURCE_DIR}/src)
if(NOT MSVC)
    target_link_libraries(test-philox PRIVATE m)
endif()

# test-model-store: exercise ModelStore load/evict policies against real GGUFs.
add_executable(test-model-store test-model-store.cpp)
target_link_libraries(test-model-store PRIVATE acestep-core)
link_ggml_backends(test-model-store)

# test-lm-prompt: byte for byte check of Phase 2 LM prompt builders
# against the Qwen3 chat template reference.
add_executable(test-lm-prompt test-lm-prompt.cpp)
target_link_libraries(test-lm-prompt PRIVATE acestep-core)
link_ggml_backends(test-lm-prompt)

# test-dit-linear-transform: verify the DiT graph's functional adapter seam.
add_executable(test-dit-linear-transform test-dit-linear-transform.cpp)
target_link_libraries(test-dit-linear-transform PRIVATE acestep-core)
link_ggml_backends(test-dit-linear-transform)

# test-train-adapter-state: verify Gary-compatible target profiles and initialization.
add_executable(test-train-adapter-state test-train-adapter-state.cpp)
target_link_libraries(test-train-adapter-state PRIVATE acestep-core)
link_ggml_backends(test-train-adapter-state)
