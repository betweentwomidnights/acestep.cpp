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

# test-dora-backend: execute a real DoRA-row forward/backward graph on a selected backend.
add_executable(test-dora-backend test-dora-backend.cpp)
target_link_libraries(test-dora-backend PRIVATE acestep-core)
link_ggml_backends(test-dora-backend)

# test-adapter-merge: reload emitted PEFT DoRA adapters through native inference.
add_executable(test-adapter-merge test-adapter-merge.cpp)
target_link_libraries(test-adapter-merge PRIVATE acestep-core)
link_ggml_backends(test-adapter-merge)

# test-vae-latent-sampling: verify training-only posterior sampling from VAE encoder output.
add_executable(test-vae-latent-sampling test-vae-latent-sampling.cpp)
target_link_libraries(test-vae-latent-sampling PRIVATE acestep-core)
link_ggml_backends(test-vae-latent-sampling)

# test-train-dataset: verify paired WAV/metadata discovery and conditioning fields.
add_executable(test-train-dataset test-train-dataset.cpp)
target_link_libraries(test-train-dataset PRIVATE acestep-core)
link_ggml_backends(test-train-dataset)

# test-train-adapter-state: verify Gary-compatible target profiles and initialization.
add_executable(test-train-adapter-state test-train-adapter-state.cpp)
target_link_libraries(test-train-adapter-state PRIVATE acestep-core)
link_ggml_backends(test-train-adapter-state)
