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

# test-train-window: verify deterministic latent cropping and conditioning preservation.
add_executable(test-train-window test-train-window.cpp)
target_link_libraries(test-train-window PRIVATE acestep-core)
link_ggml_backends(test-train-window)

# test-train-adapter-state: verify Gary-compatible target profiles and initialization.
add_executable(test-train-adapter-state test-train-adapter-state.cpp)
target_link_libraries(test-train-adapter-state PRIVATE acestep-core)
link_ggml_backends(test-train-adapter-state)

# Register every self-contained test with CTest. Model-backed tests are
# enabled when ACE_TEST_MODELS_DIR points at a complete registry.
function(ace_register_test target)
    add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}> ${ARGN})
    set_tests_properties(${target} PROPERTIES
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        LABELS "ace;unit")
endfunction()

ace_register_test(test-philox)
ace_register_test(test-dit-linear-transform)
ace_register_test(test-dora-backend)
ace_register_test(test-adapter-merge)
ace_register_test(test-vae-latent-sampling)
ace_register_test(test-train-dataset)
ace_register_test(test-train-window)
ace_register_test(test-train-adapter-state)

set_tests_properties(
    test-dora-backend
    test-adapter-merge
    test-vae-latent-sampling
    test-train-dataset
    test-train-window
    test-train-adapter-state
    PROPERTIES LABELS "ace;training;unit")

set(ACE_TEST_MODELS_DIR "" CACHE PATH "Model registry used by ACE model-backed CTests")
if(ACE_TEST_MODELS_DIR)
    ace_register_test(test-model-store --models "${ACE_TEST_MODELS_DIR}")
    ace_register_test(test-lm-prompt --models "${ACE_TEST_MODELS_DIR}")
    set_tests_properties(test-model-store test-lm-prompt PROPERTIES LABELS "ace;models;integration")
else()
    message(STATUS "ACE_TEST_MODELS_DIR is empty; model-backed CTests are not registered")
endif()

option(ACE_TEST_CUDA "Register CUDA DoRA backend execution tests (requires a CUDA GPU at test time)" OFF)
set(ACE_TEST_CUDA_BACKEND "CUDA0" CACHE STRING "GGML CUDA backend name used by ACE CUDA CTests")
if(ACE_TEST_CUDA AND GGML_CUDA)
    foreach(weight_type F32 Q8_0 Q4_K Q5_K Q6_K)
        string(TOLOWER "${weight_type}" weight_type_lower)
        set(test_name "test-dora-backend-cuda-${weight_type_lower}")
        add_test(NAME ${test_name}
                 COMMAND $<TARGET_FILE:test-dora-backend> "${ACE_TEST_CUDA_BACKEND}" "${weight_type}")
        set_tests_properties(${test_name} PROPERTIES
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            LABELS "ace;cuda;training;integration")
    endforeach()
endif()
