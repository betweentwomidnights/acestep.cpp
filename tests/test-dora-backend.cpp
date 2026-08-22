// ABOUTME: Executes a real DoRA-row forward and backward graph on a selected GGML backend.
// ABOUTME: Verifies output values and trainable gradients through backend-resident tensors.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "train-adapter.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static ggml_backend_t open_backend(const char * name) {
    if (!name || std::strcmp(name, "CPU") == 0) {
        return ggml_backend_cpu_init();
    }
    ggml_backend_load_all();
    return ggml_backend_init_by_name(name, nullptr);
}

static void print_available_backends() {
    ggml_backend_load_all();
    std::fputs("Available devices:", stderr);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        std::fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
    }
    std::fputc('\n', stderr);
}

static bool nearly_equal(float actual, float expected) {
    return std::fabs(actual - expected) < 1e-4f;
}

int main(int argc, char ** argv) {
    const char * backend_name = argc > 1 ? argv[1] : "CPU";
    ggml_backend_t backend = open_backend(backend_name);
    if (!backend) {
        std::fprintf(stderr, "failed to initialize backend device: %s\n", backend_name);
        print_available_backends();
        return 1;
    }

    ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fputs("failed to create DoRA backend test context\n", stderr);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ggml_tensor * adapter_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ggml_tensor * adapter_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2);
    ggml_tensor * magnitude = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor * base_norm_sq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_set_input(weight);
    ggml_set_input(input);
    ggml_set_input(base_norm_sq);
    ggml_set_param(adapter_a);
    ggml_set_param(adapter_b);
    ggml_set_param(magnitude);
    ggml_set_input(adapter_a);
    ggml_set_input(adapter_b);
    ggml_set_input(magnitude);

    DiTAdapterTransform adapters;
    adapters.params.emplace(weight, DiTAdapterParam {
        /*.a =*/adapter_a,
        /*.b =*/adapter_b,
        /*.magnitude =*/magnitude,
        /*.base_norm_sq =*/base_norm_sq,
        /*.scale =*/0.5f,
        /*.dora_rows =*/true,
    });
    ggml_tensor * transformed = dit_adapter_linear_transform(ctx, &adapters, weight, input);
    ggml_tensor * loss = ggml_sum(ctx, ggml_sqr(ctx, transformed));
    ggml_set_loss(loss);
    ggml_set_output(transformed);
    ggml_set_output(loss);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(graph, loss);
    ggml_build_backward_expand(ctx, graph, nullptr);
    ggml_tensor * adapter_a_grad = ggml_graph_get_grad(graph, adapter_a);
    ggml_tensor * adapter_b_grad = ggml_graph_get_grad(graph, adapter_b);
    ggml_tensor * magnitude_grad = ggml_graph_get_grad(graph, magnitude);
    if (!adapter_a_grad || !adapter_b_grad || !magnitude_grad) {
        std::fputs("DoRA backend graph did not produce all trainable gradients\n", stderr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    ggml_set_output(adapter_a_grad);
    ggml_set_output(adapter_b_grad);
    ggml_set_output(magnitude_grad);

    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        std::fputs("failed to allocate DoRA backend graph\n", stderr);
        if (allocator) {
            ggml_gallocr_free(allocator);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    const float weight_data[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float input_data[] = { 2.0f, -1.0f };
    const float adapter_a_data[] = { 0.5f, -1.0f };
    const float adapter_b_data[] = { 2.0f, -0.5f };
    const float magnitude_data[] = { 1.0f, 2.0f };
    const float base_norm_sq_data[] = { 5.0f, 25.0f };
    ggml_backend_tensor_set(weight, weight_data, 0, sizeof(weight_data));
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    ggml_backend_tensor_set(adapter_a, adapter_a_data, 0, sizeof(adapter_a_data));
    ggml_backend_tensor_set(adapter_b, adapter_b_data, 0, sizeof(adapter_b_data));
    ggml_backend_tensor_set(magnitude, magnitude_data, 0, sizeof(magnitude_data));
    ggml_backend_tensor_set(base_norm_sq, base_norm_sq_data, 0, sizeof(base_norm_sq_data));

    ggml_graph_reset(graph);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "DoRA graph execution failed on %s\n", backend_name);
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    const float expected_0 = 2.0f / std::sqrt(3.25f);
    const float expected_1 = 3.0f / std::sqrt(26.328125f);
    const float expected_loss = expected_0 * expected_0 + expected_1 * expected_1;
    float actual_loss = 0.0f;
    ggml_backend_tensor_get(loss, &actual_loss, 0, sizeof(actual_loss));
    if (!nearly_equal(actual_loss, expected_loss)) {
        std::fprintf(stderr,
                     "DoRA loss mismatch on %s: got %f, expected %f\n",
                     backend_name,
                     actual_loss,
                     expected_loss);
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    for (ggml_tensor * gradient : { adapter_a_grad, adapter_b_grad, magnitude_grad }) {
        std::vector<float> values((size_t) ggml_nelements(gradient));
        ggml_backend_tensor_get(gradient, values.data(), 0, values.size() * sizeof(float));
        bool has_signal = false;
        for (float value : values) {
            if (!std::isfinite(value)) {
                std::fprintf(stderr, "DoRA produced a non-finite gradient on %s\n", backend_name);
                ggml_gallocr_free(allocator);
                ggml_free(ctx);
                ggml_backend_free(backend);
                return 1;
            }
            has_signal = has_signal || std::fabs(value) > 1e-8f;
        }
        if (!has_signal) {
            std::fprintf(stderr, "DoRA produced an empty gradient on %s\n", backend_name);
            ggml_gallocr_free(allocator);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 1;
        }
    }

    std::printf("DoRA-row backend forward/backward: OK (%s)\n", ggml_backend_name(backend));
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
