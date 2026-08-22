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
    return std::fabs(actual - expected) < 1e-3f * std::fmax(1.0f, std::fabs(expected));
}

static bool parse_weight_type(const char * name, ggml_type & type) {
    if (std::strcmp(name, "F32") == 0) {
        type = GGML_TYPE_F32;
        return true;
    }
    if (std::strcmp(name, "Q8_0") == 0) {
        type = GGML_TYPE_Q8_0;
        return true;
    }
    if (std::strcmp(name, "Q4_K") == 0) {
        type = GGML_TYPE_Q4_K;
        return true;
    }
    if (std::strcmp(name, "Q5_K") == 0) {
        type = GGML_TYPE_Q5_K;
        return true;
    }
    if (std::strcmp(name, "Q6_K") == 0) {
        type = GGML_TYPE_Q6_K;
        return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    const char * backend_name = argc > 1 ? argv[1] : "CPU";
    ggml_type    weight_type  = GGML_TYPE_F32;
    if (argc > 2 && !parse_weight_type(argv[2], weight_type)) {
        std::fprintf(stderr, "unsupported DoRA test weight type: %s (expected F32, Q8_0, Q4_K, Q5_K, or Q6_K)\n",
                     argv[2]);
        return 1;
    }
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

    const int64_t input_width  = weight_type == GGML_TYPE_F32 ? 2 : ggml_blck_size(weight_type);
    ggml_tensor * weight       = ggml_new_tensor_2d(ctx, weight_type, input_width, 2);
    ggml_tensor * input        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_width, 1);
    ggml_tensor * adapter_a    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_width, 1);
    ggml_tensor * adapter_b    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2);
    ggml_tensor * magnitude    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
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
    adapters.params.emplace(weight, DiTAdapterParam{
                                        /*.a =*/adapter_a,
                                        /*.b =*/adapter_b,
                                        /*.magnitude =*/magnitude,
                                        /*.base_norm_sq =*/base_norm_sq,
                                        /*.scale =*/0.5f,
                                        /*.dora_rows =*/true,
                                    });
    ggml_tensor * transformed = dit_adapter_linear_transform(ctx, &adapters, weight, input);
    ggml_tensor * loss        = ggml_sum(ctx, ggml_sqr(ctx, transformed));
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

    std::vector<float> weight_data((size_t) input_width * 2);
    std::vector<float> input_data((size_t) input_width);
    std::vector<float> adapter_a_data((size_t) input_width);
    std::vector<float> adapter_b_data = { 2.0f, -0.5f };
    std::vector<float> magnitude_data = { 1.0f, 2.0f };
    if (weight_type == GGML_TYPE_F32) {
        weight_data    = { 1.0f, 2.0f, 3.0f, 4.0f };
        input_data     = { 2.0f, -1.0f };
        adapter_a_data = { 0.5f, -1.0f };
    } else {
        for (int64_t i = 0; i < input_width; ++i) {
            input_data[(size_t) i]     = 0.125f * (float) ((i * 7) % 13 - 6);
            adapter_a_data[(size_t) i] = 0.0625f * (float) ((i * 5) % 17 - 8);
            for (int64_t row = 0; row < 2; ++row) {
                const int64_t index         = row * input_width + i;
                weight_data[(size_t) index] = 0.1f * (float) ((index * 11 + row * 3) % 23 - 11);
            }
        }
        adapter_b_data = { 0.75f, -0.4f };
        magnitude_data = { 1.25f, 0.9f };
    }

    std::vector<float>   stored_weight_data = weight_data;
    std::vector<uint8_t> quantized_weight_data;
    if (ggml_is_quantized(weight_type)) {
        const size_t row_size = ggml_row_size(weight_type, input_width);
        quantized_weight_data.resize(row_size * 2);
        ggml_quantize_chunk(weight_type, weight_data.data(), quantized_weight_data.data(), 0, 2, input_width, nullptr);
        const auto * traits = ggml_get_type_traits(weight_type);
        for (int64_t row = 0; row < 2; ++row) {
            traits->to_float(quantized_weight_data.data() + row * row_size,
                             stored_weight_data.data() + row * input_width, input_width);
        }
        ggml_backend_tensor_set(weight, quantized_weight_data.data(), 0, quantized_weight_data.size());
    } else {
        ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    }

    std::vector<float> base_norm_sq_data(2, 0.0f);
    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t i = 0; i < input_width; ++i) {
            const float value = stored_weight_data[(size_t) (row * input_width + i)];
            base_norm_sq_data[(size_t) row] += value * value;
        }
    }

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(adapter_a, adapter_a_data.data(), 0, adapter_a_data.size() * sizeof(float));
    ggml_backend_tensor_set(adapter_b, adapter_b_data.data(), 0, adapter_b_data.size() * sizeof(float));
    ggml_backend_tensor_set(magnitude, magnitude_data.data(), 0, magnitude_data.size() * sizeof(float));
    ggml_backend_tensor_set(base_norm_sq, base_norm_sq_data.data(), 0, base_norm_sq_data.size() * sizeof(float));

    ggml_graph_reset(graph);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "DoRA graph execution failed on %s\n", backend_name);
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    float expected_loss = 0.0f;
    for (int64_t row = 0; row < 2; ++row) {
        float effective_dot     = 0.0f;
        float effective_norm_sq = 0.0f;
        for (int64_t i = 0; i < input_width; ++i) {
            const float effective_weight = stored_weight_data[(size_t) (row * input_width + i)] +
                                           0.5f * adapter_b_data[(size_t) row] * adapter_a_data[(size_t) i];
            effective_dot += effective_weight * input_data[(size_t) i];
            effective_norm_sq += effective_weight * effective_weight;
        }
        const float output = magnitude_data[(size_t) row] * effective_dot / std::sqrt(effective_norm_sq);
        expected_loss += output * output;
    }
    float actual_loss = 0.0f;
    ggml_backend_tensor_get(loss, &actual_loss, 0, sizeof(actual_loss));
    if (!nearly_equal(actual_loss, expected_loss)) {
        std::fprintf(stderr, "DoRA loss mismatch on %s: got %f, expected %f\n", backend_name, actual_loss,
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

    std::printf("DoRA-row backend forward/backward: OK (%s, %s frozen weight)\n", ggml_backend_name(backend),
                ggml_type_name(weight_type));
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
