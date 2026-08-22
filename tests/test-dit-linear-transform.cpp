// ABOUTME: Exercises the DiT linear-transform callback used by trainable adapters.
// ABOUTME: Verifies callback dispatch and the unchanged frozen-weight fallback path.

#include "dit-graph.h"
#include "ggml-cpu.h"
#include "train-adapter.h"

#include <cmath>
#include <cstdio>

struct TransformProbe {
    int calls;
};

static ggml_tensor * scale_linear_output(ggml_context * ctx,
                                         void *         data,
                                         ggml_tensor *  weight,
                                         ggml_tensor *  input) {
    auto * probe = static_cast<TransformProbe *>(data);
    probe->calls++;
    return ggml_scale(ctx, ggml_mul_mat(ctx, weight, input), 2.0f);
}

static bool nearly_equal(float actual, float expected) {
    return std::fabs(actual - expected) < 1e-4f;
}

int main() {
    ggml_init_params params = {
        /*.mem_size   =*/1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fputs("failed to create ggml context\n", stderr);
        return 1;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);

    TransformProbe probe = {};
    DiTGGML        model = {};
    model.linear_transform      = scale_linear_output;
    model.linear_transform_data = &probe;

    ggml_tensor * transformed = dit_ggml_linear(ctx, &model, weight, input);
    if (probe.calls != 1 || transformed->op != GGML_OP_SCALE) {
        std::fputs("linear transform callback was not used\n", stderr);
        ggml_free(ctx);
        return 1;
    }

    ggml_tensor * frozen = dit_ggml_linear(ctx, nullptr, weight, input);
    if (probe.calls != 1 || frozen->op != GGML_OP_MUL_MAT) {
        std::fputs("frozen linear path did not use the base weight\n", stderr);
        ggml_free(ctx);
        return 1;
    }

    ggml_free(ctx);

    ggml_init_params compute_params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    ctx = ggml_init(compute_params);
    if (!ctx) {
        std::fputs("failed to create compute context\n", stderr);
        return 1;
    }

    weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ggml_tensor * adapter_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ggml_tensor * adapter_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2);
    ggml_tensor * magnitude = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor * base_norm_sq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);

    const float weight_data[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float input_data[] = { 2.0f, -1.0f };
    const float adapter_a_data[] = { 0.5f, -1.0f };
    const float adapter_b_data[] = { 2.0f, -0.5f };
    const float magnitude_data[] = { 1.0f, 2.0f };
    const float base_norm_sq_data[] = { 5.0f, 25.0f };
    std::memcpy(weight->data, weight_data, sizeof(weight_data));
    std::memcpy(input->data, input_data, sizeof(input_data));
    std::memcpy(adapter_a->data, adapter_a_data, sizeof(adapter_a_data));
    std::memcpy(adapter_b->data, adapter_b_data, sizeof(adapter_b_data));
    std::memcpy(magnitude->data, magnitude_data, sizeof(magnitude_data));
    std::memcpy(base_norm_sq->data, base_norm_sq_data, sizeof(base_norm_sq_data));

    DiTAdapterTransform adapters;
    adapters.params.emplace(weight, DiTAdapterParam {
        /*.a =*/adapter_a,
        /*.b =*/adapter_b,
        /*.magnitude =*/magnitude,
        /*.base_norm_sq =*/base_norm_sq,
        /*.scale =*/0.5f,
        /*.dora_rows =*/true,
    });
    model.linear_transform = dit_adapter_linear_transform;
    model.linear_transform_data = &adapters;

    transformed = dit_ggml_linear(ctx, &model, weight, input);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, transformed);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu || ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) {
        std::fputs("DoRA graph computation failed\n", stderr);
        ggml_free(ctx);
        return 1;
    }

    const float * output = static_cast<const float *>(transformed->data);
    const float expected_0 = 2.0f / std::sqrt(3.25f);
    const float expected_1 = 3.0f / std::sqrt(26.328125f);
    if (!nearly_equal(output[0], expected_0) || !nearly_equal(output[1], expected_1)) {
        std::fprintf(stderr,
                     "DoRA output mismatch: got [%f, %f], expected [%f, %f]\n",
                     output[0],
                     output[1],
                     expected_0,
                     expected_1);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return 1;
    }

    ggml_tensor * weight_row = ggml_view_2d(ctx, weight, 2, 1, weight->nb[1], weight->nb[1]);
    ggml_tensor * row_output = dit_ggml_linear(ctx, &model, weight_row, input);
    ggml_cgraph * row_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(row_graph, row_output);
    if (ggml_backend_graph_compute(cpu, row_graph) != GGML_STATUS_SUCCESS) {
        std::fputs("DoRA weight-view graph computation failed\n", stderr);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return 1;
    }
    if (!nearly_equal(static_cast<const float *>(row_output->data)[0], expected_1)) {
        std::fprintf(stderr,
                     "DoRA weight-view output mismatch: got %f, expected %f\n",
                     static_cast<const float *>(row_output->data)[0],
                     expected_1);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return 1;
    }

    ggml_set_param(adapter_a);
    ggml_set_param(adapter_b);
    ggml_set_param(magnitude);
    ggml_tensor * loss = ggml_sum(ctx, ggml_sqr(ctx, transformed));
    ggml_set_loss(loss);
    ggml_cgraph * training_graph = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(training_graph, loss);
    ggml_build_backward_expand(ctx, training_graph, nullptr);
    ggml_graph_reset(training_graph);
    if (ggml_backend_graph_compute(cpu, training_graph) != GGML_STATUS_SUCCESS) {
        std::fputs("DoRA backward graph computation failed\n", stderr);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return 1;
    }

    ggml_tensor * adapter_a_grad = ggml_graph_get_grad(training_graph, adapter_a);
    ggml_tensor * adapter_b_grad = ggml_graph_get_grad(training_graph, adapter_b);
    ggml_tensor * magnitude_grad = ggml_graph_get_grad(training_graph, magnitude);
    if (!adapter_a_grad || !adapter_b_grad || !magnitude_grad) {
        std::fputs("DoRA parameters did not all receive gradients\n", stderr);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return 1;
    }

    const float analytical = static_cast<const float *>(adapter_a_grad->data)[0];
    const float saved = static_cast<float *>(adapter_a->data)[0];
    auto loss_at = [&](float value) {
        static_cast<float *>(adapter_a->data)[0] = value;
        ggml_graph_reset(training_graph);
        if (ggml_backend_graph_compute(cpu, training_graph) != GGML_STATUS_SUCCESS) {
            return NAN;
        }
        return static_cast<const float *>(loss->data)[0];
    };
    const float step = 1e-3f;
    const float numerical = (loss_at(saved + step) - loss_at(saved - step)) / (2.0f * step);
    static_cast<float *>(adapter_a->data)[0] = saved;
    if (!std::isfinite(numerical) || std::fabs(analytical - numerical) > 2e-2f) {
        std::fprintf(stderr,
                     "DoRA gradient mismatch: analytical %f, numerical %f\n",
                     analytical,
                     numerical);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_free(cpu);
    ggml_free(ctx);
    std::puts("DiT functional LoRA/DoRA transform: OK");
    return 0;
}
