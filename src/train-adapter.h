// ABOUTME: Builds functional LoRA and DoRA-row linear transforms for ACE-Step training graphs.
// ABOUTME: Keeps frozen base weights in stored precision without materializing effective weights.

#pragma once

#include "ggml.h"

#include <unordered_map>

struct DiTAdapterParam {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    struct ggml_tensor * magnitude;
    struct ggml_tensor * base_norm_sq;
    float                scale;
    bool                 dora_rows;
};

struct DiTAdapterTransform {
    std::unordered_map<struct ggml_tensor *, DiTAdapterParam> params;
};

static struct ggml_tensor * dit_adapter_stop_gradient(struct ggml_context * ctx, struct ggml_tensor * tensor) {
    struct ggml_tensor * result = ggml_view_tensor(ctx, tensor);
    result->src[0]              = tensor;
    return result;
}

static struct ggml_tensor * dit_adapter_linear_transform(struct ggml_context * ctx,
                                                         void *                data,
                                                         struct ggml_tensor *  weight,
                                                         struct ggml_tensor *  input) {
    auto * adapters      = static_cast<DiTAdapterTransform *>(data);
    auto   found         = adapters->params.find(weight);
    size_t output_offset = 0;
    if (found == adapters->params.end() && weight->view_src) {
        found = adapters->params.find(weight->view_src);
        if (found != adapters->params.end()) {
            output_offset = weight->view_offs / weight->view_src->nb[1];
        }
    }
    if (found == adapters->params.end()) {
        return ggml_mul_mat(ctx, weight, input);
    }

    DiTAdapterParam adapter = found->second;
    if (output_offset > 0 || weight->ne[1] != found->first->ne[1]) {
        adapter.b = ggml_view_2d(ctx, adapter.b, adapter.b->ne[0], weight->ne[1], adapter.b->nb[1],
                                 output_offset * adapter.b->nb[1]);
        if (adapter.dora_rows) {
            adapter.magnitude =
                ggml_view_1d(ctx, adapter.magnitude, weight->ne[1], output_offset * adapter.magnitude->nb[0]);
            adapter.base_norm_sq =
                ggml_view_1d(ctx, adapter.base_norm_sq, weight->ne[1], output_offset * adapter.base_norm_sq->nb[0]);
        }
    }
    struct ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    struct ggml_tensor * ax     = ggml_mul_mat(ctx, adapter.a, input);
    struct ggml_tensor * bax    = ggml_mul_mat(ctx, adapter.b, ax);
    output                      = ggml_add(ctx, output, ggml_scale(ctx, bax, adapter.scale));

    if (!adapter.dora_rows) {
        return output;
    }

    const int64_t        out        = output->ne[0];
    struct ggml_tensor * weight_t_a = ggml_mul_mat(ctx, weight, adapter.a);
    struct ggml_tensor * b_t        = ggml_cont(ctx, ggml_transpose(ctx, adapter.b));
    struct ggml_tensor * cross_term =
        ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, ggml_mul(ctx, weight_t_a, b_t))));
    struct ggml_tensor * a_t_a     = ggml_mul_mat(ctx, adapter.a, adapter.a);
    struct ggml_tensor * a_t_a_b   = ggml_mul_mat(ctx, a_t_a, adapter.b);
    struct ggml_tensor * update_sq = ggml_sum_rows(ctx, ggml_mul(ctx, adapter.b, a_t_a_b));
    struct ggml_tensor * norm_sq =
        ggml_add(ctx,
                 ggml_add(ctx, adapter.base_norm_sq,
                          ggml_scale(ctx, ggml_reshape_1d(ctx, cross_term, out), 2.0f * adapter.scale)),
                 ggml_scale(ctx, ggml_reshape_1d(ctx, update_sq, out), adapter.scale * adapter.scale));
    struct ggml_tensor * norm      = dit_adapter_stop_gradient(ctx, ggml_sqrt(ctx, norm_sq));
    struct ggml_tensor * row_scale = ggml_div(ctx, adapter.magnitude, norm);
    return ggml_mul(ctx, output, row_scale);
}
