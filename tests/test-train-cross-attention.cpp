// Cross-attention backward regression with grouped heads and unequal lengths.
// Finite differences exercise production graph construction on the selected backend.
#include "dit-graph.h"
#include "train-diffusion.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const auto device =
        argc > 1 ? ggml_backend_dev_by_name(argv[1]) : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = device ? ggml_backend_dev_init(device, nullptr) : nullptr;
    if (!backend) {
        std::fputs("FAIL: requested backend unavailable\n", stderr);
        return 1;
    }
    ggml_context * ctx     = ggml_init({ 16 * 1024 * 1024, nullptr, true });
    DiTGGML        model   = {};
    model.cfg.hidden_size  = 16;
    model.cfg.head_dim     = 4;
    model.cfg.n_heads      = 4;
    model.cfg.n_kv_heads   = 2;
    model.cfg.rms_norm_eps = 1e-6f;
    DiTGGMLLayer layer     = {};
    layer.ca_q_proj        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16);
    layer.ca_k_proj        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 8);
    layer.ca_v_proj        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 8);
    layer.ca_o_proj        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16);
    layer.ca_q_norm        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    layer.ca_k_norm        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_param(layer.ca_q_proj);
    ggml_set_param(layer.ca_k_proj);
    auto * input      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 3, 1);
    auto * encoder    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 5, 1);
    auto * prediction = dit_ggml_build_cross_attn(ctx, &model, &layer, input, encoder, nullptr, nullptr, 3, 5, 1);
    auto * probe      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 3, 1);
    auto * loss       = ggml_sum(ctx, ggml_mul(ctx, prediction, probe));
    ggml_set_loss(loss);
    ggml_set_output(loss);
    auto * forward = ggml_new_graph_custom(ctx, 2048, false);
    ggml_build_forward_expand(forward, loss);
    // This also catches the unsafe layout on CPU-only CI, where RMSNorm's
    // backward itself supports strides and finite differences alone would pass.
    int norms = 0;
    for (int i = 0; i < ggml_graph_n_nodes(forward); ++i) {
        auto * node = ggml_graph_node(forward, i);
        if (node->op == GGML_OP_MUL_MAT) {
            // Small finite differences require F32 arithmetic. Vulkan's
            // default F16 matmul rounding can exceed the perturbation itself.
            ggml_mul_mat_set_prec(node, GGML_PREC_F32);
        }
        if (node->op == GGML_OP_RMS_NORM) {
            ++norms;
            if (!ggml_is_contiguous(node->src[0])) {
                std::fputs("FAIL: cross-attention RMSNorm must receive contiguous head rows\n", stderr);
                ggml_free(ctx);
                ggml_backend_free(backend);
                return 1;
            }
        }
    }
    auto * backward = ggml_new_graph_custom(ctx, 4096, true);
    ggml_build_forward_expand(backward, loss);
    ggml_build_backward_expand(ctx, backward, nullptr);
    auto * q_gradient = ggml_graph_get_grad(backward, layer.ca_q_proj);
    auto * k_gradient = ggml_graph_get_grad(backward, layer.ca_k_proj);
    ggml_set_output(q_gradient);
    ggml_set_output(k_gradient);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer || norms != 2) {
        std::fputs("FAIL: graph allocation or RMSNorm inventory\n", stderr);
        return 1;
    }
    auto fill = [](ggml_tensor * tensor, float phase, bool norm = false) {
        std::vector<float> values(size_t(ggml_nelements(tensor)));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = (norm ? 1.0f : 0.0f) + 0.3f * std::sin(float(i) * 0.17f + phase);
        }
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
        return values;
    };
    auto q = fill(layer.ca_q_proj, 0.1f);
    auto k = fill(layer.ca_k_proj, 0.5f);
    fill(layer.ca_v_proj, 0.9f);
    fill(layer.ca_o_proj, 1.3f);
    fill(layer.ca_q_norm, 0.3f, true);
    fill(layer.ca_k_norm, 0.7f, true);
    fill(input, 1.7f);
    fill(encoder, 2.1f);
    fill(probe, 2.5f);
    ggml_graph_reset(backward);
    bool               ok = ggml_backend_graph_compute(backend, backward) == GGML_STATUS_SUCCESS;
    std::vector<float> dq(q.size()), dk(k.size());
    ggml_backend_tensor_get(q_gradient, dq.data(), 0, dq.size() * sizeof(float));
    ggml_backend_tensor_get(k_gradient, dk.data(), 0, dk.size() * sizeof(float));
    auto check = [&](ggml_tensor * parameter, std::vector<float> values, const std::vector<float> & gradients) {
        for (size_t i : { size_t(0), size_t(7), size_t(32), size_t(73) }) {
            const float original = values[i];
            // Check two central-difference scales above GPU input rounding.
            // Agreement at both prevents selecting a lucky perturbation.
            for (float eps : { 0.01f, 0.02f }) {
                float losses[2];
                for (int side = 0; side < 2; ++side) {
                    values[i] = original + (side == 0 ? eps : -eps);
                    ggml_backend_tensor_set(parameter, values.data(), 0, values.size() * sizeof(float));
                    if (ggml_backend_graph_compute(backend, forward) != GGML_STATUS_SUCCESS) {
                        return false;
                    }
                    ggml_backend_tensor_get(loss, &losses[side], 0, sizeof(float));
                }
                values[i]             = original;
                const float numerical = (losses[0] - losses[1]) / (2 * eps);
                if (!std::isfinite(gradients[i]) ||
                    std::fabs(numerical - gradients[i]) >
                        2e-3f + 0.02f * std::max(std::fabs(numerical), std::fabs(gradients[i]))) {
                    std::fprintf(stderr, "FAIL: gradient[%zu] eps=%g analytic=%g numerical=%g\n", i, eps, gradients[i],
                                 numerical);
                    return false;
                }
            }
        }
        ggml_backend_tensor_set(parameter, values.data(), 0, values.size() * sizeof(float));
        return true;
    };
    ok            = ok && check(layer.ca_q_proj, q, dq) && check(layer.ca_k_proj, k, dk);
    // ggml's broadcasted MUL_MAT backward cannot reduce grouped heads across
    // multiple batch items. Both trainers must reject this before graph build.
    model.backend = backend;
    ACETrainAdapterState  state;
    ACETrainDiTGraph      training;
    ACETrainDiTCheckpoint checkpoint;
    std::string           error;
    ok = ok && !ace_build_train_dit_graph(model, state, 6, 5, 2, training, error) &&
         error.find("accumulation") != std::string::npos;
    ok = ok && !ace_build_train_dit_checkpoint(model, state, 6, 5, 2, checkpoint, error) &&
         error.find("accumulation") != std::string::npos;
    std::fprintf(stderr, "%s: production cross-attention gradients on %s\n", ok ? "PASS" : "FAIL",
                 ggml_backend_dev_name(device));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
