// ABOUTME: Applies inference LoRA/DoRA without modifying or requantizing base weights.
// ABOUTME: Precomputes DoRA row gains and scaled B factors once per adapter load.
#pragma once

#include "adapter-merge.h"

#include <algorithm>
#include <limits>

struct AdapterFunctionalParam {
    ggml_tensor *      weight    = nullptr;
    ggml_tensor *      a         = nullptr;
    ggml_tensor *      b         = nullptr;
    ggml_tensor *      gain      = nullptr;
    ggml_tensor *      magnitude = nullptr;
    ggml_tensor *      norm_sq   = nullptr;
    float              scale     = 1.0f;
    int64_t            rank      = 0;
    std::vector<float> host_a, host_b, host_magnitude, host_norm_sq;
};

struct AdapterFunctional {
    ggml_context *                            ctx                 = nullptr;
    ggml_backend_buffer_t                     buffer              = nullptr;
    float                                     strength            = 1.0f;
    bool                                      prepared            = false;
    bool                                      normalized_strength = false;
    std::vector<AdapterFunctionalParam>       params;
    std::unordered_map<ggml_tensor *, size_t> lookup;
};

static void adapter_functional_free(AdapterFunctional & state) {
    if (state.buffer) {
        ggml_backend_buffer_free(state.buffer);
    }
    if (state.ctx) {
        ggml_free(state.ctx);
    }
    state = {};
}

// Called while pending copies still refer to the mapped GGUF. Nothing in the
// base context or its pending-copy list is modified by this consumer.
static bool adapter_functional_load(AdapterFunctional & state,
                                    WeightCtx *         weights,
                                    const GGUFModel &   gf,
                                    const char *        path,
                                    float               strength,
                                    ggml_backend_t      backend,
                                    bool                normalized_strength = false) {
    adapter_functional_free(state);
    if (!std::isfinite(strength)) {
        return false;
    }
    state.strength            = strength;
    state.normalized_strength = normalized_strength;
    auto consume              = [&](const std::string & name, const void * raw, ggml_type type, int64_t in, int64_t out,
                       int64_t rank, float scale, const std::vector<float> & a, const std::vector<float> & b,
                       const std::vector<float> & magnitude) {
        AdapterFunctionalParam p;
        p.weight = ggml_get_tensor(weights->ctx, name.c_str());
        if (!p.weight || p.weight->ne[0] != in || p.weight->ne[1] != out) {
            fprintf(stderr, "[Adapter] Functional target must be unfused: %s\n", name.c_str());
            return false;
        }
        auto finite = [](const std::vector<float> & values) {
            return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
        };
        if (!finite(a) || !finite(b) || !finite(magnitude)) {
            return false;
        }
        p.rank           = rank;
        p.scale          = scale;
        p.host_a         = a;
        p.host_b         = b;
        p.host_magnitude = magnitude;
        if (!magnitude.empty()) {
            // One host row at a time: no full dense base copy or GPU roundtrip.
            std::vector<float> row((size_t) in);
            p.host_norm_sq.resize((size_t) out);
            const size_t stride = ggml_row_size(type, in);
            for (int64_t o = 0; o < out; ++o) {
                if (!adapter_dequant((const uint8_t *) raw + (size_t) o * stride, row.data(), in, 1, type)) {
                    return false;
                }
                double norm = 0;
                for (float v : row) {
                    norm += (double) v * v;
                }
                p.host_norm_sq[(size_t) o] = (float) norm;
            }
        } else {
            for (float & v : p.host_b) {
                v *= scale * strength;
            }
        }
        state.lookup.emplace(p.weight, state.params.size());
        state.params.push_back(std::move(p));
        return true;
    };
    if (!adapter_merge(weights, gf, path, strength, backend, consume)) {
        adapter_functional_free(state);
        return false;
    }
    return true;
}

// Base weights must already be allocated. Only low-rank intermediates appear
// in this graph: ||W+sBA||^2 = ||W||^2 + 2s<W,BA> + s^2||BA||^2.
// Once B and gain are prepared, inference is gain*(W*x) + B_scaled*(A*x).
static bool adapter_functional_prepare(AdapterFunctional & state, ggml_backend_t backend) {
    if (state.prepared) {
        return true;
    }
    if (state.ctx || state.buffer) {
        return false;  // do not reuse a partial failed allocation
    }
    if (state.params.empty()) {
        return true;
    }
    ggml_init_params ip{ (state.params.size() * 6 + 16) * ggml_tensor_overhead() + 1024, nullptr, true };
    state.ctx = ggml_init(ip);
    if (!state.ctx) {
        return false;
    }
    for (auto & p : state.params) {
        const int64_t in = p.weight->ne[0], out = p.weight->ne[1];
        p.a = ggml_new_tensor_2d(state.ctx, GGML_TYPE_F32, in, p.rank);
        p.b = ggml_new_tensor_2d(state.ctx, GGML_TYPE_F32, p.rank, out);
        if (!p.host_magnitude.empty()) {
            p.magnitude = ggml_new_tensor_2d(state.ctx, GGML_TYPE_F32, 1, out);
            p.norm_sq   = ggml_new_tensor_2d(state.ctx, GGML_TYPE_F32, 1, out);
            p.gain      = ggml_new_tensor_1d(state.ctx, GGML_TYPE_F32, out);
        }
    }
    state.buffer = ggml_backend_alloc_ctx_tensors(state.ctx, backend);
    if (!state.buffer) {
        return false;
    }
    ggml_backend_buffer_set_usage(state.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto & p : state.params) {
        ggml_backend_tensor_set(p.a, p.host_a.data(), 0, p.host_a.size() * sizeof(float));
        ggml_backend_tensor_set(p.b, p.host_b.data(), 0, p.host_b.size() * sizeof(float));
        if (p.gain) {
            ggml_backend_tensor_set(p.magnitude, p.host_magnitude.data(), 0, p.host_magnitude.size() * sizeof(float));
            ggml_backend_tensor_set(p.norm_sq, p.host_norm_sq.data(), 0, p.host_norm_sq.size() * sizeof(float));
        }
        std::vector<float>().swap(p.host_a);
        std::vector<float>().swap(p.host_b);
        std::vector<float>().swap(p.host_magnitude);
        std::vector<float>().swap(p.host_norm_sq);
    }
    const size_t     capacity = state.params.size() * 32 + 128;
    ggml_init_params gp{ capacity * ggml_tensor_overhead() + ggml_graph_overhead_custom(capacity, false) + 1024 * 1024,
                         nullptr, true };
    ggml_context *   ctx = ggml_init(gp);
    if (!ctx) {
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, capacity, false);
    for (auto & p : state.params) {
        if (!p.gain) {
            continue;
        }
        auto        wa         = ggml_mul_mat(ctx, p.weight, p.a);
        auto        bt         = ggml_cont(ctx, ggml_transpose(ctx, p.b));
        auto        cross      = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, ggml_mul(ctx, wa, bt))));
        auto        aa         = ggml_mul_mat(ctx, p.a, p.a);
        auto        update_sq  = ggml_sum_rows(ctx, ggml_mul(ctx, p.b, ggml_mul_mat(ctx, aa, p.b)));
        // SA3/PEFT-style strength scales BA BEFORE normalization. Keep the
        // strength-one graph identical to delta mode for a strict regression control.
        const bool  normalized = state.normalized_strength && state.strength != 1.0f;
        const float norm_scale = normalized ? p.scale * state.strength : p.scale;
        auto        n          = ggml_add(ctx, p.norm_sq, ggml_scale(ctx, cross, 2 * norm_scale));
        n                      = ggml_add(ctx, n, ggml_scale(ctx, update_sq, norm_scale * norm_scale));
        n                      = ggml_clamp(ctx, n, 1e-20f, std::numeric_limits<float>::max());
        auto row               = ggml_div(ctx, p.magnitude, ggml_sqrt(ctx, n));
        // Legacy delta mode interpolates the FULL normalized DoRA update.
        // Normalized mode preserves the learned magnitude instead; zero is
        // explicitly bypassed by adapter_functional_linear in both modes.
        auto gain              = normalized ? row : ggml_scale_bias(ctx, row, state.strength, 1 - state.strength);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, ggml_reshape_1d(ctx, gain, p.weight->ne[1]), p.gain));
        auto scaled_b = ggml_mul(ctx, p.b, ggml_scale(ctx, row, p.scale * state.strength));
        // All reads of B for this target precede this final in-place write.
        ggml_build_forward_expand(graph, ggml_cpy(ctx, scaled_b, p.b));
    }
    bool ok = true;
    if (ggml_graph_n_nodes(graph)) {
        auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ok         = alloc && ggml_gallocr_alloc_graph(alloc, graph) &&
             ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        if (alloc) {
            ggml_gallocr_free(alloc);
        }
    }
    ggml_free(ctx);
    for (const auto & p : state.params) {
        if (!ok || !p.gain) {
            continue;
        }
        std::vector<float> gain((size_t) p.weight->ne[1]);
        ggml_backend_tensor_get(p.gain, gain.data(), 0, gain.size() * sizeof(float));
        ok = std::all_of(gain.begin(), gain.end(), [](float v) { return std::isfinite(v); });
    }
    if (ok) {
        fprintf(stderr,
                "[Adapter] Functional ready: %zu projections, %.1f MiB, base weights unchanged, strength-mode=%s\n",
                state.params.size(), ggml_backend_buffer_get_size(state.buffer) / (1024.0 * 1024.0),
                state.normalized_strength ? "normalized" : "delta");
    }
    state.prepared = ok;
    return ok;
}

static ggml_tensor * adapter_functional_linear(ggml_context * ctx,
                                               void *         data,
                                               ggml_tensor *  weight,
                                               ggml_tensor *  input) {
    auto & state  = *static_cast<AdapterFunctional *>(data);
    auto   output = ggml_mul_mat(ctx, weight, input);
    if (state.strength == 0) {
        return output;
    }
    auto   it     = state.lookup.find(weight);
    size_t offset = 0;
    if (it == state.lookup.end() && weight->view_src) {
        it     = state.lookup.find(weight->view_src);
        offset = weight->view_offs / weight->view_src->nb[1];
    }
    if (it == state.lookup.end()) {
        return output;
    }
    const auto & p    = state.params[it->second];
    auto         b    = p.b;
    auto         gain = p.gain;
    if (offset || weight->ne[1] != p.weight->ne[1]) {
        b = ggml_view_2d(ctx, b, b->ne[0], weight->ne[1], b->nb[1], offset * b->nb[1]);
        if (gain) {
            gain = ggml_view_1d(ctx, gain, weight->ne[1], offset * sizeof(float));
        }
    }
    if (gain) {
        output = ggml_mul(ctx, output, gain);
    }
    return ggml_add(ctx, output, ggml_mul_mat(ctx, b, ggml_mul_mat(ctx, p.a, input)));
}
