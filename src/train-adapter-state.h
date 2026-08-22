// ABOUTME: Selects ACE-Step adapter targets and initializes LoRA or DoRA-row trainable state.
// ABOUTME: Mirrors Gary's attention and balanced projection profiles over unfused DiT weights.

#pragma once

#include "dit.h"
#include "train-adapter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct ACETrainAdapterTarget {
    struct ggml_tensor * weight;
    std::string          weight_name;
    std::string          module_name;
    int64_t              in;
    int64_t              out;
    int                  rank;
    int                  alpha;
    int                  base_rank;
    int                  base_alpha;
    std::string          module_profile;
};

struct ACETrainAdapterParam {
    ACETrainAdapterTarget target;
    std::vector<float>    a;
    std::vector<float>    b;
    std::vector<float>    magnitude;
    std::vector<float>    base_norm_sq;
};

struct ACETrainAdapterState {
    std::string                       adapter_type;
    std::string                       module_profile;
    int                               base_rank;
    int                               base_alpha;
    std::vector<ACETrainAdapterParam> params;
};

struct ACETrainAdapterGraphParam {
    ACETrainAdapterTarget target;
    struct ggml_tensor *  a;
    struct ggml_tensor *  b;
    struct ggml_tensor *  magnitude;
    struct ggml_tensor *  base_norm_sq;
};

struct ACETrainAdapterGraphState {
    std::vector<ACETrainAdapterGraphParam> params;
    DiTAdapterTransform                    transform;
};

static int ace_train_adapter_round(double value) {
    const double lower    = std::floor(value);
    const double fraction = value - lower;
    if (fraction < 0.5) {
        return (int) lower;
    }
    if (fraction > 0.5) {
        return (int) lower + 1;
    }
    const int lower_integer = (int) lower;
    return (lower_integer % 2 == 0) ? lower_integer : lower_integer + 1;
}

static bool ace_train_adapter_targets(const DiTGGML &                      model,
                                      const std::string &                  profile,
                                      int                                  base_rank,
                                      int                                  base_alpha,
                                      std::vector<ACETrainAdapterTarget> & targets,
                                      std::string &                        error) {
    targets.clear();
    error.clear();
    if (profile != "attention" && profile != "balanced") {
        error = "module profile must be attention or balanced";
        return false;
    }
    if (base_rank <= 0 || base_alpha <= 0) {
        error = "rank and alpha must be greater than zero";
        return false;
    }

    auto profile_value = [&](const std::string & module_name, bool alpha) {
        double multiplier = 1.0;
        if (profile == "balanced") {
            if (module_name == "self_attn.q_proj") {
                multiplier = 1.0 / 4.0;
            } else if (module_name == "self_attn.k_proj") {
                multiplier = 3.0 / 8.0;
            } else if (module_name == "self_attn.v_proj") {
                multiplier = 5.0 / 4.0;
            } else if (module_name == "self_attn.o_proj") {
                multiplier = 7.0 / 8.0;
            } else if (module_name == "cross_attn.k_proj" || module_name == "mlp.gate_proj") {
                multiplier = 5.0 / 8.0;
            } else if (module_name == "cross_attn.v_proj") {
                multiplier = 1.0 / 2.0;
            } else if (module_name == "cross_attn.o_proj" || module_name == "mlp.up_proj" ||
                       module_name == "mlp.down_proj") {
                multiplier = 3.0 / 4.0;
            }
        }
        const int rank = std::max(1, ace_train_adapter_round((double) base_rank * multiplier));
        return alpha ? std::max(1, ace_train_adapter_round((double) base_alpha * rank / base_rank)) : rank;
    };

    auto add_target = [&](struct ggml_tensor * weight, const std::string & module_name) {
        if (!weight || ggml_n_dims(weight) != 2) {
            error = "training projection is missing or is not two-dimensional: " + module_name;
            return false;
        }
        ACETrainAdapterTarget target;
        target.weight         = weight;
        target.weight_name    = ggml_get_name(weight);
        target.module_name    = module_name;
        target.in             = weight->ne[0];
        target.out            = weight->ne[1];
        target.rank           = profile_value(module_name, false);
        target.alpha          = profile_value(module_name, true);
        target.base_rank      = base_rank;
        target.base_alpha     = base_alpha;
        target.module_profile = profile;
        targets.push_back(std::move(target));
        return true;
    };

    for (int i = 0; i < model.cfg.n_layers; ++i) {
        const DiTGGMLLayer & layer = model.layers[i];
        if (layer.sa_qkv || layer.sa_qk || layer.ca_qkv || layer.ca_kv || layer.gate_up) {
            error = "adapter training requires projection fusion to be disabled while loading the DiT";
            targets.clear();
            return false;
        }
        if (!add_target(layer.sa_q_proj, "self_attn.q_proj") || !add_target(layer.sa_k_proj, "self_attn.k_proj") ||
            !add_target(layer.sa_v_proj, "self_attn.v_proj") || !add_target(layer.sa_o_proj, "self_attn.o_proj") ||
            !add_target(layer.ca_q_proj, "cross_attn.q_proj") || !add_target(layer.ca_k_proj, "cross_attn.k_proj") ||
            !add_target(layer.ca_v_proj, "cross_attn.v_proj") || !add_target(layer.ca_o_proj, "cross_attn.o_proj")) {
            targets.clear();
            return false;
        }
        if (profile == "balanced" &&
            (!add_target(layer.gate_proj, "mlp.gate_proj") || !add_target(layer.up_proj, "mlp.up_proj") ||
             !add_target(layer.down_proj, "mlp.down_proj"))) {
            targets.clear();
            return false;
        }
    }
    return true;
}

static bool ace_train_tensor_to_f32(struct ggml_tensor * tensor, std::vector<float> & values, std::string & error) {
    const int64_t count = ggml_nelements(tensor);
    values.resize((size_t) count);
    if (tensor->type == GGML_TYPE_F32) {
        if (tensor->buffer) {
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        } else if (tensor->data) {
            std::memcpy(values.data(), tensor->data, values.size() * sizeof(float));
        } else {
            error = "weight tensor has no readable storage: " + std::string(ggml_get_name(tensor));
            return false;
        }
        return true;
    }

    const size_t               bytes = ggml_nbytes(tensor);
    std::vector<unsigned char> packed(bytes);
    if (tensor->buffer) {
        ggml_backend_tensor_get(tensor, packed.data(), 0, bytes);
    } else if (tensor->data) {
        std::memcpy(packed.data(), tensor->data, bytes);
    } else {
        error = "weight tensor has no readable storage: " + std::string(ggml_get_name(tensor));
        return false;
    }

    const struct ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (!traits->to_float) {
        error = "weight tensor type cannot be converted to F32: " + std::string(ggml_get_name(tensor));
        return false;
    }
    const int64_t row_elements = tensor->ne[0];
    const int64_t rows         = count / row_elements;
    const size_t  row_bytes    = ggml_row_size(tensor->type, row_elements);
    for (int64_t row = 0; row < rows; ++row) {
        traits->to_float(packed.data() + (size_t) row * row_bytes, values.data() + (size_t) row * row_elements,
                         row_elements);
    }
    return true;
}

static bool ace_init_train_adapter_state(const std::vector<ACETrainAdapterTarget> & targets,
                                         const std::string &                        adapter_type,
                                         unsigned long long                         seed,
                                         ACETrainAdapterState &                     state,
                                         std::string &                              error) {
    state = {};
    error.clear();
    if (adapter_type != "lora" && adapter_type != "dora-rows") {
        error = "adapter type must be lora or dora-rows";
        return false;
    }
    state.adapter_type = adapter_type;
    if (!targets.empty()) {
        state.module_profile = targets.front().module_profile;
        state.base_rank      = targets.front().base_rank;
        state.base_alpha     = targets.front().base_alpha;
    }
    std::mt19937_64 random(seed);

    for (const ACETrainAdapterTarget & target : targets) {
        if (target.rank <= 0 || target.rank > target.in || target.rank > target.out) {
            error = "rank is infeasible for target " + target.weight_name;
            state = {};
            return false;
        }
        ACETrainAdapterParam param;
        param.target = target;
        param.a.resize((size_t) target.rank * target.in);
        param.b.assign((size_t) target.out * target.rank, 0.0f);
        const float                           bound = 1.0f / std::sqrt((float) target.in);
        std::uniform_real_distribution<float> distribution(-bound, bound);
        for (float & value : param.a) {
            value = distribution(random);
        }

        if (adapter_type == "dora-rows") {
            std::vector<float> weight;
            if (!ace_train_tensor_to_f32(target.weight, weight, error)) {
                state = {};
                return false;
            }
            param.magnitude.resize((size_t) target.out);
            param.base_norm_sq.resize((size_t) target.out);
            for (int64_t out = 0; out < target.out; ++out) {
                double norm_sq = 0.0;
                for (int64_t in = 0; in < target.in; ++in) {
                    const double value = weight[(size_t) out * target.in + in];
                    norm_sq += value * value;
                }
                param.magnitude[(size_t) out]    = (float) std::sqrt(norm_sq);
                param.base_norm_sq[(size_t) out] = (float) (norm_sq + (double) target.in * 1e-12);
            }
        }
        state.params.push_back(std::move(param));
    }
    return true;
}

static bool ace_build_train_adapter_graph_state(struct ggml_context *        ctx,
                                                const ACETrainAdapterState & state,
                                                ACETrainAdapterGraphState &  graph_state,
                                                std::string &                error) {
    graph_state = {};
    error.clear();
    if (!ctx) {
        error = "adapter graph context is null";
        return false;
    }
    const bool dora_rows = state.adapter_type == "dora-rows";
    if (!dora_rows && state.adapter_type != "lora") {
        error = "adapter type must be lora or dora-rows";
        return false;
    }

    for (const ACETrainAdapterParam & host : state.params) {
        const ACETrainAdapterTarget & target = host.target;
        if (host.a.size() != (size_t) target.in * target.rank || host.b.size() != (size_t) target.out * target.rank) {
            error       = "adapter host tensor shape mismatch for " + target.weight_name;
            graph_state = {};
            return false;
        }
        if (dora_rows &&
            (host.magnitude.size() != (size_t) target.out || host.base_norm_sq.size() != (size_t) target.out)) {
            error       = "DoRA host tensor shape mismatch for " + target.weight_name;
            graph_state = {};
            return false;
        }

        ACETrainAdapterGraphParam graph_param = {};
        graph_param.target                    = target;
        graph_param.a                         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, target.in, target.rank);
        graph_param.b                         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, target.rank, target.out);
        ggml_set_param(graph_param.a);
        ggml_set_param(graph_param.b);
        ggml_set_input(graph_param.a);
        ggml_set_input(graph_param.b);

        std::string stem = target.weight_name;
        if (stem.size() >= 7 && stem.compare(stem.size() - 7, 7, ".weight") == 0) {
            stem.resize(stem.size() - 7);
        }
        ggml_set_name(graph_param.a, (stem + ".lora_A").c_str());
        ggml_set_name(graph_param.b, (stem + ".lora_B").c_str());

        if (dora_rows) {
            graph_param.magnitude    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, target.out);
            graph_param.base_norm_sq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, target.out);
            ggml_set_param(graph_param.magnitude);
            ggml_set_input(graph_param.magnitude);
            ggml_set_input(graph_param.base_norm_sq);
            ggml_set_name(graph_param.magnitude, (stem + ".lora_magnitude_vector").c_str());
            ggml_set_name(graph_param.base_norm_sq, (stem + ".base_norm_sq").c_str());
        }

        graph_state.transform.params.emplace(target.weight, DiTAdapterParam{
                                                                /*.a =*/graph_param.a,
                                                                /*.b =*/graph_param.b,
                                                                /*.magnitude =*/graph_param.magnitude,
                                                                /*.base_norm_sq =*/graph_param.base_norm_sq,
                                                                /*.scale =*/(float) target.alpha / target.rank,
                                                                /*.dora_rows =*/dora_rows,
                                                            });
        graph_state.params.push_back(std::move(graph_param));
    }
    return true;
}

static bool ace_upload_train_adapter_state(const ACETrainAdapterGraphState & graph_state,
                                           const ACETrainAdapterState &      state,
                                           std::string &                     error) {
    error.clear();
    if (graph_state.params.size() != state.params.size()) {
        error = "adapter graph and host state target counts differ";
        return false;
    }

    auto upload = [&](struct ggml_tensor * tensor, const std::vector<float> & values) {
        if (!tensor || ggml_nelements(tensor) != (int64_t) values.size()) {
            return false;
        }
        const size_t bytes = values.size() * sizeof(float);
        if (tensor->buffer) {
            ggml_backend_tensor_set(tensor, values.data(), 0, bytes);
            return true;
        }
        if (tensor->data) {
            std::memcpy(tensor->data, values.data(), bytes);
            return true;
        }
        return false;
    };

    for (size_t i = 0; i < state.params.size(); ++i) {
        const ACETrainAdapterParam &      host  = state.params[i];
        const ACETrainAdapterGraphParam & graph = graph_state.params[i];
        if (!upload(graph.a, host.a) || !upload(graph.b, host.b) ||
            (graph.magnitude && !upload(graph.magnitude, host.magnitude)) ||
            (graph.base_norm_sq && !upload(graph.base_norm_sq, host.base_norm_sq))) {
            error = "failed to upload adapter tensors for " + host.target.weight_name;
            return false;
        }
    }
    return true;
}
