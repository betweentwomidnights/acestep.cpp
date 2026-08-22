// ABOUTME: Applies native AdamW updates to ACE-Step LoRA and DoRA-row adapter parameters.
// ABOUTME: Downloads real GGML gradients, clips their global norm, and uploads updated graph inputs.

#pragma once

#include "train-dit-graph.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct ACETrainAdamWConfig {
    float learning_rate    = 1e-4f;
    float beta1            = 0.9f;
    float beta2            = 0.999f;
    float epsilon          = 1e-8f;
    float weight_decay     = 0.01f;
    float max_gradient_norm = 1.0f;
};

struct ACETrainAdamWTensorState {
    std::vector<float> first_moment;
    std::vector<float> second_moment;
};

struct ACETrainAdapterOptimizerParam {
    ACETrainAdamWTensorState a;
    ACETrainAdamWTensorState b;
    ACETrainAdamWTensorState magnitude;
};

struct ACETrainAdapterOptimizer {
    int                                        step = 0;
    std::vector<ACETrainAdapterOptimizerParam> params;
};

static bool ace_train_adamw_update(std::vector<float> &          parameter,
                                   const std::vector<float> &    gradient,
                                   ACETrainAdamWTensorState &    moments,
                                   const ACETrainAdamWConfig &   config,
                                   int                           step,
                                   float                         gradient_scale,
                                   std::string &                 error) {
    if (parameter.size() != gradient.size()) {
        error = "AdamW parameter and gradient sizes differ";
        return false;
    }
    if (moments.first_moment.size() != parameter.size()) {
        moments.first_moment.assign(parameter.size(), 0.0f);
        moments.second_moment.assign(parameter.size(), 0.0f);
    }
    const float beta1_correction = 1.0f - std::pow(config.beta1, (float) step);
    const float beta2_correction = 1.0f - std::pow(config.beta2, (float) step);
    for (size_t i = 0; i < parameter.size(); ++i) {
        const float grad = gradient[i] * gradient_scale;
        moments.first_moment[i] =
            config.beta1 * moments.first_moment[i] + (1.0f - config.beta1) * grad;
        moments.second_moment[i] =
            config.beta2 * moments.second_moment[i] + (1.0f - config.beta2) * grad * grad;
        const float corrected_first  = moments.first_moment[i] / beta1_correction;
        const float corrected_second = moments.second_moment[i] / beta2_correction;
        parameter[i] -= config.learning_rate *
                        (corrected_first / (std::sqrt(corrected_second) + config.epsilon) +
                         config.weight_decay * parameter[i]);
    }
    return true;
}

static bool ace_train_adapter_adamw_step(ACETrainDiTGraph &             training,
                                         ACETrainAdapterState &         state,
                                         ACETrainAdapterOptimizer &     optimizer,
                                         const ACETrainAdamWConfig &    config,
                                         std::string &                  error) {
    error.clear();
    if (training.adapters.params.size() != state.params.size() || state.params.empty()) {
        error = "training graph and adapter state target counts differ";
        return false;
    }
    if (config.learning_rate <= 0.0f || config.beta1 < 0.0f || config.beta1 >= 1.0f || config.beta2 < 0.0f ||
        config.beta2 >= 1.0f || config.epsilon <= 0.0f || config.weight_decay < 0.0f ||
        config.max_gradient_norm < 0.0f) {
        error = "invalid AdamW configuration";
        return false;
    }

    struct Gradients {
        std::vector<float> a;
        std::vector<float> b;
        std::vector<float> magnitude;
    };
    std::vector<Gradients> gradients(state.params.size());
    double                 global_norm_sq = 0.0;
    auto download = [&](struct ggml_tensor * parameter, std::vector<float> & values) {
        if (!parameter) {
            values.clear();
            return true;
        }
        struct ggml_tensor * gradient = ggml_graph_get_grad(training.graph, parameter);
        if (!gradient) {
            gradient = ggml_graph_get_grad_acc(training.graph, parameter);
        }
        if (!gradient) {
            error = "adapter parameter has no gradient: " + std::string(ggml_get_name(parameter));
            return false;
        }
        values.resize((size_t) ggml_nelements(gradient));
        ggml_backend_tensor_get(gradient, values.data(), 0, values.size() * sizeof(float));
        for (float value : values) {
            if (!std::isfinite(value)) {
                error = "adapter gradient is not finite: " + std::string(ggml_get_name(parameter));
                return false;
            }
            global_norm_sq += (double) value * value;
        }
        return true;
    };

    for (size_t i = 0; i < state.params.size(); ++i) {
        const ACETrainAdapterGraphParam & graph = training.adapters.params[i];
        if (!download(graph.a, gradients[i].a) || !download(graph.b, gradients[i].b) ||
            !download(graph.magnitude, gradients[i].magnitude)) {
            return false;
        }
    }

    const double global_norm = std::sqrt(global_norm_sq);
    const float gradient_scale =
        (config.max_gradient_norm > 0.0f && global_norm > config.max_gradient_norm) ?
            (float) (config.max_gradient_norm / global_norm) :
            1.0f;
    const int next_step = optimizer.step + 1;
    if (optimizer.params.size() != state.params.size()) {
        optimizer.params.resize(state.params.size());
    }
    for (size_t i = 0; i < state.params.size(); ++i) {
        ACETrainAdapterParam &          parameter = state.params[i];
        ACETrainAdapterOptimizerParam & moments   = optimizer.params[i];
        if (!ace_train_adamw_update(
                parameter.a, gradients[i].a, moments.a, config, next_step, gradient_scale, error) ||
            !ace_train_adamw_update(
                parameter.b, gradients[i].b, moments.b, config, next_step, gradient_scale, error) ||
            (!parameter.magnitude.empty() &&
             !ace_train_adamw_update(parameter.magnitude,
                                     gradients[i].magnitude,
                                     moments.magnitude,
                                     config,
                                     next_step,
                                     gradient_scale,
                                     error))) {
            return false;
        }
    }
    optimizer.step = next_step;
    return ace_upload_train_adapter_state(training.adapters, state, error);
}
