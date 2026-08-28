// ABOUTME: Applies native AdamW updates to ACE-Step LoRA and DoRA-row adapter parameters.
// ABOUTME: Downloads real GGML gradients, clips their global norm, and uploads updated graph inputs.

#pragma once

#include "train-dit-graph.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct ACETrainAdamWConfig {
    float learning_rate     = 1e-4f;
    float beta1             = 0.9f;
    float beta2             = 0.999f;
    float epsilon           = 1e-8f;
    float weight_decay      = 0.01f;
    float max_gradient_norm = 1.0f;
};

enum ACETrainSchedule {
    ACE_TRAIN_SCHEDULE_CONSTANT,
    ACE_TRAIN_SCHEDULE_LINEAR,
    ACE_TRAIN_SCHEDULE_COSINE,
    ACE_TRAIN_SCHEDULE_COSINE_RESTARTS,
};

static float ace_train_learning_rate(float            base_learning_rate,
                                     int              step,
                                     int              total_steps,
                                     int              warmup_steps,
                                     ACETrainSchedule schedule,
                                     int              restart_count = 1) {
    if (base_learning_rate <= 0.0f || total_steps <= 0) {
        return 0.0f;
    }
    const int clamped_step   = std::max(0, std::min(step, total_steps));
    const int clamped_warmup = std::max(0, std::min(warmup_steps, total_steps));
    if (clamped_warmup > 0 && clamped_step < clamped_warmup) {
        const float progress = (float) clamped_step / (float) clamped_warmup;
        return base_learning_rate * (0.1f + 0.9f * progress);
    }
    if (schedule == ACE_TRAIN_SCHEDULE_CONSTANT || clamped_warmup == total_steps) {
        return base_learning_rate;
    }

    const int   remaining_steps = total_steps - clamped_warmup;
    const float progress        = (float) (clamped_step - clamped_warmup) / (float) remaining_steps;
    float       factor          = 1.0f;
    if (schedule == ACE_TRAIN_SCHEDULE_LINEAR) {
        factor = 1.0f - 0.99f * progress;
    } else {
        float cosine_progress = progress;
        if (schedule == ACE_TRAIN_SCHEDULE_COSINE_RESTARTS && clamped_step < total_steps) {
            const int cycles = std::max(1, restart_count);
            cosine_progress  = progress * (float) cycles;
            cosine_progress -= std::floor(cosine_progress);
        }
        constexpr float pi = 3.14159265358979323846f;
        factor             = 0.01f + 0.99f * 0.5f * (1.0f + std::cos(pi * cosine_progress));
    }
    return base_learning_rate * factor;
}

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

struct ACETrainAdapterGradientParam {
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> magnitude;
};

struct ACETrainAdapterGradientAccumulator {
    int                                       microbatch_count = 0;
    int                                       example_count    = 0;
    std::vector<ACETrainAdapterGradientParam> params;
};

static bool ace_train_adamw_update(std::vector<float> &        parameter,
                                   const std::vector<float> &  gradient,
                                   ACETrainAdamWTensorState &  moments,
                                   const ACETrainAdamWConfig & config,
                                   int                         step,
                                   float                       gradient_scale,
                                   std::string &               error) {
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
        const float grad             = gradient[i] * gradient_scale;
        moments.first_moment[i]      = config.beta1 * moments.first_moment[i] + (1.0f - config.beta1) * grad;
        moments.second_moment[i]     = config.beta2 * moments.second_moment[i] + (1.0f - config.beta2) * grad * grad;
        const float corrected_first  = moments.first_moment[i] / beta1_correction;
        const float corrected_second = moments.second_moment[i] / beta2_correction;
        parameter[i] -= config.learning_rate * (corrected_first / (std::sqrt(corrected_second) + config.epsilon) +
                                                config.weight_decay * parameter[i]);
    }
    return true;
}

static bool ace_train_adapter_accumulate_gradient_subset(const ACETrainAdapterGraphState &    graph_state,
                                                         struct ggml_cgraph *                 graph,
                                                         const std::vector<size_t> &          indices,
                                                         ACETrainAdapterGradientAccumulator & accumulator,
                                                         int                                  example_count,
                                                         bool                                 finish_microbatch,
                                                         std::string &                        error);

static bool ace_train_adapter_accumulate_gradients(ACETrainDiTGraph &                   training,
                                                   ACETrainAdapterGradientAccumulator & accumulator,
                                                   int                                  example_count,
                                                   std::string &                        error) {
    std::vector<size_t> indices(training.adapters.params.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    return ace_train_adapter_accumulate_gradient_subset(training.adapters, training.graph, indices, accumulator,
                                                        example_count, true, error);
}

static bool ace_train_adapter_accumulate_gradient_subset(const ACETrainAdapterGraphState &    graph_state,
                                                         struct ggml_cgraph *                 graph,
                                                         const std::vector<size_t> &          indices,
                                                         ACETrainAdapterGradientAccumulator & accumulator,
                                                         int                                  example_count,
                                                         bool                                 finish_microbatch,
                                                         std::string &                        error) {
    error.clear();
    if (example_count <= 0 || !graph || graph_state.params.empty() ||
        (!accumulator.params.empty() && accumulator.params.size() != graph_state.params.size())) {
        error = "training graph and gradient accumulator target counts differ";
        return false;
    }
    if (accumulator.params.empty()) {
        accumulator.params.resize(graph_state.params.size());
    }

    auto accumulate = [&](struct ggml_tensor * parameter, std::vector<float> & sum) {
        if (!parameter) {
            sum.clear();
            return true;
        }
        struct ggml_tensor * gradient = ggml_graph_get_grad(graph, parameter);
        if (!gradient) {
            gradient = ggml_graph_get_grad_acc(graph, parameter);
        }
        if (!gradient) {
            error = "adapter parameter has no gradient: " + std::string(ggml_get_name(parameter));
            return false;
        }
        std::vector<float> values((size_t) ggml_nelements(gradient));
        if (sum.empty()) {
            sum.assign(values.size(), 0.0f);
        } else if (sum.size() != values.size()) {
            error = "adapter gradient shape changed between microbatches: " + std::string(ggml_get_name(parameter));
            return false;
        }
        values.resize((size_t) ggml_nelements(gradient));
        ggml_backend_tensor_get(gradient, values.data(), 0, values.size() * sizeof(float));
        for (size_t i = 0; i < values.size(); ++i) {
            const float value = values[i];
            if (!std::isfinite(value)) {
                error = "adapter gradient is not finite: " + std::string(ggml_get_name(parameter));
                return false;
            }
            sum[i] += value * (float) example_count;
        }
        return true;
    };

    for (size_t i : indices) {
        if (i >= graph_state.params.size()) {
            error = "adapter gradient subset index is out of range";
            return false;
        }
        const ACETrainAdapterGraphParam & graph_param = graph_state.params[i];
        ACETrainAdapterGradientParam &    sums        = accumulator.params[i];
        if (!accumulate(graph_param.a, sums.a) || !accumulate(graph_param.b, sums.b) ||
            !accumulate(graph_param.magnitude, sums.magnitude)) {
            return false;
        }
    }
    if (finish_microbatch) {
        accumulator.microbatch_count += 1;
        accumulator.example_count += example_count;
    }
    return true;
}

static bool ace_train_adapter_adamw_step_accumulated(ACETrainAdapterGraphState &          graph_state,
                                                     ACETrainAdapterState &               state,
                                                     ACETrainAdapterOptimizer &           optimizer,
                                                     const ACETrainAdamWConfig &          config,
                                                     ACETrainAdapterGradientAccumulator & accumulator,
                                                     std::string &                        error) {
    error.clear();
    if (graph_state.params.size() != state.params.size() || state.params.empty() ||
        accumulator.params.size() != state.params.size() || accumulator.microbatch_count <= 0 ||
        accumulator.example_count <= 0) {
        error = "training graph, adapter state, and accumulated gradient target counts differ";
        return false;
    }
    if (config.learning_rate <= 0.0f || config.beta1 < 0.0f || config.beta1 >= 1.0f || config.beta2 < 0.0f ||
        config.beta2 >= 1.0f || config.epsilon <= 0.0f || config.weight_decay < 0.0f ||
        config.max_gradient_norm < 0.0f) {
        error = "invalid AdamW configuration";
        return false;
    }

    const float accumulation_scale = 1.0f / (float) accumulator.example_count;
    double      global_norm_sq     = 0.0;
    auto        add_norm           = [&](const std::vector<float> & gradient) {
        for (float value : gradient) {
            const double averaged = (double) value * accumulation_scale;
            global_norm_sq += averaged * averaged;
        }
    };
    for (const ACETrainAdapterGradientParam & gradient : accumulator.params) {
        add_norm(gradient.a);
        add_norm(gradient.b);
        add_norm(gradient.magnitude);
    }

    const double global_norm    = std::sqrt(global_norm_sq);
    const float  clip_scale     = (config.max_gradient_norm > 0.0f && global_norm > config.max_gradient_norm) ?
                                      (float) (config.max_gradient_norm / global_norm) :
                                      1.0f;
    const float  gradient_scale = accumulation_scale * clip_scale;
    const int    next_step      = optimizer.step + 1;
    if (optimizer.params.size() != state.params.size()) {
        optimizer.params.resize(state.params.size());
    }
    for (size_t i = 0; i < state.params.size(); ++i) {
        ACETrainAdapterParam &               parameter = state.params[i];
        ACETrainAdapterOptimizerParam &      moments   = optimizer.params[i];
        const ACETrainAdapterGradientParam & gradient  = accumulator.params[i];
        if (!ace_train_adamw_update(parameter.a, gradient.a, moments.a, config, next_step, gradient_scale, error) ||
            !ace_train_adamw_update(parameter.b, gradient.b, moments.b, config, next_step, gradient_scale, error) ||
            (!parameter.magnitude.empty() &&
             !ace_train_adamw_update(parameter.magnitude, gradient.magnitude, moments.magnitude, config, next_step,
                                     gradient_scale, error))) {
            return false;
        }
    }
    optimizer.step = next_step;
    if (!ace_upload_train_adapter_state(graph_state, state, error)) {
        return false;
    }
    accumulator = {};
    return true;
}

static bool ace_train_adapter_adamw_step_accumulated(ACETrainDiTGraph &                   training,
                                                     ACETrainAdapterState &               state,
                                                     ACETrainAdapterOptimizer &           optimizer,
                                                     const ACETrainAdamWConfig &          config,
                                                     ACETrainAdapterGradientAccumulator & accumulator,
                                                     std::string &                        error) {
    return ace_train_adapter_adamw_step_accumulated(training.adapters, state, optimizer, config, accumulator, error);
}

static bool ace_train_adapter_adamw_step(ACETrainDiTGraph &          training,
                                         ACETrainAdapterState &      state,
                                         ACETrainAdapterOptimizer &  optimizer,
                                         const ACETrainAdamWConfig & config,
                                         std::string &               error) {
    ACETrainAdapterGradientAccumulator accumulator;
    return ace_train_adapter_accumulate_gradients(training, accumulator, 1, error) &&
           ace_train_adapter_adamw_step_accumulated(training, state, optimizer, config, accumulator, error);
}
