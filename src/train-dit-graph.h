// ABOUTME: Builds ACE-Step DiT forward-and-backward graphs for native LoRA and DoRA-row training.
// ABOUTME: Owns adapter parameters, flow-matching inputs, loss, gradients, and reusable graph allocation.

#pragma once

#include "dit-graph.h"
#include "ggml-alloc.h"
#include "train-adapter-state.h"

#include <string>

struct ACETrainDiTGraph {
    struct ggml_context *  ctx;
    ggml_gallocr_t         allocator;
    struct ggml_cgraph *   graph;
    struct ggml_tensor *   input_latents;
    struct ggml_tensor *   encoder_hidden;
    struct ggml_tensor *   timestep;
    struct ggml_tensor *   reference_timestep;
    struct ggml_tensor *   positions;
    struct ggml_tensor *   self_attention_mask;
    struct ggml_tensor *   cross_attention_mask;
    struct ggml_tensor *   target_velocity;
    struct ggml_tensor *   velocity;
    struct ggml_tensor *   loss;
    ACETrainAdapterGraphState adapters;
};

static void ace_free_train_dit_graph(ACETrainDiTGraph & training) {
    if (training.allocator) {
        ggml_gallocr_free(training.allocator);
    }
    if (training.ctx) {
        ggml_free(training.ctx);
    }
    training = {};
}

static bool ace_build_train_dit_graph(DiTGGML &                   model,
                                      const ACETrainAdapterState & state,
                                      int                           temporal_length,
                                      int                           encoder_sequence_length,
                                      int                           batch_size,
                                      ACETrainDiTGraph &            training,
                                      std::string &                 error) {
    training = {};
    error.clear();
    if (!model.backend) {
        error = "DiT backend is not initialized";
        return false;
    }
    if (temporal_length <= 0 || encoder_sequence_length <= 0 || batch_size <= 0 ||
        temporal_length % model.cfg.patch_size != 0) {
        error = "invalid DiT training graph dimensions";
        return false;
    }

    const size_t graph_capacity = 65536;
    const size_t context_size = graph_capacity * ggml_tensor_overhead() +
                                ggml_graph_overhead_custom(graph_capacity, true) + 64 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/context_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    training.ctx = ggml_init(params);
    if (!training.ctx) {
        error = "failed to create DiT training graph context";
        return false;
    }

    if (!ace_build_train_adapter_graph_state(training.ctx, state, training.adapters, error)) {
        ace_free_train_dit_graph(training);
        return false;
    }

    DiTGGMLLinearTransform saved_transform = model.linear_transform;
    void *                 saved_data      = model.linear_transform_data;
    model.linear_transform                 = dit_adapter_linear_transform;
    model.linear_transform_data            = &training.adapters.transform;
    training.graph = dit_ggml_build_graph(&model,
                                          training.ctx,
                                          temporal_length,
                                          encoder_sequence_length,
                                          batch_size,
                                          &training.input_latents,
                                          &training.velocity,
                                          true);
    model.linear_transform      = saved_transform;
    model.linear_transform_data = saved_data;
    if (!training.graph || !training.velocity) {
        error = "failed to build DiT forward graph";
        ace_free_train_dit_graph(training);
        return false;
    }

    training.encoder_hidden = ggml_graph_get_tensor(training.graph, "enc_hidden");
    training.timestep = ggml_graph_get_tensor(training.graph, "t");
    training.reference_timestep = ggml_graph_get_tensor(training.graph, "t_r");
    training.positions = ggml_graph_get_tensor(training.graph, "positions");
    training.self_attention_mask = ggml_graph_get_tensor(training.graph, "sa_mask_sw");
    training.cross_attention_mask = ggml_graph_get_tensor(training.graph, "ca_mask");
    if (!training.encoder_hidden || !training.timestep || !training.reference_timestep || !training.positions ||
        !training.self_attention_mask || !training.cross_attention_mask) {
        error = "DiT training graph is missing a required input";
        ace_free_train_dit_graph(training);
        return false;
    }

    training.target_velocity = ggml_new_tensor_3d(training.ctx,
                                                  GGML_TYPE_F32,
                                                  model.cfg.out_channels,
                                                  temporal_length,
                                                  batch_size);
    ggml_set_name(training.target_velocity, "target_velocity");
    ggml_set_input(training.target_velocity);
    struct ggml_tensor * squared_error =
        ggml_sqr(training.ctx, ggml_sub(training.ctx, training.velocity, training.target_velocity));
    const float element_count = (float) ((int64_t) model.cfg.out_channels * temporal_length * batch_size);
    training.loss = ggml_scale(training.ctx, ggml_sum(training.ctx, squared_error), 1.0f / element_count);
    ggml_set_name(training.loss, "flow_matching_loss");
    ggml_set_loss(training.loss);
    ggml_set_output(training.loss);
    ggml_build_forward_expand(training.graph, training.loss);
    ggml_build_backward_expand(training.ctx, training.graph, nullptr);

    for (const ACETrainAdapterGraphParam & param : training.adapters.params) {
        struct ggml_tensor * trainable[] = { param.a, param.b, param.magnitude };
        for (struct ggml_tensor * tensor : trainable) {
            if (!tensor) {
                continue;
            }
            struct ggml_tensor * gradient = ggml_graph_get_grad(training.graph, tensor);
            if (!gradient) {
                gradient = ggml_graph_get_grad_acc(training.graph, tensor);
            }
            if (!gradient) {
                error = "adapter parameter has no gradient: " + std::string(ggml_get_name(tensor));
                ace_free_train_dit_graph(training);
                return false;
            }
            ggml_set_output(gradient);
        }
    }

    training.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    if (!training.allocator || !ggml_gallocr_alloc_graph(training.allocator, training.graph)) {
        error = "failed to allocate DiT training graph";
        ace_free_train_dit_graph(training);
        return false;
    }
    if (!ace_upload_train_adapter_state(training.adapters, state, error)) {
        ace_free_train_dit_graph(training);
        return false;
    }
    return true;
}
