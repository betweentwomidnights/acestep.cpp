// ABOUTME: Builds gradient-checkpointed ACE-Step DiT training graphs with per-layer recomputation.
// ABOUTME: Keeps full-track training memory bounded by one transformer layer's backward workspace.

#pragma once

#include "dit-graph.h"
#include "ggml-alloc.h"
#include "train-adapter-state.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct ACETrainDiTCheckpointBlock {
    struct ggml_context * ctx       = nullptr;
    struct ggml_cgraph *  graph     = nullptr;
    ggml_gallocr_t        allocator = nullptr;
    std::vector<size_t>   param_indices;
};

struct ACETrainDiTCheckpoint {
    ggml_backend_t        backend           = nullptr;
    struct ggml_context * persistent_ctx    = nullptr;
    ggml_backend_buffer_t persistent_buffer = nullptr;

    struct ggml_tensor *              input_latents             = nullptr;
    struct ggml_tensor *              encoder_hidden            = nullptr;
    struct ggml_tensor *              timestep                  = nullptr;
    struct ggml_tensor *              reference_timestep        = nullptr;
    struct ggml_tensor *              positions                 = nullptr;
    struct ggml_tensor *              self_attention_mask       = nullptr;
    struct ggml_tensor *              full_self_attention_mask  = nullptr;
    struct ggml_tensor *              cross_attention_mask      = nullptr;
    struct ggml_tensor *              target_velocity           = nullptr;
    struct ggml_tensor *              loss_weights              = nullptr;
    struct ggml_tensor *              saved_timestep_projection = nullptr;
    struct ggml_tensor *              saved_timestep_embedding  = nullptr;
    struct ggml_tensor *              saved_encoder_hidden      = nullptr;
    std::vector<struct ggml_tensor *> boundaries;
    struct ggml_tensor *              gradient_a = nullptr;
    struct ggml_tensor *              gradient_b = nullptr;
    ACETrainAdapterGraphState         adapters;

    struct ggml_context *                   forward_ctx       = nullptr;
    struct ggml_cgraph *                    forward_graph     = nullptr;
    ggml_gallocr_t                          forward_allocator = nullptr;
    struct ggml_context *                   tail_ctx          = nullptr;
    struct ggml_cgraph *                    tail_graph        = nullptr;
    ggml_gallocr_t                          tail_allocator    = nullptr;
    struct ggml_tensor *                    velocity          = nullptr;
    struct ggml_tensor *                    loss              = nullptr;
    std::vector<ggml_gallocr_t>             block_allocators;
    std::vector<ACETrainDiTCheckpointBlock> blocks;

    int layer_count             = 0;
    int temporal_length         = 0;
    int sequence_length         = 0;
    int encoder_sequence_length = 0;
    int batch_size              = 0;

    struct ggml_tensor * gradient_input(int layer) const {
        return ((layer_count - 1 - layer) % 2 == 0) ? gradient_a : gradient_b;
    }

    struct ggml_tensor * gradient_output(int layer) const {
        return ((layer_count - 1 - layer) % 2 == 0) ? gradient_b : gradient_a;
    }
};

static void ace_free_train_dit_checkpoint(ACETrainDiTCheckpoint & training) {
    if (training.forward_allocator) {
        ggml_gallocr_free(training.forward_allocator);
    }
    if (training.tail_allocator) {
        ggml_gallocr_free(training.tail_allocator);
    }
    for (ggml_gallocr_t allocator : training.block_allocators) {
        if (allocator) {
            ggml_gallocr_free(allocator);
        }
    }
    for (ACETrainDiTCheckpointBlock & block : training.blocks) {
        if (block.ctx) {
            ggml_free(block.ctx);
        }
    }
    if (training.forward_ctx) {
        ggml_free(training.forward_ctx);
    }
    if (training.tail_ctx) {
        ggml_free(training.tail_ctx);
    }
    if (training.persistent_buffer) {
        ggml_backend_buffer_free(training.persistent_buffer);
    }
    if (training.persistent_ctx) {
        ggml_free(training.persistent_ctx);
    }
    training = {};
}

static int ace_train_checkpoint_layer_for_weight(const std::string & name, int layer_count) {
    const std::string prefix = "decoder.layers.";
    if (name.compare(0, prefix.size(), prefix) != 0) {
        return -1;
    }
    const size_t begin = prefix.size();
    const size_t end   = name.find('.', begin);
    if (end == std::string::npos || end == begin) {
        return -1;
    }
    for (size_t i = begin; i < end; ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return -1;
        }
    }
    const int layer = std::atoi(name.substr(begin, end - begin).c_str());
    return layer >= 0 && layer < layer_count ? layer : -1;
}

static bool ace_build_train_dit_checkpoint(DiTGGML &                    model,
                                           const ACETrainAdapterState & state,
                                           int                          temporal_length,
                                           int                          encoder_sequence_length,
                                           int                          batch_size,
                                           ACETrainDiTCheckpoint &      training,
                                           std::string &                error) {
    training = {};
    error.clear();
    if (!model.backend) {
        error = "DiT backend is not initialized";
        return false;
    }
    if (temporal_length <= 0 || encoder_sequence_length <= 0 || batch_size <= 0 || model.cfg.patch_size <= 0 ||
        temporal_length % model.cfg.patch_size != 0 || model.cfg.n_layers <= 0 || !model.cond_emb_w) {
        error = "invalid checkpointed DiT training graph dimensions";
        return false;
    }

    training.backend                                     = model.backend;
    training.layer_count                                 = model.cfg.n_layers;
    training.temporal_length                             = temporal_length;
    training.sequence_length                             = temporal_length / model.cfg.patch_size;
    training.encoder_sequence_length                     = encoder_sequence_length;
    training.batch_size                                  = batch_size;
    const int                        hidden_size         = model.cfg.hidden_size;
    const int                        encoder_hidden_size = (int) model.cond_emb_w->ne[0];
    const int                        sequence_length     = training.sequence_length;
    const ggml_backend_buffer_type_t buffer_type         = ggml_backend_get_default_buffer_type(model.backend);

    const DiTGGMLLinearTransform saved_transform     = model.linear_transform;
    void * const                 saved_data          = model.linear_transform_data;
    const bool                   saved_flash         = model.use_flash_attn;
    bool                         transform_installed = false;
    auto                         restore_model       = [&]() {
        if (transform_installed) {
            model.linear_transform      = saved_transform;
            model.linear_transform_data = saved_data;
            model.use_flash_attn        = saved_flash;
            transform_installed         = false;
        }
    };
    auto fail = [&](const std::string & message) {
        error = message;
        restore_model();
        ace_free_train_dit_checkpoint(training);
        return false;
    };

    const size_t            persistent_tensor_count = 24 + (size_t) model.cfg.n_layers + 1 + state.params.size() * 4;
    struct ggml_init_params persistent_params       = {
        /*.mem_size   =*/ggml_tensor_overhead() * (persistent_tensor_count + 64),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    training.persistent_ctx = ggml_init(persistent_params);
    if (!training.persistent_ctx) {
        return fail("failed to create persistent checkpoint tensor context");
    }

    struct ggml_context * pctx = training.persistent_ctx;
    training.input_latents =
        ggml_new_tensor_3d(pctx, GGML_TYPE_F32, model.cfg.in_channels, temporal_length, batch_size);
    training.encoder_hidden =
        ggml_new_tensor_3d(pctx, GGML_TYPE_F32, encoder_hidden_size, encoder_sequence_length, batch_size);
    training.timestep           = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, batch_size);
    training.reference_timestep = ggml_new_tensor_1d(pctx, GGML_TYPE_F32, batch_size);
    training.positions          = ggml_new_tensor_1d(pctx, GGML_TYPE_I32, sequence_length * batch_size);
    training.self_attention_mask =
        ggml_new_tensor_4d(pctx, GGML_TYPE_F16, sequence_length, sequence_length, 1, batch_size);
    training.full_self_attention_mask =
        ggml_new_tensor_4d(pctx, GGML_TYPE_F16, sequence_length, sequence_length, 1, batch_size);
    training.cross_attention_mask =
        ggml_new_tensor_4d(pctx, GGML_TYPE_F16, encoder_sequence_length, sequence_length, 1, batch_size);
    training.target_velocity =
        ggml_new_tensor_3d(pctx, GGML_TYPE_F32, model.cfg.out_channels, temporal_length, batch_size);
    training.loss_weights              = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, temporal_length, batch_size);
    training.saved_timestep_projection = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, 6 * hidden_size, batch_size);
    training.saved_timestep_embedding  = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, hidden_size, batch_size);
    training.saved_encoder_hidden =
        ggml_new_tensor_3d(pctx, GGML_TYPE_F32, hidden_size, encoder_sequence_length, batch_size);
    training.boundaries.resize((size_t) model.cfg.n_layers + 1);
    for (struct ggml_tensor *& boundary : training.boundaries) {
        boundary = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, hidden_size, sequence_length, batch_size);
        ggml_set_param(boundary);
    }
    training.gradient_a = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, hidden_size, sequence_length, batch_size);
    training.gradient_b = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, hidden_size, sequence_length, batch_size);

    for (struct ggml_tensor * input :
         { training.input_latents, training.encoder_hidden, training.timestep, training.reference_timestep,
           training.positions, training.self_attention_mask, training.full_self_attention_mask,
           training.cross_attention_mask, training.target_velocity, training.loss_weights }) {
        ggml_set_input(input);
    }
    ggml_set_name(training.input_latents, "input_latents");
    ggml_set_name(training.encoder_hidden, "enc_hidden");
    ggml_set_name(training.timestep, "t");
    ggml_set_name(training.reference_timestep, "t_r");
    ggml_set_name(training.positions, "positions");
    ggml_set_name(training.self_attention_mask, "sa_mask_sw");
    ggml_set_name(training.full_self_attention_mask, "sa_mask");
    ggml_set_name(training.cross_attention_mask, "ca_mask");
    ggml_set_name(training.target_velocity, "target_velocity");
    ggml_set_name(training.loss_weights, "loss_weights");

    if (!ace_build_train_adapter_graph_state(pctx, state, training.adapters, error)) {
        return fail(error);
    }
    training.persistent_buffer = ggml_backend_alloc_ctx_tensors(pctx, model.backend);
    if (!training.persistent_buffer) {
        return fail("failed to allocate persistent checkpoint tensors");
    }
    if (!ace_upload_train_adapter_state(training.adapters, state, error)) {
        return fail(error);
    }

    std::vector<std::vector<size_t>> layer_param_indices((size_t) model.cfg.n_layers);
    for (size_t i = 0; i < state.params.size(); ++i) {
        const int layer = ace_train_checkpoint_layer_for_weight(state.params[i].target.weight_name, model.cfg.n_layers);
        if (layer < 0) {
            return fail("checkpointed training target is not owned by a decoder layer: " +
                        state.params[i].target.weight_name);
        }
        layer_param_indices[(size_t) layer].push_back(i);
    }

    auto graph_context = [](size_t graph_capacity, bool gradients) {
        struct ggml_init_params params = {
            /*.mem_size   =*/graph_capacity * ggml_tensor_overhead() * 3 +
                ggml_graph_overhead_custom(graph_capacity, gradients),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        return ggml_init(params);
    };
    auto keep_parameter_gradients = [&](struct ggml_cgraph * graph, const std::vector<size_t> & indices) {
        for (size_t index : indices) {
            const ACETrainAdapterGraphParam & param = training.adapters.params[index];
            for (struct ggml_tensor * tensor : { param.a, param.b, param.magnitude }) {
                if (!tensor) {
                    continue;
                }
                struct ggml_tensor * gradient = ggml_graph_get_grad(graph, tensor);
                if (!gradient) {
                    gradient = ggml_graph_get_grad_acc(graph, tensor);
                }
                if (gradient) {
                    ggml_set_output(gradient);
                }
            }
        }
    };

    model.linear_transform      = dit_adapter_linear_transform;
    model.linear_transform_data = &training.adapters.transform;
    model.use_flash_attn        = false;
    transform_installed         = true;

    // F: one forward pass stores the head output and every transformer-layer boundary.
    {
        constexpr size_t graph_capacity = 32768;
        training.forward_ctx            = graph_context(graph_capacity, false);
        if (!training.forward_ctx) {
            return fail("failed to create checkpoint forward context");
        }
        struct ggml_context * ctx = training.forward_ctx;
        training.forward_graph    = ggml_new_graph_custom(ctx, graph_capacity, false);

        struct ggml_tensor * tproj_t = nullptr;
        struct ggml_tensor * temb_t =
            dit_ggml_build_temb(ctx, &model, &model.time_embed, training.timestep, &tproj_t, "_t");
        struct ggml_tensor * tproj_r = nullptr;
        struct ggml_tensor * temb_r =
            dit_ggml_build_temb(ctx, &model, &model.time_embed_r,
                                ggml_sub(ctx, training.timestep, training.reference_timestep), &tproj_r, "_r");
        struct ggml_tensor * temb  = ggml_add(ctx, temb_t, temb_r);
        struct ggml_tensor * tproj = ggml_add(ctx, tproj_t, tproj_r);
        ggml_build_forward_expand(training.forward_graph, ggml_cpy(ctx, temb, training.saved_timestep_embedding));
        ggml_build_forward_expand(training.forward_graph, ggml_cpy(ctx, tproj, training.saved_timestep_projection));

        struct ggml_tensor * patched = ggml_reshape_3d(
            ctx, training.input_latents, model.cfg.in_channels * model.cfg.patch_size, sequence_length, batch_size);
        struct ggml_tensor * hidden = dit_ggml_linear_bias(ctx, &model, model.proj_in_w, model.proj_in_b, patched);
        ggml_build_forward_expand(training.forward_graph, ggml_cpy(ctx, hidden, training.boundaries.front()));

        struct ggml_tensor * encoder =
            dit_ggml_linear_bias(ctx, &model, model.cond_emb_w, model.cond_emb_b, training.encoder_hidden);
        ggml_build_forward_expand(training.forward_graph, ggml_cpy(ctx, encoder, training.saved_encoder_hidden));

        for (int layer = 0; layer < model.cfg.n_layers; ++layer) {
            struct ggml_tensor * mask =
                model.layers[layer].layer_type == 0 ? training.self_attention_mask : training.full_self_attention_mask;
            hidden = dit_ggml_build_layer(ctx, &model, layer, training.boundaries[(size_t) layer],
                                          training.saved_timestep_projection, training.saved_encoder_hidden,
                                          training.positions, mask, training.cross_attention_mask, sequence_length,
                                          encoder_sequence_length, batch_size, false);
            ggml_build_forward_expand(training.forward_graph,
                                      ggml_cpy(ctx, hidden, training.boundaries[(size_t) layer + 1]));
        }
        training.forward_allocator = ggml_gallocr_new(buffer_type);
        if (!training.forward_allocator ||
            !ggml_gallocr_alloc_graph(training.forward_allocator, training.forward_graph)) {
            return fail("failed to allocate checkpoint forward graph");
        }
    }

    // T: reconstruct the output head and real flow-matching loss, then seed dL/dx_last.
    {
        constexpr size_t graph_capacity = 4096;
        training.tail_ctx               = graph_context(graph_capacity, true);
        if (!training.tail_ctx) {
            return fail("failed to create checkpoint tail context");
        }
        struct ggml_context * ctx = training.tail_ctx;
        training.tail_graph       = ggml_new_graph_custom(ctx, graph_capacity, true);

        struct ggml_tensor * scale_shift = model.out_scale_shift;
        if (scale_shift->type != GGML_TYPE_F32) {
            scale_shift = ggml_cast(ctx, scale_shift, GGML_TYPE_F32);
        }
        struct ggml_tensor * modulation =
            ggml_add(ctx, ggml_concat(ctx, training.saved_timestep_embedding, training.saved_timestep_embedding, 0),
                     ggml_reshape_1d(ctx, scale_shift, 2 * hidden_size));
        const size_t         hidden_bytes = (size_t) hidden_size * sizeof(float);
        struct ggml_tensor * shift =
            ggml_view_3d(ctx, modulation, hidden_size, 1, batch_size, hidden_bytes, modulation->nb[1], 0);
        struct ggml_tensor * scale =
            ggml_view_3d(ctx, modulation, hidden_size, 1, batch_size, hidden_bytes, modulation->nb[1], hidden_bytes);
        struct ggml_tensor * normalized =
            dit_ggml_rms_norm_weighted(ctx, training.boundaries.back(), model.norm_out, model.cfg.rms_norm_eps);
        normalized        = dit_ggml_adaln(ctx, normalized, scale, shift, model.scalar_one);
        training.velocity = dit_ggml_linear_bias(ctx, &model, model.proj_out_w, model.proj_out_b, normalized);
        training.velocity =
            ggml_reshape_3d(ctx, training.velocity, model.cfg.out_channels, temporal_length, batch_size);
        ggml_set_name(training.velocity, "velocity");
        ggml_set_output(training.velocity);

        struct ggml_tensor * squared_error = ggml_sqr(ctx, ggml_sub(ctx, training.velocity, training.target_velocity));
        struct ggml_tensor * per_frame_error =
            ggml_reshape_2d(ctx, ggml_sum_rows(ctx, squared_error), temporal_length, batch_size);
        per_frame_error = ggml_scale(ctx, per_frame_error, 1.0f / (float) model.cfg.out_channels);
        training.loss   = ggml_scale(ctx, ggml_sum(ctx, ggml_mul(ctx, per_frame_error, training.loss_weights)),
                                     1.0f / (float) batch_size);
        ggml_set_name(training.loss, "flow_matching_loss");
        ggml_set_loss(training.loss);
        ggml_set_output(training.loss);
        ggml_build_forward_expand(training.tail_graph, training.loss);
        ggml_build_backward_expand(ctx, training.tail_graph, nullptr);
        struct ggml_tensor * input_gradient = ggml_graph_get_grad(training.tail_graph, training.boundaries.back());
        if (!input_gradient) {
            return fail("checkpoint tail graph produced no boundary gradient");
        }
        ggml_build_forward_expand(training.tail_graph,
                                  ggml_cpy(ctx, input_gradient, training.gradient_input(model.cfg.n_layers - 1)));
        training.tail_allocator = ggml_gallocr_new(buffer_type);
        if (!training.tail_allocator || !ggml_gallocr_alloc_graph(training.tail_allocator, training.tail_graph)) {
            return fail("failed to allocate checkpoint tail graph");
        }
    }

    // B_l: recompute one layer and backpropagate its vector-Jacobian product.
    training.blocks.resize((size_t) model.cfg.n_layers);
    for (int layer = 0; layer < model.cfg.n_layers; ++layer) {
        constexpr size_t             graph_capacity = 16384;
        ACETrainDiTCheckpointBlock & block          = training.blocks[(size_t) layer];
        block.param_indices                         = layer_param_indices[(size_t) layer];
        block.ctx                                   = graph_context(graph_capacity, true);
        if (!block.ctx) {
            return fail("failed to create checkpoint block context");
        }
        block.graph = ggml_new_graph_custom(block.ctx, graph_capacity, true);
        struct ggml_tensor * mask =
            model.layers[layer].layer_type == 0 ? training.self_attention_mask : training.full_self_attention_mask;
        struct ggml_tensor * output = dit_ggml_build_layer(
            block.ctx, &model, layer, training.boundaries[(size_t) layer], training.saved_timestep_projection,
            training.saved_encoder_hidden, training.positions, mask, training.cross_attention_mask, sequence_length,
            encoder_sequence_length, batch_size, false);
        struct ggml_tensor * vjp = ggml_sum(block.ctx, ggml_mul(block.ctx, output, training.gradient_input(layer)));
        ggml_set_loss(vjp);
        ggml_build_forward_expand(block.graph, vjp);
        ggml_build_backward_expand(block.ctx, block.graph, nullptr);
        struct ggml_tensor * input_gradient = ggml_graph_get_grad(block.graph, training.boundaries[(size_t) layer]);
        if (!input_gradient) {
            return fail("checkpoint block graph produced no boundary gradient");
        }
        ggml_build_forward_expand(block.graph, ggml_cpy(block.ctx, input_gradient, training.gradient_output(layer)));
        keep_parameter_gradients(block.graph, block.param_indices);
    }

    // Blocks with identical graph shapes safely share a workspace reserved once at build time.
    std::map<int, size_t> shape_allocators;
    for (ACETrainDiTCheckpointBlock & block : training.blocks) {
        const int shape = ggml_graph_n_nodes(block.graph);
        auto      found = shape_allocators.find(shape);
        if (found == shape_allocators.end()) {
            ggml_gallocr_t allocator = ggml_gallocr_new(buffer_type);
            if (!allocator) {
                return fail("failed to create checkpoint block allocator");
            }
            training.block_allocators.push_back(allocator);
            if (!ggml_gallocr_reserve(allocator, block.graph)) {
                return fail("failed to reserve checkpoint block allocator");
            }
            found = shape_allocators.emplace(shape, training.block_allocators.size() - 1).first;
        }
        block.allocator = training.block_allocators[found->second];
    }

    restore_model();
    double block_memory_mib = 0.0;
    for (ggml_gallocr_t allocator : training.block_allocators) {
        block_memory_mib += ggml_gallocr_get_buffer_size(allocator, 0) / (1024.0 * 1024.0);
    }
    std::fprintf(stderr,
                 "[Ace-Train] Checkpointed graphs: F=%d nodes %.1f MiB, T=%d nodes %.1f MiB, "
                 "B=%d layers/%zu shapes %.1f MiB, persistent=%.1f MiB\n",
                 ggml_graph_n_nodes(training.forward_graph),
                 ggml_gallocr_get_buffer_size(training.forward_allocator, 0) / (1024.0 * 1024.0),
                 ggml_graph_n_nodes(training.tail_graph),
                 ggml_gallocr_get_buffer_size(training.tail_allocator, 0) / (1024.0 * 1024.0), model.cfg.n_layers,
                 training.block_allocators.size(), block_memory_mib,
                 ggml_backend_buffer_get_size(training.persistent_buffer) / (1024.0 * 1024.0));
    return true;
}
