// ABOUTME: Tests ACE-Step adapter target profiles and trainable parameter initialization.
// ABOUTME: Covers Gary-compatible ranks, DoRA magnitudes, identity initialization, and fusion rejection.

#include "adapter-merge.h"
#include "dit-graph.h"
#include "ggml-cpu.h"
#include "train-adapter-checkpoint.h"
#include "train-adapter-optimizer.h"
#include "train-adapter-state.h"
#include "train-checkpoint.h"
#include "train-diffusion.h"
#include "train-dit-graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

static ggml_tensor * make_weight(ggml_context * ctx, const char * name, int64_t in, int64_t out) {
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, out);
    ggml_set_name(weight, name);
    float * data = static_cast<float *>(weight->data);
    for (int64_t i = 0; i < in * out; ++i) {
        data[i] = 1.0f;
    }
    return weight;
}

static ggml_tensor * make_matrix(ggml_context * ctx,
                                 const char *   name,
                                 int64_t        in,
                                 int64_t        out,
                                 float          value) {
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, out);
    ggml_set_name(tensor, name);
    float * data = static_cast<float *>(tensor->data);
    for (int64_t i = 0; i < in * out; ++i) {
        data[i] = value;
    }
    return tensor;
}

static ggml_tensor * make_vector(ggml_context * ctx, const char * name, int64_t length, float value) {
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length);
    ggml_set_name(tensor, name);
    float * data = static_cast<float *>(tensor->data);
    for (int64_t i = 0; i < length; ++i) {
        data[i] = value;
    }
    return tensor;
}

static int fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

int main() {
    using TrainingLoader = bool (*)(DiTGGML *, const char *, const char *, float, bool);
    TrainingLoader training_loader = dit_ggml_load;
    (void) training_loader;
    using TrainingGraphBuilder = ggml_cgraph * (*)(
        DiTGGML *, ggml_context *, int, int, int, ggml_tensor **, ggml_tensor **, bool);
    TrainingGraphBuilder training_graph_builder = dit_ggml_build_graph;
    (void) training_graph_builder;

    ggml_init_params params = {
        /*.mem_size   =*/32 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return fail("failed to create ggml context");
    }

    DiTGGML model = {};
    model.cfg.n_layers = 1;
    DiTGGMLLayer & layer = model.layers[0];
    layer.sa_q_proj = make_weight(ctx, "decoder.layers.0.self_attn.q_proj.weight", 128, 256);
    layer.sa_k_proj = make_weight(ctx, "decoder.layers.0.self_attn.k_proj.weight", 128, 256);
    layer.sa_v_proj = make_weight(ctx, "decoder.layers.0.self_attn.v_proj.weight", 128, 256);
    layer.sa_o_proj = make_weight(ctx, "decoder.layers.0.self_attn.o_proj.weight", 128, 256);
    layer.ca_q_proj = make_weight(ctx, "decoder.layers.0.cross_attn.q_proj.weight", 128, 256);
    layer.ca_k_proj = make_weight(ctx, "decoder.layers.0.cross_attn.k_proj.weight", 128, 256);
    layer.ca_v_proj = make_weight(ctx, "decoder.layers.0.cross_attn.v_proj.weight", 128, 256);
    layer.ca_o_proj = make_weight(ctx, "decoder.layers.0.cross_attn.o_proj.weight", 128, 256);
    layer.gate_proj = make_weight(ctx, "decoder.layers.0.mlp.gate_proj.weight", 128, 256);
    layer.up_proj = make_weight(ctx, "decoder.layers.0.mlp.up_proj.weight", 128, 256);
    layer.down_proj = make_weight(ctx, "decoder.layers.0.mlp.down_proj.weight", 128, 256);

    std::string error;
    std::vector<ACETrainAdapterTarget> attention;
    if (!ace_train_adapter_targets(model, "attention", 64, 128, attention, error)) {
        std::fprintf(stderr, "FAIL: attention target enumeration: %s\n", error.c_str());
        ggml_free(ctx);
        return 1;
    }
    if (attention.size() != 8) {
        ggml_free(ctx);
        return fail("attention profile must contain eight projection families per layer");
    }

    std::vector<ACETrainAdapterTarget> balanced;
    if (!ace_train_adapter_targets(model, "balanced", 64, 128, balanced, error)) {
        std::fprintf(stderr, "FAIL: balanced target enumeration: %s\n", error.c_str());
        ggml_free(ctx);
        return 1;
    }
    const int expected_ranks[] = { 16, 24, 80, 56, 64, 40, 32, 48, 40, 48, 48 };
    if (balanced.size() != 11) {
        ggml_free(ctx);
        return fail("balanced profile must contain eleven projection families per layer");
    }
    for (size_t i = 0; i < balanced.size(); ++i) {
        if (balanced[i].rank != expected_ranks[i] || balanced[i].alpha != expected_ranks[i] * 2) {
            std::fprintf(stderr,
                         "FAIL: balanced profile rank/alpha at %zu: got %d/%d\n",
                         i,
                         balanced[i].rank,
                         balanced[i].alpha);
            ggml_free(ctx);
            return 1;
        }
    }

    ACETrainAdapterState state;
    if (!ace_init_train_adapter_state(balanced, "dora-rows", 42, state, error)) {
        std::fprintf(stderr, "FAIL: DoRA state initialization: %s\n", error.c_str());
        ggml_free(ctx);
        return 1;
    }
    if (state.params.size() != balanced.size()) {
        ggml_free(ctx);
        return fail("one trainable parameter set is required per target");
    }
    const ACETrainAdapterParam & first = state.params.front();
    if (first.a.size() != 128 * 16 || first.b.size() != 256 * 16 || first.magnitude.size() != 256 ||
        first.base_norm_sq.size() != 256) {
        ggml_free(ctx);
        return fail("DoRA tensor shapes do not match the target");
    }
    const float bound = 1.0f / std::sqrt(128.0f);
    for (float value : first.a) {
        if (value < -bound || value > bound) {
            ggml_free(ctx);
            return fail("LoRA A initialization exceeded Kaiming-uniform bounds");
        }
    }
    for (float value : first.b) {
        if (value != 0.0f) {
            ggml_free(ctx);
            return fail("LoRA B must start at zero for identity initialization");
        }
    }
    const float expected_magnitude = std::sqrt(128.0f);
    if (std::fabs(first.magnitude[0] - expected_magnitude) > 1e-5f ||
        std::fabs(first.base_norm_sq[0] - 128.0f) > 1e-4f) {
        ggml_free(ctx);
        return fail("DoRA magnitude and base norm must initialize from frozen weight rows");
    }

    ggml_init_params graph_params = {
        /*.mem_size   =*/32 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    ggml_context * graph_ctx = ggml_init(graph_params);
    ACETrainAdapterGraphState graph_state;
    if (!ace_build_train_adapter_graph_state(graph_ctx, state, graph_state, error)) {
        std::fprintf(stderr, "FAIL: adapter graph state: %s\n", error.c_str());
        ggml_free(graph_ctx);
        ggml_free(ctx);
        return 1;
    }
    if (graph_state.params.size() != state.params.size() ||
        graph_state.transform.params.size() != state.params.size()) {
        ggml_free(graph_ctx);
        ggml_free(ctx);
        return fail("graph state must bind every host parameter set");
    }
    const ACETrainAdapterGraphParam & first_graph = graph_state.params.front();
    if (first_graph.a->ne[0] != 128 || first_graph.a->ne[1] != 16 || first_graph.b->ne[0] != 16 ||
        first_graph.b->ne[1] != 256 || !(first_graph.a->flags & GGML_TENSOR_FLAG_PARAM) ||
        !(first_graph.b->flags & GGML_TENSOR_FLAG_PARAM) ||
        !(first_graph.magnitude->flags & GGML_TENSOR_FLAG_PARAM)) {
        ggml_free(graph_ctx);
        ggml_free(ctx);
        return fail("adapter graph tensor shapes or parameter flags are incorrect");
    }
    if (!ace_upload_train_adapter_state(graph_state, state, error) ||
        static_cast<const float *>(first_graph.a->data)[0] != first.a[0] ||
        static_cast<const float *>(first_graph.magnitude->data)[0] != first.magnitude[0]) {
        std::fprintf(stderr, "FAIL: adapter graph upload: %s\n", error.c_str());
        ggml_free(graph_ctx);
        ggml_free(ctx);
        return 1;
    }
    const auto transform_param = graph_state.transform.params.find(first.target.weight);
    if (transform_param == graph_state.transform.params.end() || !transform_param->second.dora_rows ||
        std::fabs(transform_param->second.scale - 2.0f) > 1e-6f) {
        ggml_free(graph_ctx);
        ggml_free(ctx);
        return fail("graph transform must bind DoRA parameters with alpha divided by rank");
    }
    ggml_free(graph_ctx);

    const std::filesystem::path checkpoint_dir =
        std::filesystem::temp_directory_path() /
        ("ace-adapter-checkpoint-" + std::to_string(std::random_device {}()));
    if (!ace_save_train_adapter_checkpoint(checkpoint_dir.string(), state, error)) {
        std::fprintf(stderr, "FAIL: save PEFT checkpoint: %s\n", error.c_str());
        ggml_free(ctx);
        return 1;
    }
    STFile saved_adapter;
    const std::filesystem::path adapter_path = checkpoint_dir / "adapter_model.safetensors";
    if (!st_open(&saved_adapter, adapter_path.string().c_str()) || saved_adapter.entries.size() != 33) {
        std::filesystem::remove_all(checkpoint_dir);
        ggml_free(ctx);
        return fail("saved DoRA checkpoint must contain A, B, and magnitude for every target");
    }
    const std::string expected_a_key =
        "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight";
    const std::string expected_magnitude_key =
        "base_model.model.layers.0.self_attn.q_proj.lora_magnitude_vector.default";
    const STEntry * saved_a = nullptr;
    const STEntry * saved_magnitude = nullptr;
    for (const STEntry & entry : saved_adapter.entries) {
        if (entry.name == expected_a_key) {
            saved_a = &entry;
        } else if (entry.name == expected_magnitude_key) {
            saved_magnitude = &entry;
        }
    }
    if (!saved_a || saved_a->n_dims != 2 || saved_a->shape[0] != 16 || saved_a->shape[1] != 128 ||
        !saved_magnitude || saved_magnitude->n_dims != 1 || saved_magnitude->shape[0] != 256 ||
        static_cast<const float *>(st_data(saved_adapter, *saved_a))[0] != first.a[0]) {
        st_close(&saved_adapter);
        std::filesystem::remove_all(checkpoint_dir);
        ggml_free(ctx);
        return fail("saved PEFT tensor names, shapes, or data are incorrect");
    }
    st_close(&saved_adapter);
    adapter_config config;
    if (!adapter_read_config(checkpoint_dir.string().c_str(), config) || !config.use_dora ||
        config.rank_pattern["self_attn.q_proj"] != 16 || config.rank_pattern["self_attn.v_proj"] != 80 ||
        config.alpha_pattern["self_attn.q_proj"] != 32 || config.alpha_pattern["self_attn.v_proj"] != 160 ||
        adapter_config_value_for_weight(config.alpha_pattern,
                                        "decoder.layers.0.self_attn.q_proj.weight",
                                        config.lora_alpha) != 32 ||
        adapter_config_value_for_weight(config.alpha_pattern,
                                        "decoder.layers.0.self_attn.v_proj.weight",
                                        config.lora_alpha) != 160) {
        std::filesystem::remove_all(checkpoint_dir);
        ggml_free(ctx);
        return fail("adapter_config.json must preserve semantic DoRA rank and alpha metadata");
    }
    ACETrainAdapterState resumed;
    if (!ace_load_train_adapter_checkpoint(checkpoint_dir.string(), balanced, resumed, error) ||
        resumed.adapter_type != "dora-rows" || resumed.params.size() != state.params.size() ||
        resumed.params[0].a != state.params[0].a || resumed.params[0].b != state.params[0].b ||
        resumed.params[0].magnitude != state.params[0].magnitude ||
        resumed.params[0].base_norm_sq != state.params[0].base_norm_sq) {
        std::fprintf(stderr, "FAIL: resume PEFT checkpoint: %s\n", error.c_str());
        std::filesystem::remove_all(checkpoint_dir);
        ggml_free(ctx);
        return 1;
    }
    std::filesystem::remove_all(checkpoint_dir);

    ggml_init_params tiny_params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    ggml_context * tiny_ctx = ggml_init(tiny_params);
    DiTGGML tiny = {};
    tiny.cfg.hidden_size = 4;
    tiny.cfg.intermediate_size = 8;
    tiny.cfg.n_heads = 1;
    tiny.cfg.n_kv_heads = 1;
    tiny.cfg.head_dim = 4;
    tiny.cfg.n_layers = 1;
    tiny.cfg.in_channels = 4;
    tiny.cfg.out_channels = 2;
    tiny.cfg.patch_size = 1;
    tiny.cfg.sliding_window = 2;
    tiny.cfg.rope_theta = 10000.0f;
    tiny.cfg.rms_norm_eps = 1e-5f;
    tiny.backend = ggml_backend_cpu_init();
    tiny.cpu_backend = tiny.backend;
    tiny.use_flash_attn = true;

    auto set_timestep_weights = [&](DiTGGMLTembWeights & weights, const char * prefix) {
        const std::string p(prefix);
        weights.linear_1_w = make_matrix(tiny_ctx, (p + ".linear_1.weight").c_str(), 256, 4, 0.01f);
        weights.linear_1_b = make_vector(tiny_ctx, (p + ".linear_1.bias").c_str(), 4, 0.0f);
        weights.linear_2_w = make_matrix(tiny_ctx, (p + ".linear_2.weight").c_str(), 4, 4, 0.01f);
        weights.linear_2_b = make_vector(tiny_ctx, (p + ".linear_2.bias").c_str(), 4, 0.0f);
        weights.time_proj_w = make_matrix(tiny_ctx, (p + ".time_proj.weight").c_str(), 4, 24, 0.01f);
        weights.time_proj_b = make_vector(tiny_ctx, (p + ".time_proj.bias").c_str(), 24, 0.0f);
    };
    set_timestep_weights(tiny.time_embed, "decoder.time_embed");
    set_timestep_weights(tiny.time_embed_r, "decoder.time_embed_r");
    tiny.proj_in_w = make_matrix(tiny_ctx, "decoder.proj_in.1.weight", 4, 4, 0.01f);
    tiny.proj_in_b = make_vector(tiny_ctx, "decoder.proj_in.1.bias", 4, 0.0f);
    tiny.cond_emb_w = make_matrix(tiny_ctx, "decoder.condition_embedder.weight", 4, 4, 0.01f);
    tiny.cond_emb_b = make_vector(tiny_ctx, "decoder.condition_embedder.bias", 4, 0.0f);

    DiTGGMLLayer & tiny_layer = tiny.layers[0];
    tiny_layer.self_attn_norm = make_vector(tiny_ctx, "tiny.self_attn_norm", 4, 1.0f);
    tiny_layer.sa_q_proj = make_matrix(tiny_ctx, "decoder.layers.0.self_attn.q_proj.weight", 4, 4, 0.01f);
    tiny_layer.sa_k_proj = make_matrix(tiny_ctx, "decoder.layers.0.self_attn.k_proj.weight", 4, 4, 0.01f);
    tiny_layer.sa_v_proj = make_matrix(tiny_ctx, "decoder.layers.0.self_attn.v_proj.weight", 4, 4, 0.01f);
    tiny_layer.sa_q_norm = make_vector(tiny_ctx, "tiny.self_attn.q_norm", 4, 1.0f);
    tiny_layer.sa_k_norm = make_vector(tiny_ctx, "tiny.self_attn.k_norm", 4, 1.0f);
    tiny_layer.sa_o_proj = make_matrix(tiny_ctx, "decoder.layers.0.self_attn.o_proj.weight", 4, 4, 0.01f);
    tiny_layer.cross_attn_norm = make_vector(tiny_ctx, "tiny.cross_attn_norm", 4, 1.0f);
    tiny_layer.ca_q_proj = make_matrix(tiny_ctx, "decoder.layers.0.cross_attn.q_proj.weight", 4, 4, 0.01f);
    tiny_layer.ca_k_proj = make_matrix(tiny_ctx, "decoder.layers.0.cross_attn.k_proj.weight", 4, 4, 0.01f);
    tiny_layer.ca_v_proj = make_matrix(tiny_ctx, "decoder.layers.0.cross_attn.v_proj.weight", 4, 4, 0.01f);
    tiny_layer.ca_q_norm = make_vector(tiny_ctx, "tiny.cross_attn.q_norm", 4, 1.0f);
    tiny_layer.ca_k_norm = make_vector(tiny_ctx, "tiny.cross_attn.k_norm", 4, 1.0f);
    tiny_layer.ca_o_proj = make_matrix(tiny_ctx, "decoder.layers.0.cross_attn.o_proj.weight", 4, 4, 0.01f);
    tiny_layer.mlp_norm = make_vector(tiny_ctx, "tiny.mlp_norm", 4, 1.0f);
    tiny_layer.gate_proj = make_matrix(tiny_ctx, "decoder.layers.0.mlp.gate_proj.weight", 4, 8, 0.01f);
    tiny_layer.up_proj = make_matrix(tiny_ctx, "decoder.layers.0.mlp.up_proj.weight", 4, 8, 0.01f);
    tiny_layer.down_proj = make_matrix(tiny_ctx, "decoder.layers.0.mlp.down_proj.weight", 8, 4, 0.01f);
    tiny_layer.scale_shift_table = make_matrix(tiny_ctx, "tiny.scale_shift", 4, 6, 0.0f);
    tiny_layer.layer_type = 0;

    tiny.norm_out = make_vector(tiny_ctx, "decoder.norm_out.weight", 4, 1.0f);
    tiny.out_scale_shift = make_matrix(tiny_ctx, "decoder.scale_shift_table", 4, 2, 0.0f);
    tiny.proj_out_w = make_matrix(tiny_ctx, "decoder.proj_out.1.weight", 4, 2, 0.01f);
    tiny.proj_out_b = make_vector(tiny_ctx, "decoder.proj_out.1.bias", 2, 0.0f);
    tiny.scalar_one = make_vector(tiny_ctx, "scalar_one", 1, 1.0f);

    std::vector<ACETrainAdapterTarget> tiny_targets;
    if (!ace_train_adapter_targets(tiny, "balanced", 1, 1, tiny_targets, error)) {
        std::fprintf(stderr, "FAIL: tiny targets: %s\n", error.c_str());
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return 1;
    }
    ACETrainAdapterState tiny_state;
    if (!ace_init_train_adapter_state(tiny_targets, "dora-rows", 7, tiny_state, error)) {
        std::fprintf(stderr, "FAIL: tiny state: %s\n", error.c_str());
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return 1;
    }
    ACETrainDiTGraph tiny_training;
    if (!ace_build_train_dit_graph(tiny, tiny_state, 2, 1, 1, tiny_training, error)) {
        std::fprintf(stderr, "FAIL: tiny training graph: %s\n", error.c_str());
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return 1;
    }
    if (!tiny.use_flash_attn) {
        ace_free_train_dit_graph(tiny_training);
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return fail("training graph must preserve the model flash-attention setting");
    }

    const float latent_data[] = { 0.1f, -0.2f, 0.3f, -0.4f, 0.2f, 0.1f, -0.1f, 0.4f };
    const float encoder_data[] = { 0.2f, -0.1f, 0.05f, 0.3f };
    const float target_data[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    const int32_t position_data[] = { 0, 1 };
    const ggml_fp16_t self_mask[] = { ggml_fp32_to_fp16(0.0f), ggml_fp32_to_fp16(0.0f),
                                      ggml_fp32_to_fp16(0.0f), ggml_fp32_to_fp16(0.0f) };
    const ggml_fp16_t cross_mask[] = { ggml_fp32_to_fp16(0.0f), ggml_fp32_to_fp16(0.0f) };
    const float timestep = 0.5f;
    const float loss_weights[] = { 0.5f, 0.5f };
    ggml_backend_tensor_set(tiny_training.input_latents, latent_data, 0, sizeof(latent_data));
    ggml_backend_tensor_set(tiny_training.encoder_hidden, encoder_data, 0, sizeof(encoder_data));
    ggml_backend_tensor_set(tiny_training.target_velocity, target_data, 0, sizeof(target_data));
    ggml_backend_tensor_set(tiny_training.positions, position_data, 0, sizeof(position_data));
    ggml_backend_tensor_set(tiny_training.self_attention_mask, self_mask, 0, sizeof(self_mask));
    ggml_backend_tensor_set(tiny_training.cross_attention_mask, cross_mask, 0, sizeof(cross_mask));
    ggml_backend_tensor_set(tiny_training.timestep, &timestep, 0, sizeof(timestep));
    ggml_backend_tensor_set(tiny_training.reference_timestep, &timestep, 0, sizeof(timestep));
    ggml_backend_tensor_set(tiny_training.loss_weights, loss_weights, 0, sizeof(loss_weights));
    ggml_graph_reset(tiny_training.graph);
    if (ggml_backend_graph_compute(tiny.backend, tiny_training.graph) != GGML_STATUS_SUCCESS) {
        ace_free_train_dit_graph(tiny_training);
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return fail("tiny real DiT forward/backward graph did not execute");
    }
    float tiny_loss = 0.0f;
    ggml_backend_tensor_get(tiny_training.loss, &tiny_loss, 0, sizeof(tiny_loss));
    float largest_gradient = 0.0f;
    for (const ACETrainAdapterGraphParam & param : tiny_training.adapters.params) {
        ggml_tensor * trainable[] = { param.b, param.magnitude };
        for (ggml_tensor * tensor : trainable) {
            ggml_tensor * gradient = ggml_graph_get_grad(tiny_training.graph, tensor);
            std::vector<float> gradient_data((size_t) ggml_nelements(gradient));
            ggml_backend_tensor_get(gradient,
                                    gradient_data.data(),
                                    0,
                                    gradient_data.size() * sizeof(float));
            for (float value : gradient_data) {
                largest_gradient = std::max(largest_gradient, std::fabs(value));
            }
        }
    }
    const bool nonzero_gradient = largest_gradient > 1e-10f;
    if (!std::isfinite(tiny_loss) || !nonzero_gradient) {
        std::fprintf(stderr, "FAIL: tiny training evidence loss=%f largest_gradient=%f\n", tiny_loss, largest_gradient);
        ace_free_train_dit_graph(tiny_training);
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return fail("tiny training graph must produce finite loss and a nonzero adapter gradient");
    }
    std::vector<float> parameters_before;
    for (const ACETrainAdapterParam & param : tiny_state.params) {
        parameters_before.insert(parameters_before.end(), param.a.begin(), param.a.end());
        parameters_before.insert(parameters_before.end(), param.b.begin(), param.b.end());
        parameters_before.insert(parameters_before.end(), param.magnitude.begin(), param.magnitude.end());
    }
    ACETrainAdapterOptimizer optimizer;
    ACETrainAdamWConfig optimizer_config;
    optimizer_config.learning_rate = 1e-3f;
    optimizer_config.weight_decay = 0.01f;
    optimizer_config.max_gradient_norm = 1.0f;
    if (std::fabs(ace_train_learning_rate(1e-3f, 0, 10, 2, ACE_TRAIN_SCHEDULE_COSINE) - 1e-4f) > 1e-8f ||
        std::fabs(ace_train_learning_rate(1e-3f, 2, 10, 2, ACE_TRAIN_SCHEDULE_COSINE) - 1e-3f) > 1e-8f ||
        std::fabs(ace_train_learning_rate(1e-3f, 10, 10, 2, ACE_TRAIN_SCHEDULE_COSINE) - 1e-5f) > 1e-8f) {
        ace_free_train_dit_graph(tiny_training);
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return fail("warmup and cosine learning-rate schedule does not match the trainer contract");
    }
    ACETrainAdapterGradientAccumulator accumulated_gradients;
    if (!ace_train_adapter_accumulate_gradients(tiny_training, accumulated_gradients, error) ||
        !ace_train_adapter_accumulate_gradients(tiny_training, accumulated_gradients, error) ||
        accumulated_gradients.microbatch_count != 2 ||
        !ace_train_adapter_adamw_step_accumulated(
            tiny_training, tiny_state, optimizer, optimizer_config, accumulated_gradients, error) ||
        optimizer.step != 1 || accumulated_gradients.microbatch_count != 0) {
        std::fprintf(stderr, "FAIL: native AdamW step: %s\n", error.c_str());
        ace_free_train_dit_graph(tiny_training);
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return 1;
    }
    std::vector<float> parameters_after;
    for (const ACETrainAdapterParam & param : tiny_state.params) {
        parameters_after.insert(parameters_after.end(), param.a.begin(), param.a.end());
        parameters_after.insert(parameters_after.end(), param.b.begin(), param.b.end());
        parameters_after.insert(parameters_after.end(), param.magnitude.begin(), param.magnitude.end());
    }
    float uploaded_magnitude = 0.0f;
    ggml_backend_tensor_get(tiny_training.adapters.params[0].magnitude,
                            &uploaded_magnitude,
                            0,
                            sizeof(uploaded_magnitude));
    if (parameters_before == parameters_after || uploaded_magnitude != tiny_state.params[0].magnitude[0]) {
        ace_free_train_dit_graph(tiny_training);
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return fail("AdamW must update host parameters and upload them for the next graph replay");
    }
    const std::filesystem::path training_checkpoint_dir =
        std::filesystem::temp_directory_path() /
        ("ace-training-checkpoint-" + std::to_string(std::random_device {}()));
    ACETrainAdapterState resumed_training_state;
    ACETrainAdapterOptimizer resumed_optimizer;
    int resumed_epochs = 0;
    if (!ace_save_train_checkpoint(
            training_checkpoint_dir.string(), tiny_state, optimizer, 3, error) ||
        !ace_load_train_checkpoint(training_checkpoint_dir.string(),
                                   tiny_targets,
                                   resumed_training_state,
                                   resumed_optimizer,
                                   resumed_epochs,
                                   error) ||
        resumed_epochs != 3 || resumed_optimizer.step != optimizer.step ||
        resumed_training_state.params[0].magnitude != tiny_state.params[0].magnitude ||
        resumed_optimizer.params[0].b.first_moment != optimizer.params[0].b.first_moment ||
        resumed_optimizer.params[0].magnitude.second_moment != optimizer.params[0].magnitude.second_moment) {
        std::fprintf(stderr, "FAIL: full native training checkpoint: %s\n", error.c_str());
        std::filesystem::remove_all(training_checkpoint_dir);
        ace_free_train_dit_graph(tiny_training);
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return 1;
    }
    std::filesystem::remove_all(training_checkpoint_dir);
    ace_free_train_dit_graph(tiny_training);

    std::vector<ACETrainDiffusionExample> examples(2);
    examples[0].target_latents = { 0.1f, -0.2f };
    examples[0].context_latents = { 0.5f, 0.6f };
    examples[0].encoder_hidden = { 0.2f, -0.1f, 0.05f, 0.3f };
    examples[0].real_encoder_sequence_length = 1;
    examples[0].real_temporal_length = 1;
    examples[1].target_latents = { -0.4f, 0.3f, -0.2f, 0.1f };
    examples[1].context_latents = { -0.8f, -0.7f, -0.6f, -0.5f };
    examples[1].encoder_hidden = { -0.3f, 0.05f, -0.1f, 0.2f };
    examples[1].real_encoder_sequence_length = 1;
    examples[1].real_temporal_length = 2;
    const std::vector<float> null_condition = { 0.9f, 0.8f, 0.7f, 0.6f };
    const std::vector<float> silence_latents = { 0.01f, 0.02f, 0.03f, 0.04f };
    std::vector<ACETrainDiffusionExample> collated_examples;
    int collated_temporal_length = 0;
    int collated_encoder_sequence_length = 0;
    if (!ace_collate_training_examples(tiny,
                                       examples,
                                       silence_latents,
                                       null_condition,
                                       collated_examples,
                                       collated_temporal_length,
                                       collated_encoder_sequence_length,
                                       error) ||
        collated_temporal_length != 2 || collated_encoder_sequence_length != 1 ||
        collated_examples[0].target_latents != std::vector<float>({ 0.1f, -0.2f, 0.0f, 0.0f }) ||
        collated_examples[0].context_latents != std::vector<float>({ 0.5f, 0.6f, 0.03f, 0.04f })) {
        std::fprintf(stderr, "FAIL: collate variable training examples: %s\n", error.c_str());
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return 1;
    }
    examples = std::move(collated_examples);
    ACETrainDiffusionConfig diffusion_config;
    diffusion_config.timestep_mean = 0.0f;
    diffusion_config.timestep_std = 0.0f;
    diffusion_config.cfg_dropout = 1.0f;
    diffusion_config.min_snr = true;
    diffusion_config.min_snr_gamma = 5.0f;
    if (std::fabs(ace_train_timestep_from_logits(-2.0f, 1.0f) - ace_train_sigmoid(1.0f)) > 1e-7f) {
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return fail("ACE-Step timestep sampling must keep the maximum of the sampled pair");
    }
    const float expected_low_t_weight = 5.0f / 81.0f;
    if (std::fabs(ace_train_flow_min_snr_weight(0.1f, 5.0f) - expected_low_t_weight) > 1e-6f ||
        ace_train_flow_min_snr_weight(0.5f, 5.0f) != 1.0f ||
        ace_train_flow_min_snr_weight(0.9f, 5.0f) != 1.0f) {
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return fail("flow Min-SNR weights do not match Gary's timestep convention");
    }
    ACETrainDiffusionBatch batch;
    if (!ace_prepare_train_diffusion_batch(
            tiny,
            examples,
            collated_temporal_length,
            collated_encoder_sequence_length,
            null_condition,
            1234,
            diffusion_config,
            batch,
            error)) {
        std::fprintf(stderr, "FAIL: prepare diffusion batch: %s\n", error.c_str());
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return 1;
    }
    if (batch.timesteps.size() != 2 || batch.timesteps[0] != 0.5f || batch.timesteps[1] != 0.5f ||
        batch.reference_timesteps != batch.timesteps || batch.input_latents.size() != 16 ||
        batch.target_velocity.size() != 8 || batch.encoder_hidden.size() != 8 || batch.loss_weights.size() != 4 ||
        batch.loss_weights != std::vector<float>({ 1.0f, 0.0f, 0.5f, 0.5f }) ||
        batch.positions != std::vector<int32_t>({ 0, 1, 0, 1 })) {
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return fail("prepared diffusion batch dimensions or fixed timesteps are incorrect");
    }
    if (!std::isinf(ggml_fp16_to_fp32(batch.self_attention_mask[1])) ||
        ggml_fp16_to_fp32(batch.self_attention_mask[1]) >= 0.0f ||
        ggml_fp16_to_fp32(batch.self_attention_mask[0]) != 0.0f ||
        ggml_fp16_to_fp32(batch.self_attention_mask[2]) != 0.0f ||
        !std::isinf(ggml_fp16_to_fp32(batch.self_attention_mask[3]))) {
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return fail("self attention must hide padded keys without creating an all-masked padded query");
    }
    for (size_t sample = 0; sample < examples.size(); ++sample) {
        for (int time = 0; time < 2; ++time) {
            const size_t input_offset = sample * 8 + (size_t) time * 4;
            const size_t latent_offset = sample * 4 + (size_t) time * 2;
            if (batch.input_latents[input_offset] != examples[sample].context_latents[(size_t) time * 2] ||
                batch.input_latents[input_offset + 1] != examples[sample].context_latents[(size_t) time * 2 + 1]) {
                ggml_backend_free(tiny.backend);
                ggml_free(tiny_ctx);
                ggml_free(ctx);
                return fail("prepared diffusion input must pack context channels first");
            }
            for (int channel = 0; channel < 2; ++channel) {
                const size_t local_index = (size_t) time * 2 + channel;
                const size_t index = latent_offset + (size_t) channel;
                const float expected_xt =
                    examples[sample].target_latents[local_index] + 0.5f * batch.target_velocity[index];
                if (std::fabs(batch.input_latents[input_offset + 2 + channel] - expected_xt) > 1e-6f) {
                    std::fprintf(stderr,
                                 "FAIL: interpolation sample=%zu time=%d channel=%d input=%f expected=%f velocity=%f\n",
                                 sample,
                                 time,
                                 channel,
                                 batch.input_latents[input_offset + 2 + channel],
                                 expected_xt,
                                 batch.target_velocity[index]);
                    ggml_backend_free(tiny.backend);
                    ggml_free(tiny_ctx);
                    ggml_free(ctx);
                    return fail("prepared noisy latent does not satisfy the flow-matching interpolation");
                }
            }
        }
        for (size_t i = 0; i < null_condition.size(); ++i) {
            if (batch.encoder_hidden[sample * null_condition.size() + i] != null_condition[i]) {
                ggml_backend_free(tiny.backend);
                ggml_free(tiny_ctx);
                ggml_free(ctx);
                return fail("CFG dropout must replace the complete sample condition");
            }
        }
    }
    ACETrainDiffusionBatch replay;
    if (!ace_prepare_train_diffusion_batch(
            tiny, examples, 2, 1, null_condition, 1234, diffusion_config, replay, error) ||
        replay.input_latents != batch.input_latents || replay.target_velocity != batch.target_velocity) {
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return fail("diffusion batch preparation must be deterministic for a fixed seed");
    }

    ACETrainDiTGraph batch_training;
    if (!ace_build_train_dit_graph(tiny, tiny_state, 2, 1, 2, batch_training, error)) {
        std::fprintf(stderr, "FAIL: batched training graph: %s\n", error.c_str());
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return 1;
    }
    if (batch_training.timestep->ne[0] != 2 || batch_training.reference_timestep->ne[0] != 2 ||
        !ace_upload_train_diffusion_batch(batch_training, batch, error)) {
        std::fprintf(stderr, "FAIL: upload prepared diffusion batch: %s\n", error.c_str());
        ace_free_train_dit_graph(batch_training);
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return 1;
    }
    ACETrainAdapterOptimizer batch_optimizer;
    float batch_loss = 0.0f;
    if (!ace_train_adapter_step(batch_training,
                                tiny_state,
                                batch_optimizer,
                                optimizer_config,
                                batch,
                                batch_loss,
                                error) ||
        batch_optimizer.step != 1 || !std::isfinite(batch_loss)) {
        std::fprintf(stderr, "FAIL: prepared native training step: %s loss=%f\n", error.c_str(), batch_loss);
        ace_free_train_dit_graph(batch_training);
        ggml_backend_free(tiny.backend);
        ggml_free(tiny_ctx);
        ggml_free(ctx);
        return 1;
    }
    ace_free_train_dit_graph(batch_training);
    ggml_backend_free(tiny.backend);
    ggml_free(tiny_ctx);

    DiTGGML fused = {};
    fused.cfg.n_layers = 1;
    fused.layers[0].sa_qkv = make_weight(ctx, "", 128, 768);
    std::vector<ACETrainAdapterTarget> rejected;
    if (ace_train_adapter_targets(fused, "balanced", 64, 128, rejected, error) ||
        error.find("fusion") == std::string::npos) {
        ggml_free(ctx);
        return fail("training target inventory must reject fused projection weights");
    }

    ggml_free(ctx);
    std::puts("ACE-Step adapter target profile and state: OK");
    return 0;
}
