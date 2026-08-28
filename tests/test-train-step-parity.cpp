// ABOUTME: Emits a deterministic real-model native training step for PyTorch parity checks.
// ABOUTME: Saves exact batch tensors, adapter gradients, and pre/post AdamW checkpoints.

#include "debug.h"
#include "dit.h"
#include "train-adapter-checkpoint.h"
#include "train-adapter-optimizer.h"
#include "train-diffusion.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static bool write_gradients(const std::filesystem::path &              path,
                            const ACETrainAdapterState &               state,
                            const ACETrainAdapterGradientAccumulator & gradients,
                            std::string &                              error) {
    if (state.params.size() != gradients.params.size()) {
        error = "adapter state and gradient inventory differ";
        return false;
    }
    std::vector<ACETrainCheckpointTensor> tensors;
    tensors.reserve(state.params.size() * 3);
    for (size_t i = 0; i < state.params.size(); ++i) {
        const ACETrainAdapterParam &         parameter = state.params[i];
        const ACETrainAdapterGradientParam & gradient  = gradients.params[i];
        const std::string                    module    = ace_checkpoint_module_path(parameter.target.weight_name);
        tensors.push_back({
            module + ".lora_A.weight", { parameter.target.rank, parameter.target.in },
               &gradient.a
        });
        tensors.push_back({
            module + ".lora_B.weight", { parameter.target.out, parameter.target.rank },
               &gradient.b
        });
        if (!parameter.magnitude.empty()) {
            tensors.push_back({ module + ".lora_magnitude_vector", { parameter.target.out }, &gradient.magnitude });
        }
    }
    return ace_write_safetensors(path, tensors, error);
}

static double gradient_norm(const ACETrainAdapterGradientAccumulator & gradients) {
    double squared = 0.0;
    auto   add     = [&](const std::vector<float> & values) {
        for (float value : values) {
            squared += (double) value * value;
        }
    };
    for (const ACETrainAdapterGradientParam & gradient : gradients.params) {
        add(gradient.a);
        add(gradient.b);
        add(gradient.magnitude);
    }
    return std::sqrt(squared);
}

static void dump_batch(const std::filesystem::path &  output,
                       const DiTGGML &                model,
                       const ACETrainDiffusionBatch & batch,
                       const std::vector<float> &     velocity,
                       float                          loss) {
    DebugDumper dumper;
    debug_init(&dumper, output.string().c_str());
    const int batch_shape[]   = { batch.batch_size, batch.temporal_length, model.cfg.in_channels };
    const int target_shape[]  = { batch.batch_size, batch.temporal_length, model.cfg.out_channels };
    const int encoder_shape[] = { batch.batch_size, batch.encoder_sequence_length, (int) model.cond_emb_w->ne[0] };
    const int time_shape[]    = { batch.batch_size };
    debug_dump(&dumper, "input_latents", batch.input_latents.data(), batch_shape, 3);
    debug_dump(&dumper, "encoder_hidden", batch.encoder_hidden.data(), encoder_shape, 3);
    debug_dump(&dumper, "target_velocity", batch.target_velocity.data(), target_shape, 3);
    debug_dump(&dumper, "native_velocity", velocity.data(), target_shape, 3);
    debug_dump(&dumper, "timesteps", batch.timesteps.data(), time_shape, 1);
    debug_dump(&dumper, "native_loss", &loss, time_shape, 1);
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <DiT-GGUF> <output-dir>\n", argv[0]);
        return 1;
    }
    const std::filesystem::path output(argv[2]);
    std::error_code             filesystem_error;
    std::filesystem::create_directories(output, filesystem_error);
    if (filesystem_error) {
        std::fprintf(stderr, "FAIL: cannot create output directory: %s\n", filesystem_error.message().c_str());
        return 1;
    }

    std::string error;
    DiTGGML     model = {};
    if (!dit_ggml_load(&model, argv[1], nullptr, 1.0f, false)) {
        std::fprintf(stderr, "FAIL: could not load %s\n", argv[1]);
        return 1;
    }

    std::vector<ACETrainAdapterTarget> targets;
    ACETrainAdapterState               state;
    constexpr int                      base_rank  = 8;
    constexpr int                      base_alpha = 16;
    if (!ace_train_adapter_targets(model, "balanced", base_rank, base_alpha, targets, error) ||
        !ace_init_train_adapter_state(targets, "dora-rows", 123, state, error) ||
        !ace_save_train_adapter_checkpoint((output / "initial").string(), state, error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        dit_ggml_free(&model);
        return 1;
    }

    const int                temporal_length  = model.cfg.patch_size * 4;
    const int                encoder_length   = 8;
    const int                output_channels  = model.cfg.out_channels;
    const int                context_channels = model.cfg.in_channels - output_channels;
    const int                encoder_width    = (int) model.cond_emb_w->ne[0];
    ACETrainDiffusionExample example;
    example.target_latents.resize((size_t) temporal_length * output_channels);
    example.context_latents.resize((size_t) temporal_length * context_channels);
    example.encoder_hidden.resize((size_t) encoder_length * encoder_width);
    for (size_t i = 0; i < example.target_latents.size(); ++i) {
        example.target_latents[i] = 0.25f * std::sin((float) i * 0.017f);
    }
    for (size_t i = 0; i < example.context_latents.size(); ++i) {
        example.context_latents[i] = 0.1f * std::cos((float) i * 0.011f);
    }
    for (size_t i = 0; i < example.encoder_hidden.size(); ++i) {
        example.encoder_hidden[i] = 0.05f * std::sin((float) i * 0.007f);
    }
    example.real_temporal_length         = temporal_length;
    example.real_encoder_sequence_length = encoder_length;

    std::vector<float>      null_condition((size_t) encoder_width, 0.0f);
    ACETrainDiffusionConfig diffusion;
    diffusion.timestep_mean = std::log(1.0f / 3.0f);  // sigmoid(log(1/3)) = 0.25
    diffusion.timestep_std  = 0.0f;
    diffusion.cfg_dropout   = 0.0f;
    diffusion.min_snr       = true;
    diffusion.min_snr_gamma = 5.0f;
    ACETrainDiffusionBatch batch;
    if (!ace_prepare_train_diffusion_batch(model, { example }, temporal_length, encoder_length, null_condition, 456,
                                           diffusion, batch, error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        dit_ggml_free(&model);
        return 1;
    }

    ACETrainDiTGraph graph = {};
    if (!ace_build_train_dit_graph(model, state, temporal_length, encoder_length, 1, graph, error, true)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        dit_ggml_free(&model);
        return 1;
    }
    float                              loss = 0.0f;
    ACETrainAdapterGradientAccumulator gradients;
    if (!ace_compute_train_adapter_gradients(graph, batch, loss, error) ||
        !ace_train_adapter_accumulate_gradients(graph, gradients, 1, error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        ace_free_train_dit_graph(graph);
        dit_ggml_free(&model);
        return 1;
    }

    std::vector<float> velocity((size_t) temporal_length * output_channels);
    ggml_backend_tensor_get(graph.velocity_snapshot, velocity.data(), 0, velocity.size() * sizeof(float));
    dump_batch(output, model, batch, velocity, loss);
    const double native_gradient_norm = gradient_norm(gradients);
    if (!write_gradients(output / "native_gradients.safetensors", state, gradients, error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        ace_free_train_dit_graph(graph);
        dit_ggml_free(&model);
        return 1;
    }

    ACETrainAdapterOptimizer optimizer;
    ACETrainAdamWConfig      adamw;
    adamw.learning_rate     = 3e-5f;
    adamw.weight_decay      = 0.01f;
    adamw.max_gradient_norm = 1.0f;
    if (!ace_train_adapter_adamw_step_accumulated(graph.adapters, state, optimizer, adamw, gradients, error) ||
        !ace_save_train_adapter_checkpoint((output / "native_updated").string(), state, error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        ace_free_train_dit_graph(graph);
        dit_ggml_free(&model);
        return 1;
    }

    std::ofstream metrics(output / "native_metrics.json", std::ios::trunc);
    metrics << "{\n"
            << "  \"loss\": " << loss << ",\n"
            << "  \"gradient_norm\": " << native_gradient_norm << ",\n"
            << "  \"learning_rate\": " << adamw.learning_rate << ",\n"
            << "  \"weight_decay\": " << adamw.weight_decay << ",\n"
            << "  \"max_gradient_norm\": " << adamw.max_gradient_norm << "\n"
            << "}\n";

    ace_free_train_dit_graph(graph);
    dit_ggml_free(&model);
    std::fprintf(stderr, "PASS: native training parity fixture written to %s\n", output.string().c_str());
    return 0;
}
