// Opt-in real-data multi-update replay using production checkpointed backward and AdamW.
#include "debug.h"
#include "dit.h"
#include "train-adapter-checkpoint.h"
#include "train-diffusion.h"
#include "train-window.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>

static bool save_gradients(const std::filesystem::path &              path,
                           const ACETrainAdapterState &               state,
                           const ACETrainAdapterGradientAccumulator & gradients,
                           std::string &                              error) {
    std::vector<ACETrainCheckpointTensor> tensors;
    for (size_t i = 0; i < state.params.size(); ++i) {
        const auto & p      = state.params[i];
        const auto & g      = gradients.params[i];
        const auto   module = ace_checkpoint_module_path(p.target.weight_name);
        tensors.push_back({
            module + ".lora_A.weight", { p.target.rank, p.target.in },
               &g.a
        });
        tensors.push_back({
            module + ".lora_B.weight", { p.target.out, p.target.rank },
               &g.b
        });
        if (!p.magnitude.empty()) {
            tensors.push_back({ module + ".lora_magnitude_vector", { p.target.out }, &g.magnitude });
        }
    }
    return ace_write_safetensors(path, tensors, error);
}

// Replay the retained, fully valid short-window fixture with identical adapter
// weights. Reconstruct masks through the production batch builder, then replace
// its random inputs with captures. No optimizer update is performed.
static void trace_tensor(DebugDumper & dump, const std::string & name, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor));
    std::vector<float> values(size_t(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    const int shape[] = { int(tensor->ne[2]), int(tensor->ne[1]), int(tensor->ne[0]) };
    debug_dump(&dump, name.c_str(), values.data(), shape, 3);
}

static int replay_update(int argc, char ** argv) {
    const bool tracing = argc == 7 && std::string(argv[6]) == "--trace";
    if ((argc != 6 && !tracing) || std::filesystem::exists(argv[5])) {
        std::fprintf(stderr,
                     "Usage: %s replay <dit.gguf> <before-adapter> <captured-update> <fresh-output> [--trace]\n",
                     argv[0]);
        return 1;
    }
    const std::filesystem::path input(argv[4]), output(argv[5]);
    DiTGGML                     model = {};
    std::string                 error;
    if (!dit_ggml_load(&model, argv[2], nullptr, 1.0f, false)) {
        return 1;
    }
    std::vector<ACETrainAdapterTarget> targets;
    ACETrainAdapterState               state;
    bool                               ok = ace_train_adapter_targets(model, "balanced", 8, 16, targets, error) &&
              ace_load_train_adapter_checkpoint(argv[3], targets, state, error, "dora-rows");
    std::vector<int> shape;
    const auto       combined = debug_load((input / "input_latents.bin").string().c_str(), shape);
    const int        temporal = shape.size() == 3 && shape[0] == 1 && shape[2] == model.cfg.in_channels ? shape[1] : 0;
    auto             encoder  = debug_load((input / "encoder_hidden.bin").string().c_str(), shape);
    const int sequence = shape.size() == 3 && shape[0] == 1 && shape[2] == model.cond_emb_w->ne[0] ? shape[1] : 0;
    if (temporal <= 0 || sequence <= 0) {
        std::fprintf(stderr, "FAIL replay: invalid captured tensor shapes\n");
        dit_ggml_free(&model);
        return 1;
    }
    ACETrainDiffusionExample example;
    example.target_latents.resize(size_t(temporal) * model.cfg.out_channels);
    example.context_latents.resize(size_t(temporal) * (model.cfg.in_channels - model.cfg.out_channels));
    example.encoder_hidden               = encoder;
    example.real_temporal_length         = temporal;
    example.real_encoder_sequence_length = sequence;
    ACETrainDiffusionConfig config;
    config.cfg_dropout = 0;
    ACETrainDiffusionBatch batch;
    ok = ok && temporal > 0 && sequence > 0 &&
         ace_prepare_train_diffusion_batch(model, { example }, temporal, sequence, {}, 0, config, batch, error);
    batch.input_latents       = combined;
    batch.encoder_hidden      = std::move(encoder);
    batch.target_velocity     = debug_load((input / "target_velocity.bin").string().c_str(), shape);
    batch.timesteps           = debug_load((input / "timesteps.bin").string().c_str(), shape);
    batch.reference_timesteps = batch.timesteps;
    if (batch.timesteps.size() != 1) {
        ok = false;
    } else {
        std::fill(batch.loss_weights.begin(), batch.loss_weights.end(),
                  ace_train_flow_min_snr_weight(batch.timesteps[0], 5) / temporal);
    }
    ACETrainDiTCheckpoint              graph = {};
    ACETrainAdapterGradientAccumulator gradients;
    DebugDumper                        trace_dump = {};
    ACETrainCheckpointTrace            trace;
    if (tracing) {
        const auto directory = output / "trace";
        std::filesystem::create_directories(directory);
        debug_init(&trace_dump, directory.string().c_str());
        trace.data    = &trace_dump;
        trace.forward = [](void * data, const ACETrainDiTCheckpoint & checkpoint) {
            auto & dump = *static_cast<DebugDumper *>(data);
            trace_tensor(dump, "temb", checkpoint.saved_timestep_embedding);
            trace_tensor(dump, "tproj", checkpoint.saved_timestep_projection);
            trace_tensor(dump, "condition", checkpoint.saved_encoder_hidden);
            for (size_t i = 0; i < checkpoint.boundaries.size(); ++i) {
                trace_tensor(dump, "boundary-" + std::to_string(i), checkpoint.boundaries[i]);
            }
        };
        trace.gradient = [](void * data, int boundary, const ggml_tensor * gradient) {
            trace_tensor(*static_cast<DebugDumper *>(data), "gradient-" + std::to_string(boundary), gradient);
        };
    }
    float loss = 0;
    ok         = ok && ace_build_train_dit_checkpoint(model, state, temporal, sequence, 1, graph, error, true) &&
         ace_compute_train_adapter_gradients_checkpointed(graph, batch, gradients, 1, loss, error,
                                                          tracing ? &trace : nullptr);
    if (ok) {
        std::filesystem::create_directories(output);
        ok = save_gradients(output / "native_gradients.safetensors", state, gradients, error);
        std::vector<float> velocity(size_t(temporal) * model.cfg.out_channels);
        ggml_backend_tensor_get(graph.velocity_snapshot, velocity.data(), 0, velocity.size() * sizeof(float));
        DebugDumper dump;
        debug_init(&dump, output.string().c_str());
        const int target_shape[] = { 1, temporal, model.cfg.out_channels };
        debug_dump(&dump, "native_velocity", velocity.data(), target_shape, 3);
        std::ofstream metrics(output / "native_metrics.json");
        metrics << std::setprecision(12) << "{\"loss\":" << loss << "}\n";
    }
    ace_free_train_dit_checkpoint(graph);
    dit_ggml_free(&model);
    if (!ok) {
        std::fprintf(stderr, "FAIL replay: %s\n", error.c_str());
    }
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "replay") {
        return replay_update(argc, argv);
    }
    if (argc < 5 || argc > 8) {
        std::fprintf(stderr,
                     "Usage: %s <dit.gguf> <preprocess-fixture-dir> <fresh-output> <steps> "
                     "[full] [--snapshot-step N]\n",
                     argv[0]);
        return 1;
    }
    bool full     = false;
    int  snapshot = 0;
    for (int i = 5; i < argc; ++i) {
        if (std::string(argv[i]) == "full") {
            full = true;
        } else if (std::string(argv[i]) == "--snapshot-step" && i + 1 < argc) {
            snapshot = std::atoi(argv[++i]);
            if (snapshot < 1) {
                return 1;
            }
        } else {
            return 1;
        }
    }
    const std::filesystem::path input(argv[2]), output(argv[3]);
    const int                   steps = std::atoi(argv[4]);
    if (steps < 1 || steps > 1000 || snapshot > steps || std::filesystem::exists(output)) {
        return 1;
    }
    std::filesystem::create_directories(output);
    DiTGGML     model = {};
    std::string error;
    if (!dit_ggml_load(&model, argv[1], nullptr, 1.0f, false)) {
        return 1;
    }
    std::vector<ACETrainAdapterTarget> targets;
    ACETrainAdapterState               state;
    bool                               ok = ace_train_adapter_targets(model, "balanced", 8, 16, targets, error) &&
              ace_init_train_adapter_state(targets, "dora-rows", 42, state, error) &&
              ace_save_train_adapter_checkpoint((output / "initial").string(), state, error);
    std::vector<ACETrainDiffusionExample> examples;
    std::vector<float>                    null_condition;
    for (size_t i = 0; ok && std::filesystem::is_directory(input / std::to_string(i)); ++i) {
        const auto               dir = input / std::to_string(i);
        ACETrainDiffusionExample example;
        std::vector<int>         shape;
        example.target_latents               = debug_load((dir / "target.bin").string().c_str(), shape);
        example.context_latents              = debug_load((dir / "context.bin").string().c_str(), shape);
        example.encoder_hidden               = debug_load((dir / "encoder.bin").string().c_str(), shape);
        example.real_encoder_sequence_length = shape.empty() ? 0 : shape[0];
        std::ifstream     metadata(dir / "sample.json");
        const std::string json((std::istreambuf_iterator<char>(metadata)), {});
        yyjson_doc *      doc = yyjson_read(json.data(), json.size(), 0);
        example.real_temporal_length =
            int(yyjson_get_int(yyjson_obj_get(yyjson_doc_get_root(doc), "real_temporal_length")));
        yyjson_doc_free(doc);
        if (i == 0) {
            null_condition = debug_load((dir / "null_condition.bin").string().c_str(), shape);
        }
        ok = !example.target_latents.empty() && !example.context_latents.empty() && !example.encoder_hidden.empty() &&
             example.real_temporal_length > 0;
        examples.push_back(std::move(example));
    }
    ok = ok && !examples.empty();
    ACETrainAdapterOptimizer optimizer;
    ACETrainAdamWConfig      adamw;
    adamw.weight_decay      = 0.01f;
    adamw.max_gradient_norm = 1.0f;
    ACETrainDiffusionConfig diffusion;
    diffusion.cfg_dropout   = 0.15f;
    diffusion.timestep_mean = 0.0f;
    diffusion.timestep_std  = 1.0f;
    diffusion.min_snr       = true;
    diffusion.min_snr_gamma = 5.0f;
    ACETrainWindowConfig window_config;
    window_config.min_seconds = 4;
    window_config.max_seconds = 8;
    std::ofstream metrics(output / "metrics.jsonl");
    metrics << std::setprecision(12);
    for (int step = 0; ok && step < steps; ++step) {
        const auto               started = std::chrono::steady_clock::now();
        const size_t             index   = size_t(step) % examples.size();
        ACETrainDiffusionExample example;
        ACETrainWindow           window;
        ok                                          = ace_train_crop_example(examples[index], model.cfg.out_channels,
                                                                             model.cfg.in_channels - model.cfg.out_channels, model.cfg.patch_size, window_config,
                                                                             42, int(step / examples.size()), index, example, window, error);
        const int                          temporal = int(example.target_latents.size() / model.cfg.out_channels);
        ACETrainDiffusionBatch             batch;
        ACETrainDiTCheckpoint              graph      = {};
        ACETrainDiTGraph                   full_graph = {};
        ACETrainAdapterGradientAccumulator gradients;
        float                              loss = 0;
        ok = ok && ace_prepare_train_diffusion_batch(model, { example }, temporal, example.real_encoder_sequence_length,
                                                     null_condition, 4242 + step, diffusion, batch, error);
        if (ok && full) {
            ok = ace_build_train_dit_graph(model, state, temporal, example.real_encoder_sequence_length, 1, full_graph,
                                           error, true) &&
                 ace_compute_train_adapter_gradients(full_graph, batch, loss, error) &&
                 ace_train_adapter_accumulate_gradients(full_graph, gradients, 1, error);
        } else if (ok) {
            ok = ace_build_train_dit_checkpoint(model, state, temporal, example.real_encoder_sequence_length, 1, graph,
                                                error, true) &&
                 ace_compute_train_adapter_gradients_checkpointed(graph, batch, gradients, 1, loss, error);
        }
        if (ok) {
            const auto dir = output / std::to_string(step + 1);
            std::filesystem::create_directory(dir);
            if (step == 0 || step + 1 == snapshot) {
                ok = save_gradients(dir / "native_gradients.safetensors", state, gradients, error);
            }
            DebugDumper dump;
            debug_init(&dump, dir.string().c_str());
            const int input_shape[]   = { 1, temporal, model.cfg.in_channels };
            const int target_shape[]  = { 1, temporal, model.cfg.out_channels };
            const int encoder_shape[] = { 1, example.real_encoder_sequence_length, int(null_condition.size()) };
            debug_dump(&dump, "input_latents", batch.input_latents.data(), input_shape, 3);
            debug_dump(&dump, "encoder_hidden", batch.encoder_hidden.data(), encoder_shape, 3);
            debug_dump(&dump, "target_velocity", batch.target_velocity.data(), target_shape, 3);
            debug_dump_1d(&dump, "timesteps", batch.timesteps.data(), 1);
            std::vector<float> velocity(size_t(temporal) * model.cfg.out_channels);
            ggml_backend_tensor_get(full ? full_graph.velocity_snapshot : graph.velocity_snapshot, velocity.data(), 0,
                                    velocity.size() * sizeof(float));
            debug_dump(&dump, "native_velocity", velocity.data(), target_shape, 3);
            double grad_norm = 0;
            adamw.learning_rate =
                ace_train_learning_rate(3e-4f, step, steps, std::min(2, steps), ACE_TRAIN_SCHEDULE_COSINE);
            ok = ace_train_adapter_adamw_step_accumulated(full ? full_graph.adapters : graph.adapters, state, optimizer,
                                                          adamw, gradients, error, &grad_norm);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            metrics << "{\"update\":" << step + 1 << ",\"example\":" << index
                    << ",\"offset_frames\":" << window.first_latent << ",\"frames\":" << temporal
                    << ",\"loss\":" << loss << ",\"lr\":" << adamw.learning_rate << ",\"grad_norm\":" << grad_norm
                    << ",\"seconds\":" << seconds << "}\n";
            metrics.flush();
            if (ok && ((step + 1) % examples.size() == 0 || step + 1 == steps || step + 1 == snapshot ||
                       step + 2 == snapshot)) {
                ok = ace_save_train_adapter_checkpoint((dir / "updated").string(), state, error);
            }
            std::fprintf(stderr, "[Trajectory] %d/%d loss=%g grad=%g seconds=%.2f\n", step + 1, steps, loss, grad_norm,
                         seconds);
        }
        ace_free_train_dit_checkpoint(graph);
        ace_free_train_dit_graph(full_graph);
    }
    dit_ggml_free(&model);
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    }
    return ok ? 0 : 1;
}
