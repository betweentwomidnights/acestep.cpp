// ABOUTME: Runs native ACE-Step LoRA and DoRA-row training from paired WAV and metadata files.
// ABOUTME: Reuses GGUF preprocessing, batches variable lengths, accumulates gradients, and saves PEFT adapters.

#include "dit.h"
#include "model-registry.h"
#include "model-store.h"
#include "pipeline-synth.h"
#include "pipeline-train.h"
#include "train-checkpoint.h"
#include "train-diffusion.h"
#include "version.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct ACETrainCommand {
    const char *             models_dir          = nullptr;
    const char *             dataset_dir         = nullptr;
    const char *             output_dir          = nullptr;
    const char *             resume_dir          = nullptr;
    const char *             dit_name            = nullptr;
    const char *             text_name           = nullptr;
    const char *             vae_name            = nullptr;
    std::string              adapter_type        = "dora-rows";
    std::string              module_profile      = "balanced";
    int                      rank                = 64;
    int                      alpha               = 128;
    int                      epochs              = 10;
    int                      batch_size          = 1;
    int                      accumulation        = 1;
    int                      warmup_steps        = 10;
    int                      restart_count       = 1;
    uint64_t                 seed                = 42;
    float                    learning_rate       = 1e-4f;
    float                    weight_decay        = 0.01f;
    float                    max_grad_norm       = 1.0f;
    float                    cfg_dropout         = 0.15f;
    bool                     min_snr             = false;
    float                    min_snr_gamma       = 5.0f;
    bool                     checkpoint_backward = true;
    bool                     validate_dataset    = false;
    ACETrainSchedule         schedule            = ACE_TRAIN_SCHEDULE_COSINE;
    ACETrainPreprocessConfig preprocess;
};

static void usage(const char * program) {
    std::fprintf(stderr,
                 "acestep.cpp %s native adapter trainer\n\n"
                 "Usage: %s --models <dir> --dataset <dir> --output <dir> [options]\n\n"
                 "Required:\n"
                 "  --models <dir>             ACE-Step GGUF model directory\n"
                 "  --dataset <dir>            Basename-paired WAV and metadata text files\n"
                 "  --output <dir>             PEFT adapter output directory\n\n"
                 "Dataset check:\n"
                 "  --validate-dataset         Validate --dataset pairs without loading models\n\n"
                 "Adapter:\n"
                 "  --adapter-type <name>      lora or dora-rows; must match --resume (default: dora-rows)\n"
                 "  --profile <name>           attention or balanced (default: balanced)\n"
                 "  --rank <N>                 Base rank (default: 64)\n"
                 "  --alpha <N>                Base alpha (default: 128)\n"
                 "  --resume <dir>             Resume full native state or PEFT weights\n\n"
                 "Training:\n"
                 "  --epochs <N>               Epoch count (default: 10)\n"
                 "  --batch-size <N>            Microbatch size (default: 1)\n"
                 "  --accumulation <N>          Microbatches per optimizer step (default: 1)\n"
                 "  --learning-rate <F>         Peak AdamW learning rate (default: 0.0001)\n"
                 "  --weight-decay <F>          AdamW weight decay (default: 0.01)\n"
                 "  --max-grad-norm <F>         Global gradient clip (default: 1.0)\n"
                 "  --warmup-steps <N>          Linear warmup steps (default: 10)\n"
                 "  --schedule <name>           constant, linear, cosine, cosine-restarts\n"
                 "  --restart-count <N>         Cosine restart cycles (default: 1)\n"
                 "  --cfg-dropout <F>           Conditioning dropout (default: 0.15)\n"
                 "  --min-snr [gamma]           Enable flow Min-SNR weighting (default gamma: 5)\n"
                 "  --checkpoint-backward <B>  Recompute layers during backward (default: true)\n"
                 "  --seed <N>                  Deterministic seed (default: 42)\n\n"
                 "Conditioning:\n"
                 "  --use-genre                 Use genre instead of caption when available\n"
                 "  --tag-position <name>       prepend, append, or replace\n\n"
                 "Model selection:\n"
                 "  --dit <name>                DiT filename in the model registry\n"
                 "  --text-encoder <name>        Text encoder filename in the model registry\n"
                 "  --vae <name>                VAE filename in the model registry\n\n"
                 "Set GGML_BACKEND=CUDA0, Vulkan0, MTL0, or CPU to select a device.\n",
                 ACE_VERSION, program);
}

static bool read_integer(const char * value, int & result) {
    if (!value || !*value) {
        return false;
    }
    char *     end    = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end || parsed < 0 || parsed > 2147483647L) {
        return false;
    }
    result = (int) parsed;
    return true;
}

static bool read_seed(const char * value, uint64_t & result) {
    if (!value || !*value || *value == '-') {
        return false;
    }
    char *                   end    = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (!end || *end) {
        return false;
    }
    result = (uint64_t) parsed;
    return true;
}

static bool read_float(const char * value, float & result) {
    if (!value || !*value) {
        return false;
    }
    char *      end    = nullptr;
    const float parsed = std::strtof(value, &end);
    if (!end || *end || !std::isfinite(parsed)) {
        return false;
    }
    result = parsed;
    return true;
}

static bool read_bool(const char * value, bool & result) {
    if (!value) {
        return false;
    }
    if (!std::strcmp(value, "true") || !std::strcmp(value, "1") || !std::strcmp(value, "on")) {
        result = true;
        return true;
    }
    if (!std::strcmp(value, "false") || !std::strcmp(value, "0") || !std::strcmp(value, "off")) {
        result = false;
        return true;
    }
    return false;
}

static bool parse_command(int argc, char ** argv, ACETrainCommand & command) {
    for (int i = 1; i < argc; ++i) {
        const char * option = argv[i];
        auto         next   = [&]() -> const char * {
            return i + 1 < argc ? argv[++i] : nullptr;
        };
        if (!std::strcmp(option, "--models")) {
            command.models_dir = next();
        } else if (!std::strcmp(option, "--dataset")) {
            command.dataset_dir = next();
        } else if (!std::strcmp(option, "--output")) {
            command.output_dir = next();
        } else if (!std::strcmp(option, "--validate-dataset")) {
            command.validate_dataset = true;
        } else if (!std::strcmp(option, "--resume")) {
            command.resume_dir = next();
        } else if (!std::strcmp(option, "--dit")) {
            command.dit_name = next();
        } else if (!std::strcmp(option, "--text-encoder")) {
            command.text_name = next();
        } else if (!std::strcmp(option, "--vae")) {
            command.vae_name = next();
        } else if (!std::strcmp(option, "--adapter-type")) {
            const char * value = next();
            if (!value) {
                return false;
            }
            command.adapter_type = value;
        } else if (!std::strcmp(option, "--profile")) {
            const char * value = next();
            if (!value) {
                return false;
            }
            command.module_profile = value;
        } else if (!std::strcmp(option, "--rank")) {
            if (!read_integer(next(), command.rank)) {
                return false;
            }
        } else if (!std::strcmp(option, "--alpha")) {
            if (!read_integer(next(), command.alpha)) {
                return false;
            }
        } else if (!std::strcmp(option, "--epochs")) {
            if (!read_integer(next(), command.epochs)) {
                return false;
            }
        } else if (!std::strcmp(option, "--batch-size")) {
            if (!read_integer(next(), command.batch_size)) {
                return false;
            }
        } else if (!std::strcmp(option, "--accumulation")) {
            if (!read_integer(next(), command.accumulation)) {
                return false;
            }
        } else if (!std::strcmp(option, "--warmup-steps")) {
            if (!read_integer(next(), command.warmup_steps)) {
                return false;
            }
        } else if (!std::strcmp(option, "--restart-count")) {
            if (!read_integer(next(), command.restart_count)) {
                return false;
            }
        } else if (!std::strcmp(option, "--seed")) {
            if (!read_seed(next(), command.seed)) {
                return false;
            }
        } else if (!std::strcmp(option, "--learning-rate")) {
            if (!read_float(next(), command.learning_rate)) {
                return false;
            }
        } else if (!std::strcmp(option, "--weight-decay")) {
            if (!read_float(next(), command.weight_decay)) {
                return false;
            }
        } else if (!std::strcmp(option, "--max-grad-norm")) {
            if (!read_float(next(), command.max_grad_norm)) {
                return false;
            }
        } else if (!std::strcmp(option, "--cfg-dropout")) {
            if (!read_float(next(), command.cfg_dropout)) {
                return false;
            }
        } else if (!std::strcmp(option, "--min-snr")) {
            command.min_snr = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                if (!read_float(next(), command.min_snr_gamma)) {
                    return false;
                }
            }
        } else if (!std::strcmp(option, "--checkpoint-backward")) {
            if (!read_bool(next(), command.checkpoint_backward)) {
                return false;
            }
        } else if (!std::strcmp(option, "--schedule")) {
            const char * value = next();
            if (!value) {
                return false;
            }
            if (!std::strcmp(value, "constant")) {
                command.schedule = ACE_TRAIN_SCHEDULE_CONSTANT;
            } else if (!std::strcmp(value, "linear")) {
                command.schedule = ACE_TRAIN_SCHEDULE_LINEAR;
            } else if (!std::strcmp(value, "cosine")) {
                command.schedule = ACE_TRAIN_SCHEDULE_COSINE;
            } else if (!std::strcmp(value, "cosine-restarts")) {
                command.schedule = ACE_TRAIN_SCHEDULE_COSINE_RESTARTS;
            } else {
                return false;
            }
        } else if (!std::strcmp(option, "--use-genre")) {
            command.preprocess.use_genre = true;
        } else if (!std::strcmp(option, "--tag-position")) {
            const char * value = next();
            if (!value) {
                return false;
            }
            if (!std::strcmp(value, "prepend")) {
                command.preprocess.tag_position = ACE_TRAIN_TAG_PREPEND;
            } else if (!std::strcmp(value, "append")) {
                command.preprocess.tag_position = ACE_TRAIN_TAG_APPEND;
            } else if (!std::strcmp(value, "replace")) {
                command.preprocess.tag_position = ACE_TRAIN_TAG_REPLACE;
            } else {
                return false;
            }
        } else if (!std::strcmp(option, "--help") || !std::strcmp(option, "-h")) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "[Ace-Train] Unknown option: %s\n", option);
            return false;
        }
    }
    const bool required_paths = command.validate_dataset ?
                                    command.dataset_dir != nullptr :
                                    command.models_dir && command.dataset_dir && command.output_dir;
    return required_paths && command.rank > 0 && command.alpha > 0 && command.epochs > 0 && command.batch_size > 0 &&
           command.accumulation > 0 && command.restart_count > 0 && command.learning_rate > 0.0f &&
           command.weight_decay >= 0.0f && command.max_grad_norm >= 0.0f && command.cfg_dropout >= 0.0f &&
           command.cfg_dropout <= 1.0f && command.min_snr_gamma > 0.0f &&
           (command.adapter_type == "lora" || command.adapter_type == "dora-rows") &&
           (command.module_profile == "attention" || command.module_profile == "balanced");
}

static const ModelEntry * choose_model(const std::vector<ModelEntry> & entries, const char * name) {
    if (entries.empty()) {
        return nullptr;
    }
    return name ? registry_find(entries, name) : &entries[0];
}

static bool preprocess_dataset(const ACETrainCommand &                 command,
                               const ModelEntry &                      dit_entry,
                               const ModelEntry &                      text_entry,
                               const ModelEntry &                      vae_entry,
                               std::vector<ACETrainDiffusionExample> & prepared,
                               std::vector<float> &                    silence_latents,
                               std::vector<float> &                    null_condition,
                               std::string &                           error) {
    std::vector<ACETrainingExample> sources;
    if (!ace_training_load_dataset(command.dataset_dir, sources, error)) {
        return false;
    }
    ModelStore *   store = store_create(EVICT_STRICT);
    AceSynthParams params;
    ace_synth_default_params(&params);
    params.dit_path          = dit_entry.path.c_str();
    params.text_encoder_path = text_entry.path.c_str();
    params.vae_path          = vae_entry.path.c_str();
    AceSynth * synth         = ace_synth_load(store, &params);
    if (!synth) {
        store_free(store);
        error = "could not initialize ACE-Step preprocessing models";
        return false;
    }

    std::fprintf(stderr, "[Ace-Train] Preprocessing %zu examples in model phases\n", sources.size());
    if (!ace_prepare_training_dataset(synth, sources, command.seed, command.preprocess, prepared, error)) {
        ace_synth_free(synth);
        store_free(store);
        return false;
    }
    const DiTMeta * metadata = store_dit_meta(store, dit_entry.path.c_str());
    if (!metadata || !ace_training_null_condition(synth, null_condition, error)) {
        ace_synth_free(synth);
        store_free(store);
        return false;
    }
    silence_latents = metadata->silence_full;
    ace_synth_free(synth);
    store_free(store);
    return true;
}

static bool train_adapter(const ACETrainCommand &                       command,
                          const ModelEntry &                            dit_entry,
                          const std::vector<ACETrainDiffusionExample> & prepared,
                          const std::vector<float> &                    silence_latents,
                          const std::vector<float> &                    null_condition,
                          std::string &                                 error) {
    std::string base_model_fingerprint;
    if (!ace_file_fingerprint(dit_entry.path, base_model_fingerprint, error)) {
        return false;
    }
    DiTGGML model = {};
    if (!dit_ggml_load(&model, dit_entry.path.c_str(), nullptr, 1.0f, false)) {
        error = "could not load the unfused DiT training model";
        return false;
    }

    std::vector<ACETrainAdapterTarget> targets;
    if (!ace_train_adapter_targets(model, command.module_profile, command.rank, command.alpha, targets, error)) {
        dit_ggml_free(&model);
        return false;
    }
    ACETrainAdapterState     state;
    ACETrainAdapterOptimizer optimizer;
    int                      completed_epochs = 0;
    if (command.resume_dir) {
        ACETrainCheckpointKind checkpoint_kind;
        if (!ace_train_checkpoint_kind(command.resume_dir, checkpoint_kind, error)) {
            dit_ggml_free(&model);
            return false;
        }
        const bool loaded =
            checkpoint_kind == ACE_TRAIN_CHECKPOINT_FULL ?
                ace_load_train_checkpoint(command.resume_dir, targets, state, optimizer, completed_epochs,
                                          base_model_fingerprint, command.adapter_type, error) :
                ace_load_train_adapter_checkpoint(command.resume_dir, targets, state, error, command.adapter_type);
        if (!loaded) {
            dit_ggml_free(&model);
            return false;
        }
    } else if (!ace_init_train_adapter_state(targets, command.adapter_type, command.seed, state, error)) {
        dit_ggml_free(&model);
        return false;
    }

    const int batches_per_epoch = ((int) prepared.size() + command.batch_size - 1) / command.batch_size;
    const int updates_per_epoch = (batches_per_epoch + command.accumulation - 1) / command.accumulation;
    const int total_steps       = updates_per_epoch * command.epochs;
    if (completed_epochs > command.epochs) {
        error = "resume checkpoint has completed more epochs than requested";
        dit_ggml_free(&model);
        return false;
    }
    ACETrainAdapterGradientAccumulator accumulator;
    ACETrainAdamWConfig                optimizer_config;
    optimizer_config.weight_decay      = command.weight_decay;
    optimizer_config.max_gradient_norm = command.max_grad_norm;
    ACETrainDiffusionConfig diffusion_config;
    diffusion_config.cfg_dropout   = command.cfg_dropout;
    diffusion_config.min_snr       = command.min_snr;
    diffusion_config.min_snr_gamma = command.min_snr_gamma;

    std::vector<size_t> order(prepared.size());
    std::iota(order.begin(), order.end(), 0);
    for (int epoch = completed_epochs; epoch < command.epochs; ++epoch) {
        std::mt19937_64 shuffle(command.seed + (uint64_t) epoch);
        std::shuffle(order.begin(), order.end(), shuffle);
        double epoch_loss    = 0.0;
        int    epoch_batches = 0;
        for (int batch_index = 0; batch_index < batches_per_epoch; ++batch_index) {
            const int                             first = batch_index * command.batch_size;
            const int                             last  = std::min(first + command.batch_size, (int) prepared.size());
            std::vector<ACETrainDiffusionExample> sources;
            sources.reserve((size_t) (last - first));
            for (int i = first; i < last; ++i) {
                sources.push_back(prepared[order[(size_t) i]]);
            }

            std::vector<ACETrainDiffusionExample> collated;
            int                                   temporal_length         = 0;
            int                                   encoder_sequence_length = 0;
            if (!ace_collate_training_examples(model, sources, silence_latents, null_condition, collated,
                                               temporal_length, encoder_sequence_length, error)) {
                dit_ggml_free(&model);
                return false;
            }
            ACETrainDiffusionBatch batch;
            const uint64_t         batch_seed = command.seed + (uint64_t) epoch * 1000000ULL + (uint64_t) batch_index;
            if (!ace_prepare_train_diffusion_batch(model, collated, temporal_length, encoder_sequence_length,
                                                   null_condition, batch_seed, diffusion_config, batch, error)) {
                dit_ggml_free(&model);
                return false;
            }
            ACETrainDiTGraph            graph;
            ACETrainDiTCheckpoint       checkpoint;
            ACETrainAdapterGraphState * graph_adapters = nullptr;
            if (command.checkpoint_backward) {
                if (!ace_build_train_dit_checkpoint(model, state, temporal_length, encoder_sequence_length,
                                                    (int) collated.size(), checkpoint, error)) {
                    dit_ggml_free(&model);
                    return false;
                }
                graph_adapters = &checkpoint.adapters;
            } else {
                if (!ace_build_train_dit_graph(model, state, temporal_length, encoder_sequence_length,
                                               (int) collated.size(), graph, error)) {
                    dit_ggml_free(&model);
                    return false;
                }
                graph_adapters = &graph.adapters;
            }
            float      loss = 0.0f;
            const bool gradients_ok =
                command.checkpoint_backward ?
                    ace_compute_train_adapter_gradients_checkpointed(checkpoint, batch, accumulator,
                                                                     (int) collated.size(), loss, error) :
                    (ace_compute_train_adapter_gradients(graph, batch, loss, error) &&
                     ace_train_adapter_accumulate_gradients(graph, accumulator, (int) collated.size(), error));
            if (!gradients_ok) {
                ace_free_train_dit_checkpoint(checkpoint);
                ace_free_train_dit_graph(graph);
                dit_ggml_free(&model);
                return false;
            }
            epoch_loss += loss;
            epoch_batches += 1;

            const bool update =
                accumulator.microbatch_count == command.accumulation || batch_index + 1 == batches_per_epoch;
            if (update) {
                optimizer_config.learning_rate =
                    ace_train_learning_rate(command.learning_rate, optimizer.step, total_steps, command.warmup_steps,
                                            command.schedule, command.restart_count);
                if (!ace_train_adapter_adamw_step_accumulated(*graph_adapters, state, optimizer, optimizer_config,
                                                              accumulator, error)) {
                    ace_free_train_dit_checkpoint(checkpoint);
                    ace_free_train_dit_graph(graph);
                    dit_ggml_free(&model);
                    return false;
                }
                std::fprintf(stderr, "[Ace-Train] epoch=%d/%d step=%d/%d batch=%d/%d loss=%.6f lr=%.8g\n", epoch + 1,
                             command.epochs, optimizer.step, total_steps, batch_index + 1, batches_per_epoch, loss,
                             optimizer_config.learning_rate);
            }
            ace_free_train_dit_checkpoint(checkpoint);
            ace_free_train_dit_graph(graph);
        }
        const std::string checkpoint =
            std::string(command.output_dir) + "/checkpoint-epoch-" + std::to_string(epoch + 1);
        if (!ace_save_train_checkpoint(checkpoint, state, optimizer, epoch + 1, base_model_fingerprint, error)) {
            dit_ggml_free(&model);
            return false;
        }
        std::fprintf(stderr, "[Ace-Train] epoch=%d mean_loss=%.6f checkpoint=%s\n", epoch + 1,
                     epoch_loss / (double) epoch_batches, checkpoint.c_str());
    }

    const bool saved =
        ace_save_train_checkpoint(command.output_dir, state, optimizer, command.epochs, base_model_fingerprint, error);
    dit_ggml_free(&model);
    return saved;
}

int main(int argc, char ** argv) {
    ACETrainCommand command;
    if (!parse_command(argc, argv, command)) {
        usage(argv[0]);
        return 1;
    }

    if (command.validate_dataset) {
        std::vector<ACETrainingExample> examples;
        std::string                     error;
        if (!ace_training_load_dataset(command.dataset_dir, examples, error)) {
            std::fprintf(stderr, "[Ace-Train] ERROR: %s\n", error.c_str());
            return 1;
        }
        double duration = 0.0;
        for (const ACETrainingExample & example : examples) {
            int samples = 0;
            if (!ace_validate_training_audio(example.audio_path.string(), samples, error)) {
                std::fprintf(stderr, "[Ace-Train] ERROR: %s\n", error.c_str());
                return 1;
            }
            duration += (double) samples / 48000.0;
        }
        std::fprintf(stderr, "[Ace-Train] Dataset valid: %zu paired examples, %.2f minutes decoded at 48 kHz\n",
                     examples.size(), duration / 60.0);
        return 0;
    }

    ModelRegistry registry;
    if (!registry_scan(&registry, command.models_dir)) {
        std::fprintf(stderr, "[Ace-Train] ERROR: could not scan models directory\n");
        return 1;
    }
    const ModelEntry * dit          = choose_model(registry.dit, command.dit_name);
    const ModelEntry * text_encoder = choose_model(registry.text_enc, command.text_name);
    const ModelEntry * vae          = choose_model(registry.vae, command.vae_name);
    if (!dit || !text_encoder || !vae) {
        std::fprintf(stderr, "[Ace-Train] ERROR: selected DiT, text encoder, or VAE model was not found\n");
        return 1;
    }

    std::string                           error;
    std::vector<ACETrainDiffusionExample> prepared;
    std::vector<float>                    silence_latents;
    std::vector<float>                    null_condition;
    if (!preprocess_dataset(command, *dit, *text_encoder, *vae, prepared, silence_latents, null_condition, error) ||
        !train_adapter(command, *dit, prepared, silence_latents, null_condition, error)) {
        std::fprintf(stderr, "[Ace-Train] ERROR: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "[Ace-Train] Complete: %s\n", command.output_dir);
    return 0;
}
