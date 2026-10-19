// ABOUTME: Runs native ACE-Step LoRA and DoRA-row training from paired WAV and metadata files.
// ABOUTME: Reuses GGUF preprocessing, batches variable lengths, accumulates gradients, and saves PEFT adapters.

#include "cache-build.h"
#include "dit.h"
#include "model-registry.h"
#include "model-store.h"
#include "pipeline-synth.h"
#include "pipeline-train.h"
#include "train-checkpoint.h"
#include "train-cli-control.h"
#include "train-diffusion.h"
#include "train-history.h"
#include "train-preprocess-cache.h"
#include "train-prompt.h"
#include "train-resume.h"
#include "train-window.h"
#include "version.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#ifdef __APPLE__
#    include <crt_externs.h>
#endif

struct ACETrainCommand {
    const char *             models_dir             = nullptr;
    const char *             dataset_dir            = nullptr;
    const char *             output_dir             = nullptr;
    const char *             resume_dir             = nullptr;
    const char *             dit_name               = nullptr;
    const char *             text_name              = nullptr;
    const char *             vae_name               = nullptr;
    std::string              adapter_type           = "dora-rows";
    std::string              module_profile         = "balanced";
    int                      rank                   = 64;
    int                      alpha                  = 128;
    int                      epochs                 = 10;
    int                      batch_size             = 1;
    int                      accumulation           = 1;
    int                      warmup_steps           = 10;
    int                      restart_count          = 1;
    int                      checkpoint_every       = 1;
    int                      checkpoint_every_steps = 0;
    uint64_t                 seed                   = 42;
    float                    learning_rate          = 1e-4f;
    float                    weight_decay           = 0.01f;
    float                    max_grad_norm          = 1.0f;
    float                    cfg_dropout            = 0.15f;
    float                    timestep_mean          = -0.4f;
    float                    timestep_std           = 1.0f;
    bool                     min_snr                = false;
    float                    min_snr_gamma          = 5.0f;
    bool                     checkpoint_backward    = true;
    bool                     validate_dataset       = false;
    bool                     allow_legacy_resume    = false;
    int                      max_steps              = 0;
    int                      examples_per_epoch     = 0;
    int                      genre_ratio            = 0;
    ACETrainWindowConfig     window;
    ACETrainSchedule         schedule = ACE_TRAIN_SCHEDULE_COSINE;
    ACETrainPreprocessConfig preprocess;
    bool                     preprocess_cache      = true;
    const char *             preprocess_cache_dir  = nullptr;
    int                      cache_memory_mb       = 256;
    int                      preprocess_batch_size = 1;
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
                 "  --allow-legacy-resume       Accept old state without a recorded plan (explicit opt-in)\n"
                 "Training:\n"
                 "  --epochs <N>               Epoch count (default: 10)\n"
                 "  --steps <N>                Total optimizer-step stop target within the epoch schedule budget\n"
                 "  --batch-size <N>            Microbatch size (default: 1)\n"
                 "  --accumulation <N>          Microbatches per optimizer step (default: 1)\n"
                 "  --learning-rate <F>         Peak AdamW learning rate (default: 0.0001)\n"
                 "  --weight-decay <F>          AdamW weight decay (default: 0.01)\n"
                 "  --max-grad-norm <F>         Global gradient clip (default: 1.0)\n"
                 "  --warmup-steps <N>          Linear warmup steps (default: 10)\n"
                 "  --schedule <name>           constant, linear, cosine, cosine-restarts\n"
                 "  --restart-count <N>         Cosine restart cycles (default: 1)\n"
                 "  --cfg-dropout <F>           Conditioning dropout (default: 0.15)\n"
                 "  --timestep-mu <F>           Logit-normal timestep mean (default: -0.4)\n"
                 "  --timestep-sigma <F>        Logit-normal timestep stddev (default: 1.0)\n"
                 "  --min-snr [gamma]           Enable flow Min-SNR weighting (default gamma: 5)\n"
                 "  --checkpoint-backward <B>  Recompute layers during backward (default: true)\n"
                 "  --checkpoint-every <N>     Save intermediate state every N epochs; 0 disables (default: 1)\n"
                 "  --checkpoint-every-steps N Save state every N optimizer updates; 0 disables (default: 0)\n"
                 "  --window-min-seconds <F>   Minimum rotating audio crop; 0 disables (default: 0)\n"
                 "  --window-max-seconds <F>   Maximum rotating audio crop; set with minimum\n"
                 "  --examples-per-epoch <N>   Deterministic epoch subset; 0 uses all (default: 0)\n"
                 "  --seed <N>                  Deterministic seed (default: 42)\n\n"
                 "Conditioning:\n"
                 "  --preprocess-cache <B>     Reuse disk-backed prepared tensors (default: true)\n"
                 "  --preprocess-cache-dir DIR Cache root (default: DATASET/.ace-train-cache)\n"
                 "  --cache-memory-mb N        Prepared-tensor LRU budget; 0 streams (default: 256)\n"
                 "  --preprocess-batch-size N  Maximum cold-preprocessing examples in RAM (default: 1)\n"
                 "  --use-genre                 Use genre instead of caption when available\n"
                 "  --genre-ratio <0-100>       Rotate caption/genre prompts (default: 0; excludes --use-genre)\n"
                 "  --tag-position <name>       prepend, append, or replace\n\n"
                 "Model selection:\n"
                 "  --dit <name>                DiT filename in the model registry\n"
                 "  --text-encoder <name>        Text encoder filename in the model registry\n"
                 "  --vae <name>                VAE filename in the model registry\n\n"
                 "Set GGML_BACKEND=CUDA0, Vulkan0, MTL0, or CPU to select a device.\n",
                 ACE_VERSION, program);
    std::fputs(ace_training::TrainCliControl::usage(), stderr);
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
        } else if (!std::strcmp(option, "--allow-legacy-resume")) {
            command.allow_legacy_resume = true;
        } else if (!std::strcmp(option, "--steps")) {
            if (!read_integer(next(), command.max_steps) || command.max_steps == 0) {
                return false;
            }
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
        } else if (!std::strcmp(option, "--timestep-mu")) {
            if (!read_float(next(), command.timestep_mean)) {
                return false;
            }
        } else if (!std::strcmp(option, "--timestep-sigma")) {
            if (!read_float(next(), command.timestep_std)) {
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
        } else if (!std::strcmp(option, "--checkpoint-every")) {
            if (!read_integer(next(), command.checkpoint_every)) {
                return false;
            }
        } else if (!std::strcmp(option, "--checkpoint-every-steps")) {
            if (!read_integer(next(), command.checkpoint_every_steps)) {
                return false;
            }
        } else if (!std::strcmp(option, "--window-min-seconds")) {
            if (!read_float(next(), command.window.min_seconds)) {
                return false;
            }
        } else if (!std::strcmp(option, "--window-max-seconds")) {
            if (!read_float(next(), command.window.max_seconds)) {
                return false;
            }
        } else if (!std::strcmp(option, "--examples-per-epoch")) {
            if (!read_integer(next(), command.examples_per_epoch)) {
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
        } else if (!std::strcmp(option, "--genre-ratio")) {
            if (!read_integer(next(), command.genre_ratio) || command.genre_ratio > 100) {
                return false;
            }
            command.preprocess.genre_variants = command.genre_ratio > 0;
        } else if (!std::strcmp(option, "--use-genre")) {
            command.preprocess.use_genre = true;
        } else if (!std::strcmp(option, "--preprocess-cache")) {
            if (!read_bool(next(), command.preprocess_cache)) {
                return false;
            }
        } else if (!std::strcmp(option, "--preprocess-cache-dir")) {
            command.preprocess_cache_dir = next();
            if (!command.preprocess_cache_dir || !*command.preprocess_cache_dir) {
                return false;
            }
        } else if (!std::strcmp(option, "--cache-memory-mb")) {
            if (!read_integer(next(), command.cache_memory_mb)) {
                return false;
            }
        } else if (!std::strcmp(option, "--preprocess-batch-size")) {
            if (!read_integer(next(), command.preprocess_batch_size) || command.preprocess_batch_size == 0) {
                return false;
            }
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
           command.cfg_dropout <= 1.0f && command.timestep_std >= 0.0f && command.min_snr_gamma > 0.0f &&
           ace_train_window_config_valid(command.window) &&
           !(command.preprocess.use_genre && command.genre_ratio > 0) &&
           (command.adapter_type == "lora" || command.adapter_type == "dora-rows") &&
           (command.module_profile == "attention" || command.module_profile == "balanced");
}

static const ModelEntry * choose_model(const std::vector<ModelEntry> & entries, const char * name) {
    if (entries.empty()) {
        return nullptr;
    }
    // Never choose a different family/encoding just because it sorts first.
    return name ? registry_find(entries, name) : entries.size() == 1 ? &entries[0] : nullptr;
}

// SA3 fingerprints every input/setting that changes the update trajectory.
// Content identities allow moving an unchanged dataset/model installation.
static bool make_resume_plan(const ACETrainCommand &                 command,
                             const ModelEntry &                      dit,
                             const ModelEntry &                      text,
                             const ModelEntry &                      vae,
                             const std::vector<ACETrainingExample> & sources,
                             ACETrainResumePlan &                    plan,
                             std::string &                           error) {
    if (sources.empty() || sources.size() > INT_MAX) {
        error = "invalid training dataset size";
        return false;
    }
    plan.dataset_size = int(sources.size());
    plan.epochs       = command.epochs;
    plan.accumulation = command.accumulation;
    const int count =
        command.examples_per_epoch > 0 ? std::min(command.examples_per_epoch, plan.dataset_size) : plan.dataset_size;
    plan.batches_per_epoch = int((int64_t(count) + command.batch_size - 1) / command.batch_size);
    plan.updates_per_epoch = int((int64_t(plan.batches_per_epoch) + command.accumulation - 1) / command.accumulation);
    const int64_t total    = int64_t(plan.updates_per_epoch) * command.epochs;
    if (total > INT_MAX || (command.max_steps > 0 && command.max_steps > total)) {
        error = "training step budget exceeds the epoch plan or supported integer range";
        return false;
    }
    plan.total_steps     = int(total);
    yyjson_mut_doc * doc = yyjson_mut_doc_new(nullptr);
    if (!doc) {
        throw std::bad_alloc();
    }
    auto * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_int(doc, root, "schema_version", 1);
    // v2 fixes CUDA cross-attention RMSNorm backward on permuted views.
    // Older optimizer state must not silently resume under changed gradients.
    yyjson_mut_obj_add_str(doc, root, "algorithm", "ace-native-epoch-seeds-v2");
    yyjson_mut_obj_add_str(doc, root, "preprocessing", ACE_TRAIN_PREPROCESS_ALGORITHM);
    // CUDA arithmetic overrides materially change the optimizer trajectory.
    // Omit default policies to retain compatibility with prior native plans.
    const char * tf32 = std::getenv("GGML_CUDA_TF32");
    if (tf32 && tf32[0] == '0') {
        yyjson_mut_obj_add_bool(doc, root, "cuda_tf32", false);
    }
    if (const char * compute = std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE")) {
        std::string policy(compute);
        std::transform(policy.begin(), policy.end(), policy.begin(),
                       [](unsigned char c) { return char(std::tolower(c)); });
        if (policy == "f32") {
            policy = "fp32";
        } else if (policy == "f16") {
            policy = "fp16";
        }
        if (policy == "fp32" || policy == "fp16" || policy == "bf16") {
            yyjson_mut_obj_add_strcpy(doc, root, "cuda_cublas_compute_type", policy.c_str());
        }
    }
    auto * settings = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "settings", settings);
    auto integer = [&](const char * name, int value) {
        yyjson_mut_obj_add_int(doc, settings, name, value);
    };
    auto real = [&](const char * name, float value) {
        yyjson_mut_obj_add_real(doc, settings, name, value);
    };
    yyjson_mut_obj_add_strcpy(doc, settings, "adapter_type", command.adapter_type.c_str());
    yyjson_mut_obj_add_strcpy(doc, settings, "profile", command.module_profile.c_str());
    integer("rank", command.rank);
    integer("alpha", command.alpha);
    integer("epochs", command.epochs);
    integer("batch_size", command.batch_size);
    integer("accumulation", command.accumulation);
    integer("warmup_steps", command.warmup_steps);
    integer("schedule", int(command.schedule));
    integer("restart_count", command.restart_count);
    integer("examples_per_epoch", command.examples_per_epoch);
    integer("batches_per_epoch", plan.batches_per_epoch);
    integer("total_steps", plan.total_steps);
    yyjson_mut_obj_add_uint(doc, settings, "seed", command.seed);
    real("learning_rate", command.learning_rate);
    real("weight_decay", command.weight_decay);
    real("max_grad_norm", command.max_grad_norm);
    real("cfg_dropout", command.cfg_dropout);
    real("timestep_mean", command.timestep_mean);
    real("timestep_std", command.timestep_std);
    real("min_snr_gamma", command.min_snr_gamma);
    real("window_min_seconds", command.window.min_seconds);
    real("window_max_seconds", command.window.max_seconds);
    real("adam_beta1", 0.9f);
    real("adam_beta2", 0.999f);
    real("adam_epsilon", 1e-8f);
    yyjson_mut_obj_add_bool(doc, settings, "min_snr", command.min_snr);
    yyjson_mut_obj_add_bool(doc, settings, "checkpoint_backward", command.checkpoint_backward);
    yyjson_mut_obj_add_bool(doc, settings, "use_genre", command.preprocess.use_genre);
    integer("tag_position", int(command.preprocess.tag_position));
    // Preserve existing caption-only / global-genre plan compatibility.
    if (command.genre_ratio > 0) {
        integer("genre_ratio", command.genre_ratio);
        yyjson_mut_obj_add_str(doc, settings, "prompt_sampling", ACE_TRAIN_PROMPT_ALGORITHM);
    }
    auto * models = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "models", models);
    auto fingerprint = [&](yyjson_mut_val * obj, const char * name, const std::filesystem::path & path) {
        std::string identity;
        if (!ace_file_fingerprint(path, identity, error)) {
            return false;
        }
        yyjson_mut_obj_add_strcpy(doc, obj, name, identity.c_str());
        if (obj == models && std::strcmp(name, "dit") == 0) {
            plan.base_fingerprint = identity;
        }
        return true;
    };
    bool ok = fingerprint(models, "dit", std::filesystem::u8path(dit.path)) &&
              fingerprint(models, "text_encoder", std::filesystem::u8path(text.path)) &&
              fingerprint(models, "vae", std::filesystem::u8path(vae.path));
    auto * dataset = yyjson_mut_arr(doc);
    yyjson_mut_obj_add_val(doc, root, "dataset", dataset);
    for (const auto & source : sources) {
        if (!ok) {
            break;
        }
        auto * item = yyjson_mut_obj(doc);
        yyjson_mut_arr_add_val(dataset, item);
        yyjson_mut_obj_add_strcpy(doc, item, "id", source.name.c_str());
        ok = fingerprint(item, "audio", source.audio_path) && fingerprint(item, "metadata", source.metadata_path);
    }
    char * json = ok ? yyjson_mut_write(doc, 0, nullptr) : nullptr;
    yyjson_mut_doc_free(doc);
    if (ok && !json) {
        error = "cannot serialize training plan";
        ok    = false;
    }
    if (json) {
        plan.json = json;
        free(json);
    }
    return ok;
}

struct ACETrainCancel {
    ace_training::TrainCliControl * control;
    bool                            cancelled = false;
};

static bool train_cancel(void * opaque) {
    auto & request    = *static_cast<ACETrainCancel *>(opaque);
    request.cancelled = request.cancelled || request.control->should_cancel();
    return request.cancelled;
}

static bool prepare_cache_identities(const ACETrainCommand &    command,
                                     const ACETrainResumePlan & plan,
                                     const AceSynthParams &     params,
                                     ACETrainPreparedDataset &  prepared,
                                     std::string &              error) {
    ggml_backend_load_all();
    auto device = ace_runtime_device(std::getenv("GGML_BACKEND"));
    if (!device) {
        error = "requested preprocessing device is unavailable";
        return false;
    }
    std::string backend = std::string(ggml_backend_dev_name(device)) + ":" + ggml_backend_dev_description(device);
    auto        reg     = ggml_backend_dev_backend_reg(device);
    auto features = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (features) {
        for (auto feature = features(reg); feature && feature->name; ++feature) {
            backend += std::string(";") + feature->name + "=" + (feature->value ? feature->value : "");
        }
    }
    std::vector<std::string> environment;
#ifdef _WIN32
    char ** variables = _environ;
#elif defined(__APPLE__)
    char ** variables = *_NSGetEnviron();
#else
    extern char ** environ;
    char **        variables = environ;
#endif
    for (char ** item = variables; item && *item; ++item) {
        if (std::strncmp(*item, "GGML_", 5) == 0) {
            environment.emplace_back(*item);
        }
    }
    std::sort(environment.begin(), environment.end());
    yyjson_doc * input = yyjson_read(plan.json.data(), plan.json.size(), 0);
    auto *       root  = input ? yyjson_doc_get_root(input) : nullptr;
    if (!root) {
        yyjson_doc_free(input);
        error = "invalid plan for preprocessing cache";
        return false;
    }
    auto * settings = yyjson_obj_get(root, "settings");
    auto * dataset  = yyjson_obj_get(root, "dataset");
    prepared.identities.clear();
    prepared.genre_variants = command.preprocess.genre_variants;
    for (size_t i = 0; i < yyjson_arr_size(dataset); ++i) {
        yyjson_mut_doc * doc = yyjson_mut_doc_new(nullptr);
        auto *           key = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, key);
        yyjson_mut_obj_add_int(doc, key, "schema_version", 1);
        yyjson_mut_obj_add_str(doc, key, "algorithm", ACE_TRAIN_PREPROCESS_ALGORITHM);
        yyjson_mut_obj_add_str(doc, key, "application", ACE_VERSION);
        yyjson_mut_obj_add_str(doc, key, "release", ACE_RELEASE_VERSION);
        yyjson_mut_obj_add_str(doc, key, "ggml", ACE_GGML_REVISION);
        yyjson_mut_obj_add_str(doc, key, "build", ACE_CACHE_CONFIG_ID ":" ACE_CACHE_BUILD_CONFIGURATION);
        yyjson_mut_obj_add_strcpy(doc, key, "backend", backend.c_str());
        yyjson_mut_obj_add_int(doc, key, "threads", backend_cpu_n_threads());
        uint32_t endian = 1;
        yyjson_mut_obj_add_bool(doc, key, "little_endian", *(unsigned char *) &endian == 1);
        auto * env = yyjson_mut_arr(doc);
        for (const auto & value : environment) {
            yyjson_mut_arr_add_strcpy(doc, env, value.c_str());
        }
        yyjson_mut_obj_add_val(doc, key, "environment", env);
        yyjson_mut_obj_add_val(doc, key, "models", yyjson_val_mut_copy(doc, yyjson_obj_get(root, "models")));
        yyjson_mut_obj_add_val(doc, key, "source", yyjson_val_mut_copy(doc, yyjson_arr_get(dataset, i)));
        // The original VAE posterior uses seed + dataset index, independent of chunking/cache misses.
        yyjson_mut_obj_add_uint(doc, key, "posterior_seed", command.seed + uint64_t(i));
        for (const char * name : { "tag_position", "use_genre" }) {
            yyjson_mut_obj_add_val(doc, key, name, yyjson_val_mut_copy(doc, yyjson_obj_get(settings, name)));
        }
        if (prepared.genre_variants) {
            yyjson_mut_obj_add_str(doc, key, "prompt_variants", "caption-and-genre-v1");
        }
        yyjson_mut_obj_add_int(doc, key, "vae_chunk", params.vae_chunk);
        yyjson_mut_obj_add_int(doc, key, "vae_overlap", params.vae_overlap);
        yyjson_mut_obj_add_bool(doc, key, "flash_attention", params.use_fa);
        yyjson_mut_obj_add_bool(doc, key, "clamp_fp16", params.clamp_fp16);
        char * text = yyjson_mut_write(doc, 0, nullptr);
        yyjson_mut_doc_free(doc);
        if (!text) {
            yyjson_doc_free(input);
            error = "cannot serialize cache identity";
            return false;
        }
        prepared.identities.emplace_back(text);
        free(text);
    }
    yyjson_doc_free(input);
    prepared.directory    = command.preprocess_cache_dir ?
                                std::filesystem::u8path(command.preprocess_cache_dir) :
                                std::filesystem::u8path(command.dataset_dir) / ".ace-train-cache";
    prepared.memory_limit = size_t(command.cache_memory_mb) * 1024 * 1024;
    return true;
}

static bool preprocess_dataset(const ACETrainCommand &                 command,
                               const ModelEntry &                      dit_entry,
                               const ModelEntry &                      text_entry,
                               const ModelEntry &                      vae_entry,
                               ACETrainPreparedDataset &               prepared,
                               std::vector<std::string> &              example_names,
                               std::vector<float> &                    silence_latents,
                               std::vector<float> &                    null_condition,
                               const std::vector<ACETrainingExample> & sources,
                               const ACETrainResumePlan &              plan,
                               ACETrainCancel &                        cancellation,
                               std::string &                           error) {
    example_names.clear();
    example_names.reserve(sources.size());
    for (const ACETrainingExample & source : sources) {
        example_names.push_back(source.name);
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

    const DiTMeta * metadata = store_dit_meta(store, dit_entry.path.c_str());
    if (!metadata || !ace_training_null_condition(synth, null_condition, error)) {
        ace_synth_free(synth);
        store_free(store);
        return false;
    }
    if (command.batch_size > 1 && metadata->cfg.n_kv_heads != metadata->cfg.n_heads) {
        ace_synth_free(synth);
        store_free(store);
        error = "grouped-query attention training requires --batch-size 1; use --accumulation";
        return false;
    }
    silence_latents           = metadata->silence_full;
    prepared.output_channels  = metadata->cfg.out_channels;
    prepared.context_channels = metadata->cfg.in_channels - metadata->cfg.out_channels;
    prepared.patch_size       = metadata->cfg.patch_size;
    prepared.hidden_size      = int(null_condition.size());
    prepared.null_condition   = null_condition;
    prepared.max_latents      = silence_latents.size();
    prepared.genre_variants   = command.preprocess.genre_variants;
    prepared.genre_eligible.clear();
    for (const auto & source : sources) {
        prepared.genre_eligible.push_back(!source.metadata.genre.empty());
    }
    std::fprintf(stderr, "[Ace-Train] Preprocessing %zu examples in model phases\n", sources.size());
    bool ok = true;
    if (!command.preprocess_cache) {
        ok = ace_prepare_training_dataset(synth, sources, command.seed, command.preprocess, prepared.memory, error,
                                          train_cancel, &cancellation);
    } else if (!prepare_cache_identities(command, plan, params, prepared, error)) {
        ok = false;
    } else {
        const size_t chunk_size = size_t(command.preprocess_batch_size);
        for (size_t first = 0; first < sources.size() && ok; first += chunk_size) {
            if (train_cancel(&cancellation)) {
                break;
            }
            const size_t last     = std::min(sources.size(), first + chunk_size);
            bool         complete = true;
            for (size_t i = first; i < last; ++i) {
                ACETrainDiffusionExample example;
                std::string              miss;
                if (prepared.read_raw(i, example, miss)) {
                    prepared.observe(example);
                } else {
                    complete = false;
                    if (std::filesystem::exists(prepared.path(i))) {
                        ++prepared.rebuilt;
                    }
                }
            }
            if (complete) {
                prepared.reused += last - first;
                continue;
            }
            std::vector<ACETrainingExample>       chunk(sources.begin() + (std::ptrdiff_t) first,
                                                        sources.begin() + (std::ptrdiff_t) last);
            std::vector<ACETrainDiffusionExample> examples;
            ok = ace_prepare_training_dataset(synth, chunk, command.seed + uint64_t(first), command.preprocess,
                                              examples, error, train_cancel, &cancellation);
            for (size_t i = 0; ok && i < examples.size(); ++i) {
                if (train_cancel(&cancellation)) {
                    break;
                }
                // Remove chunk-local padding; get() restores original dataset-wide null padding.
                auto & example = examples[i];
                example.encoder_hidden.resize(size_t(example.real_encoder_sequence_length) *
                                              size_t(prepared.hidden_size));
                example.genre_encoder_hidden.resize(size_t(example.genre_encoder_sequence_length) *
                                                    size_t(prepared.hidden_size));
                prepared.observe(example);
                ok = prepared.write(first + i, example, error);
            }
        }
        std::fprintf(stderr,
                     "[Ace-Train] preprocess cache: %zu reused, %zu written, %zu corrupt/incompatible rebuilt; "
                     "LRU budget=%d MiB; chunk=%d -> %s\n",
                     prepared.reused, prepared.written, prepared.rebuilt, command.cache_memory_mb,
                     command.preprocess_batch_size, prepared.directory.u8string().c_str());
    }
    ace_synth_free(synth);
    store_free(store);
    return ok;
}

static bool train_adapter(const ACETrainCommand &          command,
                          const ModelEntry &               dit_entry,
                          ACETrainPreparedDataset &        prepared,
                          const std::vector<std::string> & example_names,
                          const std::vector<float> &       silence_latents,
                          const std::vector<float> &       null_condition,
                          const ACETrainResumePlan &       plan,
                          ACETrainResumeProgress           progress,
                          const std::string &              resume_path,
                          ACETrainCancel &                 cancellation,
                          std::string &                    error) {
    const std::string & base_model_fingerprint = plan.base_fingerprint;
    auto &              control                = *cancellation.control;
    if (prepared.empty() || example_names.size() != prepared.size()) {
        error = "prepared training examples and source names do not match";
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
    if (!resume_path.empty()) {
        ACETrainCheckpointKind checkpoint_kind;
        if (!ace_train_checkpoint_kind(resume_path, checkpoint_kind, error)) {
            dit_ggml_free(&model);
            return false;
        }
        const bool loaded =
            checkpoint_kind == ACE_TRAIN_CHECKPOINT_FULL ?
                ace_load_train_checkpoint(resume_path, targets, state, optimizer, completed_epochs,
                                          base_model_fingerprint, command.adapter_type, error) :
                ace_load_train_adapter_checkpoint(resume_path, targets, state, error, command.adapter_type);
        if (!loaded) {
            dit_ggml_free(&model);
            return false;
        }
        if (checkpoint_kind == ACE_TRAIN_CHECKPOINT_FULL &&
            (completed_epochs != progress.epoch || optimizer.step != int64_t(progress.epoch) * plan.updates_per_epoch +
                                                                         progress.next_batch / command.accumulation)) {
            error = "resume checkpoint counters do not match the validated continuation";
            dit_ggml_free(&model);
            return false;
        }
    } else if (!ace_init_train_adapter_state(targets, command.adapter_type, command.seed, state, error)) {
        dit_ggml_free(&model);
        return false;
    }

    const int examples_per_epoch = command.examples_per_epoch > 0 ?
                                       std::min(command.examples_per_epoch, (int) prepared.size()) :
                                       (int) prepared.size();
    const int batches_per_epoch  = plan.batches_per_epoch;
    const int total_steps        = plan.total_steps;
    const int stop_step          = command.max_steps > 0 ? command.max_steps : total_steps;
    if (optimizer.step > stop_step) {
        error = "resume step exceeds --steps (the total target optimizer step)";
        dit_ggml_free(&model);
        return false;
    }
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
    diffusion_config.timestep_mean = command.timestep_mean;
    diffusion_config.timestep_std  = command.timestep_std;
    diffusion_config.min_snr       = command.min_snr;
    diffusion_config.min_snr_gamma = command.min_snr_gamma;

    ace_training::TrainHistory history;
    if (!history.open(std::filesystem::u8path(command.output_dir), optimizer.step, error)) {
        dit_ggml_free(&model);
        return false;
    }
    double              update_loss_sum = 0;
    std::vector<size_t> order(prepared.size());
    auto                save = [&](const std::string & directory, bool final) {
        control.phase   = "saving";
        control.message = "Saving native adapter and trainer continuation.";
        control.publish();
        if (!ace_save_train_checkpoint(directory, state, optimizer, progress.epoch, base_model_fingerprint, error,
                                                      ace_train_continuation(plan, progress))) {
            return false;
        }
        std::filesystem::path generation;
        if (!adapter_checkpoint_directory(directory, generation, error)) {
            return false;
        }
        control.adapter_checkpoint = std::filesystem::absolute(generation).u8string();
        control.state_checkpoint   = control.adapter_checkpoint;
        if (final) {
            control.final_adapter = std::filesystem::absolute(std::filesystem::u8path(directory)).u8string();
        }
        control.phase = "training";
        control.publish();
        return true;
    };
    bool restore_order = progress.next_batch > 0;
    bool stop          = optimizer.step >= stop_step;
    control.step       = optimizer.step;
    control.max_steps  = stop_step;
    control.status     = "running";
    control.phase      = "training";
    control.publish();
    auto step_started = std::chrono::steady_clock::now();
    if (command.window.max_seconds > 0.0f) {
        std::fprintf(
            stderr,
            "[Ace-Train] Rotating windows %.2f-%.2fs, full-song caption/lyrics/duration conditioning retained\n",
            command.window.min_seconds, command.window.max_seconds);
    }
    if (examples_per_epoch != (int) prepared.size()) {
        std::fprintf(stderr, "[Ace-Train] Using %d/%zu examples per epoch (deterministic shuffled subset)\n",
                     examples_per_epoch, prepared.size());
    }
    for (int epoch = progress.epoch; epoch < command.epochs && !stop; ++epoch) {
        if (restore_order) {
            order         = progress.order;
            restore_order = false;
        } else {
            std::iota(order.begin(), order.end(), 0);
            std::mt19937_64 shuffle(command.seed + (uint64_t) epoch);
            std::shuffle(order.begin(), order.end(), shuffle);
        }
        progress.order            = order;
        const auto genre_selected = ace_train_genre_selection(
            prepared.genre_eligible, order, size_t(examples_per_epoch), command.genre_ratio, command.seed, epoch);
        if (command.genre_ratio > 0) {
            std::fprintf(stderr, "[Ace-Train] prompt mix epoch=%d genre=%zu/%d ratio=%d%%\n", epoch + 1,
                         size_t(std::count(genre_selected.begin(), genre_selected.end(), true)), examples_per_epoch,
                         command.genre_ratio);
        }
        double epoch_loss    = progress.epoch_loss;
        int    epoch_batches = progress.epoch_batches;
        for (int batch_index = progress.next_batch; batch_index < batches_per_epoch; ++batch_index) {
            // Finish any accumulated microbatches before observing cancellation:
            // a checkpoint never contains a half-finished optimizer update.
            if (accumulator.microbatch_count == 0 && train_cancel(&cancellation)) {
                stop = true;
                break;
            }
            const int                             first = batch_index * command.batch_size;
            const int                             last  = std::min(first + command.batch_size, examples_per_epoch);
            std::vector<ACETrainDiffusionExample> sources;
            sources.reserve((size_t) (last - first));
            for (int i = first; i < last; ++i) {
                const size_t             source_index = order[(size_t) i];
                ACETrainDiffusionExample windowed;
                ACETrainWindow           window;
                const int                output_channels  = model.cfg.out_channels;
                const int                context_channels = model.cfg.in_channels - output_channels;
                const auto               prepared_example = prepared.get(source_index, error);
                if (!prepared_example) {
                    dit_ggml_free(&model);
                    return false;
                }
                if (!ace_train_crop_example(*prepared_example, output_channels, context_channels, model.cfg.patch_size,
                                            command.window, command.seed, epoch, source_index, windowed, window,
                                            error)) {
                    dit_ggml_free(&model);
                    return false;
                }
                if (genre_selected[source_index]) {
                    if (prepared_example->genre_encoder_hidden.empty()) {
                        error = "selected genre conditioning is missing from prepared example";
                        dit_ggml_free(&model);
                        return false;
                    }
                    windowed.encoder_hidden               = prepared_example->genre_encoder_hidden;
                    windowed.real_encoder_sequence_length = prepared_example->genre_encoder_sequence_length;
                }
                windowed.genre_encoder_hidden.clear();
                windowed.genre_encoder_sequence_length = 0;
                const float full_seconds =
                    (float) prepared_example->real_temporal_length / ACE_TRAIN_LATENTS_PER_SECOND;
                const float offset_seconds = (float) window.first_latent / ACE_TRAIN_LATENTS_PER_SECOND;
                const float window_seconds = (float) windowed.real_temporal_length / ACE_TRAIN_LATENTS_PER_SECOND;
                if (command.window.max_seconds > 0.0f) {
                    std::fprintf(stderr,
                                 "[Ace-Train] window epoch=%d example=\"%s\" offset=%.2fs duration=%.2fs full=%.2fs "
                                 "conditioning=full-song\n",
                                 epoch + 1, example_names[source_index].c_str(), offset_seconds, window_seconds,
                                 full_seconds);
                }
                sources.push_back(std::move(windowed));
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
            ACETrainDiTGraph            graph          = {};
            ACETrainDiTCheckpoint       checkpoint     = {};
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
            update_loss_sum += double(loss) * collated.size();
            epoch_batches += 1;
            progress.epoch         = epoch;
            progress.next_batch    = batch_index + 1;
            progress.epoch_loss    = epoch_loss;
            progress.epoch_batches = epoch_batches;
            if (batch_index + 1 == batches_per_epoch) {
                progress.epoch         = epoch + 1;
                progress.next_batch    = 0;
                progress.epoch_loss    = 0;
                progress.epoch_batches = 0;
                progress.order.clear();
            }

            const bool update =
                accumulator.microbatch_count == command.accumulation || batch_index + 1 == batches_per_epoch;
            if (update) {
                const int    update_examples     = accumulator.example_count;
                const int    update_microbatches = accumulator.microbatch_count;
                const double update_loss         = update_loss_sum / update_examples;
                double       grad_norm           = 0;
                optimizer_config.learning_rate =
                    ace_train_learning_rate(command.learning_rate, optimizer.step, total_steps, command.warmup_steps,
                                            command.schedule, command.restart_count);
                if (!ace_train_adapter_adamw_step_accumulated(*graph_adapters, state, optimizer, optimizer_config,
                                                              accumulator, error, &grad_norm)) {
                    ace_free_train_dit_checkpoint(checkpoint);
                    ace_free_train_dit_graph(graph);
                    dit_ggml_free(&model);
                    return false;
                }
                std::fprintf(stderr, "[Ace-Train] epoch=%d/%d step=%d/%d batch=%d/%d loss=%.6f lr=%.8g\n", epoch + 1,
                             command.epochs, optimizer.step, total_steps, batch_index + 1, batches_per_epoch,
                             update_loss, optimizer_config.learning_rate);
                control.step          = optimizer.step;
                control.epoch         = epoch + 1;
                control.loss          = update_loss;
                control.learning_rate = optimizer_config.learning_rate;
                control.step_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - step_started).count();
                if (!history.append(epoch + 1, optimizer.step, optimizer_config.learning_rate, update_loss, grad_norm,
                                    control.step_seconds, update_microbatches, update_examples, error)) {
                    ace_free_train_dit_checkpoint(checkpoint);
                    ace_free_train_dit_graph(graph);
                    dit_ggml_free(&model);
                    return false;
                }
                update_loss_sum = 0;
                control.message = "Optimizer update completed.";
                control.publish();
                step_started = std::chrono::steady_clock::now();
                stop         = optimizer.step >= stop_step || train_cancel(&cancellation);
            }
            ace_free_train_dit_checkpoint(checkpoint);
            ace_free_train_dit_graph(graph);
            if (update && !stop && command.checkpoint_every_steps > 0 &&
                optimizer.step % command.checkpoint_every_steps == 0) {
                if (!save(std::string(command.output_dir) + "/checkpoint-step-" + std::to_string(optimizer.step),
                          false)) {
                    dit_ggml_free(&model);
                    return false;
                }
                step_started = std::chrono::steady_clock::now();
            }
            if (stop) {
                break;
            }
        }
        if (progress.epoch != epoch + 1) {
            break;
        }
        const bool final_epoch = epoch + 1 == command.epochs;
        const bool checkpoint_due =
            !final_epoch && command.checkpoint_every > 0 && (epoch + 1) % command.checkpoint_every == 0;
        if (checkpoint_due) {
            const std::string checkpoint =
                std::string(command.output_dir) + "/checkpoint-epoch-" + std::to_string(epoch + 1);
            std::fprintf(stderr, "[Ace-Train] saving epoch checkpoint=%s\n", checkpoint.c_str());
            if (!save(checkpoint, false)) {
                dit_ggml_free(&model);
                return false;
            }
            std::fprintf(stderr, "[Ace-Train] epoch=%d mean_loss=%.6f checkpoint=%s\n", epoch + 1,
                         epoch_loss / (double) epoch_batches, checkpoint.c_str());
        } else {
            std::fprintf(stderr, "[Ace-Train] epoch=%d mean_loss=%.6f\n", epoch + 1,
                         epoch_loss / (double) epoch_batches);
        }
    }

    if (optimizer.step == 0 && cancellation.cancelled) {
        dit_ggml_free(&model);
        return true;
    }
    if (progress.next_batch == 0) {
        progress.order.clear();
    }
    std::fprintf(stderr, "[Ace-Train] saving final checkpoint=%s\n", command.output_dir);
    const bool saved = save(command.output_dir, true);
    dit_ggml_free(&model);
    return saved;
}

static bool run_training(const ACETrainCommand & command, ACETrainCancel & cancellation, std::string & error) {
    auto & control = *cancellation.control;
    if (command.validate_dataset) {
        std::vector<ACETrainingExample> examples;
        if (!ace_training_load_dataset(command.dataset_dir, examples, error)) {
            return false;
        }
        double duration = 0.0;
        control.status  = "running";
        control.phase   = "validating";
        control.publish();
        for (const ACETrainingExample & example : examples) {
            if (train_cancel(&cancellation)) {
                return true;
            }
            int samples = 0;
            if (!ace_validate_training_audio(example.audio_path.string(), samples, error)) {
                return false;
            }
            duration += (double) samples / 48000.0;
        }
        std::fprintf(stderr, "[Ace-Train] Dataset valid: %zu paired examples, %.2f minutes decoded at 48 kHz\n",
                     examples.size(), duration / 60.0);
        return true;
    }

    ModelRegistry registry;
    if (!registry_scan(&registry, command.models_dir)) {
        error = "could not scan models directory";
        return false;
    }
    const ModelEntry * dit          = choose_model(registry.dit, command.dit_name);
    const ModelEntry * text_encoder = choose_model(registry.text_enc, command.text_name);
    const ModelEntry * vae          = choose_model(registry.vae, command.vae_name);
    if (!dit || !text_encoder || !vae) {
        error =
            "model selection is missing or ambiguous; use --dit, --text-encoder, and --vae with exact filenames when "
            "multiple variants are installed";
        return false;
    }

    control.status  = "running";
    control.phase   = "validating";
    control.message = "Validating model, dataset and training-plan identities.";
    control.publish();
    std::vector<ACETrainingExample> sources;
    ACETrainResumePlan              plan;
    if (!ace_training_load_dataset(command.dataset_dir, sources, error) ||
        !make_resume_plan(command, *dit, *text_encoder, *vae, sources, plan, error)) {
        return false;
    }
    control.max_steps = command.max_steps > 0 ? command.max_steps : plan.total_steps;
    ACETrainResumeProgress progress;
    std::string            resume_path;
    if (command.resume_dir) {
        std::filesystem::path  resolved;
        ACETrainCheckpointKind kind = ACE_TRAIN_CHECKPOINT_ADAPTER;
        if (!adapter_checkpoint_directory(command.resume_dir, resolved, error) ||
            !ace_train_checkpoint_kind(resolved.u8string(), kind, error)) {
            return false;
        }
        resume_path = resolved.u8string();
        int step    = 0;
        if (kind == ACE_TRAIN_CHECKPOINT_FULL) {
            bool legacy = false;
            if (!ace_train_read_progress(resume_path, plan, command.allow_legacy_resume, progress, step, error,
                                         &legacy)) {
                return false;
            }
            control.step  = step;
            control.epoch = progress.epoch;
            control.message =
                legacy ? "WARNING: legacy checkpoint has no recorded plan; original settings cannot be validated." :
                         "Native full-state resume plan validated.";
        } else {
            control.message = "PEFT weight-only restart: optimizer, cursor and schedule start from zero.";
        }
        if (step > control.max_steps) {
            error = "resume step exceeds the total target optimizer step";
            return false;
        }
        std::fprintf(stderr, "[Ace-Train] %s\n", control.message.c_str());
    }
    const auto output = std::filesystem::u8path(command.output_dir);
    if (std::filesystem::exists(output / "checkpoint.current") ||
        std::filesystem::exists(output / "trainer_state.json")) {
        std::filesystem::path current;
        std::error_code       ec;
        if (resume_path.empty() || !adapter_checkpoint_directory(command.output_dir, current, error) ||
            !std::filesystem::equivalent(current, std::filesystem::u8path(resume_path), ec) || ec) {
            error =
                "output already contains a native checkpoint; resume that current checkpoint or choose a new output "
                "directory";
            return false;
        }
    }
    if (train_cancel(&cancellation)) {
        return true;
    }
    control.phase   = "pre-encode";
    control.message = "Preparing training audio and conditioning.";
    control.publish();
    ACETrainPreparedDataset  prepared;
    std::vector<std::string> example_names;
    std::vector<float>       silence_latents;
    std::vector<float>       null_condition;
    if (!preprocess_dataset(command, *dit, *text_encoder, *vae, prepared, example_names, silence_latents,
                            null_condition, sources, plan, cancellation, error)) {
        return cancellation.cancelled;
    }
    if (train_cancel(&cancellation)) {
        return true;
    }
    return train_adapter(command, *dit, prepared, example_names, silence_latents, null_condition, plan, progress,
                         resume_path, cancellation, error);
}

int main(int argc, char ** argv) {
    if (argc == 2 && !std::strcmp(argv[1], "--version")) {
        std::puts(ACE_RELEASE_VERSION);
        return 0;
    }
    ace_training::TrainCliControl control;
    std::vector<char *>           training_args;
    std::string                   error;
    ACETrainCommand               command;
    try {
        if (!control.parse(argc, argv, training_args, error)) {
            std::fprintf(stderr, "[Ace-Train] %s\n", error.c_str());
            return 2;
        }
        if (control.info) {
            std::fputs(control.capabilities(), stdout);
            return 0;
        }
        if (!parse_command(int(training_args.size()), training_args.data(), command)) {
            usage(argv[0]);
            return 2;
        }
        ACETrainCancel cancellation{ &control };
        control.publish();
        bool ok = true;
        if (train_cancel(&cancellation)) {
            control.message = "Cancelled before loading models.";
        } else {
            ok = run_training(command, cancellation, error);
        }
        control.status = !ok ? "failed" : cancellation.cancelled ? "cancelled" : "completed";
        control.phase  = control.status;
        control.error  = ok ? "" : error;
        if (ok && cancellation.cancelled && control.message != "Cancelled before loading models.") {
            control.message = control.state_checkpoint.empty() ?
                                  "Cancelled before the first optimizer update." :
                                  "Cancelled at an optimizer boundary; native continuation is saved.";
        }
        control.publish();
        std::fprintf(stderr, "[Ace-Train] %s%s%s\n", control.status.c_str(), ok ? "" : ": ", ok ? "" : error.c_str());
        return ok ? 0 : 1;
    } catch (const std::exception & exception) {
        control.status = control.phase = "failed";
        control.error                  = exception.what();
        try {
            control.publish();
        } catch (...) {
        }
        std::fprintf(stderr, "[Ace-Train] ERROR: %s\n", exception.what());
        return 1;
    }
}
