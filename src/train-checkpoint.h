// ABOUTME: Saves and resumes complete ACE-Step native adapter training state.
// ABOUTME: Stores AdamW moments and progress beside PEFT-compatible adapter artifacts.

#pragma once

#include "train-adapter-checkpoint.h"
#include "train-adapter-optimizer.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

enum ACETrainCheckpointKind {
    ACE_TRAIN_CHECKPOINT_ADAPTER,
    ACE_TRAIN_CHECKPOINT_FULL,
};

static bool ace_file_fingerprint(const std::filesystem::path & path,
                                 std::string &                 fingerprint,
                                 std::string &                 error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open file for fingerprinting: " + path.string();
        return false;
    }
    uint64_t          hash = 14695981039346656037ULL;
    uint64_t          size = 0;
    std::vector<char> buffer(1024 * 1024);
    while (input) {
        input.read(buffer.data(), (std::streamsize) buffer.size());
        const std::streamsize count = input.gcount();
        for (std::streamsize i = 0; i < count; ++i) {
            hash ^= (uint8_t) buffer[(size_t) i];
            hash *= 1099511628211ULL;
        }
        size += (uint64_t) count;
    }
    if (!input.eof()) {
        error = "cannot read file for fingerprinting: " + path.string();
        return false;
    }
    char value[64];
    std::snprintf(value,
                  sizeof(value),
                  "fnv1a64:%016llx:%llu",
                  (unsigned long long) hash,
                  (unsigned long long) size);
    fingerprint = value;
    error.clear();
    return true;
}

static const char * const ace_train_checkpoint_files[] = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer_state.safetensors",
    "trainer_state.json",
};

static bool ace_write_train_checkpoint_manifest(const std::filesystem::path & directory,
                                                const std::string &           base_model_fingerprint,
                                                std::string &                 error) {
    error.clear();
    std::vector<std::string> fingerprints;
    for (const char * name : ace_train_checkpoint_files) {
        std::string fingerprint;
        if (!ace_file_fingerprint(directory / name, fingerprint, error)) {
            return false;
        }
        fingerprints.push_back(std::move(fingerprint));
    }
    std::ofstream output(directory / "checkpoint_manifest.json", std::ios::trunc);
    if (!output) {
        error = "cannot create checkpoint_manifest.json";
        return false;
    }
    output << "{\"format_version\":1,\"base_model_fingerprint\":\"" << base_model_fingerprint
           << "\",\"files\":{";
    for (size_t i = 0; i < fingerprints.size(); ++i) {
        output << (i == 0 ? "" : ",") << "\"" << ace_train_checkpoint_files[i] << "\":\""
               << fingerprints[i] << "\"";
    }
    output << "}}\n";
    if (!output.good()) {
        error = "cannot write checkpoint_manifest.json";
        return false;
    }
    return true;
}

static bool ace_validate_train_checkpoint_manifest(const std::filesystem::path & directory,
                                                   const std::string &           expected_base_fingerprint,
                                                   std::string &                 error) {
    error.clear();
    std::ifstream input(directory / "checkpoint_manifest.json");
    if (!input) {
        error = "checkpoint_manifest.json is missing";
        return false;
    }
    const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    yyjson_doc * document = yyjson_read(contents.data(), contents.size(), 0);
    yyjson_val * root = document ? yyjson_doc_get_root(document) : nullptr;
    yyjson_val * version = root ? yyjson_obj_get(root, "format_version") : nullptr;
    yyjson_val * base = root ? yyjson_obj_get(root, "base_model_fingerprint") : nullptr;
    yyjson_val * files = root ? yyjson_obj_get(root, "files") : nullptr;
    if (!yyjson_is_obj(root) || !yyjson_is_int(version) || yyjson_get_int(version) != 1 || !yyjson_is_str(base) ||
        !*yyjson_get_str(base) ||
        !yyjson_is_obj(files) || (!expected_base_fingerprint.empty() && expected_base_fingerprint != yyjson_get_str(base))) {
        if (document) {
            yyjson_doc_free(document);
        }
        error = "checkpoint manifest is invalid or belongs to a different base model";
        return false;
    }
    for (const char * name : ace_train_checkpoint_files) {
        yyjson_val * saved = yyjson_obj_get(files, name);
        std::string current;
        if (!yyjson_is_str(saved) || !ace_file_fingerprint(directory / name, current, error) ||
            current != yyjson_get_str(saved)) {
            yyjson_doc_free(document);
            if (error.empty()) {
                error = "checkpoint generation is incomplete or corrupt";
            }
            return false;
        }
    }
    yyjson_doc_free(document);
    error.clear();
    return true;
}

static bool ace_train_checkpoint_kind(const std::string &      directory,
                                      ACETrainCheckpointKind & kind,
                                      std::string &            error) {
    const std::filesystem::path path(directory);
    const bool has_manifest = std::filesystem::is_regular_file(path / "checkpoint_manifest.json");
    const bool has_pending = std::filesystem::is_regular_file(path / "checkpoint.pending");
    const bool has_progress = std::filesystem::is_regular_file(path / "trainer_state.json");
    const bool has_optimizer = std::filesystem::is_regular_file(path / "optimizer_state.safetensors");
    if (has_manifest) {
        if (!ace_validate_train_checkpoint_manifest(path, "", error)) {
            return false;
        }
        kind = ACE_TRAIN_CHECKPOINT_FULL;
        return true;
    }
    if (has_pending || has_progress || has_optimizer) {
        error = "training checkpoint is incomplete";
        return false;
    }
    kind = ACE_TRAIN_CHECKPOINT_ADAPTER;
    error.clear();
    return true;
}

static std::string ace_optimizer_tensor_name(const ACETrainAdapterTarget & target,
                                             const char *                  parameter,
                                             const char *                  moment) {
    return "optimizer." + ace_checkpoint_module_path(target.weight_name) + "." + parameter + "." + moment;
}

static bool ace_save_train_checkpoint(const std::string &               directory,
                                      const ACETrainAdapterState &      state,
                                      const ACETrainAdapterOptimizer &  optimizer,
                                      int                               completed_epochs,
                                      const std::string &               base_model_fingerprint,
                                      std::string &                     error) {
    error.clear();
    if (completed_epochs < 0 || optimizer.step < 0 || optimizer.params.size() != state.params.size() ||
        state.params.empty() || base_model_fingerprint.empty()) {
        error = "adapter and optimizer state cannot form a complete training checkpoint";
        return false;
    }

    std::vector<ACETrainCheckpointTensor> tensors;
    auto add_moments = [&](const ACETrainAdapterParam &     parameter,
                           const char *                      name,
                           const std::vector<float> &        values,
                           const ACETrainAdamWTensorState & moments) {
        if (values.empty()) {
            return moments.first_moment.empty() && moments.second_moment.empty();
        }
        if (moments.first_moment.size() != values.size() || moments.second_moment.size() != values.size()) {
            return false;
        }
        tensors.push_back({ ace_optimizer_tensor_name(parameter.target, name, "first_moment"),
                            { (int64_t) values.size() },
                            &moments.first_moment });
        tensors.push_back({ ace_optimizer_tensor_name(parameter.target, name, "second_moment"),
                            { (int64_t) values.size() },
                            &moments.second_moment });
        return true;
    };
    for (size_t i = 0; i < state.params.size(); ++i) {
        const ACETrainAdapterParam & parameter = state.params[i];
        const ACETrainAdapterOptimizerParam & moments = optimizer.params[i];
        if (!add_moments(parameter, "lora_A", parameter.a, moments.a) ||
            !add_moments(parameter, "lora_B", parameter.b, moments.b) ||
            !add_moments(parameter, "magnitude", parameter.magnitude, moments.magnitude)) {
            error = "optimizer moments do not match adapter parameters for " + parameter.target.weight_name;
            return false;
        }
    }

    const std::filesystem::path path(directory);
    const std::filesystem::path parent = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    const std::filesystem::path staging =
        parent / (path.filename().string() + ".checkpoint-stage-" + std::to_string(std::random_device {}()));
    std::error_code filesystem_error;
    std::filesystem::create_directories(staging, filesystem_error);
    if (filesystem_error) {
        error = "cannot create checkpoint staging directory: " + filesystem_error.message();
        return false;
    }
    auto discard_staging = [&]() {
        std::error_code ignored;
        std::filesystem::remove_all(staging, ignored);
    };
    if (!ace_save_train_adapter_checkpoint(staging.string(), state, error) ||
        !ace_write_safetensors(staging / "optimizer_state.safetensors", tensors, error)) {
        discard_staging();
        return false;
    }
    std::ofstream progress(staging / "trainer_state.json", std::ios::trunc);
    if (!progress) {
        error = "cannot create trainer_state.json";
        discard_staging();
        return false;
    }
    progress << "{\"completed_epochs\":" << completed_epochs << ",\"optimizer_step\":" << optimizer.step
             << ",\"base_model_fingerprint\":\"" << base_model_fingerprint << "\"}\n";
    progress.close();
    if (!progress.good()) {
        error = "cannot write trainer_state.json";
        discard_staging();
        return false;
    }
    if (!ace_write_train_checkpoint_manifest(staging, base_model_fingerprint, error)) {
        discard_staging();
        return false;
    }

    std::filesystem::create_directories(path, filesystem_error);
    if (filesystem_error) {
        error = "cannot create checkpoint directory: " + filesystem_error.message();
        discard_staging();
        return false;
    }
    {
        std::ofstream pending(path / "checkpoint.pending", std::ios::trunc);
        pending << "1\n";
        if (!pending.good()) {
            error = "cannot create checkpoint publication marker";
            discard_staging();
            return false;
        }
    }
    auto publish = [&](const char * name) {
        std::error_code publish_error;
        std::filesystem::remove(path / name, publish_error);
        publish_error.clear();
        std::filesystem::rename(staging / name, path / name, publish_error);
        if (publish_error) {
            error = "cannot publish checkpoint file " + std::string(name) + ": " + publish_error.message();
            return false;
        }
        return true;
    };
    for (const char * name : ace_train_checkpoint_files) {
        if (!publish(name)) {
            discard_staging();
            return false;
        }
    }
    if (!publish("checkpoint_manifest.json")) {
        discard_staging();
        return false;
    }
    std::filesystem::remove(path / "checkpoint.pending", filesystem_error);
    if (filesystem_error) {
        error = "cannot finalize checkpoint publication: " + filesystem_error.message();
        discard_staging();
        return false;
    }
    discard_staging();
    return true;
}

static bool ace_load_train_checkpoint(const std::string &                    directory,
                                      const std::vector<ACETrainAdapterTarget> & targets,
                                      ACETrainAdapterState &                   state,
                                      ACETrainAdapterOptimizer &               optimizer,
                                      int &                                    completed_epochs,
                                      const std::string &                      expected_base_fingerprint,
                                      const std::string &                      expected_adapter_type,
                                      std::string &                            error) {
    state = {};
    optimizer = {};
    completed_epochs = 0;
    error.clear();
    const std::filesystem::path path(directory);
    if (expected_base_fingerprint.empty()) {
        error = "expected base model fingerprint is missing";
        return false;
    }
    if (!ace_validate_train_checkpoint_manifest(path, expected_base_fingerprint, error) ||
        !ace_load_train_adapter_checkpoint(directory, targets, state, error, expected_adapter_type)) {
        return false;
    }

    std::ifstream progress(path / "trainer_state.json");
    if (!progress) {
        state = {};
        error = "trainer_state.json is missing or invalid";
        return false;
    }
    const std::string contents((std::istreambuf_iterator<char>(progress)), std::istreambuf_iterator<char>());
    yyjson_doc * document = yyjson_read(contents.data(), contents.size(), 0);
    yyjson_val * root = document ? yyjson_doc_get_root(document) : nullptr;
    yyjson_val * epochs = root ? yyjson_obj_get(root, "completed_epochs") : nullptr;
    yyjson_val * step = root ? yyjson_obj_get(root, "optimizer_step") : nullptr;
    yyjson_val * fingerprint = root ? yyjson_obj_get(root, "base_model_fingerprint") : nullptr;
    if (!yyjson_is_obj(root) || !yyjson_is_int(epochs) || !yyjson_is_int(step) || !yyjson_is_str(fingerprint) ||
        yyjson_get_int(epochs) < 0 || yyjson_get_int(step) < 0 ||
        expected_base_fingerprint != yyjson_get_str(fingerprint)) {
        if (document) {
            yyjson_doc_free(document);
        }
        state = {};
        completed_epochs = 0;
        error = "trainer_state.json is missing or invalid";
        return false;
    }
    completed_epochs = (int) yyjson_get_int(epochs);
    const int optimizer_step = (int) yyjson_get_int(step);
    yyjson_doc_free(document);

    STFile file;
    if (!st_open(&file, (path / "optimizer_state.safetensors").string().c_str())) {
        state = {};
        completed_epochs = 0;
        error = "cannot open optimizer_state.safetensors";
        return false;
    }
    auto find_entry = [&](const std::string & name) -> const STEntry * {
        for (const STEntry & entry : file.entries) {
            if (entry.name == name) {
                return &entry;
            }
        }
        return nullptr;
    };
    auto load_moments = [&](const ACETrainAdapterParam & parameter,
                            const char *                  name,
                            const std::vector<float> &    values,
                            ACETrainAdamWTensorState &    moments) {
        if (values.empty()) {
            moments = {};
            return true;
        }
        const STEntry * first = find_entry(ace_optimizer_tensor_name(parameter.target, name, "first_moment"));
        const STEntry * second = find_entry(ace_optimizer_tensor_name(parameter.target, name, "second_moment"));
        if (!first || !second || first->n_dims != 1 || second->n_dims != 1 ||
            first->shape[0] != (int64_t) values.size() || second->shape[0] != (int64_t) values.size()) {
            return false;
        }
        moments.first_moment.resize(values.size());
        moments.second_moment.resize(values.size());
        return adapter_to_f32(st_data(file, *first),
                              moments.first_moment.data(),
                              (int64_t) values.size(),
                              first->dtype) &&
               adapter_to_f32(st_data(file, *second),
                              moments.second_moment.data(),
                              (int64_t) values.size(),
                              second->dtype);
    };

    optimizer.step = optimizer_step;
    optimizer.params.resize(state.params.size());
    for (size_t i = 0; i < state.params.size(); ++i) {
        const ACETrainAdapterParam & parameter = state.params[i];
        ACETrainAdapterOptimizerParam & moments = optimizer.params[i];
        if (!load_moments(parameter, "lora_A", parameter.a, moments.a) ||
            !load_moments(parameter, "lora_B", parameter.b, moments.b) ||
            !load_moments(parameter, "magnitude", parameter.magnitude, moments.magnitude)) {
            st_close(&file);
            state = {};
            optimizer = {};
            completed_epochs = 0;
            error = "optimizer tensor shape mismatch for " + parameter.target.weight_name;
            return false;
        }
    }
    st_close(&file);
    return true;
}
