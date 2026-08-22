// ABOUTME: Saves and resumes complete ACE-Step native adapter training state.
// ABOUTME: Stores AdamW moments and progress beside PEFT-compatible adapter artifacts.

#pragma once

#include "adapter-checkpoint-directory.h"
#include "train-adapter-checkpoint.h"
#include "train-adapter-optimizer.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <unistd.h>
#endif

enum ACETrainCheckpointKind {
    ACE_TRAIN_CHECKPOINT_ADAPTER,
    ACE_TRAIN_CHECKPOINT_FULL,
};

static bool ace_file_fingerprint(const std::filesystem::path & path, std::string & fingerprint, std::string & error) {
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
    std::snprintf(value, sizeof(value), "fnv1a64:%016llx:%llu", (unsigned long long) hash, (unsigned long long) size);
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

static bool ace_sync_checkpoint_path(const std::filesystem::path & path, bool directory, std::string & error) {
#ifdef _WIN32
    if (directory) {
        return true;
    }
    const DWORD flags = directory ? FILE_FLAG_BACKUP_SEMANTICS : FILE_ATTRIBUTE_NORMAL;
    HANDLE      handle =
        CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE || !FlushFileBuffers(handle)) {
        if (handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
        }
        error = "cannot make checkpoint path durable: " + path.string();
        return false;
    }
    CloseHandle(handle);
#else
    const int descriptor = open(path.c_str(), O_RDONLY);
    if (descriptor < 0 || fsync(descriptor) != 0) {
        if (descriptor >= 0) {
            close(descriptor);
        }
        error = "cannot make checkpoint path durable: " + path.string();
        return false;
    }
    close(descriptor);
#endif
    return true;
}

static bool ace_checkpoint_files_equal(const std::filesystem::path & first,
                                       const std::filesystem::path & second,
                                       bool &                        equal,
                                       std::string &                 error) {
    std::error_code filesystem_error;
    const uintmax_t first_size = std::filesystem::file_size(first, filesystem_error);
    if (filesystem_error) {
        error = "cannot inspect checkpoint file " + first.string() + ": " + filesystem_error.message();
        return false;
    }
    const uintmax_t second_size = std::filesystem::file_size(second, filesystem_error);
    if (filesystem_error) {
        error = "cannot inspect checkpoint file " + second.string() + ": " + filesystem_error.message();
        return false;
    }
    if (first_size != second_size) {
        equal = false;
        return true;
    }
    std::ifstream first_input(first, std::ios::binary);
    std::ifstream second_input(second, std::ios::binary);
    if (!first_input || !second_input) {
        error = "cannot compare checkpoint files";
        return false;
    }
    equal = std::equal(std::istreambuf_iterator<char>(first_input), std::istreambuf_iterator<char>(),
                       std::istreambuf_iterator<char>(second_input));
    return true;
}

static bool ace_replace_checkpoint_file(const std::filesystem::path & source,
                                        const std::filesystem::path & destination,
                                        std::string &                 error) {
#ifdef _WIN32
    const bool replaced = MoveFileExW(source.wstring().c_str(), destination.wstring().c_str(),
                                      MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    const bool replaced = ::rename(source.c_str(), destination.c_str()) == 0;
#endif
    if (!replaced) {
        error = "cannot publish checkpoint file " + destination.filename().string();
    }
    return replaced;
}

static bool ace_publish_peft_checkpoint(const std::filesystem::path & directory,
                                        const std::filesystem::path & relative_generation,
                                        const std::string &           token,
                                        std::string &                 error) {
    const std::filesystem::path generation = directory / relative_generation;
    const std::filesystem::path config     = directory / "adapter_config.json";
    const std::filesystem::path model      = directory / "adapter_model.safetensors";
    std::error_code             filesystem_error;
    auto inspect = [&](const std::filesystem::path & path, std::filesystem::file_status & status) {
        status = std::filesystem::symlink_status(path, filesystem_error);
        if (filesystem_error == std::make_error_code(std::errc::no_such_file_or_directory)) {
            filesystem_error.clear();
        }
        if (filesystem_error) {
            error = "cannot inspect checkpoint file " + path.filename().string() + ": " + filesystem_error.message();
            return false;
        }
        return true;
    };
    std::filesystem::file_status config_status;
    std::filesystem::file_status model_status;
    if (!inspect(config, config_status) || !inspect(model, model_status)) {
        return false;
    }
    const bool config_exists = std::filesystem::exists(config_status);
    const bool model_exists  = std::filesystem::exists(model_status);
    if ((config_exists && !std::filesystem::is_regular_file(config_status)) ||
        (model_exists && (!config_exists || !std::filesystem::is_regular_file(model_status)))) {
        error = "checkpoint directory contains incompatible PEFT files";
        return false;
    }
    if (config_exists) {
        bool equal = false;
        if (!ace_checkpoint_files_equal(config, generation / "adapter_config.json", equal, error)) {
            return false;
        }
        if (!equal) {
            error = "checkpoint adapter configuration cannot change between generations";
            return false;
        }
    } else {
        const std::filesystem::path temporary = directory / ("adapter_config.json.publish-" + token);
        std::filesystem::copy_file(generation / "adapter_config.json", temporary, filesystem_error);
        if (filesystem_error || !ace_sync_checkpoint_path(temporary, false, error) ||
            !ace_replace_checkpoint_file(temporary, config, error)) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            if (error.empty()) {
                error = "cannot publish adapter_config.json: " + filesystem_error.message();
            }
            return false;
        }
    }

    const std::filesystem::path temporary = directory / ("adapter_model.safetensors.publish-" + token);
    std::filesystem::create_hard_link(generation / "adapter_model.safetensors", temporary, filesystem_error);
    if (filesystem_error) {
        filesystem_error.clear();
        std::filesystem::copy_file(generation / "adapter_model.safetensors", temporary, filesystem_error);
    }
    if (filesystem_error || !ace_sync_checkpoint_path(temporary, false, error) ||
        !ace_replace_checkpoint_file(temporary, model, error)) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        if (error.empty()) {
            error = "cannot publish adapter_model.safetensors: " + filesystem_error.message();
        }
        return false;
    }
    return ace_sync_checkpoint_path(directory, true, error);
}

static bool ace_publish_train_checkpoint_generation(const std::filesystem::path & directory,
                                                    const std::filesystem::path & relative_generation,
                                                    const std::string &           token,
                                                    std::string &                 error) {
    const std::filesystem::path pointer   = directory / "checkpoint.current";
    const std::filesystem::path temporary = directory / ("checkpoint.current-" + token);
    std::error_code             filesystem_error;
#ifdef _WIN32
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output << relative_generation.generic_string();
    output.close();
    if (!output.good() || !ace_sync_checkpoint_path(temporary, false, error)) {
        std::filesystem::remove(temporary, filesystem_error);
        if (error.empty()) {
            error = "cannot create checkpoint generation pointer";
        }
        return false;
    }
#else
    std::filesystem::create_directory_symlink(relative_generation, temporary, filesystem_error);
    if (filesystem_error) {
        error = "cannot create checkpoint generation pointer: " + filesystem_error.message();
        return false;
    }
#endif
#ifdef _WIN32
    const bool published = MoveFileExW(temporary.wstring().c_str(), pointer.wstring().c_str(),
                                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    const bool published = ::rename(temporary.c_str(), pointer.c_str()) == 0;
#endif
    if (!published) {
        std::filesystem::remove(temporary, filesystem_error);
        error = "cannot publish checkpoint generation pointer";
        return false;
    }
    return ace_sync_checkpoint_path(directory, true, error);
}

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
    output << "{\"format_version\":1,\"base_model_fingerprint\":\"" << base_model_fingerprint << "\",\"files\":{";
    for (size_t i = 0; i < fingerprints.size(); ++i) {
        output << (i == 0 ? "" : ",") << "\"" << ace_train_checkpoint_files[i] << "\":\"" << fingerprints[i] << "\"";
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
    yyjson_doc *      document = yyjson_read(contents.data(), contents.size(), 0);
    yyjson_val *      root     = document ? yyjson_doc_get_root(document) : nullptr;
    yyjson_val *      version  = root ? yyjson_obj_get(root, "format_version") : nullptr;
    yyjson_val *      base     = root ? yyjson_obj_get(root, "base_model_fingerprint") : nullptr;
    yyjson_val *      files    = root ? yyjson_obj_get(root, "files") : nullptr;
    if (!yyjson_is_obj(root) || !yyjson_is_int(version) || yyjson_get_int(version) != 1 || !yyjson_is_str(base) ||
        !*yyjson_get_str(base) || !yyjson_is_obj(files) ||
        (!expected_base_fingerprint.empty() && expected_base_fingerprint != yyjson_get_str(base))) {
        if (document) {
            yyjson_doc_free(document);
        }
        error = "checkpoint manifest is invalid or belongs to a different base model";
        return false;
    }
    for (const char * name : ace_train_checkpoint_files) {
        yyjson_val * saved = yyjson_obj_get(files, name);
        std::string  current;
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
    std::filesystem::path path;
    if (!adapter_checkpoint_directory(directory, path, error)) {
        return false;
    }
    const bool has_manifest = std::filesystem::is_regular_file(path / "checkpoint_manifest.json");
    const bool has_pending  = std::filesystem::is_regular_file(std::filesystem::path(directory) / "checkpoint.pending");
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

static bool ace_save_train_checkpoint(const std::string &              directory,
                                      const ACETrainAdapterState &     state,
                                      const ACETrainAdapterOptimizer & optimizer,
                                      int                              completed_epochs,
                                      const std::string &              base_model_fingerprint,
                                      std::string &                    error) {
    error.clear();
    if (completed_epochs < 0 || optimizer.step < 0 || optimizer.params.size() != state.params.size() ||
        state.params.empty() || base_model_fingerprint.empty()) {
        error = "adapter and optimizer state cannot form a complete training checkpoint";
        return false;
    }

    std::vector<ACETrainCheckpointTensor> tensors;
    auto add_moments = [&](const ACETrainAdapterParam & parameter, const char * name, const std::vector<float> & values,
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
        const ACETrainAdapterParam &          parameter = state.params[i];
        const ACETrainAdapterOptimizerParam & moments   = optimizer.params[i];
        if (!add_moments(parameter, "lora_A", parameter.a, moments.a) ||
            !add_moments(parameter, "lora_B", parameter.b, moments.b) ||
            !add_moments(parameter, "magnitude", parameter.magnitude, moments.magnitude)) {
            error = "optimizer moments do not match adapter parameters for " + parameter.target.weight_name;
            return false;
        }
    }

    const std::filesystem::path path(directory);
    const std::string token = std::to_string(std::random_device{}()) + "-" + std::to_string(std::random_device{}());
    const std::filesystem::path generations = path / ".checkpoint-generations";
    const std::filesystem::path relative_generation =
        std::filesystem::path(".checkpoint-generations") / ("generation-" + token);
    const std::filesystem::path generation = path / relative_generation;
    const std::filesystem::path staging    = generations / (".stage-" + token);
    std::error_code             filesystem_error;
    std::filesystem::create_directories(path, filesystem_error);
    if (!filesystem_error) {
        std::filesystem::create_directories(generations, filesystem_error);
    }
    if (!filesystem_error) {
        std::filesystem::create_directory(staging, filesystem_error);
    }
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
    for (const char * name : ace_train_checkpoint_files) {
        if (!ace_sync_checkpoint_path(staging / name, false, error)) {
            discard_staging();
            return false;
        }
    }
    if (!ace_sync_checkpoint_path(staging / "checkpoint_manifest.json", false, error) ||
        !ace_sync_checkpoint_path(staging, true, error)) {
        discard_staging();
        return false;
    }
    std::filesystem::rename(staging, generation, filesystem_error);
    if (filesystem_error || !ace_sync_checkpoint_path(generations, true, error)) {
        if (error.empty()) {
            error = "cannot finalize checkpoint generation: " + filesystem_error.message();
        }
        discard_staging();
        return false;
    }
    if (!ace_publish_peft_checkpoint(path, relative_generation, token, error) ||
        !ace_sync_checkpoint_path(path, true, error) ||
        !ace_publish_train_checkpoint_generation(path, relative_generation, token, error)) {
        return false;
    }
    return true;
}

static bool ace_load_train_checkpoint(const std::string &                        directory,
                                      const std::vector<ACETrainAdapterTarget> & targets,
                                      ACETrainAdapterState &                     state,
                                      ACETrainAdapterOptimizer &                 optimizer,
                                      int &                                      completed_epochs,
                                      const std::string &                        expected_base_fingerprint,
                                      const std::string &                        expected_adapter_type,
                                      std::string &                              error) {
    state            = {};
    optimizer        = {};
    completed_epochs = 0;
    error.clear();
    std::filesystem::path path;
    if (expected_base_fingerprint.empty()) {
        error = "expected base model fingerprint is missing";
        return false;
    }
    if (!adapter_checkpoint_directory(directory, path, error)) {
        return false;
    }
    if (!ace_validate_train_checkpoint_manifest(path, expected_base_fingerprint, error) ||
        !ace_load_train_adapter_checkpoint(path.string(), targets, state, error, expected_adapter_type)) {
        return false;
    }

    std::ifstream progress(path / "trainer_state.json");
    if (!progress) {
        state = {};
        error = "trainer_state.json is missing or invalid";
        return false;
    }
    const std::string contents((std::istreambuf_iterator<char>(progress)), std::istreambuf_iterator<char>());
    yyjson_doc *      document    = yyjson_read(contents.data(), contents.size(), 0);
    yyjson_val *      root        = document ? yyjson_doc_get_root(document) : nullptr;
    yyjson_val *      epochs      = root ? yyjson_obj_get(root, "completed_epochs") : nullptr;
    yyjson_val *      step        = root ? yyjson_obj_get(root, "optimizer_step") : nullptr;
    yyjson_val *      fingerprint = root ? yyjson_obj_get(root, "base_model_fingerprint") : nullptr;
    if (!yyjson_is_obj(root) || !yyjson_is_int(epochs) || !yyjson_is_int(step) || !yyjson_is_str(fingerprint) ||
        yyjson_get_int(epochs) < 0 || yyjson_get_int(step) < 0 ||
        expected_base_fingerprint != yyjson_get_str(fingerprint)) {
        if (document) {
            yyjson_doc_free(document);
        }
        state            = {};
        completed_epochs = 0;
        error            = "trainer_state.json is missing or invalid";
        return false;
    }
    completed_epochs         = (int) yyjson_get_int(epochs);
    const int optimizer_step = (int) yyjson_get_int(step);
    yyjson_doc_free(document);

    STFile file;
    if (!st_open(&file, (path / "optimizer_state.safetensors").string().c_str())) {
        state            = {};
        completed_epochs = 0;
        error            = "cannot open optimizer_state.safetensors";
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
    auto load_moments = [&](const ACETrainAdapterParam & parameter, const char * name,
                            const std::vector<float> & values, ACETrainAdamWTensorState & moments) {
        if (values.empty()) {
            moments = {};
            return true;
        }
        const STEntry * first  = find_entry(ace_optimizer_tensor_name(parameter.target, name, "first_moment"));
        const STEntry * second = find_entry(ace_optimizer_tensor_name(parameter.target, name, "second_moment"));
        if (!first || !second || first->n_dims != 1 || second->n_dims != 1 ||
            first->shape[0] != (int64_t) values.size() || second->shape[0] != (int64_t) values.size()) {
            return false;
        }
        moments.first_moment.resize(values.size());
        moments.second_moment.resize(values.size());
        return adapter_to_f32(st_data(file, *first), moments.first_moment.data(), (int64_t) values.size(),
                              first->dtype) &&
               adapter_to_f32(st_data(file, *second), moments.second_moment.data(), (int64_t) values.size(),
                              second->dtype);
    };

    optimizer.step = optimizer_step;
    optimizer.params.resize(state.params.size());
    for (size_t i = 0; i < state.params.size(); ++i) {
        const ACETrainAdapterParam &    parameter = state.params[i];
        ACETrainAdapterOptimizerParam & moments   = optimizer.params[i];
        if (!load_moments(parameter, "lora_A", parameter.a, moments.a) ||
            !load_moments(parameter, "lora_B", parameter.b, moments.b) ||
            !load_moments(parameter, "magnitude", parameter.magnitude, moments.magnitude)) {
            st_close(&file);
            state            = {};
            optimizer        = {};
            completed_epochs = 0;
            error            = "optimizer tensor shape mismatch for " + parameter.target.weight_name;
            return false;
        }
    }
    st_close(&file);
    return true;
}
