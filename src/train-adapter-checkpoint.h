// ABOUTME: Writes ACE-Step LoRA and DoRA-row parameters in PEFT-compatible checkpoint files.
// ABOUTME: Emits adapter_model.safetensors and adapter_config.json for Gary and native inference.

#pragma once

#include "adapter-config.h"
#include "safetensors.h"
#include "train-adapter-state.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

struct ACETrainCheckpointTensor {
    std::string                name;
    std::vector<int64_t>       shape;
    const std::vector<float> * values;
};

static std::string ace_checkpoint_module_path(const std::string & weight_name) {
    std::string path = weight_name;
    if (path.compare(0, 8, "decoder.") == 0) {
        path.erase(0, 8);
    }
    if (path.size() >= 7 && path.compare(path.size() - 7, 7, ".weight") == 0) {
        path.resize(path.size() - 7);
    }
    return "base_model.model." + path;
}

static bool ace_checkpoint_key_ends_with(const std::string & key, const char * suffix) {
    const size_t length = std::strlen(suffix);
    return key.size() >= length && key.compare(key.size() - length, length, suffix) == 0;
}

static std::string ace_checkpoint_adapter_key(const std::string & key) {
    struct KeySuffix {
        const char * serialized;
        const char * canonical;
    };
    static const KeySuffix suffixes[] = {
        { ".lora_A.default.weight", ".lora_A.weight" },
        { ".lora_A.weight", ".lora_A.weight" },
        { ".lora_B.default.weight", ".lora_B.weight" },
        { ".lora_B.weight", ".lora_B.weight" },
        { ".lora_magnitude_vector.default.weight", ".lora_magnitude_vector" },
        { ".lora_magnitude_vector.default", ".lora_magnitude_vector" },
        { ".lora_magnitude_vector.weight", ".lora_magnitude_vector" },
        { ".lora_magnitude_vector", ".lora_magnitude_vector" },
    };
    for (const KeySuffix & suffix : suffixes) {
        if (ace_checkpoint_key_ends_with(key, suffix.serialized)) {
            return key.substr(0, key.size() - std::strlen(suffix.serialized)) + suffix.canonical;
        }
    }
    return "";
}

static bool ace_write_safetensors(const std::filesystem::path &              path,
                                  const std::vector<ACETrainCheckpointTensor> & tensors,
                                  std::string &                                error) {
    std::string header = "{\"__metadata__\":{\"format\":\"pt\"}";
    size_t      offset = 0;
    for (const ACETrainCheckpointTensor & tensor : tensors) {
        size_t elements = 1;
        for (int64_t dimension : tensor.shape) {
            if (dimension <= 0) {
                error = "checkpoint tensor has an invalid dimension: " + tensor.name;
                return false;
            }
            elements *= (size_t) dimension;
        }
        if (!tensor.values || tensor.values->size() != elements) {
            error = "checkpoint tensor data does not match its shape: " + tensor.name;
            return false;
        }
        const size_t end = offset + elements * sizeof(float);
        header += ",\"" + tensor.name + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (size_t i = 0; i < tensor.shape.size(); ++i) {
            if (i > 0) {
                header += ",";
            }
            header += std::to_string(tensor.shape[i]);
        }
        header += "],\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(end) + "]}";
        offset = end;
    }
    header += "}";
    while (header.size() % 8 != 0) {
        header.push_back(' ');
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "cannot create " + path.string();
        return false;
    }
    const uint64_t header_size = (uint64_t) header.size();
    output.write(reinterpret_cast<const char *>(&header_size), sizeof(header_size));
    output.write(header.data(), (std::streamsize) header.size());
    for (const ACETrainCheckpointTensor & tensor : tensors) {
        output.write(reinterpret_cast<const char *>(tensor.values->data()),
                     (std::streamsize) (tensor.values->size() * sizeof(float)));
    }
    if (!output) {
        error = "failed while writing " + path.string();
        return false;
    }
    return true;
}

static bool ace_write_adapter_config(const std::filesystem::path & path,
                                     const ACETrainAdapterState & state,
                                     std::string &                error) {
    std::vector<std::string>       modules;
    std::set<std::string>          seen;
    std::map<std::string, int>     ranks;
    std::map<std::string, int>     alphas;
    for (const ACETrainAdapterParam & param : state.params) {
        const std::string & module = param.target.module_name;
        if (seen.insert(module).second) {
            modules.push_back(module);
        }
        ranks[module]  = param.target.rank;
        alphas[module] = param.target.alpha;
    }

    std::ofstream output(path, std::ios::trunc);
    if (!output) {
        error = "cannot create " + path.string();
        return false;
    }
    output << "{\n"
           << "  \"alpha_pattern\": {";
    bool first = true;
    for (const auto & entry : alphas) {
        output << (first ? "\n" : ",\n") << "    \"" << entry.first << "\": " << entry.second;
        first = false;
    }
    output << (alphas.empty() ? "" : "\n  ") << "},\n"
           << "  \"base_model_name_or_path\": \"ACE-Step-v1.5\",\n"
           << "  \"bias\": \"none\",\n"
           << "  \"fan_in_fan_out\": false,\n"
           << "  \"inference_mode\": true,\n"
           << "  \"lora_alpha\": " << state.base_alpha << ",\n"
           << "  \"lora_dropout\": 0.0,\n"
           << "  \"modules_to_save\": null,\n"
           << "  \"peft_type\": \"LORA\",\n"
           << "  \"r\": " << state.base_rank << ",\n"
           << "  \"rank_pattern\": {";
    first = true;
    for (const auto & entry : ranks) {
        output << (first ? "\n" : ",\n") << "    \"" << entry.first << "\": " << entry.second;
        first = false;
    }
    output << (ranks.empty() ? "" : "\n  ") << "},\n"
           << "  \"revision\": null,\n"
           << "  \"target_modules\": [";
    for (size_t i = 0; i < modules.size(); ++i) {
        output << (i == 0 ? "\n" : ",\n") << "    \"" << modules[i] << "\"";
    }
    output << (modules.empty() ? "" : "\n  ") << "],\n"
           << "  \"task_type\": null,\n"
           << "  \"use_dora\": " << (state.adapter_type == "dora-rows" ? "true" : "false") << ",\n"
           << "  \"use_rslora\": false\n"
           << "}\n";
    if (!output) {
        error = "failed while writing " + path.string();
        return false;
    }
    return true;
}

static bool ace_save_train_adapter_checkpoint(const std::string &          directory,
                                              const ACETrainAdapterState & state,
                                              std::string &                error) {
    error.clear();
    if (state.params.empty() || state.base_rank <= 0 || state.base_alpha <= 0 ||
        (state.adapter_type != "lora" && state.adapter_type != "dora-rows")) {
        error = "adapter state is incomplete";
        return false;
    }
    std::error_code filesystem_error;
    std::filesystem::create_directories(directory, filesystem_error);
    if (filesystem_error) {
        error = "cannot create checkpoint directory: " + filesystem_error.message();
        return false;
    }

    std::vector<ACETrainCheckpointTensor> tensors;
    tensors.reserve(state.params.size() * (state.adapter_type == "dora-rows" ? 3 : 2));
    for (const ACETrainAdapterParam & param : state.params) {
        const std::string module = ace_checkpoint_module_path(param.target.weight_name);
        tensors.push_back({ module + ".lora_A.weight", { param.target.rank, param.target.in }, &param.a });
        tensors.push_back({ module + ".lora_B.weight", { param.target.out, param.target.rank }, &param.b });
        if (state.adapter_type == "dora-rows") {
            tensors.push_back({ module + ".lora_magnitude_vector", { param.target.out }, &param.magnitude });
        }
    }

    const std::filesystem::path path(directory);
    if (!ace_write_safetensors(path / "adapter_model.safetensors", tensors, error)) {
        return false;
    }
    return ace_write_adapter_config(path / "adapter_config.json", state, error);
}

static bool ace_load_train_adapter_checkpoint(const std::string &                    directory,
                                              const std::vector<ACETrainAdapterTarget> & targets,
                                              ACETrainAdapterState &                   state,
                                              std::string &                            error,
                                              const std::string & expected_adapter_type = "") {
    state = {};
    error.clear();
    if (targets.empty()) {
        error = "adapter target inventory is empty";
        return false;
    }
    const std::filesystem::path path(directory);
    adapter_config config;
    if (!adapter_read_config(directory.c_str(), config)) {
        error = "adapter_config.json is missing or invalid";
        return false;
    }
    if (config.rank != targets.front().base_rank || config.lora_alpha != targets.front().base_alpha) {
        error = "checkpoint base rank or alpha does not match the requested training configuration";
        return false;
    }
    if (!expected_adapter_type.empty() &&
        ((expected_adapter_type == "dora-rows") != config.use_dora ||
         (expected_adapter_type != "lora" && expected_adapter_type != "dora-rows"))) {
        error = "checkpoint adapter type does not match the requested adapter type";
        return false;
    }
    const std::set<std::string> saved_modules(config.target_modules.begin(), config.target_modules.end());
    std::set<std::string>       requested_modules;
    for (const ACETrainAdapterTarget & target : targets) {
        requested_modules.insert(target.module_name);
    }
    if (saved_modules.size() != config.target_modules.size() || saved_modules != requested_modules) {
        error = "checkpoint target modules do not match the requested module profile";
        return false;
    }
    for (const ACETrainAdapterTarget & target : targets) {
        const int saved_rank =
            adapter_config_value_for_weight(config.rank_pattern, target.weight_name, config.rank);
        const int saved_alpha =
            adapter_config_value_for_weight(config.alpha_pattern, target.weight_name, config.lora_alpha);
        if (saved_rank != target.rank || saved_alpha != target.alpha) {
            error = "checkpoint rank or alpha does not match target " + target.weight_name;
            return false;
        }
    }

    STFile file;
    if (!st_open(&file, (path / "adapter_model.safetensors").string().c_str())) {
        error = "cannot open adapter_model.safetensors";
        return false;
    }
    std::set<std::string> expected_keys;
    for (const ACETrainAdapterTarget & target : targets) {
        const std::string module = ace_checkpoint_module_path(target.weight_name);
        expected_keys.insert(module + ".lora_A.weight");
        expected_keys.insert(module + ".lora_B.weight");
        if (config.use_dora) {
            expected_keys.insert(module + ".lora_magnitude_vector");
        }
    }
    std::set<std::string> serialized_keys;
    for (const STEntry & entry : file.entries) {
        const std::string key = ace_checkpoint_adapter_key(entry.name);
        if (!key.empty() && !serialized_keys.insert(key).second) {
            error = "checkpoint contains duplicate adapter tensor aliases";
            st_close(&file);
            return false;
        }
    }
    if (serialized_keys != expected_keys) {
        error = "checkpoint adapter tensor inventory does not match the requested model";
        st_close(&file);
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

    state.module_profile  = targets.front().module_profile;
    state.base_rank       = config.rank;
    state.base_alpha      = config.lora_alpha;
    bool adapter_type_set = false;
    bool dora_rows       = false;
    for (const ACETrainAdapterTarget & target : targets) {
        const std::string module = ace_checkpoint_module_path(target.weight_name);
        const STEntry *   a = find_entry(module + ".lora_A.weight");
        const STEntry *   b = find_entry(module + ".lora_B.weight");
        const STEntry *   magnitude = find_entry(module + ".lora_magnitude_vector");
        if (!a) {
            a = find_entry(module + ".lora_A.default.weight");
        }
        if (!b) {
            b = find_entry(module + ".lora_B.default.weight");
        }
        if (!magnitude) {
            magnitude = find_entry(module + ".lora_magnitude_vector.weight");
        }
        if (!magnitude) {
            magnitude = find_entry(module + ".lora_magnitude_vector.default");
        }
        if (!magnitude) {
            magnitude = find_entry(module + ".lora_magnitude_vector.default.weight");
        }
        if (!a || !b || a->n_dims != 2 || b->n_dims != 2 || a->shape[0] != target.rank ||
            a->shape[1] != target.in || b->shape[0] != target.out || b->shape[1] != target.rank) {
            error = "checkpoint adapter tensor shape mismatch for " + target.weight_name;
            st_close(&file);
            state = {};
            return false;
        }
        if (!adapter_type_set) {
            dora_rows       = magnitude != nullptr;
            adapter_type_set = true;
            if (dora_rows != config.use_dora) {
                error = "checkpoint DoRA tensors do not match adapter_config.json";
                st_close(&file);
                state = {};
                return false;
            }
        } else if ((magnitude != nullptr) != dora_rows) {
            error = "checkpoint mixes LoRA and DoRA targets";
            st_close(&file);
            state = {};
            return false;
        }

        ACETrainAdapterParam param;
        param.target = target;
        param.a.resize((size_t) target.rank * target.in);
        param.b.resize((size_t) target.out * target.rank);
        if (!adapter_to_f32(st_data(file, *a), param.a.data(), (int64_t) param.a.size(), a->dtype) ||
            !adapter_to_f32(st_data(file, *b), param.b.data(), (int64_t) param.b.size(), b->dtype)) {
            error = "unsupported checkpoint tensor dtype for " + target.weight_name;
            st_close(&file);
            state = {};
            return false;
        }
        if (dora_rows) {
            int64_t magnitude_elements = 1;
            for (int dimension = 0; dimension < magnitude->n_dims; ++dimension) {
                magnitude_elements *= magnitude->shape[dimension];
            }
            if (magnitude_elements != target.out) {
                error = "checkpoint DoRA magnitude shape mismatch for " + target.weight_name;
                st_close(&file);
                state = {};
                return false;
            }
            param.magnitude.resize((size_t) magnitude_elements);
            if (!adapter_to_f32(st_data(file, *magnitude),
                                param.magnitude.data(),
                                magnitude_elements,
                                magnitude->dtype)) {
                error = "unsupported DoRA magnitude dtype for " + target.weight_name;
                st_close(&file);
                state = {};
                return false;
            }
            std::vector<float> weight;
            if (!ace_train_tensor_to_f32(target.weight, weight, error)) {
                st_close(&file);
                state = {};
                return false;
            }
            param.base_norm_sq.resize((size_t) target.out);
            for (int64_t out = 0; out < target.out; ++out) {
                double norm_sq = 0.0;
                for (int64_t in = 0; in < target.in; ++in) {
                    const double value = weight[(size_t) out * target.in + in];
                    norm_sq += value * value;
                }
                param.base_norm_sq[(size_t) out] = (float) (norm_sq + (double) target.in * 1e-12);
            }
        }
        state.params.push_back(std::move(param));
    }
    state.adapter_type = dora_rows ? "dora-rows" : "lora";
    st_close(&file);
    return true;
}
