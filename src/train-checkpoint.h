// ABOUTME: Saves and resumes complete ACE-Step native adapter training state.
// ABOUTME: Stores AdamW moments and progress beside PEFT-compatible adapter artifacts.

#pragma once

#include "train-adapter-checkpoint.h"
#include "train-adapter-optimizer.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static std::string ace_optimizer_tensor_name(const ACETrainAdapterTarget & target,
                                             const char *                  parameter,
                                             const char *                  moment) {
    return "optimizer." + ace_checkpoint_module_path(target.weight_name) + "." + parameter + "." + moment;
}

static bool ace_save_train_checkpoint(const std::string &               directory,
                                      const ACETrainAdapterState &      state,
                                      const ACETrainAdapterOptimizer &  optimizer,
                                      int                               completed_epochs,
                                      std::string &                     error) {
    error.clear();
    if (completed_epochs < 0 || optimizer.step < 0 || optimizer.params.size() != state.params.size() ||
        state.params.empty()) {
        error = "adapter and optimizer state cannot form a complete training checkpoint";
        return false;
    }
    if (!ace_save_train_adapter_checkpoint(directory, state, error)) {
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
    if (!ace_write_safetensors(path / "optimizer_state.safetensors", tensors, error)) {
        return false;
    }
    std::ofstream progress(path / "trainer_state.json", std::ios::trunc);
    if (!progress) {
        error = "cannot create trainer_state.json";
        return false;
    }
    progress << "{\"completed_epochs\":" << completed_epochs << ",\"optimizer_step\":" << optimizer.step << "}\n";
    if (!progress.good()) {
        error = "cannot write trainer_state.json";
        return false;
    }
    return true;
}

static bool ace_load_train_checkpoint(const std::string &                    directory,
                                      const std::vector<ACETrainAdapterTarget> & targets,
                                      ACETrainAdapterState &                   state,
                                      ACETrainAdapterOptimizer &               optimizer,
                                      int &                                    completed_epochs,
                                      std::string &                            error) {
    state = {};
    optimizer = {};
    completed_epochs = 0;
    error.clear();
    if (!ace_load_train_adapter_checkpoint(directory, targets, state, error)) {
        return false;
    }

    const std::filesystem::path path(directory);
    std::ifstream progress(path / "trainer_state.json");
    if (!progress) {
        state = {};
        error = "trainer_state.json is missing or invalid";
        return false;
    }
    const std::string contents((std::istreambuf_iterator<char>(progress)), std::istreambuf_iterator<char>());
    int optimizer_step = 0;
    if (std::sscanf(contents.c_str(), "{\"completed_epochs\":%d,\"optimizer_step\":%d}",
                    &completed_epochs, &optimizer_step) != 2 ||
        completed_epochs < 0 || optimizer_step < 0) {
        state = {};
        completed_epochs = 0;
        error = "trainer_state.json is missing or invalid";
        return false;
    }

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
