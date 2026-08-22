// ABOUTME: Parses PEFT adapter configuration into rank, alpha, and DoRA semantics.
// ABOUTME: Resolves module-specific pattern values for checkpoint resume and native inference.

#pragma once

#include "yyjson.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

struct adapter_config {
    int                        rank       = 0;
    int                        lora_alpha = 0;
    bool                       use_dora   = false;
    std::map<std::string, int> rank_pattern;
    std::map<std::string, int> alpha_pattern;
};

static void adapter_read_pattern(yyjson_val * value, std::map<std::string, int> & pattern) {
    if (!value || !yyjson_is_obj(value)) {
        return;
    }
    yyjson_obj_iter iterator = yyjson_obj_iter_with(value);
    yyjson_val *    key;
    while ((key = yyjson_obj_iter_next(&iterator))) {
        yyjson_val * item = yyjson_obj_iter_get_val(key);
        if (yyjson_is_str(key) && yyjson_is_int(item) && yyjson_get_int(item) > 0) {
            pattern[yyjson_get_str(key)] = (int) yyjson_get_int(item);
        }
    }
}

static bool adapter_read_config(const char * dir, adapter_config & config) {
    config           = {};
    std::string path = std::string(dir) + "/adapter_config.json";

    FILE * file = fopen(path.c_str(), "rb");
    if (!file) {
        return false;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return false;
    }
    long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return false;
    }

    std::vector<char> data((size_t) length);
    size_t            read = fread(data.data(), 1, data.size(), file);
    fclose(file);
    if (read != data.size()) {
        return false;
    }

    yyjson_doc * document = yyjson_read(data.data(), data.size(), 0);
    if (!document) {
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(document);
    if (!yyjson_is_obj(root)) {
        yyjson_doc_free(document);
        return false;
    }

    yyjson_val * rank = yyjson_obj_get(root, "r");
    if (rank && yyjson_is_int(rank) && yyjson_get_int(rank) > 0) {
        config.rank = (int) yyjson_get_int(rank);
    }
    yyjson_val * alpha = yyjson_obj_get(root, "lora_alpha");
    if ((!alpha || !yyjson_is_int(alpha)) && yyjson_obj_get(root, "alpha")) {
        alpha = yyjson_obj_get(root, "alpha");
    }
    if (alpha && yyjson_is_int(alpha) && yyjson_get_int(alpha) > 0) {
        config.lora_alpha = (int) yyjson_get_int(alpha);
    }
    yyjson_val * use_dora = yyjson_obj_get(root, "use_dora");
    if (use_dora && yyjson_is_bool(use_dora)) {
        config.use_dora = yyjson_get_bool(use_dora);
    }
    adapter_read_pattern(yyjson_obj_get(root, "rank_pattern"), config.rank_pattern);
    adapter_read_pattern(yyjson_obj_get(root, "alpha_pattern"), config.alpha_pattern);
    yyjson_doc_free(document);

    if (config.lora_alpha > 0) {
        fprintf(stderr, "[Adapter] adapter_config.json: alpha=%d, %zu module overrides\n", config.lora_alpha,
                config.alpha_pattern.size());
    }
    return config.rank > 0 && config.lora_alpha > 0;
}

static bool adapter_module_matches_pattern(const std::string & module, const std::string & pattern) {
    if (pattern.empty()) {
        return false;
    }
    if (pattern[0] == '^') {
        return module == pattern.substr(1);
    }
    if (module == pattern) {
        return true;
    }
    return module.size() > pattern.size() &&
           module.compare(module.size() - pattern.size(), pattern.size(), pattern) == 0 &&
           module[module.size() - pattern.size() - 1] == '.';
}

static int adapter_config_value_for_weight(const std::map<std::string, int> & pattern,
                                           const std::string &                gguf_name,
                                           int                                fallback) {
    std::string module = gguf_name;
    if (module.compare(0, 8, "decoder.") == 0) {
        module.erase(0, 8);
    }
    if (module.size() >= 7 && module.compare(module.size() - 7, 7, ".weight") == 0) {
        module.resize(module.size() - 7);
    }
    for (const auto & entry : pattern) {
        if (adapter_module_matches_pattern(module, entry.first)) {
            return entry.second;
        }
    }
    return fallback;
}
