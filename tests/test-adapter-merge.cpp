// ABOUTME: Reloads emitted PEFT DoRA adapters through the native GGUF merge path.
// ABOUTME: Verifies module alpha resolution, normalization, and adapter-strength interpolation.

#include "adapter-merge.h"
#include "ggml-cpu.h"
#include "train-adapter-checkpoint.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <utility>
#include <vector>

struct temporary_directory {
    std::filesystem::path path;

    ~temporary_directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

static bool nearly_equal(float actual, float expected) {
    return std::fabs(actual - expected) < 1e-4f * std::fmax(1.0f, std::fabs(expected));
}

static ACETrainAdapterParam make_param(const char *         weight_name,
                                       const char *         module_name,
                                       int                  rank,
                                       int                  alpha,
                                       std::vector<float>   a,
                                       std::vector<float>   b,
                                       std::vector<float>   magnitude) {
    ACETrainAdapterParam param;
    param.target.weight_name    = weight_name;
    param.target.module_name    = module_name;
    param.target.in             = 2;
    param.target.out            = 2;
    param.target.rank           = rank;
    param.target.alpha          = alpha;
    param.target.base_rank      = 2;
    param.target.base_alpha     = 8;
    param.target.module_profile = "balanced";
    param.a                     = std::move(a);
    param.b                     = std::move(b);
    param.magnitude             = std::move(magnitude);
    return param;
}

static bool write_base_gguf(const std::filesystem::path & path,
                            const std::vector<float> &    q_weight,
                            const std::vector<float> &    v_weight) {
    ggml_init_params params = {
        /*.mem_size   =*/1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    ggml_context * context = ggml_init(params);
    if (!context) {
        return false;
    }
    ggml_tensor * q = ggml_new_tensor_2d(context, GGML_TYPE_F32, 2, 2);
    ggml_tensor * v = ggml_new_tensor_2d(context, GGML_TYPE_F32, 2, 2);
    ggml_set_name(q, "decoder.layers.0.self_attn.q_proj.weight");
    ggml_set_name(v, "decoder.layers.0.self_attn.v_proj.weight");
    std::memcpy(q->data, q_weight.data(), q_weight.size() * sizeof(float));
    std::memcpy(v->data, v_weight.data(), v_weight.size() * sizeof(float));

    gguf_context * file = gguf_init_empty();
    gguf_add_tensor(file, q);
    gguf_add_tensor(file, v);
    const bool written = gguf_write_to_file(file, path.string().c_str(), false);
    gguf_free(file);
    ggml_free(context);
    return written;
}

static std::vector<float> expected_weight(const std::vector<float> &       base,
                                          const ACETrainAdapterParam & param,
                                          float                         adapter_scale) {
    std::vector<float> fully_merged(base.size());
    const float        lora_scale = (float) param.target.alpha / (float) param.target.rank;
    for (int64_t out = 0; out < param.target.out; ++out) {
        float norm_squared = 0.0f;
        for (int64_t in = 0; in < param.target.in; ++in) {
            float delta = 0.0f;
            for (int64_t rank = 0; rank < param.target.rank; ++rank) {
                delta += param.b[(size_t) out * param.target.rank + rank] *
                         param.a[(size_t) rank * param.target.in + in];
            }
            const size_t index = (size_t) out * param.target.in + in;
            fully_merged[index] = base[index] + lora_scale * delta;
            norm_squared += fully_merged[index] * fully_merged[index];
        }
        const float magnitude_scale = param.magnitude[(size_t) out] / std::sqrt(norm_squared);
        for (int64_t in = 0; in < param.target.in; ++in) {
            const size_t index = (size_t) out * param.target.in + in;
            fully_merged[index] *= magnitude_scale;
        }
    }

    std::vector<float> expected(base.size());
    for (size_t i = 0; i < base.size(); ++i) {
        expected[i] = base[i] + adapter_scale * (fully_merged[i] - base[i]);
    }
    return expected;
}

static bool reload_adapter(const std::filesystem::path & directory,
                           float                         adapter_scale,
                           std::vector<float> &          q_weight,
                           std::vector<float> &          v_weight) {
    GGUFModel model = {};
    if (!gf_load(&model, (directory / "base.gguf").string().c_str())) {
        return false;
    }

    WeightCtx weights = {};
    wctx_init(&weights, 2);
    ggml_tensor * q = gf_load_tensor(&weights, model, "decoder.layers.0.self_attn.q_proj.weight");
    ggml_tensor * v = gf_load_tensor(&weights, model, "decoder.layers.0.self_attn.v_proj.weight");
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend || !adapter_merge(&weights, model, directory.string().c_str(), adapter_scale, backend) ||
        !wctx_alloc(&weights, backend)) {
        wctx_free(&weights);
        if (backend) {
            ggml_backend_free(backend);
        }
        gf_close(&model);
        return false;
    }

    q_weight.resize(4);
    v_weight.resize(4);
    ggml_backend_tensor_get(q, q_weight.data(), 0, q_weight.size() * sizeof(float));
    ggml_backend_tensor_get(v, v_weight.data(), 0, v_weight.size() * sizeof(float));
    wctx_free(&weights);
    ggml_backend_free(backend);
    gf_close(&model);
    return true;
}

static bool rejects_partial_merge(const std::filesystem::path & directory) {
    GGUFModel model = {};
    if (!gf_load(&model, (directory / "base.gguf").string().c_str())) {
        return false;
    }

    WeightCtx weights = {};
    wctx_init(&weights, 1);
    gf_load_tensor(&weights, model, "decoder.layers.0.self_attn.q_proj.weight");
    ggml_backend_t backend = ggml_backend_cpu_init();
    const bool rejected = backend && !adapter_merge(&weights, model, directory.string().c_str(), 1.0f, backend);
    wctx_free(&weights);
    if (backend) {
        ggml_backend_free(backend);
    }
    gf_close(&model);
    return rejected;
}

int main() {
    temporary_directory directory {
        std::filesystem::temp_directory_path() /
        ("ace-adapter-merge-" + std::to_string(std::random_device {}())),
    };
    std::error_code filesystem_error;
    std::filesystem::create_directories(directory.path, filesystem_error);
    if (filesystem_error) {
        std::fprintf(stderr, "failed to create adapter merge fixture: %s\n", filesystem_error.message().c_str());
        return 1;
    }

    const std::vector<float> q_base = { 1.0f, 2.0f, 3.0f, 4.0f };
    const std::vector<float> v_base = { 1.0f, 2.0f, 3.0f, 4.0f };
    if (!write_base_gguf(directory.path / "base.gguf", q_base, v_base)) {
        std::fputs("failed to write adapter merge GGUF fixture\n", stderr);
        return 1;
    }

    ACETrainAdapterState state;
    state.adapter_type  = "dora-rows";
    state.module_profile = "balanced";
    state.base_rank      = 2;
    state.base_alpha     = 8;
    state.params.push_back(make_param("decoder.layers.0.self_attn.q_proj.weight",
                                      "self_attn.q_proj",
                                      1,
                                      2,
                                      { 0.123456f, -0.654321f },
                                      { 0.333333f, -0.222222f },
                                      { 1.234567f, 2.345678f }));
    state.params.push_back(make_param("decoder.layers.0.self_attn.v_proj.weight",
                                      "self_attn.v_proj",
                                      2,
                                      6,
                                      { 0.712345f, -0.134567f, 0.245678f, 0.856789f },
                                      { 0.934567f, 0.312345f, -0.456789f, 0.623456f },
                                      { 1.456789f, 2.567891f }));

    std::string error;
    if (!ace_save_train_adapter_checkpoint(directory.path.string(), state, error)) {
        std::fprintf(stderr, "failed to emit PEFT adapter: %s\n", error.c_str());
        return 1;
    }

    adapter_config config;
    if (!adapter_read_config(directory.path.string().c_str(), config) || !config.use_dora ||
        config.rank_pattern["self_attn.q_proj"] != 1 || config.rank_pattern["self_attn.v_proj"] != 2 ||
        adapter_config_value_for_weight(config.alpha_pattern,
                                        "decoder.layers.0.self_attn.q_proj.weight",
                                        config.lora_alpha) != 2 ||
        adapter_config_value_for_weight(config.alpha_pattern,
                                        "decoder.layers.0.self_attn.v_proj.weight",
                                        config.lora_alpha) != 6) {
        std::fputs("emitted PEFT configuration did not preserve module semantics\n", stderr);
        return 1;
    }

    for (float adapter_scale : { 0.0f, 0.5f, 1.0f }) {
        std::vector<float> q_actual;
        std::vector<float> v_actual;
        if (!reload_adapter(directory.path, adapter_scale, q_actual, v_actual)) {
            std::fprintf(stderr, "native adapter reload failed at scale %.2f\n", adapter_scale);
            return 1;
        }
        const std::vector<float> q_expected = expected_weight(q_base, state.params[0], adapter_scale);
        const std::vector<float> v_expected = expected_weight(v_base, state.params[1], adapter_scale);
        for (size_t i = 0; i < q_expected.size(); ++i) {
            if (!nearly_equal(q_actual[i], q_expected[i]) || !nearly_equal(v_actual[i], v_expected[i])) {
                std::fprintf(stderr,
                             "native DoRA reload mismatch at scale %.2f index %zu: q=%f/%f v=%f/%f\n",
                             adapter_scale,
                             i,
                             q_actual[i],
                             q_expected[i],
                             v_actual[i],
                             v_expected[i]);
                return 1;
            }
        }
    }
    if (!rejects_partial_merge(directory.path)) {
        std::fputs("native adapter reload accepted a partial merge\n", stderr);
        return 1;
    }

    std::puts("PEFT DoRA native save/reload parity: OK");
    return 0;
}
