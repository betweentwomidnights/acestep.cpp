// ABOUTME: Reloads emitted PEFT DoRA adapters through the native GGUF merge path.
// ABOUTME: Verifies module alpha resolution, normalization, and adapter-strength interpolation.

#include "adapter-merge.h"
#include "ggml-cpu.h"
#include "train-adapter-checkpoint.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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

static ACETrainAdapterParam make_param(const char *       weight_name,
                                       const char *       module_name,
                                       int                rank,
                                       int                alpha,
                                       std::vector<float> a,
                                       std::vector<float> b,
                                       std::vector<float> magnitude) {
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

struct base_weight_fixture {
    const char *               name;
    const std::vector<float> * values;
};

static bool write_base_gguf(const std::filesystem::path & path, const std::vector<base_weight_fixture> & weights) {
    ggml_init_params params = {
        /*.mem_size   =*/1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/false,
    };
    ggml_context * context = ggml_init(params);
    if (!context) {
        return false;
    }
    gguf_context * file = gguf_init_empty();
    for (const base_weight_fixture & weight : weights) {
        if (!weight.values || weight.values->size() != 4) {
            gguf_free(file);
            ggml_free(context);
            return false;
        }
        ggml_tensor * tensor = ggml_new_tensor_2d(context, GGML_TYPE_F32, 2, 2);
        ggml_set_name(tensor, weight.name);
        std::memcpy(tensor->data, weight.values->data(), weight.values->size() * sizeof(float));
        gguf_add_tensor(file, tensor);
    }
    const bool written = gguf_write_to_file(file, path.string().c_str(), false);
    gguf_free(file);
    ggml_free(context);
    return written;
}

static std::vector<float> expected_weight(const std::vector<float> &   base,
                                          const ACETrainAdapterParam & param,
                                          float                        adapter_scale) {
    std::vector<float> fully_merged(base.size());
    const float        lora_scale = (float) param.target.alpha / (float) param.target.rank;
    for (int64_t out = 0; out < param.target.out; ++out) {
        float norm_squared = 0.0f;
        for (int64_t in = 0; in < param.target.in; ++in) {
            float delta = 0.0f;
            for (int64_t rank = 0; rank < param.target.rank; ++rank) {
                delta +=
                    param.b[(size_t) out * param.target.rank + rank] * param.a[(size_t) rank * param.target.in + in];
            }
            const size_t index  = (size_t) out * param.target.in + in;
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
    ggml_tensor *  q       = gf_load_tensor(&weights, model, "decoder.layers.0.self_attn.q_proj.weight");
    ggml_tensor *  v       = gf_load_tensor(&weights, model, "decoder.layers.0.self_attn.v_proj.weight");
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

static bool rejects_adapter_merge(const std::filesystem::path &     directory,
                                  const std::vector<const char *> & staged_weights) {
    GGUFModel model = {};
    if (!gf_load(&model, (directory / "base.gguf").string().c_str())) {
        return false;
    }

    WeightCtx weights = {};
    wctx_init(&weights, (int) staged_weights.size());
    for (const char * name : staged_weights) {
        gf_load_tensor(&weights, model, name);
    }
    ggml_backend_t backend  = ggml_backend_cpu_init();
    const bool     rejected = backend && !adapter_merge(&weights, model, directory.string().c_str(), 1.0f, backend);
    wctx_free(&weights);
    if (backend) {
        ggml_backend_free(backend);
    }
    gf_close(&model);
    return rejected;
}

int main() {
    temporary_directory directory{
        std::filesystem::temp_directory_path() / ("ace-adapter-merge-" + std::to_string(std::random_device{}())),
    };
    std::error_code filesystem_error;
    std::filesystem::create_directories(directory.path, filesystem_error);
    if (filesystem_error) {
        std::fprintf(stderr, "failed to create adapter merge fixture: %s\n", filesystem_error.message().c_str());
        return 1;
    }

    const std::vector<float> q_base = { 1.0f, 2.0f, 3.0f, 4.0f };
    const std::vector<float> v_base = { 1.0f, 2.0f, 3.0f, 4.0f };
    if (!write_base_gguf(directory.path / "base.gguf", {
                                                           { "decoder.layers.0.self_attn.q_proj.weight", &q_base },
                                                           { "decoder.layers.0.self_attn.v_proj.weight", &v_base },
    })) {
        std::fputs("failed to write adapter merge GGUF fixture\n", stderr);
        return 1;
    }

    ACETrainAdapterState state;
    state.adapter_type   = "dora-rows";
    state.module_profile = "balanced";
    state.base_rank      = 2;
    state.base_alpha     = 8;
    state.params.push_back(make_param("decoder.layers.0.self_attn.q_proj.weight", "self_attn.q_proj", 1, 2,
                                      { 0.123456f, -0.654321f }, { 0.333333f, -0.222222f }, { 1.234567f, 2.345678f }));
    state.params.push_back(make_param("decoder.layers.0.self_attn.v_proj.weight", "self_attn.v_proj", 2, 6,
                                      { 0.712345f, -0.134567f, 0.245678f, 0.856789f },
                                      { 0.934567f, 0.312345f, -0.456789f, 0.623456f }, { 1.456789f, 2.567891f }));

    std::string error;
    if (!ace_save_train_adapter_checkpoint(directory.path.string(), state, error)) {
        std::fprintf(stderr, "failed to emit PEFT adapter: %s\n", error.c_str());
        return 1;
    }

    adapter_config config;
    if (!adapter_read_config(directory.path.string().c_str(), config) || !config.use_dora ||
        config.rank_pattern["self_attn.q_proj"] != 1 || config.rank_pattern["self_attn.v_proj"] != 2 ||
        adapter_config_value_for_weight(config.alpha_pattern, "decoder.layers.0.self_attn.q_proj.weight",
                                        config.lora_alpha) != 2 ||
        adapter_config_value_for_weight(config.alpha_pattern, "decoder.layers.0.self_attn.v_proj.weight",
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
                std::fprintf(stderr, "native DoRA reload mismatch at scale %.2f index %zu: q=%f/%f v=%f/%f\n",
                             adapter_scale, i, q_actual[i], q_expected[i], v_actual[i], v_expected[i]);
                return 1;
            }
        }
    }

    const std::filesystem::path published            = directory.path / "published";
    const std::filesystem::path published_generation = published / ".checkpoint-generations" / "generation-test";
    std::filesystem::create_directories(published_generation, filesystem_error);
    std::filesystem::copy_file(directory.path / "base.gguf", published / "base.gguf", filesystem_error);
    std::filesystem::copy_file(directory.path / "adapter_config.json", published_generation / "adapter_config.json",
                               filesystem_error);
    std::filesystem::copy_file(directory.path / "adapter_model.safetensors",
                               published_generation / "adapter_model.safetensors", filesystem_error);
    std::filesystem::create_directories(published / ".checkpoint-generations" / "generation-interrupted",
                                        filesystem_error);
    bool pointer_written = false;
    {
        std::ofstream pointer(published / "checkpoint.current", std::ios::binary | std::ios::trunc);
        pointer << (std::filesystem::path(".checkpoint-generations") / "generation-test").generic_string();
        pointer_written = pointer.good();
    }
    std::vector<float> published_q;
    std::vector<float> published_v;
    if (filesystem_error || !pointer_written || !reload_adapter(published, 1.0f, published_q, published_v)) {
        std::fputs("native adapter reload did not honor the selected checkpoint generation\n", stderr);
        return 1;
    }
    if (!rejects_adapter_merge(directory.path, { "decoder.layers.0.self_attn.q_proj.weight" })) {
        std::fputs("native adapter reload accepted a partial merge\n", stderr);
        return 1;
    }

    const std::filesystem::path incomplete = directory.path / "incomplete";
    std::filesystem::create_directories(incomplete, filesystem_error);
    std::filesystem::copy_file(directory.path / "base.gguf", incomplete / "base.gguf", filesystem_error);
    std::filesystem::copy_file(directory.path / "adapter_config.json", incomplete / "adapter_config.json",
                               filesystem_error);
    const std::string                     q_module = ace_checkpoint_module_path(state.params[0].target.weight_name);
    const std::string                     v_module = ace_checkpoint_module_path(state.params[1].target.weight_name);
    std::vector<ACETrainCheckpointTensor> incomplete_tensors = {
        { q_module + ".lora_A.weight",         { 1, 2 }, &state.params[0].a         },
        { q_module + ".lora_B.weight",         { 2, 1 }, &state.params[0].b         },
        { q_module + ".lora_magnitude_vector", { 2 },    &state.params[0].magnitude },
        { v_module + ".lora_A.weight",         { 2, 2 }, &state.params[1].a         },
        { v_module + ".lora_B.weight",         { 2, 2 }, &state.params[1].b         },
    };
    if (filesystem_error ||
        !ace_write_safetensors(incomplete / "adapter_model.safetensors", incomplete_tensors, error) ||
        !rejects_adapter_merge(incomplete, { "decoder.layers.0.self_attn.q_proj.weight" })) {
        std::fputs("native PEFT reload accepted a missing DoRA magnitude\n", stderr);
        return 1;
    }

    const std::filesystem::path missing_instance = directory.path / "missing-instance";
    ACETrainAdapterState        missing_instance_state;
    missing_instance_state.adapter_type   = "dora-rows";
    missing_instance_state.module_profile = "attention";
    missing_instance_state.base_rank      = 2;
    missing_instance_state.base_alpha     = 8;
    missing_instance_state.params.push_back(make_param("decoder.layers.1.self_attn.q_proj.weight", "self_attn.q_proj",
                                                       1, 2, { 0.123456f, -0.654321f }, { 0.333333f, -0.222222f },
                                                       { 1.234567f, 2.345678f }));
    if (!ace_save_train_adapter_checkpoint(missing_instance.string(), missing_instance_state, error) ||
        !write_base_gguf(missing_instance / "base.gguf",
                         {
                             { "decoder.layers.0.self_attn.q_proj.weight", &q_base },
                             { "decoder.layers.1.self_attn.q_proj.weight", &q_base },
    }) ||
        !rejects_adapter_merge(missing_instance, {
                                                     "decoder.layers.0.self_attn.q_proj.weight",
                                                     "decoder.layers.1.self_attn.q_proj.weight",
                                                 })) {
        std::fputs("native PEFT reload accepted a missing target module instance\n", stderr);
        return 1;
    }

    const std::filesystem::path invalid_config = directory.path / "invalid-config";
    std::filesystem::create_directories(invalid_config, filesystem_error);
    std::filesystem::copy_file(directory.path / "base.gguf", invalid_config / "base.gguf", filesystem_error);
    std::filesystem::copy_file(directory.path / "adapter_model.safetensors",
                               invalid_config / "adapter_model.safetensors", filesystem_error);
    std::ofstream invalid_config_file(invalid_config / "adapter_config.json", std::ios::trunc);
    invalid_config_file << "{\n";
    invalid_config_file.close();
    if (filesystem_error || !invalid_config_file.good() ||
        !rejects_adapter_merge(invalid_config, { "decoder.layers.0.self_attn.q_proj.weight" })) {
        std::fputs("native PEFT reload accepted an invalid adapter configuration\n", stderr);
        return 1;
    }

    std::puts("PEFT DoRA native save/reload parity: OK");
    return 0;
}
