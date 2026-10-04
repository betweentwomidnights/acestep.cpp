// ABOUTME: Checks mixed-precision quantization with and without layer-count metadata.
// ABOUTME: Uses small synthetic tensors to verify the real quantizer's layer policy.

#define main ace_quantize_main
#include "../tools/quantize.cpp"
#undef main

#include <chrono>
#include <filesystem>
#include <string>

int main() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir    = std::filesystem::temp_directory_path() / ("ace-quantize-test-" + std::to_string(unique));
    std::filesystem::create_directory(dir);
    ggml_init_params params  = { 256 * 1024, nullptr, false };
    ggml_context *   tensors = ggml_init(params);
    gguf_context *   model   = gguf_init_empty();
    gguf_set_val_str(model, "general.architecture", "acestep-dit");
    for (int layer = 0; layer < 12; ++layer) {
        ggml_tensor *     tensor = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, 256, 4);
        const std::string name   = "model.layers." + std::to_string(layer) + ".self_attn.v_proj.weight";
        ggml_set_name(tensor, name.c_str());
        float * data = (float *) tensor->data;
        for (int i = 0; i < 1024; ++i) {
            data[i] = (float) (i % 29 - 14) / 13.0f;
        }
        gguf_add_tensor(model, tensor);
    }
    int result = 0;
    for (bool metadata : { false, true }) {
        if (metadata) {
            gguf_set_val_u32(model, "acestep-dit.block_count", 12);
        }
        std::string input  = (dir / (metadata ? "with.gguf" : "without.gguf")).string();
        std::string output = (dir / (metadata ? "with-q4.gguf" : "without-q4.gguf")).string();
        if (!gguf_write_to_file(model, input.c_str(), false)) {
            result = 1;
            break;
        }
        char   program[] = "quantize";
        char   type[]    = "Q4_K_M";
        char * args[]    = { program, &input[0], &output[0], type };
        if (ace_quantize_main(4, args) != 0) {
            result = 1;
            break;
        }
        gguf_init_params read_params = { true, nullptr };
        gguf_context *   quantized   = gguf_init_from_file(output.c_str(), read_params);
        if (!quantized || gguf_get_n_tensors(quantized) != 12) {
            if (quantized) {
                gguf_free(quantized);
            }
            result = 1;
            break;
        }
        for (int layer = 0; layer < 12; ++layer) {
            // With 12 layers: first 1, last 1, and every third projection use Q6_K.
            const ggml_type expected = (layer == 0 || layer == 11 || layer % 3 == 0) ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
            if (gguf_get_tensor_type(quantized, layer) != expected) {
                fprintf(stderr, "FAIL: layer %d policy differs (metadata=%d)\n", layer, (int) metadata);
                result = 1;
            }
        }
        gguf_free(quantized);
    }
    gguf_free(model);
    ggml_free(tensors);
    std::filesystem::remove_all(dir);
    if (!result) {
        fprintf(stderr, "PASS: Q4_K_M layer policy is independent of block-count metadata\n");
    }
    return result;
}
