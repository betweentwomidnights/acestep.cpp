// Opt-in production condition encoder replay on explicitly shared feature tensors.
#include "cond-enc.h"
#include "debug.h"

#include <filesystem>

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "Usage: %s <dit.gguf> <features-directory> <fresh-output> [no-fa]\n", argv[0]);
        return 1;
    }
    if (argc == 5 && std::string(argv[4]) != "no-fa") {
        return 1;
    }
    const std::filesystem::path input(argv[2]), output(argv[3]);
    if (std::filesystem::exists(output)) {
        return 1;
    }
    std::vector<int> text_shape, lyric_shape, timbre_shape;
    const auto       text   = debug_load((input / "text.bin").string().c_str(), text_shape);
    const auto       lyric  = debug_load((input / "lyric.bin").string().c_str(), lyric_shape);
    const auto       timbre = debug_load((input / "timbre.bin").string().c_str(), timbre_shape);
    if (text_shape.size() != 2 || lyric_shape.size() != 2 || timbre_shape.size() != 2 || text_shape[1] != 1024 ||
        lyric_shape[1] != 1024 || timbre_shape[1] != 64 || text.empty() || lyric.empty() || timbre.empty()) {
        return 1;
    }
    CondGGML model = {};
    if (!cond_ggml_load(&model, argv[1])) {
        return 1;
    }
    if (argc == 5) {
        model.use_flash_attn = false;
    }
    std::vector<float> encoded;
    int                frames = 0;
    const bool ok = cond_ggml_forward(&model, text.data(), text_shape[0], lyric.data(), lyric_shape[0], timbre.data(),
                                      timbre_shape[0], encoded, &frames);
    if (ok) {
        std::filesystem::create_directories(output);
        DebugDumper dump;
        debug_init(&dump, output.string().c_str());
        debug_dump_2d(&dump, "encoder", encoded.data(), frames, model.lyric_cfg.hidden_size);
    }
    cond_ggml_free(&model);
    std::fprintf(stderr, "%s: shared-feature condition encoder (%d frames)\n", ok ? "PASS" : "FAIL", frames);
    return ok ? 0 : 1;
}
