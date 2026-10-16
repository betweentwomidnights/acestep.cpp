// Opt-in shared-noise VAE reference emitter using production tiling and sampling.
#include "vendor/minimp3/minimp3.h"
#define AUDIO_IO_MP3DEC_IMPL
#include "audio-io.h"
#include "debug.h"
#include "vae-enc.h"
#include "yyjson.h"

#include <filesystem>
#include <fstream>

struct PosteriorDump {
    DebugDumper      dump;
    yyjson_mut_doc * doc;
    yyjson_mut_val * tiles;
    int              index = 0;
};

static void observe_posterior(const float * raw,
                              int           frames,
                              int           trim,
                              int           count,
                              int           destination,
                              int           audio_offset,
                              int           audio_count,
                              void *        userdata) {
    auto &            output = *static_cast<PosteriorDump *>(userdata);
    const std::string name   = "tile-" + std::to_string(output.index++);
    debug_dump_2d(&output.dump, name.c_str(), raw, 128, frames);
    auto * tile = yyjson_mut_obj(output.doc);
    yyjson_mut_obj_add_strcpy(output.doc, tile, "file", (name + ".bin").c_str());
    yyjson_mut_obj_add_int(output.doc, tile, "audio_offset", audio_offset);
    yyjson_mut_obj_add_int(output.doc, tile, "audio_count", audio_count);
    yyjson_mut_obj_add_int(output.doc, tile, "raw_frames", frames);
    yyjson_mut_obj_add_int(output.doc, tile, "trim_start", trim);
    yyjson_mut_obj_add_int(output.doc, tile, "core_frames", count);
    yyjson_mut_obj_add_int(output.doc, tile, "write_start", destination);
    yyjson_mut_arr_append(output.tiles, tile);
}

static void dump_weight(const DebugDumper & dump, const std::string & name, ggml_tensor * tensor) {
    std::vector<float> values(size_t(ggml_nelements(tensor)));
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(values.size());
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(raw.data(), values.data(), int64_t(raw.size()));
    } else {
        GGML_ASSERT(tensor->type == GGML_TYPE_F32);
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    }
    int       shape[4];
    const int dimensions = ggml_n_dims(tensor);
    for (int i = 0; i < dimensions; ++i) {
        shape[i] = int(tensor->ne[dimensions - 1 - i]);
    }
    debug_dump(&dump, name.c_str(), values.data(), shape, dimensions);
}

static void dump_loaded_weights(const DebugDumper & dump, VAEEncoder & model) {
    auto conv = [&](const std::string & name, ggml_tensor * w, ggml_tensor * b) {
        dump_weight(dump, name + ".weight", w);
        dump_weight(dump, name + ".bias", b);
    };
    auto snake = [&](const std::string & name, ggml_tensor * alpha, ggml_tensor * inverse_beta) {
        dump_weight(dump, name + ".alpha_effective", alpha);
        dump_weight(dump, name + ".inverse_beta_effective", inverse_beta);
    };
    conv("encoder.conv1", model.c1w, model.c1b);
    for (int i = 0; i < 5; ++i) {
        auto &     block  = model.blk[i];
        const auto prefix = "encoder.block." + std::to_string(i);
        for (int r = 0; r < 3; ++r) {
            auto &     unit = block.ru[r];
            const auto name = prefix + ".res_unit" + std::to_string(r + 1);
            conv(name + ".conv1", unit.c1w, unit.c1b);
            conv(name + ".conv2", unit.c2w, unit.c2b);
            snake(name + ".snake1", unit.s1a, unit.s1b);
            snake(name + ".snake2", unit.s2a, unit.s2b);
        }
        conv(prefix + ".conv1", block.dw, block.db);
        snake(prefix + ".snake1", block.sa, block.sb);
    }
    conv("encoder.conv2", model.c2w, model.c2b);
    snake("encoder.snake1", model.sa, model.sb);
}

int main(int argc, char ** argv) {
    if (argc != 7 && argc != 8) {
        std::fprintf(stderr,
                     "Usage: %s <vae.gguf> <audio.wav> <noise.bin> <fresh-output> "
                     "<chunk-latents> <overlap-latents> [legacy-seed]\n",
                     argv[0]);
        return 1;
    }
    const std::filesystem::path output(argv[4]);
    const int                   chunk = std::atoi(argv[5]), overlap = std::atoi(argv[6]);
    const uint64_t              seed = argc == 8 ? std::strtoull(argv[7], nullptr, 10) : 42;
    std::vector<int>            shape;
    const auto                  noise = debug_load(argv[3], shape);
    if (std::filesystem::exists(output) || chunk < 1 || overlap < 0 || chunk <= 2 * overlap || noise.empty()) {
        return 1;
    }
    int     samples = 0;
    float * planar  = audio_read_48k(argv[2], &samples);
    if (!planar || samples < 1920 || noise.size() < size_t(samples / 1920 + 64) * 64) {
        free(planar);
        return 1;
    }
    float * audio = audio_planar_to_interleaved(planar, samples);
    free(planar);
    if (!audio) {
        return 1;
    }
    std::filesystem::create_directories(output / "weights");
    VAEEncoder model = {};
    vae_enc_load(&model, argv[1]);
    DebugDumper weights;
    debug_init(&weights, (output / "weights").string().c_str());
    dump_loaded_weights(weights, model);
    PosteriorDump dump = {};
    debug_init(&dump.dump, output.string().c_str());
    dump.doc    = yyjson_mut_doc_new(nullptr);
    auto * root = yyjson_mut_obj(dump.doc);
    yyjson_mut_doc_set_root(dump.doc, root);
    dump.tiles = yyjson_mut_arr(dump.doc);
    yyjson_mut_obj_add_val(dump.doc, root, "tiles", dump.tiles);
    yyjson_mut_obj_add_int(dump.doc, root, "audio_samples", samples);
    yyjson_mut_obj_add_int(dump.doc, root, "chunk_latents", chunk);
    yyjson_mut_obj_add_int(dump.doc, root, "overlap_latents", overlap);
    yyjson_mut_obj_add_strcpy(dump.doc, root, "backend", ggml_backend_name(model.backend));
    VAEEncLatentSampler sampler(seed);
    sampler.shared_noise        = noise.data();
    sampler.shared_noise_size   = noise.size();
    sampler.observe             = observe_posterior;
    sampler.observer_data       = &dump;
    const int          capacity = samples / 1920 + 64;
    std::vector<float> sampled(size_t(capacity) * 64), legacy(sampled.size());
    const int frames = vae_enc_encode_tiled(&model, audio, samples, sampled.data(), capacity, chunk, overlap, &sampler);
    const int legacy_frames =
        vae_enc_encode_tiled_sampled(&model, audio, samples, legacy.data(), capacity, seed, chunk, overlap);
    const bool ok = frames > 0 && frames == legacy_frames && sampler.shared_noise_offset == size_t(frames) * 64;
    if (ok) {
        debug_dump_2d(&dump.dump, "sampled", sampled.data(), frames, 64);
        debug_dump_2d(&dump.dump, "legacy-seeded", legacy.data(), frames, 64);
        yyjson_mut_obj_add_int(dump.doc, root, "frames", frames);
        yyjson_mut_obj_add_uint(dump.doc, root, "legacy_seed", seed);
        char * json = yyjson_mut_write(dump.doc, YYJSON_WRITE_PRETTY, nullptr);
        if (json) {
            std::ofstream(output / "posterior.json") << json << '\n';
            free(json);
        }
    }
    yyjson_mut_doc_free(dump.doc);
    vae_enc_free(&model);
    free(audio);
    std::fprintf(stderr, "%s: shared-noise posterior fixture (%d frames)\n", ok ? "PASS" : "FAIL", frames);
    return ok ? 0 : 1;
}
