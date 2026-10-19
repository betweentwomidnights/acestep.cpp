// Opt-in real-audio fixture: dump production training preprocessing without a DiT update.
#include "vendor/minimp3/minimp3.h"
#define AUDIO_IO_MP3DEC_IMPL
#include "audio-io.h"
#include "debug.h"
#include "model-store.h"
#include "pipeline-train.h"
#include "train-preprocess-cache.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "Usage: %s <dit.gguf> <text.gguf> <vae.gguf> <dataset> <fresh-output>\n", argv[0]);
        return 1;
    }
    const std::filesystem::path     output(argv[5]);
    std::vector<ACETrainingExample> sources;
    std::string                     error;
    if (std::filesystem::exists(output) || !ace_training_load_dataset(argv[4], sources, error)) {
        std::fprintf(stderr, "FAIL: output must be fresh; dataset: %s\n", error.c_str());
        return 1;
    }
    std::filesystem::create_directories(output);
    ModelStore * store = store_create(EVICT_STRICT);
    bool         ok    = true;
    for (size_t i = 0; i < sources.size() && ok; ++i) {
        const auto destination = output / std::to_string(i);
        std::filesystem::create_directory(destination);
        const std::string dump_dir = destination.string();
        AceSynthParams    params;
        ace_synth_default_params(&params);
        params.dit_path                = argv[1];
        params.text_encoder_path       = argv[2];
        params.vae_path                = argv[3];
        params.dump_dir                = dump_dir.c_str();
        AceSynth *               synth = ace_synth_load(store, &params);
        ACETrainDiffusionExample prepared;
        ok = synth && ace_prepare_training_example(synth, sources[i], 42 + i, {}, prepared, error);
        if (ok) {
            DebugDumper dump;
            debug_init(&dump, dump_dir.c_str());
            debug_dump_2d(&dump, "target", prepared.target_latents.data(), int(prepared.target_latents.size() / 64),
                          64);
            debug_dump_2d(&dump, "context", prepared.context_latents.data(), int(prepared.context_latents.size() / 128),
                          128);
            std::vector<float> null_condition;
            ok = ace_training_null_condition(synth, null_condition, error);
            if (ok) {
                debug_dump_1d(&dump, "null_condition", null_condition.data(), int(null_condition.size()));
                debug_dump_2d(&dump, "encoder", prepared.encoder_hidden.data(), prepared.real_encoder_sequence_length,
                              int(null_condition.size()));
                int     samples = 0;
                float * audio   = audio_read_48k(sources[i].audio_path.string().c_str(), &samples);
                ok              = audio && samples > 0;
                if (ok) {
                    const uint64_t hash = ace_cache_hash(audio, size_t(samples) * 2 * sizeof(float));
                    debug_dump_2d(&dump, "audio_planar", audio, 2, samples);
                    std::ofstream    meta(destination / "sample.json");
                    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
                    auto *           root = yyjson_mut_obj(doc);
                    yyjson_mut_doc_set_root(doc, root);
                    yyjson_mut_obj_add_strcpy(doc, root, "name", sources[i].name.c_str());
                    yyjson_mut_obj_add_uint(doc, root, "audio_samples", samples);
                    yyjson_mut_obj_add_uint(doc, root, "audio_planar_f32_fnv1a64", hash);
                    yyjson_mut_obj_add_uint(doc, root, "posterior_seed", 42 + i);
                    yyjson_mut_obj_add_int(doc, root, "real_temporal_length", prepared.real_temporal_length);
                    yyjson_mut_obj_add_int(doc, root, "encoder_length", prepared.real_encoder_sequence_length);
                    yyjson_mut_obj_add_int(doc, root, "vae_chunk", params.vae_chunk);
                    yyjson_mut_obj_add_int(doc, root, "vae_overlap", params.vae_overlap);
                    char * json = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY, nullptr);
                    ok          = json != nullptr;
                    if (json) {
                        meta << json << '\n';
                        free(json);
                        ok = bool(meta);
                    }
                    yyjson_mut_doc_free(doc);
                }
                free(audio);
            }
        }
        ace_synth_free(synth);
        std::fprintf(stderr, "[Preprocess-Fixture] %zu/%zu %s: %s\n", i + 1, sources.size(), sources[i].name.c_str(),
                     ok ? "PASS" : error.c_str());
    }
    store_free(store);
    return ok ? 0 : 1;
}
