// Exercise the real CLI plan builder without loading GGUF tensors.
#define main ace_train_main
#include "../tools/ace-train.cpp"
#undef main

static void expect(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    namespace fs        = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("ace-plan-" + std::to_string(std::random_device{}()));
    fs::create_directories(root);
    auto write = [](const fs::path & path, const char * contents) {
        std::ofstream file(path);
        file << contents;
    };
    ModelEntry dit, text, vae;
    dit.path  = (root / "dit").u8string();
    text.path = (root / "text").u8string();
    vae.path  = (root / "vae").u8string();
    write(dit.path, "frozen base");
    write(text.path, "text encoder");
    write(vae.path, "VAE");
    std::vector<ACETrainingExample> sources(3);
    for (size_t i = 0; i < sources.size(); ++i) {
        sources[i].name          = std::to_string(i);
        sources[i].audio_path    = root / (std::to_string(i) + ".wav");
        sources[i].metadata_path = root / (std::to_string(i) + ".txt");
        write(sources[i].audio_path, "audio");
        write(sources[i].metadata_path, "metadata");
    }
    ACETrainCommand command;
    command.epochs       = 3;
    command.accumulation = 2;
    ACETrainResumePlan plan;
    std::string        error;
    expect(make_resume_plan(command, dit, text, vae, sources, plan, error), "build native plan");
    expect(plan.json.find(ACE_TRAIN_PREPROCESS_ALGORITHM) != std::string::npos,
           "preprocessing algorithm is part of the immutable plan");
    expect(plan.total_steps == 6 && plan.batches_per_epoch == 3, "partial accumulation counts as an update");
    ACETrainResumeProgress original;
    original.next_batch    = 2;
    original.epoch_batches = 2;
    original.epoch_loss    = 1.25;
    original.order         = { 2, 0, 1 };
    auto restore           = [&](const ACETrainResumeProgress & cursor, const ACETrainResumePlan & expected, int epoch,
                       int step) {
        const auto             json = ace_train_continuation(plan, cursor);
        yyjson_doc *           doc  = yyjson_read(json.data(), json.size(), 0);
        ACETrainResumeProgress actual;
        const bool ok = ace_train_restore_progress(yyjson_doc_get_root(doc), expected, epoch, step, actual, error);
        if (ok) {
            expect(actual.order == cursor.order && actual.epoch_loss == cursor.epoch_loss,
                             "cursor round trip is exact");
        }
        yyjson_doc_free(doc);
        return ok;
    };
    expect(restore(original, plan, 0, 1), "mid-epoch optimizer-boundary cursor is valid");
    {
        auto   legacy   = plan;
        auto * doc      = yyjson_read(plan.json.data(), plan.json.size(), 0);
        auto * modified = yyjson_doc_mut_copy(doc, nullptr);
        yyjson_mut_obj_remove_key(yyjson_mut_doc_get_root(modified), "preprocessing");
        char * json = yyjson_mut_write(modified, 0, nullptr);
        legacy.json = json;
        free(json);
        yyjson_mut_doc_free(modified);
        yyjson_doc_free(doc);
        const auto             state = ace_train_continuation(legacy, original);
        auto *                 saved = yyjson_read(state.data(), state.size(), 0);
        ACETrainResumeProgress actual;
        expect(!ace_train_restore_progress(yyjson_doc_get_root(saved), plan, 0, 1, actual, error),
               "resume plans predating corrected VAE preprocessing are rejected");
        yyjson_doc_free(saved);
    }
    auto invalid       = original;
    invalid.next_batch = invalid.epoch_batches = 1;
    expect(!restore(invalid, plan, 0, 1), "partial accumulated update is not resumable");
    invalid       = original;
    invalid.order = { 0, 0, 2 };
    expect(!restore(invalid, plan, 0, 1), "duplicate shuffle index rejected");
    expect(!restore(original, plan, 0, 2), "cursor and optimizer step must agree");
    ACETrainResumeProgress finished;
    finished.epoch = 3;
    expect(restore(finished, plan, 3, 6), "final cursor is valid");
    auto changed                   = command;
    changed.output_dir             = "another-location";
    changed.checkpoint_every       = 0;
    changed.checkpoint_every_steps = 2;
    changed.max_steps              = 1;
    changed.preprocess_cache       = false;
    changed.preprocess_cache_dir   = "relocated-cache";
    changed.cache_memory_mb        = 0;
    changed.preprocess_batch_size  = 2;
    ACETrainResumePlan other;
    expect(make_resume_plan(changed, dit, text, vae, sources, other, error) && restore(original, other, 0, 1),
           "output/cadence/stop target do not change the update trajectory");
    // Exercise the real cache identity builder independently of model loading.
#ifdef _WIN32
    _putenv_s("GGML_BACKEND", "CPU");
#else
    setenv("GGML_BACKEND", "CPU", 1);
#endif
    AceSynthParams params;
    ace_synth_default_params(&params);
    const auto dataset_path = root.u8string();
    command.dataset_dir     = dataset_path.c_str();
    ACETrainPreparedDataset cache, alternative;
    expect(prepare_cache_identities(command, plan, params, cache, error), "build cache input identities");
    expect(prepare_cache_identities(changed, other, params, alternative, error) &&
               cache.identities == alternative.identities,
           "cache/output/memory/stop controls do not invalidate tensors");
    auto cache_command = command;
    cache_command.learning_rate *= 2;
    cache_command.rank *= 2;
    expect(make_resume_plan(cache_command, dit, text, vae, sources, other, error) &&
               prepare_cache_identities(cache_command, other, params, alternative, error) &&
               cache.identities == alternative.identities,
           "optimizer and adapter options do not invalidate preprocessing");
    cache_command = command;
    cache_command.seed += 1;
    expect(make_resume_plan(cache_command, dit, text, vae, sources, other, error) &&
               prepare_cache_identities(cache_command, other, params, alternative, error) &&
               cache.identities != alternative.identities,
           "posterior seed invalidates preprocessing");
    params.vae_chunk += 1;
    expect(prepare_cache_identities(command, plan, params, alternative, error) &&
               cache.identities != alternative.identities,
           "VAE tiling settings invalidate preprocessing");
    params.vae_chunk -= 1;
    write(vae.path, "different VAE");
    expect(make_resume_plan(command, dit, text, vae, sources, other, error) &&
               prepare_cache_identities(command, other, params, alternative, error) &&
               cache.identities != alternative.identities,
           "model content invalidates preprocessing");
    write(vae.path, "VAE");
    changed        = command;
    changed.epochs = 4;
    expect(make_resume_plan(changed, dit, text, vae, sources, other, error) && !restore(original, other, 0, 1),
           "changing the schedule budget is rejected");
    changed = command;
    changed.seed += 1;
    expect(make_resume_plan(changed, dit, text, vae, sources, other, error) && !restore(original, other, 0, 1),
           "changed seed rejected");
    changed = command;
    changed.learning_rate *= 2;
    expect(make_resume_plan(changed, dit, text, vae, sources, other, error) && !restore(original, other, 0, 1),
           "changed learning rate rejected");
    changed                      = command;
    changed.preprocess.use_genre = true;
    expect(make_resume_plan(changed, dit, text, vae, sources, other, error) && !restore(original, other, 0, 1),
           "changed preprocessing rejected");
    write(sources[0].metadata_path, "changed metadata");
    expect(make_resume_plan(command, dit, text, vae, sources, other, error) && !restore(original, other, 0, 1),
           "metadata content change rejected");
    expect(prepare_cache_identities(command, other, params, alternative, error) &&
               cache.identities[0] != alternative.identities[0] && cache.identities[1] == alternative.identities[1],
           "metadata edits invalidate only the changed example");
    write(sources[0].metadata_path, "metadata");
    const auto relocated = root / "moved";
    fs::create_directory(relocated);
    for (auto & source : sources) {
        fs::copy_file(source.audio_path, relocated / source.audio_path.filename());
        fs::copy_file(source.metadata_path, relocated / source.metadata_path.filename());
        source.audio_path    = relocated / source.audio_path.filename();
        source.metadata_path = relocated / source.metadata_path.filename();
    }
    expect(make_resume_plan(command, dit, text, vae, sources, other, error) && restore(original, other, 0, 1),
           "moving identical inputs is supported");
    command.epochs = INT_MAX;
    expect(!make_resume_plan(command, dit, text, vae, sources, other, error), "step budget overflow rejected");
    fs::remove_all(root);
    std::puts("PASS: real trainer plan identities, exact cursor and safe resume validation");
}
