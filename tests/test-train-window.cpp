// ABOUTME: Verifies deterministic rotating latent windows used by memory-bounded adapter training.
// ABOUTME: Ensures audio is cropped on patch boundaries while full-song conditioning remains byte-identical.

#include "train-window.h"

#include <cstdio>
#include <string>

static int fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

int main() {
    ACETrainDiffusionExample source;
    source.real_temporal_length         = 5000;
    source.real_encoder_sequence_length = 3;
    source.target_latents.resize(5000 * 2);
    source.context_latents.resize(5000 * 3);
    source.encoder_hidden = { 10.0f, 20.0f, 30.0f };
    for (size_t i = 0; i < source.target_latents.size(); ++i) {
        source.target_latents[i] = (float) i;
    }
    for (size_t i = 0; i < source.context_latents.size(); ++i) {
        source.context_latents[i] = (float) (100000 + i);
    }

    ACETrainWindowConfig config;
    config.min_seconds = 45.0f;
    config.max_seconds = 60.0f;
    ACETrainDiffusionExample first;
    ACETrainDiffusionExample repeat;
    ACETrainWindow           first_window;
    ACETrainWindow           repeat_window;
    std::string              error;
    if (!ace_train_crop_example(source, 2, 3, 2, config, 42, 7, 3, first, first_window, error) ||
        !ace_train_crop_example(source, 2, 3, 2, config, 42, 7, 3, repeat, repeat_window, error)) {
        return fail(error.c_str());
    }
    if (!first_window.cropped || first_window.first_latent != repeat_window.first_latent ||
        first_window.latent_length != repeat_window.latent_length || first_window.first_latent % 2 != 0 ||
        first_window.latent_length % 2 != 0 || first_window.latent_length < 1126 || first_window.latent_length > 1500) {
        return fail("window selection must be deterministic, bounded, and patch-aligned");
    }
    if (first.encoder_hidden != source.encoder_hidden ||
        first.real_encoder_sequence_length != source.real_encoder_sequence_length) {
        return fail("rotating windows must preserve complete full-song conditioning");
    }
    if (first.target_latents.front() != source.target_latents[(size_t) first_window.first_latent * 2] ||
        first.context_latents.front() != source.context_latents[(size_t) first_window.first_latent * 3] ||
        first.real_temporal_length != first_window.latent_length) {
        return fail("audio latent tensors were not cropped at the selected offset");
    }

    ACETrainDiffusionExample next_epoch;
    ACETrainWindow           next_window;
    if (!ace_train_crop_example(source, 2, 3, 2, config, 42, 8, 3, next_epoch, next_window, error) ||
        (next_window.first_latent == first_window.first_latent &&
         next_window.latent_length == first_window.latent_length)) {
        return fail("successive epochs must rotate the selected audio window");
    }

    ACETrainDiffusionExample short_source = source;
    short_source.real_temporal_length     = 1000;
    short_source.target_latents.resize(1000 * 2);
    short_source.context_latents.resize(1000 * 3);
    ACETrainDiffusionExample short_result;
    ACETrainWindow           short_window;
    if (!ace_train_crop_example(short_source, 2, 3, 2, config, 42, 0, 0, short_result, short_window, error) ||
        short_window.cropped || short_result.target_latents != short_source.target_latents) {
        return fail("tracks shorter than the minimum window must remain complete");
    }

    ACETrainWindowConfig invalid;
    invalid.min_seconds = 60.0f;
    invalid.max_seconds = 45.0f;
    if (ace_train_window_config_valid(invalid)) {
        return fail("an inverted window range must be rejected");
    }

    std::puts("ACE-Step rotating training windows: OK");
    return 0;
}
