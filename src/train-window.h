// ABOUTME: Selects reproducible latent-space windows for memory-bounded ACE adapter training.
// ABOUTME: Crops only audio tensors so full-song text and lyric conditioning remain intact for baseline experiments.

#pragma once

#include "train-diffusion.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

constexpr float ACE_TRAIN_LATENTS_PER_SECOND = 25.0f;

struct ACETrainWindowConfig {
    float min_seconds = 0.0f;
    float max_seconds = 0.0f;
};

struct ACETrainWindow {
    int  first_latent  = 0;
    int  latent_length = 0;
    bool cropped       = false;
};

static bool ace_train_window_config_valid(const ACETrainWindowConfig & config) {
    const bool disabled = config.min_seconds == 0.0f && config.max_seconds == 0.0f;
    return disabled || (std::isfinite(config.min_seconds) && std::isfinite(config.max_seconds) &&
                        config.min_seconds > 0.0f && config.max_seconds >= config.min_seconds);
}

static uint64_t ace_train_window_mix(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

static ACETrainWindow ace_train_select_window(int                          real_temporal_length,
                                               int                          padded_temporal_length,
                                               int                          patch_size,
                                               const ACETrainWindowConfig & config,
                                               uint64_t                     seed,
                                               int                          epoch,
                                               size_t                       example_index) {
    ACETrainWindow result;
    result.latent_length = padded_temporal_length;
    if (!ace_train_window_config_valid(config) || config.max_seconds == 0.0f || real_temporal_length <= 0 ||
        padded_temporal_length <= 0 || patch_size <= 0) {
        return result;
    }

    const int available_patches = real_temporal_length / patch_size;
    const int minimum_patches =
        std::max(1, (int) std::ceil(config.min_seconds * ACE_TRAIN_LATENTS_PER_SECOND / (float) patch_size));
    if (available_patches < minimum_patches) {
        return result;
    }
    const int maximum_patches = std::min(
        available_patches,
        std::max(minimum_patches,
                 (int) std::floor(config.max_seconds * ACE_TRAIN_LATENTS_PER_SECOND / (float) patch_size)));

    uint64_t random = ace_train_window_mix(seed ^ ((uint64_t) (epoch + 1) * 0xd1b54a32d192ed03ULL) ^
                                           ((uint64_t) (example_index + 1) * 0x94d049bb133111ebULL));
    const int window_patches = minimum_patches + (int) (random % (uint64_t) (maximum_patches - minimum_patches + 1));
    random                   = ace_train_window_mix(random);
    const int first_patch    = (int) (random % (uint64_t) (available_patches - window_patches + 1));

    result.first_latent  = first_patch * patch_size;
    result.latent_length = window_patches * patch_size;
    result.cropped       = result.first_latent != 0 || result.latent_length != padded_temporal_length;
    return result;
}

static bool ace_train_crop_example(const ACETrainDiffusionExample & source,
                                   int                              output_channels,
                                   int                              context_channels,
                                   int                              patch_size,
                                   const ACETrainWindowConfig &     config,
                                   uint64_t                         seed,
                                   int                              epoch,
                                   size_t                           example_index,
                                   ACETrainDiffusionExample &       destination,
                                   ACETrainWindow &                 window,
                                   std::string &                    error) {
    destination = {};
    window      = {};
    error.clear();
    if (output_channels <= 0 || context_channels <= 0 || patch_size <= 0 || source.real_temporal_length <= 0 ||
        source.target_latents.empty() || source.target_latents.size() % (size_t) output_channels != 0) {
        error = "cannot window an invalid training example";
        return false;
    }
    const int padded_temporal_length = (int) (source.target_latents.size() / (size_t) output_channels);
    if (padded_temporal_length % patch_size != 0 || source.real_temporal_length > padded_temporal_length ||
        source.context_latents.size() != (size_t) padded_temporal_length * context_channels) {
        error = "training example tensor shapes do not support latent windowing";
        return false;
    }

    window = ace_train_select_window(source.real_temporal_length, padded_temporal_length, patch_size, config, seed,
                                     epoch, example_index);
    if (!window.cropped) {
        destination = source;
        return true;
    }

    const size_t target_first = (size_t) window.first_latent * output_channels;
    const size_t target_count = (size_t) window.latent_length * output_channels;
    destination.target_latents.assign(source.target_latents.begin() + (std::ptrdiff_t) target_first,
                                      source.target_latents.begin() +
                                          (std::ptrdiff_t) (target_first + target_count));
    const size_t context_first = (size_t) window.first_latent * context_channels;
    const size_t context_count = (size_t) window.latent_length * context_channels;
    destination.context_latents.assign(source.context_latents.begin() + (std::ptrdiff_t) context_first,
                                       source.context_latents.begin() +
                                           (std::ptrdiff_t) (context_first + context_count));
    destination.encoder_hidden                 = source.encoder_hidden;
    destination.real_temporal_length           = window.latent_length;
    destination.real_encoder_sequence_length   = source.real_encoder_sequence_length;
    return true;
}
