// ABOUTME: Exposes ACE-Step model-backed preprocessing for native adapter training.
// ABOUTME: Maps paired dataset examples into sampled latents and DiT conditioning tensors.

#pragma once

#include "pipeline-synth.h"
#include "train-dataset.h"
#include "train-diffusion.h"

#include <cstdint>
#include <string>
#include <vector>

enum ACETrainTagPosition {
    ACE_TRAIN_TAG_PREPEND,
    ACE_TRAIN_TAG_APPEND,
    ACE_TRAIN_TAG_REPLACE,
};

struct ACETrainPreprocessConfig {
    ACETrainTagPosition tag_position = ACE_TRAIN_TAG_PREPEND;
    bool                use_genre    = false;
};

AceRequest ace_training_request(const ACETrainingExample &       example,
                                float                            duration,
                                const ACETrainPreprocessConfig & config);

bool ace_prepare_training_example(AceSynth *                       synth,
                                  const ACETrainingExample &       source,
                                  uint64_t                         seed,
                                  const ACETrainPreprocessConfig & config,
                                  ACETrainDiffusionExample &       prepared,
                                  std::string &                    error);

bool ace_prepare_training_dataset(AceSynth *                              synth,
                                  const std::vector<ACETrainingExample> & sources,
                                  uint64_t                                seed,
                                  const ACETrainPreprocessConfig &        config,
                                  std::vector<ACETrainDiffusionExample> & prepared,
                                  std::string &                           error);

bool ace_training_null_condition(const AceSynth * synth, std::vector<float> & condition, std::string & error);

bool ace_validate_training_audio(const std::string & path, int & samples, std::string & error);
