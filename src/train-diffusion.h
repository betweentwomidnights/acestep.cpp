// ABOUTME: Prepares ACE-Step flow-matching batches and executes native adapter training steps.
// ABOUTME: Packs context and noisy latents, applies CFG dropout, uploads graph inputs, and runs AdamW.

#pragma once

#include "train-adapter-optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct ACETrainDiffusionConfig {
    float timestep_mean = -0.4f;
    float timestep_std  = 1.0f;
    float cfg_dropout   = 0.15f;
    bool  min_snr       = false;
    float min_snr_gamma = 5.0f;
};

struct ACETrainDiffusionExample {
    std::vector<float> target_latents;
    std::vector<float> context_latents;
    std::vector<float> encoder_hidden;
    int                real_temporal_length         = 0;
    int                real_encoder_sequence_length = 0;
};

struct ACETrainDiffusionBatch {
    int batch_size              = 0;
    int temporal_length         = 0;
    int encoder_sequence_length = 0;

    std::vector<float>       input_latents;
    std::vector<float>       encoder_hidden;
    std::vector<float>       target_velocity;
    std::vector<float>       loss_weights;
    std::vector<float>       timesteps;
    std::vector<float>       reference_timesteps;
    std::vector<int32_t>     positions;
    std::vector<ggml_fp16_t> self_attention_mask;
    std::vector<ggml_fp16_t> cross_attention_mask;
};

static float ace_train_sigmoid(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-value));
    }
    const float exponential = std::exp(value);
    return exponential / (1.0f + exponential);
}

static float ace_train_timestep_from_logits(float first, float second) {
    return std::max(ace_train_sigmoid(first), ace_train_sigmoid(second));
}

static float ace_train_flow_min_snr_weight(float timestep, float gamma) {
    const float clamped = std::max(1e-4f, std::min(1.0f - 1e-4f, timestep));
    const float ratio = (1.0f - clamped) / clamped;
    const float snr = std::min(1e6f, ratio * ratio);
    return std::min(snr, gamma) / std::max(snr, 1e-6f);
}

static bool ace_collate_training_examples(const DiTGGML &                              model,
                                          const std::vector<ACETrainDiffusionExample> & sources,
                                          const std::vector<float> &                   silence_latents,
                                          const std::vector<float> &                   null_condition,
                                          std::vector<ACETrainDiffusionExample> &       collated,
                                          int &                                        temporal_length,
                                          int &                                        encoder_sequence_length,
                                          std::string &                                error) {
    collated.clear();
    temporal_length = 0;
    encoder_sequence_length = 0;
    error.clear();
    if (sources.empty() || model.cfg.out_channels <= 0 || model.cfg.in_channels <= model.cfg.out_channels ||
        model.cfg.patch_size <= 0 || !model.cond_emb_w) {
        error = "invalid model or empty source batch for training collation";
        return false;
    }

    const int output_channels = model.cfg.out_channels;
    const int context_channels = model.cfg.in_channels - output_channels;
    const int hidden_size = (int) model.cond_emb_w->ne[0];
    for (const ACETrainDiffusionExample & source : sources) {
        if (source.target_latents.empty() || source.target_latents.size() % (size_t) output_channels != 0 ||
            source.encoder_hidden.empty() || source.encoder_hidden.size() % (size_t) hidden_size != 0) {
            error = "training example has an invalid latent or conditioning shape";
            return false;
        }
        const int source_temporal_length = (int) (source.target_latents.size() / (size_t) output_channels);
        const int source_encoder_length = (int) (source.encoder_hidden.size() / (size_t) hidden_size);
        if (source_temporal_length % model.cfg.patch_size != 0 ||
            source.context_latents.size() != (size_t) source_temporal_length * context_channels ||
            source.real_temporal_length <= 0 || source.real_temporal_length > source_temporal_length ||
            source.real_encoder_sequence_length <= 0 ||
            source.real_encoder_sequence_length > source_encoder_length) {
            error = "training example lengths do not match its tensor shapes";
            return false;
        }
        temporal_length = std::max(temporal_length, source_temporal_length);
        encoder_sequence_length = std::max(encoder_sequence_length, source_encoder_length);
    }
    if (context_channels < output_channels ||
        silence_latents.size() < (size_t) temporal_length * output_channels ||
        null_condition.size() != (size_t) hidden_size) {
        error = "training padding values do not match the DiT dimensions";
        return false;
    }

    collated.resize(sources.size());
    for (size_t sample = 0; sample < sources.size(); ++sample) {
        const ACETrainDiffusionExample & source = sources[sample];
        ACETrainDiffusionExample & destination = collated[sample];
        destination.target_latents.assign((size_t) temporal_length * output_channels, 0.0f);
        std::copy(source.target_latents.begin(), source.target_latents.end(), destination.target_latents.begin());
        destination.context_latents.resize((size_t) temporal_length * context_channels);
        for (int time = 0; time < temporal_length; ++time) {
            float * context = destination.context_latents.data() + (size_t) time * context_channels;
            const float * silence = silence_latents.data() + (size_t) time * output_channels;
            std::copy(silence, silence + output_channels, context);
            std::fill(context + output_channels, context + context_channels, 1.0f);
        }
        std::copy(source.context_latents.begin(),
                  source.context_latents.end(),
                  destination.context_latents.begin());
        destination.encoder_hidden.resize((size_t) encoder_sequence_length * hidden_size);
        for (int position = 0; position < encoder_sequence_length; ++position) {
            std::copy(null_condition.begin(),
                      null_condition.end(),
                      destination.encoder_hidden.begin() + (size_t) position * hidden_size);
        }
        std::copy(source.encoder_hidden.begin(), source.encoder_hidden.end(), destination.encoder_hidden.begin());
        destination.real_temporal_length = source.real_temporal_length;
        destination.real_encoder_sequence_length = source.real_encoder_sequence_length;
    }
    return true;
}

static bool ace_prepare_train_diffusion_batch(const DiTGGML &                              model,
                                              const std::vector<ACETrainDiffusionExample> & examples,
                                              int                                          temporal_length,
                                              int                                          encoder_sequence_length,
                                              const std::vector<float> &                   null_condition,
                                              uint64_t                                     seed,
                                              const ACETrainDiffusionConfig &              config,
                                              ACETrainDiffusionBatch &                     batch,
                                              std::string &                                error) {
    batch = {};
    error.clear();
    if (examples.empty() || temporal_length <= 0 || encoder_sequence_length <= 0 || model.cfg.patch_size <= 0 ||
        temporal_length % model.cfg.patch_size != 0 || model.cfg.out_channels <= 0 ||
        model.cfg.in_channels <= model.cfg.out_channels || !model.cond_emb_w) {
        error = "invalid diffusion batch dimensions or model configuration";
        return false;
    }
    if (!std::isfinite(config.timestep_mean) || !std::isfinite(config.timestep_std) ||
        config.timestep_std < 0.0f || !std::isfinite(config.cfg_dropout) || config.cfg_dropout < 0.0f ||
        config.cfg_dropout > 1.0f || !std::isfinite(config.min_snr_gamma) || config.min_snr_gamma <= 0.0f) {
        error = "invalid diffusion sampling configuration";
        return false;
    }

    const int batch_size       = (int) examples.size();
    const int output_channels  = model.cfg.out_channels;
    const int context_channels = model.cfg.in_channels - output_channels;
    const int hidden_size      = (int) model.cond_emb_w->ne[0];
    const int sequence_length  = temporal_length / model.cfg.patch_size;
    const size_t target_size   = (size_t) output_channels * temporal_length;
    const size_t context_size  = (size_t) context_channels * temporal_length;
    const size_t encoder_size  = (size_t) hidden_size * encoder_sequence_length;
    if (config.cfg_dropout > 0.0f && null_condition.size() != (size_t) hidden_size) {
        error = "null condition width does not match the encoder hidden width";
        return false;
    }
    for (const ACETrainDiffusionExample & example : examples) {
        if (example.target_latents.size() != target_size || example.context_latents.size() != context_size ||
            example.encoder_hidden.size() != encoder_size || example.real_temporal_length <= 0 ||
            example.real_temporal_length > temporal_length || example.real_encoder_sequence_length < 0 ||
            example.real_encoder_sequence_length > encoder_sequence_length) {
            error = "diffusion example tensor shape does not match the requested batch";
            return false;
        }
    }

    batch.batch_size              = batch_size;
    batch.temporal_length         = temporal_length;
    batch.encoder_sequence_length = encoder_sequence_length;
    batch.input_latents.resize((size_t) model.cfg.in_channels * temporal_length * batch_size);
    batch.encoder_hidden.resize(encoder_size * batch_size);
    batch.target_velocity.resize(target_size * batch_size);
    batch.loss_weights.resize((size_t) temporal_length * batch_size);
    batch.timesteps.resize(batch_size);
    batch.reference_timesteps.resize(batch_size);
    batch.positions.resize((size_t) sequence_length * batch_size);
    batch.self_attention_mask.resize((size_t) sequence_length * sequence_length * batch_size);
    batch.cross_attention_mask.resize((size_t) encoder_sequence_length * sequence_length * batch_size);

    std::mt19937_64                       generator(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::normal_distribution<float>       normal(0.0f, 1.0f);
    std::vector<float>                    noise(target_size * batch_size);
    std::vector<bool>                     drop_condition((size_t) batch_size);

    for (int sample = 0; sample < batch_size; ++sample) {
        drop_condition[(size_t) sample] = config.cfg_dropout > 0.0f && uniform(generator) < config.cfg_dropout;
    }
    for (float & value : noise) {
        value = normal(generator);
    }
    for (int sample = 0; sample < batch_size; ++sample) {
        const ACETrainDiffusionExample & example = examples[(size_t) sample];
        float * encoder_destination = batch.encoder_hidden.data() + (size_t) sample * encoder_size;
        if (drop_condition[(size_t) sample]) {
            for (int position = 0; position < encoder_sequence_length; ++position) {
                std::copy(null_condition.begin(),
                          null_condition.end(),
                          encoder_destination + (size_t) position * hidden_size);
            }
        } else {
            std::copy(example.encoder_hidden.begin(), example.encoder_hidden.end(), encoder_destination);
        }
    }

    std::vector<float> first_logits((size_t) batch_size);
    for (int sample = 0; sample < batch_size; ++sample) {
        first_logits[(size_t) sample] = config.timestep_mean + config.timestep_std * normal(generator);
    }
    for (int sample = 0; sample < batch_size; ++sample) {
        const float second_logit = config.timestep_mean + config.timestep_std * normal(generator);
        const float timestep = ace_train_timestep_from_logits(first_logits[(size_t) sample], second_logit);
        batch.timesteps[(size_t) sample] = timestep;
        batch.reference_timesteps[(size_t) sample] = timestep;
        const float sample_weight = config.min_snr ? ace_train_flow_min_snr_weight(timestep, config.min_snr_gamma)
                                                   : 1.0f;
        for (int time = 0; time < temporal_length; ++time) {
            batch.loss_weights[(size_t) sample * temporal_length + time] =
                time < examples[(size_t) sample].real_temporal_length
                    ? sample_weight / (float) examples[(size_t) sample].real_temporal_length
                    : 0.0f;
        }
    }

    for (int sample = 0; sample < batch_size; ++sample) {
        const ACETrainDiffusionExample & example = examples[(size_t) sample];
        for (int time = 0; time < temporal_length; ++time) {
            float * input = batch.input_latents.data() +
                            ((size_t) sample * temporal_length + time) * model.cfg.in_channels;
            const float * context = example.context_latents.data() + (size_t) time * context_channels;
            std::copy(context, context + context_channels, input);
            for (int channel = 0; channel < output_channels; ++channel) {
                const size_t latent_index = (size_t) time * output_channels + channel;
                const size_t batch_index = (size_t) sample * target_size + latent_index;
                const float data = example.target_latents[latent_index];
                const float sampled_noise = noise[batch_index];
                input[context_channels + channel] =
                    batch.timesteps[(size_t) sample] * sampled_noise +
                    (1.0f - batch.timesteps[(size_t) sample]) * data;
                batch.target_velocity[batch_index] = sampled_noise - data;
            }
        }

        for (int position = 0; position < sequence_length; ++position) {
            batch.positions[(size_t) sample * sequence_length + position] = position;
        }
        const int real_sequence_length =
            (example.real_temporal_length + model.cfg.patch_size - 1) / model.cfg.patch_size;
        for (int query = 0; query < sequence_length; ++query) {
            for (int key = 0; key < sequence_length; ++key) {
                const int distance = query > key ? query - key : key - query;
                const bool within_window = model.cfg.sliding_window <= 0 || sequence_length <= model.cfg.sliding_window ||
                                           distance <= model.cfg.sliding_window;
                const bool visible = key < real_sequence_length && within_window;
                const size_t offset = (size_t) sample * sequence_length * sequence_length +
                                      (size_t) query * sequence_length + key;
                batch.self_attention_mask[offset] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
            }
            for (int key = 0; key < encoder_sequence_length; ++key) {
                const size_t offset = (size_t) sample * encoder_sequence_length * sequence_length +
                                      (size_t) query * encoder_sequence_length + key;
                batch.cross_attention_mask[offset] =
                    ggml_fp32_to_fp16(key < example.real_encoder_sequence_length ? 0.0f : -INFINITY);
            }
        }
    }
    return true;
}

static bool ace_upload_train_diffusion_batch(ACETrainDiTGraph &             training,
                                             const ACETrainDiffusionBatch & batch,
                                             std::string &                  error) {
    error.clear();
    auto matches = [](const ggml_tensor * tensor, const std::vector<float> & values) {
        return tensor && (size_t) ggml_nelements(tensor) == values.size();
    };
    if (batch.batch_size <= 0 || batch.temporal_length <= 0 || batch.encoder_sequence_length <= 0 ||
        !matches(training.input_latents, batch.input_latents) ||
        !matches(training.encoder_hidden, batch.encoder_hidden) ||
        !matches(training.target_velocity, batch.target_velocity) ||
        !matches(training.loss_weights, batch.loss_weights) ||
        !matches(training.timestep, batch.timesteps) ||
        !matches(training.reference_timestep, batch.reference_timesteps) || !training.positions ||
        (size_t) ggml_nelements(training.positions) != batch.positions.size() || !training.self_attention_mask ||
        (size_t) ggml_nelements(training.self_attention_mask) != batch.self_attention_mask.size() ||
        !training.cross_attention_mask ||
        (size_t) ggml_nelements(training.cross_attention_mask) != batch.cross_attention_mask.size()) {
        error = "prepared diffusion batch does not match the training graph";
        return false;
    }

    ggml_backend_tensor_set(
        training.input_latents, batch.input_latents.data(), 0, batch.input_latents.size() * sizeof(float));
    ggml_backend_tensor_set(
        training.encoder_hidden, batch.encoder_hidden.data(), 0, batch.encoder_hidden.size() * sizeof(float));
    ggml_backend_tensor_set(training.target_velocity,
                            batch.target_velocity.data(),
                            0,
                            batch.target_velocity.size() * sizeof(float));
    ggml_backend_tensor_set(
        training.loss_weights, batch.loss_weights.data(), 0, batch.loss_weights.size() * sizeof(float));
    ggml_backend_tensor_set(
        training.timestep, batch.timesteps.data(), 0, batch.timesteps.size() * sizeof(float));
    ggml_backend_tensor_set(training.reference_timestep,
                            batch.reference_timesteps.data(),
                            0,
                            batch.reference_timesteps.size() * sizeof(float));
    ggml_backend_tensor_set(
        training.positions, batch.positions.data(), 0, batch.positions.size() * sizeof(int32_t));
    ggml_backend_tensor_set(training.self_attention_mask,
                            batch.self_attention_mask.data(),
                            0,
                            batch.self_attention_mask.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(training.cross_attention_mask,
                            batch.cross_attention_mask.data(),
                            0,
                            batch.cross_attention_mask.size() * sizeof(ggml_fp16_t));
    return true;
}

static bool ace_compute_train_adapter_gradients(ACETrainDiTGraph &             training,
                                                const ACETrainDiffusionBatch & batch,
                                                float &                        loss,
                                                std::string &                  error) {
    loss = 0.0f;
    if (!ace_upload_train_diffusion_batch(training, batch, error)) {
        return false;
    }
    ggml_graph_reset(training.graph);
    if (!training.backend) {
        error = "DiT training backend is not initialized";
        return false;
    }
    const enum ggml_status status = ggml_backend_graph_compute(training.backend, training.graph);
    if (status != GGML_STATUS_SUCCESS) {
        error = "DiT training graph execution failed";
        return false;
    }
    ggml_backend_tensor_get(training.loss, &loss, 0, sizeof(loss));
    if (!std::isfinite(loss)) {
        error = "DiT training loss is not finite";
        return false;
    }
    return true;
}

static bool ace_train_adapter_step(ACETrainDiTGraph &             training,
                                   ACETrainAdapterState &         state,
                                   ACETrainAdapterOptimizer &     optimizer,
                                   const ACETrainAdamWConfig &    optimizer_config,
                                   const ACETrainDiffusionBatch & batch,
                                   float &                        loss,
                                   std::string &                  error) {
    return ace_compute_train_adapter_gradients(training, batch, loss, error) &&
           ace_train_adapter_adamw_step(training, state, optimizer, optimizer_config, error);
}
