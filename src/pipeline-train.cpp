// ABOUTME: Prepares real audio and metadata through ACE-Step's VAE and conditioning models.
// ABOUTME: Produces padded training tensors while retaining exact temporal and text lengths.

#include "pipeline-train.h"

#include "audio-io.h"
#include "pipeline-synth-impl.h"
#include "pipeline-synth-ops.h"
#include "task-types.h"
#include "vae-enc.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

static std::string ace_training_caption(const ACETrainingMetadata & metadata,
                                        const ACETrainPreprocessConfig & config) {
    std::string caption = config.use_genre && !metadata.genre.empty() ? metadata.genre : metadata.caption;
    if (metadata.custom_tag.empty()) {
        return caption;
    }
    if (config.tag_position == ACE_TRAIN_TAG_REPLACE) {
        return metadata.custom_tag;
    }
    if (config.tag_position == ACE_TRAIN_TAG_APPEND) {
        return caption.empty() ? metadata.custom_tag : caption + ", " + metadata.custom_tag;
    }
    return caption.empty() ? metadata.custom_tag : metadata.custom_tag + ", " + caption;
}

AceRequest ace_training_request(const ACETrainingExample & example,
                                float                      duration,
                                const ACETrainPreprocessConfig & config) {
    AceRequest request;
    request.caption         = ace_training_caption(example.metadata, config);
    request.lyrics          = example.metadata.lyrics;
    request.bpm             = example.metadata.bpm;
    request.duration        = duration;
    request.keyscale        = example.metadata.key;
    request.timesignature   = std::to_string(example.metadata.time_signature);
    request.vocal_language  = "unknown";
    request.task_type       = TASK_TEXT2MUSIC;
    if (request.lyrics.empty() && example.metadata.is_instrumental) {
        request.lyrics = "[Instrumental]";
    }
    return request;
}

bool ace_training_null_condition(const AceSynth * synth, std::vector<float> & condition, std::string & error) {
    condition.clear();
    error.clear();
    if (!synth || !synth->meta || synth->meta->null_cond_cpu.empty()) {
        error = "DiT model does not expose a null conditioning vector";
        return false;
    }
    condition = synth->meta->null_cond_cpu;
    return true;
}

bool ace_validate_training_audio(const std::string & path, int & samples, std::string & error) {
    samples = 0;
    error.clear();
    float * audio = audio_read_48k(path.c_str(), &samples);
    if (!audio || samples <= 0) {
        free(audio);
        error = "could not decode dataset audio at 48 kHz: " + path;
        return false;
    }
    free(audio);
    return true;
}

bool ace_prepare_training_example(AceSynth *                       synth,
                                  const ACETrainingExample &       source,
                                  uint64_t                         seed,
                                  const ACETrainPreprocessConfig & config,
                                  ACETrainDiffusionExample &       prepared,
                                  std::string &                    error) {
    prepared = {};
    error.clear();
    if (!synth || !synth->store || !synth->meta || synth->Oc <= 0 || synth->ctx_ch < synth->Oc ||
        synth->meta->cfg.patch_size <= 0) {
        error = "invalid ACE-Step synthesis context for training preprocessing";
        return false;
    }

    int     audio_length = 0;
    float * planar = audio_read_48k(source.audio_path.string().c_str(), &audio_length);
    if (!planar || audio_length <= 0) {
        free(planar);
        error = "could not decode training audio at 48 kHz: " + source.audio_path.string();
        return false;
    }
    float * audio = audio_planar_to_interleaved(planar, audio_length);
    free(planar);
    if (!audio) {
        error = "could not allocate interleaved training audio: " + source.audio_path.string();
        return false;
    }

    const int maximum_latent_length = audio_length / 1920 + 64;
    std::vector<float> target((size_t) maximum_latent_length * synth->Oc);
    int real_temporal_length = -1;
    {
        VAEEncoder * encoder = store_require_vae_enc(synth->store, synth->vae_enc_key);
        if (!encoder) {
            free(audio);
            error = "could not load the VAE encoder for training preprocessing";
            return false;
        }
        ModelHandle encoder_guard(synth->store, encoder);
        real_temporal_length = vae_enc_encode_tiled_sampled(encoder,
                                                            audio,
                                                            audio_length,
                                                            target.data(),
                                                            maximum_latent_length,
                                                            seed,
                                                            synth->params.vae_chunk,
                                                            synth->params.vae_overlap);
    }
    free(audio);
    if (real_temporal_length <= 0) {
        error = "VAE training posterior encoding failed: " + source.audio_path.string();
        return false;
    }

    const int patch_size = synth->meta->cfg.patch_size;
    const int temporal_length =
        ((real_temporal_length + patch_size - 1) / patch_size) * patch_size;
    if ((size_t) temporal_length * synth->Oc > synth->meta->silence_full.size()) {
        error = "training audio exceeds the DiT model's maximum latent duration: " + source.audio_path.string();
        return false;
    }
    target.resize((size_t) temporal_length * synth->Oc, 0.0f);

    const float duration = (float) audio_length / 48000.0f;
    const AceRequest request = ace_training_request(source, duration, config);
    SynthState state = {};
    state.Oc                = synth->Oc;
    state.ctx_ch            = synth->ctx_ch;
    state.rr                = request;
    state.duration          = duration;
    state.instruction_str   = DIT_INSTR_TEXT2MUSIC;
    state.use_source_context = false;
    state.is_repaint        = false;
    state.is_lego_region    = false;
    debug_init(&state.dbg, synth->params.dump_dir);
    ops_encode_timbre(synth, nullptr, 0, nullptr, 0, state);
    if (ops_encode_text(synth, &request, 1, state) != 0 || state.per_enc_S.empty() || state.per_enc_S[0] <= 0 ||
        state.enc_hidden.empty()) {
        error = "ACE-Step text conditioning failed for training example: " + source.metadata_path.string();
        return false;
    }

    prepared.target_latents = std::move(target);
    prepared.context_latents.resize((size_t) temporal_length * synth->ctx_ch);
    for (int time = 0; time < temporal_length; ++time) {
        float * destination = prepared.context_latents.data() + (size_t) time * synth->ctx_ch;
        const float * silence = synth->meta->silence_full.data() + (size_t) time * synth->Oc;
        std::copy(silence, silence + synth->Oc, destination);
        std::fill(destination + synth->Oc, destination + synth->ctx_ch, 1.0f);
    }
    prepared.encoder_hidden = std::move(state.enc_hidden);
    prepared.real_temporal_length = real_temporal_length;
    prepared.real_encoder_sequence_length = state.per_enc_S[0];
    return true;
}
