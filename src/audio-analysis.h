#pragma once
// Native sidecar suggestions ported from betweentwomidnights/sa3.cpp,
// commit 3b7bfbf2939983ff3411dea92e5357f04a0ec6ad (MIT).
// Keep these estimators independent of ggml, models and the inference WAV reader.
#include "vendor/sa3-analysis/audio_analysis.h"
#include "yyjson.h"

namespace ace_analysis {
using Result = sa3::analysis::Result;

// Bounded in-memory decoder for HTTP uploads. The vendored file reader is used
// by the standalone CLI; both feed the identical original-rate mono estimator.
inline Result analyze_wav_buffer(const std::string &  bytes,
                                 double &             duration,
                                 std::vector<float> * stereo = nullptr,
                                 int *                rate   = nullptr) {
    using namespace sa3::wav_detail;
    const auto * data = reinterpret_cast<const uint8_t *>(bytes.data());
    const size_t size = bytes.size();
    if (size < 12 || memcmp(data, "RIFF", 4) || memcmp(data + 8, "WAVE", 4)) {
        throw std::invalid_argument("analysis requires a PCM or float WAV upload");
    }
    uint16_t        format = 0, channels = 0, bits = 0;
    uint32_t        sample_rate = 0;
    const uint8_t * audio       = nullptr;
    size_t          audio_size  = 0;
    for (size_t pos = 12; pos + 8 <= size;) {
        const uint32_t count = u32le(data + pos + 4);
        const auto *   id    = data + pos;
        pos += 8;
        if (count > size - pos) {
            throw std::invalid_argument("truncated WAV chunk");
        }
        if (!memcmp(id, "fmt ", 4)) {
            if (count < 16) {
                throw std::invalid_argument("invalid WAV format chunk");
            }
            format      = u16le(data + pos);
            channels    = u16le(data + pos + 2);
            sample_rate = u32le(data + pos + 4);
            bits        = u16le(data + pos + 14);
            if (format == 0xfffe && count >= 40) {
                format = u16le(data + pos + 24);
            }
        } else if (!memcmp(id, "data", 4)) {
            audio      = data + pos;
            audio_size = count;
        }
        pos += size_t(count) + (count & 1);
    }
    const bool pcm      = format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32);
    const bool floating = format == 3 && (bits == 32 || bits == 64);
    if ((!pcm && !floating) || !audio || channels < 1 || channels > 64 || sample_rate < 1000 || sample_rate > 384000) {
        throw std::invalid_argument("unsupported WAV format, channel count or sample rate");
    }
    const size_t width = bits / 8, frame = width * channels;
    if (!audio_size || audio_size % frame) {
        throw std::invalid_argument("empty or incomplete WAV frames");
    }
    const size_t count = audio_size / frame;
    duration           = double(count) / sample_rate;
    if (duration > 600) {
        throw std::length_error("audio exceeds max duration (10 min)");
    }
    std::vector<float> mono(count);
    if (stereo) {
        stereo->resize(count * 2);
    }
    if (rate) {
        *rate = int(sample_rate);
    }
    for (size_t i = 0; i < count; ++i) {
        double sum = 0;
        for (size_t ch = 0; ch < channels; ++ch) {
            const auto * p = audio + (i * channels + ch) * width;
            double       value;
            if (floating && bits == 64) {
                memcpy(&value, p, sizeof(double));
            } else if (floating) {
                float sample;
                memcpy(&sample, p, sizeof(float));
                value = sample;
            } else if (bits == 8) {
                value = (int(p[0]) - 128) / 128.0;
            } else if (bits == 16) {
                value = int16_t(u16le(p)) / 32768.0;
            } else if (bits == 24) {
                value = s24le(p) / 8388608.0;
            } else {
                value = s32le(p) / 2147483648.0;
            }
            if (!std::isfinite(value)) {
                throw std::invalid_argument("audio contains non-finite samples");
            }
            sum += value;
            if (stereo && ch < 2) {
                (*stereo)[ch * count + i] = float(std::max(-1.0, std::min(1.0, value)));
                if (channels == 1) {
                    (*stereo)[count + i] = (*stereo)[i];
                }
            }
        }
        mono[i] = float(sum / channels);
    }
    return sa3::analysis::analyze(mono, int(sample_rate));
}

inline yyjson_mut_val * result_json(yyjson_mut_doc * doc, const Result & result) {
    auto * obj = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_int(doc, obj, "schema_version", 1);
    yyjson_mut_obj_add_bool(doc, obj, "ok", true);
    if (result.bpm) {
        yyjson_mut_obj_add_int(doc, obj, "bpm", *result.bpm);
    } else {
        yyjson_mut_obj_add_null(doc, obj, "bpm");
    }
    yyjson_mut_obj_add_strcpy(doc, obj, "keyscale", result.keyscale.c_str());
    std::string suggestion;
    if (result.bpm) {
        suggestion = std::to_string(*result.bpm) + " bpm";
    }
    if (!result.keyscale.empty()) {
        suggestion += (suggestion.empty() ? "" : ", ") + result.keyscale;
    }
    yyjson_mut_obj_add_strcpy(doc, obj, "suggestion", suggestion.c_str());
    yyjson_mut_obj_add_str(doc, obj, "bpm_source", result.bpm ? "local" : "missing");
    yyjson_mut_obj_add_str(doc, obj, "key_source", result.keyscale.empty() ? "missing" : "local");
    auto confidence = [&](const char * key, const std::optional<double> & value) {
        if (value) {
            yyjson_mut_obj_add_real(doc, obj, key, std::nearbyint(*value * 10000) / 10000);
        } else {
            yyjson_mut_obj_add_null(doc, obj, key);
        }
    };
    confidence("bpm_confidence", result.bpm_confidence);
    confidence("key_confidence", result.key_confidence);
    return obj;
}
}  // namespace ace_analysis
