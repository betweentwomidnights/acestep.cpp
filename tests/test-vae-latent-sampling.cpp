// ABOUTME: Tests deterministic sampling from the ACE-Step VAE encoder's diagonal Gaussian output.
// ABOUTME: Verifies mean-only inference extraction and training posterior standard deviations.

#include "vae-enc.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

int main() {
    // Every sub-frame tail around representative tile
    // boundaries. Retained cores must cover each complete frame exactly once.
    for (const auto & geometry : {
             std::pair<int, int>{ 375,  50 },
              { 1024, 64 },
              { 256,  64 },
              { 8,    0  }
    }) {
        const int chunk = geometry.first * 1920, overlap = geometry.second * 1920;
        const int stride = chunk - 2 * overlap;
        for (int tail = 0; tail < 1920; ++tail) {
            for (int boundary : { 0, 1, geometry.second - 1, geometry.second, geometry.second + 1 }) {
                if (boundary < 0) {
                    continue;
                }
                const int samples = 3 * stride + boundary * 1920 + tail;
                int       written = 0;
                for (int start = 0; start < samples; start += stride) {
                    const int end       = std::min(samples, start + stride);
                    const int win_start = std::max(0, start - overlap);
                    const int win_end   = std::min(samples, end + overlap);
                    int       trim = 0, count = 0;
                    vae_enc_core_bounds((win_end - win_start) / 1920, start, end, win_start, trim, count);
                    if (written != start / 1920 || trim != (start - win_start) / 1920 ||
                        count != end / 1920 - start / 1920) {
                        return fail("tiled VAE loses or duplicates a complete core frame");
                    }
                    written += count;
                }
                if (written != samples / 1920) {
                    return fail("tiled VAE frame count differs from untiled downsampling");
                }
            }
        }
    }
    if (std::fabs(vae_enc_standard_deviation(0.0f) - (std::log(2.0f) + 1e-4f)) > 1e-6f ||
        std::fabs(vae_enc_standard_deviation(100.0f) - 100.0001f) > 1e-4f ||
        std::fabs(vae_enc_standard_deviation(-100.0f) - 1e-4f) > 1e-7f) {
        return fail("VAE posterior scale must use numerically stable softplus plus epsilon");
    }

    const int          temporal_length = 2;
    std::vector<float> raw((size_t) 128 * temporal_length);
    for (int channel = 0; channel < 64; ++channel) {
        for (int time = 0; time < temporal_length; ++time) {
            raw[(size_t) channel * temporal_length + time]        = (float) (channel * 10 + time);
            raw[(size_t) (64 + channel) * temporal_length + time] = -100.0f;
        }
    }

    std::vector<float> means((size_t) 64 * temporal_length);
    vae_enc_extract_latents(raw.data(), temporal_length, 0, temporal_length, means.data(), 0, nullptr);
    for (int time = 0; time < temporal_length; ++time) {
        for (int channel = 0; channel < 64; ++channel) {
            if (means[(size_t) time * 64 + channel] != (float) (channel * 10 + time)) {
                return fail("mean-only VAE extraction changed the encoder channel layout");
            }
        }
    }

    VAEEncLatentSampler first(42);
    VAEEncLatentSampler replay(42);
    VAEEncLatentSampler different(43);
    std::vector<float>  sampled_a(means.size());
    std::vector<float>  sampled_b(means.size());
    std::vector<float>  sampled_c(means.size());
    vae_enc_extract_latents(raw.data(), temporal_length, 0, temporal_length, sampled_a.data(), 0, &first);
    vae_enc_extract_latents(raw.data(), temporal_length, 0, temporal_length, sampled_b.data(), 0, &replay);
    vae_enc_extract_latents(raw.data(), temporal_length, 0, temporal_length, sampled_c.data(), 0, &different);
    if (sampled_a != sampled_b || sampled_a == sampled_c || sampled_a == means) {
        return fail("VAE posterior sampling must be seed-deterministic and seed-sensitive");
    }
    for (float value : sampled_a) {
        if (!std::isfinite(value)) {
            return fail("VAE posterior sampling produced a non-finite latent");
        }
    }

    // Explicit shared draws must follow retained frame/channel order, even
    // when tiled extraction trims overlap and appends to an existing output.
    std::vector<float>  noise(64, 0.5f), shared(128, -1.0f);
    VAEEncLatentSampler controlled(42);
    controlled.shared_noise      = noise.data();
    controlled.shared_noise_size = noise.size();
    for (int channel = 0; channel < 64; ++channel) {
        raw[(size_t) (64 + channel) * temporal_length + 1] = 0.0f;
    }
    vae_enc_extract_latents(raw.data(), temporal_length, 1, 1, shared.data(), 1, &controlled);
    if (controlled.shared_noise_offset != 64) {
        return fail("shared posterior noise cursor does not match retained frames");
    }
    for (int channel = 0; channel < 64; ++channel) {
        const float expected = float(channel * 10 + 1) + vae_enc_standard_deviation(0.0f) * 0.5f;
        if (shared[channel] != -1.0f || std::fabs(shared[64 + channel] - expected) > 1e-5f) {
            return fail("shared posterior noise changed trimming, layout or standard deviation math");
        }
    }

    std::puts("VAE training posterior sampling: OK");
    return 0;
}
