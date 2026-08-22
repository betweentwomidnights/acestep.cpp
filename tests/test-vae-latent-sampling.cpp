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

    std::puts("VAE training posterior sampling: OK");
    return 0;
}
