// Deterministic caption/genre substitutions, following Carey's rotating prompt
// policy in betweentwomidnights/gary-localhost-installer at f0ff85ae (data_module.py).
// Native RNG/shuffle order is retained; this is policy, not Python RNG parity.
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

static constexpr const char * ACE_TRAIN_PROMPT_ALGORITHM = "ace-rotating-prompts-v1";

// Each track remains in the ordinary epoch order exactly once. The optional
// examples-per-epoch subset is the sampling population, so ratios describe the
// tracks actually trained. Missing genres retain caption conditioning.
static std::vector<bool> ace_train_genre_selection(const std::vector<bool> &   eligible,
                                                   const std::vector<size_t> & order,
                                                   size_t                      count,
                                                   int                         ratio,
                                                   uint64_t                    seed,
                                                   int                         epoch) {
    std::vector<bool> selected(eligible.size(), false);
    if (ratio <= 0 || count == 0) {
        return selected;
    }
    std::vector<size_t> cycle;
    for (size_t i = 0; i < count; ++i) {
        if (eligible.at(order.at(i))) {
            cycle.push_back(order[i]);
        }
    }
    // Stable eligibility order makes the all-tracks cycle independent of the
    // epoch shuffle; std::shuffle intentionally retains the native RNG policy.
    std::sort(cycle.begin(), cycle.end());
    std::mt19937_64 generator(seed + 10000);
    std::shuffle(cycle.begin(), cycle.end(), generator);
    const auto draws = [&](uint64_t e) {
        const uint64_t tracks = e * uint64_t(count);
        return (tracks / 100) * uint64_t(ratio) + ((tracks % 100) * uint64_t(ratio) + 50) / 100;
    };
    const uint64_t first = draws(uint64_t(epoch));
    const size_t   n     = std::min(cycle.size(), size_t(draws(uint64_t(epoch) + 1) - first));
    for (size_t j = 0; j < n; ++j) {
        selected[cycle[(first + j) % cycle.size()]] = true;
    }
    return selected;
}
