// ABOUTME: Verifies the genre vocabulary's loading, prefix lookups and next-byte enumeration.
// ABOUTME: The understand pass constrains its genres field with exactly these queries.

#include "genre-vocab.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>

static int fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

int main(int argc, char ** argv) {
    const auto path = std::filesystem::temp_directory_path() /
                      ("ace-genre-vocab-" + std::to_string((unsigned long long) std::random_device{}()) + ".txt");
    {
        std::ofstream out(path, std::ios::binary);
        out << "# comment\r\nPop\r\npop rock\r\n\r\npop, rock, jazz\r\n  Rock  \r\nrock\r\n";
    }
    auto vocab = GenreVocab::load(path);
    std::filesystem::remove(path);
    if (!vocab) {
        return fail("the fixture vocabulary did not load");
    }
    const std::vector<std::string> expected = { "pop", "pop rock", "pop, rock, jazz", "rock" };
    if (vocab->entries != expected) {
        return fail("entries must be lowercased, trimmed, sorted and unique, without comments");
    }
    if (!vocab->has_prefix("") || !vocab->has_prefix("po") || !vocab->has_prefix("pop, r") ||
        vocab->has_prefix("pox") || vocab->has_prefix("rocks")) {
        return fail("prefix lookups");
    }
    if (!vocab->contains("pop") || vocab->contains("po") || !vocab->contains("rock")) {
        return fail("complete-entry lookups");
    }
    std::vector<unsigned char> next;
    bool                       complete = false;
    vocab->next_bytes("", next, complete);
    if (next != std::vector<unsigned char>{ 'p', 'r' } || complete) {
        return fail("the root offers each first byte once and is not a genre");
    }
    vocab->next_bytes("pop", next, complete);
    if (next != std::vector<unsigned char>{ ' ', ',' } || !complete) {
        return fail("\"pop\" is complete and continues with a space or a comma");
    }
    vocab->next_bytes("rock", next, complete);
    if (!next.empty() || !complete) {
        return fail("\"rock\" is complete and ends there");
    }
    vocab->next_bytes("jazz", next, complete);
    if (!next.empty() || complete) {
        return fail("an unknown prefix offers nothing");
    }

    // The vendored vocabulary itself, when the build passes its path.
    if (argc > 1) {
        auto real = GenreVocab::load(argv[1]);
        if (!real || real->entries.size() < 170000 || !real->contains("16-bit jrpg") || !real->has_prefix("lo-fi")) {
            return fail("the vendored ACE-Step genre vocabulary did not load as expected");
        }
        std::printf("ACE-Step genre vocabulary: %zu genres\n", real->entries.size());
    }
    std::puts("ACE-Step genre vocabulary contract: OK");
    return 0;
}
