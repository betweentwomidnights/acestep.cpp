#pragma once
// genre-vocab.h: ACE-Step's genre vocabulary (vendor/ace-step/genres_vocab.txt).
//
// 178k genre strings, the vocabulary the reference LM's understand pass
// constrains its "genres:" field to. Kept independent of the LM so any genre
// source can be checked or normalized against the same list. Entries are
// lowercased (ASCII) and sorted, so prefix questions are binary searches over
// one vector instead of a pointer-heavy trie.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

static std::string genre_lower(std::string text) {
    for (char & ch : text) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = (char) (ch - 'A' + 'a');
        }
    }
    return text;
}

struct GenreVocab {
    std::vector<std::string> entries;  // lowercased, sorted, unique

    // Load one genre per line; blank lines and "#" comments are skipped.
    static std::shared_ptr<const GenreVocab> load(const std::filesystem::path & path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return nullptr;
        }
        auto        vocab = std::make_shared<GenreVocab>();
        std::string line;
        while (std::getline(input, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
            size_t start = line.find_first_not_of(" \t");
            if (start == std::string::npos || line[start] == '#') {
                continue;
            }
            vocab->entries.push_back(genre_lower(line.substr(start)));
        }
        std::sort(vocab->entries.begin(), vocab->entries.end());
        vocab->entries.erase(std::unique(vocab->entries.begin(), vocab->entries.end()), vocab->entries.end());
        if (vocab->entries.empty()) {
            return nullptr;
        }
        return vocab;
    }

    // The first entry at or after `prefix`; entries starting with it follow.
    std::vector<std::string>::const_iterator lower(const std::string & prefix) const {
        return std::lower_bound(entries.begin(), entries.end(), prefix);
    }

    // Whether some genre starts with `prefix` (the empty prefix included).
    bool has_prefix(const std::string & prefix) const {
        auto it = lower(prefix);
        return it != entries.end() && it->compare(0, prefix.size(), prefix) == 0;
    }

    bool contains(const std::string & genre) const {
        auto it = lower(genre);
        return it != entries.end() && *it == genre;
    }

    // The distinct bytes that can follow `prefix` in some genre, ascending,
    // and whether `prefix` is itself a complete genre.
    void next_bytes(const std::string & prefix, std::vector<unsigned char> & out, bool & complete) const {
        out.clear();
        complete       = false;
        auto       it  = lower(prefix);
        const auto end = entries.end();
        while (it != end && it->compare(0, prefix.size(), prefix) == 0) {
            if (it->size() == prefix.size()) {
                complete = true;
                ++it;
                continue;
            }
            unsigned char next = (unsigned char) (*it)[prefix.size()];
            out.push_back(next);
            if (next == 0xff) {
                break;
            }
            // Skip every entry sharing this byte: jump to the next byte's range.
            std::string bound = prefix;
            bound.push_back((char) (next + 1));
            it = std::lower_bound(it, end, bound);
        }
    }
};

// The process-wide vocabulary: ACE_GENRES_VOCAB when set, else
// genres_vocab.txt beside the executable. Null when neither loads, so callers
// skip genre generation rather than fail. Loaded once (genre-vocab.cpp).
std::shared_ptr<const GenreVocab> genre_vocab();
