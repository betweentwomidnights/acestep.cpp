// ABOUTME: Discovers paired audio and metadata files for native ACE-Step adapter training.
// ABOUTME: Converts the dataset text contract into structured music and lyric conditioning fields.

#pragma once

#include <charconv>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <vector>

struct ACETrainingMetadata {
    std::string caption;
    std::string genre;
    int         bpm = 0;
    std::string key;
    int         time_signature  = 0;
    bool        is_instrumental = false;
    std::string custom_tag;
    std::string lyrics;
};

struct ACETrainingExample {
    std::string           name;
    std::filesystem::path audio_path;
    std::filesystem::path metadata_path;
    ACETrainingMetadata   metadata;
};

static std::string ace_training_trim(const std::string & value) {
    const size_t first = value.find_first_not_of(" \t\r");
    if (first == std::string::npos) {
        return {};
    }
    const size_t last = value.find_last_not_of(" \t\r");
    return value.substr(first, last - first + 1);
}

static bool ace_training_read_integer(const std::string & value, int & result) {
    const std::string trimmed = ace_training_trim(value);
    if (trimmed.empty()) {
        return false;
    }
    const char * first  = trimmed.data();
    const char * last   = first + trimmed.size();
    const auto   parsed = std::from_chars(first, last, result);
    return parsed.ec == std::errc{} && parsed.ptr == last;
}

static bool ace_training_read_metadata(const std::filesystem::path & path,
                                       ACETrainingMetadata &         metadata,
                                       std::string &                 error) {
    metadata = {};
    error.clear();

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "could not open metadata file: " + path.string();
        return false;
    }
    const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (!input.good() && !input.eof()) {
        error = "could not read metadata file: " + path.string();
        return false;
    }

    bool   has_caption        = false;
    bool   has_genre          = false;
    bool   has_bpm            = false;
    bool   has_key            = false;
    bool   has_time_signature = false;
    bool   has_instrumental   = false;
    bool   has_custom_tag     = false;
    bool   has_lyrics         = false;
    size_t offset             = 0;
    while (offset <= contents.size()) {
        const size_t newline = contents.find('\n', offset);
        const size_t end     = newline == std::string::npos ? contents.size() : newline;
        std::string  line    = contents.substr(offset, end - offset);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (has_lyrics) {
            if (!metadata.lyrics.empty()) {
                metadata.lyrics.push_back('\n');
            }
            metadata.lyrics += line;
        } else {
            const size_t colon = line.find(':');
            if (colon == std::string::npos) {
                if (!ace_training_trim(line).empty()) {
                    error = "invalid metadata line in " + path.string() + ": " + line;
                    return false;
                }
            } else {
                const std::string field = ace_training_trim(line.substr(0, colon));
                const std::string value = ace_training_trim(line.substr(colon + 1));
                if (field == "caption") {
                    metadata.caption = value;
                    has_caption      = true;
                } else if (field == "genre") {
                    metadata.genre = value;
                    has_genre      = true;
                } else if (field == "bpm") {
                    if (!ace_training_read_integer(value, metadata.bpm)) {
                        error = "invalid bpm in metadata file: " + path.string();
                        return false;
                    }
                    has_bpm = true;
                } else if (field == "key") {
                    metadata.key = value;
                    has_key      = true;
                } else if (field == "signature") {
                    if (!ace_training_read_integer(value, metadata.time_signature)) {
                        error = "invalid signature in metadata file: " + path.string();
                        return false;
                    }
                    has_time_signature = true;
                } else if (field == "is_instrumental") {
                    if (value == "true") {
                        metadata.is_instrumental = true;
                    } else if (value == "false") {
                        metadata.is_instrumental = false;
                    } else {
                        error = "invalid is_instrumental value in metadata file: " + path.string();
                        return false;
                    }
                    has_instrumental = true;
                } else if (field == "custom_tag") {
                    metadata.custom_tag = value;
                    has_custom_tag      = true;
                } else if (field == "lyrics") {
                    metadata.lyrics = value;
                    has_lyrics      = true;
                }
            }
        }

        if (newline == std::string::npos) {
            break;
        }
        offset = newline + 1;
    }

    while (!metadata.lyrics.empty() && metadata.lyrics.back() == '\n') {
        metadata.lyrics.pop_back();
    }
    if (!has_caption || !has_genre || !has_bpm || !has_key || !has_time_signature || !has_instrumental ||
        !has_custom_tag || !has_lyrics || metadata.caption.empty() || metadata.bpm <= 0 ||
        metadata.time_signature <= 0 || (!metadata.is_instrumental && metadata.lyrics.empty())) {
        error = "metadata file is missing required ACE-Step conditioning fields: " + path.string();
        return false;
    }
    return true;
}

static bool ace_training_load_dataset(const std::filesystem::path &     directory,
                                      std::vector<ACETrainingExample> & examples,
                                      std::string &                     error) {
    examples.clear();
    error.clear();

    std::error_code status_error;
    if (!std::filesystem::is_directory(directory, status_error) || status_error) {
        error = "training dataset is not a readable directory: " + directory.string();
        return false;
    }

    struct Pair {
        std::filesystem::path audio_path;
        std::filesystem::path metadata_path;
    };

    std::map<std::string, Pair> pairs;
    std::error_code             iterator_error;
    for (std::filesystem::directory_iterator iterator(directory, iterator_error), end; iterator != end;
         iterator.increment(iterator_error)) {
        if (iterator_error) {
            error = "could not enumerate training dataset: " + directory.string();
            return false;
        }
        std::error_code entry_error;
        if (!iterator->is_regular_file(entry_error) || entry_error) {
            continue;
        }
        const std::filesystem::path & path      = iterator->path();
        const std::string             extension = path.extension().string();
        if (extension == ".wav") {
            pairs[path.stem().string()].audio_path = path;
        } else if (extension == ".txt") {
            pairs[path.stem().string()].metadata_path = path;
        }
    }
    if (iterator_error) {
        error = "could not enumerate training dataset: " + directory.string();
        return false;
    }
    if (pairs.empty()) {
        error = "training dataset contains no WAV and metadata pairs: " + directory.string();
        return false;
    }

    examples.reserve(pairs.size());
    for (const auto & entry : pairs) {
        const std::string & name = entry.first;
        const Pair &        pair = entry.second;
        if (pair.audio_path.empty()) {
            error = "metadata file has no matching audio file: " + pair.metadata_path.filename().string();
            return false;
        }
        if (pair.metadata_path.empty()) {
            error = "audio file has no matching metadata file: " + pair.audio_path.filename().string();
            return false;
        }

        ACETrainingExample example;
        example.name          = name;
        example.audio_path    = pair.audio_path;
        example.metadata_path = pair.metadata_path;
        if (!ace_training_read_metadata(pair.metadata_path, example.metadata, error)) {
            return false;
        }
        examples.push_back(std::move(example));
    }
    return true;
}
