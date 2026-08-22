// ABOUTME: Tests ACE-Step training dataset discovery against real temporary files.
// ABOUTME: Verifies structured metadata, multiline lyrics, stable ordering, and pair validation.

#include "pipeline-train.h"
#include "train-dataset.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

static int fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

static bool write_file(const std::filesystem::path & path, const std::string & contents) {
    std::ofstream output(path, std::ios::binary);
    output.write(contents.data(), (std::streamsize) contents.size());
    return output.good();
}

int main() {
    const auto directory = std::filesystem::temp_directory_path() /
                           ("ace-training-dataset-" + std::to_string((unsigned long long) std::random_device{}()));
    std::filesystem::create_directories(directory);

    const std::string metadata =
        "caption: sparse electronic pop\r\n"
        "genre: alternative pop\r\n"
        "bpm: 70\r\n"
        "key: E minor\r\n"
        "signature: 4\r\n"
        "is_instrumental: false\r\n"
        "custom_tag: reference_voice\r\n"
        "lyrics:\r\n"
        "[Verse]\r\n"
        "Synthetic test words only\r\n";

    if (!write_file(directory / "second.wav", "RIFFtest") || !write_file(directory / "second.txt", metadata) ||
        !write_file(directory / "first.wav", "RIFFtest") || !write_file(directory / "first.txt", metadata)) {
        std::filesystem::remove_all(directory);
        return fail("could not create the temporary training dataset");
    }

    std::vector<ACETrainingExample> examples;
    std::string                     error;
    if (!ace_training_load_dataset(directory, examples, error)) {
        std::filesystem::remove_all(directory);
        return fail(error.c_str());
    }
    if (examples.size() != 2 || examples[0].name != "first" || examples[1].name != "second") {
        std::filesystem::remove_all(directory);
        return fail("dataset examples must be returned in stable basename order");
    }

    const ACETrainingMetadata & parsed = examples[0].metadata;
    if (parsed.caption != "sparse electronic pop" || parsed.genre != "alternative pop" || parsed.bpm != 70 ||
        parsed.key != "E minor" || parsed.time_signature != 4 || parsed.is_instrumental ||
        parsed.custom_tag != "reference_voice" || parsed.lyrics != "[Verse]\nSynthetic test words only") {
        std::filesystem::remove_all(directory);
        return fail("metadata fields or multiline lyrics were not parsed exactly");
    }

    ACETrainPreprocessConfig preprocess;
    AceRequest               request = ace_training_request(examples[0], 42.5f, preprocess);
    if (request.caption != "reference_voice, sparse electronic pop" || request.lyrics != parsed.lyrics ||
        request.bpm != 70 || request.keyscale != "E minor" || request.timesignature != "4" ||
        request.duration != 42.5f || request.seed != -1 || request.lm_batch_size != 1 ||
        request.output_format != "mp3" || request.adapter_scale != 1.0f) {
        std::filesystem::remove_all(directory);
        return fail("default training request must match Gary's prepended-tag caption semantics");
    }
    preprocess.use_genre    = true;
    preprocess.tag_position = ACE_TRAIN_TAG_APPEND;
    request                 = ace_training_request(examples[0], 42.5f, preprocess);
    if (request.caption != "alternative pop, reference_voice") {
        std::filesystem::remove_all(directory);
        return fail("training request must support Gary's genre and appended-tag prompt mode");
    }

    if (!write_file(directory / "unpaired.wav", "RIFFtest") || ace_training_load_dataset(directory, examples, error) ||
        error.find("unpaired.wav") == std::string::npos) {
        std::filesystem::remove_all(directory);
        return fail("an unpaired audio file must fail with its filename in the error");
    }

    std::filesystem::remove_all(directory);
    std::puts("ACE-Step training dataset contract: OK");
    return 0;
}
