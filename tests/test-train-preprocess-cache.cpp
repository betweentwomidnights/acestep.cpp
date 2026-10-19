#include "train-preprocess-cache.h"

static void expect(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    namespace fs         = std::filesystem;
    const auto root      = fs::temp_directory_path() / ("ace-cache-" + std::to_string(std::random_device{}()));
    auto       configure = [&](ACETrainPreparedDataset & data) {
        data.directory  = root / fs::u8path("tensors-\xC3\xA9");
        data.identities = { "models/audio/metadata/seed/device:one", "models/audio/metadata/seed/device:two" };
        data.output_channels  = 2;
        data.context_channels = 3;
        data.patch_size       = 2;
        data.hidden_size      = 4;
        data.max_latents      = 64;
        data.null_condition   = { 10, 20, 30, 40 };
        data.memory_limit     = 144;
    };
    ACETrainPreparedDataset data, stream;
    configure(data);
    configure(stream);
    ACETrainDiffusionExample a, b, loaded;
    a.target_latents = { 1, 2, 3, 4, 5, 6, 0, 0 };
    a.context_latents.assign(12, 0.5f);
    a.encoder_hidden               = { 1, 2, 3, 4, 5, 6, 7, 8 };
    a.real_temporal_length         = 3;
    a.real_encoder_sequence_length = 2;
    b                              = a;
    b.real_encoder_sequence_length = 4;
    b.encoder_hidden.assign(16, 0.25f);
    std::string error;
    expect(data.write(0, a, error) && data.write(1, b, error), "atomic cache writes");
    expect(data.read_raw(0, loaded, error) && loaded.target_latents == a.target_latents &&
               loaded.context_latents == a.context_latents && loaded.encoder_hidden == a.encoder_hidden &&
               loaded.real_temporal_length == 3,
           "lossless tensor/length round trip");
    data.observe(a);
    data.observe(b);
    auto first = data.get(0, error);
    expect(first && first->encoder_hidden.size() == 16 && first->encoder_hidden[8] == 10 &&
               first->encoder_hidden[15] == 40 && first->real_encoder_sequence_length == 2,
           "dataset-wide null padding");
    expect(data.get(0, error) == first, "LRU storage reused");
    auto second = data.get(1, error);
    expect(second && data.resident_bytes() == 144 && data.resident_bytes() <= data.memory_limit,
           "byte budget enforced");
    expect(data.get(0, error) != first, "least recently used entry evicted");
    stream.max_encoder_length = 4;
    stream.memory_limit       = 0;
    expect(bool(stream.get(0, error)) && stream.resident_bytes() == 0, "zero budget streams");
    stream.memory_limit = 1;
    expect(bool(stream.get(1, error)) && stream.resident_bytes() == 0, "oversized example bypasses residency");
    stream.identities[0] = "different identity";
    fs::copy_file(data.path(0), stream.path(0));
    expect(!stream.read_raw(0, loaded, error), "copied wrong-identity entry rejected");
    stream.identities = data.identities;
    {
        std::fstream corrupt(data.path(0), std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekg(-1, std::ios::end);
        char value = 0;
        corrupt.read(&value, 1);
        value ^= 1;
        corrupt.seekp(-1, std::ios::end);
        corrupt.write(&value, 1);
    }
    expect(!stream.read_raw(0, loaded, error), "checksum corruption rejected");
    expect(data.write(0, a, error) && stream.read_raw(0, loaded, error), "atomic repair");
    fs::resize_file(data.path(0), 20);
    expect(!stream.read_raw(0, loaded, error), "truncation rejected");
    expect(data.write(0, a, error), "rewrite complete entry");
    {
        std::fstream corrupt(data.path(0), std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekp(8);
        char bytes[8];
        std::memset(bytes, 255, 8);
        corrupt.write(bytes, 8);
    }
    expect(!stream.read_raw(0, loaded, error), "malformed header bounded before allocation");
    auto invalid              = a;
    invalid.target_latents[0] = NAN;
    expect(!data.write(0, invalid, error), "nonfinite tensors rejected");
    expect(data.write(0, a, error), "publish valid replacement");
    // Padding a nearly full vector often grows capacity geometrically. Logical
    // shape bytes must not undercount that allocation against the LRU budget.
    ACETrainPreparedDataset allocated;
    configure(allocated);
    allocated.max_encoder_length           = 4;
    auto near_full                         = a;
    near_full.real_encoder_sequence_length = 3;
    near_full.encoder_hidden.assign(12, 0.75f);
    expect(allocated.write(0, near_full, error), "prepare vector-growth fixture");
    auto grown = allocated.get(0, error);
    expect(bool(grown), "read vector-growth fixture");
    const size_t actual_bytes =
        (grown->target_latents.capacity() + grown->context_latents.capacity() + grown->encoder_hidden.capacity()) *
        sizeof(float);
    expect(allocated.resident_bytes() == (actual_bytes <= allocated.memory_limit ? actual_bytes : 0),
           "budget accounts for allocated capacity after padding");
    expect(data.write(0, a, error), "restore original entry");
    {
        ACETrainPreparedDataset variants;
        configure(variants);
        variants.genre_variants = true;
        variants.identities[0] += "/caption-and-genre";
        auto mixed                          = a;
        mixed.genre_encoder_sequence_length = 3;
        mixed.genre_encoder_hidden.assign(12, 0.125f);
        variants.observe(mixed);
        variants.observe(b);
        expect(variants.write(0, mixed, error) && variants.read_raw(0, loaded, error) &&
                   loaded.genre_encoder_hidden == mixed.genre_encoder_hidden &&
                   loaded.target_latents == a.target_latents && loaded.genre_encoder_sequence_length == 3,
               "both prompts round trip with shared audio");
        auto padded = variants.get(0, error);
        expect(padded && padded->genre_encoder_hidden.size() == 16 && padded->genre_encoder_hidden[12] == 10 &&
                   variants.resident_bytes() == 0,
               "genre padding and both conditioning allocations count toward budget");
        mixed.genre_encoder_sequence_length = 4;
        expect(!variants.write(0, mixed, error), "invalid genre length rejected");
        mixed = a;
        expect(variants.write(1, mixed, error) && variants.read_raw(1, loaded, error) &&
                   loaded.genre_encoder_hidden.empty(),
               "missing genre remains caption-only in variant cache");
    }
#ifdef _WIN32
    HANDLE lock = CreateFileW(data.path(0).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    expect(lock != INVALID_HANDLE_VALUE, "simulate persistent reader lock");
    expect(!data.write(0, b, error), "publication failure reported");
    CloseHandle(lock);
    expect(stream.read_raw(0, loaded, error) && loaded.encoder_hidden == a.encoder_hidden, "previous entry preserved");
#endif
    for (const auto & item : fs::directory_iterator(data.directory)) {
        expect(item.path().u8string().find(".tmp-") == std::string::npos, "staging files cleaned");
    }
    fs::remove_all(root);
    std::puts("PASS cache integrity, atomic publication, padding and bounded LRU");
}
