// Disk-backed prepared examples, following SA3's identity-checked pre-encode workflow.
// ACE caches full-song latents AND conditioning; windows remain a training operation.
#pragma once

#include "train-checkpoint.h"
#include "train-diffusion.h"

#include <climits>
#include <cmath>
#include <cstring>
#include <list>
#include <memory>

static uint64_t ace_cache_hash(const void * bytes, size_t size, uint64_t hash = 14695981039346656037ULL) {
    const auto * data = static_cast<const unsigned char *>(bytes);
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ data[i]) * 1099511628211ULL;
    }
    return hash;
}

static std::string ace_cache_name(const std::string & identity) {
    char name[32];
    std::snprintf(name, sizeof(name), "%016llx.acecache",
                  (unsigned long long) ace_cache_hash(identity.data(), identity.size()));
    return name;
}

// F32 cache payloads are local to this ABI/backend/build, never portable datasets.
// A complete identity lives inside each checksummed file, so even a hash collision
// cannot return a different example's tensors.
class ACETrainPreparedDataset {
  public:
    std::vector<ACETrainDiffusionExample> memory;  // Explicit uncached compatibility mode only.
    std::vector<std::string>              identities;
    std::filesystem::path                 directory;
    size_t                                memory_limit    = 256ULL * 1024 * 1024;
    int                                   output_channels = 0, context_channels = 0, patch_size = 0, hidden_size = 0;
    size_t                                max_latents        = 0;
    int                                   max_encoder_length = 0;
    std::vector<float>                    null_condition;
    size_t                                reused = 0, written = 0, rebuilt = 0;
    bool                                  genre_variants = false;
    std::vector<bool>                     genre_eligible;

    size_t size() const { return identities.empty() ? memory.size() : identities.size(); }

    bool empty() const { return size() == 0; }

    bool disk() const { return !identities.empty(); }

    size_t resident_bytes() const { return resident; }

    std::filesystem::path path(size_t index) const { return directory / ace_cache_name(identities.at(index)); }

    bool valid(const ACETrainDiffusionExample & example, bool unpadded) const {
        const auto & t = example.target_latents;
        const auto & c = example.context_latents;
        const auto & e = example.encoder_hidden;
        if (output_channels <= 0 || context_channels <= 0 || patch_size <= 0 || hidden_size <= 0 || t.empty() ||
            t.size() > max_latents || t.size() % size_t(output_channels) != 0 ||
            t.size() / size_t(output_channels) % size_t(patch_size) != 0 ||
            c.size() != t.size() / size_t(output_channels) * size_t(context_channels) ||
            example.real_temporal_length <= 0 ||
            size_t(example.real_temporal_length) > t.size() / size_t(output_channels) ||
            example.real_encoder_sequence_length <= 0 || example.real_encoder_sequence_length > 65536 || e.empty() ||
            e.size() % size_t(hidden_size) != 0 ||
            size_t(example.real_encoder_sequence_length) > e.size() / size_t(hidden_size) ||
            (unpadded && e.size() != size_t(example.real_encoder_sequence_length) * size_t(hidden_size))) {
            return false;
        }
        const auto & g  = example.genre_encoder_hidden;
        const int    gs = example.genre_encoder_sequence_length;
        if ((g.empty() && gs != 0) || (!g.empty() && (gs <= 0 || gs > 65536 || g.size() % size_t(hidden_size) != 0 ||
                                                      size_t(gs) > g.size() / size_t(hidden_size) ||
                                                      (unpadded && g.size() != size_t(gs) * size_t(hidden_size))))) {
            return false;
        }
        for (const auto * values : { &t, &c, &e, &g }) {
            for (float value : *values) {
                if (!std::isfinite(value)) {
                    return false;
                }
            }
        }
        return true;
    }

    // Missing/corrupt entries are misses during preparation. During training a
    // damaged entry is a hard error: GPU models must never be swapped to rebuild it.
    bool read_raw(size_t index, ACETrainDiffusionExample & result, std::string & error) const {
        result = {};
        error  = "missing, incompatible or corrupt preprocessing cache entry: " + path(index).u8string();
        std::error_code ec;
        const uintmax_t file_size = std::filesystem::file_size(path(index), ec);
        if (ec || file_size < 24) {
            return false;
        }
        std::ifstream input(path(index), std::ios::binary);
        char          magic[8] = {};
        input.read(magic, 8);
        uint64_t   header_size = 0;
        const bool variants    = !std::memcmp(magic, "ACETC002", 8);
        if (!input || (!variants && std::memcmp(magic, "ACETC001", 8)) || variants != genre_variants ||
            !read_u64(input, header_size) || header_size > 65536 || header_size > file_size - 24) {
            return false;
        }
        std::string header(size_t(header_size), '\0');
        input.read(header.data(), (std::streamsize) header.size());
        yyjson_doc * doc      = input ? yyjson_read(header.data(), header.size(), 0) : nullptr;
        auto *       root     = doc ? yyjson_doc_get_root(doc) : nullptr;
        auto *       key      = yyjson_obj_get(root, "identity");
        uint64_t     sizes[4] = {}, temporal = 0, sequence = 0, genre_sequence = 0;
        auto         number = [&](const char * name, uint64_t & value) {
            auto * v = yyjson_obj_get(root, name);
            if (!yyjson_is_int(v) || yyjson_get_sint(v) < 0) {
                return false;
            }
            value = yyjson_get_uint(v);
            return true;
        };
        bool ok = yyjson_is_str(key) && identities.at(index) == yyjson_get_str(key) && number("target", sizes[0]) &&
                  number("context", sizes[1]) && number("encoder", sizes[2]) &&
                  number("real_temporal_length", temporal) && number("real_encoder_sequence_length", sequence) &&
                  temporal > 0 && temporal <= INT_MAX && sequence > 0 && sequence <= 65536;
        if (variants) {
            ok = ok && number("genre_encoder", sizes[3]) && number("genre_sequence", genre_sequence) &&
                 genre_sequence <= 65536;
        }
        yyjson_doc_free(doc);
        uint64_t count = 0;
        for (uint64_t size : sizes) {
            if (size > (file_size - 24 - header_size) / sizeof(float) || size > SIZE_MAX / sizeof(float) ||
                count > UINT64_MAX - size) {
                ok = false;
                break;
            }
            count += size;
        }
        if (!ok || count != (file_size - 24 - header_size) / sizeof(float) ||
            count * sizeof(float) != file_size - 24 - header_size || sizes[0] > max_latents || output_channels <= 0 ||
            context_channels <= 0 || hidden_size <= 0 || sizes[0] % size_t(output_channels) ||
            sizes[1] != sizes[0] / size_t(output_channels) * size_t(context_channels) ||
            sizes[2] != sequence * size_t(hidden_size) || sizes[3] != genre_sequence * size_t(hidden_size)) {
            return false;
        }
        result.real_temporal_length          = int(temporal);
        result.real_encoder_sequence_length  = int(sequence);
        result.genre_encoder_sequence_length = int(genre_sequence);
        uint64_t hash                        = ace_cache_hash(header.data(), header.size());
        size_t   i                           = 0;
        for (auto * values : { &result.target_latents, &result.context_latents, &result.encoder_hidden,
                               &result.genre_encoder_hidden }) {
            values->resize(size_t(sizes[i++]));
            input.read(reinterpret_cast<char *>(values->data()), (std::streamsize)(values->size() * sizeof(float)));
            hash = ace_cache_hash(values->data(), values->size() * sizeof(float), hash);
        }
        uint64_t stored_hash = 0;
        if (!input || !read_u64(input, stored_hash) || stored_hash != hash || !valid(result, true)) {
            result = {};
            return false;
        }
        error.clear();
        return true;
    }

    bool write(size_t index, const ACETrainDiffusionExample & example, std::string & error) {
        error.clear();
        if (!valid(example, true)) {
            error = "cannot cache invalid/nonfinite prepared tensors";
            return false;
        }
        yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
        auto *           root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_strcpy(doc, root, "identity", identities.at(index).c_str());
        yyjson_mut_obj_add_uint(doc, root, "target", example.target_latents.size());
        yyjson_mut_obj_add_uint(doc, root, "context", example.context_latents.size());
        yyjson_mut_obj_add_uint(doc, root, "encoder", example.encoder_hidden.size());
        yyjson_mut_obj_add_int(doc, root, "real_temporal_length", example.real_temporal_length);
        yyjson_mut_obj_add_int(doc, root, "real_encoder_sequence_length", example.real_encoder_sequence_length);
        if (genre_variants) {
            yyjson_mut_obj_add_uint(doc, root, "genre_encoder", example.genre_encoder_hidden.size());
            yyjson_mut_obj_add_int(doc, root, "genre_sequence", example.genre_encoder_sequence_length);
        } else if (!example.genre_encoder_hidden.empty()) {
            yyjson_mut_doc_free(doc);
            error = "genre conditioning requires a variant cache identity";
            return false;
        }
        char * data = yyjson_mut_write(doc, 0, nullptr);
        yyjson_mut_doc_free(doc);
        if (!data) {
            error = "cannot serialize preprocessing cache header";
            return false;
        }
        std::string header(data);
        free(data);
        if (header.size() > 65536) {
            error = "preprocessing cache identity is too large";
            return false;
        }
        const auto destination = path(index);
        // Staging stays beside its destination; concurrent writers use distinct names.
        const auto stage =
            std::filesystem::u8path(destination.u8string() + ".tmp-" + std::to_string(std::random_device{}()) + "-" +
                                    std::to_string(std::random_device{}()));
        try {
            std::filesystem::create_directories(directory);
            std::ofstream output(stage, std::ios::binary | std::ios::trunc);
            output.write(genre_variants ? "ACETC002" : "ACETC001", 8);
            write_u64(output, header.size());
            output.write(header.data(), (std::streamsize) header.size());
            uint64_t hash = ace_cache_hash(header.data(), header.size());
            for (const auto * values : { &example.target_latents, &example.context_latents, &example.encoder_hidden,
                                         &example.genre_encoder_hidden }) {
                const size_t bytes = values->size() * sizeof(float);
                output.write(reinterpret_cast<const char *>(values->data()), (std::streamsize) bytes);
                hash = ace_cache_hash(values->data(), bytes, hash);
            }
            write_u64(output, hash);
            output.close();
            if (!output || !ace_sync_checkpoint_path(stage, false, error) || !replace(stage, destination, error)) {
                if (error.empty()) {
                    error = "cannot write preprocessing cache: " + destination.u8string();
                }
                std::error_code ignored;
                std::filesystem::remove(stage, ignored);
                return false;
            }
            for (auto it = lru.begin(); it != lru.end(); ++it) {
                if (it->first == index) {
                    resident -= tensor_bytes(*it->second);
                    lru.erase(it);
                    break;
                }
            }
            ++written;
            return true;
        } catch (const std::filesystem::filesystem_error & exception) {
            std::error_code ignored;
            std::filesystem::remove(stage, ignored);
            error = exception.what();
            return false;
        }
    }

    void observe(const ACETrainDiffusionExample & example) {
        max_encoder_length = std::max(max_encoder_length, example.real_encoder_sequence_length);
        max_encoder_length = std::max(max_encoder_length, example.genre_encoder_sequence_length);
    }

    std::shared_ptr<const ACETrainDiffusionExample> get(size_t index, std::string & error) {
        error.clear();
        if (!disk()) {
            return std::shared_ptr<const ACETrainDiffusionExample>(&memory.at(index),
                                                                   [](const ACETrainDiffusionExample *) {});
        }
        for (auto it = lru.begin(); it != lru.end(); ++it) {
            if (it->first == index) {
                auto result = it->second;
                lru.splice(lru.begin(), lru, it);
                return result;
            }
        }
        auto result = std::make_shared<ACETrainDiffusionExample>();
        if (!read_raw(index, *result, error)) {
            return {};
        }
        if (max_encoder_length <
                std::max(result->real_encoder_sequence_length, result->genre_encoder_sequence_length) ||
            null_condition.size() != size_t(hidden_size)) {
            error = "invalid dataset-wide conditioning padding";
            return {};
        }
        const int length = result->real_encoder_sequence_length;
        result->encoder_hidden.resize(size_t(max_encoder_length) * size_t(hidden_size));
        for (int s = length; s < max_encoder_length; ++s) {
            std::copy(null_condition.begin(), null_condition.end(),
                      result->encoder_hidden.begin() + size_t(s) * size_t(hidden_size));
        }
        if (!result->genre_encoder_hidden.empty()) {
            result->genre_encoder_hidden.resize(size_t(max_encoder_length) * size_t(hidden_size));
            for (int s = result->genre_encoder_sequence_length; s < max_encoder_length; ++s) {
                std::copy(null_condition.begin(), null_condition.end(),
                          result->genre_encoder_hidden.begin() + size_t(s) * size_t(hidden_size));
            }
        }
        const size_t bytes = tensor_bytes(*result);
        while (!lru.empty() && (bytes > memory_limit || resident > memory_limit - bytes)) {
            resident -= tensor_bytes(*lru.back().second);
            lru.pop_back();
        }
        if (bytes <= memory_limit) {
            resident += bytes;
            lru.emplace_front(index, result);
        }
        return result;
    }

  private:
    static bool replace(const std::filesystem::path & stage,
                        const std::filesystem::path & destination,
                        std::string &                 error) {
#ifdef _WIN32
        for (int attempt = 0; attempt < 101; ++attempt) {
            if (MoveFileExW(stage.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                return true;
            }
            DWORD code = GetLastError();
            if (code != ERROR_ACCESS_DENIED && code != ERROR_SHARING_VIOLATION && code != ERROR_LOCK_VIOLATION) {
                break;
            }
            Sleep(20);
        }
#else
        if (::rename(stage.c_str(), destination.c_str()) == 0) {
            return true;
        }
#endif
        error = "cannot atomically publish preprocessing cache: " + destination.u8string();
        return false;
    }

    std::list<std::pair<size_t, std::shared_ptr<const ACETrainDiffusionExample>>> lru;
    size_t                                                                        resident = 0;

    static size_t tensor_bytes(const ACETrainDiffusionExample & e) {
        // resize() may reserve more than the logical padded shape. Bound the
        // actual owned tensor allocation, including that spare capacity.
        return (e.target_latents.capacity() + e.context_latents.capacity() + e.encoder_hidden.capacity() +
                e.genre_encoder_hidden.capacity()) *
               sizeof(float);
    }

    static void write_u64(std::ostream & output, uint64_t value) {
        unsigned char bytes[8];
        for (int i = 0; i < 8; ++i) {
            bytes[i] = (unsigned char) (value >> (8 * i));
        }
        output.write(reinterpret_cast<const char *>(bytes), 8);
    }

    static bool read_u64(std::istream & input, uint64_t & value) {
        unsigned char bytes[8] = {};
        input.read(reinterpret_cast<char *>(bytes), 8);
        value = 0;
        for (int i = 0; i < 8; ++i) {
            value |= uint64_t(bytes[i]) << (8 * i);
        }
        return bool(input);
    }
};
