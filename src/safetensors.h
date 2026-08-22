// ABOUTME: Parses memory-mapped safetensors files and exposes validated tensor entries.
// ABOUTME: Rejects malformed shapes, dtypes, offsets, and byte spans before data access.
#pragma once
// safetensors.h: minimal read only safetensors parser
//
// Format: 8 byte LE header length, JSON header, raw tensor data.
// We mmap the file, validate the header, and expose tensor entries
// with name, dtype, shape, and a pointer to the raw data.
//
// Only handles the flat safetensors JSON structure.

#include "yyjson.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

struct STEntry {
    std::string name;
    std::string dtype;  // "F32", "BF16", "F16"
    int64_t     shape[4];
    int         n_dims;
    size_t      data_start;  // byte offset from data section
    size_t      data_end;
};

struct STFile {
    uint8_t *            mapping;
    size_t               file_size;
    size_t               data_offset;  // 8 + header_len (start of tensor data)
    std::vector<STEntry> entries;
#ifdef _WIN32
    HANDLE fh, mh;
#else
    int fd;
#endif
};

static bool st_parse(STFile * st, const char * hdr, size_t len) {
    yyjson_doc * document = yyjson_read(hdr, len, 0);
    yyjson_val * root     = document ? yyjson_doc_get_root(document) : nullptr;
    if (!yyjson_is_obj(root)) {
        if (document) {
            yyjson_doc_free(document);
        }
        return false;
    }
    auto reject = [&]() {
        yyjson_doc_free(document);
        st->entries.clear();
        return false;
    };
    std::set<std::string> names;
    yyjson_obj_iter       entries = yyjson_obj_iter_with(root);
    yyjson_val *          key;
    while ((key = yyjson_obj_iter_next(&entries))) {
        yyjson_val * value = yyjson_obj_iter_get_val(key);
        if (!yyjson_is_str(key) || !names.insert(yyjson_get_str(key)).second) {
            return reject();
        }
        if (strcmp(yyjson_get_str(key), "__metadata__") == 0) {
            if (!yyjson_is_obj(value)) {
                return reject();
            }
            continue;
        }
        if (!yyjson_is_obj(value)) {
            return reject();
        }
        yyjson_val * dtype   = yyjson_obj_get(value, "dtype");
        yyjson_val * shape   = yyjson_obj_get(value, "shape");
        yyjson_val * offsets = yyjson_obj_get(value, "data_offsets");
        if (!yyjson_is_str(dtype) || !yyjson_is_arr(shape) || !yyjson_is_arr(offsets)) {
            return reject();
        }

        STEntry entry              = {};
        entry.name                 = yyjson_get_str(key);
        entry.dtype                = yyjson_get_str(dtype);
        yyjson_arr_iter dimensions = yyjson_arr_iter_with(shape);
        yyjson_val *    dimension;
        while ((dimension = yyjson_arr_iter_next(&dimensions))) {
            if (entry.n_dims >= 4 || !yyjson_is_int(dimension) || yyjson_get_int(dimension) < 0) {
                return reject();
            }
            entry.shape[entry.n_dims++] = yyjson_get_int(dimension);
        }
        yyjson_arr_iter offset_values = yyjson_arr_iter_with(offsets);
        yyjson_val *    start         = yyjson_arr_iter_next(&offset_values);
        yyjson_val *    end           = yyjson_arr_iter_next(&offset_values);
        if (!yyjson_is_int(start) || !yyjson_is_int(end) || yyjson_arr_iter_next(&offset_values) ||
            yyjson_get_int(start) < 0 || yyjson_get_int(end) < 0 ||
            (uint64_t) yyjson_get_int(start) > (uint64_t) SIZE_MAX ||
            (uint64_t) yyjson_get_int(end) > (uint64_t) SIZE_MAX) {
            return reject();
        }
        entry.data_start = (size_t) yyjson_get_int(start);
        entry.data_end   = (size_t) yyjson_get_int(end);
        st->entries.push_back(std::move(entry));
    }
    yyjson_doc_free(document);
    return true;
}

static void st_close(STFile * st);

static size_t st_dtype_size(const std::string & dtype) {
    if (dtype == "F32") {
        return 4;
    }
    if (dtype == "BF16" || dtype == "F16") {
        return 2;
    }
    return 0;
}

static bool st_validate_entries(const STFile & st) {
    const size_t                           data_size = st.file_size - st.data_offset;
    std::vector<std::pair<size_t, size_t>> spans;
    spans.reserve(st.entries.size());
    for (const STEntry & entry : st.entries) {
        const size_t element_size = st_dtype_size(entry.dtype);
        if (entry.name.empty() || element_size == 0 || entry.n_dims < 0 || entry.n_dims > 4 ||
            entry.data_start > entry.data_end || entry.data_end > data_size) {
            return false;
        }
        size_t elements = 1;
        for (int dimension = 0; dimension < entry.n_dims; ++dimension) {
            if (entry.shape[dimension] < 0) {
                return false;
            }
            const size_t extent = (size_t) entry.shape[dimension];
            if (extent != 0 && elements > SIZE_MAX / extent) {
                return false;
            }
            elements *= extent;
        }
        if (elements > SIZE_MAX / element_size || entry.data_end - entry.data_start != elements * element_size) {
            return false;
        }
        spans.emplace_back(entry.data_start, entry.data_end);
    }
    std::sort(spans.begin(), spans.end());
    size_t expected_start = 0;
    for (const auto & span : spans) {
        if (span.first != expected_start) {
            return false;
        }
        expected_start = span.second;
    }
    return expected_start == data_size;
}

static bool st_open(STFile * st, const char * path) {
    *st = {};

#ifdef _WIN32
    st->fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (st->fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[Safetensors] Cannot open %s\n", path);
        return false;
    }
    LARGE_INTEGER li;
    GetFileSizeEx(st->fh, &li);
    st->file_size = (size_t) li.QuadPart;
    st->mh        = CreateFileMappingA(st->fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!st->mh) {
        CloseHandle(st->fh);
        return false;
    }
    st->mapping = (uint8_t *) MapViewOfFile(st->mh, FILE_MAP_READ, 0, 0, 0);
    if (!st->mapping) {
        CloseHandle(st->mh);
        CloseHandle(st->fh);
        return false;
    }
#else
    st->fd = open(path, O_RDONLY);
    if (st->fd < 0) {
        fprintf(stderr, "[Safetensors] Cannot open %s\n", path);
        return false;
    }
    struct stat sb;
    fstat(st->fd, &sb);
    st->file_size = (size_t) sb.st_size;
    st->mapping   = (uint8_t *) mmap(NULL, st->file_size, PROT_READ, MAP_PRIVATE, st->fd, 0);
    if (st->mapping == MAP_FAILED) {
        close(st->fd);
        st->mapping = NULL;
        fprintf(stderr, "[Safetensors] Mmap failed %s\n", path);
        return false;
    }
#endif

    // first 8 bytes: LE u64 header length
    if (st->file_size < 8) {
        fprintf(stderr, "[Safetensors] File too small %s\n", path);
        st_close(st);
        return false;
    }
    uint64_t hdr_len;
    memcpy(&hdr_len, st->mapping, 8);
    if (hdr_len > (uint64_t) (st->file_size - 8)) {
        fprintf(stderr, "[Safetensors] Header overflows file %s\n", path);
        st_close(st);
        return false;
    }
    st->data_offset = 8 + (size_t) hdr_len;

    // parse JSON header
    if (!st_parse(st, (const char *) st->mapping + 8, (size_t) hdr_len)) {
        fprintf(stderr, "[Safetensors] Failed to parse header %s\n", path);
        st_close(st);
        return false;
    }
    if (!st_validate_entries(*st)) {
        fprintf(stderr, "[Safetensors] Invalid tensor entry %s\n", path);
        st_close(st);
        return false;
    }

    fprintf(stderr, "[Safetensors] %s: %zu tensors\n", path, st->entries.size());
    return true;
}

static void st_close(STFile * st) {
#ifdef _WIN32
    if (st->mapping) {
        UnmapViewOfFile(st->mapping);
    }
    if (st->mh) {
        CloseHandle(st->mh);
    }
    if (st->fh && st->fh != INVALID_HANDLE_VALUE) {
        CloseHandle(st->fh);
    }
#else
    if (st->mapping) {
        munmap(st->mapping, st->file_size);
    }
    if (st->fd >= 0) {
        close(st->fd);
    }
#endif
    *st = {};
}

// Get raw data pointer for a tensor entry
static inline const void * st_data(const STFile & st, const STEntry & e) {
    return st.mapping + st.data_offset + e.data_start;
}
