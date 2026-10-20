// genre-vocab.cpp: locate and load the genre vocabulary once per process.

#include "genre-vocab.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__APPLE__)
#    include <mach-o/dyld.h>
#else
#    include <unistd.h>
#endif

// The executable's folder with a trailing separator, where release packages
// place genres_vocab.txt; empty when it cannot be found.
static std::string executable_dir() {
    std::string path;
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768);
    DWORD                length = GetModuleFileNameW(nullptr, buffer.data(), (DWORD) buffer.size());
    if (length == 0 || length >= buffer.size()) {
        return "";
    }
    int bytes = WideCharToMultiByte(CP_UTF8, 0, buffer.data(), (int) length, nullptr, 0, nullptr, nullptr);
    path.resize((size_t) bytes);
    WideCharToMultiByte(CP_UTF8, 0, buffer.data(), (int) length, path.data(), bytes, nullptr, nullptr);
#elif defined(__APPLE__)
    char     buffer[4096];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) != 0) {
        return "";
    }
    path = buffer;
#else
    char    buffer[4096];
    ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length <= 0) {
        return "";
    }
    path.assign(buffer, (size_t) length);
#endif
    size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
}

std::shared_ptr<const GenreVocab> genre_vocab() {
    static std::once_flag                    once;
    static std::shared_ptr<const GenreVocab> vocab;
    std::call_once(once, [] {
        const char * override_path = std::getenv("ACE_GENRES_VOCAB");
        std::string  path = override_path && *override_path ? override_path : executable_dir() + "genres_vocab.txt";
#ifdef _WIN32
        // Paths are UTF-8; std::ifstream on Windows needs the wide form.
        int                  wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::vector<wchar_t> wpath((size_t) (wide > 0 ? wide : 1));
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
        vocab = GenreVocab::load(std::filesystem::path(wpath.data()));
#else
        vocab = GenreVocab::load(std::filesystem::path(path));
#endif
        if (vocab) {
            fprintf(stderr, "[Genres] %zu genres from %s\n", vocab->entries.size(), path.c_str());
        } else {
            fprintf(stderr, "[Genres] WARNING: no genre vocabulary at %s; genres are skipped\n", path.c_str());
        }
    });
    return vocab;
}
