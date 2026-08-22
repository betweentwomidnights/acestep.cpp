// ABOUTME: Resolves adapter directories backed by immutable checkpoint generations.
// ABOUTME: Validates the atomic generation pointer before consumers open checkpoint files.

#pragma once

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

static bool adapter_checkpoint_directory(const std::filesystem::path & requested,
                                         std::filesystem::path &       resolved,
                                         std::string &                 error) {
    error.clear();
    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(requested, filesystem_error)) {
        if (filesystem_error && std::filesystem::exists(requested)) {
            error = "cannot inspect adapter checkpoint path: " + filesystem_error.message();
            return false;
        }
        resolved = requested;
        return true;
    }
    const std::filesystem::path pointer = requested / "checkpoint.current";
    const std::filesystem::file_status pointer_status = std::filesystem::symlink_status(pointer, filesystem_error);
    if (filesystem_error == std::make_error_code(std::errc::no_such_file_or_directory)) {
        filesystem_error.clear();
    }
    if (filesystem_error) {
        error = "cannot inspect checkpoint generation pointer: " + filesystem_error.message();
        return false;
    }
    if (!std::filesystem::exists(pointer_status)) {
        if (std::filesystem::is_directory(requested / ".checkpoint-generations")) {
            error = "checkpoint generation pointer is missing";
            return false;
        }
        resolved = requested;
        return true;
    }
    std::filesystem::path target;
    if (std::filesystem::is_symlink(pointer_status)) {
        target = std::filesystem::read_symlink(pointer, filesystem_error);
    } else if (std::filesystem::is_regular_file(pointer_status)) {
        std::ifstream input(pointer, std::ios::binary);
        const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (!input.is_open() || input.bad() || contents.empty()) {
            error = "checkpoint generation pointer is invalid";
            return false;
        }
        target = std::filesystem::path(contents);
    } else {
        error = "checkpoint generation pointer is invalid";
        return false;
    }
    const std::string generation_name = target.filename().string();
    if (filesystem_error || target.is_absolute() || target.parent_path() != ".checkpoint-generations" ||
        generation_name.size() <= 11 || generation_name.compare(0, 11, "generation-") != 0) {
        error = "checkpoint generation pointer is invalid";
        return false;
    }
    const std::filesystem::path generation = requested / target;
    if (!std::filesystem::is_directory(generation)) {
        error = "checkpoint generation is missing";
        return false;
    }
    resolved = generation;
    return true;
}
