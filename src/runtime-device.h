// Runtime device names shared by the CLI, server and supervisor probe.
#pragma once

#include "ggml-backend.h"

#include <cctype>
#include <string>

static ggml_backend_dev_t ace_runtime_device(const char * requested) {
    if (!requested || !*requested || std::string(requested) == "auto") {
        for (auto type :
             { GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU, GGML_BACKEND_DEVICE_TYPE_CPU }) {
            if (auto dev = ggml_backend_dev_by_type(type)) {
                return dev;
            }
        }
        return nullptr;
    }
    std::string name(requested);
    for (char & c : name) {
        c = (char) std::tolower((unsigned char) c);
    }
    if (name == "auto") {
        return ace_runtime_device(nullptr);
    }
    ggml_backend_dev_t alias          = nullptr;
    int                alias_priority = 4;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto        dev = ggml_backend_dev_get(i);
        std::string exact(ggml_backend_dev_name(dev));
        std::string backend(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)));
        for (char & c : exact) {
            c = (char) std::tolower((unsigned char) c);
        }
        for (char & c : backend) {
            c = (char) std::tolower((unsigned char) c);
        }
        if (name == exact) {
            return dev;
        }
        if (name == backend || (name == "metal" && backend == "mtl")) {
            // Registry order may put an integrated GPU before a discrete one.
            // A backend alias follows auto's preference; exact names stay exact.
            auto type     = ggml_backend_dev_type(dev);
            int  priority = type == GGML_BACKEND_DEVICE_TYPE_GPU  ? 0 :
                            type == GGML_BACKEND_DEVICE_TYPE_IGPU ? 1 :
                            type == GGML_BACKEND_DEVICE_TYPE_CPU  ? 2 :
                                                                    3;
            if (priority < alias_priority) {
                alias          = dev;
                alias_priority = priority;
            }
        }
    }
    return alias;
}
