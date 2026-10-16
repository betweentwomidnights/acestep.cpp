// Exercise the same registry/proc-address interface as distributed DL builds.
#pragma once
#include "backend.h"

static ggml_backend_t test_cpu_backend() {
    ggml_backend_load_all();
    return cpu_backend_new(2);
}

static void test_cpu_abort_callback(ggml_backend_t backend, ggml_abort_callback callback, void * data) {
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_abort =
        (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
    if (!set_abort) {
        std::fputs("CPU backend has no abort callback interface\n", stderr);
        std::abort();
    }
    set_abort(backend, callback, data);
}
