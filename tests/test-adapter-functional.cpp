// ABOUTME: Tests unmerged LoRA/DoRA inference against dense FP32 reference math.
// ABOUTME: Covers quantized base immutability, strength interpolation and batched inputs.
#include "adapter-functional.h"
#include "backend.h"
#include "train-adapter-checkpoint.h"

#include <cstdio>
#include <filesystem>
#include <random>

static void print_available_backends() {
    ggml_backend_load_all();
    std::fputs("Available devices:", stderr);
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        std::fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
    }
    std::fputc('\n', stderr);
}

static bool run_case(ggml_backend_t       backend,
                     ggml_type            type,
                     bool                 dora,
                     float                strength,
                     bool                 normalized,
                     std::vector<float> & result) {
    const int                             in = 256, out = 32, rank = 8, seq = 5, batch = 2;
    const char *                          name = "decoder.layers.0.self_attn.q_proj.weight";
    std::mt19937                          rng(123);
    std::uniform_real_distribution<float> random(-0.2f, 0.2f);
    std::vector<float>                    base(in * out), a(in * rank), b(rank * out), m(out), x(in * seq * batch);
    for (auto & v : base) {
        v = random(rng);
    }
    for (auto & v : a) {
        v = random(rng);
    }
    for (auto & v : b) {
        v = random(rng);
    }
    for (auto & v : x) {
        v = random(rng);
    }
    std::vector<uint8_t> packed(ggml_row_size(type, in) * out);
    adapter_requant(base.data(), packed.data(), base.size(), in, type);
    adapter_dequant(packed.data(), base.data(), in, out, type);
    for (int o = 0; o < out; ++o) {
        double n = 0;
        for (int i = 0; i < in; ++i) {
            n += (double) base[o * in + i] * base[o * in + i];
        }
        m[o] = (float) std::sqrt(n) * (0.8f + 0.02f * o);
    }
    const auto dir =
        std::filesystem::temp_directory_path() / ("ace-functional-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);

    struct Cleanup {
        std::filesystem::path path;

        ~Cleanup() {
            std::error_code e;
            std::filesystem::remove_all(path, e);
        }
    } cleanup{ dir };

    ggml_context * gc = ggml_init({ 1024 * 1024, nullptr, false });
    auto           gt = ggml_new_tensor_2d(gc, type, in, out);
    ggml_set_name(gt, name);
    std::memcpy(gt->data, packed.data(), packed.size());
    auto g = gguf_init_empty();
    gguf_add_tensor(g, gt);
    bool ok = gguf_write_to_file(g, (dir / "base.gguf").string().c_str(), false);
    gguf_free(g);
    ggml_free(gc);
    if (!ok) {
        return false;
    }
    ACETrainAdapterState host{};
    host.adapter_type   = dora ? "dora-rows" : "lora";
    host.base_rank      = rank;
    host.base_alpha     = 2 * rank;
    host.module_profile = "attention";
    ACETrainAdapterParam p{};
    p.target.weight_name = name;
    p.target.module_name = "self_attn.q_proj";
    p.target.in          = in;
    p.target.out         = out;
    p.target.rank        = rank;
    p.target.alpha       = 2 * rank;
    p.target.base_rank   = rank;
    p.target.base_alpha  = 2 * rank;
    p.a                  = a;
    p.b                  = b;
    if (dora) {
        p.magnitude = m;
    }
    host.params.push_back(p);
    std::string error;
    if (!ace_save_train_adapter_checkpoint(dir.string(), host, error)) {
        fprintf(stderr, "save: %s\n", error.c_str());
        return false;
    }
    GGUFModel gf{};
    if (!gf_load(&gf, (dir / "base.gguf").string().c_str())) {
        return false;
    }
    WeightCtx wc{};
    wctx_init(&wc, 1);
    auto              w = gf_load_tensor(&wc, gf, name);
    AdapterFunctional state;
    ok = adapter_functional_load(state, &wc, gf, dir.string().c_str(), strength, backend, normalized) &&
         wctx_alloc(&wc, backend) && adapter_functional_prepare(state, backend) &&
         adapter_functional_prepare(state, backend);  // cached normalization must be idempotent
    gf_close(&gf);
    if (!ok) {
        adapter_functional_free(state);
        wctx_free(&wc);
        return false;
    }
    std::vector<uint8_t> after(packed.size());
    ggml_backend_tensor_get(w, after.data(), 0, after.size());
    ok = after == packed;
    std::vector<float> expected(out * seq * batch);
    for (int o = 0; o < out; ++o) {
        std::vector<double> full(in);
        double              norm = 0;
        for (int i = 0; i < in; ++i) {
            double delta = 0;
            for (int r = 0; r < rank; ++r) {
                delta += (double) b[o * rank + r] * a[r * in + i];
            }
            full[i] = base[o * in + i] + 2 * delta * (normalized && dora ? strength : 1.0f);
            norm += full[i] * full[i];
        }
        for (int n = 0; n < seq * batch; ++n) {
            for (int i = 0; i < in; ++i) {
                double effective = dora ? full[i] * m[o] / std::sqrt(norm) : full[i];
                if (!normalized || !dora) {
                    effective = base[o * in + i] + strength * (effective - base[o * in + i]);
                }
                if (strength == 0) {
                    effective = base[o * in + i];  // explicit bypass, even if m differs from ||W||
                }
                expected[n * out + o] += (float) (effective * x[n * in + i]);
            }
        }
    }
    auto ctx   = ggml_init({ 4 * 1024 * 1024, nullptr, true });
    auto input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in, seq, batch);
    ggml_set_input(input);
    auto y = adapter_functional_linear(ctx, &state, w, input);
    ggml_set_output(y);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ok         = ok && ggml_gallocr_alloc_graph(alloc, graph);
    std::vector<float> actual(expected.size()), first;
    double             rel = 0;
    if (ok) {
        for (int repeat = 0; repeat < 2; ++repeat) {
            ggml_backend_tensor_set(input, x.data(), 0, x.size() * 4);
            ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
            if (!ok) {
                break;
            }
            ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * 4);
            if (repeat == 0) {
                first = actual;
            } else {
                ok = actual == first;
            }
        }
        double err = 0, ref = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            double d = actual[i] - expected[i];
            err += d * d;
            ref += (double) expected[i] * expected[i];
        }
        // CUDA's default F32 matmul uses reduced-precision arithmetic: even
        // strength=0 differs from the dense CPU reference by about 8e-4.
        const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        rel            = std::sqrt(err / ref);
        ok             = ok && rel < (type == GGML_TYPE_F32 ? (cpu ? 2e-5 : 0.002) : 0.015);
        ggml_backend_tensor_get(w, after.data(), 0, after.size());
        ok = ok && after == packed;
    }
    printf("%s %s strength=%.2f mode=%s relative_error=%.8g unchanged=%s %s\n", ggml_type_name(type),
           dora ? "DoRA" : "LoRA", strength, normalized ? "normalized" : "delta", rel, after == packed ? "yes" : "NO",
           ok ? "PASS" : "FAIL");
    result = actual;
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    adapter_functional_free(state);
    wctx_free(&wc);
    return ok;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const char * backend_name = argc > 1 ? argv[1] : "CPU";
    auto         backend      = ggml_backend_init_by_name(backend_name, nullptr);
    if (!backend) {
        std::fprintf(stderr, "failed to initialize backend device: %s\n", backend_name);
        print_available_backends();
        return 1;
    }
    bool ok = true;
    for (auto type : { GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
        for (bool dora : { false, true }) {
            for (float s : { -0.5f, 0.0f, 0.5f, 1.0f, 2.0f }) {
                std::vector<float> delta_result, normalized_result;
                bool               pair_ok = run_case(backend, type, dora, s, false, delta_result);
                pair_ok                    = run_case(backend, type, dora, s, true, normalized_result) && pair_ok;
                if (!dora || s == 0 || s == 1) {
                    const bool identical = delta_result == normalized_result;
                    printf("mode equivalence %s %s strength=%.2f: %s\n", ggml_type_name(type), dora ? "DoRA" : "LoRA",
                           s, identical ? "EXACT" : "FAIL");
                    pair_ok = pair_ok && identical;
                }
                ok = pair_ok && ok;
            }
        }
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
