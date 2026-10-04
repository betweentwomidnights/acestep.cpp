// ABOUTME: Exercises server request handling and process shutdown without model weights.
// ABOUTME: Includes the server to test its handlers and fatal-exit log capture directly.

#define main ace_server_main
#include "../tools/ace-server.cpp"
#undef main

#include <chrono>
#include <cmath>
#include <filesystem>

static int fail(const char * message) {
    fprintf(stderr, "FAIL: %s\n", message);
    return 1;
}

static int test_float_roundtrip() {
    const float values[] = { 1.005f, 0.7f, 0.000123456f, std::nextafter(1.0f, 2.0f), -0.1234567f };
    for (float value : values) {
        AceRequest input;
        request_init(&input);
        input.guidance_scale = value;
        input.adapter_scale  = value;
        input.latent_shift   = value;
        for (bool sparse : { false, true }) {
            AceRequest  output;
            std::string json = request_to_json(&input, sparse);
            if (!request_parse_json(&output, json.c_str()) || output.guidance_scale != value ||
                output.adapter_scale != value || output.latent_shift != value) {
                return fail("request JSON must preserve float values exactly");
            }
        }
    }
    httplib::Request  req;
    httplib::Response res;
    handle_props(req, res);
    yyjson_doc * doc = yyjson_read(res.body.c_str(), res.body.size(), 0);
    if (!doc) {
        return fail("props must contain valid JSON");
    }
    yyjson_val * defaults    = yyjson_obj_get(yyjson_doc_get_root(doc), "default");
    const float  temperature = (float) yyjson_get_num(yyjson_obj_get(defaults, "lm_temperature"));
    yyjson_doc_free(doc);
    if (temperature != 0.85f) {
        return fail("props must preserve the request defaults");
    }
    return 0;
}

static int test_multipart() {
    // Advertise the pipeline without loading weights; only exercise the queueing handler.
    g_registry.dit.resize(1);
    g_registry.text_enc.resize(1);
    g_registry.vae.resize(1);
    const char * bodies[] = { "{\"caption\":\"one\"}", "[{\"caption\":\"one\"},{\"caption\":\"two\"}]", "[]",
                              "[broken" };
    for (bool file_part : { false, true }) {
        for (int i = 0; i < 4; ++i) {
            httplib::Request req;
            req.set_header("Content-Type", "multipart/form-data; boundary=test");
            if (file_part) {
                req.form.files.emplace(
                    "request", httplib::FormData{ "request", bodies[i], "request.json", "application/json", {} });
            } else {
                req.form.fields.emplace("request", httplib::FormField{ "request", bodies[i], {} });
            }
            req.form.files.emplace(
                "src_latents",
                httplib::FormData{
                    "src_latents", std::string(64 * 4, '\0'), "source.vae", "application/octet-stream", {} });
            req.form.files.emplace(
                "ref_latents",
                httplib::FormData{
                    "ref_latents", std::string(64 * 4, '\0'), "reference.vae", "application/octet-stream", {} });
            httplib::Response res;
            res.status               = 200;
            const uint64_t first_log = log_seq;
            {
                LogCapture capture;
                handle_synth(req, res);
            }
            const bool accepted = i < 2;
            if ((accepted && (res.status != 200 || g_work_queue.size() != 1)) ||
                (!accepted && (res.status != 400 || !g_work_queue.empty()))) {
                fprintf(stderr, "multipart case %d file=%d: status=%d body=%s\n", i, (int) file_part, res.status,
                        res.body.c_str());
                return fail("multipart objects/batches must queue; empty/malformed batches must reject");
            }
            if (accepted) {
                const std::string count = i == 0 ? "(1 requests)" : "(2 requests)";
                bool              found = false;
                for (uint64_t n = first_log; n < log_seq; ++n) {
                    found |= log_ring[n & LOG_RING_MASK].find(count) != std::string::npos;
                }
                if (!found) {
                    return fail("multipart must retain every request in the batch");
                }
            }
            // No worker runs here: queued requests hold latents but load no models.
            g_work_queue.clear();
            g_jobs.clear();
        }
    }
    g_registry = {};
    return 0;
}

static int test_bind_failure() {
    httplib::Server occupied;
    // Disable httplib's default reuse option so this fixture actually owns the port.
    occupied.set_socket_options([](socket_t sock) {
#ifdef _WIN32
        httplib::set_socket_opt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1);
#else
        (void) sock;
#endif
    });
    const int port = occupied.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        return fail("could not reserve a loopback port");
    }
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir    = std::filesystem::temp_directory_path() / ("ace-server-test-" + std::to_string(unique));
    std::filesystem::create_directory(dir);
    gguf_context * model = gguf_init_empty();
    gguf_set_val_str(model, "general.architecture", "acestep-lm");
    const bool written = gguf_write_to_file(model, (dir / "stub.gguf").string().c_str(), true);
    gguf_free(model);
    if (!written) {
        std::filesystem::remove_all(dir);
        return fail("could not write the metadata-only model fixture");
    }
    std::string model_dir = dir.string();
    std::string port_arg  = std::to_string(port);
    char        program[] = "ace-server";
    char        models[]  = "--models";
    char        option[]  = "--port";
    char *      args[]    = { program, models, &model_dir[0], option, &port_arg[0] };
    const int   result    = ace_server_main(5, args);
    g_svr                 = nullptr;
    std::filesystem::remove_all(dir);
    if (result != 1) {
        return fail("an occupied port must make the actual server entry point return 1");
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 2 && !strcmp(argv[1], "--fatal-log")) {
        LogCapture        capture;
        const std::string payload(256 * 1024, 'x');
        fwrite(payload.data(), 1, payload.size(), stderr);
        fputs("FATAL-TAIL", stderr);  // Also exercise a partial final line.
        std::exit(17);                // Skip the capture destructor, as the loaders do.
    }
    if (test_float_roundtrip() || test_multipart() || test_bind_failure()) {
        return 1;
    }
    fprintf(stderr, "PASS: server float round trips, multipart batches, and bind failure\n");
    return 0;
}
