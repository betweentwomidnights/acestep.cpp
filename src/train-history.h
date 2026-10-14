// Metric names follow betweentwomidnights/sa3.cpp's train_job.h (MIT).
// A diagnostic journal, independent of checkpoint selection and optimizer state.
#pragma once

#include "yyjson.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace ace_training {

struct TrainHistory {
    std::ofstream file;
    std::string   session;
    int           start_update = 0;

    bool open(const std::filesystem::path & output, int step, std::string & error) {
        const auto      path = output / "metrics.jsonl";
        std::error_code ec;
        std::filesystem::create_directories(output, ec);
        if (ec) {
            error = "cannot create training history directory: " + ec.message();
            return false;
        }
        // A killed writer can leave a partial final record. Preserve all complete
        // records, and never concatenate the new session onto a torn JSON line.
        if (std::filesystem::is_regular_file(path, ec)) {
            std::ifstream input(path, std::ios::binary | std::ios::ate);
            if (!input) {
                error = "cannot read training history: " + path.u8string();
                return false;
            }
            auto end = input.tellg();
            if (end < 0) {
                error = "cannot inspect training history length";
                return false;
            }
            auto keep = end;
            while (keep > 0) {
                input.seekg(keep - std::streamoff(1));
                char c = 0;
                if (!input.get(c)) {
                    error = "cannot inspect training history tail";
                    return false;
                }
                if (c == '\n') {
                    break;
                }
                keep -= std::streamoff(1);
            }
            input.close();
            if (keep != end) {
                std::filesystem::resize_file(path, static_cast<uintmax_t>(keep), ec);
                if (ec) {
                    error = "cannot repair incomplete training history tail: " + ec.message();
                    return false;
                }
            }
        }
        file.open(path, std::ios::binary | std::ios::app);
        if (!file) {
            error = "cannot append training history: " + path.u8string();
            return false;
        }
        session = std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "-" +
                  std::to_string(std::random_device{}());
        start_update = step;
        return true;
    }

    bool append(int           epoch,
                int           update,
                double        lr,
                double        loss,
                double        grad_norm,
                double        step_s,
                int           microbatches,
                int           examples,
                std::string & error) {
        if (!file || !std::isfinite(lr) || !std::isfinite(loss) || !std::isfinite(grad_norm) ||
            !std::isfinite(step_s) || epoch <= 0 || update <= start_update || microbatches <= 0 || examples <= 0) {
            error = "invalid training history sample or unwritable metrics.jsonl";
            return false;
        }
        yyjson_mut_doc * doc = yyjson_mut_doc_new(nullptr);
        if (!doc) {
            error = "cannot allocate training history JSON";
            return false;
        }
        auto * root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_int(doc, root, "epoch", epoch);
        yyjson_mut_obj_add_int(doc, root, "update", update);
        yyjson_mut_obj_add_str(doc, root, "split", "train");
        yyjson_mut_obj_add_real(doc, root, "lr", lr);
        yyjson_mut_obj_add_real(doc, root, "loss", loss);
        yyjson_mut_obj_add_real(doc, root, "grad_norm", grad_norm);
        yyjson_mut_obj_add_real(doc, root, "step_s", step_s);
        yyjson_mut_obj_add_str(doc, root, "session", session.c_str());
        yyjson_mut_obj_add_int(doc, root, "start_update", start_update);
        yyjson_mut_obj_add_int(doc, root, "microbatches", microbatches);
        yyjson_mut_obj_add_int(doc, root, "examples", examples);
        size_t length = 0;
        char * json   = yyjson_mut_write(doc, 0, &length);
        yyjson_mut_doc_free(doc);
        if (!json) {
            error = "cannot serialize training history";
            return false;
        }
        file.write(json, static_cast<std::streamsize>(length));
        std::free(json);
        file.put('\n');
        file.flush();
        if (!file) {
            error = "cannot write or flush training history metrics.jsonl";
            return false;
        }
        return true;
    }
};

}  // namespace ace_training
