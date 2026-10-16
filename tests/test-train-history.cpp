#include "train-history.h"

#include <cassert>
#include <limits>
#include <sstream>

static void expect(bool ok) {
    if (!ok) {
        std::abort();
    }
}

int main() {
    namespace fs     = std::filesystem;
    const auto  root = fs::temp_directory_path() / ("ace-history-" + std::to_string(std::random_device{}()));
    std::string error, first_session;
    {
        ace_training::TrainHistory history;
        expect(history.open(root, 0, error));
        first_session = history.session;
        expect(history.append(1, 1, 0.0001, 1.25, 2.5, 0.3, 2, 3, error));
        expect(!history.append(1, 2, 0.0001, std::numeric_limits<double>::infinity(), 2.5, 0.3, 1, 1, error));
    }
    // Simulate interrupted publication and a rollback to an earlier checkpoint.
    {
        std::ofstream tail(root / "metrics.jsonl", std::ios::app);
        tail << "{\"update\":";
    }
    {
        ace_training::TrainHistory history;
        expect(history.open(root, 0, error));
        expect(history.session != first_session);
        expect(history.append(1, 1, 0.0001, 1.0, 2.0, 0.2, 1, 1, error));
    }
    std::ifstream file(root / "metrics.jsonl");
    std::string   line;
    int           count = 0;
    while (std::getline(file, line)) {
        auto * doc = yyjson_read(line.data(), line.size(), 0);
        expect(doc != nullptr);
        auto * sample = yyjson_doc_get_root(doc);
        expect(yyjson_get_int(yyjson_obj_get(sample, "update")) == 1);
        expect(yyjson_get_int(yyjson_obj_get(sample, "epoch")) == 1);
        expect(yyjson_get_int(yyjson_obj_get(sample, "start_update")) == 0);
        expect(yyjson_is_real(yyjson_obj_get(sample, "grad_norm")));
        yyjson_doc_free(doc);
        ++count;
    }
    expect(count == 2);
    file.close();
    {
        std::ofstream blocked(root / "blocked");
        blocked << "file";
    }
    ace_training::TrainHistory blocked;
    expect(!blocked.open(root / "blocked", 0, error));
    fs::remove_all(root);
    std::puts("PASS: append, resume sessions, incomplete tail and history errors");
}
