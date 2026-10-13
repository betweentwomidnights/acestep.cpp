// Native continuation mirrors sa3.cpp's compatibility + cursor contract.
// ACE keeps PEFT/optimizer files in its existing immutable checkpoint generations.
#pragma once

#include "train-checkpoint.h"

#include <climits>
#include <cmath>
#include <numeric>
#include <stdexcept>

struct ACETrainResumeProgress {
    int                 epoch         = 0;
    int                 next_batch    = 0;
    double              epoch_loss    = 0;
    int                 epoch_batches = 0;
    std::vector<size_t> order;
};

struct ACETrainResumePlan {
    std::string json;
    std::string base_fingerprint;
    int         dataset_size      = 0;
    int         epochs            = 0;
    int         accumulation      = 0;
    int         batches_per_epoch = 0;
    int         updates_per_epoch = 0;
    int         total_steps       = 0;
};

static std::string ace_train_continuation(const ACETrainResumePlan & plan, const ACETrainResumeProgress & progress) {
    yyjson_doc * immutable = yyjson_read(plan.json.data(), plan.json.size(), 0);
    if (!immutable) {
        throw std::runtime_error("invalid training plan");
    }
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    auto *           root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_int(doc, root, "state_version", 1);
    yyjson_mut_obj_add_val(doc, root, "plan", yyjson_val_mut_copy(doc, yyjson_doc_get_root(immutable)));
    yyjson_doc_free(immutable);
    yyjson_mut_obj_add_int(doc, root, "epoch", progress.epoch);
    yyjson_mut_obj_add_int(doc, root, "next_batch", progress.next_batch);
    yyjson_mut_obj_add_int(doc, root, "epoch_batches", progress.epoch_batches);
    yyjson_mut_obj_add_real(doc, root, "epoch_loss", progress.epoch_loss);
    auto * order = yyjson_mut_arr(doc);
    for (size_t index : progress.order) {
        yyjson_mut_arr_add_uint(doc, order, index);
    }
    yyjson_mut_obj_add_val(doc, root, "order", order);
    char * text = yyjson_mut_write(doc, 0, nullptr);
    yyjson_mut_doc_free(doc);
    if (!text) {
        throw std::runtime_error("cannot serialize training continuation");
    }
    std::string result(text);
    free(text);
    return result;
}

static bool ace_train_restore_progress(yyjson_val *               root,
                                       const ACETrainResumePlan & plan,
                                       int                        completed_epochs,
                                       int                        step,
                                       ACETrainResumeProgress &   progress,
                                       std::string &              error) {
    progress     = {};
    auto invalid = [&]() {
        error = "resume continuation has an invalid cursor, order, or optimizer step";
        return false;
    };
    if (!yyjson_is_obj(root) || yyjson_get_int(yyjson_obj_get(root, "state_version")) != 1 || plan.accumulation <= 0 ||
        plan.batches_per_epoch <= 0) {
        error = "unsupported training continuation version";
        return false;
    }
    yyjson_doc * expected   = yyjson_read(plan.json.data(), plan.json.size(), 0);
    const bool   compatible = expected && yyjson_equals(yyjson_obj_get(root, "plan"), yyjson_doc_get_root(expected));
    yyjson_doc_free(expected);
    if (!compatible) {
        error =
            "resume checkpoint is incompatible with the current model, dataset, or training settings (including "
            "schedule budget)";
        return false;
    }
    auto integer = [&](const char * name, int & value) {
        auto * node = yyjson_obj_get(root, name);
        if (!yyjson_is_int(node) || yyjson_get_sint(node) < 0 || yyjson_get_uint(node) > INT_MAX) {
            return false;
        }
        value = int(yyjson_get_uint(node));
        return true;
    };
    if (!integer("epoch", progress.epoch) || !integer("next_batch", progress.next_batch) ||
        !integer("epoch_batches", progress.epoch_batches)) {
        return invalid();
    }
    auto * loss = yyjson_obj_get(root, "epoch_loss");
    if (!yyjson_is_num(loss)) {
        return invalid();
    }
    progress.epoch_loss = yyjson_get_num(loss);
    if (!std::isfinite(progress.epoch_loss) || progress.epoch_loss < 0 || progress.epoch != completed_epochs ||
        progress.epoch > plan.epochs || progress.next_batch >= plan.batches_per_epoch ||
        progress.next_batch % plan.accumulation != 0 || progress.epoch_batches != progress.next_batch ||
        (progress.next_batch == 0 && progress.epoch_loss != 0) ||
        (progress.epoch == plan.epochs && progress.next_batch != 0) ||
        int64_t(progress.epoch) * plan.updates_per_epoch + progress.next_batch / plan.accumulation != step) {
        return invalid();
    }
    auto * order = yyjson_obj_get(root, "order");
    if (!yyjson_is_arr(order)) {
        return invalid();
    }
    if (yyjson_arr_size(order) != size_t(plan.dataset_size) &&
        !(progress.next_batch == 0 && yyjson_arr_size(order) == 0)) {
        return invalid();
    }
    std::vector<bool> seen(size_t(plan.dataset_size), false);
    size_t            index, count;
    yyjson_val *      value;
    yyjson_arr_foreach(order, index, count, value) {
        if (!yyjson_is_int(value) || yyjson_get_uint(value) >= size_t(plan.dataset_size)) {
            return invalid();
        }
        size_t item = size_t(yyjson_get_uint(value));
        if (seen[item]) {
            return invalid();
        }
        seen[item] = true;
        progress.order.push_back(item);
    }
    return true;
}

static bool ace_train_read_progress(const std::string &        directory,
                                    const ACETrainResumePlan & plan,
                                    bool                       allow_legacy,
                                    ACETrainResumeProgress &   progress,
                                    int &                      step,
                                    std::string &              error,
                                    bool *                     legacy = nullptr) {
    if (legacy) {
        *legacy = false;
    }
    if (!ace_validate_train_checkpoint_manifest(std::filesystem::u8path(directory), plan.base_fingerprint, error)) {
        return false;
    }
    std::ifstream input(std::filesystem::u8path(directory) / "trainer_state.json", std::ios::binary);
    std::string   contents((std::istreambuf_iterator<char>(input)), {});
    yyjson_doc *  doc        = yyjson_read(contents.data(), contents.size(), 0);
    auto *        root       = doc ? yyjson_doc_get_root(doc) : nullptr;
    auto *        epochs     = yyjson_obj_get(root, "completed_epochs");
    auto *        saved_step = yyjson_obj_get(root, "optimizer_step");
    bool          ok = yyjson_is_int(epochs) && yyjson_get_uint(epochs) <= INT_MAX && yyjson_is_int(saved_step) &&
              yyjson_get_uint(saved_step) <= INT_MAX;
    if (!ok) {
        error = "invalid trainer state counters";
    }
    auto * continuation = yyjson_obj_get(root, "continuation");
    if (ok && !continuation) {
        if (!allow_legacy) {
            error =
                "checkpoint predates resume-plan persistence; use --allow-legacy-resume only with the original "
                "training settings";
            ok = false;
        } else {
            if (legacy) {
                *legacy = true;
            }
            progress       = {};
            progress.epoch = int(yyjson_get_uint(epochs));
            step           = int(yyjson_get_uint(saved_step));
            ok             = progress.epoch <= plan.epochs && int64_t(progress.epoch) * plan.updates_per_epoch == step;
            if (!ok) {
                error = "legacy checkpoint counters do not match this epoch plan";
            }
        }
    } else if (ok) {
        step = int(yyjson_get_uint(saved_step));
        ok   = ace_train_restore_progress(continuation, plan, int(yyjson_get_uint(epochs)), step, progress, error);
    }
    yyjson_doc_free(doc);
    return ok;
}
