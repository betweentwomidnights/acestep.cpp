#pragma once
// job-progress.h: what a long pipeline call is doing, for a caller polling
// from another thread.
//
// The thread that runs a job installs a JobProgress with JobProgressScope;
// the VAE encoder, the DiT sampler and the VAE decoder report their tile or
// step into it. With none installed, reports do nothing, so the CLI tools and
// the trainer pay nothing for it.

#include <atomic>

enum JobStage : int {
    JOB_STAGE_QUEUED = 0,  // waiting for the GPU worker
    JOB_STAGE_LOAD,        // started: loading models and preparing inputs
    JOB_STAGE_ENCODE,      // VAE-encoding source audio, per tile
    JOB_STAGE_DIT,         // DiT sampling, per step
    JOB_STAGE_DECODE,      // VAE-decoding the result, per tile
};

struct JobProgress {
    std::atomic<int> stage{ JOB_STAGE_QUEUED };
    std::atomic<int> step{ 0 };   // steps or tiles done in this stage
    std::atomic<int> total{ 0 };  // steps or tiles this stage has
};

inline thread_local JobProgress * g_job_progress = nullptr;

// Record that `step` of `total` are done in `stage`. A reader may briefly see
// one stage's step beside another's total; the next report settles it.
static inline void job_progress_report(JobStage stage, int step, int total) {
    if (JobProgress * progress = g_job_progress) {
        progress->total.store(total, std::memory_order_relaxed);
        progress->step.store(step, std::memory_order_relaxed);
        progress->stage.store(stage, std::memory_order_relaxed);
    }
}

// Installs `progress` for the calling thread until the scope ends.
struct JobProgressScope {
    explicit JobProgressScope(JobProgress * progress) : previous(g_job_progress) { g_job_progress = progress; }

    ~JobProgressScope() { g_job_progress = previous; }

    JobProgressScope(const JobProgressScope &)             = delete;
    JobProgressScope & operator=(const JobProgressScope &) = delete;

    JobProgress * previous;
};

static inline const char * job_stage_name(int stage) {
    switch (stage) {
        case JOB_STAGE_LOAD:
            return "load";
        case JOB_STAGE_ENCODE:
            return "encode";
        case JOB_STAGE_DIT:
            return "dit";
        case JOB_STAGE_DECODE:
            return "decode";
        default:
            return "queued";
    }
}
