#!/bin/sh
# ABOUTME: Builds acestep against whatever ggml revision is checked out and
# ABOUTME: runs its test suite, for pull requests that move the pointer.
#
# The shared ggml fork backs several projects (audiocraft.cpp, sa3.cpp,
# acestep.cpp, yue2.cpp), so a change there can break a consumer that the
# change's author never builds. Every consumer carries this script under the
# same path, which lets one job validate any of them without knowing how any
# of them are built: check the repo out, point its ggml at the revision under
# test, run ci/ggml-check.sh.
#
# CPU only and no model weights, so it runs on an ordinary hosted runner. That
# means a backend-specific change compiles here but is not exercised: the CUDA
# and Metal paths still need a machine that has one. What this does catch is
# the common case, a ggml change that alters an interface or a behaviour some
# consumer depends on.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

build=${BUILD_DIR:-build-ggml-check}
jobs=${JOBS:-}
if [ -z "$jobs" ]; then
    jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
fi

if [ ! -f ggml/CMakeLists.txt ]; then
    echo "ci/ggml-check.sh: ggml/ is empty; clone with --recurse-submodules" >&2
    exit 1
fi

# Say which revision is under test. On a bump PR this is the whole point of the
# run, and it is the first thing worth seeing in the log.
printf 'acestep.cpp  %s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
printf 'ggml         %s\n' "$(git -C ggml rev-parse HEAD 2>/dev/null || echo unknown)"

# ACE_TEST_MODELS_DIR and ACE_TEST_CUDA are both left unset, which is what
# keeps this to the nine self-contained tests. tests/CMakeLists.txt already
# draws those lines, so this only has to not cross them.
# GGML_METAL has to be turned off rather than merely left alone. ggml defaults
# it ON for Apple, and ACE_METAL=OFF only declines to force it ON, so a macOS
# runner would build and run Metal while this script claims to be CPU only.
# That aborted three of audiocraft.cpp's tests on a virtualised runner.
cmake -S . -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DGGML_METAL=OFF
cmake --build "$build" --config Release -j "$jobs"

# --output-on-failure so a red test explains itself in the job log rather than
# only naming which one died.
ctest --test-dir "$build" --build-config Release --output-on-failure
