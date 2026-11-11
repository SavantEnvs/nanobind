#!/usr/bin/env bash
#
# mayhem/build.sh — build this repo's fuzz harness(es). EDIT per repo.
#
# Runs inside the commit image (mayhem/Dockerfile) as `mayhem` in /mayhem. The base image
# (ghcr.io/savantenvs/base) already exports the build contract — use these, don't redefine:
#   CC, CXX             stock clang / clang++
#   LIB_FUZZING_ENGINE  -fsanitize=fuzzer   (link into each harness that has a LLVMFuzzer entry)
#   SANITIZER_FLAGS     -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
#                       (ASan + UBSan, both set to HALT — so Mayhem catches memory AND UB defects)
#   DEBUG_FLAGS         -g -gdwarf-3   (DWARF debug info — always on for fuzz/standalone builds,
#                       independent of the sanitizer off-switch; DWARF version must be < 4)
#   RUST_DEBUG_FLAGS    -C debuginfo=2 -C force-frame-pointers=yes -C llvm-args=-gdwarf-3
#                       (thread through RUSTFLAGS on every cargo-fuzz build)
#   GO_DEBUG_FLAGS      -gcflags=all=-N -l
#                       (thread through go build / go-fuzz-build so the linked ELF keeps symbols)
#   SRC                 /mayhem (the repo source)
#
# Contract: build EVERYTHING here — one runnable binary per fuzz harness, AND the project's test
# suite (so mayhem/test.sh only has to RUN it, never compile). Keep it ADDITIVE (build upstream as
# upstream documents; don't edit upstream files). IMPORTANT: build the PROJECT ITSELF with
# $SANITIZER_FLAGS and $DEBUG_FLAGS (not just the harness) so the fuzzed code is instrumented
# AND carries DWARF < 4 symbols — otherwise ASan/UBSan only see the harness, not the library
# you're trying to find bugs in, and backtraces won't resolve project source lines. Build the TEST
# suite with the project's NORMAL flags (a clean, independent build) so test.sh stays an honest
# functional oracle and won't false-fail on benign UB. Leave the test binary/runner where test.sh
# expects it.
set -euo pipefail

# clang rejects SOURCE_DATE_EPOCH='' (empty) — it must be unset or a valid integer.
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH

# Build knobs come from the ENVIRONMENT (overridable), with sane defaults — no if-statements,
# just parameter-expansion fallbacks. Default sanitizers are the base's ASan+UBSan (halting); override
# per build via the Dockerfile's `--build-arg SANITIZER_FLAGS="..."`.
# NB: SANITIZER_FLAGS uses `=` (no colon) on purpose — `=` only fills when the var is UNSET, so an
# explicit EMPTY value (`--build-arg SANITIZER_FLAGS=`) is honored and builds with NO sanitizers
# (useful when you want the program's natural crash / full backtrace, not an ASan report). The other
# knobs use `:=` (default on empty too). MAYHEM_JOBS sets build parallelism (falls back to nproc).
: "${SANITIZER_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer}"
# DEBUG_FLAGS carries DWARF debug info INDEPENDENTLY of the sanitizer off-switch (so an empty
# SANITIZER_FLAGS still yields DWARF symbols). DWARF MUST be < 4 (Mayhem triage can't read >=4); clang-19's
# plain `-g` emits DWARF-5, so `-gdwarf-3` is explicit. Apply $DEBUG_FLAGS to the fuzz/harness/standalone
# builds (NOT the test/oracle build). Rust/Go carry the same intent via their language flags below.
: "${DEBUG_FLAGS:=-g -gdwarf-3}"
: "${RUST_DEBUG_FLAGS:=-C debuginfo=2 -C force-frame-pointers=yes -C llvm-args=-gdwarf-3}"
: "${GO_DEBUG_FLAGS:=-gcflags=all=-N -l}"
: "${CC:=clang}" ; : "${CXX:=clang++}" ; : "${LIB_FUZZING_ENGINE:=-fsanitize=fuzzer}"
: "${MAYHEM_JOBS:=$(nproc)}"
# COVERAGE_FLAGS: empty by default → no effect on the normal oracle build. Set it via the Dockerfile's
# `--build-arg COVERAGE_FLAGS="-fprofile-instr-generate -fcoverage-mapping"` to instrument the TEST
# build for source-coverage measurement (how much of the project the test suite actually exercises —
# a quality signal for the oracle; complements the anti-reward-hack sabotage check). APPEND it to the
# test build's compile+link flags in step 3 (NOT the fuzz build); after `test.sh` runs, merge with
# `llvm-profdata` and report with `llvm-cov`. Empty value is honored (`=`, not `:=`).
: "${COVERAGE_FLAGS=}"
export SANITIZER_FLAGS DEBUG_FLAGS RUST_DEBUG_FLAGS GO_DEBUG_FLAGS CC CXX LIB_FUZZING_ENGINE MAYHEM_JOBS COVERAGE_FLAGS

cd "$SRC"

# nanobind is a C++ binding library, so its fuzz target is an embedded-CPython libFuzzer harness that drives
# the extension modules built from nanobind's own tests/test_*.cpp (see mayhem/fuzz_nanobind.cpp).
ROBIN_MAP=/opt/toolchains/deps/robin-map-1.4.0   # pinned in the Dockerfile; ext/robin_map is an empty submodule
PY_INCLUDES="$(python3-config --includes)"
PY_LDFLAGS="$(python3-config --embed --ldflags)"
NB_CMAKE=(-G Ninja -DNB_USE_SUBMODULE_DEPS=OFF -DCMAKE_PREFIX_PATH="$ROBIN_MAP" -DNB_TEST=ON
          -DNB_TEST_FREE_THREADED=OFF -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX")

# Test extensions the fuzzers import (tests/CMakeLists.txt names them test_<name>_ext).
FUZZ_MODULES=(functions accessor callbacks exception enum make_iterator issue chrono classes holders
              intrusive stl stl_bind_map stl_bind_vector ndarray jax tensorflow eigen)
FUZZ_TARGETS=("${FUZZ_MODULES[@]/%/_ext}")
FUZZ_TARGETS=("${FUZZ_TARGETS[@]/#/test_}")

# 1) The project itself, sanitized: nanobind's runtime plus the test extensions, ASan+UBSan+SanCov so that
#    the fuzzed code (src/*.cpp, include/nanobind/**) is instrumented. Debug keeps nanobind_strip from
#    stripping the DWARF out of the modules; -O1 keeps fuzzing fast. -fno-sanitize=function: nanobind
#    deliberately calls the intrusive-pointer callbacks (src/nb_func.cpp, nb::intrusive_ptr) through a
#    `void (*)(void *, PyObject *)` cast of a lambda taking the derived type; UBSan's function check flags
#    that on the first use of any intrusive class and would stop every run there.
cmake -S . -B build/fuzz "${NB_CMAKE[@]}" -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_FLAGS="-fsanitize=fuzzer-no-link $SANITIZER_FLAGS -fno-sanitize=function $DEBUG_FLAGS" \
    -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link $SANITIZER_FLAGS -fno-sanitize=function $DEBUG_FLAGS" \
    -DCMAKE_C_FLAGS_DEBUG="-O1" -DCMAKE_CXX_FLAGS_DEBUG="-O1" \
    -DCMAKE_SHARED_LINKER_FLAGS="$SANITIZER_FLAGS" -DCMAKE_MODULE_LINKER_FLAGS="$SANITIZER_FLAGS"
cmake --build build/fuzz -j"$MAYHEM_JOBS" --target "${FUZZ_TARGETS[@]}"

# 2) The harness, three times (one per extension group), each as a fuzzer and as a standalone reproducer.
$CC $SANITIZER_FLAGS $DEBUG_FLAGS -c "$STANDALONE_FUZZ_MAIN" -o /tmp/standalone_main.o
$CXX $SANITIZER_FLAGS $DEBUG_FLAGS -c "$SRC/mayhem/lsan_off.cc" -o /tmp/lsan_off.o
for group in functions stl ndarray; do
  for mode in fuzzer standalone; do
    if [ "$mode" = fuzzer ]; then
      engine=($LIB_FUZZING_ENGINE); out=/mayhem/fuzz_$group
    else
      engine=(/tmp/standalone_main.o); out=/mayhem/fuzz_$group-standalone
    fi
    $CXX -std=c++17 -O1 $SANITIZER_FLAGS $DEBUG_FLAGS -DNBFUZZ_GROUP="\"$group\"" $PY_INCLUDES \
        "$SRC/mayhem/fuzz_nanobind.cpp" /tmp/lsan_off.o "${engine[@]}" $PY_LDFLAGS -o "$out"
  done
done

# 3) The test suite, normal flags and a separate tree. nb_python is a tiny launcher (Py_BytesMain) so the
#    suite runs under a binary of ours instead of /usr/bin/python3; test.sh then depends on our build.
cmake -S . -B build/tests "${NB_CMAKE[@]}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="$COVERAGE_FLAGS" -DCMAKE_CXX_FLAGS="$COVERAGE_FLAGS"
cmake --build build/tests -j"$MAYHEM_JOBS"
$CC -O2 $COVERAGE_FLAGS $PY_INCLUDES "$SRC/mayhem/nb_python.c" $PY_LDFLAGS -o build/nb_python
