#!/usr/bin/env bash
#
# mayhem/test.sh — RUN this repo's OWN functional test suite (already built by mayhem/build.sh).
# exit 0 = pass. EDIT per repo. PATCH-grade oracle: after an agent patches the source, the grader
# rebuilds (build.sh) then runs this. DELETE this file if the repo has no meaningful tests.
#
# IMPORTANT:
#  * Must assert BEHAVIOR/OUTPUT, not just exit status. The oracle has to check asserted values /
#    golden-output diffs / known-answer results — so a PATCH that "fixes" a bug by making the program
#    exit(0) (or any no-op) FAILS here. Running inputs and checking only "exit 0 / didn't crash" is
#    NOT a functional test (it's trivially reward-hackable) — use the project's real assertion suite.
#  * Do NOT build here — mayhem/build.sh already compiled the test suite (with the project's normal
#    flags). This script only RUNS the pre-built tests and reports counts. If the test runner is
#    missing, that's a build.sh bug — fail loudly rather than silently rebuilding.
#  * REQUIRED OUTPUT — a CTRF (https://ctrf.io) summary so Mayhem/the PATCH grader reads the counts:
#      - writes a CTRF JSON report to ${CTRF_REPORT:-$SRC/ctrf-report.json}, and
#      - prints a one-line `CTRF {...}` marker to stdout (same JSON, compact).
#    Only `results.summary` (with tests/passed/failed/pending/skipped/other) is required.
#    Use the emit_ctrf helper below; it computes tests = passed+failed+skipped and sets the exit
#    code (0 iff failed==0). Map your framework's output to passed/failed/skipped.
set -uo pipefail
[ -n "${SOURCE_DATE_EPOCH:-}" ] || unset SOURCE_DATE_EPOCH
: "${MAYHEM_JOBS:=$(nproc)}"   # build parallelism; env-overridable, falls back to nproc (use -j"$MAYHEM_JOBS")
cd "$SRC"

# emit_ctrf <tool> <passed> <failed> [skipped] [pending] [other]
# Writes a CTRF report (file + stdout `CTRF {...}` marker) and returns non-zero iff failed>0.
emit_ctrf() {
  local tool="$1" passed="$2" failed="$3" skipped="${4:-0}" pending="${5:-0}" other="${6:-0}"
  local tests=$(( passed + failed + skipped + pending + other ))
  cat > "${CTRF_REPORT:-$SRC/ctrf-report.json}" <<JSON
{
  "results": {
    "tool": { "name": "$tool" },
    "summary": {
      "tests": $tests,
      "passed": $passed,
      "failed": $failed,
      "pending": $pending,
      "skipped": $skipped,
      "other": $other
    }
  }
}
JSON
  printf 'CTRF {"results":{"tool":{"name":"%s"},"summary":{"tests":%d,"passed":%d,"failed":%d,"pending":%d,"skipped":%d,"other":%d}}}\n' \
    "$tool" "$tests" "$passed" "$failed" "$pending" "$skipped" "$other"
  [ "$failed" -eq 0 ]
}

# Oracle: nanobind's own pytest suite, run by the pre-built nb_python launcher (build.sh, step 3) against the
# normal-flags extensions in build/tests/tests. Run via our launcher, not /usr/bin/python3, so that
# neutering the project's binaries makes the suite produce no pytest summary at all.
PY=/mayhem/build/nb_python
TESTDIR="$SRC/build/tests/tests"
[ -x "$PY" ] || { echo "missing $PY: mayhem/build.sh did not run" >&2; emit_ctrf "pytest" 0 1 0; exit 1; }
[ -d "$TESTDIR" ] || { echo "missing $TESTDIR: mayhem/build.sh did not run" >&2; emit_ctrf "pytest" 0 1 0; exit 1; }

# Known-answer probes: values asserted here come from nanobind's documented binding behaviour.
probe_fail=0
probe() {  # probe <name> <expected stdout> <python code>
  local got
  got="$(cd "$TESTDIR" && "$PY" -c "$3" 2>&1)"
  if [ "$got" = "$2" ]; then echo "probe $1: ok"; else echo "probe $1: FAIL (expected '$2', got '$got')"; probe_fail=$((probe_fail+1)); fi
}
probe functions "2 44 -2" 'import test_functions_ext as t; print(t.test_02(5, 3), t.test_03(5, 3), t.test_02(3, 5))'
probe overload "1 2" 'import test_functions_ext as t; print(t.test_05(0), t.test_05(0.0))'
probe stl "['abc', None]" 'import test_stl_ext as t; print(t.vector_optional_str(["abc", None]))'
probe error "TypeError" 'import test_functions_ext as t
try: t.test_02(unexpected=27)
except TypeError: print("TypeError")'

out="$(cd "$TESTDIR" && "$PY" -m pytest -q -p no:cacheprovider --color=no . 2>&1)"; rc=$?
printf '%s\n' "$out" | tail -n 25
summary="$(printf '%s\n' "$out" | grep -E '^=*[ ]*([0-9]+ (passed|failed|skipped|xfailed|xpassed|error|errors|deselected|warnings?)(, )?)+ in ' | tail -n 1)"
count() { printf '%s\n' "$summary" | grep -oE "[0-9]+ $1" | head -n 1 | grep -oE '^[0-9]+' || true; }
passed="$(count passed)"; failed="$(count failed)"; errors="$(count 'errors?')"; skipped="$(count skipped)"
passed=${passed:-0}; failed=${failed:-0}; errors=${errors:-0}; skipped=${skipped:-0}
failed=$(( failed + errors + probe_fail ))
# A run with no pytest summary (crash, early exit) counts as a failure, never as an empty pass.
if [ -z "$summary" ] || { [ "$rc" -ne 0 ] && [ "$failed" -eq 0 ]; }; then failed=$(( failed + 1 )); fi
emit_ctrf "pytest" $(( passed + 4 - probe_fail )) "$failed" "$skipped"
