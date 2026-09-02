#!/usr/bin/env bash
# Phase-0 baseline: run the UNMODIFIED original evaluation into a scratchpad mirror.
#
# One process per test, because process_zed_vicon_calibration leaks a matplotlib
# figure on every call (SynchronizationNew.py:2230 calls plt.figure() and never
# closes it) and stage 3 invokes it ~100 times per test.  A single process for all
# four tests reaches >2 GB within minutes and would OOM.  Per-test processes bound
# peak RSS to one test's worth and reset it in between.
#
# This is not a deviation from how the published numbers were made: the reference
# artifacts are themselves timestamped in two groups (test2/3/5 at 16:37-16:52 and
# test1 at 18:00-18:01 on 2026-08-18), so the published run was already split by
# test.  Unification is within a test in any case.
#
# Writes nothing outside $MIRROR.  PYTHONDONTWRITEBYTECODE=1 keeps __pycache__ in
# the read-only source tree untouched (its Aug-26 bytecode is forensic evidence).
set -uo pipefail

MIRROR="${1:?usage: run_baseline.sh <mirror-dir> <log-prefix>}"
LOGPRE="${2:?}"
PROJ="${LEGACY_PROJECT_DIR:?set LEGACY_PROJECT_DIR to the original ZED_Project checkout}"
DATA="${LEGACY_DATA_DIR:-$PROJ/Zohar_Experiment/23_10_25/data}"

cd "$PROJ" || exit 1

for T in 2 3 5 1; do
  echo "### test$T starting $(date +%H:%M:%S)"
  env -u DISPLAY -u WAYLAND_DISPLAY \
      PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      ./.venv/bin/python3 plot_path_unified.py \
        --results-dir "$MIRROR/results" --data-dir "$DATA" \
        --tests "$T" --pipelines 1,2,3 --unified --noplot \
      > "${LOGPRE}-test${T}.log" 2>&1 &
  pid=$!
  peak=0
  while kill -0 "$pid" 2>/dev/null; do
    r=$(awk '/VmRSS/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
    [ -n "$r" ] && [ "$r" -gt "$peak" ] 2>/dev/null && peak=$r
    sleep 5
  done
  wait "$pid"; rc=$?
  n=$(find "$MIRROR/results/comparison" -name '*_unified_angles.csv' 2>/dev/null | wc -l)
  echo "### test$T done $(date +%H:%M:%S) rc=$rc peakRSS=$((peak/1024))MB cells_total=$n"
  [ "$rc" -ne 0 ] && { echo "### ABORT: test$T exited $rc"; tail -25 "${LOGPRE}-test${T}.log"; exit "$rc"; }
done
echo "### ALL TESTS COMPLETE"
