#!/usr/bin/env bash
# Pipeline II: cuVSLAM over a ZED SVO2 recording.
#
# ENV: run on the HOST. Drives the Isaac ROS container over `docker exec`; see
#      ../../environments/isaac_ros/ for building and starting it.
#
# Three phases, all inside the container:
#   1. play the SVO2 back through the ZED wrapper and track it with cuVSLAM
#   2. capture the optimised trajectory when playback ends, and parse it
#   3. convert to the common pose schema
#
# Usage:
#   run_zed_svo2.sh <output-name> <svo-file-in-container> [pose-csv]
#
# The SVO2 path is the path *inside the container*. The workspace is normally
# mounted at /workspaces/<name>, so a file at the workspace root is simply
# /workspaces/isaac_ros-dev/test2.svo2.
set -euo pipefail

CONTAINER=${CONTAINER_NAME:-isaac_ros_dev-x86_64-container}
EXEC_USER=${EXEC_USER:-admin}
WS=${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}
RELEASE_IN_CONTAINER=${RELEASE_IN_CONTAINER:-${WS}}

OUTPUT_NAME=${1:?usage: run_zed_svo2.sh <output-name> <svo-file-in-container> [pose-csv]}
SVO_FILE=${2:?}
FINAL_CSV=${3:-}

OUT_REL="slam_output/${OUTPUT_NAME}"
OUT_DIR="${WS}/${OUT_REL}"
POSES_RAW=/tmp/poses_response.txt

# Bring the container up if it is not already, so the driver works from a cold
# machine. Without this, every failure mode below -- container never started,
# container stopped, container removed when its terminal closed -- surfaces as
# the same opaque "No such container" from the docker daemon.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../environments/isaac_ros/ensure-container.sh
source "${HERE}/../../environments/isaac_ros/ensure-container.sh"

in_container() {
  docker exec -i --user "${EXEC_USER}" \
    ${ROS_DOMAIN_ID:+-e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}"} \
    ${ROS_LOCALHOST_ONLY:+-e ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY}"} \
    -e ISAAC_ROS_WS="${RELEASE_IN_CONTAINER}" \
    "${CONTAINER}" bash -lc "$1"
}

# Leftover nodes from an interrupted run hold the ROS graph and the GPU, and the
# next launch then fails in ways that look like a data problem. Clear them first.
preflight() {
  echo "== preflight: clearing any leftover SLAM processes"
  in_container '
    set -eo pipefail
    mapfile -t PROCS < <(ps -eo pid,cmd | grep -E "visual_slam.*launch\.py|component_container|monitor_svo_end\.py|static_transform_publisher" | grep -v grep || true)
    if [[ ${#PROCS[@]} -gt 0 ]]; then
      printf "   killing: %s\n" "${PROCS[@]}"
      mapfile -t PIDS < <(printf "%s\n" "${PROCS[@]}" | awk "{print \$1}")
      for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done
      sleep 5
      for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null || true; done
    fi'
}

launch() {
  echo "== phase 1: playback and tracking (40% of real time)"
  in_container "
    set -eo pipefail
    cd ${WS} && set +u && source install/setup.bash && set -u
    rm -f ${POSES_RAW}
    ros2 launch ${RELEASE_IN_CONTAINER}/extract/cuvslam/launch/isaac_ros_visual_slam_zed_svo2.launch.py \
      svo_file_path:=${SVO_FILE} svo_loop:=${SVO_LOOP:-false}"
}

collect() {
  echo "== phase 2: capturing and parsing the optimised trajectory"
  # Parsing happens INSIDE the container on purpose: /tmp is not shared with the
  # host, so the response file only exists here.
  in_container "
    set -eo pipefail
    cd ${WS} && mkdir -p ${OUT_DIR}
    [ -f ${POSES_RAW} ] || { echo 'no poses response at ${POSES_RAW}; did playback reach the end?' >&2; exit 1; }
    cp ${POSES_RAW} ${OUT_DIR}/poses_response.txt
    python3 ${RELEASE_IN_CONTAINER}/extract/cuvslam/export_poses.py \
      ${OUT_DIR}/poses_response.txt -o ${OUT_DIR}/poses.csv"
}

convert() {
  echo "== phase 3: converting to the common pose schema"
  local target="${FINAL_CSV:-${OUT_DIR}/${OUTPUT_NAME}_cuVSLAM_results.csv}"
  in_container "
    set -eo pipefail
    python3 ${RELEASE_IN_CONTAINER}/extract/common/tracks_to_common_csv.py \
      ${OUT_DIR}/poses.csv -o ${target}"
  echo "== done: ${target}"
}

ensure_container "${ISAAC_ROS_WS_HOST:-}"
preflight
launch
collect
convert
