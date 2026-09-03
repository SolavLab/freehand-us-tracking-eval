#!/usr/bin/env bash
# Pipeline III: cuVSLAM over a converted RealSense ROS 2 bag.
#
# ENV: run on the HOST. Drives the Isaac ROS container over `docker exec`; see
#      ../../environments/isaac_ros/ for building and starting it.
#
# Same three phases as run_zed_svo2.sh, differing only in what is played back:
# a rosbag2 directory produced by ../convert/realsense_bag_to_rosbag2.py rather
# than an SVO2 file. Playback is at 40% of real time with the original
# timestamps preserved, for the same reason.
#
# Usage:
#   run_realsense_bag.sh <output-name> <bag-dir-in-container> [pose-csv]
#
# Note on the original: its post-run validation was inside an unquoted heredoc,
# so `${TOPIC_LIST}` expanded to a single array element and the generated Python
# was a syntax error -- the check never ran, and under `set -e` it took the
# script's exit status with it. The equivalent check here is a normal function.
set -euo pipefail

CONTAINER=${CONTAINER_NAME:-isaac_ros_dev-x86_64-container}
EXEC_USER=${EXEC_USER:-admin}
WS=${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}
RELEASE_IN_CONTAINER=${RELEASE_IN_CONTAINER:-${WS}}

OUTPUT_NAME=${1:?usage: run_realsense_bag.sh <output-name> <bag-dir-in-container> [pose-csv]}
BAG_PATH=${2:?}
FINAL_CSV=${3:-}

OUT_DIR="${WS}/slam_output/${OUTPUT_NAME}"
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
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    -e ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}" \
    -e ISAAC_ROS_WS="${RELEASE_IN_CONTAINER}" \
    "${CONTAINER}" bash -lc "$1"
}

preflight() {
  echo "== preflight: clearing any leftover SLAM processes"
  in_container '
    set -eo pipefail
    mapfile -t PROCS < <(ps -eo pid,cmd | grep -E "visual_slam.*launch\.py|component_container|ros2 bag play|static_transform_publisher" | grep -v grep || true)
    if [[ ${#PROCS[@]} -gt 0 ]]; then
      printf "   killing: %s\n" "${PROCS[@]}"
      mapfile -t PIDS < <(printf "%s\n" "${PROCS[@]}" | awk "{print \$1}")
      for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done
      sleep 5
      for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null || true; done
    fi'
}

check_bag() {
  echo "== checking the input bag"
  in_container "
    set -eo pipefail
    [ -d ${BAG_PATH} ] || { echo 'no such bag directory: ${BAG_PATH}' >&2; exit 1; }
    [ -f ${BAG_PATH}/metadata.yaml ] || { echo 'not a rosbag2 directory (no metadata.yaml): ${BAG_PATH}' >&2; exit 1; }
    python3 - <<'PY'
import sys, yaml
m = yaml.safe_load(open('${BAG_PATH}/metadata.yaml'))['rosbag2_bagfile_information']
counts = {t['topic_metadata']['name']: t['message_count'] for t in m['topics_with_message_count']}
duration = m['duration']['nanoseconds'] / 1e9
required = ['/camera/left/image_raw', '/camera/right/image_raw', '/imu']
missing = [t for t in required if t not in counts]
if missing:
    sys.exit('bag is missing required topics: %s' % missing)
imu_hz = counts['/imu'] / duration
img_hz = counts['/camera/left/image_raw'] / duration
print('   %.1f s, images %.1f Hz, IMU %.1f Hz' % (duration, img_hz, imu_hz))
# A bag whose IMU was truncated during conversion will still play and still
# produce a trajectory -- just a worse one, silently. Refuse it here.
if imu_hz < 200:
    sys.exit('IMU is only %.1f Hz; expected about 400. The bag was truncated '
             'during conversion -- see extract/README.md.' % imu_hz)
PY"
}

launch() {
  echo "== phase 1: playback and tracking (40% of real time)"
  in_container "
    set -eo pipefail
    cd ${WS} && set +u && source install/setup.bash && set -u
    rm -f ${POSES_RAW}
    ros2 launch ${RELEASE_IN_CONTAINER}/extract/cuvslam/launch/isaac_ros_visual_slam_rosbag.launch.py \
      rosbag_path:=${BAG_PATH}"
}

collect() {
  echo "== phase 2: capturing and parsing the optimised trajectory"
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
check_bag
launch
collect
convert
