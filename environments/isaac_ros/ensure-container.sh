#!/usr/bin/env bash
# Make sure the Isaac ROS container is up, starting it if it is not.
#
# Sourced by the pipeline drivers so they can be run from a cold machine. Safe
# to call when the container is already running.
#
# Three cases are handled separately, because they need different actions and
# because "docker exec" fails identically for all of them with a message that
# says nothing useful:
#
#   running          use it
#   exists, stopped  docker start
#   does not exist   launch it
#
# The third case needs care. Isaac ROS's run_dev.sh runs `docker run -it --rm`
# with /bin/bash as the command and no way to pass a different one, so the
# container lives exactly as long as that interactive shell -- which is why it
# disappears the moment you close the terminal you started it from. To get a
# container that survives for a scripted run, it is launched under a pty with
# its stdin held open by a FIFO, detached from this shell.
set -uo pipefail

CONTAINER="${CONTAINER_NAME:-isaac_ros_dev-x86_64-container}"
IMAGE_KEY="${IMAGE_KEY:-ros2_humble.realsense.zed.custom}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"

container_state() {
  if [ -n "$(docker ps --quiet --filter status=running --filter "name=^/${CONTAINER}$" 2>/dev/null)" ]; then
    echo running
  elif [ -n "$(docker ps --all --quiet --filter "name=^/${CONTAINER}$" 2>/dev/null)" ]; then
    echo stopped
  else
    echo absent
  fi
}

container_ready() {
  # The container is ready only when its entrypoint has FINISHED.
  #
  # workspace-entrypoint.sh ends with `exec gosu admin "$@"`, so PID 1 becomes
  # the container command once setup is complete. Until then PID 1 is still the
  # entrypoint, and the workspace is being rebuilt underneath you: BOTH
  # entrypoint additions run a colcon build on every start --
  # 01_my_setup.user.sh builds isaac_ros_visual_slam and zed-entrypoint.sh
  # builds zed_wrapper. That takes minutes.
  #
  # Checking for install/setup.bash instead is not enough, and is an easy
  # mistake: the file is left over from the previous build, so it exists
  # immediately while the libraries beside it are being replaced. Launching
  # then fails with missing .so files and missing Python modules, which look
  # like a broken image rather than a race.
  local pid1
  pid1="$(docker exec "${CONTAINER}" ps -p 1 -o comm= 2>/dev/null | tr -d '[:space:]')" || return 1
  case "${pid1}" in
    sleep|"${CONTAINER_CMD_COMM:-sleep}") ;;
    *) return 1 ;;
  esac
  docker exec "${CONTAINER}" bash -lc \
    '[ -f "${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/install/setup.bash" ]' >/dev/null 2>&1
}

launch_container() {
  local ws="${1:?workspace path required}"
  local image="${IMAGE_NAME:-isaac_ros_dev-x86_64:latest}"

  if ! docker image inspect "${image}" >/dev/null 2>&1; then
    echo "error: image '${image}' not found. Build it first:" >&2
    echo "         cd environments/isaac_ros && ./run-container.sh" >&2
    echo "       (that is Isaac ROS's own run_dev.sh, which builds and enters" >&2
    echo "        the container interactively.)" >&2
    return 1
  fi

  # Started DETACHED with `sleep infinity`, rather than through run_dev.sh.
  #
  # run_dev.sh runs `docker run -it --rm` with /bin/bash as the command and no
  # way to pass another, so its container lives exactly as long as the
  # interactive shell that started it -- close the terminal and the pipeline
  # loses its container mid-run. For a scripted pipeline the container has to
  # outlive the shell that asked for it.
  #
  # The arguments below mirror run_dev.sh's x86_64 path (its Jetson-only mounts
  # are omitted). The two ZED mounts are required: the SDK reads its per-camera
  # calibration and neural resources from the host.
  #
  # ROS_DOMAIN_ID is INHERITED from the host, never defaulted here. run_dev.sh
  # passes it bare (`-e ROS_DOMAIN_ID`) for the same reason. Forcing a value
  # instead puts the container on a different DDS domain from the shell that
  # started it, and DDS domains are isolated -- so `ros2 topic list` on the host
  # shows only its own /parameter_events and /rosout while the pipeline runs
  # perfectly well, unseen, next door. That is a genuinely confusing failure and
  # it costs nothing to avoid.
  echo "   starting '${CONTAINER}' detached from ${image}"
  docker run -d \
    --name "${CONTAINER}" \
    --privileged --network host --ipc=host \
    --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e ISAAC_ROS_WS=/workspaces/isaac_ros-dev \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    -e USER="${USER:-$(id -un)}" \
    -e HOST_USER_UID="$(id -u)" \
    -e HOST_USER_GID="$(id -g)" \
    ${DISPLAY:+-e DISPLAY="${DISPLAY}"} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "${ws}":/workspaces/isaac_ros-dev \
    -v /etc/localtime:/etc/localtime:ro \
    -v /usr/local/zed/settings:/usr/local/zed/settings \
    -v /usr/local/zed/resources:/usr/local/zed/resources \
    --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
    --workdir /workspaces/isaac_ros-dev \
    "${image}" \
    sleep infinity >/dev/null || {
      echo "error: docker run failed" >&2
      return 1
    }

  local waited=0
  while [ "${waited}" -lt "${READY_TIMEOUT}" ]; do
    if [ "$(container_state)" = running ] && container_ready; then
      echo "   container up after ${waited}s"
      return 0
    fi
    if [ "$(container_state)" = absent ]; then
      echo "error: the container exited during startup. Logs:" >&2
      docker logs "${CONTAINER}" 2>&1 | tail -20 >&2
      return 1
    fi
    if [ $((waited % 30)) -eq 0 ] && [ "${waited}" -gt 0 ]; then
      echo "   still initialising (${waited}s) -- the entrypoint rebuilds"
      echo "   isaac_ros_visual_slam and zed_wrapper on every start"
    fi
    sleep 3; waited=$((waited + 3))
  done

  echo "error: container did not become ready within ${READY_TIMEOUT}s. Logs:" >&2
  docker logs "${CONTAINER}" 2>&1 | tail -20 >&2
  return 1
}

check_domain_match() {
  local host="${ROS_DOMAIN_ID:-0}" inside
  inside="$(docker exec "${CONTAINER}" printenv ROS_DOMAIN_ID 2>/dev/null || echo 0)"
  inside="${inside:-0}"
  if [ "${host}" != "${inside}" ]; then
    echo "   warning: this shell is on ROS_DOMAIN_ID=${host} but the running container" >&2
    echo "            is on ${inside}. DDS domains are isolated, so 'ros2 topic list'" >&2
    echo "            here will not show the pipeline's topics. Recreate the container" >&2
    echo "            with a matching domain:  docker rm -f ${CONTAINER}" >&2
  fi
}

ensure_container() {
  local ws="${1:-${ISAAC_ROS_WS_HOST:-}}"
  local state
  state="$(container_state)"
  echo "== container '${CONTAINER}': ${state}"

  case "${state}" in
    running)
      check_domain_match
      if container_ready; then return 0; fi
      echo "   running but the workspace is not built; run environments/isaac_ros/build-workspace.sh" >&2
      return 1
      ;;
    stopped)
      echo "   starting it"
      docker start "${CONTAINER}" >/dev/null || return 1
      ;;
    absent)
      [ -n "${ws}" ] || {
        echo "error: the container does not exist and no workspace path was given." >&2
        echo "       Pass one, or set ISAAC_ROS_WS_HOST." >&2
        return 1
      }
      launch_container "${ws}" || return 1
      ;;
  esac

  local waited=0
  while [ "${waited}" -lt "${READY_TIMEOUT}" ]; do
    container_ready && { echo "   ready after ${waited}s"; return 0; }
    if [ $((waited % 30)) -eq 0 ] && [ "${waited}" -gt 0 ]; then
      echo "   still initialising (${waited}s)"
    fi
    sleep 3; waited=$((waited + 3))
  done
  echo "error: container is up but ${ISAAC_ROS_WS:-/workspaces/isaac_ros-dev}/install/setup.bash is missing." >&2
  echo "       Build the workspace: environments/isaac_ros/build-workspace.sh" >&2
  return 1
}

# Allow running this file directly as a command, as well as sourcing it.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  ensure_container "${1:-}"
fi
