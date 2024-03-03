#!/usr/bin/env bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/home/docker/catkin_ws/devel/setup.bash"

exec "/usr/local/bin/fixuid" "-q" "$@"