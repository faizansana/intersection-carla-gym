FROM osrf/ros:noetic-desktop-full
SHELL [ "/bin/bash", "-c" ]

# Install foxglove bridge
RUN apt-get update && \
    apt-get install ros-noetic-foxglove-bridge -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

