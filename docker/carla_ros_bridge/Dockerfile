ARG CARLA_VERSION
FROM carlasim/carla:${CARLA_VERSION} as carla_api

FROM ros:noetic-ros-base
SHELL [ "/bin/bash", "-c" ]
ARG CARLA_VERSION

# Set up apt repo
RUN apt-get update && \
    apt-get install -y lsb-release software-properties-common python-is-python3 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install catkin build system
RUN apt-get update && \
    apt-get install -y build-essential python3-catkin-tools wget curl python3-rosdep python3-rospkg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Carla Python API
RUN apt-get update && \
    apt-get install -y python3-pip python3-pygame && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=carla_api --chown=root /home/carla/PythonAPI /opt/carla/PythonAPI
WORKDIR /opt/carla/PythonAPI
RUN python3 -m pip install --upgrade pip
RUN if [[ "$CARLA_VERSION" > "0.9.11" ]]; then \
    python3 -m pip install "carla==$CARLA_VERSION"; \
    fi

# Add a docker user so we that created files in the docker container are owned by a non-root user
RUN addgroup --gid 1000 docker && \
    adduser --uid 1000 --ingroup docker --home /home/docker --shell /bin/bash --disabled-password --gecos "" docker && \
    echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# Remap the docker user and group to be the same uid and group as the host user.
# Any created files by the docker container will be owned by the host user.
RUN USER=docker && \
    GROUP=docker && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\npaths:\n  - /home/docker/" > /etc/fixuid/config.yml
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

USER docker:docker

ENV PYTHONPATH "${PYTHONPATH}:/opt/carla/PythonAPI/carla/dist/carla-${CARLA_VERSION}-py3.7-linux-x86_64.egg"
RUN if [ "${CARLA_VERSION}" = "0.9.10.1" ]; then \
      echo "export PYTHONPATH=$PYTHONPATH:/opt/carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg" >> ~/.bashrc ; \
    fi
ENV DEBIAN_FRONTEND noninteractive
RUN sudo chsh -s /bin/bash
ENV SHELL=/bin/bash

# ================= Repositories ===================
# Set up workspace
RUN mkdir -p ~/catkin_ws/src
WORKDIR /home/docker/catkin_ws/src

RUN if [ "${CARLA_VERSION}" = "0.9.13" ]; then \
      git clone --depth 1 --recursive --branch master https://github.com/carla-simulator/ros-bridge.git ; \
    else \
      git clone --depth 1 --recursive --branch "${CARLA_VERSION}" https://github.com/carla-simulator/ros-bridge.git ; \
    fi

# Download source dependencies
RUN sudo rm /etc/ros/rosdep/sources.list.d/20-default.list || true && \
    sudo rosdep init && \
    sudo apt-get update && \
    rosdep update && \
    rosdep install --from-paths . --ignore-src -r -y

# Fix for missing rviz and pcl_ros
RUN sudo apt-get install -y ros-noetic-rviz ros-noetic-pcl-ros

WORKDIR /home/docker/catkin_ws
RUN catkin config --extend /opt/ros/noetic && \
    catkin build && \
    rm -rf .catkin_tools build

# Entrypoint will run before any CMD on launch. Sources ~/opt/<ROS_DISTRO>/setup.bash and ~/catkin_ws/devel/setup.bash
COPY ./ros_entrypoint.sh /home/docker/ros_entrypoint.sh
RUN sudo chmod +x ~/ros_entrypoint.sh
ENTRYPOINT ["/home/docker/ros_entrypoint.sh"]

# ================= Develop ===================
USER root:root
# Supervisor to manage multiple processes (code-server, roslaunch)
RUN apt-get install -y wget curl gdb supervisor

COPY --chown=docker ./carla_ros_bridge/supervisord.conf /etc/supervisor/supervisord.conf
RUN chown -R docker:docker /etc/supervisor
RUN chmod 777 /var/log/supervisor/

USER docker:docker
RUN echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc

CMD ["/usr/bin/supervisord"]