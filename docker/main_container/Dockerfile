ARG CARLA_VERSION=0.9.10.1
ARG CUDA_VERSION=12.0.0
FROM carlasim/carla:${CARLA_VERSION} AS carla

ARG CUDA_VERSION=12.0.0
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu20.04 AS base
SHELL [ "/bin/bash", "-c" ]

ARG CARLA_VERSION=0.9.13

ENV DEBIAN_FRONTEND noninteractive
ENV CARLA_VERSION $CARLA_VERSION
RUN apt-get update && \
    apt-get install -y sudo curl wget git python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install ROS Noetic Desktop Full
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-get update && \
    apt-get install -y --no-install-recommends ros-noetic-desktop-full && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Install ROS dependencies for building packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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

# Install vnc server
RUN apt-get update && \
    apt-get install -y lxde x11vnc xvfb mesa-utils supervisor && \
    apt-get purge -y light-locker && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --chown=docker ./main_container/supervisord.conf /etc/supervisor/supervisord.conf
RUN chown -R docker:docker /etc/supervisor
RUN chmod 777 /var/log/supervisor/

COPY --from=carla /home/carla/PythonAPI /opt/carla/PythonAPI

# Setup python environment
USER docker:docker
ENV SHELL=/bin/bash
ENV CARLA_VERSION ${CARLA_VERSION:?}
ENV PYTHON_VENV_NAME driving

COPY --chown=docker:docker ./main_container/requirements.txt /tmp
WORKDIR /home/docker
RUN python3 -m venv $PYTHON_VENV_NAME && \
    source $PYTHON_VENV_NAME/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    echo "source /home/docker/$PYTHON_VENV_NAME/bin/activate" >> ~/.bashrc

# Install carla python api if carla v0.9.12 or greater, otherwise add the egg to the python path
RUN if [[ "$CARLA_VERSION" > "0.9.11" ]]; then \
    source $PYTHON_VENV_NAME/bin/activate && \
    python -m pip install "carla==$CARLA_VERSION"; \
    else \
    CARLA_VERSION=$(echo $CARLA_VERSION | sed 's/\.[^.]*$//') && \
    echo "export PYTHONPATH=$PYTHONPATH:/opt/carla/PythonAPI/carla/dist/carla-${CARLA_VERSION}-py3.7-linux-x86_64.egg" >> ~/.bashrc; \
    fi
ENV DISPLAY=:1.0
WORKDIR /home/docker
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
ENTRYPOINT ["fixuid"]
CMD ["/usr/bin/supervisord"]