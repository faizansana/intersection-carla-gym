ARG CARLA_VERSION=0.9.13
ARG CUDA_VERSION=11.8.0
FROM carlasim/carla:${CARLA_VERSION} AS carla

ARG CUDA_VERSION=11.8.0
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04 AS base
SHELL [ "/bin/bash", "-c" ]

ARG CARLA_VERSION=0.9.13

ENV DEBIAN_FRONTEND noninteractive
ENV CARLA_VERSION $CARLA_VERSION
RUN apt-get update && \
    apt-get install software-properties-common -y && \
    # Needed for installing python3.8 in ubuntu 22.04
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y sudo curl wget git python3.8 python3.8-venv iputils-ping && \
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

COPY --chown=docker ./supervisord.conf /etc/supervisor/supervisord.conf
RUN chown -R docker:docker /etc/supervisor
RUN chmod 777 /var/log/supervisor/

COPY --from=carla /home/carla/PythonAPI /opt/carla/PythonAPI

# Setup python environment
USER docker:docker
ENV SHELL=/bin/bash
ENV CARLA_VERSION ${CARLA_VERSION:?}
ENV PYTHON_VENV_NAME custom_carla_gym

COPY --chown=docker:docker ./requirements.txt /tmp
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
ENTRYPOINT ["fixuid"]
CMD ["/usr/bin/supervisord"]