ARG CARLA_VERSION=0.9.13
ARG CUDA_VERSION=11.8.0
FROM carlasim/carla:${CARLA_VERSION} AS carla

ARG CUDA_VERSION=11.8.0
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu20.04 AS base
SHELL [ "/bin/bash", "-c" ]

ARG CARLA_VERSION=0.9.13

ENV DEBIAN_FRONTEND noninteractive
ENV CARLA_VERSION $CARLA_VERSION
RUN apt-get update && apt-get install -y sudo curl wget git python3-venv iputils-ping

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
USER root:root
RUN apt-get update && apt-get install -y lxde x11vnc xvfb mesa-utils && apt-get purge -y light-locker
RUN apt-get install -y supervisor
EXPOSE 5900

COPY --chown=docker ./supervisord.conf /etc/supervisor/supervisord.conf
RUN chown -R docker:docker /etc/supervisor
RUN chmod 777 /var/log/supervisor/

COPY --from=carla /home/carla/PythonAPI /opt/carla/PythonAPI

# Install miniconda
USER docker:docker
ENV SHELL=/bin/bash
ENV CARLA_VERSION ${CARLA_VERSION:?}
ENV CONDA_DIR /home/docker/conda
ENV CONDA_ENV_NAME custom_carla_gym
WORKDIR /home/docker/Downloads

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
     rm ~/miniconda.sh && \
     echo "export PATH=$CONDA_DIR/bin:$PATH" >> ~/.bashrc
# Putting conda in path to use 'conda activate'
ENV PATH=$CONDA_DIR/bin:$PATH


COPY --chown=docker:docker ./environment.yml /tmp
RUN conda update conda && \
    conda env create -n $CONDA_ENV_NAME -f /tmp/environment.yml && \
    conda clean -a -y

RUN conda init bash && \
    echo "conda activate $CONDA_ENV_NAME" >> ~/.bashrc
# Install carla python api if carla v0.9.12 or greater, otherwise add the egg to the python path
RUN if [[ "$CARLA_VERSION" > "0.9.11" ]]; then \
    ${CONDA_DIR}/envs/$CONDA_ENV_NAME/bin/python -m pip install "carla==$CARLA_VERSION"; \
    else \
    echo "export PYTHONPATH=$PYTHONPATH:/opt/carla/PythonAPI/carla/dist/carla-${CARLA_VERSION}-py3.7-linux-x86_64.egg" >> ~/.bashrc; \
    fi
ENV DISPLAY=:1.0
WORKDIR /home/docker
ENTRYPOINT ["fixuid"]
CMD ["/usr/bin/supervisord"]